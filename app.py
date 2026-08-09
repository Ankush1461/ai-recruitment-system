# Optional Hugging Face Spaces compatibility: spaces must be imported before
# any CUDA/PyTorch modules, otherwise spaces.reloading throws RuntimeError.
try:
    import spaces  # type: ignore # noqa: F401
except Exception:
    pass

import json
import os
import socket
import threading
import time

import auth
import backup
import db
import embeddings
import vectorstore
from ui import demo


def register_google_oauth_routes() -> None:
    """Attach /auth/google/start (the website-style Google sign-in entry).

    Gradio rebuilds its FastAPI app inside launch(), so the route must be
    added to the *running* app instance after the server is up.
    """
    from fastapi import Request as FastAPIRequest
    from fastapi.responses import RedirectResponse

    app = demo.app  # type: ignore[attr-defined]

    def _redirect_uri(request) -> str:
        # The redirect URI is the app's own origin (scheme + host only). The
        # Referer can be the app page itself — sometimes WITH a query string
        # (?state=...&code=... from a previous Google callback) — and that
        # query must never leak into Google's redirect_uri (Google only
        # accepts a registered URI with no path/query). Opaque or missing
        # origins fall back to how the app is bound.
        origin = auth.normalize_redirect_uri(
            request.headers.get("origin") or request.headers.get("referer") or ""
        )
        if origin:
            return origin
        host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
        port = os.getenv("PORT", "7861")
        if os.getenv("SPACE_ID"):
            return f"https://{os.getenv('SPACE_ID')}.hf.space/"
        return f"http://{host}:{port}/"

    @app.get("/auth/google/start")
    @app.get("/auth/google/start/")
    def google_start(request: FastAPIRequest):
        if not auth.google_enabled():
            return RedirectResponse("/?oauth=disabled", status_code=302)
        try:
            # The redirect URI is decided HERE (from the start request, whose
            # headers reliably carry the app's own origin) and stored with the
            # attempt, so the callback exchange reuses the EXACT same value.
            # Google requires the token-exchange redirect_uri to match the
            # authorization request's precisely — re-deriving it from the
            # callback request's headers (Referer = accounts.google.com, no
            # app Origin) would mismatch and fail every sign-in.
            redirect_uri = _redirect_uri(request)
            state, verifier = auth.new_google_attempt(redirect_uri)
            url = auth.build_google_auth_url(redirect_uri, state, verifier)
            # The callback is correlated through the `state` query parameter
            # (validated + consumed server-side in the load handler).
            return RedirectResponse(url, status_code=302)
        except Exception:
            return RedirectResponse("/?oauth=error", status_code=302)


# ================================================================
# API auth gate — every Gradio API call requires a valid session
# ================================================================
# The login gate only hides the UI; the underlying event endpoints
# (/gradio_api/call/*, /gradio_api/queue/join) are reachable by anyone who
# can POST JSON. This pure-ASGI middleware sits in front of the live app and
# rejects such calls with 401 unless the payload carries a valid session
# token — EXCEPT the login/auth events themselves (which must be callable
# before a session exists). Handlers additionally run inside auth.user_scope
# (per-request thread isolation), so this gate is the boundary-level second
# line of defense, not the only one.
#
# The middleware reads the request body to find the token and re-serves the
# buffered body downstream, so Gradio still receives the full payload.

_OPEN_EVENTS = {"_on_auth_mode", "_on_auth_submit", "_on_logout", "_on_page_load"}


class _ApiAuthGate:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()
        is_event_submit = method == "POST" and (
            path.startswith("/gradio_api/call/")
            or path == "/gradio_api/queue/join"
        )
        if not is_event_submit:
            # Page loads, static assets, file uploads, SSE result streams and
            # the GET /gradio_api/call/{api}/{event_id} poll all pass through.
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        api_name = None
        if path.startswith("/gradio_api/call/"):
            rest = path[len("/gradio_api/call/"):].rstrip("/")
            # Gradio 5+/6 serve the POST route as /gradio_api/call/v2/<name>
            # (the non-v2 alias still exists but the v2 path is canonical).
            # Strip the version segment so open events (login, logout, page
            # load) resolve to their real api_name instead of "v2" — which
            # would otherwise 401 them before a session exists.
            rest = rest.removeprefix("v2/")
            api_name = rest.split("/")[0] if rest else None
        try:
            payload = json.loads(body.decode("utf-8", "replace") or "{}")
        except Exception:
            payload = {}

        if api_name is None:
            # /queue/join carries no api_name in the URL — resolve it from the
            # event index so anonymous (lambda) handlers are still gated.
            api_name = _event_name_for(payload)

        if os.getenv("AUTH_GATE_DEBUG") and api_name in _OPEN_EVENTS:

            data = payload.get("data")
            print(
                f"[gate] {api_name} fn_index={payload.get('fn_index')} "
                f"data0={str(data[0])[:30] if isinstance(data, list) and data else None!r}",
                flush=True,
            )

        allowed = api_name in _OPEN_EVENTS or _payload_has_session(payload)
        if not allowed:
            from fastapi.responses import JSONResponse

            response = JSONResponse(
                status_code=401,
                content={
                    "error": "Not authenticated",
                    "detail": "A valid session is required to call this API.",
                },
            )
            await response(scope, receive, send)
            return

        sent = {"sent": False}

        async def re_receive():
            if not sent["sent"]:
                sent["sent"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, re_receive, send)


async def _read_body(receive) -> bytes:
    """Drain the ASGI receive channel into a byte string."""
    body = b""
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    return body


def _payload_has_session(payload: dict) -> bool:
    """True when the event payload carries a valid session token.

    The session token (BrowserState) is a top-level string in `data`. Values
    are length-filtered before a cheap users.db lookup so resume text, paths
    and messages are never scanned as tokens.
    """
    data = payload.get("data")
    if not isinstance(data, list):
        return False
    for value in data:
        if (
            isinstance(value, str)
            and 40 <= len(value) <= 64
            and auth.get_user_by_session(value)
        ):
            return True
    return False


def _event_name_for(payload: dict) -> str | None:
    """Resolve the fn_index in a queue/join payload to its event name."""
    try:
        fn_index = payload.get("fn_index")
        if not isinstance(fn_index, int):
            return None
        blocks = demo.app.get_blocks()  # type: ignore[attr-defined]
        fn = blocks.fns[fn_index]
        return getattr(fn, "api_name", None)
    except Exception:
        return None


def install_api_auth_gate() -> None:
    """Wrap the RUNNING app with the session gate (after launch — Gradio
    rebuilds its FastAPI app inside launch()).

    Starlette refuses add_middleware() once the app has started, so the gate
    is spliced directly in front of the live middleware stack. New requests
    hit Starlette.__call__ -> self.middleware_stack, which now starts with
    the gate, while the previous stack (routes, CORS, etc.) runs underneath.
    """
    app = demo.app  # type: ignore[attr-defined]
    app.middleware_stack = _ApiAuthGate(app.middleware_stack)
    print("[boot] API auth gate installed (every Gradio API call requires a session)")


def find_free_port(start: int, max_tries: int = 50) -> int:
    """Return the first free port at or above `start`, or raise if none is found.

    Uses the same bind-check Gradio performs, so a port held by a lingering
    previous instance is skipped instead of crashing the app.
    """
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise OSError(
        f"No free port found in range {start}-{start + max_tries - 1}. "
        "Kill any lingering `py app.py` processes (Task Manager or "
        "`taskkill /F /IM python.exe`) and retry."
    )


if __name__ == "__main__":
    # Restore accounts + per-user data from the HF backup dataset repo FIRST,
    # before any DB is opened, so a rebuilt (wiped) Space comes back with
    # every account, job and candidate intact. No-op when backup isn't
    # configured or this disk already holds data.
    if backup.restore_if_needed():
        print("[boot] data restored from the HF backup dataset repo")
    # Global identity store (users.db) is created before anything else so
    # the login gate can serve accounts on first boot.
    auth.init_db()
    db.init_db()
    # Rebuild the vector index once if the embedding model changed (EN+DE
    # support) — the resume_text in SQLite is the source of truth.
    reindexed = vectorstore.maybe_reindex_all()
    if reindexed:
        print(f"[boot] re-indexed {reindexed} candidate(s) with the new embedding model")
    # Preload the embedding model on a daemon thread (overlaps Gradio launch)
    # so candidate ingest is never blocked by model loading — ingest itself is
    # deferred to a background thread, so this only shrinks the window in
    # which a freshly ingested candidate has no vectors yet.
    is_hf_space = bool(os.getenv("SPACE_ID") or os.getenv("SYSTEM") == "spaces")
    default_port = "7860" if is_hf_space else "7861"
    requested_port = int(os.getenv("PORT", default_port))
    port = find_free_port(requested_port)
    if port != requested_port:
        print(
            f"[boot] port {requested_port} is in use (an old instance may still be "
            f"running) — starting on port {port} instead."
        )
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0" if is_hf_space else "127.0.0.1")

    # launch() blocks, so run it in a thread and register the OAuth routes on
    # the live app once the server is up (Gradio rebuilds its FastAPI app
    # inside launch()).
    def _serve() -> None:
        demo.launch(
            server_name=server_name,
            server_port=port,
            share=os.getenv("GRADIO_SHARE", "0").lower() in ("1", "true"),
            ssr_mode=False,
        )

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    for _ in range(120):
        try:
            import socket as _s

            with _s.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    try:
        register_google_oauth_routes()
        print("[boot] Google OAuth route /auth/google/start registered")
    except Exception as e:
        print(f"[boot] warning: could not register Google OAuth route: {e}")
    try:
        install_api_auth_gate()
    except Exception as e:
        print(f"[boot] warning: could not install the API auth gate: {e}")
    # Periodic backup to the HF dataset repo (no-op when not configured).
    backup.start_backup_timer()
    t.join()
