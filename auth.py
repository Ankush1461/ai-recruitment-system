# ================================================================
# Accounts - login (email/password + Google) & per-user isolation
# ================================================================
# Identity is stored in one shared users.db (emails, PBKDF2 hashes, Google
# ids). Each account's *data* (recruiter.db, Chroma vectors, CSV exports)
# lives under data/users/<user_id>/ so every user sees only their own jobs,
# candidates, screenings and interviews.
#
# Google sign-in uses the standard OAuth 2.0 authorization-code redirect
# flow with PKCE: the browser goes to Google's consent page and comes back
# with a one-time code, which is exchanged server-side for a verified
# id_token (see _verify_id_token). Requires GOOGLE_CLIENT_ID and
# GOOGLE_CLIENT_SECRET in .env; without them the Google button is disabled
# with a friendly hint.

from __future__ import annotations

import base64
import gc
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Any

import config
import db
import reports
import vectorstore
import video_interview

# Google id_token verification (signature + audience + issuer + expiry). Uses
# google-auth's standard verifier — certs are fetched once from Google and
# cached. When the library is missing, Google sign-in is disabled safely.
# The module-level names are pre-declared (Any) so the optional-dependency
# fallback keeps its type while the library remains importable at runtime.
_google_requests: Any
_google_id_token: Any
try:
    from google.auth.transport import requests as _google_requests
    from google.oauth2 import id_token as _google_id_token

    _GOOGLE_AUTH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _google_requests = None
    _google_id_token = None  # type: ignore[assignment]
    _GOOGLE_AUTH_AVAILABLE = False

_USERS_DB_PATH = str(config.USERS_DB_PATH)
_USER_DATA_DIR = str(config.USER_DATA_DIR)

# Google OAuth 2.0 endpoints
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_SCOPE = "openid email profile"

# The signed-in user for THIS THREAD (None when logged out). Switches
# db/vectorstore/reports to that user's private storage. Per-thread so two
# concurrent requests (Gradio runs handlers in a thread pool) can never leak
# each other's data — auth.set_active_user / auth.user_scope are the ONLY
# writers, and every data handler runs inside auth.user_scope.
_thread = threading.local()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{secrets.token_hex(12)}"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_USERS_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    if config.SQLITE_WAL:
        # read-only filesystem — default journal mode still works
        with suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class AuthLockedError(Exception):
    """Raised when a sign-in/registration attempt is rejected by rate
    limiting (too many failed attempts for this email or device)."""


def init_db() -> None:
    """Create the users + sessions tables if they do not exist (global
    identity store). Sessions let a logged-in user survive page reloads:
    the token is stored in the browser and resolved from the DB on load."""
    os.makedirs(os.path.dirname(_USERS_DB_PATH) or ".", exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT DEFAULT '',
                name TEXT DEFAULT '',
                provider TEXT DEFAULT 'email',   -- 'email' | 'google'
                google_id TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT DEFAULT ''
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL,          -- 'login:<email>' | 'ip:<addr>' | 'reg:<addr>'
                success INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_attempts_scope ON auth_attempts (scope, created_at)"
        )
        _migrate_sessions(conn)


def _migrate_sessions(conn: sqlite3.Connection) -> None:
    """Add expires_at to sessions (older builds lack it) and backfill from
    created_at so previously-issued tokens get the configured TTL."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    except sqlite3.OperationalError:
        return  # table unavailable — skip the migration entirely (fail-open)
    if "expires_at" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN expires_at TEXT DEFAULT ''")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "expires_at" not in cols:
        return
    rows = conn.execute(
        "SELECT token, created_at FROM sessions "
        "WHERE expires_at IS NULL OR expires_at = ''"
    ).fetchall()
    for row in rows:
        created = (row["created_at"] or "").strip()
        try:
            base = datetime.fromisoformat(created)
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
        except ValueError:
            base = datetime.now(timezone.utc)
        expires = (base + timedelta(days=config.SESSION_TTL_DAYS)).replace(microsecond=0)
        conn.execute(
            "UPDATE sessions SET expires_at = ? WHERE token = ?",
            (expires.isoformat(), row["token"]),
        )


# ---- Password hashing (PBKDF2-HMAC-SHA256, stdlib only) --------------------

def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 210_000)
    return (
        "pbkdf2_sha256$210000$"
        + base64.urlsafe_b64encode(salt).decode()
        + "$"
        + base64.urlsafe_b64encode(dk).decode()
    )


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, dk_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_b64)
        dk = base64.urlsafe_b64decode(dk_b64)
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iters)
        )
        return hmac.compare_digest(candidate, dk)
    except Exception:
        return False


# ---- Brute-force protection (rate limiting / lockout) ------------------------
# Attempts are recorded in auth_attempts (persisted in users.db). The policy:
#   - > AUTH_MAX_FAILED_ATTEMPTS failed sign-ins per email within the window
#     locks that account for AUTH_LOCKOUT_WINDOW seconds.
#   - > AUTH_IP_MAX_FAILED_ATTEMPTS failed sign-ins per IP locks the device.
#   - > AUTH_MAX_REGISTRATIONS_PER_IP account creations per IP within the
#     registration window is rejected.
# Rate limiting is best-effort: recording failures must never break auth itself.


def _record_auth_attempt(scope: str, success: bool) -> None:
    """Record one sign-in/registration attempt for rate limiting.

    Also prunes attempts older than the longest configured window so the
    table can never grow unbounded (lazy cleanup, like sessions).
    """
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO auth_attempts (scope, success, created_at) VALUES (?, ?, ?)",
                (scope, 1 if success else 0, _utc_now()),
            )
            oldest = datetime.now(timezone.utc) - timedelta(
                seconds=max(config.AUTH_LOCKOUT_WINDOW, config.AUTH_REGISTER_WINDOW)
            )
            conn.execute(
                "DELETE FROM auth_attempts WHERE created_at < ?",
                (oldest.isoformat(),),
            )
    except Exception:
        pass


def _recent_auth_attempts(scope: str, window_seconds: int, success: int | None = None) -> int:
    """Count attempts in the last `window_seconds` for a scope.

    success=None counts all attempts; success=0 counts only failures.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).isoformat()
    try:
        with _connect() as conn:
            if success is None:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM auth_attempts "
                    "WHERE scope = ? AND created_at > ?",
                    (scope, cutoff),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM auth_attempts "
                    "WHERE scope = ? AND success = ? AND created_at > ?",
                    (scope, success, cutoff),
                ).fetchone()
        return int(row["n"] or 0) if row else 0
    except Exception:
        return 0


def _clear_auth_attempts(scope: str) -> None:
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM auth_attempts WHERE scope = ?", (scope,))
    except Exception:
        pass


def login_locked_out(email: str, ip: str = "") -> str | None:
    """Return a user-facing error when brute-force limits are exceeded, else None.

    Checks the per-email failure count first, then the per-IP failure count.
    """
    email = (email or "").strip().lower()
    if email:
        failures = _recent_auth_attempts(
            f"login:{email}", config.AUTH_LOCKOUT_WINDOW, success=0
        )
        if failures >= config.AUTH_MAX_FAILED_ATTEMPTS:
            minutes = max(1, config.AUTH_LOCKOUT_WINDOW // 60)
            return (
                "Too many failed sign-in attempts. "
                f"Please wait about {minutes} minute(s) and try again."
            )
    if ip and config.AUTH_ENFORCE_IP_LIMITS:
        failures = _recent_auth_attempts(
            f"ip:{ip}", config.AUTH_LOCKOUT_WINDOW, success=0
        )
        if failures >= config.AUTH_IP_MAX_FAILED_ATTEMPTS:
            return "Too many failed attempts from this device. Please try again later."
    return None


# ---- User records ----------------------------------------------------------

def _row_to_user(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return dict(row)


def get_user_by_email(email: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
    return _row_to_user(row)


def get_user_by_google_id(google_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE google_id = ?", (google_id,)
        ).fetchone()
    return _row_to_user(row)


def register_user(email: str, password: str, name: str = "", ip: str = "") -> dict:
    """Create an email/password account. Raises ValueError on invalid input
    or when the email is already registered; raises AuthLockedError when the
    device has created too many accounts (per-IP registration cap)."""
    email = (email or "").strip().lower()
    password = password or ""
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise ValueError("Enter a valid email address.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if ip and config.AUTH_ENFORCE_IP_LIMITS:
        recent = _recent_auth_attempts(f"reg:{ip}", config.AUTH_REGISTER_WINDOW)
        if recent >= config.AUTH_MAX_REGISTRATIONS_PER_IP:
            raise AuthLockedError(
                "Too many accounts created from this device. Please try again later."
            )
    if get_user_by_email(email):
        raise ValueError("An account with that email already exists - sign in instead.")
    uid = _new_id("usr_")
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, name, provider, google_id, created_at) "
            "VALUES (?, ?, ?, ?, 'email', '', ?)",
            (uid, email, _hash_password(password), (name or "").strip(), now),
        )
    if ip and config.AUTH_ENFORCE_IP_LIMITS:
        _record_auth_attempt(f"reg:{ip}", True)
    user = get_user_by_email(email)
    assert user is not None  # just inserted
    _maybe_migrate_legacy(user)  # first account inherits existing local data
    return user


def authenticate(email: str, password: str, ip: str = "") -> dict | None:
    """Verify credentials; returns the user dict or None.

    Enforces brute-force lockout per email and per IP: once the failure
    threshold is exceeded, raises AuthLockedError (even for the correct
    password) until the window elapses. A successful login resets the
    per-email failure counter.
    """
    email = (email or "").strip().lower()
    locked = login_locked_out(email, ip)
    if locked:
        raise AuthLockedError(locked)
    user = get_user_by_email(email)
    if not user or not user.get("password_hash"):
        _record_auth_attempt(f"login:{email}", False)
        if ip and config.AUTH_ENFORCE_IP_LIMITS:
            _record_auth_attempt(f"ip:{ip}", False)
        return None
    if not _verify_password(password or "", user["password_hash"]):
        _record_auth_attempt(f"login:{email}", False)
        if ip and config.AUTH_ENFORCE_IP_LIMITS:
            _record_auth_attempt(f"ip:{ip}", False)
        return None
    _clear_auth_attempts(f"login:{email}")
    _maybe_migrate_legacy(user)
    return user


def get_or_create_google_user(google_id: str, email: str, name: str = "") -> dict:
    """Return the account bound to a Google id, creating it on first sign-in.
    An existing email/password account is upgraded to also accept Google."""
    user = get_user_by_google_id(google_id)
    if user:
        return user
    email = (email or "").strip().lower()
    # Google always sends an email with the `email` scope, but be defensive:
    # derive a unique placeholder so the account is still usable if it is absent.
    if not email:
        email = f"g-{google_id[:12]}@google-oauth.local"
    existing = get_user_by_email(email)
    if existing:
        with _connect() as conn:
            conn.execute(
                "UPDATE users SET google_id = ?, provider = 'google' WHERE id = ?",
                (google_id, existing["id"]),
            )
        user = get_user_by_email(email)  # re-fetch: reflects the new google_id
    else:
        uid = _new_id("usr_")
        now = _utc_now()
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash, name, provider, google_id, created_at) "
                "VALUES (?, ?, '', ?, 'google', ?, ?)",
                (uid, email, (name or "").strip(), google_id, now),
            )
        user = get_user_by_email(email)
    assert user is not None  # created or upgraded above
    _maybe_migrate_legacy(user)
    return user


# ---- Sessions (remember login across page reloads) -------------------------

def create_session(user_id: str) -> str:
    """Issue a persistent login token for a user. The token is stored in
    users.db and kept in the browser, so a page reload (or a server restart)
    can restore the session until it expires (SESSION_TTL_DAYS). Stale and
    expired sessions are pruned on issue."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    expires = (now + timedelta(days=config.SESSION_TTL_DAYS)).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), expires),
        )
        conn.execute(
            "DELETE FROM sessions WHERE expires_at != '' AND expires_at < ?",
            (now.isoformat(),),
        )
    return token


def get_user_by_session(token: str | None) -> dict | None:
    """Resolve a session token to its user, or None when invalid/unknown/expired.

    Expired tokens are rejected AND removed (lazy cleanup) so a dead token
    cannot accumulate in the table."""
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT u.id, u.email, u.password_hash, u.name, u.provider, "
            "u.google_id, u.created_at "
            "FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token = ? AND (s.expires_at = '' OR s.expires_at > ?)",
            (token, _utc_now()),
        ).fetchone()
        if row is None:
            # Reject + drop expired/orphaned rows.
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    return _row_to_user(row)


def delete_session(token: str | None) -> None:
    """Invalidate a session (used on logout)."""
    # Guard against API clients that pass a raw component payload (a dict
    # update) instead of the token string — never let logout crash on that.
    if not isinstance(token, str) or not token:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def get_user_by_id(user_id: str) -> dict | None:
    """The full user row for an account id (used by the Profile tab)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return _row_to_user(row)


def update_user_name(user_id: str, name: str) -> dict | None:
    """Rename the account. Returns the refreshed user dict (None if gone)."""
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET name = ? WHERE id = ?",
            ((name or "").strip(), user_id),
        )
    return get_user_by_id(user_id)


def delete_user(user_id: str) -> bool:
    """Permanently remove the account and everything tied to it.

    Deletes the user row, every session for it, the auth-attempt history for
    the account's email (brute-force lockout counters), and the user's whole
    private data directory (recruiter.db, chroma, exports, media). Returns
    True when a row was actually removed.
    """
    email = ""
    with _connect() as conn:
        row = conn.execute(
            "SELECT email FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            return False
        email = row["email"] or ""
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        if email:
            conn.execute(
                "DELETE FROM auth_attempts WHERE scope = ?", (f"login:{email}",)
            )
    with _reindex_lock:
        _reindex_checked.discard(user_id)
    # Release this thread's handles to the user's files first so Windows can
    # actually remove the directory: the Chroma persistent client keeps
    # chroma.sqlite3 open (and thus the whole dir locked) until closed.
    if (getattr(_thread, "user", None) or {}).get("id") == user_id:
        vectorstore.close_active_chroma()
        set_active_user(None)
    if os.path.isdir(_user_root(user_id)):
        # Best-effort: Chroma can take a moment to release its file handle
        # after close(), so retry briefly before giving up on Windows.
        gc.collect()
        for _ in range(4):
            shutil.rmtree(_user_root(user_id), ignore_errors=True)
            if not os.path.isdir(_user_root(user_id)):
                break
            time.sleep(0.2)
    return True


# ---- Per-user storage ------------------------------------------------------

def _user_root(user_id: str) -> str:
    return os.path.join(_USER_DATA_DIR, user_id)


def user_storage(user_id: str) -> dict[str, str]:
    """The private db / chroma / export / media paths for one user."""
    root = _user_root(user_id)
    return {
        "db": os.path.join(root, "recruiter.db"),
        "chroma": os.path.join(root, "chroma"),
        "exports": os.path.join(root, "exports"),
        "media": os.path.join(root, "media"),
    }


# Users whose vector store was already checked for an embedding-model change
# THIS process. The boot-time reindex (app.py) only covers the legacy default
# store; each account's private store is checked lazily on first access so an
# EMBEDDING_MODEL switch rebuilds every account's vectors too.
_reindex_checked: set[str] = set()
_reindex_lock = threading.Lock()


def _smtp_encryption_key(user: dict | None) -> bytes | None:
    """Per-account key for at-rest SMTP password encryption (db layer).

    Derived from the account's stored PBKDF2 password hash — the same secret
    that protects the login — with a fixed app-tag salt, so a leaked per-user
    .db file cannot reveal the SMTP password without also knowing the account
    password hash. Google-only accounts have no password hash → None (their
    SMTP password stays plaintext, fail-open). Changing the account password
    changes this key, which invalidates the stored ciphertext (detected via
    its HMAC tag) until the recruiter re-enters the SMTP password.
    """
    ph = (user or {}).get("password_hash") or ""
    if not ph:
        return None
    # A single domain-separated HMAC suffices: the input is ALREADY the
    # 210k-iteration PBKDF2 password hash (full 256-bit entropy, not a
    # low-entropy password), so no extra stretching is needed — and this runs
    # on EVERY request via set_active_user, so it must stay cheap.
    return hmac.new(b"talentiq-smtp-v1", ph.encode("utf-8"), hashlib.sha256).digest()


def set_active_user(user: dict | None) -> None:
    """Switch THIS THREAD to a user's private storage (or back to global)."""
    # Guard against stale thread state: a user_scope's finally restores the
    # PREVIOUS thread user, which may be an account that was just deleted
    # (the delete handler runs inside user_scope). Re-activating it would
    # recreate its data dirs and Chroma index. Verify the account still
    # exists before switching to it.
    if user is not None and get_user_by_id(user.get("id", "")) is None:
        user = None
    _thread.user = user
    if user is None:
        db.set_active_db(None)
        db.set_active_encryption_key(None)
        vectorstore.set_active_chroma(None)
        reports.set_export_dir(None)
        video_interview.set_active_media_dir(None)
        return
    storage = user_storage(user["id"])
    os.makedirs(os.path.dirname(storage["db"]), exist_ok=True)
    db.set_active_db(storage["db"])
    db.set_active_encryption_key(_smtp_encryption_key(user))
    vectorstore.set_active_chroma(storage["chroma"])
    reports.set_export_dir(storage["exports"])
    video_interview.set_active_media_dir(storage["media"])
    # Per-user vector reindex (once per user per process): cheap when the
    # store is already on the current embedding model (one metadata read),
    # and guarded by vectorstore's own lock file when it is not.
    already_checked = True
    with _reindex_lock:
        if user["id"] not in _reindex_checked:
            _reindex_checked.add(user["id"])
            already_checked = False
    if not already_checked:
        # best-effort — SQLite is the source of truth, it reindexes on demand
        with suppress(Exception):
            vectorstore.maybe_reindex_all(db_path=storage["db"])


def active_user() -> dict | None:
    """The user active on THIS thread (None when logged out)."""
    return getattr(_thread, "user", None)


@contextmanager
def user_scope(token: str | None) -> Iterator[dict]:
    """Validate a session token and run the block inside that user's private
    storage (per-thread, restoring the previous thread user on exit).

    Raises PermissionError when the token is missing or invalid — every
    workspace handler enters this scope, so an unauthenticated API call can
    never read or write another user's (or the global) data.

        with auth.user_scope(token) as user:
            ...  # db / vectorstore / reports are THIS user's
    """
    user = get_user_by_session(token)
    if not user:
        raise PermissionError(
            "Your session has expired — please sign in again."
        )
    previous = active_user()
    try:
        set_active_user(user)
        yield user
    finally:
        set_active_user(previous)


def _maybe_migrate_legacy(user: dict) -> None:
    """First sign-in: give the account the app's pre-existing local data.

    The original single-user build stored everything in recruiter.db + the
    root chroma_db. When the very first account is created, copy that data
    into the account's private storage so nothing is lost - but only if the
    account has no data yet and the legacy DB actually has content.
    """
    storage = user_storage(user["id"])
    user_db = storage["db"]
    if os.path.exists(user_db) and os.path.getsize(user_db) > 0:
        return
    legacy = str(config.DB_PATH)
    if not os.path.exists(legacy) or os.path.getsize(legacy) == 0:
        return
    # The legacy data is claimed by exactly one account: the first one that
    # signs in after the upgrade. A marker file next to the legacy DB records
    # the handover so later accounts start with a clean workspace.
    marker = os.path.join(os.path.dirname(legacy), ".legacy_migrated")
    if os.path.exists(marker):
        return
    try:
        with db.connect(legacy) as conn:
            n = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if not n:
            return
    except Exception:
        return
    try:
        os.makedirs(os.path.dirname(user_db), exist_ok=True)
        shutil.copyfile(legacy, user_db)
        # Mark the handover as soon as the DB copy succeeded, so a later
        # best-effort step failing can never strand the user_db without a
        # marker (which would silently skip both migration and retry).
        with open(marker, "w", encoding="utf-8") as f:
            f.write("Migrated to per-user storage; delete this file to re-claim.\n")
        # Copy the legacy vector index alongside (same candidate ids) if present.
        legacy_chroma = str(config.CHROMA_DIR)
        if os.path.isdir(legacy_chroma) and not os.path.exists(storage["chroma"]):
            shutil.copytree(
                legacy_chroma,
                storage["chroma"],
                ignore=shutil.ignore_patterns(".reindex.lock"),
            )
    except Exception:
        pass  # best-effort; SQLite is the source of truth and reindexes anyway


# ---- Google OAuth 2.0 authorization-code flow (website-style sign-in) ---
# The standard redirect flow: browser -> Google consent -> back to the app
# with a one-time code that is exchanged for an id_token. PKCE keeps the
# flow safe even when no client secret is configured.
#
# The PKCE verifier + state for an in-flight attempt are kept SERVER-SIDE
# (in-process dict, 10-minute TTL) and correlated via an opaque cookie set on
# the /auth/google/start redirect. The load handler validates the state from
# the callback against the stored one and exchanges the code with the stored
# verifier — no client-side JS or browser storage is involved.

_GOOGLE_STATE_TTL = 600  # seconds
# state-token -> {"verifier", "created_at"}
_google_attempts: dict[str, dict] = {}
_google_attempts_lock = threading.Lock()


def _prune_google_attempts() -> None:
    now = time.time()
    with _google_attempts_lock:
        to_remove = [t for t, v in _google_attempts.items() if now - v["created_at"] > _GOOGLE_STATE_TTL]
        for tok in to_remove:
            _google_attempts.pop(tok, None)


def normalize_redirect_uri(raw: str) -> str:
    """Turn a raw Origin/Referer header value into a canonical OAuth redirect
    URI (scheme://host + trailing slash), or '' when it is not usable.

    Rejects empty values and opaque origins ("null"), and drops any path or
    query string — a Referer from the app's own callback URL
    (`/?state=...&code=...`) must never leak those params into Google's
    redirect_uri, because Google only accepts a registered URI with no
    path/query and would reject the whole sign-in.
    """
    value = (raw or "").strip()
    if not value or "://" not in value:
        return ""
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}/"


def new_google_attempt(redirect_uri: str | None = None) -> tuple[str, str]:
    """Start a Google sign-in attempt. Returns (state_token, verifier).

    The token is stored server-side (10-minute TTL) together with the PKCE
    verifier AND the redirect URI the browser started with. The callback
    exchange reuses that exact stored URI, so the token-exchange request can
    never diverge from the authorization request — Google requires the two
    redirect_uris to match exactly, and the callback request's own headers
    (Referer = accounts.google.com, often no usable Origin) would otherwise
    produce a mismatch. A server restart invalidates any in-flight attempt
    (the user simply clicks the button again).
    """
    verifier, _ = new_pkce_pair()
    state = secrets.token_hex(16)
    _prune_google_attempts()
    with _google_attempts_lock:
        _google_attempts[state] = {
            "verifier": verifier,
            "redirect_uri": redirect_uri,
            "created_at": time.time(),
        }
    return state, verifier


def pop_google_attempt(state: str) -> dict | None:
    """Validate + consume a callback state token; returns the stored attempt
    ({"verifier", "redirect_uri"}) or None when unknown/expired/already used.
    """
    if not state:
        return None
    with _google_attempts_lock:
        attempt = _google_attempts.pop(state, None)
    if not attempt:
        return None
    if time.time() - attempt["created_at"] > _GOOGLE_STATE_TTL:
        return None
    return {
        "verifier": attempt["verifier"],
        "redirect_uri": attempt.get("redirect_uri"),
    }


def google_enabled() -> bool:
    return bool(config.GOOGLE_CLIENT_ID and config.GOOGLE_CLIENT_SECRET)

def new_pkce_pair() -> tuple[str, str]:
    """(verifier, challenge) for RFC 7636 PKCE (S256). The verifier is kept
    server-side until the redirect returns; the challenge goes to Google."""
    verifier = base64.urlsafe_b64encode(os.urandom(48)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _pkce_challenge(verifier: str) -> str:
    """S256 PKCE challenge for a verifier."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def build_google_auth_url(redirect_uri: str, state: str, verifier: str) -> str:
    """Build the accounts.google.com authorization URL for the redirect flow.

    `redirect_uri` must be registered as an *Authorized redirect URI* on the
    OAuth client (e.g. `http://localhost:7861/` locally, or the Space's
    `https://...hf.space/` when deployed).
    """
    params = {
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": _GOOGLE_SCOPE,
        "state": state,
        "code_challenge": _pkce_challenge(verifier),
        "code_challenge_method": "S256",
        "access_type": "online",
        "prompt": "select_account",
    }
    return _GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_google_code(redirect_uri: str, code: str, verifier: str) -> dict | None:
    """Exchange a one-time authorization code for an id_token and return the
    signed-in user (creating/linking the account as needed). Returns None if
    the code is invalid or the exchange fails.

    The client secret is sent when available; PKCE alone suffices otherwise.
    """
    params = {
        "client_id": config.GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "code": code,
        "code_verifier": verifier,
        "grant_type": "authorization_code",
    }
    if config.GOOGLE_CLIENT_SECRET:
        params["client_secret"] = config.GOOGLE_CLIENT_SECRET
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(_GOOGLE_TOKEN_URL, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            token = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # Google rejected the exchange (stale/used code, mismatched
        # redirect_uri, ...). Log the reason so a failed Google sign-in is
        # never a mystery — Google's `error`/`error_description` fields carry
        # no secrets, so printing them is safe.
        try:
            body = json.loads(e.read().decode() or "{}")
            err = body.get("error") or f"HTTP {e.code}"
            desc = body.get("error_description") or ""
            print(f"[google] token exchange rejected: {err} {desc}".strip())
        except Exception:
            print(f"[google] token exchange rejected: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"[google] token exchange failed: {type(e).__name__}: {e}")
        return None
    id_token = token.get("id_token", "")
    claims = _verify_id_token(id_token, config.GOOGLE_CLIENT_ID)
    google_id = str(claims.get("sub") or "")
    email = str(claims.get("email") or "")
    if not google_id:
        return None
    name = str(claims.get("name") or "")
    return get_or_create_google_user(google_id, email, name)


def _verify_id_token(token: str, audience: str) -> dict:
    """Verify a Google id_token JWT and return its claims ({} on any failure).

    Uses google-auth's standard verifier: the RSA signature is checked
    against Google's published certificates (fetched once, cached), and the
    audience (`aud` must equal our OAuth client id), issuer
    (accounts.google.com) and expiry (`exp`) are all enforced. A token that
    fails any check — including a forged or replay token — is rejected.

    A clock-skew tolerance (config.GOOGLE_CLOCK_SKEW_SECONDS) is passed for
    the `iat`/`exp` checks: without it, a token Google just issued fails as
    "Token used too early" whenever this machine's clock is even a second
    behind Google's servers, turning every Google login into a false
    "Google sign-in failed".
    """
    if not token or not _GOOGLE_AUTH_AVAILABLE or not audience or _google_id_token is None or _google_requests is None:
        return {}
    try:
        claims = _google_id_token.verify_oauth2_token(
            token,
            _google_requests.Request(),
            audience=audience,
            clock_skew_in_seconds=config.GOOGLE_CLOCK_SKEW_SECONDS,
        )
        return dict(claims or {})
    except Exception as e:
        # Log WHY a token failed verification (signature / audience / expiry /
        # a real clock-skew beyond tolerance) — a silent {} turns every
        # Google login failure into a mystery.
        print(f"[google] id_token verification failed: {type(e).__name__}: {e}")
        return {}
