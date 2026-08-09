# ================================================================
# 🧠 Unified LLM Client — Groq (primary) + Ollama (local fallback)
# ================================================================

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

import config
import db

load_dotenv()


@dataclass
class LLMResponse:
    """Wrapper around an LLM chat response."""
    text: str
    model: str
    provider: str  # "groq" or "ollama"
    raw: Any = None


class LLMClient:
    """Unified interface to Groq and Ollama."""

    def __init__(self, provider: str, client: Any, model: str):
        self.provider = provider
        self._client = client
        self.model = model

    def chat(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = False,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a chat completion request."""
        if self.provider == "groq":
            return self._chat_groq(prompt, system, json_mode, temperature)
        elif self.provider == "ollama":
            return self._chat_ollama(prompt, system, json_mode, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def chat_json(self, prompt: str, system: str = "", temperature: float = 0.3) -> dict:
        """Send a chat request and parse JSON response. Raises on parse failure."""
        resp = self.chat(prompt, system, json_mode=True, temperature=temperature)
        text = resp.text.strip()

        # Strip markdown fences if the model wraps JSON in ```json ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        return json.loads(text)

    # -- Groq ----------------------------------------------------------

    def _chat_groq(
        self, prompt: str, system: str, json_mode: bool, temperature: float
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._retry_call(lambda: self._client.chat.completions.create(**kwargs))
        text = response.choices[0].message.content or ""
        self._log_usage(response)

        return LLMResponse(
            text=text,
            model=self.model,
            provider="groq",
            raw=response,
        )

    # -- Resilience helpers ------------------------------------------------

    def _retry_call(
        self,
        fn: Any,
        max_attempts: int | None = None,
        base_delay: float | None = None,
    ) -> Any:
        """Call fn with exponential backoff on transient errors (429 / 5xx)."""
        attempts = max_attempts or config.LLM_MAX_ATTEMPTS
        delay = base_delay or config.LLM_BASE_DELAY
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                if attempt >= attempts or not _is_transient(exc):
                    raise
                time.sleep(delay * (2 ** (attempt - 1)))
        return fn()  # pragma: no cover — loop above always returns or raises

    def _log_usage(self, response: Any) -> None:
        """Persist token usage for one LLM call into the audit log (best effort)."""
        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return
            db.audit(
                "llm_call",
                "llm",
                "",
                json.dumps({
                    "provider": self.provider,
                    "model": self.model,
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                }),
            )
        except Exception:
            pass

    # -- Ollama --------------------------------------------------------

    def _chat_ollama(
        self, prompt: str, system: str, json_mode: bool, temperature: float
    ) -> LLMResponse:
        import ollama as ollama_lib

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        if json_mode:
            kwargs["format"] = "json"

        response = ollama_lib.chat(**kwargs)
        text = response.get("message", {}).get("content", "")

        return LLMResponse(
            text=text,
            model=self.model,
            provider="ollama",
            raw=response,
        )


def _is_transient(exc: Exception) -> bool:
    """True for retryable errors: 408/429/5xx or transport-style exceptions."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 429) or 500 <= status < 600
    name = type(exc).__name__.lower()
    return any(
        k in name
        for k in ("rate", "connection", "timeout", "internal", "overloaded", "unavailable")
    )


def get_llm_client(
    user_api_key: str = "",
    model: str | None = None,
) -> tuple[LLMClient | None, str | None]:
    """Initialize the best available LLM client.

    Priority:
    1. Groq (with user-provided key or GROQ_API_KEY from env)
    2. Ollama local (if running on localhost:11434)

    Args:
        user_api_key: Optional user-supplied Groq key (overrides env).
        model: Override the configured model — used for hybrid routing so
            low-stakes calls (e.g. follow-up detection) can use the cheap
            fast model while scoring stays on the strong one.

    Returns:
        (client, error_message) — client is None if error_message is set.
    """
    # --- Try Groq first ---
    api_key = (
        user_api_key.strip().strip("'\"")
        if user_api_key and user_api_key.strip()
        else config.GROQ_API_KEY
    )
    if api_key:
        try:
            from groq import Groq

            groq_client = Groq(api_key=api_key)
            # Model policy: default = strongest (GROQ_MODEL); an override
            # (e.g. GROQ_FAST_MODEL) routes low-stakes calls to the cheap model.
            chosen = model or config.GROQ_MODEL
            return (
                LLMClient(provider="groq", client=groq_client, model=chosen),
                None,
            )
        except Exception:
            pass  # Fall through to Ollama

    # --- Try Ollama fallback ---
    try:
        import ollama as ollama_lib

        # Check if Ollama is reachable
        models = ollama_lib.list()
        available = [m.get("name", "") for m in models.get("models", [])]
        if available:
            # Prefer OLLAMA_MODEL, fall back to first available
            chosen = config.OLLAMA_MODEL if config.OLLAMA_MODEL in available else available[0]
            return (
                LLMClient(provider="ollama", client=None, model=chosen),
                None,
            )
    except Exception:
        pass

    return (
        None,
        (
            "⚠️ **No LLM available.** Set `GROQ_API_KEY` in your `.env` file, "
            "or start Ollama locally (`ollama serve`)."
        ),
    )


def get_fast_llm_client(user_api_key: str = "") -> tuple[LLMClient | None, str | None]:
    """Client on the fast/cheap model for low-stakes LLM calls.

    Hybrid routing: high-stakes scoring (screening, interview evaluation)
    uses the strong GROQ_MODEL; trivial decisions (follow-up detection) use
    GROQ_FAST_MODEL so the request/day budget lasts much longer.
    Falls back to the main model if GROQ_FAST_MODEL is unset.
    """
    return get_llm_client(user_api_key, model=config.GROQ_FAST_MODEL or None)
