# ================================================================
# 🧬 Local Embeddings — sentence-transformers (zero API cost)
# ================================================================
from __future__ import annotations

import threading
from contextlib import suppress
from typing import Any

try:
    import spaces  # type: ignore
except Exception:
    class _DummySpaces:
        @staticmethod
        def GPU(func: Any = None, **kwargs: Any) -> Any:
            if func is None:
                return lambda f: f
            return func

    spaces = _DummySpaces()  # type: ignore

from sentence_transformers import SentenceTransformer

import config

# Multilingual paraphrase MiniLM-L12 — 384-dim, 50+ languages (EN + DE), still
# CPU-friendly (~120 MB, downloads on first run, cached in ~/.cache/huggingface/).
# Override with EMBEDDING_MODEL. Switching models invalidates existing vectors —
# vectorstore.maybe_reindex_all() rebuilds them automatically on the next boot.
_MODEL_NAME = config.EMBEDDING_MODEL
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()

# Serializes encode() calls: model loading is expensive, and multiple
# background ingest threads may embed at the same time (torch forward passes
# on one model are safe, but the LOAD is not — the lock covers both).
_embed_lock = threading.Lock()


def model_name() -> str:
    """Configured embedding model id (used to detect model switches)."""
    return _MODEL_NAME


def get_model() -> SentenceTransformer:
    """Lazy singleton — loads model once (thread-safe), reuses across calls."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(_MODEL_NAME)
    return _model


def warm() -> None:
    """Preload the model (best-effort, never raises).

    Called from a daemon thread at boot so the model is already in memory by
    the time the first candidate is ingested — the UI never waits on loading.
    """
    with suppress(Exception):
        get_model()


@spaces.GPU
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings. Returns list of 384-dim float vectors."""
    model = get_model()
    with _embed_lock:
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_single(text: str) -> list[float]:
    """Embed a single text string. Returns one 384-dim float vector."""
    return embed_texts([text])[0]
