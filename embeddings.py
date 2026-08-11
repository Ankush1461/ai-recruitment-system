# ================================================================
# 🧬 Local Embeddings — sentence-transformers (zero API cost)
# ================================================================
# ruff: noqa: I001 — import order is intentional: zerogpu (→ `import spaces`)
# must load BEFORE sentence-transformers (torch) so HF's spaces-before-CUDA
# rule holds on ZeroGPU Spaces; isort would reorder it.
from __future__ import annotations

import threading
from contextlib import suppress
from functools import lru_cache

import config

# ZeroGPU helpers import the `spaces` package BEFORE sentence-transformers so
# HF's spaces-before-CUDA import rule holds. wrap_gpu() auto-routes the model
# calls through @spaces.GPU on ZeroGPU Spaces with CPU fallback (see zerogpu.py).
from zerogpu import pick_device, wrap_gpu

from sentence_transformers import SentenceTransformer

# Multilingual paraphrase MiniLM-L12 — 384-dim, 50+ languages (EN + DE), still
# CPU-friendly (~120 MB, downloads on first run, cached in ~/.cache/huggingface/).
# Override with EMBEDDING_MODEL. Switching models invalidates existing vectors —
# vectorstore.maybe_reindex_all() rebuilds them automatically on the next boot.
_MODEL_NAME = config.EMBEDDING_MODEL
# Device is picked by PROBING for a real GPU (zerogpu.pick_device): on ZeroGPU
# Spaces, torch.cuda.is_available() returns True even OUTSIDE @spaces.GPU (CUDA
# emulation), so an implicit device="cuda" triggers a real CUDA init that ZeroGPU
# rejects ("Low-level CUDA init (`torch._C._cuda_init`) reached..."). CPU on
# plain CPU Spaces / local dev / the ZeroGPU main process; CUDA only inside the
# ZeroGPU worker processes (real GPU), where the worker's fork-inherited CPU
# model is re-deviced below.
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
    """Lazy singleton — loads model once (thread-safe), reuses across calls.

    On ZeroGPU, the GPU worker is a fork of the main process, so it can
    inherit the boot-warmed CPU-loaded model. sentence-transformers pins its
    internal target device at construction, so a fork-inherited CPU model is
    discarded and reloaded on the real GPU inside the worker (weights are
    cached on disk, so the reload is fast).
    """
    global _model
    device = pick_device()
    with _model_lock:
        if _model is not None:
            first = next(_model.parameters(), None)
            if first is not None and first.device.type != device:
                _model = None  # fork-inherited model on the wrong device
        if _model is None:
            _model = SentenceTransformer(_MODEL_NAME, device=device)
    return _model


def warm() -> None:
    """Preload the model (best-effort, never raises).

    Called from a daemon thread at boot so the model is already in memory by
    the time the first candidate is ingested — the UI never waits on loading.
    """
    with suppress(Exception):
        get_model()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings. Returns list of 384-dim float vectors."""
    model = get_model()
    with _embed_lock:
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


# Route through ZeroGPU when the runtime is present (auto-detected), with
# automatic CPU fallback on any GPU failure (quota exhausted, ...).
embed_texts = wrap_gpu(embed_texts)


@lru_cache(maxsize=512)
def _embed_single_vec(text: str) -> tuple[float, ...]:
    """Embedded vector as an immutable tuple (cache value)."""
    return tuple(embed_texts([text])[0])


def embed_single(text: str) -> list[float]:
    """Embed a single text string. Returns one fresh 384-dim float vector.

    LRU-cached internally: the same query text is embedded once, then reused
    — e.g. the same JD requirements re-embedded for every candidate in a
    deep-screen batch, or across repeated rank runs. Embedding is
    deterministic, so the cached vector is identical to a fresh computation.
    The cache stores an immutable tuple; each call returns a fresh list, so
    callers can never mutate a shared cached vector. Bounded at 512 entries
    (~1 MB) so a large variety of query texts can't grow memory.
    """
    return list(_embed_single_vec(text))
