# ================================================================
# 🔀 Cross-Encoder Rerank — improve top-k retrieval quality
# ================================================================
# ruff: noqa: I001 — import order is intentional: zerogpu (→ `import spaces`)
# must load BEFORE sentence-transformers (torch) so HF's spaces-before-CUDA
# rule holds on ZeroGPU Spaces; isort would reorder it.

from __future__ import annotations

import os
import threading
from contextlib import suppress

import config

# ZeroGPU helpers import the `spaces` package BEFORE sentence-transformers so
# HF's spaces-before-CUDA import rule holds. wrap_gpu() auto-routes the model
# calls through @spaces.GPU on ZeroGPU Spaces with CPU fallback (see zerogpu.py).
from zerogpu import pick_device, wrap_gpu

from sentence_transformers import CrossEncoder

# Multilingual cross-encoder — works for EN and DE retrieval hits
# (~470 MB first download, cached afterwards). Override via RERANK_MODEL.
_MODEL_NAME = config.RERANK_MODEL
# Device is picked by PROBING for a real GPU (zerogpu.pick_device): on ZeroGPU
# Spaces, torch.cuda.is_available() returns True even OUTSIDE @spaces.GPU (CUDA
# emulation), so an implicit device="cuda" raises "Low-level CUDA init
# (`torch._C._cuda_init`) reached...". CPU on plain CPU Spaces / local dev /
# the ZeroGPU main process; CUDA only inside ZeroGPU worker processes (real
# GPU), where the fork-inherited CPU model is re-deviced below.
# Set RERANK_ENABLED=0 to skip the cross-encoder entirely (saves ~80 MB RAM
# and CPU time on Hugging Face free tier — pure vector retrieval only).
_ENABLED = os.getenv("RERANK_ENABLED", "1").lower() not in ("0", "false", "no", "off")
# Scoring input cap (see config.RERANK_MAX_CHARS).
_MAX_SCORE_CHARS = config.RERANK_MAX_CHARS
_model: CrossEncoder | None = None
_model_lock = threading.Lock()


def get_model() -> CrossEncoder:
    """Lazy singleton — loads model once (thread-safe), reuses across calls.

    The lock mirrors embeddings.get_model() so the boot-time warm() thread can
    never race the first deep-screen into loading the ~470 MB cross-encoder
    twice on a CPU Space. On ZeroGPU, the worker is a fork of the main process
    and can inherit the boot-warmed CPU-loaded model; CrossEncoder pins its
    internal device at construction, so an inherited CPU model is discarded
    and reloaded on the real GPU inside the worker (weights are cached).
    """
    global _model
    device = pick_device()
    with _model_lock:
        if _model is not None:
            first = next(_model.parameters(), None)
            if first is not None and first.device.type != device:
                _model = None  # fork-inherited model on the wrong device
        if _model is None:
            _model = CrossEncoder(_MODEL_NAME, device=device)
    return _model


def warm() -> None:
    """Preload the cross-encoder (best-effort, never raises).

    Called from a daemon thread at boot so the first deep-screen never pays
    model loading inline — on CPU Spaces that load can take tens of seconds.
    No-op when RERANK_ENABLED=0.
    """
    if not _ENABLED:
        return
    with suppress(Exception):
        get_model()


def rerank(query: str, hits: list[dict], top_k: int = 3) -> list[dict]:
    """Rerank retrieval hits with a cross-encoder.

    Args:
        query: The search query (e.g. a JD requirement).
        hits: List of dicts with at least a "text" field.
        top_k: Number of results to keep after reranking.

    Returns:
        Top-k hits sorted by cross-encoder score (descending).
        Each hit gains/overwrites a "rerank_score" field.
        When RERANK_ENABLED=0, returns hits[:top_k] unchanged (fast path).
    """
    if not hits:
        return []
    if not _ENABLED:
        return hits[:top_k]
    if len(hits) == 1:
        hits[0]["rerank_score"] = hits[0].get("score", 0.0)
        return hits[:top_k]

    model = get_model()
    # Score only the head of each chunk: cost scales with input length, and
    # the chunk head carries the relevance signal. Full text is untouched in
    # the returned hits.
    pairs = [(query, h["text"][:_MAX_SCORE_CHARS]) for h in hits]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(hits, scores, strict=False),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    out: list[dict] = []
    for hit, score in ranked[:top_k]:
        enriched = dict(hit)
        enriched["rerank_score"] = round(float(score), 4)
        # Prefer rerank score for display when available
        enriched["score"] = enriched["rerank_score"]
        out.append(enriched)

    return out


# Route through ZeroGPU when the runtime is present (auto-detected), with
# automatic CPU fallback on any GPU failure (quota exhausted, ...).
rerank = wrap_gpu(rerank)
