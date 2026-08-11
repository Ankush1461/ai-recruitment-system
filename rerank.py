# ================================================================
# 🔀 Cross-Encoder Rerank — improve top-k retrieval quality
# ================================================================

from __future__ import annotations

import os
import threading
from contextlib import suppress

import config

# Optional ZeroGPU acceleration (Hugging Face GPU Spaces only). Enabled via
# ZEROGPU_ENABLED=1; default OFF runs the cross-encoder on plain CPU — no
# ZeroGPU quota consumed, works on CPU Spaces. The import stays above
# sentence-transformers so HF's spaces-before-CUDA import rule holds.
if config.ZEROGPU_ENABLED:
    try:
        import spaces as _spaces  # type: ignore
    except Exception:
        _spaces = None
else:
    _spaces = None

from sentence_transformers import CrossEncoder

# Multilingual cross-encoder — works for EN and DE retrieval hits
# (~470 MB first download, cached afterwards). Override via RERANK_MODEL.
_MODEL_NAME = config.RERANK_MODEL
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
    twice on a CPU Space.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = CrossEncoder(_MODEL_NAME)
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


# Route through ZeroGPU only when explicitly enabled (default: plain CPU).
if _spaces is not None:
    rerank = _spaces.GPU(rerank)
