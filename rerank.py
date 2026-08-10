# ================================================================
# 🔀 Cross-Encoder Rerank — improve top-k retrieval quality
# ================================================================

from __future__ import annotations

import os
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

from sentence_transformers import CrossEncoder

import config

# Multilingual cross-encoder — works for EN and DE retrieval hits
# (~470 MB first download, cached afterwards). Override via RERANK_MODEL.
_MODEL_NAME = config.RERANK_MODEL
# Set RERANK_ENABLED=0 to skip the cross-encoder entirely (saves ~80 MB RAM
# and CPU time on Hugging Face free tier — pure vector retrieval only).
_ENABLED = os.getenv("RERANK_ENABLED", "1").lower() not in ("0", "false", "no", "off")
_model: CrossEncoder | None = None


def get_model() -> CrossEncoder:
    """Lazy singleton — loads model once, reuses across calls."""
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


@spaces.GPU
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
    pairs = [(query, h["text"]) for h in hits]
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
