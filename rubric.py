# ================================================================
# 📐 Rubric Scoring — Weighted hiring dimensions
# ================================================================

from __future__ import annotations

from typing import Any

# Weights must sum to 1.0
RUBRIC_WEIGHTS: dict[str, float] = {
    "must_have_skills": 0.40,
    "experience": 0.25,
    "projects": 0.20,
    "education_extras": 0.15,
}

RUBRIC_LABELS: dict[str, str] = {
    "must_have_skills": "Must-have skills",
    "experience": "Relevant experience",
    "projects": "Projects / impact",
    "education_extras": "Education / extras",
}

# Strictness knobs — lower these to pass more candidates, raise to filter harder.
PASS_THRESHOLD = 55
MUST_HAVE_MIN = 4  # out of 10 — hard gate for PASS


def compute_weighted_score(dimensions: dict[str, Any]) -> int:
    """Compute 0-100 overall score from per-dimension 0-10 scores.

    Accepts either:
      {"must_have_skills": 8, ...}
    or:
      {"must_have_skills": {"score": 8, "evidence": "..."}, ...}
    """
    total = 0.0
    for key, weight in RUBRIC_WEIGHTS.items():
        raw = dimensions.get(key, 0)
        score = (
            float(raw.get("score", 0)) if isinstance(raw, dict) else float(raw or 0)
        )
        score = max(0.0, min(10.0, score))
        total += score * weight

    # dimensions are /10; weight sum = 1 → scale to /100
    return round(total * 10)


def must_have_score(dimensions: dict[str, Any]) -> float:
    raw = dimensions.get("must_have_skills", 0)
    if isinstance(raw, dict):
        return float(raw.get("score", 0))
    return float(raw or 0)


def apply_verdict(score: int, dimensions: dict[str, Any] | None = None) -> str:
    """PASS requires overall >= threshold AND must-have skills >= min."""
    if score < PASS_THRESHOLD:
        return "FAIL"
    if dimensions is not None and must_have_score(dimensions) < MUST_HAVE_MIN:
        return "FAIL"
    return "PASS"


def normalize_rubric(raw: dict | None) -> dict[str, dict]:
    """Normalize LLM rubric payload into consistent shape."""
    out: dict[str, dict] = {}
    raw = raw or {}
    for key in RUBRIC_WEIGHTS:
        item = raw.get(key, {})
        if isinstance(item, dict):
            out[key] = {
                "score": max(0, min(10, int(item.get("score", 0)))),
                "evidence": str(item.get("evidence", "")),
            }
        else:
            out[key] = {
                "score": max(0, min(10, int(item or 0))),
                "evidence": "",
            }
    return out


def verdict_badge(verdict: str | None) -> str:
    """Render a verdict as a styled HTML badge for markdown output.

    PASS / RECOMMEND → green badge; FAIL / REJECT → red badge;
    empty → plain em-dash (no badge).
    """
    v = (verdict or "").strip()
    if not v:
        return "—"
    low = v.lower()
    if "recommend" in low or "pass" in low:
        return f"<span class='badge badge-pass'>{v}</span>"
    return f"<span class='badge badge-fail'>{v}</span>"


def format_rubric_markdown(rubric: dict[str, dict]) -> str:
    if not rubric:
        return ""
    lines = ["### 📐 Rubric Breakdown\n"]
    for key, weight in RUBRIC_WEIGHTS.items():
        item = rubric.get(key, {})
        label = RUBRIC_LABELS.get(key, key)
        score = item.get("score", 0)
        evidence = item.get("evidence", "")
        lines.append(
            f"- **{label}** ({int(weight * 100)}%): **{score}/10**"
            + (f" — _{evidence}_" if evidence else "")
        )
    lines.append("")
    return "\n".join(lines)


def rubric_prompt_block() -> str:
    """JSON schema fragment for the screening LLM prompt."""
    return """
  "rubric": {
    "must_have_skills": {"score": <0-10>, "evidence": "<resume quote or gap>"},
    "experience": {"score": <0-10>, "evidence": "<resume quote or gap>"},
    "projects": {"score": <0-10>, "evidence": "<resume quote or gap>"},
    "education_extras": {"score": <0-10>, "evidence": "<resume quote or gap>"}
  },
""".strip()
