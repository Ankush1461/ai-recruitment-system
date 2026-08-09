# ================================================================
# 📊 Corpus Ranking — Hybrid semantic + keyword shortlist
# ================================================================

from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass, field

import chunking
import db
import skill_model
import vectorstore
from screening import stable_candidate_id

# Category-aware keyword credit (skill_model.py): when literal token overlap is
# zero but the fine-tuned skill classifier (if available) sees the requirement
# and the resume in the same skill family, credit a partial match — e.g. JD
# says "Amazon Web Services", resume says "AWS"; JD says "PostgreSQL", resume
# says "postgres". Fail-open: without a trained model this never fires.
_SKILL_CATEGORY_CREDIT = 0.5
_SKILL_CAT_MIN_CONF = 0.5
_SKILL_REQ_TOKENS_MAX = 4
_SKILL_RESUME_TOKENS_MAX = 24


@dataclass
class RankResult:
    candidate_id: str
    name: str
    hybrid_score: float
    semantic_score: float
    keyword_score: float
    top_evidence: list[dict] = field(default_factory=list)

    def to_stored(self) -> dict:
        return asdict(self)


# A requirement token counts as a hit only when it appears at a token
# boundary in the resume. Letters are "sticky" so SQL never matches NoSQL
# and Java never matches JavaScript; digits and the common compound
# separators (- + / . #) are boundaries so "React" still matches
# "react-native", "Python" matches "python3", and "C#" matches "c#".
#
# Accepted tradeoff: this is stricter than substring matching, so genuine
# hits that are substrings of a larger word ("postgres" in "PostgreSQL",
# "ML" in "MLflow", German compounds like "datenbank" in "Datenbanken")
# no longer count as keyword hits — the semantic retrieval score covers
# those. Do NOT loosen this back to substring matching without reintroducing
# the NoSQL/JavaScript-class false positives.
_TOKEN_START = r"(?<![a-z])"
_TOKEN_END = r"(?![a-z])"

_STOP_WORDS = {
    "and", "or", "the", "a", "an", "to", "of", "in", "for", "with",
    "on", "at", "by", "from", "as", "is", "are", "be", "this", "that",
    "using", "use", "experience", "strong", "ability", "years", "year",
    # German stopwords / generic CV filler (resumes may be in German)
    "und", "oder", "der", "die", "das", "den", "dem", "des", "ein",
    "eine", "einen", "einer", "einem", "eines", "mit", "für", "fuer",
    "von", "vom", "auf", "zu", "zum", "zur", "im", "am",
    "bei", "aus", "als", "wie", "nicht", "ist", "sind", "war", "waren",
    "wird", "werden", "wurde", "sowie", "auch", "nach", "über", "ueber",
    "durch", "ohne", "gegen", "zwischen", "unter", "sehr", "gute",
    "guter", "gutes", "starke", "starker", "starkes", "umfangreiche",
    "kenntnisse", "kenntnissen", "erfahrung", "erfahrungen", "fähigkeiten",
    "fähigkeit", "jahre", "jahren", "arbeit", "arbeiten", "team",
    "projekt", "projekte", "bereich", "thema", "schwerpunkt", "praxis",
    "zusammenarbeit", "eigenständig", "selbstständig", "verantwortung",
    "aufgaben", "tätigkeiten", "methoden", "tools", "systeme", "umfeld",
}


def _tokens_of(text: str) -> list[str]:
    """Lowercased content tokens (stopwords + pure digits removed)."""
    return [
        t for t in re.findall(r"[a-z0-9+#./-]{2,}", text.lower())
        if t not in _STOP_WORDS and not t.isdigit()
    ]


def _category_credit(req_tokens: list[str], resume_label_set: set[str]) -> float:
    """Partial keyword credit when the fine-tuned skill classifier sees the
    requirement and the resume in the same skill family but no literal token
    matched (e.g. JD 'Amazon Web Services' vs resume 'AWS')."""
    if not resume_label_set:
        return 0.0
    for t in req_tokens[:_SKILL_REQ_TOKENS_MAX]:
        pair = skill_model.classify_token(t)
        if pair is None:
            continue
        label, conf = pair
        if conf >= _SKILL_CAT_MIN_CONF and label in resume_label_set:
            return _SKILL_CATEGORY_CREDIT
    return 0.0


def _keyword_overlap(
    requirement: str,
    resume_text: str,
    resume_label_set: set[str] | None = None,
) -> float:
    """Token-boundary overlap ratio for must-have keyword signal.

    When literal overlap is zero, falls back to skill-category matching via
    the fine-tuned classifier (``resume_label_set`` = category labels already
    detected in the resume) — so a JD requirement still earns partial credit
    when only *related* wording appears in the resume.
    """
    tokens = _tokens_of(requirement)
    if not tokens:
        return 0.0
    resume_l = resume_text.lower()
    hits = sum(
        1 for t in tokens
        if re.search(_TOKEN_START + re.escape(t) + _TOKEN_END, resume_l)
    )
    ratio = hits / len(tokens)
    if ratio == 0.0 and resume_label_set:
        ratio = _category_credit(tokens, resume_label_set)
    return ratio


# Candidates whose vectors are being embedded on background threads (deferred
# ingest / deferred re-index). Ranking waits briefly for them so a just-added
# candidate is scored with real semantic vectors instead of keyword-only.
_pending_index: set[str] = set()
_pending_lock = threading.Lock()
_INDEX_WAIT_MAX = 3.0  # seconds — long enough for a warm-model embed, capped
# so ranking can never stall behind a slow (cold-model) load.


def _ingest_core(
    resume_text: str,
    name: str | None,
    source: str,
    job_id: str | None,
) -> tuple[dict, str, list[dict]]:
    """Validate + chunk + write the SQLite row and job link. Returns
    (candidate, candidate_id, chunks) — shared by the sync and deferred
    ingest paths."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Empty resume text")

    candidate_id = stable_candidate_id(resume_text)
    chunks = chunking.chunk_resume(resume_text, candidate_id)
    if not chunks:
        raise ValueError("Could not chunk resume")

    cand = db.upsert_candidate(
        candidate_id, resume_text, name=name, source=source, job_id=job_id
    )
    # Link into the owning job's pipeline immediately so the Jobs tab shows it
    # right away (not only after the next ranking run).
    if job_id:
        db.add_candidate_to_job(job_id, candidate_id, status="shortlisted")
    return cand, candidate_id, chunks


def _spawn_background_index(token: str, candidate_id: str, chunks: list[dict]) -> None:
    """Embed + index on a daemon thread inside the caller's user scope."""
    with _pending_lock:
        _pending_index.add(candidate_id)

    def _index() -> None:
        try:
            import auth  # local import — ui/ranking both depend on auth

            with auth.user_scope(token):
                vectorstore.index_resume(candidate_id, chunks)
        except Exception as e:
            # The candidate is safely in SQLite (the source of truth); the
            # index can be rebuilt later (maybe_reindex_all, or the next
            # ranking run's lazy index) — never fail the ingest over a
            # background embed.
            print(f"[index] background embed failed for {candidate_id}: {e}", flush=True)
        finally:
            with _pending_lock:
                _pending_index.discard(candidate_id)

    threading.Thread(target=_index, daemon=True).start()


def _wait_for_pending_index(ids: set[str]) -> None:
    """Briefly wait until every candidate with pending background vectors is
    indexed (or the cap expires), so a rank run scores them semantically."""
    if not ids:
        return
    deadline = time.time() + _INDEX_WAIT_MAX
    while time.time() < deadline:
        with _pending_lock:
            pending = ids & _pending_index
        if not pending:
            return
        time.sleep(0.2)


def ingest_candidate_deferred(
    token: str,
    resume_text: str,
    name: str | None = None,
    source: str = "upload",
    job_id: str | None = None,
) -> dict:
    """Ingest a candidate without making the UI wait on embedding.

    The SQLite row and the job-pipeline link are written synchronously, so the
    candidate appears in every tab instantly. Chunking + embedding + the
    Chroma upsert run on a daemon thread that re-enters the caller's user
    scope; the next ranking run briefly waits for the vectors to land (see
    _wait_for_pending_index), so keyword-only scoring is a rare fallback.
    Returns the stored candidate dict.
    """
    cand, candidate_id, chunks = _ingest_core(resume_text, name, source, job_id)
    _spawn_background_index(token, candidate_id, chunks)
    return cand


def update_candidate_record(
    candidate_id: str,
    *,
    name: str | None = None,
    resume_text: str | None = None,
    deferred_token: str | None = None,
) -> dict:
    """Edit candidate; re-index Chroma when resume text changes.

    Pass `deferred_token` (the caller's session token) to run the re-embed on
    a background thread instead of blocking the UI — the record update itself
    is always synchronous.
    """
    existing = db.get_candidate(candidate_id)
    if not existing:
        raise ValueError("Candidate not found")

    text_changed = (
        resume_text is not None
        and resume_text.strip()
        and resume_text.strip() != existing["resume_text"].strip()
    )
    updated = db.update_candidate(
        candidate_id,
        name=name,
        resume_text=resume_text if (text_changed or resume_text is not None) else None,
    )
    if not updated:
        raise ValueError("Update failed")

    if text_changed:
        chunks = chunking.chunk_resume(updated["resume_text"], candidate_id)
        if not chunks:
            raise ValueError("Could not re-chunk updated resume")
        if deferred_token:
            _spawn_background_index(deferred_token, candidate_id, chunks)
        else:
            vectorstore.index_resume(candidate_id, chunks, force=True)

    return updated


def remove_candidate(candidate_id: str) -> None:
    """Delete candidate from SQLite and Chroma."""
    vectorstore.clear_candidate(candidate_id)
    db.delete_candidate(candidate_id)


def rank_candidates_for_job(job_id: str, top_n: int | None = None) -> list[RankResult]:
    """Rank all stored candidates against a job using hybrid retrieval.

    Does NOT call the LLM — fast shortlist for deep screening.
    hybrid = 0.7 * semantic + 0.3 * keyword  (both on 0-100 scale)
    """
    job = db.get_job(job_id)
    if not job:
        return []

    requirements = chunking.split_jd_requirements(job["description"])
    if not requirements:
        requirements = [{
            "id": "req_0",
            "requirement": job["description"][:300],
            "text": job["description"][:300],
        }]

    # Candidates are job-specific — rank only this job's candidates.
    candidates = db.list_candidates(job_id=job_id)
    # A candidate ingested seconds ago may still be embedding on a background
    # thread — wait briefly so it is scored with real vectors, not keyword-only.
    _wait_for_pending_index({c["id"] for c in candidates})
    # Fine-tuned skill classifier (optional, fail-open): one batched pass per
    # candidate to detect its skill-category labels, then cheap token-level
    # lookups per requirement — never blocks ranking when no model is present.
    use_skill_cats = skill_model.available()
    results: list[RankResult] = []

    for cand_meta in candidates:
        full = db.get_candidate(cand_meta["id"])
        if not full:
            continue
        cid = full["id"]
        resume = full["resume_text"]

        resume_label_set: set[str] = set()
        if use_skill_cats:
            # Only confident resume-side predictions count — a spurious
            # low-confidence label would otherwise let the category credit fire
            # on weak evidence (e.g. a summary word like "built").
            resume_label_set = {
                pair[0]
                for pair in skill_model.classify_tokens(
                    _tokens_of(resume)[:_SKILL_RESUME_TOKENS_MAX]
                ).values()
                if pair[1] >= _SKILL_CAT_MIN_CONF
            }

        if vectorstore.get_candidate_count(cid) == 0:
            chunks = chunking.chunk_resume(resume, cid)
            if chunks:
                vectorstore.index_resume(cid, chunks)

        per_req_scores: list[float] = []
        evidence: list[dict] = []
        for req in requirements:
            hits = vectorstore.search_resume(
                req["text"], cid, top_k=2, use_rerank=False
            )
            if hits:
                best = max(
                    max(0.0, min(1.0, float(h["score"]))) for h in hits
                )
                per_req_scores.append(best)
                evidence.append({
                    "requirement": req["requirement"][:80],
                    "section": hits[0]["section"],
                    "text": hits[0]["text"][:160],
                    "score": hits[0]["score"],
                })
            else:
                per_req_scores.append(0.0)

        semantic = (
            (sum(per_req_scores) / len(per_req_scores)) * 100 if per_req_scores else 0.0
        )

        global_hits = vectorstore.search_resume(
            job["description"][:500], cid, top_k=3, use_rerank=False
        )
        if global_hits:
            g = sum(max(0.0, min(1.0, h["score"])) for h in global_hits) / len(global_hits)
            semantic = 0.6 * semantic + 0.4 * (g * 100)

        keyword = (
            sum(
                _keyword_overlap(r["requirement"], resume, resume_label_set)
                for r in requirements
            )
            / len(requirements)
        ) * 100

        hybrid = 0.7 * semantic + 0.3 * keyword

        results.append(
            RankResult(
                candidate_id=cid,
                name=full.get("name") or "Unknown",
                hybrid_score=round(hybrid, 1),
                semantic_score=round(semantic, 1),
                keyword_score=round(keyword, 1),
                top_evidence=evidence[:4],
            )
        )

    results.sort(key=lambda r: r.hybrid_score, reverse=True)
    if top_n is not None:
        results = results[:top_n]
    return results


def rank_and_save_shortlist(
    job_id: str,
    top_n: int | None = None,
) -> list[RankResult]:
    """Rank candidates for one job and persist the shortlist snapshot.

    Every ranked candidate is also linked to the job's pipeline so each job
    keeps its own candidate list (independent of other jobs).
    """
    results = rank_candidates_for_job(job_id, top_n=top_n)
    db.save_shortlist(
        job_id,
        [r.to_stored() for r in results],
        top_n=top_n,
    )
    for r in results:
        db.add_candidate_to_job(job_id, r.candidate_id, status="shortlisted")
    return results


def rank_jobs_batch(
    job_ids: list[str],
    top_n: int | None = None,
) -> dict[str, list[RankResult]]:
    """Rank and save a separate shortlist for each selected job."""
    out: dict[str, list[RankResult]] = {}
    for jid in job_ids:
        if not jid:
            continue
        out[jid] = rank_and_save_shortlist(jid, top_n=top_n)
    return out


def load_shortlist_results(job_id: str, db_path: str | None = None) -> list[RankResult]:
    """Load the latest saved shortlist for a job (empty if none).

    Entries whose candidate was deleted after the snapshot was saved are
    dropped at load time, so a removed candidate can never reappear in the
    Shortlist or Interview tabs (the stored JSON is a snapshot; SQLite is the
    source of truth for what still exists).
    """
    row = db.get_latest_shortlist(job_id, db_path)
    if not row:
        return []
    raw = db.parse_json_field(row.get("results_json"), [])
    results: list[RankResult] = []
    for item in raw:
        candidate_id = item.get("candidate_id", "")
        if not candidate_id or db.get_candidate(candidate_id, db_path) is None:
            continue
        results.append(
            RankResult(
                candidate_id=candidate_id,
                name=item.get("name", "Unknown"),
                hybrid_score=float(item.get("hybrid_score", 0)),
                semantic_score=float(item.get("semantic_score", 0)),
                keyword_score=float(item.get("keyword_score", 0)),
                top_evidence=item.get("top_evidence") or [],
            )
        )
    return results


def format_ranking_table(results: list[RankResult]) -> list[list]:
    rows = []
    for i, r in enumerate(results, 1):
        rows.append([
            i,
            r.name,
            r.hybrid_score,
            r.semantic_score,
            r.keyword_score,
        ])
    return rows


def format_ranking_markdown(results: list[RankResult], job_title: str = "") -> str:
    title = f" — {job_title}" if job_title else ""
    if not results:
        return f"*No shortlist{title}. Rank candidates for this job first.*"
    lines = [f"### Shortlist{title}\n"]
    lines.append("| # | Candidate | Hybrid | Semantic | Keyword |")
    lines.append("|---|---|---:|---:|---:|")
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r.name} | {r.hybrid_score} | {r.semantic_score} | "
            f"{r.keyword_score} |"
        )
    lines.append(
        "\n_Hybrid = 70% retrieval similarity + 30% keyword overlap. "
        "Deep-screen top candidates for rubric LLM scores._"
    )
    return "\n".join(lines)


def format_multi_job_shortlists_markdown(
    results_by_job: dict[str, list[RankResult]],
) -> str:
    if not results_by_job:
        return "*Select one or more jobs and run ranking.*"
    parts = ["## Multi-job shortlists\n"]
    parts.append(
        "_Each job keeps an independent ranked shortlist — candidates are "
        "scored against that JD only._\n"
    )
    for jid, results in results_by_job.items():
        job = db.get_job(jid)
        title = job["title"] if job else jid
        parts.append(format_ranking_markdown(results, job_title=title))
        parts.append("\n---\n")
    return "\n".join(parts)
