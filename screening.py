# ================================================================
# 🎯 RAG Screening Pipeline — Retrieve → Rubric JSON → Persist
# ================================================================

from __future__ import annotations

import hashlib
import json
import math
from contextlib import suppress
from dataclasses import dataclass, field

import chunking
import db
import prompts
import rubric
import vectorstore
from llm import get_llm_client


@dataclass
class ScreeningResult:
    """Structured output from resume screening."""
    score: int = 0
    verdict: str = "FAIL"
    summary: str = ""
    strengths: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    interview_focus: list[str] = field(default_factory=list)
    evidence_snippets: list[dict] = field(default_factory=list)
    candidate_id: str = ""
    rubric: dict = field(default_factory=dict)
    screening_db_id: str = ""
    raw_json: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalResult:
    """Structured output from interview evaluation."""
    per_question: list[dict] = field(default_factory=list)
    average_score: float = 0.0
    verdict: str = ""
    feedback: str = ""
    raw_json: dict = field(default_factory=dict)
    error: str | None = None


def stable_candidate_id(resume_text: str) -> str:
    digest = hashlib.sha256(resume_text.strip().encode("utf-8")).hexdigest()
    return digest[:16]


def screen_candidate(
    resume_text: str,
    job_description: str,
    user_api_key: str = "",
    job_id: str | None = None,
    persist: bool = True,
) -> ScreeningResult:
    """RAG screening with weighted rubric; optionally persists to SQLite."""
    if not resume_text or not resume_text.strip():
        return ScreeningResult(error="No resume text provided.")
    if not job_description or not job_description.strip():
        return ScreeningResult(error="No job description provided.")

    client, err = get_llm_client(user_api_key)
    if client is None or err:
        return ScreeningResult(error=err or "No LLM available.")

    candidate_id = stable_candidate_id(resume_text)

    if persist:
        with suppress(Exception):
            db.upsert_candidate(candidate_id, resume_text, source="screen")

    chunks = chunking.chunk_resume(resume_text, candidate_id)
    if not chunks:
        return ScreeningResult(error="Could not chunk resume text.", candidate_id=candidate_id)

    vectorstore.index_resume(candidate_id, chunks)

    requirements = chunking.split_jd_requirements(job_description)
    if not requirements:
        requirements = [{
            "id": "req_0",
            "requirement": job_description.strip(),
            "text": job_description.strip(),
        }]

    evidence_blocks: list[str] = []
    all_evidence: list[dict] = []

    for req in requirements:
        hits = vectorstore.search_resume(req["text"], candidate_id, top_k=3, use_rerank=True)
        block = f'Requirement: "{req["requirement"]}"\nRelevant resume evidence:'
        if hits:
            for h in hits:
                block += f'\n  - [{h["section"]}] "{h["text"][:1500]}"'
                all_evidence.append({
                    "requirement": req["requirement"],
                    "section": h["section"],
                    "text": h["text"][:1500],
                    "similarity": h["score"],
                })
        else:
            block += "\n  - (No matching evidence found in resume)"
        evidence_blocks.append(block)

    evidence_prompt = "\n\n".join(evidence_blocks)

    system_prompt = prompts.SCREENING_SYSTEM_PROMPT
    user_prompt = prompts.build_screening_user_prompt(evidence_prompt, job_description)

    try:
        result_json = client.chat_json(user_prompt, system=system_prompt, temperature=0.3)
    except Exception as e:
        try:
            resp = client.chat(user_prompt, system=system_prompt, temperature=0.3)
            text = resp.text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result_json = json.loads(text[start:end])
            else:
                return ScreeningResult(
                    error=f"LLM did not return valid JSON: {e!s}",
                    evidence_snippets=all_evidence,
                    candidate_id=candidate_id,
                )
        except Exception as e2:
            return ScreeningResult(
                error=f"LLM evaluation failed: {e2!s}",
                evidence_snippets=all_evidence,
                candidate_id=candidate_id,
            )

    normalized = rubric.normalize_rubric(result_json.get("rubric"))
    score = rubric.compute_weighted_score(normalized)
    verdict = rubric.apply_verdict(score, normalized)

    report = {
        "strengths": result_json.get("strengths", []),
        "gaps": result_json.get("gaps", []),
        "interview_focus": result_json.get("interview_focus", []),
        "evidence_snippets": all_evidence,
        "rubric": normalized,
        "summary": result_json.get("summary", ""),
        "score": score,
        "verdict": verdict,
    }

    screening_db_id = ""
    if persist and job_id:
        try:
            job = db.get_job(job_id)
            if job and not db.parse_json_field(job.get("requirements_json"), []):
                with db.connect() as conn:
                    conn.execute(
                        "UPDATE jobs SET requirements_json = ? WHERE id = ?",
                        (json.dumps(requirements), job_id),
                    )
            row = db.save_screening(
                job_id=job_id,
                candidate_id=candidate_id,
                score=score,
                verdict=verdict,
                rubric=normalized,
                report=report,
                summary=result_json.get("summary", ""),
            )
            screening_db_id = row.get("id", "")
            # Candidate belongs to this job's pipeline (each job keeps its own list)
            db.add_candidate_to_job(job_id, candidate_id, status=verdict.lower())
        except Exception:
            pass

    return ScreeningResult(
        score=score,
        verdict=verdict,
        summary=result_json.get("summary", ""),
        strengths=result_json.get("strengths", []),
        gaps=result_json.get("gaps", []),
        interview_focus=result_json.get("interview_focus", []),
        evidence_snippets=all_evidence,
        candidate_id=candidate_id,
        rubric=normalized,
        screening_db_id=screening_db_id,
        raw_json=result_json,
    )


def deep_screen_candidate(
    job_id: str,
    candidate_id: str,
    user_api_key: str = "",
) -> ScreeningResult:
    job = db.get_job(job_id)
    cand = db.get_candidate(candidate_id)
    if not job:
        return ScreeningResult(error="Job not found.")
    if not cand:
        return ScreeningResult(error="Candidate not found.")
    return screen_candidate(
        cand["resume_text"],
        job["description"],
        user_api_key=user_api_key,
        job_id=job_id,
        persist=True,
    )


def generate_interview_questions(screening_result: ScreeningResult) -> list[str]:
    questions = list(screening_result.interview_focus)
    for gap in screening_result.gaps:
        if len(questions) >= 3:
            break
        q = (
            f"Can you describe your experience with "
            f"{gap.get('requirement', 'this area')}? {gap.get('detail', '')}"
        )
        questions.append(q.strip())
    return questions[:3]


def suggest_interview_questions(
    job_id: str,
    candidate_id: str,
    user_api_key: str = "",
    language: str = "English",
) -> tuple[list[str], str | None]:
    """AI-generate exactly 10 tailored interview questions for a candidate.

    Uses the fast/cheap model (hybrid routing — question generation is a
    creative task, not scoring). Falls back to the rule-based questions
    (padded to 10) if the LLM is unavailable or returns too few.
    """
    fallback = _fallback_questions(job_id, candidate_id, language)
    job = db.get_job(job_id) or {}
    cand = db.get_candidate(candidate_id)
    if not job or not cand:
        return fallback, None

    focus, gaps = [], []
    scr = db.latest_screening(job_id, candidate_id)
    if scr:
        report = db.parse_json_field(scr.get("report_json"), {})
        focus = [str(f) for f in report.get("interview_focus", [])][:5]
        gaps = [
            f"{g.get('requirement', '')}: {g.get('detail', '')}"
            for g in report.get("gaps", [])[:5]
        ]

    from llm import get_fast_llm_client

    client, err = get_fast_llm_client(user_api_key)
    if client is None or err:
        return fallback, None
    try:
        data = client.chat_json(
            prompts.build_suggest_questions_prompt(
                job.get("title", ""),
                job.get("description", ""),
                cand.get("resume_text", ""),
                focus,
                gaps,
                language=language,
            ),
            system=prompts.SUGGEST_QUESTIONS_SYSTEM_PROMPT,
            temperature=0.7,
        )
        questions = [
            str(q).strip()
            for q in (data.get("questions") or [])
            if str(q).strip()
        ][:10]
        if len(questions) >= 5:
            return questions, None
    except Exception:
        pass
    return fallback, None


def _fallback_questions(job_id: str, candidate_id: str, language: str = "English") -> list[str]:
    """Rule-based question set (used when the LLM can't be reached)."""
    scr = db.latest_screening(job_id, candidate_id)
    result = screening_result_from_db(scr) if scr else None
    questions = list(generate_interview_questions(result)) if result else []
    if language == "German":
        generic = [
            "Beschreiben Sie eine anspruchsvolle technische Entscheidung, die Sie kürzlich getroffen haben, und warum.",
            "Wie bleiben Sie fachlich auf dem neuesten Stand und lernen neue Technologien?",
            "Beschreiben Sie eine Situation mit einem schwierigen Stakeholder. Wie sind Sie damit umgegangen?",
            "Auf welches Projekt sind Sie am stolzesten, und welchen konkreten Beitrag haben Sie geleistet?",
            "Wie gehen Sie an die Fehlersuche bei einem Produktionsproblem heran, das Sie noch nie gesehen haben?",
            "Erzählen Sie von einer Situation, in der Sie eine Anforderung ablehnen mussten. Was ist passiert?",
            "Wie würden Sie Ihre Kernkompetenz einem nicht-technischen Kollegen erklären?",
            "Wie arbeiten Sie am liebsten mit funktionsübergreifenden Teams zusammen?",
            "Wie möchten Sie Ihre Fähigkeiten in den nächsten zwei Jahren weiterentwickeln?",
            "Beschreiben Sie ein Projekt mit knappem Zeitrahmen – wie haben Sie geplant und geliefert?",  # noqa: RUF001 — intentional typographic en dash in German copy
        ]
    else:
        generic = [
            "Walk me through a challenging technical decision you made recently and why.",
            "How do you keep your skills current and learn new technologies?",
            "Describe a time you worked with a difficult stakeholder. How did you handle it?",
            "What project are you most proud of, and what was your specific contribution?",
            "How do you approach debugging a production issue you have never seen before?",
            "Tell me about a time you had to push back on a requirement. What happened?",
            "How would you explain your core skill to a non-technical colleague?",
            "What is your preferred way to collaborate with cross-functional teams?",
            "Where do you see your skills growing in the next two years?",
            "Describe a project that had a tight deadline — how did you plan and deliver it?",
        ]
    out: list[str] = []
    for q in questions + generic:
        if len(out) >= 10:
            break
        if q not in out:
            out.append(q)
    return out[:10]


def screening_result_from_db(row: dict | None) -> ScreeningResult | None:
    """Rebuild a ScreeningResult from a `screenings` row (shared helper)."""
    if not row:
        return None
    report = db.parse_json_field(row.get("report_json"), {})
    return ScreeningResult(
        score=int(row.get("score", 0)),
        verdict=row.get("verdict", "FAIL"),
        summary=row.get("summary", ""),
        strengths=report.get("strengths", []),
        gaps=report.get("gaps", []),
        interview_focus=report.get("interview_focus", []),
        evidence_snippets=report.get("evidence_snippets", []),
        candidate_id=row.get("candidate_id", ""),
        raw_json=report,
    )


def _build_resume_context_for_eval(
    questions: list[str],
    screening_result: ScreeningResult | None,
) -> str:
    blocks: list[str] = []
    if screening_result:
        if screening_result.strengths:
            blocks.append("### Screening strengths (claimed evidence)")
            for s in screening_result.strengths[:5]:
                blocks.append(
                    f"- {s.get('skill', '')}: {s.get('evidence', '')} "
                    f"[{s.get('section', '')}]"
                )
        if screening_result.gaps:
            blocks.append("\n### Screening gaps to probe")
            for g in screening_result.gaps[:5]:
                blocks.append(f"- {g.get('requirement', '')}: {g.get('detail', '')}")

    cid = screening_result.candidate_id if screening_result else ""
    if cid and vectorstore.get_candidate_count(cid) > 0:
        blocks.append("\n### Resume evidence retrieved for each question")
        for i, q in enumerate(questions, 1):
            hits = vectorstore.search_resume(q, cid, top_k=2, use_rerank=True)
            blocks.append(f"\nQ{i} evidence:")
            if hits:
                for h in hits:
                    blocks.append(f'  - [{h["section"]}] "{h["text"][:250]}"')
            else:
                blocks.append("  - (no matching resume chunks)")
    elif screening_result and screening_result.evidence_snippets:
        blocks.append("\n### Resume evidence from screening")
        seen: set[str] = set()
        for ev in screening_result.evidence_snippets[:8]:
            key = ev.get("text", "")[:80]
            if key in seen:
                continue
            seen.add(key)
            blocks.append(
                f'- [{ev.get("section", "")}] "{ev.get("text", "")[:200]}"'
            )
    return "\n".join(blocks) if blocks else "(No resume evidence available)"


def evaluate_answers(
    questions: list[str],
    answers: str,
    user_api_key: str = "",
    screening_result: ScreeningResult | None = None,
    language: str = "English",
) -> EvalResult:
    if not questions:
        return EvalResult(error="No interview questions provided.")
    if not answers or not answers.strip():
        return EvalResult(error="No answers provided.")

    client, err = get_llm_client(user_api_key)
    if client is None or err:
        return EvalResult(error=err or "No LLM available.")

    q_str = "\n".join(f"Q{i+1}: {q}" for i, q in enumerate(questions))
    resume_context = _build_resume_context_for_eval(questions, screening_result)

    system_prompt = prompts.EVALUATION_SYSTEM_PROMPT
    user_prompt = prompts.build_evaluation_user_prompt(
        q_str, answers, resume_context, language=language
    )

    try:
        result_json = client.chat_json(user_prompt, system=system_prompt, temperature=0.3)
    except Exception as e:
        try:
            resp = client.chat(user_prompt, system=system_prompt, temperature=0.3)
            text = resp.text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result_json = json.loads(text[start:end])
            else:
                return EvalResult(error=f"LLM did not return valid JSON: {e!s}")
        except Exception as e2:
            return EvalResult(error=f"Evaluation failed: {e2!s}")

    per_question, average_score = _normalize_per_question_scores(
        result_json.get("per_question")
    )
    return EvalResult(
        per_question=per_question,
        average_score=average_score,
        verdict=result_json.get("verdict", ""),
        feedback=result_json.get("feedback", ""),
        raw_json=result_json,
    )


def _normalize_per_question_scores(raw_list: list | None) -> tuple[list[dict], float]:
    """Clamp per-question scores into 1-10 and recompute the overall average.

    The model is asked for integer scores in 1-10; out-of-range, non-finite
    (NaN/Inf), or missing values are clamped to the 1-10 bounds (a missing or
    malformed score gets the minimum 1 — the model should grade every asked
    question, so an omission is treated as the weakest possible score, never
    excluded). The average is recomputed from the (clamped) per-question
    scores instead of trusting the model's self-reported `average_score`.

    Returns (normalized per-question list, recomputed average).
    """
    out: list[dict] = []
    scores: list[float] = []
    for pq in raw_list or []:
        if not isinstance(pq, dict):
            continue
        try:
            raw = float(pq.get("score", 0))
        except (TypeError, ValueError):
            raw = 0.0
        if not math.isfinite(raw):  # NaN/Inf would poison the average
            raw = 0.0
        clamped = max(1.0, min(10.0, raw))
        scores.append(clamped)
        normalized = dict(pq)
        normalized["score"] = (
            int(clamped) if clamped == int(clamped) else round(clamped, 1)
        )
        out.append(normalized)
    average = round(sum(scores) / len(scores), 1) if scores else 0.0
    return out, average


def format_screening_markdown(result: ScreeningResult) -> str:
    if result.error:
        return f"**Error**: {result.error}"

    md = f"### Candidate Status: {rubric.verdict_badge(result.verdict)} · Score: **{result.score}/100**\n\n"

    if result.summary:
        md += f"**Executive Fit Summary:** {result.summary}\n\n"

    if result.rubric:
        md += rubric.format_rubric_markdown(result.rubric)

    if result.strengths:
        md += "---\n### Key Matching Strengths\n"
        for s in result.strengths:
            skill = s.get("skill", "")
            evidence = s.get("evidence", "")
            md += f"- **{skill}** — _{evidence}_\n"
        md += "\n"

    if result.gaps:
        md += "### Skill Gaps & Missing Qualifications\n"
        for g in result.gaps:
            md += f"- **{g.get('requirement', '')}**: {g.get('detail', '')}\n"
        md += "\n"

    if result.evidence_snippets:
        md += "---\n### Evidence Snippets (RAG + Rerank)\n"
        seen: set[str] = set()
        for ev in result.evidence_snippets[:8]:
            key = ev.get("text", "")[:80]
            if key in seen:
                continue
            seen.add(key)
            md += f"- **{ev.get('requirement', '')}** → _{ev.get('text', '')[:150]}..._\n"
        md += "\n"

    if result.verdict == "PASS":
        md += "### Next Step\nCandidate is **qualified** — proceed to the multi-turn AI interview.\n"
    else:
        md += "### Recommendation\nCandidate does **not meet** the minimum threshold "
        md += "(overall score and/or must-have skills).\n"

    return md


def format_evaluation_markdown(result: EvalResult) -> str:
    if result.error:
        return f"**Error**: {result.error}"

    md = "### Technical Interview Evaluation (RAG-Grounded)\n\n"
    if result.per_question:
        md += "#### Question-by-Question Scores\n"
        for pq in result.per_question:
            md += (
                f"- **Q{pq.get('question_num', '?')} Score: {pq.get('score', 0)}/10** "
                f"— {pq.get('assessment', '')}\n"
            )
        md += "\n"

    md += f"**Average Score: {result.average_score:.1f}/10**\n\n"
    md += f"### Final Verdict: {rubric.verdict_badge(result.verdict)}\n\n"
    if result.feedback:
        md += f"### Constructive Feedback\n{result.feedback}\n"
    return md
