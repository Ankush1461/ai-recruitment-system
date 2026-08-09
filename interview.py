# ================================================================
# 🎤 Multi-Turn Interview — Chat-style technical interview
# ================================================================

from __future__ import annotations

from dataclasses import dataclass, field

import db
import prompts
from llm import get_fast_llm_client
from screening import (
    evaluate_answers,
    format_evaluation_markdown,
    screening_result_from_db,
    suggest_interview_questions,
)


@dataclass
class InterviewSession:
    interview_id: str = ""
    job_id: str = ""
    candidate_id: str = ""
    screening_id: str = ""
    questions: list[str] = field(default_factory=list)
    answers: list[list[str]] = field(default_factory=list)  # one bucket per question
    q_index: int = 0
    followup_used: bool = False
    messages: list[dict] = field(default_factory=list)  # Gradio chatbot format
    status: str = "idle"  # idle | in_progress | completed
    language: str = "English"  # English | German — drives questions & feedback
    eval_markdown: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "interview_id": self.interview_id,
            "job_id": self.job_id,
            "candidate_id": self.candidate_id,
            "screening_id": self.screening_id,
            "questions": self.questions,
            "answers": self.answers,
            "q_index": self.q_index,
            "followup_used": self.followup_used,
            "messages": self.messages,
            "status": self.status,
            "language": self.language,
            "eval_markdown": self.eval_markdown,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> InterviewSession:
        if not data:
            return cls()
        defaults = cls()
        return cls(
            interview_id=data.get("interview_id", defaults.interview_id),
            job_id=data.get("job_id", defaults.job_id),
            candidate_id=data.get("candidate_id", defaults.candidate_id),
            screening_id=data.get("screening_id", defaults.screening_id),
            questions=data.get("questions", defaults.questions) or [],
            answers=_coerce_answer_buckets(data.get("answers", defaults.answers)),
            q_index=int(data.get("q_index", defaults.q_index) or 0),
            followup_used=bool(data.get("followup_used", defaults.followup_used)),
            messages=data.get("messages", defaults.messages) or [],
            status=data.get("status", defaults.status) or "idle",
            language=data.get("language", defaults.language) or "English",
            eval_markdown=data.get("eval_markdown", defaults.eval_markdown) or "",
            error=data.get("error"),
        )


def _coerce_answer_buckets(raw) -> list[list[str]]:
    """Normalize the serialized answers into per-question buckets.

    New sessions store `[[a1, a2], [b1], ...]` (follow-up replies share their
    question's bucket). Legacy sessions stored a flat `[a1, a2, b1, ...]` list
    — each entry becomes its own bucket so older state never crashes.
    """
    if isinstance(raw, str):  # malformed state — never char-split it
        return []
    raw = raw or []
    if raw and isinstance(raw[0], list):
        return [[str(a) for a in bucket] for bucket in raw]
    return [[str(a)] for a in raw]


def start_interview(
    job_id: str,
    candidate_id: str,
    user_api_key: str = "",
    language: str = "English",
) -> InterviewSession:
    """Start a multi-turn interview for a candidate in the job's top-N list.

    If the candidate has not been deep-screened for this job yet, a screening
    runs on demand (one LLM call) so the top-N shortlist is directly
    interviewable. Candidates with an explicit FAIL verdict are blocked.
    """
    screening = db.latest_screening(job_id, candidate_id)
    if screening and screening.get("verdict") == "FAIL":
        return InterviewSession(
            error="This candidate did not pass deep screening for this job."
        )
    if not screening or screening.get("verdict") != "PASS":
        # Auto-screen on demand so the top-N shortlist is interview-ready
        from screening import deep_screen_candidate

        auto = deep_screen_candidate(job_id, candidate_id, user_api_key)
        if auto.error:
            return InterviewSession(error=f"Auto-screening failed: {auto.error}")
        screening = db.latest_screening(job_id, candidate_id)
        if not screening or screening.get("verdict") != "PASS":
            return InterviewSession(
                error=(
                    "Candidate did not pass the on-demand screening for this job "
                    f"(score {auto.score}/100, verdict {auto.verdict})."
                )
            )

    # AI suggests exactly 10 tailored questions (JD + resume + screening focus).
    # Falls back to a rule-based set if the LLM can't be reached. The questions
    # are written in the selected interview language.
    questions, _qerr = suggest_interview_questions(
        job_id, candidate_id, user_api_key, language=language
    )
    if len(questions) < 1:
        return InterviewSession(error="No interview questions available from screening.")

    row = db.create_interview(
        job_id=job_id,
        candidate_id=candidate_id,
        screening_id=screening["id"],
        questions=questions,
    )

    first_q = questions[0]
    if language == "German":
        greeting = (
            f"**Interview gestartet** — {len(questions)} Kernfragen "
            f"(mit optionalen Nachfragen).\n\n"
        )
    else:
        greeting = (
            f"**Interview started** — {len(questions)} core questions "
            f"(with optional follow-ups).\n\n"
        )
    messages = [
        {
            "role": "assistant",
            "content": (
                greeting
                + f"**Question 1/{len(questions)}:** {first_q}"
            ),
        }
    ]

    db.update_interview(row["id"], messages=messages, status="in_progress")

    return InterviewSession(
        interview_id=row["id"],
        job_id=job_id,
        candidate_id=candidate_id,
        screening_id=screening["id"],
        questions=questions,
        answers=[[] for _ in questions],
        q_index=0,
        followup_used=False,
        messages=messages,
        status="in_progress",
        language=language,
    )


def _generic_followup(language: str) -> str:
    """Fallback follow-up probe when the LLM can't be reached."""
    if language == "German":
        return (
            "Können Sie ein konkretes Beispiel oder eine Kennzahl aus Ihrer "
            "Erfahrung nennen?"
        )
    return "Can you add a concrete example or metric from your experience?"


def _needs_followup(
    answer: str,
    question: str,
    user_api_key: str = "",
    language: str = "English",
) -> tuple[bool, str]:
    """Ask LLM if a short follow-up is warranted. Fail open (no follow-up)."""
    if len(answer.split()) >= 80:
        return False, ""

    # Low-stakes decision -> cheap/fast model (hybrid routing)
    client, err = get_fast_llm_client(user_api_key)
    if client is None or err:
        # Heuristic: very short answers get a generic probe
        if len(answer.split()) < 25:
            return True, _generic_followup(language)
        return False, ""

    try:
        data = client.chat_json(
            prompts.build_followup_user_prompt(question, answer, language=language),
            system=prompts.FOLLOWUP_SYSTEM_PROMPT,
            temperature=0.2,
        )
        if data.get("followup") and data.get("question"):
            return True, str(data["question"]).strip()
    except Exception:
        if len(answer.split()) < 25:
            return True, _generic_followup(language)
    return False, ""


def submit_answer(
    session: InterviewSession | dict,
    answer: str,
    user_api_key: str = "",
) -> InterviewSession:
    """Process one candidate answer; may ask follow-up or advance / finish."""
    if isinstance(session, dict):
        session = InterviewSession.from_dict(session)

    if session.status != "in_progress":
        session.error = "No active interview. Start one first."
        return session

    if not answer or not answer.strip():
        session.error = "Please enter an answer."
        return session

    session.error = None
    answer = answer.strip()
    messages = list(session.messages)
    messages.append({"role": "user", "content": answer})

    q_index = session.q_index
    questions = session.questions
    # Deep-copy the buckets so mutating them never aliases session state
    answers = [list(b) for b in session.answers]

    # Follow-up path (one per core question)
    if not session.followup_used:
        need, follow_q = _needs_followup(
            answer, questions[q_index], user_api_key, session.language
        )
        if need and follow_q:
            messages.append({
                "role": "assistant",
                "content": f"**Follow-up:** {follow_q}",
            })
            # Keep same q_index; the reply stays in the same question bucket
            answers[q_index].append(answer)
            session.followup_used = True
            session.messages = messages
            session.answers = answers
            db.update_interview(
                session.interview_id,
                messages=messages,
                answers=answers,
            )
            return session

    # Store answer (if follow-up already used, this is the follow-up reply)
    answers[q_index].append(answer)
    session.followup_used = False

    # Advance to next question or finish
    if q_index + 1 < len(questions):
        next_i = q_index + 1
        messages.append({
            "role": "assistant",
            "content": (
                f"**Question {next_i + 1}/{len(questions)}:** {questions[next_i]}"
            ),
        })
        session.q_index = next_i
        session.messages = messages
        session.answers = answers
        db.update_interview(
            session.interview_id,
            messages=messages,
            answers=answers,
        )
        return session

    # All questions done → evaluate
    session.messages = messages
    session.answers = answers
    session.status = "completed"

    screening = db.get_screening(session.screening_id) if session.screening_id else None
    screening_result = screening_result_from_db(screening)

    # Build the evaluation input straight from the structured question/answer
    # model (each question owns its answer bucket, follow-up replies included)
    # instead of re-parsing the chat transcript.
    combined_answers = _combine_answers_model(questions, session.answers)
    eval_result = evaluate_answers(
        questions,
        combined_answers,
        user_api_key,
        screening_result=screening_result,
        language=session.language,
    )
    eval_md = format_evaluation_markdown(eval_result)
    messages.append({
        "role": "assistant",
        "content": (
            "**Interview complete.** Here is your evaluation:\n\n" + eval_md
        ),
    })

    session.messages = messages
    session.eval_markdown = eval_md

    # Persist the NORMALIZED evaluation (clamped per-question scores + the
    # recomputed average) so the stored eval_json agrees with the
    # average_score column — never the raw, unclamped LLM payload.
    if eval_result.raw_json:
        eval_data = dict(eval_result.raw_json)
        eval_data["per_question"] = eval_result.per_question
        eval_data["average_score"] = eval_result.average_score
    else:
        eval_data = {
            "per_question": eval_result.per_question,
            "feedback": eval_result.feedback,
            "error": eval_result.error,
        }

    db.update_interview(
        session.interview_id,
        messages=messages,
        answers=answers,
        eval_data=eval_data,
        average_score=eval_result.average_score,
        verdict=eval_result.verdict,
        status="completed",
    )
    return session


def _combine_answers_model(questions: list[str], answers: list[list[str]]) -> str:
    """Build the evaluation input from the structured question/answer model.

    No transcript parsing: `answers` is one bucket per question, so follow-up
    replies stay with the question they answered and the evaluator always
    sees each answer mapped to the exact question it belongs to.
    """
    parts: list[str] = []
    for i, q in enumerate(questions):
        parts.append(f"Q{i + 1}: {q}")
        bucket = answers[i] if i < len(answers) else []
        texts = [str(a).strip() for a in bucket if str(a).strip()]
        if texts:
            parts.append(f"A{i + 1}: {' '.join(texts)}")
        else:
            parts.append(f"A{i + 1}: (no answer)")
    return "\n".join(parts)


