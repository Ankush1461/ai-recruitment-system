# ================================================================
# 📜 Prompt Registry — versioned LLM prompts (single source of truth)
# ================================================================
# Prompts are data, not inline strings: keep them here so they can be
# tuned, versioned, and A/B tested without touching pipeline code.

from __future__ import annotations

import rubric

# -- Resume screening (rubric JSON) -----------------------------------------

SCREENING_SYSTEM_PROMPT = (
    "You are a Lead AI Executive Recruiter. Evaluate resume evidence against "
    "job requirements using a weighted rubric. Respond with valid JSON only."
)


def build_screening_user_prompt(evidence_prompt: str, job_description: str) -> str:
    """Assemble the screening prompt from per-requirement evidence blocks."""
    return f"""Evaluate this candidate based on the evidence retrieved from their resume.

### Evidence per Requirement:
{evidence_prompt}

### Full Job Description:
{job_description}

Note: The resume may be written in English, German, or a mix of both —
evaluate the content regardless of the language it is written in.

Score each rubric dimension 0-10 from the evidence above, calibrated generously:
- 8-10: strong, clearly-evidenced match
- 5-7: acceptable fit — partial coverage, related or transferable skills, or minor gaps
- 4: borderline fit — some coverage with notable gaps (still viable)
- 0-3: significant gap

Calibration rules:
- Do not require an exact keyword match: count related skills, adjacent experience,
  and transferable evidence as partial credit.
- A candidate need not cover every single requirement to be viable; a couple of
  minor gaps should not pull a dimension below 5.
- When the resume is silent on a point but nothing contradicts it, lean toward the
  middle of the scale rather than the bottom.
- Education/extras is a small-weight dimension — never let it decide the verdict alone.

Dimension weights:
- must_have_skills (40%): core required technical skills
- experience (25%): years / role relevance
- projects (20%): concrete project / impact evidence
- education_extras (15%): education, certs, extras

Respond with this exact JSON structure:
{{
  {rubric.rubric_prompt_block()}
  "summary": "<2-3 sentence executive fit summary>",
  "strengths": [
    {{"skill": "<matched skill>", "evidence": "<specific resume quote>", "section": "<resume section>"}}
  ],
  "gaps": [
    {{"requirement": "<missing requirement>", "detail": "<what's missing>"}}
  ],
  "interview_focus": [
    "<specific question to probe gap or verify claim>",
    "<second question>",
    "<third question>"
  ]
}}"""


# -- Interview answer evaluation ---------------------------------------------

EVALUATION_SYSTEM_PROMPT = (
    "You are a Senior Technical Hiring Director evaluating interview responses. "
    "Grade against resume evidence. JSON only."
)


def build_evaluation_user_prompt(
    q_str: str,
    answers: str,
    resume_context: str,
    language: str = "English",
) -> str:
    """Assemble the interview-evaluation prompt."""
    language = "German" if language == "German" else "English"
    return f"""Evaluate these candidate responses using the resume evidence.

### Questions Asked:
{q_str}

### Candidate Answers:
{answers}

Note: Answers and resume evidence may be in English or German — assess the
content itself, not the language it is written in.

Write the per-question assessments, verdict, and feedback in {language}.

### Resume Evidence (RAG):
{resume_context}

Respond with this exact JSON structure:
{{
  "per_question": [
    {{
      "question_num": 1,
      "score": <integer 1-10>,
      "assessment": "<accuracy, depth, clarity, and alignment with resume evidence>"
    }}
  ],
  "average_score": <float, average of per_question scores>,
  "verdict": "RECOMMENDED FOR HIRE" or "REJECTED / FURTHER REVIEW NEEDED" (RECOMMENDED if average >= 7),
  "feedback": "<constructive feedback summary>"
}}"""


# -- Interview follow-up detection -------------------------------------------

FOLLOWUP_SYSTEM_PROMPT = "You are a technical interviewer. JSON only."


def build_followup_user_prompt(
    question: str,
    answer: str,
    language: str = "English",
) -> str:
    """Assemble the follow-up decision prompt (vague/short answers)."""
    language = "German" if language == "German" else "English"
    return f"""Question: {question}
Answer: {answer}

Decide if one short follow-up is needed because the answer is vague or shallow.
Write the follow-up question in {language}.
Respond JSON: {{"followup": true/false, "question": "<follow-up or empty>"}}"""


# -- Uploaded video interview: transcript -> Q&A pairs -------------------------

QA_PARSE_SYSTEM_PROMPT = (
    "You extract structured interview transcripts into question-answer pairs. "
    "Respond with valid JSON only."
)


def build_qa_parse_prompt(transcript: str) -> str:
    """Assemble the prompt that splits a raw transcript into Q&A pairs.

    The LLM returns both a speaker-labelled turn-by-turn transcript (`turns`)
    and the question/answer pairs extracted from it (`qa_pairs`).
    """
    return f"""Below is the raw transcript of a job interview. Reconstruct it into
speaker-labelled turns and split it into interviewer question and candidate
answer pairs.

Rules:
- The transcript is a RAW streaming speech-to-text output: no punctuation,
  occasional recognition errors, and sentences may be split across chunk
  boundaries. Rejoin fragments that clearly belong to the same sentence and
  drop filler (\"um\", \"okay\", \"uh\").
- `turns`: every utterance in order, labelled \"Interviewer\" or \"Candidate\".
  Merge consecutive utterances from the same speaker. Two speakers alternate
  in an interview — infer who is who from content, not formatting.
- `qa_pairs`: one entry per interviewer question, with the candidate's answer
  that follows it. Skip greetings and statements that are not questions.
- Skip pairs where the candidate's answer is missing.
- Keep the original wording as much as possible (the interview may be in
  English or German); only fix obvious recognition errors and dropped
  punctuation.

Respond with this exact JSON structure:
{{
  "turns": [
    {{"speaker": "Interviewer|Candidate", "text": "<utterance>"}}
  ],
  "qa_pairs": [
    {{"question": "<interviewer question>", "answer": "<candidate answer>"}}
  ]
}}

### Transcript:
{transcript}"""


# -- Interview question suggestions (10 per interview) -------------------------

SUGGEST_QUESTIONS_SYSTEM_PROMPT = (
    "You are a senior technical interviewer. You design exactly 10 tailored "
    "interview questions for a candidate based on the job description and resume. "
    "Respond with valid JSON only."
)


def build_suggest_questions_prompt(
    job_title: str,
    job_description: str,
    resume_text: str,
    screening_focus: list[str] | None = None,
    screening_gaps: list[str] | None = None,
    language: str = "English",
) -> str:
    """Assemble the prompt that produces 10 interview questions."""
    language = "German" if language == "German" else "English"
    focus = "\n".join(f"- {f}" for f in (screening_focus or [])[:5]) or "(none)"
    gaps = "\n".join(f"- {g}" for g in (screening_gaps or [])[:5]) or "(none)"
    resume_head = (resume_text or "").strip()[:2000]
    return f"""Job title: {job_title}

Job description:
{job_description}

Resume (first 2000 chars):
{resume_head}

Screening focus areas to probe:
{focus}

Known gaps / missing qualifications:
{gaps}

Design exactly 10 interview questions for this candidate that:
- Cover the must-have skills of the job description.
- Probe the screening focus areas and the known gaps.
- Mix technical depth, experience verification, and soft skills.
- Are specific to this candidate's resume, not generic.

Write every question in {language}.

Respond with this exact JSON structure:
{{
  "questions": ["<Q1>", "<Q2>", ... "<Q10>"]
}}"""
