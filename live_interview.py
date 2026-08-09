# ================================================================
# 🎙️ Live Meeting Interview — free meeting links + live transcript
# ================================================================
# The recruiter holds the call in a free Jitsi meeting (rooms are created on
# demand — the only provider the app generates links for). While the call
# runs, the app captures the audio from the browser microphone, streams it in
# rolling chunks to Groq Whisper (the same free engine the recorded-interview
# pipeline uses) and shows a live transcript.
# On "Finish & evaluate" the stitched transcript goes through the shared
# pipeline — speaker separation + Q&A parsing (video_interview.parse_qa_pairs)
# and the RAG answer evaluation (screening.evaluate_answers) — and is saved
# to the video_interviews table, exactly like a recorded interview.

from __future__ import annotations

import contextlib
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

import config
import db
import video_interview
from screening import ScreeningResult, evaluate_answers, format_evaluation_markdown

# Seconds of NEW audio to accumulate before a chunk is sent to Whisper. Short
# enough for a "live" feel (the transcript visibly updates every ~10s), long
# enough to stay well inside Groq's free-tier rate limits. Each chunk carries
# the tail of the prior transcript as a continuation hint (see
# _transcribe_audio_bytes) so Whisper keeps context across boundaries.
CHUNK_SECONDS = 10

# Characters of the already-transcribed text passed to Whisper as the `prompt`
# continuation hint — conversational context across chunk boundaries without
# bloating the request.
_PROMPT_TAIL_CHARS = 600

# Longest single Whisper request we send. At 16 kHz mono int16 (~32 KB/s) this
# stays comfortably under Groq's 25 MB upload cap (600 s ≈ 19.2 MB) while
# giving Whisper the widest context per call. Longer recordings are split into
# sequential slices whose transcripts chain into the next as the `prompt` hint.
_TRANSCRIBE_MAX_SINGLE_SECONDS = 600

# Hard cap on captured audio per session (a browser mic at 48 kHz mono is
# ~11 MB/min) — beyond this, chunks are dropped with a visible warning so a
# forgotten recording can never balloon into gigabytes of RAM.
_MAX_RECORDING_SECONDS = 2 * 60 * 60  # 2 hours

# Abandoned sessions (browser closed / user never hit Finish) are reaped by a
# lazy sweep on session create/fetch — anything untouched this long is stopped
# (which also lets its transcriber thread exit) and dropped.
_SESSION_TTL_SECONDS = 2 * 60 * 60  # 2 hours

# In-process registry of active live sessions, keyed by a random session id
# stored in the Gradio State. Live audio/transcripts are per-session in-memory
# state (never pickled through Gradio), so the background transcriber thread
# and the UI events all mutate the same object.
_REGISTRY: dict[str, LiveSession] = {}
_REGISTRY_LOCK = threading.Lock()


@dataclass
class LiveSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    job_id: str = ""
    candidate_id: str = ""
    provider: str = ""
    meeting_link: str = ""
    status: str = "idle"  # idle | recording | stopped | finalizing | completed
    language: str = "English"
    sample_rate: int = 16000
    audio_bytes: bytearray = field(default_factory=bytearray)  # int16 mono PCM
    transcribed_seconds: int = 0  # seconds of audio already sent to Whisper
    transcript: str = ""
    last_error: str | None = None
    # True while a chunk is being sent to Whisper — the status line shows
    # "Transcribing…" so the UI never looks frozen between chunk lands.
    transcribing: bool = False
    last_active: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class LiveInterviewResult:
    transcript: str = ""
    turns: list = field(default_factory=list)
    qa_pairs: list = field(default_factory=list)
    evaluation: str = ""
    average_score: float = 0.0
    verdict: str = ""
    error: str | None = None


# ---- Session registry -------------------------------------------------


def _sweep_stale_sessions() -> None:
    """Stop + drop abandoned sessions (see _SESSION_TTL_SECONDS)."""
    now = time.time()
    stale: list[str] = []
    with _REGISTRY_LOCK:
        for sid, s in _REGISTRY.items():
            with s.lock:
                if (
                    now - s.last_active > _SESSION_TTL_SECONDS
                    and s.status in ("recording", "stopped", "finalizing")
                ):
                    s.status = "stopped"  # lets the transcriber thread exit
                    stale.append(sid)
        for sid in stale:
            _REGISTRY.pop(sid, None)


def get_live_session(session_id: str) -> LiveSession | None:
    _sweep_stale_sessions()
    with _REGISTRY_LOCK:
        return _REGISTRY.get(session_id or "")


def drop_live_session(session_id: str) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.pop(session_id or "", None)


def start_live_session(
    job_id: str,
    candidate_id: str,
    provider: str = "",
    meeting_link: str = "",
    language: str = "English",
) -> LiveSession:
    """Register a new session and launch its background transcriber thread.

    Any earlier un-finished session for the SAME job + candidate is stopped
    (its transcriber thread exits) and dropped first, so starting over never
    leaks a session."""
    _sweep_stale_sessions()
    with _REGISTRY_LOCK:
        to_drop: list[str] = []
        for sid, s in _REGISTRY.items():
            if (
                s.job_id == job_id
                and s.candidate_id == candidate_id
                and s.status in ("recording", "stopped", "finalizing")
            ):
                with s.lock:
                    s.status = "stopped"
                to_drop.append(sid)
        for sid in to_drop:
            _REGISTRY.pop(sid, None)
        session = LiveSession(
            job_id=job_id,
            candidate_id=candidate_id,
            provider=provider,
            meeting_link=meeting_link,
            language=language or "English",
            status="recording",
        )
        _REGISTRY[session.session_id] = session
    threading.Thread(
        target=_transcriber_loop, args=(session, ""), daemon=True
    ).start()
    return session


def session_summary(session: LiveSession) -> str:
    """One-line status for the UI: state + captured/transcribed seconds."""
    with session.lock:
        secs = len(session.audio_bytes) // (2 * session.sample_rate)
        transcribed = session.transcribed_seconds
        status = session.status
        err = session.last_error
        transcribing = session.transcribing
    word = {
        "recording": "🔴 Recording",
        "stopped": "⏹ Stopped",
        "finalizing": "⏳ Finalizing…",
        "completed": "✅ Completed",
    }.get(status, status)
    msg = f"{word} — **{secs}s** captured, **{transcribed}s** transcribed."
    if transcribing:
        msg += " 🔍 _transcribing a segment…_"
    if err:
        msg += f"\n_last segment error: {err}_"
    return msg


def transcript_md(session: LiveSession) -> str:
    with session.lock:
        text = session.transcript
    if not text.strip():
        return "_No speech transcribed yet — is the microphone recording and are you speaking?_"
    return text.strip()


# ---- Free meeting links -------------------------------------------------


def _slugify(text: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return out[:40] or "interview"


def generate_meeting_link(job_id: str, candidate_id: str) -> str:
    """Return a FREE Jitsi meeting link — a room is created on demand
    (https://meet.jit.si/<room>): no API keys, no accounts, no time limit.
    This is the only meeting provider the app generates links for."""
    job = db.get_job(job_id) or {}
    cand = db.get_candidate(candidate_id) or {}
    slug = (
        f"talentiq-{_slugify(job.get('title') or job_id)}"
        f"-{_slugify(cand.get('name') or candidate_id)}"
        f"-{uuid.uuid4().hex[:4]}"
    )
    return f"https://meet.jit.si/{slug}"


# ---- Audio capture + streaming transcription ----------------------------


def append_audio_chunk(session: LiveSession, sample_rate, samples) -> int:
    """Append one browser-mic chunk (np int16/float array) to the session
    buffer; returns the total captured seconds."""
    arr = np.asarray(samples)
    if arr.ndim > 1:
        arr = arr[:, 0] if arr.shape[1] > 1 else arr.reshape(-1)
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.int16:
        raw = arr.tobytes()
    elif arr.dtype.kind == "f":
        raw = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    else:
        raw = arr.astype(np.int16).tobytes()
    with session.lock:
        session.last_active = time.time()
        if not session.audio_bytes and sample_rate:
            session.sample_rate = int(sample_rate)
        total = len(session.audio_bytes) // (2 * session.sample_rate)
        if total >= _MAX_RECORDING_SECONDS:
            session.last_error = (
                f"Recording capped at {_MAX_RECORDING_SECONDS // 60} minutes — "
                "click Finish & evaluate."
            )
            return total
        session.audio_bytes.extend(raw)
        return len(session.audio_bytes) // (2 * session.sample_rate)


def _transcribe_audio_bytes(
    pcm_bytes: bytes,
    sample_rate: int,
    user_api_key: str = "",
    prompt_text: str = "",
    language: str = "English",
) -> tuple[str, str | None]:
    """Whisper a raw int16 PCM chunk via Groq (the same free engine uploads
    use). Returns (text, error); tiny fragments are skipped, not failed.

    `prompt_text` (the tail of the previous transcript) is passed to Whisper
    as a continuation hint so wording/style carry across chunk boundaries
    instead of each slice being heard in isolation, and `language` pins the
    locale hint for better accuracy.
    """
    if not pcm_bytes:
        return "", None
    if len(pcm_bytes) < int(0.5 * sample_rate * 2):  # < 0.5s — nothing to hear
        return "", None
    key = user_api_key.strip().strip(chr(39) + chr(34)) or config.GROQ_API_KEY
    if not key:
        return "", "No Groq API key — live transcription needs GROQ_API_KEY."
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            video_interview.write_wav(f, pcm_bytes, sample_rate)
        from groq import Groq

        client = Groq(api_key=key)
        with open(path, "rb") as f:
            kwargs: dict = {"model": "whisper-large-v3", "file": f}
            tail = (prompt_text or "").strip()[-_PROMPT_TAIL_CHARS:]
            if tail:
                kwargs["prompt"] = tail
            kwargs["language"] = (
                "de" if str(language).lower().startswith("ger") else "en"
            )
            resp = client.audio.transcriptions.create(**kwargs)
        return (getattr(resp, "text", "") or "").strip(), None
    except Exception as e:
        return "", "Live transcription failed: " + str(e)
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)


def _transcribe_full_recording(
    pcm_bytes: bytes,
    sample_rate: int,
    user_api_key: str = "",
    language: str = "English",
) -> tuple[str, str | None]:
    """Full-context transcription of a whole recording.

    Short calls go in ONE request (Whisper is most accurate with the entire
    conversation in context). Longer calls are split into sequential slices;
    each slice's transcript is fed into the next as the `prompt` continuation
    hint so wording/style carry across boundaries. Returns (transcript, error);
    on a slice error the partial transcript so far is returned with the error.
    """
    if not pcm_bytes:
        return "", None
    total_seconds = len(pcm_bytes) // (2 * sample_rate)
    if total_seconds <= _TRANSCRIBE_MAX_SINGLE_SECONDS:
        return _transcribe_audio_bytes(
            pcm_bytes, sample_rate, user_api_key, language=language
        )
    parts: list[str] = []
    tail = ""
    start = 0
    while start < total_seconds:
        end = min(start + _TRANSCRIBE_MAX_SINGLE_SECONDS, total_seconds)
        chunk = pcm_bytes[start * 2 * sample_rate : end * 2 * sample_rate]
        text, err = _transcribe_audio_bytes(
            chunk, sample_rate, user_api_key, prompt_text=tail, language=language
        )
        if err:
            return " ".join(parts).strip(), err
        if text:
            parts.append(text)
            tail = text[-_PROMPT_TAIL_CHARS:]
        start = end
    return " ".join(parts).strip(), None


def _transcribe_pending(session: LiveSession, user_api_key: str = "") -> bool:
    """Transcribe any new audio once it reaches CHUNK_SECONDS (called by the
    background loop; returns whether a chunk was transcribed)."""
    with session.lock:
        if session.status != "recording":
            return False
        total = len(session.audio_bytes) // (2 * session.sample_rate)
        if total - session.transcribed_seconds < CHUNK_SECONDS:
            return False
        start = session.transcribed_seconds * 2 * session.sample_rate
        end = total * 2 * session.sample_rate
        chunk = bytes(session.audio_bytes[start:end])
        # The tail of what we already heard becomes the continuation hint, so
        # this chunk is transcribed with conversational context, not alone.
        prompt_tail = session.transcript[-_PROMPT_TAIL_CHARS:]
        language = session.language
        session.transcribing = True
    try:
        text, err = _transcribe_audio_bytes(
            chunk,
            session.sample_rate,
            user_api_key,
            prompt_text=prompt_tail,
            language=language,
        )
    finally:
        with session.lock:
            session.transcribing = False
            session.last_active = time.time()
    with session.lock:
        # stop/finish may have run while we were mid-call — they own the
        # remainder, so never commit stale text or offsets over them.
        if session.status != "recording":
            return False
        session.transcribed_seconds = total
        if err:
            session.last_error = err
        elif text:
            session.transcript = (session.transcript + " " + text).strip()
    return True


def _transcriber_loop(session: LiveSession, user_api_key: str = "") -> None:
    """Background daemon: keep transcribing rolling chunks while recording."""
    while True:
        time.sleep(1.0)
        if not _transcribe_pending(session, user_api_key):
            with session.lock:
                if session.status != "recording":
                    return


def stop_live_session(
    session_id: str, user_api_key: str = ""
) -> tuple[str, str | None]:
    """Stop the transcriber thread and transcribe the audio captured since
    the last segment. Returns (transcript, error)."""
    session = get_live_session(session_id)
    if session is None:
        return "", "No active live session."
    with session.lock:
        if session.status != "recording":
            return session.transcript, None
        session.status = "stopped"
        session.last_active = time.time()
        total = len(session.audio_bytes) // (2 * session.sample_rate)
        start = session.transcribed_seconds * 2 * session.sample_rate
        rest = bytes(session.audio_bytes[start : total * 2 * session.sample_rate])
        prompt_tail = session.transcript[-_PROMPT_TAIL_CHARS:]
        language = session.language
    if not rest:
        return session.transcript, None
    text, err = _transcribe_audio_bytes(
        rest,
        session.sample_rate,
        user_api_key,
        prompt_text=prompt_tail,
        language=language,
    )
    with session.lock:
        session.transcribed_seconds = total
        session.last_active = time.time()
        if err:
            session.last_error = err
        elif text:
            session.transcript = (session.transcript + " " + text).strip()
    return session.transcript, err


def _remove_file(path: str) -> None:
    try:
        if path:
            os.remove(path)
    except OSError:
        pass


def finish_live_interview(
    session_id: str,
    user_api_key: str = "",
    language: str = "English",
    override_transcript: str = "",
) -> LiveInterviewResult:
    """Finalize a live session: capture the authoritative transcript, persist
    the call recording as WAV, separate speakers + Q&A, evaluate with the
    shared RAG pipeline and save to video_interviews — the exact storage the
    uploaded-recording feature used.

    If `override_transcript` is non-empty it is the recruiter's reviewed &
    corrected copy (from the review & fix box) and is used verbatim — no
    re-transcription runs, the WAV is still saved. Otherwise the ENTIRE
    recording is re-transcribed in one full-context pass, which is far more
    accurate than stitching the live ~10s chunks together.
    """
    session = get_live_session(session_id)
    if session is None:
        return LiveInterviewResult(
            error="No active live session — start one first."
        )
    with session.lock:
        if session.status == "recording":
            session.status = "stopped"
        session.last_active = time.time()
        total = len(session.audio_bytes) // (2 * session.sample_rate)
        if total < 1:
            drop_live_session(session_id)
            return LiveInterviewResult(
                error=(
                    "No audio was captured — press the microphone's record "
                    "button during the call."
                )
            )
        full_pcm = bytes(session.audio_bytes)
        sr = session.sample_rate
        transcript = session.transcript
        job_id, candidate_id = session.job_id, session.candidate_id
        language = language or session.language or "English"

    override = (override_transcript or "").strip()
    if override:
        with session.lock:
            stitched = session.transcript
        if override == (stitched or "").strip():
            # The reviewer changed nothing — the box still holds the live
            # stitch. Fall through to the full-context re-transcription below
            # so the evaluated transcript is the accurate one, not a frozen
            # copy of the lower-quality live transcript.
            override = ""
        else:
            # The recruiter corrected the transcript after Stop — trust it.
            # Mark every byte consumed so an in-flight live chunk can't
            # append stale text on top (the session is dropped right below).
            with session.lock:
                session.transcribed_seconds = total
                session.transcript = override
            transcript = override
    if not override:
        # Authoritative full-context transcription of the whole call. On
        # failure fall back to the live stitched transcript rather than fail
        # (a partial result must never replace the complete stitch).
        text, err = _transcribe_full_recording(
            full_pcm, sr, user_api_key, language=language
        )
        if text and not err:
            transcript = text
    drop_live_session(session_id)

    transcript = (transcript or "").strip()
    if not transcript:
        return LiveInterviewResult(
            error="No speech could be transcribed from this call."
        )

    rec_path, perr = video_interview.save_live_recording(full_pcm, sr)
    if perr:
        return LiveInterviewResult(transcript=transcript, error=perr)

    turns, pairs, err2 = video_interview.parse_qa_pairs(transcript, user_api_key)
    if err2 or not pairs:
        _remove_file(rec_path)
        return LiveInterviewResult(
            transcript=transcript,
            error=err2
            or "No question-answer pairs could be extracted from the transcript.",
        )

    questions = [p["question"] for p in pairs]
    answers_txt = "\n".join(
        f"Q{i + 1}: {p['question']}\nA{i + 1}: {p['answer']}"
        for i, p in enumerate(pairs)
    )

    scr = None
    screening = db.latest_screening(job_id, candidate_id)
    if screening:
        scr = ScreeningResult(
            score=int(screening.get("score", 0) or 0),
            verdict=screening.get("verdict", "FAIL"),
            summary=screening.get("summary", ""),
            candidate_id=candidate_id,
            raw_json=db.parse_json_field(screening.get("report_json"), {}),
        )

    eval_res = evaluate_answers(
        questions,
        answers_txt,
        user_api_key,
        screening_result=scr,
        language=language or "English",
    )
    if eval_res.error:
        _remove_file(rec_path)
        return LiveInterviewResult(
            transcript=transcript, qa_pairs=pairs, error=eval_res.error
        )

    md = (
        f"### 🎙️ Live Interview Evaluation — {len(pairs)} Q&A pairs\n\n"
        + video_interview.format_qa_markdown(pairs, turns)
        + "\n\n---\n\n"
        + format_evaluation_markdown(eval_res)
    )
    eval_data = dict(eval_res.raw_json or {})
    eval_data["per_question"] = eval_res.per_question
    eval_data["average_score"] = eval_res.average_score
    eval_data["turns"] = turns

    try:
        db.save_video_interview(
            job_id=job_id,
            candidate_id=candidate_id,
            video_path=rec_path,
            transcript=transcript,
            qa_pairs=pairs,
            eval_data=eval_data,
            average_score=eval_res.average_score,
            verdict=eval_res.verdict,
        )
    except Exception:
        _remove_file(rec_path)  # a failed save must not orphan the file
        raise

    return LiveInterviewResult(
        transcript=transcript,
        turns=turns,
        qa_pairs=pairs,
        evaluation=md,
        average_score=eval_res.average_score,
        verdict=eval_res.verdict,
    )
