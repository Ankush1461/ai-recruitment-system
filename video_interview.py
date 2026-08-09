# ================================================================
# 🎙️ Video / Audio Interview Engine — shared live-interview pipeline
# ================================================================
# Shared transcript + recording engine for the live meeting interview:
# the LLM splits a raw transcript into interviewer question / candidate
# answer pairs (speaker-labelled turns), and captured browser-mic audio is
# persisted as WAV into the active media dir (per-user when signed in, the
# global MEDIA_DIR otherwise). The end-to-end flow lives in live_interview.py;
# this module holds the reusable pieces (Q&A parsing, WAV writing, recording
# persistence, markdown rendering).

from __future__ import annotations

import os
import threading
import uuid
import wave

import config
import prompts
from llm import get_llm_client

# Recordings are copied into a persistent media folder (per-user when signed
# in, the global MEDIA_DIR otherwise) so the stored video_path survives temp
# cleanup. Thread-local so two concurrent requests never write into each
# other's folder (see auth.user_scope).
_thread = threading.local()


def _active_media_dir() -> str:
    return getattr(_thread, "media_dir", None) or str(config.MEDIA_DIR)


def set_active_media_dir(path: str | None) -> None:
    """Point THIS THREAD's recording storage at a folder (per-user isolation,
    switched by auth.set_active_user). Passing None restores the default."""
    _thread.media_dir = str(path) if path else None


def parse_qa_pairs(
    transcript: str, user_api_key: str = ""
) -> tuple[list, list, str | None]:
    """Split a raw transcript into speaker-labelled turns + Q&A pairs.

    Returns (turns, qa_pairs, error). `turns` is the full reconstructed
    interview with every utterance tagged as Interviewer or Candidate;
    `qa_pairs` is the question/answer pairs extracted from those turns.
    """
    client, err = get_llm_client(user_api_key)
    if client is None or err:
        return [], [], err
    try:
        data = client.chat_json(
            prompts.build_qa_parse_prompt(transcript),
            system=prompts.QA_PARSE_SYSTEM_PROMPT,
            temperature=0.1,
        )
        turns = []
        for t in data.get("turns") or []:
            speaker = str(t.get("speaker", "")).strip()
            text = str(t.get("text", "")).strip()
            if speaker and text:
                turns.append({"speaker": speaker, "text": text})
        pairs = []
        for p in data.get("qa_pairs") or []:
            q = str(p.get("question", "")).strip()
            a = str(p.get("answer", "")).strip()
            if q and a:
                pairs.append({"question": q, "answer": a})
        return turns, pairs, None
    except Exception as e:
        return [], [], "Could not structure the transcript into Q&A: " + str(e)


def format_qa_markdown(pairs: list, turns: list | None = None) -> str:
    """Render the speaker-labelled transcript + extracted Q&A as markdown."""
    parts: list[str] = []
    if turns:
        parts.append("### 🎙️ Interview Transcript (auto-separated speakers)")
        parts.append("")
        for t in turns:
            speaker = "**Interviewer:**" if str(t.get("speaker", "")).lower().startswith("inter") else "**Candidate:**"
            parts.append(f"{speaker} {t.get('text', '')}")
            parts.append("")
    parts.append("### 🎙️ Extracted Q&A Session")
    parts.append("")
    for i, p in enumerate(pairs, 1):
        parts.append(f"**Q{i}:** {p['question']}")
        parts.append("")
        parts.append(f"> {p['answer']}")
        parts.append("")
    return chr(10).join(parts)


def write_wav(fileobj, pcm_bytes: bytes, sample_rate: int) -> None:
    """Write int16 mono PCM bytes as a WAV (stdlib wave module)."""
    with wave.open(fileobj, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)


def save_live_recording(pcm_bytes: bytes, sample_rate: int) -> tuple[str, str | None]:
    """Persist raw int16 PCM captured by the live browser mic as a WAV in the
    active media dir (per-user when signed in, global MEDIA_DIR otherwise).
    Returns (stored_path, error); stored_path is "" on error."""
    try:
        if not pcm_bytes or not sample_rate:
            return "", "No audio captured to save."
        dest_dir = _active_media_dir()
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"live_{uuid.uuid4().hex[:16]}.wav")
        with open(dest, "wb") as f:
            write_wav(f, pcm_bytes, sample_rate)
        return dest, None
    except Exception as e:
        return "", f"Could not store the recording: {e}"
