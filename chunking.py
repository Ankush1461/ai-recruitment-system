# ================================================================
# ✂️ Section-Aware Resume Chunking & JD Requirement Splitting
# ================================================================

from __future__ import annotations

import re
import uuid

# Common resume section headers (case-insensitive) — English + German
_SECTION_PATTERNS = [
    # English
    r"ABOUT\s+ME",
    r"(?:PROFESSIONAL\s+)?SUMMARY",
    r"OBJECTIVE",
    r"TECHNICAL\s+SKILLS?",
    r"SKILLS?",
    r"WORK\s+EXPERIENCE",
    r"EXPERIENCE",
    r"EMPLOYMENT(?:\s+HISTORY)?",
    r"PROJECTS?",
    r"EDUCATION(?:\s+(?:AND|&)\s+TRAINING)?",
    r"CERTIFICATIONS?",
    r"PUBLICATIONS?",
    r"AWARDS?(?:\s+(?:AND|&)\s+HONORS?)?",
    r"ACHIEVEMENTS?",
    r"VOLUNTEER(?:ING)?",
    r"INTERESTS?",
    r"LANGUAGES?",
    # German (Lebenslauf)
    r"(?:PERSÖNLICHES\s+)?PROFIL",
    r"ZUSAMMENFASSUNG",
    r"TECHNISCHE\s+(?:KENNTNISSE|FÄHIGKEITEN|FERTIGKEITEN)",
    r"KENNTNISSE",
    r"FÄHIGKEITEN",
    r"FERTIGKEITEN",
    r"BERUFSERFAHRUNG",
    r"ERFAHRUNG",
    r"WERDEGANG",
    r"PROJEKTE",
    r"AUSBILDUNG",
    r"STUDIUM",
    r"ZERTIFIZIERUNGEN",
    r"PUBLIKATIONEN",
    r"AUSZEICHNUNGEN",
    r"EHRENAMT",
    r"SPRACHEN",
    r"INTERESSEN",
]

_SECTION_RE = re.compile(
    r"^(?:" + "|".join(_SECTION_PATTERNS) + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalise_section(header: str) -> str:
    """Map a detected header (EN or DE) to a canonical section name."""
    h = header.strip().rstrip(":").upper()
    if ("SUMMARY" in h or "OBJECTIVE" in h or "ABOUT" in h
            or "ZUSAMMENFASSUNG" in h or "PROFIL" in h):
        return "SUMMARY"
    if ("SKILL" in h or "KENNTNISSE" in h
            or "FÄHIGKEITEN" in h or "FERTIGKEITEN" in h):
        return "SKILLS"
    if ("EXPERIENCE" in h or "EMPLOYMENT" in h
            or "ERFAHRUNG" in h or "WERDEGANG" in h):
        return "EXPERIENCE"
    if "PROJECT" in h or "PROJEKT" in h:
        return "PROJECTS"
    if "EDUCATION" in h or "AUSBILDUNG" in h or "STUDIUM" in h:
        return "EDUCATION"
    if "CERTIF" in h or "ZERTIFIZIERUNG" in h:
        return "CERTIFICATIONS"
    if "PUBLIKATION" in h:
        return "PUBLICATIONS"
    return h


def chunk_resume(text: str, candidate_id: str | None = None) -> list[dict]:
    """Split resume text into section-aware chunks with metadata.

    Returns a list of dicts:
        {"id": str, "section": str, "text": str, "candidate_id": str, "index": int}

    If no section headers are detected, falls back to 400-token
    sliding windows with 50-token overlap.
    """
    if not text or not text.strip():
        return []

    cid = candidate_id or uuid.uuid4().hex[:12]

    # Try section-aware split first
    chunks = _split_by_sections(text, cid)
    if chunks and not _is_degenerate_split(chunks):
        return chunks

    # Fallback: sliding window. Windows are kept small (120 words) so a short
    # resume still yields several retrievable chunks — a single giant chunk
    # would make every requirement query return the same evidence and starve
    # the LLM of section-level detail (projects/skills/certs must be
    # retrievable independently).
    return _sliding_window(text, cid, window=120, overlap=25)


def _is_degenerate_split(chunks: list[dict]) -> bool:
    """True when section detection collapsed the resume.

    Some PDFs extract their section headers OUT OF ORDER (e.g. every header
    line dumped at the END of the text). The section-aware split then swallows
    the whole resume body into one giant HEADER chunk while the detected
    "sections" become 1-word stubs — retrieval degenerates to returning the
    same chunk for every query. Detect that and fall back to sliding windows.
    """
    total_words = sum(len(c["text"].split()) for c in chunks)
    if total_words <= 0:
        return True
    header = next((c for c in chunks if c["section"] == "HEADER"), None)
    if header and len(header["text"].split()) > total_words * 0.7:
        return True
    # Stub sections: a chunk whose text is just its own header line has no
    # content — its body was absorbed elsewhere (out-of-order extraction).
    stubs = sum(
        1
        for c in chunks
        if c["section"] != "HEADER"
        and c["text"].strip().rstrip(":").upper() == c["section"]
    )
    return stubs >= 2


def _split_by_sections(text: str, cid: str) -> list[dict]:
    """Split text by detected section headers."""
    lines = text.split("\n")
    sections: list[tuple[str, int]] = []  # (section_name, start_line)

    for i, line in enumerate(lines):
        stripped = line.strip().rstrip(":")
        if _SECTION_RE.match(stripped):
            sections.append((_normalise_section(stripped), i))

    if not sections:
        return []

    chunks = []

    # Header block (everything before the first section)
    if sections[0][1] > 0:
        header_text = "\n".join(lines[: sections[0][1]]).strip()
        if header_text:
            chunks.append(
                {
                    "id": f"{cid}_header",
                    "section": "HEADER",
                    "text": header_text,
                    "candidate_id": cid,
                    "index": 0,
                }
            )

    for idx, (section_name, start) in enumerate(sections):
        end = sections[idx + 1][1] if idx + 1 < len(sections) else len(lines)
        section_text = "\n".join(lines[start:end]).strip()

        if not section_text:
            continue

        # If a section is very long (>600 tokens rough), sub-chunk it
        tokens_approx = len(section_text.split())
        if tokens_approx > 600:
            sub_chunks = _sliding_window(
                section_text, cid, window=400, overlap=50, section=section_name, start_index=len(chunks)
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                {
                    "id": f"{cid}_{section_name.lower()}_{idx}",
                    "section": section_name,
                    "text": section_text,
                    "candidate_id": cid,
                    "index": len(chunks),
                }
            )

    return chunks


def _sliding_window(
    text: str,
    cid: str,
    window: int = 400,
    overlap: int = 50,
    section: str = "GENERAL",
    start_index: int = 0,
) -> list[dict]:
    """Fallback: split text into overlapping token windows."""
    words = text.split()
    chunks = []
    i = 0
    chunk_idx = start_index

    while i < len(words):
        chunk_words = words[i : i + window]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(
                {
                    "id": f"{cid}_{section.lower()}_w{chunk_idx}",
                    "section": section,
                    "text": chunk_text,
                    "candidate_id": cid,
                    "index": chunk_idx,
                }
            )
            chunk_idx += 1
        i += window - overlap

    return chunks


def split_jd_requirements(jd_text: str) -> list[dict]:
    """Parse a job description into individual requirement items.

    Each requirement becomes a separate vector search query.
    Returns list of dicts: {"id": str, "requirement": str, "text": str}
    """
    if not jd_text or not jd_text.strip():
        return []

    requirements = []
    lines = jd_text.strip().split("\n")

    for line in lines:
        stripped = line.strip()
        # Skip empty lines, titles, section headers
        if not stripped:
            continue
        if stripped.endswith(":") and len(stripped.split()) <= 6:
            continue
        if stripped.lower().startswith((
            "position:", "title:", "role:", "company:", "location:",
            # German JD headers
            "stellenbeschreibung:", "stellenprofil:", "titel:", "firma:",
            "standort:", "ihr profil:", "ihre aufgaben:", "anforderungen:",
        )):
            continue

        # Strip bullet markers
        cleaned = re.sub(r"^[\-\*•·▪▸►➤]\s*", "", stripped).strip()
        if len(cleaned) < 10:
            continue

        requirements.append(
            {
                "id": f"req_{len(requirements)}",
                "requirement": cleaned,
                "text": cleaned,
            }
        )

    return requirements
