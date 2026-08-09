# ================================================================
# 📤 Reporting — per-job CSV hiring reports
# ================================================================

from __future__ import annotations

import csv
import io
import os
import re
import threading
from datetime import datetime, timezone

import config
import db
from ranking import load_shortlist_results

_EXPORT_DIR = str(config.EXPORT_DIR)  # default, overridable via EXPORT_DIR
# Per-thread override so concurrent requests export into THEIR user's folder.
_thread = threading.local()


def _active_export_dir() -> str:
    return getattr(_thread, "export_dir", None) or _EXPORT_DIR


def set_export_dir(path: str | None) -> None:
    """Point THIS THREAD's CSV exports at a specific folder (per-user
    isolation). Passing None clears the override (module default)."""
    _thread.export_dir = str(path) if path else None

# Column order of the per-job CSV report (full pipeline visibility).
CSV_HEADER = [
    "rank",
    "candidate_id",
    "candidate_name",
    "hybrid_score",
    "semantic_score",
    "keyword_score",
    "screening_score",
    "screening_verdict",
    "screening_summary",
    "interview_status",
    "interview_avg_score",
    "interview_verdict",
]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:60] or "job"


# Spreadsheet apps treat cells beginning with these characters as FORMULAS
# (CWE-1236) — a resume-derived name or summary like `=HYPERLINK(...)` would
# execute when the report is opened. Defang them with a leading apostrophe:
# the value displays unchanged in Excel/LibreOffice but is never evaluated.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _csv_safe_cell(value) -> str:
    s = "" if value is None else str(value)
    if s and s.lstrip().startswith(_CSV_FORMULA_PREFIXES):
        return "'" + s
    return s


def export_job_csv(job_id: str, db_path: str | None = None) -> str:
    """Generate a per-job CSV hiring report and return the file path.

    Rows = every candidate attached to the job's pipeline. Columns span the
    full workflow: hybrid rank scores, deep-screen rubric outcome, and the
    interview result. Written as UTF-8 with BOM so Excel opens it cleanly.
    """
    job = db.get_job(job_id, db_path)
    if not job:
        raise ValueError("Job not found.")

    results = load_shortlist_results(job_id, db_path)
    by_id = {r.candidate_id: r for r in results}
    screenings = {
        s["candidate_id"]: s for s in db.list_screenings(job_id, db_path)
    }
    # Chat + live (video) interviews — a finished live meeting is an interview
    # outcome and belongs in the report. When a candidate has both, the newest
    # record wins (chat rows are ordered by updated_at; live rows carry
    # created_at as their updated_at and a `live` status).
    chat_ivs = db.list_interviews(job_id, db_path)
    live_ivs = [
        dict(v, status="live", updated_at=v.get("created_at") or "")
        for v in db.list_video_interviews(job_id, db_path)
    ]
    interviews_by_cand: dict[str, dict] = {}
    for iv in chat_ivs + live_ivs:
        cid = iv["candidate_id"]
        ts = iv.get("updated_at") or iv.get("created_at") or ""
        cur = interviews_by_cand.get(cid)
        if cur is None or ts >= (
            cur.get("updated_at") or cur.get("created_at") or ""
        ):
            interviews_by_cand[cid] = iv

    # Ranked shortlist first (in rank order), then pipeline-attached candidates
    # (e.g. ingested via Talent pool), then any screened/interviewed candidates
    # not already covered — so the report truly covers the whole pipeline.
    ordered_ids = [r.candidate_id for r in results]
    for jc in db.list_job_candidates(job_id, db_path):
        cid = jc["candidate_id"]
        if cid not in ordered_ids:
            ordered_ids.append(cid)
    for cid in list(screenings) + list(interviews_by_cand):
        if cid not in ordered_ids:
            ordered_ids.append(cid)

    rows: list[dict] = []
    for cid in ordered_ids:
        cand = db.get_candidate(cid, db_path)
        if not cand:
            continue
        r = by_id.get(cid)
        scr = screenings.get(cid)
        intr = interviews_by_cand.get(cid)
        rows.append({
            "rank": (results.index(r) + 1) if r else "",
            "candidate_id": cid,
            "candidate_name": cand.get("name") or cid,
            "hybrid_score": r.hybrid_score if r else "",
            "semantic_score": r.semantic_score if r else "",
            "keyword_score": r.keyword_score if r else "",
            "screening_score": scr.get("score") if scr else "",
            "screening_verdict": scr.get("verdict") if scr else "NOT SCREENED",
            "screening_summary": (scr.get("summary") or "") if scr else "",
            "interview_status": intr.get("status") if intr else "",
            "interview_avg_score": intr.get("average_score") if intr else "",
            "interview_verdict": intr.get("verdict") if intr else "",
        })

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_HEADER, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_safe_cell(v) for k, v in row.items()})

    os.makedirs(_active_export_dir(), exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    fname = f"hiring_report_{_slugify(job['title'])}_{stamp}.csv"
    path = os.path.join(_active_export_dir(), fname)
    # utf-8-sig BOM -> Excel-friendly
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        f.write(buf.getvalue())
    return path
