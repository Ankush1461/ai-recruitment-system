# ================================================================
# 🎨 Gradio UI — Recruiter Workspace (industry-grade)
# ================================================================
from __future__ import annotations

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

import functools
import os
import shutil
import threading
import warnings
from contextlib import suppress

import gradio as gr
from dotenv import load_dotenv

import auth
import config
import db
import emailer
import live_interview
import rubric
from interview import (
    InterviewSession,
    start_interview,
    submit_answer,
)
from pdf import extract_pdf_text
from ranking import (
    format_multi_job_shortlists_markdown,
    format_ranking_markdown,
    format_ranking_table,
    ingest_candidate_deferred,
    load_shortlist_results,
    rank_and_save_shortlist,
    rank_jobs_batch,
    remove_candidate,
    update_candidate_record,
)
from reports import export_job_csv
from sample_data import SAMPLE_JOB_DESCRIPTIONS
from screening import (
    ScreeningResult,
    deep_screen_candidate,
    format_screening_markdown,
)

warnings.filterwarnings("ignore")
load_dotenv()
db.init_db()


NAV_ITEMS = ["Jobs", "Talent pool", "Shortlist", "Email", "Interview", "History & export"]

# Maximum number of shortlisted candidates to forward to interview queue
DEFAULT_INTERVIEW_N = 3


# ================================================================
# Per-request session scoping
# ================================================================
# Every workspace handler runs inside auth.user_scope: the session token
# travels with the event payload as the handler's FIRST argument (wired from
# the `session_token` BrowserState). This (a) isolates each request to its own
# user's db / vectorstore / exports even though Gradio runs handlers in a
# shared thread pool, and (b) rejects direct unauthenticated API calls with a
# visible error before they can touch any data.

def _require_session(fn):
    """Run a workspace handler inside the caller's private storage.

    The handler's first positional argument must be the session token.
    Missing/invalid tokens raise a visible error (never touch data).
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        token = args[0] if args else kwargs.get("token")
        try:
            with auth.user_scope(token):
                return fn(*args, **kwargs)
        except PermissionError:
            raise gr.Error(
                "Your session has expired — please sign in again."
            ) from None
    return wrapper


# ================================================================
# Helpers
# ================================================================

def _job_choices() -> list[tuple[str, str]]:
    """Every job labelled with its requisition ID, e.g. `REQ-1001 · Title`."""
    return [
        (f"{j.get('req_id') or '—'} · {j['title']}", j["id"])
        for j in db.list_jobs()
    ]


def _candidate_choices(job_id: str | None = None) -> list[tuple[str, str]]:
    """Candidates of one job listing (no global pool)."""
    return [(f"{c['name']}", c["id"]) for c in db.list_candidates(job_id)]


# Current "Top N for interview" value — kept in sync from the UI spinbox so
# every refresh path respects the recruiter's chosen N (not just the sync
# button). Thread-local: it is a UI preference, not shared data, so two users
# on different worker threads never clobber each other's choice.
_thread = threading.local()


def _get_int_n() -> int:
    return getattr(_thread, "int_n", DEFAULT_INTERVIEW_N)


def _set_int_n(n: int) -> None:
    _thread.int_n = max(1, min(20, int(n)))


def _qualified_choices(job_id: str, interview_n: int | None = None) -> list[tuple[str, str]]:
    if not job_id:
        return []
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for r in db.list_qualified_for_job(job_id):
        cid = r["candidate_id"]
        if cid in seen:
            continue
        seen.add(cid)
        out.append((f"{r.get('candidate_name', cid)} · {r['score']}/100", cid))
    # If interview_n is set, take only the top N by score
    if interview_n is not None and interview_n > 0:
        out = out[:interview_n]
    return out


def _interview_choices(job_id: str, interview_n: int | None = None) -> list[tuple[str, str]]:
    """Top N ranked shortlist candidates for the Interview tab.

    Each entry is labelled `#rank · name · score · status` where status is
    PASS / FAIL (latest deep-screen) or "not screened". Falls back to the
    PASS-screened list when no shortlist exists yet. PASS-screened
    candidates outside the ranked snapshot (just deep-screened, or added
    since the last rank) are appended after the ranked top N so nothing new
    hides until a manual re-rank.
    """
    n = interview_n if interview_n is not None else _get_int_n()
    if not job_id:
        return []
    results = load_shortlist_results(job_id)
    if not results:
        return _qualified_choices(job_id, n)
    if n and n > 0:
        results = results[:n]
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for i, r in enumerate(results, 1):
        cid = r.candidate_id
        if cid in seen:
            continue
        seen.add(cid)
        scr = db.latest_screening(job_id, cid)
        if scr and scr.get("verdict") == "PASS":
            badge = "PASS"
        elif scr and scr.get("verdict") == "FAIL":
            badge = "FAIL"
        else:
            badge = "not screened"
        out.append((f"#{i} {r.name} · {r.hybrid_score} · {badge}", cid))
    # Smooth sync: qualified (PASS) candidates not in the ranked snapshot
    # appear next — a candidate that just passed deep screening lands in the
    # Interview dropdown immediately, no refresh needed. These extras
    # intentionally bypass the top-N cap: they are qualified candidates
    # surfaced for smooth sync, not part of the ranked top N (the
    # no-shortlist fallback _qualified_choices truncates to N by score).
    for r in db.list_qualified_for_job(job_id):
        cid = r["candidate_id"]
        if cid in seen:
            continue
        seen.add(cid)
        name = r.get("candidate_name") or cid
        out.append((f"{name} · {r['score']}/100 · PASS", cid))
    return out


def _shortlist_cand_choices(job_id: str) -> list[tuple[str, str]]:
    """Shortlist dropdown for deep-screening: the saved ranked shortlist
    first, then every job candidate not in that snapshot (labelled "not
    ranked yet") — so a candidate just added in Talent pool appears here
    immediately, no manual re-rank needed."""
    results = load_shortlist_results(job_id) if job_id else []
    if not results:
        return _candidate_choices(job_id)
    ranked = {r.candidate_id for r in results}
    out = [
        (f"#{i} {r.name} · {r.hybrid_score}", r.candidate_id)
        for i, r in enumerate(results, 1)
    ]
    for c in db.list_candidates(job_id):
        if c["id"] not in ranked:
            scr = db.latest_screening(job_id, c["id"])
            if scr and scr.get("verdict"):
                # Already deep-screened — show the real badge, not "ranked".
                out.append(
                    (f"{c['name']} · {scr.get('score', 0)}/100 · {scr['verdict']}", c["id"])
                )
            else:
                out.append((f"{c['name']} · not ranked yet", c["id"]))
    return out


def _badge(verdict: str) -> str:
    """Render a verdict as a styled badge (shared helper)."""
    return rubric.verdict_badge(verdict)


def _fmt_dt(ts: str | None, seconds: bool = False) -> str:
    """`2026-08-05T06:32:19` → `2026-08-05 06:32` (space, no seconds by default)."""
    s = (ts or "").strip()
    if not s:
        return ""
    s = s[:19].replace("T", " ")
    return s if seconds else s[:16]


def _history_markdown(job_id: str | None = None) -> str:
    screenings = db.list_screenings(job_id)
    interviews = db.list_interviews(job_id)
    live = db.list_video_interviews(job_id)
    jobs = {j["id"]: j["title"] for j in db.list_jobs()}
    lines = ["#### Screening history\n"]
    if not screenings:
        lines.append("_No screenings yet._\n")
    else:
        lines.append("| When | Job | Candidate | Score | Verdict |")
        lines.append("|---|---|---|---:|---|")
        for s in screenings[:40]:
            lines.append(
                f"| {_fmt_dt(s.get('created_at', ''), seconds=True)} | {s.get('job_title', '')} | "
                f"{s.get('candidate_name', '')} | {s.get('score', 0)} | "
                f"{_badge(s.get('verdict', ''))} |"
            )
    lines.append("\n#### Interview history\n")
    # Chat + live (meeting) interviews in one table, newest first — a finished
    # live call is an interview outcome and belongs in History & export.
    merged: list[tuple[str, str, str, str, float, str]] = []
    for i in interviews:
        merged.append((
            i.get("updated_at") or i.get("created_at") or "",
            i.get("job_title") or "",
            i.get("candidate_name") or i.get("candidate_id", ""),
            i.get("status") or "",
            i.get("average_score") or 0,
            i.get("verdict") or "",
        ))
    for v in live:
        merged.append((
            v.get("created_at") or "",
            jobs.get(v.get("job_id", ""), ""),
            v.get("candidate_name") or v.get("candidate_id", ""),
            "live",
            v.get("average_score") or 0,
            v.get("verdict") or "",
        ))
    merged.sort(key=lambda r: r[0], reverse=True)
    if not merged:
        lines.append("_No interviews yet._\n")
    else:
        lines.append("| Updated | Job | Candidate | Status | Avg | Verdict |")
        lines.append("|---|---|---|---|---:|---|")
        for ts, job_title, cand, status, avg, verdict in merged[:40]:
            lines.append(
                f"| {_fmt_dt(ts, seconds=True)} | {job_title} | {cand} | {status} | "
                f"{avg} | {_badge(verdict)} |"
            )
    return "\n".join(lines)


def _candidates_table(job_id: str | None = None) -> list[list]:
    """Candidates of one job — checkbox Select + position `#` map rows back to
    stored candidates; raw internal IDs are never shown."""
    rows: list[list] = []
    for c in db.list_candidates(job_id):
        rows.append([False, len(rows) + 1, c["name"], c["source"], _fmt_dt(c.get("created_at"))])
    return rows


def _stats_markdown() -> str:
    """Compact pipeline KPIs shown on the Jobs tab (stat-card grid)."""
    jobs = db.list_jobs()
    cands = db.list_candidates()
    screens = db.list_screenings()
    interviews = db.list_interviews()
    live = db.list_video_interviews()
    completed = (
        sum(1 for i in interviews if i.get("status") == "completed") + len(live)
    )
    cards = [
        ("Open roles", len(jobs)),
        ("Total candidates", len(cands)),
        ("Deep screens", len(screens)),
        ("Interviews done", f"{completed}/{len(interviews) + len(live)}"),
    ]
    inner = "".join(
        f'<div class="kpi"><span class="kpi-num">{n}</span>'
        f'<span class="kpi-label">{label}</span></div>'
        for label, n in cards
    )
    return f'<div class="kpi-grid">{inner}</div>'


def _jobs_table() -> list[list]:
    """Rows for the Open roles list — checkbox Select + position `#` map rows
    back to stored jobs (same pattern as the candidate pipeline table)."""
    rows: list[list] = []
    for r in db.jobs_table_rows():
        if r:
            rows.append([False, len(rows) + 1, *r[:-1], _fmt_dt(r[-1], seconds=True)])
    return rows


# Outputs refreshed by the per-job pipeline action buttons: the full
# workspace sweep (see _candidate_full_refresh) — every tab drops a removed
# or deleted candidate immediately. Keep in sync with build_demo().

# Number of stored-data outputs returned by _auth_refresh_all() and wired as
# demo._auth_outputs
# ([*_ws_outputs, *_email_settings_refresh, *_email_template_refresh,
#   em_history, vi_history, *_profile_refresh] = 17 + 11 + 6 + 2 + 4).
# Enforced by an assertion in build_demo() so the count can never drift.
_AUTH_REFRESH_OUTPUTS: int = 40


def _normalize_job_ids(job_ids) -> list[str]:
    if not job_ids:
        return []
    if isinstance(job_ids, str):
        return [job_ids]
    return [j for j in job_ids if j]


def _job_candidates_table(job_id: str | None) -> list[list]:
    """Rows for the focus job's candidate pipeline table."""
    if not job_id:
        return []
    rows: list[list] = []
    for jc in db.list_job_candidates(job_id):
        rows.append([
            False,  # Select checkbox
            len(rows) + 1,  # # — position into this job's pipeline list
            jc.get("candidate_name") or jc["candidate_id"],
            jc.get("status") or "shortlisted",
            _fmt_dt(jc.get("created_at")),
        ])
    return rows


def _job_candidates_status(job_id: str | None) -> str:
    if not job_id:
        return "_Select a job to see its candidate pipeline._"
    n = db.job_candidate_count(job_id)
    title = (db.get_job(job_id) or {}).get("title", job_id)
    if n == 0:
        return (
            f"_No candidates for **{title}** yet._ Ingest resumes in "
            "**Talent pool** (targeting this job) or rank to build the list."
        )
    return (
        f"**{n}** candidate(s) in **{title}**'s pipeline — tick the **Select** "
        "boxes, then remove from this job or delete entirely."
    )


def refresh_workspace(job_id: str | None = None, interview_n: int | None = None):
    """Refresh shared selectors + tables after mutations."""
    jobs = _job_choices()
    selected = job_id if job_id and any(j[1] == job_id for j in jobs) else (
        jobs[0][1] if jobs else None
    )
    cands = _candidate_choices(selected)
    job_ids_all = [j[1] for j in jobs]
    qualified = _interview_choices(selected or "")
    shortlist_cands = _shortlist_cand_choices(selected or "")
    job_dd = gr.update(choices=jobs, value=selected)
    multi_jobs = gr.update(
        choices=jobs,
        value=job_ids_all[: min(3, len(job_ids_all))] if job_ids_all else [],
    )
    cand_dd = gr.update(choices=cands, value=cands[0][1] if cands else None)
    return (
        job_dd,          # focus_job
        multi_jobs,      # multi_job_dd
        job_dd,          # view_job_dd
        job_dd,          # int_job_dd
        job_dd,          # hist_job_dd
        job_dd,          # export_job_dd
        cand_dd,         # manage_cand_dd (per selected job)
        gr.update(choices=shortlist_cands, value=shortlist_cands[0][1] if shortlist_cands else None),
        gr.update(choices=qualified, value=qualified[0][1] if qualified else None),
        _jobs_table(),
        _candidates_table(selected),
        _history_markdown(selected),
        _job_candidates_table(selected),
        _job_candidates_status(selected),
        job_dd,          # tp_job_dd (Talent pool ingestion target)
        _stats_markdown(),  # jobs_kpis
        job_dd,          # em_job_dd (Email tab)
    )


# ================================================================
# Navigation (always-visible tab bar)
# ================================================================

@_require_session
def _on_nav(token, name: str):
    return (
        gr.update(visible=name == "Jobs"),
        gr.update(visible=name == "Talent pool"),
        gr.update(visible=name == "Shortlist"),
        gr.update(visible=name == "Email"),
        gr.update(visible=name == "Interview"),
        gr.update(visible=name == "History & export"),
        # Any tab switch closes the floating bubbles and the backdrop.
        gr.update(visible=False),  # profile_pop
        gr.update(visible=False),  # es_pop
        gr.update(visible=False),  # bubble_backdrop
    )


# ================================================================
# Jobs handlers
# ================================================================

@_require_session
def on_create_job(token, title, description, sample_key, req_id=None):
    """Create a job. Returns status + form clears + full workspace refresh."""
    desc = (description or "").strip()
    title = (title or "").strip()
    if sample_key and sample_key in SAMPLE_JOB_DESCRIPTIONS and not desc:
        desc = SAMPLE_JOB_DESCRIPTIONS[sample_key]
        title = title or sample_key
    if sample_key and not title:
        title = sample_key
    if not title or not desc:
        return (
            "Enter a title and description, or pick a sample JD.",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            *refresh_workspace(),
        )
    job = db.create_job(title, desc, req_id=req_id)
    return (
        f"Created **{job.get('req_id')} · {job['title']}**",
        gr.update(value=""),  # job_title
        gr.update(value=""),  # job_desc
        gr.update(value=None),  # sample_job_dd
        gr.update(value=""),  # job_req_id
        *refresh_workspace(job["id"]),
    )


@_require_session
def on_load_sample_job(token, sample_key):
    return SAMPLE_JOB_DESCRIPTIONS.get(sample_key, ""), sample_key or ""


@_require_session
def on_delete_jobs(token, table):
    """Delete job listings ticked in the Open roles list, plus everything tied
    to them (candidates, shortlists, screenings, interviews)."""
    if not table:
        return ("No jobs in the list to manage.", *refresh_workspace())
    jobs = db.list_jobs()
    positions = [int(row[1]) for row in table if row and len(row) > 1 and row[0]]
    selected = [jobs[i - 1] for i in positions if 0 < i <= len(jobs)]
    if not selected:
        return ("Tick the **Select** boxes for the job listings first.", *refresh_workspace())
    labels: list[str] = []
    # Vector cleanup must not block deletion — one flaky Chroma call aborts the
    # whole batch otherwise. SQLite is the source of truth; the index rebuilds.
    failed = 0
    for j in selected:
        labels.append(f"{j.get('req_id') or '—'} · {j['title']}")
        for c in db.list_candidates(job_id=j["id"]):
            try:
                remove_candidate(c["id"])
            except Exception:
                failed += 1
        db.delete_job(j["id"])
    suffix = f" (vector cleanup pending for {failed}) " if failed else " "
    return (
        (
            f"Deleted **{len(selected)}** job listing(s){suffix}"
            f"— {', '.join(labels)} — with their candidates, shortlists, screenings, and interviews."
        ),
        *refresh_workspace(),
    )


# ================================================================
# Candidates handlers
# ================================================================

def _cand_refresh_outputs(
    msg: str,
    prefer_id: str | None = None,
    *,
    clear_editor: bool = False,
    focus_job_id: str | None = None,
):
    jobs = _job_choices()
    job_id = focus_job_id if focus_job_id and any(j[1] == focus_job_id for j in jobs) else (
        jobs[0][1] if jobs else ""
    )
    cands = _candidate_choices(job_id or None)
    value = prefer_id if prefer_id and any(c[1] == prefer_id for c in cands) else (
        cands[0][1] if cands else None
    )
    shortlist = _shortlist_cand_choices(job_id)
    qualified = _interview_choices(job_id)
    cand_upd = gr.update(choices=cands, value=value)
    # Smooth sync: after adding/editing a candidate, pre-select it in the
    # Shortlist dropdown too (it's listed, possibly "not ranked yet") so
    # deep-screening it is one click away — no manual refresh, no re-pick.
    shortlist_val = (
        prefer_id if prefer_id and any(s[1] == prefer_id for s in shortlist)
        else (shortlist[0][1] if shortlist else None)
    )
    if clear_editor or not value:
        edit_name, edit_resume = "", ""
    else:
        cand = db.get_candidate(value)
        edit_name = (cand or {}).get("name") or ""
        edit_resume = (cand or {}).get("resume_text") or ""
    return (
        msg,
        _candidates_table(job_id or None),
        cand_upd,
        gr.update(choices=shortlist, value=shortlist_val),
        gr.update(choices=qualified, value=qualified[0][1] if qualified else None),
        edit_name,
        edit_resume,
        _job_candidates_table(job_id or None),
        _job_candidates_status(job_id or None),
        gr.update(choices=jobs, value=job_id or None),  # tp_job_dd
        # Sweep the whole workspace so a mutation — especially a deletion —
        # drops the candidate from EVERY tab immediately, not just this one.
        _jobs_table(),                                    # jobs_table (counts)
        _stats_markdown(),                                # jobs_kpis
        _history_markdown(job_id or None),                # history_md
        gr.update(choices=jobs, value=job_id or None),    # em_job_dd
        gr.update(choices=cands, value=cands[0][1] if cands else None),  # em_cand_dd
        _video_history_table(job_id or None),             # vi_history
    )


@_require_session
def on_add_resume_text(token, text, name, tp_job):
    if not tp_job:
        return _cand_refresh_outputs(
            "Select a **job listing** first — every candidate belongs to one job.",
            focus_job_id=None,
        )
    if not text or not str(text).strip():
        return _cand_refresh_outputs("Paste resume text or upload PDFs first.", focus_job_id=tp_job)
    try:
        cand = ingest_candidate_deferred(
            token, text, name=name or None, source="paste", job_id=tp_job
        )
        jt = (db.get_job(tp_job) or {}).get("title", "")
        msg = f"Ingested **{cand['name']}** for **{jt}** — indexing in the background."
        return _cand_refresh_outputs(msg, cand["id"], focus_job_id=tp_job)
    except Exception as e:
        return _cand_refresh_outputs(f"Failed: {e}", focus_job_id=tp_job)


@_require_session
def on_upload_pdfs(token, files, name_hint, tp_job):
    if not tp_job:
        return _cand_refresh_outputs(
            "Select a **job listing** first — every candidate belongs to one job.",
            focus_job_id=None,
        )
    if not files:
        return _cand_refresh_outputs("Upload one or more PDF resumes.", focus_job_id=tp_job)
    file_list = files if isinstance(files, list) else [files]
    max_bytes = config.MAX_PDF_UPLOAD_MB * 1024 * 1024
    added, errors, last_id = [], [], None
    for f in file_list:
        try:
            path = getattr(f, "name", "") or ""
            if path and os.path.getsize(path) > max_bytes:
                errors.append(
                    f"Too large: {os.path.basename(path)} "
                    f"(max {config.MAX_PDF_UPLOAD_MB} MB)"
                )
                continue
            text = extract_pdf_text(f)
            if not text.strip():
                errors.append(f"Empty: {getattr(f, 'name', f)}")
                continue
            cand = ingest_candidate_deferred(
                token, text, name=name_hint or None, source="pdf", job_id=tp_job
            )
            added.append(cand["name"])
            last_id = cand["id"]
        except Exception as e:
            errors.append(str(e))
    jt = (db.get_job(tp_job) or {}).get("title", "")
    msg = f"Ingested {len(added)} resume(s)" + (f": {', '.join(added)}" if added else ".")
    if added:
        msg += " — indexing runs in the background."
    msg += f" → all for **{jt}**"
    if errors:
        msg += f"\nErrors: {'; '.join(errors)}"
    return _cand_refresh_outputs(msg, last_id, focus_job_id=tp_job)


@_require_session
def on_load_candidate_for_edit(token, candidate_id):
    if not candidate_id:
        return "", ""
    cand = db.get_candidate(candidate_id)
    if not cand:
        return "", ""
    return cand.get("name") or "", cand.get("resume_text") or ""


@_require_session
def on_save_candidate(token, candidate_id, name, resume_text, job_id):
    if not candidate_id:
        return _cand_refresh_outputs("Select a candidate to edit.", focus_job_id=job_id)
    try:
        updated = update_candidate_record(
            candidate_id,
            name=name or None,
            resume_text=resume_text if resume_text and resume_text.strip() else None,
            deferred_token=token,
        )
        return _cand_refresh_outputs(
            f"Updated **{updated['name']}**"
            + (" · resume re-indexed" if resume_text else ""),
            updated["id"],
            focus_job_id=job_id,
        )
    except Exception as e:
        return _cand_refresh_outputs(f"Update failed: {e}", candidate_id, focus_job_id=job_id)


@_require_session
def on_delete_candidates(token, table, job_id):
    """Delete candidates ticked in the Talent pool list for the selected job."""
    if not job_id:
        return _cand_refresh_outputs(
            "Select a **job listing** first — every candidate belongs to one job.",
            focus_job_id=None,
        )
    if not table:
        return _cand_refresh_outputs("No candidates in the list to manage.", focus_job_id=job_id)
    cands = db.list_candidates(job_id)
    positions = [int(row[1]) for row in table if row and len(row) > 1 and row[0]]
    selected = [cands[i - 1] for i in positions if 0 < i <= len(cands)]
    if not selected:
        return _cand_refresh_outputs(
            "Tick the **Select** boxes for candidates first.",
            focus_job_id=job_id,
        )
    # One flaky vectorstore call shouldn't abort the whole batch (SQLite is the
    # source of truth — the index rebuilds from resume_text).
    failed = 0
    for c in selected:
        try:
            remove_candidate(c["id"])
        except Exception:
            failed += 1
    suffix = f" (vector cleanup pending for {failed}) " if failed else " "
    return _cand_refresh_outputs(
        f"Deleted **{len(selected)}** candidate(s) permanently.{suffix}".strip(),
        focus_job_id=job_id,
    )


@_require_session
def _on_tp_job_change(token, job_id):
    """Talent pool target job changed — show that job's candidates."""
    cands = _candidate_choices(job_id or None)
    return (
        gr.update(choices=cands, value=cands[0][1] if cands else None),
        _candidates_table(job_id or None),
        "",  # clear the editor
        "",
    )


# ================================================================
# Per-job candidate pipeline handlers (select + remove / delete)
# ================================================================

def _candidate_full_refresh(job_id: str | None = None):
    """Full-workspace sweep for the per-job pipeline action buttons.

    Returns exactly the _JC_ACTION_OUTPUTS set (17 workspace outputs + the
    Email candidate dropdown + video history, see build_demo), so a removed
    or deleted candidate disappears from EVERY tab immediately — Jobs KPIs,
    History, Shortlist/Interview dropdowns, Email and Video — not just the
    focus job's pipeline table.
    """
    jobs = _job_choices()
    job_id = job_id if job_id and any(j[1] == job_id for j in jobs) else (
        jobs[0][1] if jobs else ""
    )
    cands = _candidate_choices(job_id or None)
    cand_upd = gr.update(choices=cands, value=cands[0][1] if cands else None)
    job_dd = gr.update(choices=jobs, value=job_id or None)
    shortlist = _shortlist_cand_choices(job_id)
    qualified = _interview_choices(job_id)
    return (
        job_dd,          # focus_job
        gr.update(choices=jobs, value=[j[1] for j in jobs][:3]),  # multi_job_dd
        job_dd,          # view_job_dd
        job_dd,          # int_job_dd
        job_dd,          # hist_job_dd
        job_dd,          # export_job_dd
        cand_upd,        # manage_cand_dd
        gr.update(choices=shortlist, value=shortlist[0][1] if shortlist else None),  # screen_cand_dd
        gr.update(choices=qualified, value=qualified[0][1] if qualified else None),  # int_cand_dd
        _jobs_table(),   # jobs_table
        _candidates_table(job_id or None),  # cand_table
        _history_markdown(job_id or None),  # history_md
        _job_candidates_table(job_id or None),  # job_cand_table
        _job_candidates_status(job_id or None),  # job_cand_status
        job_dd,          # tp_job_dd
        _stats_markdown(),  # jobs_kpis
        job_dd,          # em_job_dd
        cand_upd,        # em_cand_dd
        _video_history_table(job_id or None),  # vi_history
    )


@_require_session
def on_job_candidate_action(token, job_id, table, action: str):
    """Remove-from-job or delete-entirely for checkbox-selected rows.

    Rows carry no internal IDs — the `#` column is the position of the row in
    the job's candidate list, which maps back to the stored candidate.

    Both actions return a full-workspace sweep so the removed/deleted
    candidates disappear from every tab immediately.
    """
    if not job_id:
        return ("Select a focus job first.", *_candidate_full_refresh())
    if not table:
        return ("No candidates to manage.", *_candidate_full_refresh(job_id))
    cands = db.list_job_candidates(job_id)
    positions = [int(row[1]) for row in table if row and len(row) > 1 and row[0]]
    selected = [
        cands[i - 1]["candidate_id"] for i in positions if 0 < i <= len(cands)
    ]
    if not selected:
        return ("Tick the **Select** boxes for candidates first.", *_candidate_full_refresh(job_id))
    if action == "remove":
        for cid in selected:
            db.remove_candidate_from_job(job_id, cid)
        msg = f"Removed **{len(selected)}** candidate(s) from this job."
    else:
        for cid in selected:
            remove_candidate(cid)
        msg = f"Deleted **{len(selected)}** candidate(s) permanently."
    return (msg, *_candidate_full_refresh(job_id))


# ================================================================
# Rank & Screen handlers
# ================================================================

def _screen_loader_markup(message: str) -> str:
    """Animated 'AI is evaluating' section shown while a deep screen runs."""
    return (
        '<div class="ai-loader"><div class="spinner"></div>'
        f'<div class="loader-text">{message}</div></div>'
    )


def _show_screen_loader(token, message: str):
    """Show the loader section (pure UI — never touches user data)."""
    return gr.update(visible=True, value=_screen_loader_markup(message))


def _hide_screen_loader(token):
    return gr.update(visible=False, value="")

@_require_session
def on_rank_multi(token, job_ids, top_n):
    ids = _normalize_job_ids(job_ids)
    if not ids:
        return (
            "Select at least one job.", "", [], gr.update(), [], "",
            gr.update(), gr.update(),  # int_cand_dd, int_job_dd
        )
    try:
        n = int(top_n) if top_n not in (None, "") else None
        if n is not None and n <= 0:
            n = None
    except (TypeError, ValueError):
        n = None
    results_by_job = rank_jobs_batch(ids, top_n=n)
    focus = ids[0]
    focus_results = results_by_job.get(focus, [])
    job = db.get_job(focus)
    title = job["title"] if job else focus
    shortlist_cands = _shortlist_cand_choices(focus)
    qualified = _interview_choices(focus)
    return (
        format_multi_job_shortlists_markdown(results_by_job),
        format_ranking_markdown(focus_results, job_title=title),
        format_ranking_table(focus_results),
        gr.update(
            choices=shortlist_cands,
            value=shortlist_cands[0][1] if shortlist_cands else None,
        ),
        _job_candidates_table(focus),
        _job_candidates_status(focus),
        # Auto-sync the Interview tab so no manual "Sync top N" click is
        # needed after ranking.
        gr.update(choices=qualified, value=qualified[0][1] if qualified else None),
        gr.update(value=focus),
    )


@_require_session
def on_view_job_shortlist(token, job_id):
    if not job_id:
        return (
            "*Select a job to view its shortlist.*", [], gr.update(), [], "",
            gr.update(), gr.update(),  # int_cand_dd, int_job_dd
        )
    results = load_shortlist_results(job_id) or rank_and_save_shortlist(job_id)
    job = db.get_job(job_id)
    title = job["title"] if job else job_id
    shortlist_cands = _shortlist_cand_choices(job_id)
    qualified = _interview_choices(job_id)
    return (
        format_ranking_markdown(results, job_title=title),
        format_ranking_table(results),
        gr.update(
            choices=shortlist_cands,
            value=shortlist_cands[0][1] if shortlist_cands else None,
        ),
        _job_candidates_table(job_id),
        _job_candidates_status(job_id),
        # Keep the Interview tab's dropdowns in step with the viewed job.
        gr.update(choices=qualified, value=qualified[0][1] if qualified else None),
        gr.update(value=job_id),
    )


@_require_session
def _sync_int_candidates(token, job_id, interview_n, focus_job):
    """Sync the top N shortlisted candidates to the Interview tab.

    Returns exactly 2 values: the interview candidate dropdown (top N from the
    ranked shortlist, with screening status) and the interview job dropdown.
    """
    if not job_id:
        job_id = focus_job or (db.list_jobs()[0]["id"] if db.list_jobs() else None)
    if not job_id:
        return gr.update(choices=[], value=None), gr.update()
    try:
        int_n = int(interview_n or DEFAULT_INTERVIEW_N)
    except (TypeError, ValueError):
        int_n = DEFAULT_INTERVIEW_N
    _set_int_n(int_n)
    qualified = _interview_choices(job_id)
    return (
        gr.update(choices=qualified, value=qualified[0][1] if qualified else None),
        gr.update(value=job_id),
    )


@_require_session
def _on_int_n_change(token, interview_n, job_id):
    """Live-sync the Interview list whenever Top N changes."""
    try:
        n = int(interview_n or DEFAULT_INTERVIEW_N)
    except (TypeError, ValueError):
        n = DEFAULT_INTERVIEW_N
    _set_int_n(n)
    choices = _interview_choices(job_id or "")
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


@_require_session
def _on_view_job_change(token, job_id):
    """Keep the Interview list in step with the focus job in the Shortlist tab."""
    choices = _interview_choices(job_id or "")
    return (
        gr.update(choices=choices, value=choices[0][1] if choices else None),
        "_Suggested questions appear here when the interview starts._",
    )


@_require_session
def on_deep_screen(token, job_id, candidate_id, interview_n):
    if not job_id or not candidate_id:
        return (
            "Select a job and a shortlisted candidate.", gr.update(), gr.update(),
            gr.update(),  # int_job_dd
        )
    result = deep_screen_candidate(job_id, candidate_id)
    md = format_screening_markdown(result)
    try:
        int_n = int(interview_n or DEFAULT_INTERVIEW_N)
    except (TypeError, ValueError):
        int_n = DEFAULT_INTERVIEW_N
    qualified = _interview_choices(job_id, int_n)
    pick = candidate_id if result.verdict == "PASS" else (
        qualified[0][1] if qualified else None
    )
    return (
        md,
        gr.update(choices=qualified, value=pick),
        _history_markdown(job_id),
        gr.update(value=job_id),  # int_job_dd
    )


# Shared progress tracker for batch deep-screen runs. B008-safe singleton:
# Gradio swaps the default for a real per-request tracker at call time, so the
# default is never actually used.
_SCREEN_PROGRESS = gr.Progress()

# Concurrent Groq evaluations for one batch deep-screen. Each candidate's
# evaluation is 1-2 min of LLM time; running up to 3 in parallel roughly
# halves the batch wall-clock on any tier (CPU basic included). Free-tier
# rate limits are absorbed by llm.py's exponential-backoff retry.
_DEEP_SCREEN_WORKERS = 3


def _deep_screen_batch(token: str, job_id: str, results, progress) -> list[str]:
    """Deep-screen ranked candidates, up to _DEEP_SCREEN_WORKERS at a time.

    Each worker re-enters the caller's user scope (thread-local DB / vector
    store — the same pattern as the background-ingest thread), so concurrent
    evaluations never leak across accounts. Reports keep input order; a
    worker that crashes is surfaced as an error block instead of aborting
    the batch.
    """
    import concurrent.futures

    total = len(results)
    if total == 0:
        return []

    def _one(r) -> ScreeningResult:
        try:
            with auth.user_scope(token):
                return deep_screen_candidate(job_id, r.candidate_id)
        except Exception as exc:
            return ScreeningResult(error=f"Deep-screen failed: {exc}")

    reports: list[str] = [""] * total
    workers = min(_DEEP_SCREEN_WORKERS, total)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, r): i for i, r in enumerate(results)}
        for done, fut in enumerate(
            concurrent.futures.as_completed(futures), 1
        ):
            i = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = ScreeningResult(error=f"Deep-screen failed: {exc}")
            reports[i] = f"## {results[i].name}\n\n{format_screening_markdown(result)}\n"
            progress(done / total, desc=f"Deep-screened {done}/{total} candidates…")
    progress(1.0, desc="Deep-screening complete")
    return reports


@_require_session
def on_deep_screen_top_n(token, job_id, top_n, interview_n, progress=_SCREEN_PROGRESS):
    if not job_id:
        return (
            "Select a focus job first.", gr.update(), gr.update(), gr.update()
        )
    try:
        n = int(top_n or 3)
    except (TypeError, ValueError):
        n = 3
    try:
        int_n = int(interview_n or DEFAULT_INTERVIEW_N)
    except (TypeError, ValueError):
        int_n = DEFAULT_INTERVIEW_N
    results = load_shortlist_results(job_id)
    results = (
        rank_and_save_shortlist(job_id, top_n=n) if not results else results[:n]
    )
    if not results:
        return (
            "No candidates to screen.", gr.update(), gr.update(), gr.update()
        )
    reports = _deep_screen_batch(token, job_id, results, progress)
    qualified = _interview_choices(job_id, int_n)
    return (
        "\n---\n".join(reports),
        gr.update(choices=qualified, value=qualified[0][1] if qualified else None),
        _history_markdown(job_id),
        gr.update(value=job_id),  # int_job_dd
    )


# ================================================================
# Interview / History
# ================================================================

def _questions_markdown(questions: list[str]) -> str:
    if not questions:
        return "_No suggested questions yet._"
    lines = ["### 🤖 AI-Suggested Interview Questions", ""]
    for i, q in enumerate(questions, 1):
        lines.append(f"**Q{i}.** {q}")
        lines.append("")
    return chr(10).join(lines)


@_require_session
def on_start_interview(token, job_id, candidate_id, language="English"):
    session = start_interview(job_id, candidate_id, language=language or "English")
    if session.error:
        return session.to_dict(), [], f"**{session.error}**", "", ""
    progress = (
        "Interview läuft." if session.language == "German" else "Interview in progress."
    )
    return (
        session.to_dict(),
        session.messages,
        progress,
        "",
        _questions_markdown(session.questions),
    )


@_require_session
def on_chat_submit(token, message, history, session_dict):
    session = InterviewSession.from_dict(session_dict)
    if not message or not str(message).strip():
        return session_dict, history or session.messages, "", session.eval_markdown
    updated = submit_answer(session, str(message))
    if updated.error:
        status = updated.error
    elif updated.status == "completed":
        status = "Interview abgeschlossen." if updated.language == "German" else "Interview complete."
    else:
        status = "Antwort aufgezeichnet." if updated.language == "German" else "Answer recorded."
    return updated.to_dict(), updated.messages, "", updated.eval_markdown or status


@_require_session
def on_refresh_history(token, job_id):
    return _history_markdown(job_id)


@_require_session
def on_export_csv(token, job_id):
    """Generate a per-job CSV report; returns status + DownloadButton update."""
    if not job_id:
        return "Select a job first.", gr.update(value=None)
    try:
        path = export_job_csv(job_id)
        title = (db.get_job(job_id) or {}).get("title", job_id)
        return (
            f"CSV report ready for **{title}** — click **Download CSV** to save it.",
            gr.update(value=path),
        )
    except Exception as e:
        return f"Export failed: {e}", gr.update(value=None)


# ================================================================
# Email handlers (shortlist notifications & interview invites)
# ================================================================

def _email_history_table() -> list[list]:
    """Recent sent emails from the audit log (only successes are recorded)."""
    return [
        [
            _fmt_dt(r.get("created_at")),
            r.get("entity_id") or "",
            r.get("detail") or "",
        ]
        for r in db.list_audit(action="email_sent", limit=20)
    ]


@_require_session
def on_email_candidate_change(token, candidate_id):
    """Auto-fill the recipient box with the email found in the resume."""
    if not candidate_id:
        return ""
    cand = db.get_candidate(candidate_id) or {}
    return emailer.extract_email(cand.get("resume_text")) or ""


@_require_session
def _on_em_job_change(token, job_id):
    cands = _candidate_choices(job_id or None)
    return (
        gr.update(choices=cands, value=cands[0][1] if cands else None),
        "",
    )


# ---- Email templates (per email type) ----------------------------------

def _tmpl_kind_from_radio(kind: str) -> str:
    """Email-type radio value → template kind ('shortlist' | 'invite')."""
    return "invite" if kind and "Interview" in str(kind) else "shortlist"


def _template_choices(kind: str) -> list[tuple[str, str]]:
    """Templates of one email type for the manager dropdown — the preferred
    template first, labelled with a ★ marker."""
    tpls = db.list_email_templates(kind)
    tpls.sort(key=lambda t: (0 if t.get("is_default") else 1, (t.get("name") or "").lower()))
    return [
        (
            (f"{t.get('name') or '(unnamed)'} ★ preferred" if t.get("is_default") else t.get("name") or "(unnamed)"),
            t["id"],
        )
        for t in tpls
    ]


def _em_template_choices(kind: str) -> list[tuple[str, str]]:
    """Compose-form dropdown: the built-in design + every saved template."""
    choices: list[tuple[str, str]] = [("(Built-in template)", "")]
    choices.extend(_template_choices(kind))
    return choices


def _tmpl_dropdown_update(choices: list[tuple[str, str]], value: str | None) -> dict:
    """gr.update for a template dropdown — never assigns a value that is not
    among the choices (Gradio crashes preprocessing with 'Value: X is not in
    the list of choices')."""
    if not value or value not in [c[1] for c in choices]:
        value = None
    return gr.update(choices=choices, value=value)


def _email_template_refresh(kind: str = "shortlist") -> tuple:
    """The 6 Email-template outputs (compose dropdown, manager dropdown,
    name, subject, body, status) loaded for one email type."""
    # Seed the built-in starter template the first time a kind has none, so
    # the feature is never an empty shell and sends always have a default.
    db.ensure_default_email_templates(kind)
    choices = _template_choices(kind)
    em_choices = _em_template_choices(kind)
    pref = db.get_default_email_template(kind)
    value = pref["id"] if pref else ""
    if pref:
        name, subject, body = (
            pref.get("name") or "", pref.get("subject") or "", pref.get("body") or ""
        )
    else:
        name = subject = body = ""
    if pref:
        status = (
            f"⭐ **{pref.get('name') or '(unnamed)'}** is the preferred template — "
            "it is pre-selected when composing this email type."
        )
    else:
        status = (
            "_No templates yet — create one below. Emails use the built-in "
            "design until you save a template._"
        )
    return (
        _tmpl_dropdown_update(em_choices, value),
        _tmpl_dropdown_update(choices, value),
        gr.update(value=name),
        gr.update(value=subject),
        gr.update(value=body),
        gr.update(value=status),
    )


@_require_session
def _on_email_kind_change(token, kind):
    """Email type switched — point both template dropdowns at that type's
    preferred template (or the built-in design when none is set)."""
    k = _tmpl_kind_from_radio(kind)
    # Seed the starter template for this kind too (it may not exist yet).
    db.ensure_default_email_templates(k)
    em_choices = _em_template_choices(k)
    choices = _template_choices(k)
    pref = db.get_default_email_template(k)
    value = pref["id"] if pref else ""
    return (
        _tmpl_dropdown_update(em_choices, value),
        _tmpl_dropdown_update(choices, value),
    )


@_require_session
def _on_tmpl_select(token, template_id):
    """Load a picked template into the editor (blank for the built-in entry)."""
    if not template_id:
        return "", "", ""
    t = db.get_email_template(template_id)
    if not t:
        return "", "", ""
    return t.get("name") or "", t.get("subject") or "", t.get("body") or ""


@_require_session
def _on_tmpl_new(token, kind):
    """Start a blank template for the current email type."""
    k = _tmpl_kind_from_radio(kind)
    return (
        _tmpl_dropdown_update(_em_template_choices(k), ""),
        _tmpl_dropdown_update(_template_choices(k), ""),
        "", "", "",
        gr.update(value=f"New **{k}** template — fill in the fields and click **Save template**."),
    )


@_require_session
def _on_tmpl_save(token, kind, template_id, name, subject, body):
    """Create or update the template in the editor and select it."""
    k = _tmpl_kind_from_radio(kind)
    name = (name or "").strip()
    if not name:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "Give the template a **name** first.",
        )
    t = db.save_email_template(
        template_id=template_id or None,
        kind=k,
        name=name,
        subject=subject or "",
        body=body or "",
    )
    return (
        _tmpl_dropdown_update(_em_template_choices(k), t["id"]),
        _tmpl_dropdown_update(_template_choices(k), t["id"]),
        gr.update(value=t.get("name") or ""),
        gr.update(),
        gr.update(),
        f"Saved **{t.get('name') or '(unnamed)'}** — select it in the compose form or mark it as preferred.",
    )


@_require_session
def _on_tmpl_set_default(token, kind, template_id):
    """Mark the selected template as the preferred one for its email type."""
    if not template_id:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "Select a template first, then **Set as preferred**.",
        )
    t = db.get_email_template(template_id)
    if not t:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "Template not found — pick one from the list.",
        )
    k = t.get("kind") or _tmpl_kind_from_radio(kind)
    db.set_default_email_template(template_id)
    return (
        _tmpl_dropdown_update(_em_template_choices(k), template_id),
        _tmpl_dropdown_update(_template_choices(k), template_id),
        gr.update(), gr.update(), gr.update(),
        f"⭐ **{t.get('name') or '(unnamed)'}** is now the preferred template for this email type.",
    )


@_require_session
def _on_tmpl_delete(token, kind, template_id):
    """Permanently delete the selected template and clear the editor."""
    if not template_id:
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "Select a template first, then **Delete template**.",
        )
    t = db.get_email_template(template_id)
    name = (t or {}).get("name") or "(unnamed)"
    db.delete_email_template(template_id)
    k = _tmpl_kind_from_radio(kind)
    pref = db.get_default_email_template(k)
    value = pref["id"] if pref else ""
    return (
        _tmpl_dropdown_update(_em_template_choices(k), value),
        _tmpl_dropdown_update(_template_choices(k), value),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        f"Deleted **{name}**.",
    )


@_require_session
def on_send_email(token, job_id, candidate_id, kind, template_id, to, message, invite_link=""):
    """Build + send a shortlist or interview-invite email via SMTP.

    `template_id` is the saved template to use ('' → the preferred template
    for that email type → built-in design when none exists)."""
    if not job_id or not candidate_id:
        return "Select a **job** and a **candidate** first.", gr.update()
    cand = db.get_candidate(candidate_id)
    if not cand:
        return "Candidate not found.", gr.update()
    job = db.get_job(job_id) or {}
    to = (to or "").strip() or (emailer.extract_email(cand.get("resume_text")) or "")
    if not to or "@" not in to:
        return (
            (
                "No email found in this resume — type the address in the "
                "**Recipient email** box."
            ),
            gr.update(),
        )
    if kind and "Interview" in str(kind):
        template = (
            db.get_email_template(template_id)
            if template_id
            else db.get_default_email_template("invite")
        )
        subject, body = emailer.build_invite_email(
            job, cand,
            extra_msg=message or "",
            invite_link=invite_link or "",
            template=template,
        )
    else:
        template = (
            db.get_email_template(template_id)
            if template_id
            else db.get_default_email_template("shortlist")
        )
        subject, body = emailer.build_shortlist_email(
            job, cand, extra_msg=message or "", template=template
        )
    res = emailer.send_email(to, subject, body)
    if not res["ok"]:
        return f"**Send failed:** {res['error']}", gr.update()
    tpl_note = f" using **{template['name']}**" if template else ""
    return (
        f"Sent **{subject}** to {to}{tpl_note} — recorded in the audit log.",
        gr.update(value=_email_history_table()),
    )


# ---- Per-account email settings (Email tab) -----------------------------

# Common free SMTP providers for the Email settings dropdown — each entry is
# (host, default port). Selecting one from the UI auto-fills the port; typing
# a custom host is still allowed (allow_custom_value).
_SMTP_PROVIDERS: dict[str, int] = {
    "smtp.gmail.com": 587,  # Gmail (app password)
    "smtp-mail.outlook.com": 587,  # Outlook / Microsoft 365
    "smtp.mail.yahoo.com": 587,  # Yahoo Mail
    "smtp.mail.me.com": 587,  # iCloud Mail
    "smtp.zoho.com": 587,  # Zoho Mail
    "smtp.gmx.com": 587,  # GMX Mail
    "smtp.sendgrid.net": 587,  # SendGrid
    "smtp.mailgun.org": 587,  # Mailgun
}

# Dropdown choices labelled with the provider name (host is the stored value).
# Deliberately a separate explicit list — the label must be the provider's
# real name, not derived from the host (which would read "Gmail —
# smtp-mail.outlook.com").
_SMTP_HOST_CHOICES: list[tuple[str, str]] = [
    ("Gmail", "smtp.gmail.com"),
    ("Outlook / Microsoft 365", "smtp-mail.outlook.com"),
    ("Yahoo Mail", "smtp.mail.yahoo.com"),
    ("iCloud Mail", "smtp.mail.me.com"),
    ("Zoho Mail", "smtp.zoho.com"),
    ("GMX Mail", "smtp.gmx.com"),
    ("SendGrid", "smtp.sendgrid.net"),
    ("Mailgun", "smtp.mailgun.org"),
]

# Providers that log in with the full mailbox address as the SMTP username
# (Gmail app password, Outlook, Yahoo, iCloud, Zoho, GMX). SendGrid/Mailgun
# are deliberately excluded — they authenticate with an API key instead of
# the from-address, so auto-filling it would be wrong.
_SMTP_USERNAME_IS_EMAIL: frozenset[str] = frozenset({
    "smtp.gmail.com",
    "smtp-mail.outlook.com",
    "smtp.mail.yahoo.com",
    "smtp.mail.me.com",
    "smtp.zoho.com",
    "smtp.gmx.com",
})


def _email_settings_values(s: dict | None = None) -> tuple:
    """The 9 settings-form values in effect (the account's own config — there
    is no .env fallback): SMTP fields + email-header branding."""
    s = s or emailer.resolved_settings()
    return (
        s["host"],
        s["port"],
        s["mail_from"],
        s["mail_from_name"],
        s["user"],
        s["password"],
        bool(s["starttls"]),
        s["company_name"],
        s["company_logo"] if os.path.isfile(s["company_logo"]) else None,
    )


def _email_settings_status(s: dict | None = None, cfg: dict | None = None) -> str:
    s = s or emailer.resolved_settings()
    cfg = cfg if cfg is not None else _saved_email_settings()
    if not cfg:
        return (
            "No SMTP sender configured yet — open **⚙️ Email settings** and save "
            "your own SMTP details (host + from-address)."
        )
    if s.get("password_unreadable"):
        return (
            "Using **your saved SMTP config** — but the saved password can't be "
            "decrypted (your account password likely changed) — re-enter it and "
            "**Save settings**."
        )
    complete = bool(
        (cfg.get("host") or "").strip() and (cfg.get("mail_from") or "").strip()
    )
    if not complete:
        return "⚠️ **Incomplete** — add a host and a from-address."
    status = (
        f"Using **your saved SMTP config** — **{s['host']}** / from "
        f"**{s['mail_from']}**."
    )
    # Google-only accounts have no password hash → the SMTP password is
    # stored plaintext (fail-open by design). Tell the recruiter instead of
    # letting them assume it is encrypted like a password account's.
    user = auth.active_user() or {}
    if not (user.get("password_hash") or "").strip():
        status += (
            "\n\n_Note: your account has no password, so the saved SMTP password "
            "is stored in plaintext (encryption needs a password-derived key)._"
        )
    return status


def _saved_email_settings() -> dict:
    """The account's own saved SMTP row ({} when none). The completeness of a
    saved account config is judged on THESE raw values — the resolved config
    (emailer.resolved_settings) must not mask a missing host/from-address the
    recruiter was asked to provide."""
    return db.get_email_settings() or {}


def _email_banner_update(s: dict | None = None, cfg: dict | None = None) -> dict:
    """Visible warning at the top of the Email tab when sends would fail:
    incomplete saved account config, an undecryptable saved password, or no
    SMTP sender at all. Hidden when the effective config is usable."""
    s = s or emailer.resolved_settings()
    cfg = cfg if cfg is not None else _saved_email_settings()
    saved_complete = bool(
        (cfg.get("host") or "").strip() and (cfg.get("mail_from") or "").strip()
    )
    if s.get("password_unreadable"):
        msg = (
            "⚠️ **Email sends will fail** — your saved SMTP password can't be "
            "decrypted (your account password likely changed). Open **⚙️ Email "
            "settings** and re-enter it."
        )
    elif cfg and not saved_complete:
        msg = (
            "⚠️ **Email sends will fail** — your saved SMTP settings are "
            "incomplete (missing host or from-address). Open **⚙️ Email "
            "settings** to fix them."
        )
    elif not (s["host"] and s["mail_from"]):
        msg = (
            "⚠️ **Email sends will fail** — no SMTP sender configured. Open "
            "**⚙️ Email settings** and add your SMTP details."
        )
    else:
        return gr.update(value="", visible=False)
    return gr.update(value=msg, visible=True)


def _email_settings_refresh(msg: str = "") -> tuple:
    """The 11 Email-settings outputs (9 form values + status + banner). `msg`
    (e.g. a save confirmation) is folded into the status line so the form's
    feedback and the config status share one visible output."""
    s = emailer.resolved_settings()
    cfg = _saved_email_settings()
    status = _email_settings_status(s, cfg)
    if msg:
        status = f"{msg}\n\n{status}"
    return (
        *_email_settings_values(s),
        status,
        _email_banner_update(s, cfg),
    )


def _profile_markdown(user: dict) -> str:
    """The Profile bubble's identity card: email, sign-in method, member since."""
    provider = "Google" if user.get("provider") == "google" else "Email & password"
    created = _fmt_dt(user.get("created_at") or "")
    return (
        f"**Email:** `{user.get('email') or ''}`  \n"
        f"**Sign-in method:** {provider}  \n"
        f"**Member since:** {created}"
    )


def _profile_refresh() -> tuple:
    """The 4 Profile-popover outputs (name, identity card, status, delete confirm)."""
    u = auth.active_user() or {}
    return (
        gr.update(value=u.get("name") or ""),   # pf_name
        gr.update(value=_profile_markdown(u)),    # pf_email
        gr.update(value=""),                     # pf_status
        gr.update(value=""),                     # pf_del_confirm
    )


@_require_session
def _on_save_profile(token, name):
    """Rename the signed-in account from the Profile popover."""
    user = auth.active_user() or {}
    updated = auth.update_user_name(user.get("id", ""), name)
    if not updated:
        return (gr.update(value="_Could not save — account not found._"), gr.update())
    # Keep the thread's active-user dict in sync so the new name is visible
    # to any later handler on this thread (e.g. a delete that reads it).
    active = auth.active_user()
    if active is not None:
        active["name"] = updated.get("name") or ""
    return (
        gr.update(value=f"Name updated to **{updated.get('name') or '—'}**."),
        gr.update(value=_profile_markdown(updated)),
    )


@_require_session
def _on_delete_account(token, confirm_text):
    """Permanently delete the signed-in account (after DELETE confirmation)
    and return to the login gate. Never touches data on a bad confirmation."""
    user = auth.active_user() or {}
    if (confirm_text or "").strip() != "DELETE":
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(value="Type **DELETE** to confirm — nothing was deleted."),
            gr.update(),
        )
    auth.delete_user(user.get("id", ""))
    auth.set_active_user(None)
    return (
        gr.update(visible=True),       # auth_view
        gr.update(visible=False),      # workspace_view
        gr.update(value=""),          # user_badge
        gr.update(value="_Account deleted — thanks for trying TalentIQ._"),  # auth_msg
        None,                          # session_token -> cleared
        gr.update(value="Sign in"),   # auth_mode
        gr.update(value=""),          # auth_email
        gr.update(value=""),          # auth_name
        gr.update(value=""),          # auth_password
        gr.update(value=""),          # pf_name
        gr.update(value=""),          # pf_email
        gr.update(value=""),          # pf_status
        gr.update(value=""),          # pf_del_confirm
    )


# Only raster formats are accepted for the email logo — SVG (and anything
# else) is rejected so a data-URI logo can never carry inline scripts.
_ALLOWED_LOGO_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
})


def _persist_company_logo(logo_path) -> str:
    """Copy an uploaded logo into this account's private folder and return the
    stored path. An empty/cleared field removes the previously stored logo.
    Raises ValueError for an oversized or unsupported upload."""
    user = auth.active_user() or {}
    if not logo_path or not str(logo_path).strip():
        # Cleared — remove the previously stored logo file (if any).
        prev = (emailer.resolved_settings().get("company_logo") or "").strip()
        if prev and os.path.isfile(prev):
            with suppress(OSError):
                os.remove(prev)
        return ""
    src = str(logo_path).strip()
    if not os.path.isfile(src):
        return ""
    ext = os.path.splitext(src)[1].lower()
    if ext not in _ALLOWED_LOGO_EXTS:
        raise ValueError("unsupported logo format — use PNG, JPG, GIF or WebP")
    max_bytes = config.MAX_LOGO_MB * 1024 * 1024
    if os.path.getsize(src) > max_bytes:
        raise ValueError(f"the logo is too large — max {config.MAX_LOGO_MB} MB")
    storage = auth.user_storage(user.get("id", ""))
    logos_dir = os.path.join(os.path.dirname(storage["db"]), "logos")
    dest = os.path.join(logos_dir, f"logo{ext}")
    # Re-saving without re-uploading feeds the stored path back in — copying
    # a file onto itself would truncate it, so keep it untouched.
    if os.path.abspath(src) == os.path.abspath(dest):
        return dest
    os.makedirs(logos_dir, exist_ok=True)
    # A new upload replaces any previous logo file.
    for name in os.listdir(logos_dir):
        with suppress(OSError):
            os.remove(os.path.join(logos_dir, name))
    shutil.copyfile(src, dest)
    return dest


def _save_email_settings_form(
    host, port, mail_from, mail_from_name, user, password, starttls,
    company_name, logo_path,
) -> tuple[bool, str, str]:
    """Persist the Email-settings form (SMTP + branding). Returns
    (saved, error_msg, logo_dest) — error_msg is non-empty when saved is False."""
    try:
        logo_dest = _persist_company_logo(logo_path)
    except ValueError as e:
        return False, f"**Logo not saved:** {e}", ""
    db.save_email_settings(
        host=host or "",
        port=port or 587,
        mail_from=mail_from or "",
        mail_from_name=mail_from_name or "",
        user=user or "",
        password=password or "",
        starttls=bool(starttls),
        company_name=company_name or "",
        company_logo=logo_dest,
    )
    return True, "", logo_dest


@_require_session
def on_save_email_settings(
    token, host, port, mail_from, mail_from_name, user, password, starttls,
    company_name, logo_path,
):
    """Persist this account's SMTP + branding settings and confirm."""
    ok, err, logo_dest = _save_email_settings_form(
        host, port, mail_from, mail_from_name, user, password, starttls,
        company_name, logo_path,
    )
    if not ok:
        return _email_settings_refresh(err)
    msg = (
        "Saved — this account now sends from "
        f"**{mail_from or '(no from-address)'}** via **{host or '(no host)'}**."
    )
    if (company_name or "").strip():
        msg += f"\n\nHeader brand: **{(company_name or '').strip()}**."
    if logo_dest:
        msg += " Company logo saved — it appears at the top of emails."
    return _email_settings_refresh(msg)


@_require_session
def on_clear_email_settings(token):
    """Drop the account's SMTP sender — sends stay disabled until a new sender
    is saved. The email branding (company name / logo) is kept."""
    db.clear_email_settings()
    msg = (
        "Cleared — your account now has no SMTP sender; sends stay disabled "
        "until you save new settings (company name / logo are kept)."
    )
    return _email_settings_refresh(msg)


@_require_session
def on_test_email_settings(
    token, host, port, mail_from, mail_from_name, user, password, starttls,
    company_name, logo_path, recipient,
):
    """Save the settings form first — the test always runs against the exact
    config shown in the bubble — then send a test email through it."""
    ok, err, _ = _save_email_settings_form(
        host, port, mail_from, mail_from_name, user, password, starttls,
        company_name, logo_path,
    )
    if not ok:
        return _email_settings_refresh(err)
    to = (recipient or "").strip() or (auth.active_user() or {}).get("email", "")
    if not to or "@" not in to:
        return _email_settings_refresh(
            "Settings saved — now enter a test recipient (or leave empty to "
            "use your account email) and click **Send test email** again."
        )
    if not emailer.is_configured():
        return _email_settings_refresh("Configure a host and from-address first.")
    subject, body = emailer.build_test_email()
    res = emailer.send_email(to, subject, body)
    if not res["ok"]:
        return _email_settings_refresh(f"**Test failed:** {res['error']}")
    return _email_settings_refresh(
        "Settings saved, then test email sent to "
        f"**{to}** — check the inbox (and spam folder)."
    )


@_require_session
def _open_email_settings(token):
    """Show the floating Email settings bubble + its click-away backdrop."""
    return gr.update(visible=True), gr.update(visible=True)


@_require_session
def _close_email_settings(token):
    """Hide the floating Email settings bubble + its click-away backdrop."""
    return gr.update(visible=False), gr.update(visible=False)


@_require_session
def _open_profile(token):
    """Show the floating Profile bubble + its click-away backdrop."""
    return gr.update(visible=True), gr.update(visible=True)


@_require_session
def _close_profile(token):
    """Hide the floating Profile bubble + its click-away backdrop."""
    return gr.update(visible=False), gr.update(visible=False)


@_require_session
def _close_bubbles(token):
    """Backdrop click: close every floating bubble (Profile + Email settings)."""
    return (
        gr.update(visible=False),  # bubble_backdrop
        gr.update(visible=False),  # profile_pop
        gr.update(visible=False),  # es_pop
    )


def _es_username_autofill(host: str, from_addr: str, username: str) -> dict:
    """Fill the SMTP username with the from-address when the picked provider
    logs in with the mailbox address (Gmail/Outlook/Yahoo/iCloud/Zoho/GMX)
    and the username field is still empty — never clobber a typed username.
    Returns a gr.update() that leaves the field unchanged otherwise."""
    if (
        (host or "").strip().lower() in _SMTP_USERNAME_IS_EMAIL
        and (from_addr or "").strip()
        and not (username or "").strip()
    ):
        return gr.update(value=(from_addr or "").strip())
    return gr.update()


@_require_session
def _on_es_host_change(token, host, from_addr, username, port):
    """Auto-fill the SMTP port when a known provider is picked from the
    dropdown, and default to 587 whenever the port is empty (custom hosts too)
    so the field never sits blank. Mirrors the from-address into the username
    for email-login providers."""
    host = (host or "").strip().lower()
    if host in _SMTP_PROVIDERS:
        port_update = gr.update(value=_SMTP_PROVIDERS[host])
    elif not port:
        port_update = gr.update(value=587)
    else:
        port_update = gr.update()
    return port_update, _es_username_autofill(host, from_addr, username)


@_require_session
def _on_es_from_change(token, host, from_addr, username):
    """When the from-address is filled after picking an email-login provider,
    mirror it into the username too (still only when username is empty)."""
    return _es_username_autofill(host, from_addr, username)


# ================================================================
# Live interview handlers (free meeting link → live transcript → Q&A → score)
# ================================================================

def _video_history_table(job_id: str | None = None) -> list[list]:
    rows = db.list_video_interviews(job_id or None)
    out: list[list] = []
    for r in rows:
        qa = len(db.parse_json_field(r.get("qa_json"), []))
        out.append([
            _fmt_dt(r.get("created_at")),
            r.get("candidate_name") or r.get("candidate_id"),
            qa,
            r.get("average_score") or 0,
            r.get("verdict") or "",
        ])
    return out


def _clip(text: str, limit: int = 6000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n_… transcript truncated for display (the full copy is stored in the database)._"


@_require_session
def on_generate_meeting_link(token, job_id, candidate_id):
    """Create a free Jitsi meeting link for the selected job + candidate."""
    if not job_id or not candidate_id:
        return "", "_Select a **job** and a **candidate** first._"
    link = live_interview.generate_meeting_link(job_id, candidate_id)
    hint = (
        "_**Jitsi** room created instantly (free, no account, no time "
        "limit) — this link opens the meeting directly. Send it via "
        "**Email → Interview invite**, or share it with the candidate._"
    )
    return link, hint


@_require_session
def on_live_start(token, job_id, candidate_id, meeting_link, language="English"):
    """Gate on selections, then start the live transcription session
    (browser-mic chunks are appended by on_live_chunk). Any stale review &
    fix text from a previous run is cleared."""
    if not job_id or not candidate_id:
        return (
            "",
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "Select a **job** and a **candidate** first.",
            gr.update(visible=False, value=""),
        )
    session = live_interview.start_live_session(
        job_id,
        candidate_id,
        meeting_link=meeting_link or "",
        language=language or "English",
    )
    return (
        session.session_id,
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        (
            "🔴 **Recording** — press the **record** button on the microphone "
            "and start the call. The transcript streams below every ~10s."
        ),
        gr.update(visible=False, value=""),  # clear stale review text
    )


@_require_session
def on_live_chunk(token, session_id, audio):
    """Append each browser-mic chunk to the active session and refresh the
    rolling transcript."""
    session = live_interview.get_live_session(session_id or "")
    if session is None:
        return (
            "No active recording — click **Start live transcription** first.",
            gr.update(),
        )
    if session.status != "recording":
        # The mic keeps streaming after Stop — don't clobber the
        # "Stopped — review & fix" status with an error; just refresh it.
        return live_interview.session_summary(session), gr.update()
    try:
        if isinstance(audio, (tuple, list)) and len(audio) == 2:
            sr, data = audio
        else:
            sr, data = None, audio
    except Exception:
        sr, data = None, None
    if data is None:
        return live_interview.session_summary(session), gr.update()
    live_interview.append_audio_chunk(session, sr or session.sample_rate, data)
    return (
        live_interview.session_summary(session),
        gr.update(value=_clip(live_interview.transcript_md(session))),
    )


def _on_live_tick(token, session_id):
    """Timer tick — push transcript/status updates even between mic chunks.
    Reads only in-memory session state, so it never touches per-user data."""
    if not token:
        return gr.update(), gr.update()
    session = live_interview.get_live_session(session_id or "")
    if session is None:
        return gr.update(), gr.update()
    return (
        live_interview.session_summary(session),
        gr.update(value=_clip(live_interview.transcript_md(session))),
    )


@_require_session
def on_live_stop(token, session_id):
    """Stop recording, transcribe the audio captured since the last segment,
    and surface the full transcript in the editable review & fix box so
    Whisper errors can be corrected before Finish & evaluate runs."""
    session = live_interview.get_live_session(session_id or "")
    if session is None:
        return "No active live session.", gr.update(), gr.update()
    text, err = live_interview.stop_live_session(session_id)
    if err:
        return f"**Could not transcribe the remainder:** {err}", gr.update(), gr.update()
    return (
        (
            "⏹ Stopped — **review & fix** any misheard words in the transcript "
            "below, then click **Finish & evaluate** to score the corrected "
            "answers."
        ),
        gr.update(value=_clip(live_interview.transcript_md(session))),
        gr.update(visible=True, value=text or ""),
    )


@_require_session
def on_live_finish(token, session_id, language="English", transcript_edit=""):
    """Run the full live pipeline: final transcript → speaker separation →
    Q&A → RAG evaluation → saved to the live interview history. Any text in
    the review & fix box is used as the authoritative transcript (so Whisper
    errors the recruiter corrected never reach the evaluation)."""
    session = live_interview.get_live_session(session_id or "")
    if session is None:
        return (
            "", gr.update(interactive=True), gr.update(interactive=False),
            gr.update(interactive=False),
            "No live session found — start one first.",
            gr.update(), gr.update(), gr.update(),
            gr.update(visible=False, value=""),
        )
    result = live_interview.finish_live_interview(
        session_id,
        language=language or "English",
        override_transcript=transcript_edit or "",
    )
    history = gr.update(value=_video_history_table(session.job_id))
    if result.error:
        return (
            "", gr.update(interactive=True), gr.update(interactive=False),
            gr.update(interactive=False),
            f"**Failed:** {result.error}",
            gr.update(value=_clip(result.transcript)),
            gr.update(),
            history,
            gr.update(visible=False, value=""),
        )
    cand = db.get_candidate(session.candidate_id) or {}
    status = (
        f"✅ Analyzed **{cand.get('name', session.candidate_id)}** — "
        f"{len(result.qa_pairs)} Q&A pair(s) from {len(result.turns)} speaker "
        f"turns, average **{result.average_score:.1f}/10 · {result.verdict}**."
    )
    return (
        "", gr.update(interactive=True), gr.update(interactive=False),
        gr.update(interactive=False),
        status,
        gr.update(value=_clip(result.transcript)),
        result.evaluation,
        history,
        gr.update(visible=False, value=""),
    )


@_require_session
def on_suggest_questions(token, job_id, candidate_id, language="English"):
    """Show the 10 AI-suggested questions for the selected job + candidate."""
    if not job_id or not candidate_id:
        return "Select a **job** and a **candidate** first."
    from screening import suggest_interview_questions

    questions, _err = suggest_interview_questions(
        job_id, candidate_id, language=language or "English"
    )
    if not questions:
        return "Could not generate questions — is a screening available?"
    return _questions_markdown(questions)


@_require_session
def _on_int_mode_change(token, mode):
    """Toggle the chat vs live-meeting panels in the Interview tab."""
    return (
        gr.update(visible=not mode or str(mode).startswith("Chat")),
        gr.update(visible=bool(mode) and str(mode).startswith("Live")),
    )


# ================================================================
# Theme & CSS
# ================================================================

custom_css = """
:root {
  --brand: #0f766e;
  --brand-dark: #115e59;
  --ink: #0f172a;
  --ink-soft: #1e293b;
  --muted: #475569;
  --placeholder: #64748b;
  --surface: #ffffff;
  --line: #e2e8f0;
  --wash: #f0fdfa;
  --pass-bg: #f0fdf4;
  --pass-border: #bbf7d0;
  --fail-bg: #fef2f2;
  --fail-border: #fecaca;
  --pass-text: #166534;
  --fail-text: #991b1b;
  --danger: #dc2626;
  --danger-dark: #b91c1c;
}

.gradio-container {
  /* Explicit emoji-capable fallbacks so labels/selected values with 🤖📊💻
     always resolve to a color-emoji font (DM Sans has no emoji glyphs). */
  font-family: 'DM Sans', 'Segoe UI', 'Segoe UI Emoji', 'Apple Color Emoji',
    'Noto Color Emoji', sans-serif !important;
  max-width: 1180px !important;
  width: 100% !important;
  margin: 0 auto !important;
  padding: 0 1rem !important;
  box-sizing: border-box !important;
  color: var(--ink-soft) !important;
}

/* The app frame sits on the page <body> — when the OS is in dark mode,
   Gradio resolves `body { background: var(--bg) }` to a dark navy that shows
   around the 1180px-wide app as a jarring black frame. Force the page canvas
   to the same light gradient so the whole window is one seamless light
   surface regardless of OS dark-mode preference. */
body {
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
}
.gradio-container, .main, .contain {
  background:
    radial-gradient(ellipse 80% 50% at 10% -10%, rgba(15, 118, 110, 0.09), transparent 55%),
    radial-gradient(ellipse 60% 40% at 100% 0%, rgba(14, 165, 233, 0.06), transparent 50%),
    linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
}

/* ===== Force light surfaces =====
   Gradio inherits the OS dark-mode preference and tags <body> with class
   "dark", which silently turns containers/dropdowns into dark navy surfaces.
   That clobbers this light design and makes text/options blend into the
   background. Pin every theme surface to light values inside the app. */
.gradio-container {
  color-scheme: light;
  --body-background-fill: #f8fafc !important;
  --body-text-color: var(--ink) !important;
  --body-text-color-subdued: var(--muted) !important;
  --background-fill-primary: #f8fafc !important;
  --background-fill-secondary: #f8fafc !important;
  --block-background-fill: #ffffff !important;
  --block-border-color: var(--line) !important;
  --border-color-primary: var(--line) !important;
  --border-color-accent: rgba(15, 118, 110, 0.3) !important;
  --border-color-accent-subdued: rgba(15, 118, 110, 0.3) !important;
  --checkbox-border-color: var(--line) !important;
  --checkbox-border-color-hover: var(--line) !important;
  --block-info-text-color: var(--muted) !important;
  --block-label-background-fill: var(--wash) !important;
  --block-label-text-color: var(--brand-dark) !important;
  --block-title-background-fill: var(--wash) !important;
  --block-title-border-color: rgba(15, 118, 110, 0.18) !important;
  --block-title-radius: 999px !important;
  --block-title-padding: 2px 10px !important;
  --block-title-text-color: var(--ink) !important;
  --input-background-fill: #ffffff !important;
  --input-background-fill-hover: #ffffff !important;
  --input-border-color: var(--line) !important;
  --input-border-color-hover: var(--line) !important;
  --input-placeholder-color: var(--placeholder) !important;
  --table-even-background-fill: #ffffff !important;
  --table-odd-background-fill: #f8fafc !important;
  --table-border-color: var(--line) !important;
  --table-text-color: var(--ink-soft) !important;
  --panel-background-fill: #ffffff !important;
  --checkbox-label-text-color: var(--ink-soft) !important;
  --button-secondary-background-fill: #ffffff !important;
  --button-secondary-background-fill-hover: var(--wash) !important;
  --button-secondary-text-color: var(--ink-soft) !important;
  --button-secondary-border-color: var(--line) !important;
}

/* ===== Better default text contrast ===== */
label, p, li, td, th, .prose, .markdown {
  color: var(--ink-soft) !important;
}
strong, b {
  color: var(--ink) !important;
}
/* NOTE: Gradio 6 renders gr.Markdown inside a `.prose` block (the old
   `.markdown` class is gone), so every markdown rule below lists both
   selectors — otherwise the typography polish silently never applies. */
.markdown p, .markdown li, .prose p, .prose li {
  color: var(--ink-soft) !important;
  font-size: 0.88rem;
  line-height: 1.6;
}
.markdown h1, .markdown h2, .markdown h3, .markdown h4,
.prose h1, .prose h2, .prose h3, .prose h4 {
  color: var(--ink) !important;
}
.markdown code, .prose code {
  background: #f1f5f9;
  color: #1e293b;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-size: 0.82rem;
}
.markdown em, .prose em {
  color: var(--ink-soft) !important;
}

/* ===== High-contrast controls =====
   Keeps every piece of text legible: dropdown options, inputs, labels,
   multi-select chips, and checkboxes — on light backgrounds only. */

/* Input text is always clearly dark on white */
input, textarea, select {
  color: var(--ink) !important;
  caret-color: var(--brand);
  background-color: var(--surface) !important;
}

/* Checkboxes & radios must NOT inherit the forced white background above —
   Gradio paints the tick/circle in WHITE on a TEAL background when checked,
   so a white background would make the tick invisible (the box looked
   checked but showed no checkmark). Let Gradio's checked styling win. */
input[type="checkbox"], input[type="radio"] {
  background-color: var(--checkbox-background-color, var(--surface)) !important;
}
input[type="checkbox"]:checked, input[type="radio"]:checked {
  background-color: var(--checkbox-background-color-selected, var(--brand)) !important;
  border-color: var(--checkbox-border-color-selected, var(--brand)) !important;
}
/* Partial-selection state (e.g. the Dataframe "Toggle all" box when some
   rows are ticked) must also paint teal — Gradio's non-important
   :indeterminate rule would otherwise lose to the white override above. */
input[type="checkbox"]:indeterminate {
  background-color: var(--checkbox-background-color-selected, var(--brand)) !important;
  border-color: var(--checkbox-border-color-selected, var(--brand)) !important;
}

/* ===== Emoji glyphs =====
   DM Sans ships no emoji glyphs, so any emoji in labels, dropdown values,
   multi-select chips or buttons resolves via the fallback stack. Push the
   color-emoji fonts onto every text-bearing control — inputs, dropdown
   internals, chips and buttons — so 🤖📊💻 always render as color glyphs
   (never monochrome or tofu) in selected values too. */
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container button,
.gradio-container .single-value,
.gradio-container .options .item,
.gradio-container .chip,
.gradio-container .selected-item,
.gradio-container .token {
  font-family: 'DM Sans', 'Segoe UI', 'Segoe UI Emoji', 'Apple Color Emoji',
    'Noto Color Emoji', sans-serif !important;
}
.gradio-container .single-value,
.gradio-container .chip,
.gradio-container .selected-item,
.gradio-container .token {
  line-height: 1.55;
}
input[disabled], textarea[disabled] {
  color: var(--muted) !important;
}

/* Field labels / block titles stay legible (not washed out) */
.block label span, .block label, .block .label, label {
  color: var(--ink-soft) !important;
}

/* Native <select> fallback (some browsers render options natively) */
select, select option {
  color: var(--ink-soft) !important;
  background-color: var(--surface) !important;
}
select option:checked {
  color: var(--brand-dark) !important;
  background-color: var(--wash) !important;
}

/* ===== Dropdowns =====
   Gradio 6 renders a dropdown as `.container > .wrap > .wrap-inner` (the
   closed box with single-value + caret). When opened, the SAME `.wrap`
   becomes the backdrop around the `ul.options` popup — and in dark mode
   that backdrop is solid slate, showing as a dark navy box around the list
   (the "odd" look). Pin the backdrop to white, round everything to match
   the app, and give the focused box a brand-teal ring. */

/* Closed dropdown box — match input radius, crisp border */
.container .wrap {
  border-radius: 10px !important;
  border-color: var(--line) !important;
}
.container .single-value {
  color: var(--ink) !important;
}

/* The selected value is typed into a bare input (no padding) that runs the
   whole box width — without right padding the text slides UNDER the caret
   icon. Give the input a right gutter sized to the caret (icon-wrap is
   ~20px + 8px gap) and let long values ellipsize instead of overlapping. */
.container input[role="combobox"] {
  padding-right: 2.1rem !important;
  overflow: hidden !important;
  text-overflow: ellipsis;
  white-space: nowrap;
}
/* Keep the caret itself tappable and unhidden; it just sits in its own
   right gutter now that the text no longer runs under it. */
.container .icon-wrap {
  width: 20px;
  height: 20px;
  right: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.container .icon-wrap svg {
  width: 16px;
  height: 16px;
  color: var(--muted);
}

/* Open popup: the same .wrap becomes the backdrop around ul.options */
.container .wrap:has(ul.options) {
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.16) !important;
  padding: 4px !important;
}

/* The option list itself */
.container .options {
  background: var(--surface) !important;
  border: none !important;
  border-radius: 8px !important;
  box-shadow: none !important;
  max-height: 300px !important;
  overflow-y: auto !important;
}
.container .options .item {
  color: var(--ink-soft) !important;
  background: transparent !important;
  padding: 0.5rem 0.8rem !important;
  font-weight: 500;
  border-radius: 6px;
  margin: 2px 4px;
  transition: background-color 0.12s ease, color 0.12s ease;
}
.container .options .item:hover {
  color: var(--ink) !important;
  background: #f1f5f9 !important;
}
.container .options .item.selected,
.container .options .item[aria-selected="true"],
.container .options .item.active {
  color: var(--brand-dark) !important;
  background: var(--wash) !important;
  font-weight: 600;
}
/* The built-in "✓" prefix span marks the selected option. Hide it ONLY for
   unselected options (Gradio tags those with its own `.hide` class) and
   show it for the selected / highlighted one — the checkmark is the
   universal "this is selected" signal in a dropdown. Scoping the hide to
   `.inner-item.hide` guarantees the selected option's checkmark can never
   be suppressed, whatever CSS prefixing Gradio applies. */
.container .options .inner-item.hide {
  display: none !important;
}
.container .options .item.selected .inner-item,
.container .options .item[aria-selected="true"] .inner-item,
.container .options .item.active .inner-item {
  display: inline-block !important;
  color: var(--brand) !important;
  font-weight: 700;
}

/* Brand-teal focus ring on the dropdown box (replaces Gradio's dark ring) */
.container:focus-within > .wrap {
  border-color: var(--brand) !important;
  box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.15) !important;
}

/* Multi-select chips / selected tokens */
.chip, .selected-item, .token, .wrap .chip {
  color: var(--brand-dark) !important;
  background: var(--wash) !important;
  border: 1px solid rgba(15, 118, 110, 0.28) !important;
}
.chip button, .selected-item button {
  color: var(--brand-dark) !important;
}

/* Native fallback for browsers that keep default appearance */
input[type="checkbox"], input[type="radio"] {
  accent-color: var(--brand);
}

/* PASS/FAIL badges */
.badge {
  display: inline-block;
  padding: 0.16rem 0.6rem;
  border-radius: 999px;
  border: 1px solid transparent;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  line-height: 1.4;
  vertical-align: middle;
}
.badge-pass {
  background: var(--pass-bg);
  color: var(--pass-text) !important;
  border-color: var(--pass-border);
}
.badge-fail {
  background: var(--fail-bg);
  color: var(--fail-text) !important;
  border-color: var(--fail-border);
}

/* ===== Contrast fixes for screening output ===== */
.markdown ul li, .prose ul li {
  margin: 0.35rem 0;
  color: var(--ink-soft) !important;
}
.markdown ul li code, .prose ul li code {
  background: #e2e8f0;
  color: #1e293b;
  font-weight: 500;
}
.markdown ul li em, .prose ul li em {
  font-size: 0.85rem;
  color: #1e293b !important;
  font-style: italic;
}

/* ---------- Brand hero ---------- */
.brand-hero {
  padding: 1.75rem 1.5rem 1.4rem;
  margin-bottom: 0.9rem;
  border-radius: 18px;
  border: 1px solid rgba(15, 118, 110, 0.18);
  background:
    linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,253,250,0.9) 100%);
  box-shadow: 0 10px 40px rgba(15, 23, 42, 0.04);
}

.brand-hero h1 {
  font-family: 'Fraunces', Georgia, serif !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  letter-spacing: -0.02em;
  margin: 0 0 0.35rem 0 !important;
}

.brand-hero .tagline {
  color: var(--muted) !important;
  font-size: 0.98rem;
  margin: 0;
}

.brand-hero .hero-badge {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--brand-dark);
  background: var(--wash);
  border: 1px solid rgba(15, 118, 110, 0.22);
  padding: 0.24rem 0.6rem;
  border-radius: 999px;
  margin-top: 0.8rem;
}
/* ---------- Always-visible nav pills ---------- */
/* The radio container (fieldset) must not inherit dark-mode surfaces; the
   pill bar lives in its inner .wrap. The auto "Radio" block-info label is
   hidden — the pills are self-explanatory. */
.nav-pills {
  margin: 0 0 1.1rem;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  overflow: visible !important;
}
.nav-pills [data-testid="block-info"] {
  display: none !important;
}
.nav-pills > .wrap:not(.hide) {
  display: flex !important;
  flex-wrap: wrap;
  gap: 0.45rem;
  padding: 0.5rem 0.6rem;
  background: rgba(255,255,255,0.92) !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
}
.nav-pills label {
  cursor: pointer;
  flex: 0 1 auto;
  text-align: center;
  padding: 0.55rem 1rem;
  border-radius: 10px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--muted) !important;
  font-weight: 600;
  font-size: 0.85rem;
  letter-spacing: 0.01em;
  transition: all 0.16s ease;
}
.nav-pills label:hover {
  color: var(--brand-dark) !important;
  background: var(--wash);
}
.nav-pills label:has(input:checked) {
  background: var(--brand) !important;
  border-color: var(--brand) !important;
  color: #fff !important;
  box-shadow: 0 2px 6px rgba(15, 118, 110, 0.18);
}
.nav-pills input[type="radio"] {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}

.tab-body {
  animation: fadeSlide 0.2s ease;
}
@keyframes fadeSlide {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ---------- AI deep-screen loader (animated 'evaluating' section) ---------- */
.ai-loader {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  padding: 0.95rem 1.15rem;
  border-radius: 12px;
  border: 1px solid rgba(15, 118, 110, 0.25);
  background: linear-gradient(135deg, rgba(240, 253, 250, 0.95), rgba(255, 255, 255, 0.98));
  box-shadow: 0 2px 12px rgba(15, 23, 42, 0.06);
  margin: 0.4rem 0;
}
.ai-loader .spinner {
  width: 26px;
  height: 26px;
  border: 3px solid rgba(15, 118, 110, 0.2);
  border-top-color: var(--brand, #0f766e);
  border-radius: 50%;
  animation: ai-spin 0.9s linear infinite;
  flex-shrink: 0;
}
@keyframes ai-spin {
  to { transform: rotate(360deg); }
}
.ai-loader .loader-text {
  color: var(--ink-soft) !important;
  font-size: 0.9rem;
  line-height: 1.5;
}
.ai-loader .loader-text strong {
  color: var(--ink) !important;
}

/* ---------- Panels & tables ---------- */
.panel {
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  padding: 1rem 1.1rem !important;
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
  margin-bottom: 0.9rem;
}

footer, .footer { display: none !important; }

.workspace-foot {
  text-align: center;
  color: var(--muted);
  font-size: 0.8rem;
  padding: 1.2rem 0 0.4rem;
  border-top: 1px solid var(--line);
  margin-top: 1.1rem;
}

/* Dataframe polish — tables are fluid and fit their panel width on every
   screen, headers never wrap, body cells wrap on small screens, and any
   overflow scrolls horizontally instead of squashing columns
   ("Can/dida/tes" never happens). Header and data rows share one padding +
   vertical alignment so columns line up at every breakpoint. */
.gradio-container .table-wrap {
  border-radius: 10px;
  border: 1px solid var(--line);
  overflow-x: auto !important;
  -webkit-overflow-scrolling: touch;
}
.gradio-container table {
  width: 100%;
  min-width: 0;
  border-collapse: collapse;
  font-size: 0.85rem;
}
.gradio-container th {
  background: #f8fafc !important;
  color: var(--ink) !important;
  font-weight: 600;
  padding: 0.45rem 0.75rem !important;
  border-bottom: 2px solid var(--line) !important;
  text-align: left;
  vertical-align: middle;
  white-space: nowrap !important;
}
/* Gradio 6 puts the header label inside a `.header-content > span` that it
   tags with a `wrap` class (`white-space: normal`) — that is why "Candidates"
   wraps to "Candidat es" even though the <th> itself says nowrap. Force the
   inner span to stay on one line so headers never wrap, and slim the cell so
   the header row reads like a real table header (Gradio's 36px cell
   min-height makes it chunky). */
.gradio-container .table-wrap th .header-content,
.gradio-container .table-wrap th .header-content span,
.gradio-container .table-wrap th span {
  white-space: nowrap !important;
  color: var(--ink) !important;
}
/* Header text must never be clipped mid-word or ellipsized — columns sized
   with enough width, and the wrapper allowed to show the full label. The th
   carries the SAME padding as td so header text aligns pixel-perfect with the
   data rows below it; the inner .cell-wrap must not add a second layer of
   padding (that pushed header text right, misaligning it with the body). */
.gradio-container .table-wrap th .header-content {
  overflow: visible !important;
}
.gradio-container .table-wrap th .cell-wrap {
  min-height: 0 !important;
  padding: 0 !important;
}
.gradio-container td {
  padding: 0.45rem 0.75rem !important;
  color: var(--ink-soft) !important;
  text-align: left;
  vertical-align: middle;
  white-space: nowrap !important;
}
/* Zebra striping + visible hover so rows read cleanly at a glance */
.gradio-container .table-wrap tbody tr:nth-child(even) td {
  background: #f8fafc !important;
}
.gradio-container .table-wrap tbody tr:hover td {
  background: #eef2f6 !important;
}
/* Markdown tables must never spill past the panel edge — let the TABLE itself
   scroll horizontally inside its block, without turning the whole prose
   container into a scroll area (that clipped callout text). */
.gradio-container .prose table,
.gradio-container .markdown table {
  display: block;
  overflow-x: auto;
  max-width: 100%;
}
/* Gradio 6's interactive Dataframe draws teal column/row selection grab
   handles over the first cell (covers the tick box + # column) — pure
   visual noise for checkbox rows, so hide them. */
.gradio-container .table-wrap .selection-button {
  display: none !important;
}
/* The per-cell "⋮" menu buttons (copy/paste cell) pop over headers and rows
   on hover, covering the labels — unused in this app, so hide them. */
.gradio-container .table-wrap .cell-menu-button {
  display: none !important;
}
/* The "Add row" (+) button on interactive Dataframes invites blank rows that
   would corrupt the pipeline/position mapping — rows are managed only through
   the app's own buttons, so hide it. (It sits OUTSIDE .table-wrap, next to the
   table block.) */
.add-row-button {
  display: none !important;
}
/* Header cells stay clean: no teal accent ring bleeding over the label
   when a column is clicked, and no hover chrome. */
.gradio-container .table-wrap .header-cell.focus {
  box-shadow: none !important;
  --ring-color: transparent !important;
}
/* No teal focus ring while tabbing through cells/headers inside a table */
.gradio-container .table-wrap *:focus-visible {
  outline: none !important;
}

/* ===== Buttons — one consistent system =====
   Three semantic variants, identical shape:
     primary   → solid brand teal   (the main action of each panel)
     secondary → white + hairline   (supporting / neutral actions)
     stop      → solid red          (destructive: delete, remove permanently)
   Same height, radius, weight and padding everywhere for consistency. */
button {
  font-weight: 600 !important;
  border-radius: 10px !important;
  min-height: 38px !important;
  padding: 0.45rem 1.1rem !important;
  letter-spacing: 0.01em;
  transition: box-shadow 0.15s ease, transform 0.15s ease,
    background-color 0.15s ease, border-color 0.15s ease,
    color 0.15s ease !important;
}
button.primary {
  background-color: var(--brand) !important;
  border: 1px solid var(--brand) !important;
  color: #ffffff !important;
  box-shadow: 0 4px 14px rgba(15, 118, 110, 0.18);
}
button.primary:hover {
  background-color: var(--brand-dark) !important;
  border-color: var(--brand-dark) !important;
  box-shadow: 0 6px 18px rgba(15, 118, 110, 0.28);
  transform: translateY(-1px);
}
button.secondary {
  background-color: #ffffff !important;
  border: 1px solid var(--line) !important;
  color: var(--ink-soft) !important;
  box-shadow: none !important;
}
button.secondary:hover {
  background-color: var(--wash) !important;
  border-color: rgba(15, 118, 110, 0.35) !important;
  color: var(--brand-dark) !important;
  transform: translateY(-1px);
}
button.stop {
  background-color: var(--danger) !important;
  border: 1px solid var(--danger) !important;
  color: #ffffff !important;
  box-shadow: 0 4px 14px rgba(220, 38, 38, 0.16);
}
button.stop:hover {
  background-color: var(--danger-dark) !important;
  border-color: var(--danger-dark) !important;
  box-shadow: 0 6px 18px rgba(220, 38, 38, 0.24);
  transform: translateY(-1px);
}
button:active {
  transform: translateY(0) scale(0.99);
}
:focus-visible {
  outline: 3px solid rgba(15, 118, 110, 0.35) !important;
  outline-offset: 2px;
  border-radius: 8px;
}

/* ===== Screening report cards ===== */
.markdown hr, .prose hr {
  margin: 1.2rem 0;
  border: 0;
  border-top: 2px solid var(--line);
}
.markdown h3, .markdown h4, .markdown h5,
.prose h3, .prose h4, .prose h5 {
  margin-top: 1.2rem !important;
  margin-bottom: 0.4rem !important;
}
.markdown h3, .prose h3 {
  font-size: 0.98rem !important;
  letter-spacing: -0.01em;
}

/* ===== KPI stat cards ===== */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.7rem;
  margin: 0.7rem 0 0.2rem;
}
.kpi {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.8rem 1rem;
  text-align: center;
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.06);
}
.kpi-num {
  display: block;
  font-size: 1.55rem;
  font-weight: 700;
  line-height: 1.15;
  color: var(--brand-dark);
}
.kpi-label {
  display: block;
  margin-top: 0.25rem;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
}

/* ===== Status callouts =====
   Plain, clean notification box: soft mint fill, subtle border, rounded
   corners on all sides, and NO left-edge decoration (no accent bar, no
   icon) — there is nothing on the left edge that could read as a bracket
   or overlap the text on any screen.

   IMPORTANT selector: Gradio marks BOTH the outer markdown box and the inner
   <span class="md prose"> with the class "prose". Styling `.status-note .prose`
   hits both, so the inner span would render a SECOND nested border + padding
   inside the box — inflating the height, pushing the text around, and (when an
   icon/bar was used) duplicating it into "two bars / two icons". The selector
   below targets only the element that carries BOTH "status-note" and "prose"
   (the outer box), so the inner span stays unstyled.

   The box is never a scroll container (overflow visible), so wrapped text
   always stays fully visible. */
.status-note { overflow: visible !important; }
.status-note.prose:not(:empty) {
  background: var(--wash);
  border: 1px solid rgba(15, 118, 110, 0.18);
  border-radius: 10px;
  padding: 0.7rem 0.9rem;
  margin: 0.6rem 0 0.2rem;
  overflow: visible !important;
}
.status-note.prose p {
  margin: 0;
}

/* ===== Email config warning banner (top of the Email tab) =====
   Amber warning box shown while the effective SMTP config would make every
   send fail (incomplete account settings, undecryptable password, or no
   SMTP at all). Hidden via `visible=False` when the config is usable. */
.email-warn.prose:not(:empty) {
  background: #fef3c7;
  border: 1px solid rgba(217, 119, 6, 0.45);
  border-radius: 10px;
  padding: 0.7rem 0.9rem;
  margin: 0 0 0.9rem;
  overflow: visible !important;
}
.email-warn.prose p {
  margin: 0;
}

/* ===== Floating Email settings bubble =====
   The ⚙️ Email settings button opens an absolutely-positioned card over the
   Email tab (gr.Popover was removed in Gradio 6, so a visibility-toggled
   panel reproduces the bubble UX). The Email tab body is the positioning
   context; the card floats top-right with a shadow and a high z-index. */
.tab-body {
  position: relative;
}
.email-settings-pop {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 60;
  width: 480px;
  max-width: 100%;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: 0 18px 44px rgba(15, 23, 42, 0.18);
  padding: 0.5rem 0.9rem 0.9rem;
}

/* ===== Floating Profile bubble =====
   The 👤 icon in the session bar opens an absolutely-positioned card over
   the workspace (same pattern as the Email settings bubble). */
.profile-pop {
  position: absolute;
  top: 46px;
  right: 12px;
  z-index: 60;
  width: 380px;
  max-width: calc(100% - 24px);
  max-height: calc(100vh - 64px);
  overflow-y: auto;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: 0 18px 44px rgba(15, 23, 42, 0.18);
  padding: 0.5rem 0.9rem 0.9rem;
}
.profile-icon-btn {
  min-width: 2.4rem !important;
  max-width: 2.4rem !important;
  height: 2.4rem !important;
  border-radius: 50% !important;
  padding: 0 !important;
  font-size: 1.05rem !important;
  border: 1px solid var(--line) !important;
  background: #fff !important;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
  transition: background 0.15s ease, box-shadow 0.15s ease !important;
}
.profile-icon-btn:hover {
  background: #f0fdfa !important;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.16) !important;
}

/* ===== Bubble click-away backdrop =====
   A transparent full-workspace overlay shown behind whichever floating
   bubble is open (Profile or Email settings). It sits below the bubbles
   (z-index 55 < 60) but above the page, so a click anywhere outside the
   bubble lands on it and closes it — same as clicking ✕. `inset: -60px`
   bleeds the overlay past the workspace's padded content box (the anchor
   is the content area, not the viewport), so clicks along the container
   edges still dismiss the bubble — don't "fix" it back to 0. */
.bubble-backdrop {
  position: absolute !important;
  inset: -60px !important;
  z-index: 55;
  background: transparent;
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  min-height: 0 !important;
}
.bubble-backdrop p,
.bubble-backdrop .prose {
  margin: 0 !important;
  min-height: 0 !important;
}

/* ===== Chatbot ===== */
.chatbot .message-item {
  border-radius: 14px !important;
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.07);
  border-color: var(--line) !important;
}
.chatbot .user-message {
  background: linear-gradient(135deg, var(--brand) 0%, var(--brand-dark) 100%) !important;
  border-color: var(--brand) !important;
  color: #ffffff !important;
}
.chatbot .user-message,
.chatbot .user-message .message-content,
.chatbot .user-message .message-text,
.chatbot .user-message p,
.chatbot .user-message li,
.chatbot .user-message strong,
.chatbot .user-message em,
.chatbot .user-message a {
  color: #ffffff !important;
}
.chatbot .user-message a {
  text-decoration: underline;
}
.chatbot .user-message code {
  background: rgba(255, 255, 255, 0.18) !important;
  color: #ffffff !important;
}
.chatbot .bot-message {
  background: #ffffff !important;
}

/* ===== Inputs & placeholders ===== */
input::placeholder, textarea::placeholder {
  color: var(--placeholder) !important;
  opacity: 1;
}
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: var(--brand) !important;
  box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12) !important;
}

/* ===== Scrollbars ===== */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}
* {
  scrollbar-width: thin;
  scrollbar-color: #cbd5e1 transparent;
}
::selection {
  background: rgba(15, 118, 110, 0.16);
}
html {
  scroll-behavior: smooth;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

/* ---------- Responsive ----------
   ⚠️ Gradio prefixes every custom-CSS rule with `.gradio-container
   .gradio-container-<ver> .contain` and, for rules inside @media blocks, it
   keeps ONLY the prefixed copy. Selectors inside media queries must therefore
   NOT start with `.gradio-container` or `.contain` — they would be double-
   prefixed and never match (this silently killed the old mobile rules). The
   container itself is already fluid (width:100% + max-width:1180px) from the
   base rules, so media blocks only restyle inner elements. */

@media (max-width: 900px) {
  .brand-hero { padding: 1.2rem 1rem 1rem; }
  .brand-hero h1 { font-size: 1.5rem !important; }
  .nav-pills .wrap {
    border-radius: 14px;
    padding: 0.35rem;
    gap: 0.3rem;
  }
  .nav-pills label {
    padding: 0.5rem 0.7rem;
    font-size: 0.8rem;
    flex: 1 1 auto;
  }
  /* Stack side-by-side panels vertically */
  .app-row {
    flex-direction: column !important;
    gap: 0.6rem !important;
  }
  .app-row > * {
    min-width: 100% !important;
    flex: 1 1 100% !important;
  }
  .interview-row {
    flex-direction: column !important;
    gap: 0.4rem !important;
  }
  .panel { padding: 0.6rem !important; }
  /* Email settings & profile popups shrink to fit */
  .email-settings-pop { width: 380px; right: 5px; }
  .profile-pop { width: 340px; right: 8px; }
  /* KPI grid: 3 columns on tablets */
  .kpi-grid { grid-template-columns: repeat(3, 1fr); }
  /* Session bar wraps gracefully */
  .session-bar { flex-wrap: wrap; gap: 0.5rem; }
}

@media (max-width: 768px) {
  /* Tighter container padding on smaller tablets */
  .brand-hero { padding: 1rem 0.8rem 0.8rem; margin-bottom: 0.7rem; border-radius: 14px; }
  .brand-hero h1 { font-size: 1.35rem !important; }
  .brand-hero .tagline { font-size: 0.9rem; }
  /* Buttons become full-width in stacked layouts */
  button { min-height: 36px !important; padding: 0.4rem 0.9rem !important; font-size: 0.85rem !important; }
  /* Tables get horizontal scroll treatment */
  .table-wrap { overflow-x: auto !important; -webkit-overflow-scrolling: touch; }
  table { font-size: 0.82rem; }
  th { padding: 0.4rem 0.6rem !important; }
  td { padding: 0.4rem 0.6rem !important; }
  /* Chatbot shrinks for more screen real estate */
  .chatbot { height: 380px !important; }
  /* Auth view: tighter padding */
  .auth-view { padding: 1.4rem 1.5rem 1.5rem !important; max-width: 400px !important; }
  .auth-hero h1 { font-size: 1.6rem; }
  /* KPI grid: 2 columns */
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .kpi { padding: 0.65rem 0.8rem; }
  .kpi-num { font-size: 1.35rem; }
  /* Popups go full-width */
  .email-settings-pop { width: calc(100% - 20px); left: 10px; right: 10px; }
  .profile-pop { width: calc(100% - 24px); left: 12px; right: 12px; }
  /* AI loader: tighter */
  .ai-loader { padding: 0.75rem 0.9rem; gap: 0.6rem; }
  .ai-loader .loader-text { font-size: 0.85rem; }
}

@media (max-width: 640px) {
  .brand-hero h1 { font-size: clamp(1.15rem, 5.5vw, 1.4rem) !important; }
  .brand-hero .tagline { font-size: 0.85rem; }
  .brand-hero .hero-badge { font-size: 0.6rem; padding: 0.18rem 0.45rem; }
  .nav-pills .wrap { gap: 0.25rem; padding: 0.3rem; }
  .nav-pills label {
    padding: 0.42rem 0.5rem;
    font-size: 0.72rem;
    border-radius: 8px;
  }
  /* Stack every field full-width; rows wrap; buttons keep natural width */
  .row, .form { flex-wrap: wrap !important; gap: 0.45rem !important; }
  .form { flex: 1 1 100% !important; min-width: 100% !important; }
  .tab-body { animation: fadeSlide 0.15s ease; }
  .markdown p, .markdown li { font-size: 0.82rem; }
  .prose p, .prose li { font-size: 0.82rem; }
  table { font-size: 0.78rem; }
  /* body cells wrap on small screens; headers keep their single-line look */
  td { white-space: normal !important; }
  .chatbot { height: 340px !important; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  /* Dropdowns & inputs smaller */
  input, textarea, select { font-size: 0.88rem !important; }
  /* Panel tighter borders */
  .panel { border-radius: 12px !important; padding: 0.5rem !important; }
  /* Badges scale down */
  .badge { font-size: 0.65rem; padding: 0.12rem 0.5rem; }
  /* Status notes compact */
  .status-note.prose:not(:empty) { padding: 0.55rem 0.7rem; font-size: 0.85rem; }
  /* Email warn banner */
  .email-warn.prose:not(:empty) { padding: 0.55rem 0.7rem; }
  /* Auth view mobile */
  .auth-view { padding: 1.2rem 1.2rem 1.4rem !important; margin: 3vh auto 0 auto !important; }
  .auth-hero h1 { font-size: 1.45rem; }
  .auth-hero .tagline { font-size: 0.82rem; }
  .auth-divider { margin: 0.8rem 0 0.7rem 0; font-size: 0.72rem; }
  .google-btn { font-size: 0.88rem; padding: 0.5rem 0.85rem; }
  /* Session bar stacks */
  .session-bar {
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.4rem;
  }
  .user-badge { text-align: center; font-size: 0.78rem; }
  /* Profile icon stays accessible */
  .profile-icon-btn {
    min-width: 2.2rem !important;
    max-width: 2.2rem !important;
    height: 2.2rem !important;
    font-size: 0.95rem !important;
  }
  /* Workspace footer */
  .workspace-foot { font-size: 0.74rem; padding: 1rem 0 0.3rem; }
  /* Screening report headings */
  .markdown h3, .prose h3 { font-size: 0.92rem !important; }
  .markdown h4, .prose h4 { font-size: 0.88rem !important; }
  /* AI loader compact */
  .ai-loader { padding: 0.65rem 0.75rem; gap: 0.5rem; border-radius: 10px; }
  .ai-loader .spinner { width: 22px; height: 22px; }
  .ai-loader .loader-text { font-size: 0.82rem; }
}

@media (max-width: 480px) {
  .nav-pills .wrap { gap: 0.2rem; padding: 0.25rem; }
  .nav-pills label { font-size: 0.68rem; padding: 0.4rem 0.35rem; }
  .panel { border-radius: 10px !important; padding: 0.4rem !important; }
  .markdown h3 { font-size: 0.88rem !important; }
  .workspace-foot { font-size: 0.72rem; }
  /* Brand hero ultra-compact */
  .brand-hero {
    padding: 0.8rem 0.7rem 0.7rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
  }
  .brand-hero h1 { font-size: clamp(1rem, 5vw, 1.2rem) !important; letter-spacing: -0.03em; }
  .brand-hero .tagline { font-size: 0.78rem; }
  .brand-hero .hero-badge { font-size: 0.55rem; padding: 0.15rem 0.4rem; margin-top: 0.5rem; }
  /* Buttons: full-width, compact */
  button {
    min-height: 34px !important;
    padding: 0.35rem 0.7rem !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
  }
  /* KPI: 2 columns, tighter */
  .kpi-grid { grid-template-columns: repeat(2, 1fr); gap: 0.5rem; }
  .kpi { padding: 0.55rem 0.7rem; border-radius: 10px; }
  .kpi-num { font-size: 1.2rem; }
  .kpi-label { font-size: 0.65rem; }
  /* Tables super-compact */
  table { font-size: 0.74rem; }
  th { padding: 0.35rem 0.5rem !important; font-size: 0.72rem; }
  td { padding: 0.35rem 0.5rem !important; }
  /* Chatbot shorter */
  .chatbot { height: 300px !important; }
  /* Auth view phone */
  .auth-view {
    max-width: calc(100% - 1rem) !important;
    margin: 2vh auto 0 auto !important;
    padding: 1rem 1rem 1.2rem !important;
    border-radius: 14px !important;
  }
  .auth-hero h1 { font-size: 1.3rem; }
  .auth-mode label { font-size: 0.78rem; padding: 0.35rem 0.4rem !important; }
  /* Inputs smaller */
  input, textarea, select { font-size: 0.84rem !important; }
  /* Email settings full screen on phone */
  .email-settings-pop {
    position: fixed;
    top: 10px; left: 10px; right: 10px;
    width: auto;
    max-height: calc(100vh - 20px);
    overflow-y: auto;
    border-radius: 12px;
    z-index: 100;
  }
  .profile-pop {
    position: fixed;
    top: 10px; left: 10px; right: 10px;
    width: auto;
    max-height: calc(100vh - 20px);
    overflow-y: auto;
    border-radius: 12px;
    z-index: 100;
  }
  /* Scrollbar thinner on mobile */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
}

@media (max-width: 360px) {
  /* Ultra-narrow phones (Galaxy S series, older iPhones) */
  .brand-hero { padding: 0.6rem; border-radius: 10px; }
  .brand-hero h1 { font-size: 1rem !important; }
  .brand-hero .tagline { font-size: 0.72rem; }
  .nav-pills label { font-size: 0.62rem; padding: 0.35rem 0.25rem; }
  .panel { padding: 0.3rem !important; border-radius: 8px !important; }
  button { font-size: 0.78rem !important; padding: 0.3rem 0.5rem !important; min-height: 32px !important; }
  .kpi-grid { grid-template-columns: 1fr 1fr; gap: 0.4rem; }
  .kpi-num { font-size: 1.05rem; }
  .kpi-label { font-size: 0.6rem; }
  .auth-view { padding: 0.8rem !important; }
  .auth-hero h1 { font-size: 1.15rem; }
  .chatbot { height: 260px !important; }
  table { font-size: 0.7rem; }
  .workspace-foot { font-size: 0.68rem; }
  .status-note.prose:not(:empty) { padding: 0.45rem 0.55rem; font-size: 0.8rem; }
  .ai-loader { padding: 0.5rem 0.6rem; }
  .ai-loader .spinner { width: 18px; height: 18px; }
  .ai-loader .loader-text { font-size: 0.78rem; }
  .badge { font-size: 0.6rem; padding: 0.1rem 0.4rem; }
}

/* ===== Login / account gate ===== */
.auth-view {
  max-width: 440px !important;
  margin: 6vh auto 0 auto !important;
  background: var(--surface) !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  padding: 1.8rem 2rem 2rem 2rem !important;
  box-shadow: 0 18px 50px -18px rgba(15, 23, 42, 0.25) !important;
}
.auth-hero { text-align: center; margin-bottom: 0.4rem; }
.auth-hero h1 {
  font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em;
  color: var(--ink); margin: 0;
}
.auth-hero .tagline {
  color: var(--muted); font-size: 0.9rem; margin: 0.25rem 0 0 0;
}
.auth-mode .wrap {
  justify-content: center; gap: 0.35rem;
  background: #f1f5f9; border-radius: 10px; padding: 0.3rem;
}
.auth-mode label {
  flex: 1; text-align: center; border-radius: 8px !important;
  border: 1px solid transparent !important; margin: 0 !important;
  font-weight: 600; font-size: 0.85rem; color: var(--muted) !important;
  padding: 0.45rem 0.5rem !important; background: transparent !important;
}
.auth-mode label:has(input:checked) {
  background: var(--brand) !important; color: #fff !important;
  box-shadow: 0 2px 8px -2px rgba(15, 118, 110, 0.5) !important;
}
.auth-view .label-wrap { margin-bottom: 0.25rem; }
.auth-view .label-wrap span { font-size: 0.8rem; font-weight: 600; }
.auth-view button.primary { width: 100%; margin-top: 0.4rem; }
.auth-divider {
  display: flex; align-items: center; gap: 0.75rem;
  margin: 1.1rem 0 0.9rem 0; color: var(--muted);
  font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
}
.auth-divider::before, .auth-divider::after {
  content: ""; flex: 1; height: 1px; background: var(--line);
}
.auth-view button.secondary {
  width: 100%; border-color: var(--line) !important;
  color: var(--ink-soft) !important; background: #fff !important;
}
.google-btn {
  display: flex; align-items: center; justify-content: center; gap: 0.6rem;
  width: 100%; padding: 0.55rem 1rem; border-radius: 10px;
  border: 1px solid var(--line); background: #fff; color: var(--ink-soft);
  font-weight: 600; font-size: 0.95rem; text-decoration: none;
  transition: box-shadow 0.15s ease, background 0.15s ease;
}
.google-btn:hover { background: #f8fafc; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12); }
.auth-view .status-note { margin-top: 0.75rem; }
.workspace-view { animation: fadeSlide 0.25s ease; position: relative; }
.session-bar {
  display: flex; align-items: center; justify-content: space-between;
  gap: 0.75rem; margin-bottom: 0.6rem;
}
.user-badge { margin: 0 !important; font-size: 0.85rem; color: var(--muted); }
.user-badge p { margin: 0; }
"""


def _theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#f0fdfa",
            c100="#ccfbf1",
            c200="#99f6e4",
            c300="#5eead4",
            c400="#2dd4bf",
            c500="#14b8a6",
            c600="#0d9488",
            c700="#0f766e",
            c800="#115e59",
            c900="#134e4a",
            c950="#042f2e",
        ),
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("DM Sans"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ).set(
        body_background_fill="#f8fafc",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_radius="12px",
        button_primary_background_fill="#0f766e",
        button_primary_background_fill_hover="#115e59",
        button_primary_text_color="#ffffff",
        body_text_color="#0f172a",
        body_text_color_subdued="#334155",
        block_title_text_color="#0f172a",
        block_label_text_color="#334155",
    )


# ================================================================
# Build UI
# ================================================================

def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="TalentIQ · AI Recruiter",
        theme=_theme(),
        css=custom_css,
    ) as demo:
        # The session token (persisted in the browser via localStorage). Every
        # workspace event passes it back as its FIRST input so each handler
        # runs inside the caller's private storage (auth.user_scope).
        session_token = gr.BrowserState(None)
        demo._session_token = session_token  # type: ignore[attr-defined]
        # reused by build_app's auth wiring
        interview_state = gr.State(InterviewSession().to_dict())

        gr.HTML(
            """
            <div class="brand-hero">
              <h1>TalentIQ</h1>
              <p class="tagline">RAG-powered screening · per-job shortlists · evidence-backed interviews</p>
              <span class="hero-badge">AI recruiting workspace</span>
            </div>
            """
        )

        # Always-visible navigation (replaces gr.Tabs "More tabs" overflow)
        nav = gr.Radio(
            choices=NAV_ITEMS,
            value="Jobs",
            label=None,
            interactive=True,
            elem_classes=["nav-pills"],
        )

        # ---------- Jobs ----------
        with gr.Column(visible=True, elem_classes=["tab-body"]) as jobs_tab:
            jobs_kpis = gr.Markdown(_stats_markdown())
            with gr.Row(elem_classes=["app-row"]):
                with gr.Column(scale=5, elem_classes=["panel"]):
                    gr.Markdown("#### Open a requisition")
                    job_title = gr.Textbox(label="Job title", placeholder="e.g. Senior LLM Engineer")
                    sample_job_dd = gr.Dropdown(
                        choices=list(SAMPLE_JOB_DESCRIPTIONS.keys()),
                        label="Load sample JD",
                        value=None,
                    )
                    job_desc = gr.Textbox(
                        label="Job description",
                        lines=12,
                        placeholder="Paste the full JD…",
                    )
                    with gr.Row():
                        job_req_id = gr.Textbox(
                            label="Req. ID (optional)",
                            placeholder="Auto-generated, e.g. REQ-1001",
                            scale=3,
                        )
                        create_job_btn = gr.Button("Create job", variant="primary", scale=2)
                with gr.Column(scale=4, elem_classes=["panel"]):
                    jobs_status = gr.Markdown(
                        "_Create a requisition to begin — sample JDs only fill the form._",
                        elem_classes=["status-note"],
                    )
                    focus_job = gr.Dropdown(
                        label="Focus job",
                        choices=_job_choices(),
                        value=(_job_choices()[0][1] if _job_choices() else None),
                    )
            # Open roles gets the FULL panel width — squeezed into the side
            # column it could never fit six columns (permanent horizontal
            # scroll + cramped headers).
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("#### Open roles")
                jobs_table = gr.Dataframe(
                    headers=["Select", "#", "Req. ID", "Title", "Candidates", "Shortlist", "Screened", "Created"],
                    datatype=["bool", "number", "str", "str", "number", "number", "number", "str"],
                    type="array",
                    interactive=True,
                    wrap=True,
                    column_widths=[110, 45, 95, "auto", 115, 105, 105, 155],
                    value=_jobs_table(),
                    label="Job listings — tick the Select boxes to delete (other cells are display-only)",
                )
                jobs_del_btn = gr.Button("Delete selected listings", variant="stop")
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("#### Candidate pipeline for focus job")
                job_cand_table = gr.Dataframe(
                    headers=["Select", "#", "Candidate", "Status", "Added"],
                    datatype=["bool", "number", "str", "str", "str"],
                    type="array",
                    interactive=True,
                    wrap=True,
                    column_widths=[110, 45, "auto", 100, 150],
                    value=_job_candidates_table(
                        _job_choices()[0][1] if _job_choices() else None
                    ),
                    label="Candidates for this job — tick checkboxes to manage",
                )
                with gr.Row():
                    jc_remove_btn = gr.Button("Remove selected from this job")
                    jc_delete_btn = gr.Button("Delete selected entirely", variant="stop")
                job_cand_status = gr.Markdown(
                    _job_candidates_status(_job_choices()[0][1] if _job_choices() else None)
                )

        # ---------- Candidates ----------
        with gr.Column(visible=False, elem_classes=["tab-body"]) as talent_tab:
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown(
                    "#### Target job for this intake\n"
                    "Select the job listing first — every candidate ingested below "
                    "is automatically linked to that job's pipeline and will appear "
                    "in its **Shortlist** ranking."
                )
                tp_job_dd = gr.Dropdown(
                    label="Job listing",
                    choices=_job_choices(),
                    value=(_job_choices()[0][1] if _job_choices() else None),
                )
            with gr.Row(elem_classes=["app-row"]):
                with gr.Column(scale=5, elem_classes=["panel"]):
                    gr.Markdown("#### Add candidates")
                    pdf_files = gr.File(
                        label="Batch PDF upload",
                        file_types=[".pdf"],
                        file_count="multiple",
                    )
                    upload_btn = gr.Button("Ingest PDFs", variant="primary")
                    cand_name = gr.Textbox(label="Name (optional override)")
                    resume_text = gr.Textbox(label="Resume text", lines=8)
                    add_text_btn = gr.Button("Ingest text")
                with gr.Column(scale=4, elem_classes=["panel"]):
                    gr.Markdown("#### Edit candidate")
                    manage_cand_dd = gr.Dropdown(
                        label="Select candidate (this job)",
                        choices=_candidate_choices(
                            _job_choices()[0][1] if _job_choices() else None
                        ),
                        value=(_candidate_choices(
                            _job_choices()[0][1] if _job_choices() else None
                        )[0][1] if _candidate_choices(
                            _job_choices()[0][1] if _job_choices() else None
                        ) else None),
                    )
                    edit_name = gr.Textbox(label="Display name")
                    edit_resume = gr.Textbox(label="Resume text", lines=8)
                    with gr.Row():
                        load_edit_btn = gr.Button("Load into editor")
                        save_cand_btn = gr.Button("Save changes", variant="primary")
                    cand_status = gr.Markdown(
                        "_Talent pool updates appear below._",
                        elem_classes=["status-note"],
                    )
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("#### Candidate list")
                cand_table = gr.Dataframe(
                    headers=["Select", "#", "Name", "Source", "Created"],
                    datatype=["bool", "number", "str", "str", "str"],
                    type="array",
                    value=_candidates_table(
                        _job_choices()[0][1] if _job_choices() else None
                    ),
                    interactive=True,
                    wrap=True,
                    column_widths=[110, 45, "auto", 90, 170],
                    label="Candidates for the selected job — tick the Select boxes to delete (other cells are display-only)",
                )
                cand_del_btn = gr.Button("Delete selected candidates", variant="stop")

        # ---------- Shortlist ----------
        with gr.Column(visible=False, elem_classes=["tab-body"]) as rank_tab:
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown(
                    "#### Rank candidates per job\n"
                    "Pick one or more jobs — each gets its **own** ranked shortlist "
                    "scored only against that JD."
                )
                with gr.Row():
                    multi_job_dd = gr.Dropdown(
                        label="Jobs to rank",
                        choices=_job_choices(),
                        value=[j[1] for j in _job_choices()[:3]],
                        multiselect=True,
                    )
                    shortlist_top_n = gr.Number(
                        label="Keep top N per job (blank = all)",
                        value=10,
                        precision=0,
                    )
                    rank_multi_btn = gr.Button("Rank selected jobs", variant="primary")
                multi_rank_md = gr.Markdown("_Run ranking to build per-job shortlists._")

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown("#### Focus job shortlist & deep screen")
                with gr.Row():
                    view_job_dd = gr.Dropdown(
                        label="Focus job",
                        choices=_job_choices(),
                        value=(_job_choices()[0][1] if _job_choices() else None),
                    )
                    screen_cand_dd = gr.Dropdown(
                        label="Candidate (shortlist order)",
                        choices=_shortlist_cand_choices(
                            _job_choices()[0][1] if _job_choices() else ""
                        ),
                        value=None,
                    )
                    screen_top_n = gr.Number(label="Deep-screen top N", value=3, precision=0)
                with gr.Row():
                    view_sl_btn = gr.Button("Load / refresh shortlist")
                    screen_one_btn = gr.Button("Deep-screen selected", variant="primary")
                    screen_top_btn = gr.Button("Deep-screen top N")
                screen_loader = gr.Markdown(
                    value="",
                    visible=False,
                    elem_id="screen_loader",
                )
                with gr.Row(elem_classes=["interview-row"]):
                    interview_n = gr.Number(
                        label="Top N for interview",
                        value=DEFAULT_INTERVIEW_N,
                        precision=0,
                        minimum=1,
                        maximum=20,
                    )
                    sync_int_btn = gr.Button("Sync top N to interview", variant="primary", scale=1)
                gr.Markdown(
                    "_Top N for interview = the **top N ranked shortlist candidates** for the "
                    "focus job. The Interview tab list updates live when you change N or the "
                    "focus job. Candidates without a PASS screening are auto-screened on "
                    "interview start._"
                )
                focus_rank_md = gr.Markdown()
                rank_table = gr.Dataframe(
                    headers=["#", "Name", "Hybrid", "Semantic", "Keyword"],
                    interactive=False,
                    column_widths=[40, "auto", 90, 110, 110],
                    label="Focus shortlist",
                )
                screen_md = gr.Markdown("_Rubric screening report appears here._")

        # ---------- Interview ----------
        with gr.Column(visible=False, elem_classes=["tab-body"]) as int_tab:
            with gr.Column(elem_classes=["panel"]):
                with gr.Row():
                    int_job_dd = gr.Dropdown(
                        label="Job",
                        choices=_job_choices(),
                        value=(_job_choices()[0][1] if _job_choices() else None),
                    )
                    int_cand_dd = gr.Dropdown(
                        label="Top N shortlist → interview",
                        choices=_interview_choices(
                            _job_choices()[0][1] if _job_choices() else ""
                        ),
                        value=None,
                    )
                    int_lang = gr.Dropdown(
                        choices=["English", "German"],
                        value="English",
                        label="Interview language",
                        scale=1,
                    )
                    int_mode = gr.Radio(
                        choices=["Chat interview", "Live meeting interview"],
                        value="Chat interview",
                        label="Interview mode",
                        scale=2,
                    )
                gr.Markdown(
                    "_The AI **suggests 10 tailored questions** when a chat interview "
                    "starts — questions, follow-ups and evaluations are written in the "
                    "selected language. **Live meeting** mode creates a **free Jitsi "
                    "meeting link** (no accounts or paid plans — the link opens the "
                    "call directly) and streams the call's transcript from your "
                    "browser microphone: "
                    "Whisper transcribes every ~10s, the AI separates interviewer and "
                    "candidate, and evaluates the answers. After **Stop** you can "
                    "**review & fix** the transcript before the evaluation runs. "
                )

                # --- Chat mode ---
                with gr.Column(visible=True) as chat_panel:
                    start_int_btn = gr.Button(
                        "Start chat interview (AI suggests 10 questions)",
                        variant="primary",
                    )
                    int_questions_md = gr.Markdown(
                        "_Suggested questions appear here when the interview starts._"
                    )
                    chatbot = gr.Chatbot(label="Interview transcript", height=380)
                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="Candidate answer",
                            placeholder="Type the answer and press Send",
                            scale=5,
                        )
                        chat_send = gr.Button("Send", variant="primary", scale=1)
                    int_status = gr.Markdown(elem_classes=["status-note"])
                    int_eval = gr.Markdown()

                # --- Live meeting mode (free Jitsi link + browser-mic live transcript) ---
                with gr.Column(visible=False) as live_panel:
                    live_gen_btn = gr.Button(
                        "Generate free Jitsi meeting link",
                        variant="secondary",
                    )
                    live_link = gr.Textbox(
                        label="Interview invite link (paste into Email → Interview invite)",
                        placeholder="Generated here — a Jitsi room that opens the meeting directly",
                        interactive=True,
                    )
                    live_link_hint = gr.Markdown(
                        "_Click **Generate free Jitsi meeting link** — a **Jitsi** room is "
                        "created instantly (free, no account, no time limit) and the link "
                        "opens the meeting directly. Send it via **Email → Interview "
                        "invite**._"
                    )
                    live_questions_btn = gr.Button(
                        "Show AI-suggested questions for this candidate",
                        variant="secondary",
                    )
                    live_questions_md = gr.Markdown(
                        "_The AI can suggest the 10 interview questions to ask before "
                        "the call — click above._"
                    )
                    with gr.Row():
                        live_start_btn = gr.Button(
                            "▶ Start live transcription (browser mic)",
                            variant="primary",
                            scale=1,
                        )
                        live_stop_btn = gr.Button(
                            "⏹ Stop & transcribe remainder",
                            variant="secondary",
                            scale=1,
                            interactive=False,
                        )
                        live_finish_btn = gr.Button(
                            "✅ Finish & evaluate (AI scores the answers)",
                            variant="primary",
                            scale=1,
                            interactive=False,
                        )
                    live_mic = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Microphone — click record during the call",
                    )
                    live_status = gr.Markdown(
                        "_1. Generate a free **Jitsi** meeting link. 2. Click "
                        "**Start live transcription**, then press **record** on the "
                        "microphone. 3. **Stop** to review & fix the transcript, then "
                        "**Finish & evaluate** when the call ends._",
                        elem_classes=["status-note"],
                    )
                    with gr.Accordion("Live transcript", open=True):
                        live_transcript = gr.Markdown(
                            "_Transcript streams here as it's transcribed._"
                        )
                    live_transcript_edit = gr.Textbox(
                        label="Review & fix the transcript before evaluating",
                        placeholder=(
                            "After **Stop**, the full transcript appears here — "
                            "correct any words Whisper misheard, then click "
                            "**Finish & evaluate** to score the corrected answers."
                        ),
                        visible=False,
                        lines=12,
                    )
                    live_result = gr.Markdown(
                        "_Speaker-separated transcript + Q&A + evaluation appear "
                        "here after **Finish & evaluate**._"
                    )
                    vi_history = gr.Dataframe(
                        headers=["When", "Candidate", "Q&A", "Avg", "Verdict"],
                        value=_video_history_table(
                            _job_choices()[0][1] if _job_choices() else None
                        ),
                        interactive=False,
                        wrap=True,
                        column_widths=[120, "auto", 60, 70, 90],
                        label="Live interview history (this job)",
                    )
                    live_state = gr.State(value="")
                    live_timer = gr.Timer(value=2)

        # ---------- History & export ----------
        with gr.Column(visible=False, elem_classes=["tab-body"]) as hist_tab:
            with gr.Column(elem_classes=["panel"]):
                with gr.Row():
                    hist_job_dd = gr.Dropdown(
                        label="Filter by job",
                        choices=_job_choices(),
                        value=(_job_choices()[0][1] if _job_choices() else None),
                    )
                    refresh_hist_btn = gr.Button("Refresh")
                history_md = gr.Markdown(
                    _history_markdown(_job_choices()[0][1] if _job_choices() else None)
                )
                gr.Markdown(
                    "#### Export per-job CSV report\n"
                    "Generates a **CSV per job title** covering every candidate in the "
                    "pipeline: rank, hybrid scores, deep-screen verdict, and interview "
                    "outcome — Excel-ready."
                )
                with gr.Row():
                    export_job_dd = gr.Dropdown(
                        label="Job",
                        choices=_job_choices(),
                        value=(_job_choices()[0][1] if _job_choices() else None),
                    )
                    export_csv_btn = gr.Button("Generate CSV report", variant="primary")
                    csv_download = gr.DownloadButton(
                        "Download CSV",
                        value=None,
                        interactive=True,
                    )
                csv_status = gr.Markdown(elem_classes=["status-note"])

        # ---------- Email ----------
        with gr.Column(visible=False, elem_classes=["tab-body"]) as email_tab:
            em_banner = gr.Markdown(
                value="",
                visible=False,
                elem_classes=["email-warn"],
            )
            with gr.Column(elem_classes=["panel"]):
                with gr.Row():
                    gr.Markdown("#### Shortlist notifications & interview invites")
                    es_open_btn = gr.Button(
                        "⚙️ Email settings", variant="secondary", scale=0
                    )
                gr.Markdown(
                    "Emails are sent over **your own SMTP server** (free — e.g. a "
                    "Gmail app password), configured **per account** via "
                    "**⚙️ Email settings**. If SMTP isn't configured the app keeps "
                    "working and shows a clear message instead."
                )
                with gr.Row():
                    em_job_dd = gr.Dropdown(
                        label="Job",
                        choices=_job_choices(),
                        value=(_job_choices()[0][1] if _job_choices() else None),
                    )
                    em_cand_dd = gr.Dropdown(
                        label="Candidate",
                        choices=_candidate_choices(
                            _job_choices()[0][1] if _job_choices() else None
                        ),
                        value=None,
                    )
                email_kind = gr.Radio(
                    choices=["Shortlist notification", "Interview invite"],
                    value="Shortlist notification",
                    label="Email type",
                )
                em_tmpl_dd = gr.Dropdown(
                    label="Email template",
                    choices=_em_template_choices("shortlist"),
                    value="",
                    info=(
                        "Pre-selected to your preferred template for this email "
                        "type — or the built-in design"
                    ),
                )
                with gr.Row():
                    em_to = gr.Textbox(
                        label="Recipient email",
                        placeholder="Auto-filled from the resume — editable",
                        scale=4,
                    )
                    em_send_btn = gr.Button("Send email", variant="primary", scale=1)
                em_msg = gr.Textbox(
                    label="Optional personal message",
                    lines=2,
                    placeholder="e.g. Interview slot suggestion, next-step details…",
                )
                em_link = gr.Textbox(
                    label="Interview invite link (optional)",
                    lines=1,
                    placeholder="e.g. https://meet.jit.si/talentiq-… — included as a button in interview invites",
                )
                em_status = gr.Markdown(
                    "_Select a job + candidate — the recipient is auto-filled from "
                    "the resume._",
                    elem_classes=["status-note"],
                )
                em_history = gr.Dataframe(
                    headers=["When", "Recipient", "Subject"],
                    value=_email_history_table(),
                    interactive=False,
                    wrap=True,
                    column_widths=[120, "auto", "auto"],
                    label="Recently sent emails",
                )

            # ---- Email templates (create / edit / preferred) ----
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown(
                    "#### 📝 Email templates\n"
                    "Create reusable templates **per email type** (shortlist "
                    "notification / interview invite), edit them any time, and "
                    "mark one as your **preferred** template — it is pre-selected "
                    "when composing that email type."
                )
                gr.Markdown(
                    "Placeholders are filled per candidate: `{{name}}`, "
                    "`{{job_title}}`, `{{req_id}}`, `{{message}}` and (for "
                    "invites) `{{invite_link}}`. Blank lines become paragraph "
                    "breaks; the built-in design is used until you save a "
                    "template."
                )
                with gr.Row():
                    tmpl_dd = gr.Dropdown(
                        label="Template",
                        choices=_template_choices("shortlist"),
                        value=None,
                        info="Pick a template to edit — or start a new one",
                        scale=4,
                    )
                    tmpl_new_btn = gr.Button(
                        "✨ New template", variant="secondary", scale=1
                    )
                tmpl_name = gr.Textbox(
                    label="Template name",
                    placeholder="e.g. Standard shortlist",
                )
                tmpl_subject = gr.Textbox(
                    label="Subject line",
                    placeholder="Shortlisted: {{job_title}}",
                )
                tmpl_body = gr.Textbox(
                    label="Email body",
                    lines=8,
                    placeholder=(
                        "Hi {{name}},\n\n"
                        "Great news — your application for {{job_title}} "
                        "({{req_id}}) has been shortlisted.\n\n{{message}}\n\n"
                        "Best regards,\nThe TalentIQ Recruiting Team"
                    ),
                )
                with gr.Row():
                    tmpl_save_btn = gr.Button("💾 Save template", variant="primary")
                    tmpl_default_btn = gr.Button(
                        "⭐ Set as preferred", variant="secondary"
                    )
                    tmpl_delete_btn = gr.Button(
                        "🗑 Delete template", variant="stop", scale=0
                    )
                tmpl_status = gr.Markdown(
                    "_No templates yet — create one above. Emails use the built-in "
                    "design until you save a template._",
                    elem_classes=["status-note"],
                )

            # Floating "Email settings" bubble — hidden until the ⚙️ button
            # opens it (gr.Popover was removed in Gradio 6; an absolutely
            # positioned panel + a visibility toggle reproduces the bubble UX).
            with gr.Column(visible=False, elem_classes=["email-settings-pop"]) as es_pop:
                with gr.Row():
                    gr.Markdown(
                        "#### Email settings (per account)\n"
                        "Configure **your own SMTP sender** here — each account "
                        "keeps its own (free) SMTP, e.g. a Gmail app password. "
                        "The app does **not** read SMTP from `.env`; until you "
                        "save, sends stay disabled."
                    )
                    es_close_btn = gr.Button("✕ Close", variant="secondary", scale=0)
                with gr.Row():
                    es_host = gr.Dropdown(
                        label="SMTP host",
                        choices=_SMTP_HOST_CHOICES,
                        value=None,
                        allow_custom_value=True,
                        info="Pick a provider, or type any host",
                        scale=2,
                    )
                    es_port = gr.Number(
                        label="SMTP port", value=587, precision=0, scale=1
                    )
                with gr.Row():
                    es_from = gr.Textbox(
                        label="From address", placeholder="you@gmail.com"
                    )
                    es_from_name = gr.Textbox(
                        label="From name", placeholder="TalentIQ Recruiter"
                    )
                with gr.Row():
                    es_user = gr.Textbox(label="SMTP username")
                    es_pass = gr.Textbox(
                        label="SMTP password",
                        type="password",
                        placeholder="Optional — some servers don't require one",
                    )
                with gr.Row():
                    es_starttls = gr.Checkbox(label="Use STARTTLS", value=True)
                gr.Markdown(
                    "**Email branding** — shown at the top of every email this "
                    "account sends."
                )
                es_company = gr.Textbox(
                    label="Company name",
                    placeholder="e.g. Acme Corp",
                )
                es_logo = gr.Image(
                    label="Company logo (optional)",
                    type="filepath",
                    height=96,
                    interactive=True,
                )
                with gr.Row():
                    es_save_btn = gr.Button("Save settings", variant="primary")
                    es_test_btn = gr.Button("Send test email", variant="secondary")
                    es_clear_btn = gr.Button("Clear settings", variant="secondary")
                es_test_to = gr.Textbox(
                    label="Test recipient",
                    placeholder="Defaults to your account email",
                )
                es_status = gr.Markdown(
                    "_Your email config will show here after you save._",
                    elem_classes=["status-note"],
                )

        # ---------- Profile (floating popover) ----------
        with gr.Column(visible=False, elem_classes=["profile-pop"]) as profile_pop:
            with gr.Row():
                gr.Markdown(
                    "#### 👤 Account\n"
                    "_Your profile, session and account settings._"
                )
                pf_close_btn = gr.Button("✕ Close", variant="secondary", scale=0)
            with gr.Row(elem_classes=["app-row"]):
                with gr.Column(scale=2, elem_classes=["panel"]):
                    gr.Markdown("#### Profile")
                    pf_name = gr.Textbox(
                        label="Full name",
                        value="",
                        placeholder="Your name",
                    )
                    pf_save_btn = gr.Button("Save name", variant="primary")
                    pf_status = gr.Markdown("", elem_classes=["status-note"])
                with gr.Column(scale=3, elem_classes=["panel"]):
                    pf_email = gr.Markdown("")
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown(
                    "#### Danger zone\n"
                    "_Permanently deletes this account and **all of its data** — "
                    "jobs, candidates, interviews, email settings and exports. "
                    "Type **DELETE** to confirm._"
                )
                with gr.Row():
                    pf_del_confirm = gr.Textbox(
                        label="Type DELETE to confirm",
                        placeholder="DELETE",
                    )
                    pf_del_btn = gr.Button(
                        "Delete my account permanently",
                        variant="stop",
                        scale=1,
                    )
            with gr.Row():
                pf_logout_btn = gr.Button("Log out", variant="secondary", scale=0)

        # Expose the Profile components to build_app's auth wiring (delete /
        # logout need the auth-gate outputs) and to the refresh sweep.
        demo._pf_outputs = [pf_name, pf_email, pf_status, pf_del_confirm]  # type: ignore[attr-defined]
        demo._pf_name = pf_name  # type: ignore[attr-defined]
        demo._pf_email = pf_email  # type: ignore[attr-defined]
        demo._pf_status = pf_status  # type: ignore[attr-defined]
        demo._pf_save_btn = pf_save_btn  # type: ignore[attr-defined]
        demo._pf_del_btn = pf_del_btn  # type: ignore[attr-defined]
        demo._pf_del_confirm = pf_del_confirm  # type: ignore[attr-defined]
        demo._pf_logout_btn = pf_logout_btn  # type: ignore[attr-defined]
        demo._pf_pop = profile_pop  # type: ignore[attr-defined]
        demo._pf_close_btn = pf_close_btn  # type: ignore[attr-defined]

        # Shared click-away backdrop for the floating bubbles (Profile and
        # Email settings): a transparent full-workspace overlay that sits
        # below the bubbles (z-index 55 < 60) so any click outside them lands
        # on it and closes them — in addition to the ✕ buttons.
        bubble_backdrop = gr.HTML("", elem_classes=["bubble-backdrop"], visible=False)
        demo._bubble_backdrop = bubble_backdrop  # type: ignore[attr-defined]

        gr.HTML(
            "<div class='workspace-foot'>"
            "TalentIQ · RAG recruiting workspace — data stays local "
            "Ankush Karmakar — 2026"
            "</div>"
        )

        # ---- Navigation wiring ----
        nav.change(
            fn=_on_nav,
            inputs=[session_token, nav],
            outputs=[
                jobs_tab, talent_tab, rank_tab, email_tab,
                int_tab, hist_tab, profile_pop, es_pop, bubble_backdrop,
            ],
        )

        # ---- Shared refresh targets ----
        _ws_outputs = [
            focus_job,
            multi_job_dd,
            view_job_dd,
            int_job_dd,
            hist_job_dd,
            export_job_dd,
            manage_cand_dd,
            screen_cand_dd,
            int_cand_dd,
            jobs_table,
            cand_table,
            history_md,
            job_cand_table,
            job_cand_status,
            tp_job_dd,
            jobs_kpis,
            em_job_dd,
        ]

        # Outputs refreshed by the per-job pipeline action buttons — the whole
        # workspace (incl. the Email candidate dropdown + video history).
        _JC_ACTION_OUTPUTS = [*_ws_outputs, em_cand_dd, vi_history]

        sample_job_dd.change(
            fn=on_load_sample_job,
            inputs=[session_token, sample_job_dd],
            outputs=[job_desc, job_title],
        )

        create_job_btn.click(
            fn=on_create_job,
            inputs=[session_token, job_title, job_desc, sample_job_dd, job_req_id],
            outputs=[jobs_status, job_title, job_desc, sample_job_dd, job_req_id, *_ws_outputs],
        )
        jobs_del_btn.click(
            fn=on_delete_jobs,
            inputs=[session_token, jobs_table],
            outputs=[jobs_status, *_ws_outputs],
        )

        _cand_outputs = [
            cand_status,
            cand_table,
            manage_cand_dd,
            screen_cand_dd,
            int_cand_dd,
            edit_name,
            edit_resume,
            job_cand_table,
            job_cand_status,
            tp_job_dd,
            jobs_table,
            jobs_kpis,
            history_md,
            em_job_dd,
            em_cand_dd,
            vi_history,
        ]

        add_text_btn.click(
            fn=on_add_resume_text,
            inputs=[session_token, resume_text, cand_name, tp_job_dd],
            outputs=_cand_outputs,
        )
        upload_btn.click(
            fn=on_upload_pdfs,
            inputs=[session_token, pdf_files, cand_name, tp_job_dd],
            outputs=_cand_outputs,
        )
        load_edit_btn.click(
            fn=on_load_candidate_for_edit,
            inputs=[session_token, manage_cand_dd],
            outputs=[edit_name, edit_resume],
        )
        manage_cand_dd.change(
            fn=on_load_candidate_for_edit,
            inputs=[session_token, manage_cand_dd],
            outputs=[edit_name, edit_resume],
        )
        save_cand_btn.click(
            fn=on_save_candidate,
            inputs=[session_token, manage_cand_dd, edit_name, edit_resume, tp_job_dd],
            outputs=_cand_outputs,
        )
        cand_del_btn.click(
            fn=on_delete_candidates,
            inputs=[session_token, cand_table, tp_job_dd],
            outputs=_cand_outputs,
        )
        tp_job_dd.change(
            fn=_on_tp_job_change,
            inputs=[session_token, tp_job_dd],
            outputs=[manage_cand_dd, cand_table, edit_name, edit_resume],
        )

        # ---- Per-job pipeline actions (select + remove / delete) ----
        jc_remove_btn.click(
            fn=lambda tk, j, t: on_job_candidate_action(tk, j, t, "remove"),
            inputs=[session_token, focus_job, job_cand_table],
            outputs=[job_cand_status, *_JC_ACTION_OUTPUTS],
        )
        jc_delete_btn.click(
            fn=lambda tk, j, t: on_job_candidate_action(tk, j, t, "delete"),
            inputs=[session_token, focus_job, job_cand_table],
            outputs=[job_cand_status, *_JC_ACTION_OUTPUTS],
        )

        rank_multi_btn.click(
            fn=on_rank_multi,
            inputs=[session_token, multi_job_dd, shortlist_top_n],
            outputs=[
                multi_rank_md, focus_rank_md, rank_table, screen_cand_dd,
                job_cand_table, job_cand_status, int_cand_dd, int_job_dd,
            ],
        )
        view_sl_btn.click(
            fn=on_view_job_shortlist,
            inputs=[session_token, view_job_dd],
            outputs=[
                focus_rank_md, rank_table, screen_cand_dd, job_cand_table,
                job_cand_status, int_cand_dd, int_job_dd,
            ],
        )
        view_job_dd.change(
            fn=on_view_job_shortlist,
            inputs=[session_token, view_job_dd],
            outputs=[
                focus_rank_md, rank_table, screen_cand_dd, job_cand_table,
                job_cand_status, int_cand_dd, int_job_dd,
            ],
        )
        # Deep-screen actions run through a visible 'AI is evaluating' loader
        # section: show it instantly, run the (slow) screening, then hide it.
        screen_one_btn.click(
            fn=lambda tk: _show_screen_loader(
                tk,
                "**AI is deep-screening the candidate** — retrieving resume "
                "evidence from the vector store and scoring the rubric "
                "dimensions…",
            ),
            inputs=[session_token],
            outputs=[screen_loader],
        ).then(
            fn=on_deep_screen,
            inputs=[session_token, view_job_dd, screen_cand_dd, interview_n],
            outputs=[screen_md, int_cand_dd, history_md, int_job_dd],
        ).then(
            fn=_hide_screen_loader,
            inputs=[session_token],
            outputs=[screen_loader],
        )
        screen_top_btn.click(
            fn=lambda tk: _show_screen_loader(
                tk,
                "**AI is deep-screening the top N candidates** — this runs "
                "one LLM evaluation per candidate and can take a minute or "
                "two on the free tier…",
            ),
            inputs=[session_token],
            outputs=[screen_loader],
        ).then(
            fn=on_deep_screen_top_n,
            inputs=[session_token, view_job_dd, screen_top_n, interview_n],
            outputs=[screen_md, int_cand_dd, history_md, int_job_dd],
        ).then(
            fn=_hide_screen_loader,
            inputs=[session_token],
            outputs=[screen_loader],
        )

        # Sync top N to interview (manual button + automatic on change)
        sync_int_btn.click(
            fn=_sync_int_candidates,
            inputs=[session_token, view_job_dd, interview_n, focus_job],
            outputs=[int_cand_dd, int_job_dd],
        )
        interview_n.change(
            fn=_on_int_n_change,
            inputs=[session_token, interview_n, view_job_dd],
            outputs=[int_cand_dd],
        )
        view_job_dd.change(
            fn=_on_view_job_change,
            inputs=[session_token, view_job_dd],
            outputs=[int_cand_dd, int_questions_md],
        )

        start_int_btn.click(
            fn=on_start_interview,
            inputs=[session_token, int_job_dd, int_cand_dd, int_lang],
            outputs=[interview_state, chatbot, int_status, int_eval, int_questions_md],
        )
        chat_send.click(
            fn=on_chat_submit,
            inputs=[session_token, chat_input, chatbot, interview_state],
            outputs=[interview_state, chatbot, chat_input, int_eval],
        )
        chat_input.submit(
            fn=on_chat_submit,
            inputs=[session_token, chat_input, chatbot, interview_state],
            outputs=[interview_state, chatbot, chat_input, int_eval],
        )

        refresh_hist_btn.click(
            fn=on_refresh_history,
            inputs=[session_token, hist_job_dd],
            outputs=[history_md],
        )
        export_csv_btn.click(
            fn=on_export_csv,
            inputs=[session_token, export_job_dd],
            outputs=[csv_status, csv_download],
        )

        # ---- Email tab wiring ----
        em_cand_dd.change(
            fn=on_email_candidate_change,
            inputs=[session_token, em_cand_dd],
            outputs=[em_to],
        )
        em_job_dd.change(
            fn=_on_em_job_change,
            inputs=[session_token, em_job_dd],
            outputs=[em_cand_dd, em_to],
        )
        em_send_btn.click(
            fn=on_send_email,
            inputs=[
                session_token, em_job_dd, em_cand_dd, email_kind, em_tmpl_dd,
                em_to, em_msg, em_link,
            ],
            outputs=[em_status, em_history],
        )

        # ---- Email template manager wiring ----
        # Switching the email type points both template dropdowns (compose +
        # manager) at that type's preferred template.
        email_kind.change(
            fn=_on_email_kind_change,
            inputs=[session_token, email_kind],
            outputs=[em_tmpl_dd, tmpl_dd],
        )
        # Picking a template loads it into the editor (the compose dropdown
        # and the manager dropdown share the editor).
        em_tmpl_dd.change(
            fn=_on_tmpl_select,
            inputs=[session_token, em_tmpl_dd],
            outputs=[tmpl_name, tmpl_subject, tmpl_body],
        )
        tmpl_dd.change(
            fn=_on_tmpl_select,
            inputs=[session_token, tmpl_dd],
            outputs=[tmpl_name, tmpl_subject, tmpl_body],
        )
        tmpl_new_btn.click(
            fn=_on_tmpl_new,
            inputs=[session_token, email_kind],
            outputs=[em_tmpl_dd, tmpl_dd, tmpl_name, tmpl_subject, tmpl_body, tmpl_status],
        )
        tmpl_save_btn.click(
            fn=_on_tmpl_save,
            inputs=[
                session_token, email_kind, tmpl_dd, tmpl_name, tmpl_subject,
                tmpl_body,
            ],
            outputs=[em_tmpl_dd, tmpl_dd, tmpl_name, tmpl_subject, tmpl_body, tmpl_status],
        )
        tmpl_default_btn.click(
            fn=_on_tmpl_set_default,
            inputs=[session_token, email_kind, tmpl_dd],
            outputs=[em_tmpl_dd, tmpl_dd, tmpl_name, tmpl_subject, tmpl_body, tmpl_status],
        )
        tmpl_delete_btn.click(
            fn=_on_tmpl_delete,
            inputs=[session_token, email_kind, tmpl_dd],
            outputs=[em_tmpl_dd, tmpl_dd, tmpl_name, tmpl_subject, tmpl_body, tmpl_status],
        )

        # ---- Email settings (per account) wiring ----
        # Order MUST match _email_settings_refresh()'s return: 9 form values
        # (host, port, from, from_name, user, password, starttls, company name,
        # logo), then the status line, then the warning banner — the same
        # order _auth_outputs uses, so Save/Clear/Test never scramble fields.
        _es_outputs = [
            es_host, es_port, es_from, es_from_name,
            es_user, es_pass, es_starttls, es_company, es_logo,
            es_status, em_banner,
        ]
        # Selecting a known provider from the SMTP host dropdown auto-fills
        # its default port (587 for the common free providers) and mirrors the
        # from-address into the username for email-login providers (Gmail et
        # al.). Typing a from-address afterwards fills the username the same
        # way (only when the username is still empty).
        es_host.change(
            fn=_on_es_host_change,
            inputs=[session_token, es_host, es_from, es_user, es_port],
            outputs=[es_port, es_user],
        )
        es_from.change(
            fn=_on_es_from_change,
            inputs=[session_token, es_host, es_from, es_user],
            outputs=[es_user],
        )
        es_save_btn.click(
            fn=on_save_email_settings,
            inputs=[
                session_token, es_host, es_port, es_from, es_from_name,
                es_user, es_pass, es_starttls, es_company, es_logo,
            ],
            outputs=_es_outputs,
        )
        es_clear_btn.click(
            fn=on_clear_email_settings,
            inputs=[session_token],
            outputs=_es_outputs,
        )
        es_test_btn.click(
            fn=on_test_email_settings,
            inputs=[
                session_token, es_host, es_port, es_from, es_from_name,
                es_user, es_pass, es_starttls, es_company, es_logo, es_test_to,
            ],
            outputs=_es_outputs,
        )
        # The ⚙️ button opens the floating settings bubble; ✕ Close hides it,
        # and clicking anywhere outside (the backdrop) closes it too.
        es_open_btn.click(
            fn=_open_email_settings,
            inputs=[session_token],
            outputs=[es_pop, bubble_backdrop],
        )
        es_close_btn.click(
            fn=_close_email_settings,
            inputs=[session_token],
            outputs=[es_pop, bubble_backdrop],
        )
        bubble_backdrop.click(
            fn=_close_bubbles,
            inputs=[session_token],
            outputs=[bubble_backdrop, profile_pop, es_pop],
        )

        # ---- Interview tab: chat vs live-meeting mode ----
        int_mode.change(
            fn=_on_int_mode_change,
            inputs=[session_token, int_mode],
            outputs=[chat_panel, live_panel],
        )
        live_gen_btn.click(
            fn=on_generate_meeting_link,
            inputs=[session_token, int_job_dd, int_cand_dd],
            outputs=[live_link, live_link_hint],
        )
        live_questions_btn.click(
            fn=on_suggest_questions,
            inputs=[session_token, int_job_dd, int_cand_dd, int_lang],
            outputs=[live_questions_md],
        )
        live_start_btn.click(
            fn=on_live_start,
            inputs=[
                session_token, int_job_dd, int_cand_dd, live_link, int_lang,
            ],
            outputs=[
                live_state, live_start_btn, live_stop_btn, live_finish_btn,
                live_status, live_transcript_edit,
            ],
        )
        live_stop_btn.click(
            fn=on_live_stop,
            inputs=[session_token, live_state],
            outputs=[live_status, live_transcript, live_transcript_edit],
        )
        live_finish_btn.click(
            fn=on_live_finish,
            inputs=[session_token, live_state, int_lang, live_transcript_edit],
            outputs=[
                live_state, live_start_btn, live_stop_btn, live_finish_btn,
                live_status, live_transcript, live_result, vi_history,
                live_transcript_edit,
            ],
        )
        live_mic.stream(
            fn=on_live_chunk,
            inputs=[session_token, live_state, live_mic],
            outputs=[live_status, live_transcript],
        )
        live_timer.tick(
            fn=_on_live_tick,
            inputs=[session_token, live_state],
            outputs=[live_status, live_transcript],
        )

        @_require_session
        def _sync_focus(token, job_id):
            qualified = _interview_choices(job_id or "")
            cands = _candidate_choices(job_id or None)
            return (
                gr.update(value=job_id),          # view_job_dd
                gr.update(value=job_id),          # int_job_dd
                gr.update(value=job_id),          # hist_job_dd
                gr.update(value=job_id),          # export_job_dd
                gr.update(choices=qualified, value=qualified[0][1] if qualified else None),  # int_cand_dd
                _history_markdown(job_id),        # history_md
                _job_candidates_table(job_id or None),   # job_cand_table
                _job_candidates_status(job_id or None),  # job_cand_status
                gr.update(value=job_id),          # tp_job_dd
                _stats_markdown(),                # jobs_kpis
                gr.update(choices=cands, value=cands[0][1] if cands else None),  # manage_cand_dd
                _candidates_table(job_id or None),       # cand_table
                gr.update(value=job_id),          # em_job_dd
            )

        focus_job.change(
            fn=_sync_focus,
            inputs=[session_token, focus_job],
            outputs=[
                view_job_dd,
                int_job_dd,
                hist_job_dd,
                export_job_dd,
                int_cand_dd,
                history_md,
                job_cand_table,
                job_cand_status,
                tp_job_dd,
                jobs_kpis,
                manage_cand_dd,
                cand_table,
                em_job_dd,
            ],
        )

        @_require_session
        def _sync_int_job(token, job_id, interview_n):
            try:
                int_n = int(interview_n or DEFAULT_INTERVIEW_N)
            except (TypeError, ValueError):
                int_n = DEFAULT_INTERVIEW_N
            qualified = _interview_choices(job_id or "", int_n)
            return (
                gr.update(
                    choices=qualified,
                    value=qualified[0][1] if qualified else None,
                ),
                "_Suggested questions appear here when the interview starts._",
            )

        int_job_dd.change(
            fn=_sync_int_job,
            inputs=[session_token, int_job_dd, interview_n],
            outputs=[int_cand_dd, int_questions_md],
        )

        # ---- Auth hooks (used by the login gate in build_app) ----
        # Everything that shows stored data — refreshed wholesale after a
        # user signs in so they only ever see THEIR jobs/candidates/history.
        demo._auth_outputs = [  # type: ignore[attr-defined]
            *_ws_outputs,
            es_host,
            es_port,
            es_from,
            es_from_name,
            es_user,
            es_pass,
            es_starttls,
            es_company,
            es_logo,
            es_status,
            em_banner,
            em_tmpl_dd,
            tmpl_dd,
            tmpl_name,
            tmpl_subject,
            tmpl_body,
            tmpl_status,
            em_history,
            vi_history,
            *demo._pf_outputs,  # type: ignore[attr-defined]
        ]
        assert len(demo._auth_outputs) == _AUTH_REFRESH_OUTPUTS, (  # type: ignore[attr-defined]
            "_AUTH_REFRESH_OUTPUTS must equal len(_ws_outputs) + 11 "
            "(email settings + branding + banner) + 6 (email templates) + 2 "
            "(email history + video history) + 4 (profile)"
        )
        # The Email settings button outputs must match the auth-refresh
        # segment for those same 9 components — otherwise Save/Clear/Test
        # scramble the form fields (values arrive in a different order).
        _es_segment = demo._auth_outputs[  # type: ignore[attr-defined]
            len(_ws_outputs): len(_ws_outputs) + len(_es_outputs)
        ]
        assert _es_segment == _es_outputs, (  # type: ignore[attr-defined]
            "_es_outputs must match the _auth_outputs email-settings segment"
        )
        demo._auth_refresh = _auth_refresh_all  # type: ignore[attr-defined]

    return demo


def _auth_refresh_all():
    """Refresh every stored-data workspace component for the signed-in user.

    When no user is signed in on this thread (login gate, failed attempt,
    expired session), returns empty placeholder updates instead of reading
    any database — so an unauthenticated API call to the open load/auth
    events can never harvest data from the global/default store.
    """
    if auth.active_user() is None:
        return (gr.update(),) * _AUTH_REFRESH_OUTPUTS
    jobs = _job_choices()
    focus = jobs[0][1] if jobs else None
    return (
        *refresh_workspace(),
        *_email_settings_refresh(),
        # Email templates — the compose + manager dropdowns for the default
        # email type (the radio resets to "Shortlist notification" on login).
        *_email_template_refresh("shortlist"),
        gr.update(value=_email_history_table()),
        gr.update(value=_video_history_table(focus)),
        *_profile_refresh(),
    )


# ---- Persistent session cookie (keeps users signed in across reloads) -----
# The session token lives in gr.BrowserState (localStorage), but Gradio 6 does
# NOT hydrate BrowserState values into the app.load event payload, so a reload
# would see token=None and drop the session. We mirror the token into a plain
# cookie client-side (via event `js` chains) and restore it from request
# headers in _on_page_load. SameSite=Lax; Max-Age matches SESSION_TTL_DAYS.
_SESSION_COOKIE = "talentiq_session"
# Max-Age derived from SESSION_TTL_DAYS so the cookie and the server-side
# session expiry can never drift apart (a stale cookie past TTL is harmless —
# it just fails to resolve — but keeping them in lockstep is the honest
# contract the comment below promises).
_SESSION_COOKIE_MAX_AGE = int(config.SESSION_TTL_DAYS) * 86400
_SESSION_COOKIE_JS = (
    "(t) => {\n"
    "  if (t && typeof t === 'string' && t.length >= 20) {\n"
    "    document.cookie = 'talentiq_session=' + encodeURIComponent(t) +\n"
    f"      '; Max-Age={_SESSION_COOKIE_MAX_AGE}; Path=/; SameSite=Lax';\n"
    "  } else {\n"
    "    document.cookie = 'talentiq_session=; Max-Age=0; Path=/';\n"
    "  }\n"
    "  return t;\n"
    "}\n"
)


def _read_session_cookie(request: gr.Request | None) -> str:
    """The remembered-session token carried by the browser cookie (if any)."""
    try:
        raw = (request.headers or {}).get("cookie", "") if request is not None else ""
    except Exception:
        return ""
    for part in raw.split(";"):
        key, _, value = part.strip().partition("=")
        if key == _SESSION_COOKIE:
            from urllib.parse import unquote
            try:
                return unquote(value.strip())
            except Exception:
                return value.strip()
    return ""


def _client_ip(request: gr.Request | None) -> str:
    """Best-effort client IP for auth rate limiting ('' when unavailable)."""
    try:
        if request is None:
            return ""
        client = getattr(request, "client", None)
        if client is None:
            return ""
        return str(getattr(client, "host", "") or "")
    except Exception:
        return ""


def build_app() -> gr.Blocks:
    """The full app: a login/register gate wrapping the recruiter workspace.

    Signing in switches db/vectorstore/reports to the user's private storage
    (auth.set_active_user) and then refreshes every workspace component, so
    each account only ever sees its own data. Logging out returns to the gate.
    """

    def _on_auth_mode(mode):
        return gr.update(visible=mode == "Create account")

    def _on_auth_submit(mode, email, name, password, token, request: gr.Request | None = None):
        # Returns: auth_view, workspace_view, auth_msg, user_badge,
        # session_token + the 19 data refreshes (24 total).
        ip = _client_ip(request)
        # The auth events are unauthenticated by definition. Never let a
        # thread-scoped user from an earlier request bleed into this one —
        # otherwise a failed login on a reused pool thread could read that
        # user's data through the refresh below.
        auth.set_active_user(None)
        if mode == "Create account":
            try:
                user = auth.register_user(email, password, name, ip=ip)
            except auth.AuthLockedError as e:
                return (
                    gr.update(), gr.update(), f"**{e}**", gr.update(),
                    gr.update(),  # token unchanged
                    *workspace._auth_refresh(),  # type: ignore[attr-defined]
                )
            except ValueError as e:
                return (
                    gr.update(), gr.update(), f"**{e}**", gr.update(),
                    gr.update(),  # token unchanged
                    *workspace._auth_refresh(),  # type: ignore[attr-defined]
                )
        else:
            try:
                user = auth.authenticate(email, password, ip=ip)
            except auth.AuthLockedError as e:
                return (
                    gr.update(), gr.update(), f"**{e}**", gr.update(),
                    gr.update(),  # token unchanged
                    *workspace._auth_refresh(),  # type: ignore[attr-defined]
                )
            if not user:
                return (
                    gr.update(), gr.update(), "**Invalid email or password.**", gr.update(),
                    gr.update(),  # token unchanged
                    *workspace._auth_refresh(),  # type: ignore[attr-defined]
                )
        auth.set_active_user(user)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=f"Welcome, **{user['email']}**"),
            gr.update(value=f"**Signed in as {user['email']}**"),
            auth.create_session(user["id"]),
            *workspace._auth_refresh(),  # type: ignore[attr-defined]
        )

    def _request_origin(request: gr.Request | None) -> str:
        """The exact page origin (scheme+host) the browser is on.

        Only used as a FALLBACK redirect URI — the Google callback exchange
        prefers the redirect URI stored with the OAuth attempt at start time
        (see _on_page_load), because the callback request's own headers
        (Referer = accounts.google.com, often no usable Origin) cannot be
        trusted to reproduce it.
        """
        if request is not None:
            origin = auth.normalize_redirect_uri(
                (request.headers or {}).get("origin")
                or (request.headers or {}).get("referer")
                or ""
            )
            if origin:
                return origin
        # Fallback: infer from how the app is bound.
        host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
        port = os.getenv("PORT", "7860")
        if os.getenv("SPACE_ID"):
            return f"https://{os.getenv('SPACE_ID')}.hf.space/"
        return f"http://{host}:{port}/"



    def _on_logout(token):
        auth.delete_session(token)
        auth.set_active_user(None)
        # Reset the whole auth form too — otherwise the mode stays on
        # "Create account" with the old email/password prefilled, and the
        # next "Sign in" click would try to RE-CREATE the account and fail
        # with "already exists" (the user is then locked out of relogin).
        return (
            gr.update(visible=True),       # auth_view
            gr.update(visible=False),      # workspace_view
            gr.update(value=""),          # user_badge
            gr.update(value="_Signed out._"),  # auth_msg
            None,                          # session_token -> cleared
            gr.update(value="Sign in"),   # auth_mode -> back to sign-in
            gr.update(value=""),          # auth_email
            gr.update(value=""),          # auth_name
            gr.update(value=""),          # auth_password
        )

    def _on_page_load(token, request: gr.Request | None = None):
        """Restore a remembered login when the page loads (or reloads).

        The session token lives in the browser (localStorage via
        gr.BrowserState) and is resolved against users.db — so a reload,
        a closed-and-reopened tab, or even a server restart keeps the user
        signed in. Without a valid token the login gate shows and any stale
        in-memory user is cleared.

        Google's redirect flow lands back here with `?code=...&state=...` in
        the URL; when present, the one-time code is exchanged for the user
        and a fresh session token is minted + stored (this event is the
        Google flow's equivalent of _on_auth_submit's create_session).
        Refreshing that callback URL re-runs this handler with the SAME
        query params but a consumed (one-time) state — that is treated as a
        plain page load, never as a failed sign-in, and the remembered
        session (if any) is restored as usual.
        """
        user = None
        oauth_msg = ""
        query = getattr(request, "query_params", None) or {}
        code = query.get("code")
        state = query.get("state")
        new_token = token
        # Gradio 6 does not hydrate BrowserState into the load-event payload,
        # so restore from the session cookie the login flow set client-side.
        if not new_token:
            new_token = _read_session_cookie(request)
        if code and state:
            # Google callback: the PKCE verifier for this attempt is stored
            # server-side under the state token (validated + consumed here).
            attempt = auth.pop_google_attempt(state)
            if attempt:
                try:
                    # The exchange MUST reuse the redirect URI the browser
                    # started with (stored on the attempt) — Google requires
                    # the token-exchange redirect_uri to match the auth
                    # request's exactly, and the callback request's own
                    # headers (Referer = accounts.google.com, no app Origin)
                    # would derive a mismatching one and reject every login.
                    user = auth.exchange_google_code(
                        attempt.get("redirect_uri") or _request_origin(request),
                        code,
                        attempt["verifier"],
                    )
                except Exception:
                    user = None
                if not user:
                    # An exchange was actually attempted and failed — surface
                    # it so the recruiter can click the button again. (If an
                    # older session still resolves below, the Welcome banner
                    # replaces this — intentional: they are signed in.)
                    oauth_msg = "**Google sign-in failed** — please try again."
                else:
                    # The Google flow never passes through _on_auth_submit,
                    # so mint the session right here (the browser stores it
                    # via the session_token output below). Without it the
                    # workspace would render once, then every later API call
                    # would be rejected by the auth gate. When the browser
                    # already holds a valid token for the SAME account, keep
                    # it — avoids churning a fresh session row on every
                    # Google re-login.
                    existing = auth.get_user_by_session(token) if token else None
                    if existing and existing.get("id") == user.get("id"):
                        new_token = token
                    else:
                        new_token = auth.create_session(user["id"])
            # No verifier for this state: the callback was already handled —
            # this is a refresh (or a replay) of the `?code=...&state=...`
            # URL, not a fresh attempt. Never claim the sign-in failed; fall
            # through to restoring the remembered session below.
        if not user:
            user = auth.get_user_by_session(new_token) if new_token else None
        if not user:
            # Logged out: never read any database (the refresh returns empty
            # placeholder updates when no user is active on this thread), and
            # drop any stale token from the browser.
            auth.set_active_user(None)
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=oauth_msg),
                gr.update(value=""),
                None,  # session_token -> cleared if it no longer resolves
                *workspace._auth_refresh(),  # type: ignore[attr-defined]
            )
        auth.set_active_user(user)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=f"Welcome, **{user['email']}**"),
            gr.update(value=f"**Signed in as {user['email']}**"),
            new_token,
            *workspace._auth_refresh(),  # type: ignore[attr-defined]
        )

    workspace = build_demo()

    with gr.Blocks(
        title="TalentIQ · AI Recruiter",
        theme=_theme(),
        css=custom_css,
    ) as app:
        with gr.Column(visible=True, elem_classes=["auth-view"]) as auth_view:
            gr.HTML(
                """
                <div class="brand-hero auth-hero">
                  <h1>TalentIQ</h1>
                  <p class="tagline">Your private AI recruiting workspace</p>
                </div>
                """
            )
            auth_mode = gr.Radio(
                choices=["Sign in", "Create account"],
                value="Sign in",
                label=None,
                elem_classes=["auth-mode"],
            )
            auth_email = gr.Textbox(label="Email", placeholder="you@example.com")
            auth_name = gr.Textbox(label="Full name", placeholder="Jane Doe", visible=False)
            auth_password = gr.Textbox(label="Password", type="password", placeholder="••••••••")
            auth_btn = gr.Button("Sign in", variant="primary")
            auth_msg = gr.Markdown("", elem_classes=["status-note"])
            gr.HTML('<div class="auth-divider"><span>or continue with</span></div>')
            # The Google button is a real link to the server-side OAuth start
            # route (302 -> Google consent), so no client JS is involved. When
            # Google is unconfigured a setup hint is shown instead.
            if auth.google_enabled():
                gr.HTML(
                    '<a class="google-btn" href="/auth/google/start">'
                    '<svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">'
                    '<path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>'
                    '<path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>'
                    '<path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>'
                    '<path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>'
                    '</svg>Continue with Google</a>'
                )
            else:
                gr.Markdown(
                    "_Google sign-in is not configured — add `GOOGLE_CLIENT_ID` and "
                    "`GOOGLE_CLIENT_SECRET` to `.env`, then restart the app._",
                    elem_classes=["status-note"],
                )

        with gr.Column(visible=False, elem_classes=["workspace-view"]) as workspace_view:
            with gr.Row(elem_classes=["session-bar"]):
                user_badge = gr.Markdown("", elem_classes=["user-badge"])
                profile_icon_btn = gr.Button(
                    "👤", elem_classes=["profile-icon-btn"], scale=0
                )
            workspace.render()

        # Persistent login token: kept in the browser (localStorage), so a
        # page reload restores the session instead of showing the gate. The
        # component is created inside build_demo (every workspace event wires
        # it as an input) and shared here for the auth events.
        session_token = workspace._session_token  # type: ignore[attr-defined]

        auth_mode.change(fn=_on_auth_mode, inputs=[auth_mode], outputs=[auth_name])
        auth_btn.click(
            fn=_on_auth_submit,
            inputs=[auth_mode, auth_email, auth_name, auth_password, session_token],
            outputs=[
                auth_view, workspace_view, auth_msg, user_badge, session_token,
                *workspace._auth_outputs,  # type: ignore[attr-defined]
            ],
        ).then(
            # Client-side: persist the freshly minted session token in a
            # cookie so a page reload can restore it (Gradio 6 does not
            # hydrate BrowserState into load-event payloads). Clears the
            # cookie when the login failed (token unchanged/None).
            js=_SESSION_COOKIE_JS,
            inputs=[session_token],
            outputs=[session_token],
        )
        # The 👤 icon opens the floating Profile bubble; ✕ Close hides it, and
        # clicking anywhere outside (the backdrop) closes it too.
        profile_icon_btn.click(
            fn=_open_profile,
            inputs=[session_token],
            outputs=[
                workspace._pf_pop,  # type: ignore[attr-defined]
                workspace._bubble_backdrop,  # type: ignore[attr-defined]
            ],
        )
        workspace._pf_close_btn.click(  # type: ignore[attr-defined]
            fn=_close_profile,
            inputs=[session_token],
            outputs=[
                workspace._pf_pop,  # type: ignore[attr-defined]
                workspace._bubble_backdrop,  # type: ignore[attr-defined]
            ],
        )
        # Profile popover: rename, log out, and permanent account deletion.
        workspace._pf_save_btn.click(  # type: ignore[attr-defined]
            fn=_on_save_profile,
            inputs=[session_token, workspace._pf_name],  # type: ignore[attr-defined]
            outputs=[workspace._pf_status, workspace._pf_email],  # type: ignore[attr-defined]
        )
        workspace._pf_logout_btn.click(  # type: ignore[attr-defined]
            fn=_on_logout,
            inputs=[session_token],
            outputs=[
                auth_view, workspace_view, user_badge, auth_msg, session_token,
                auth_mode, auth_email, auth_name, auth_password,
            ],
        ).then(
            # Logout sets the token to None — the js clears the cookie.
            js=_SESSION_COOKIE_JS,
            inputs=[session_token],
            outputs=[session_token],
        )
        workspace._pf_del_btn.click(  # type: ignore[attr-defined]
            fn=_on_delete_account,
            inputs=[session_token, workspace._pf_del_confirm],  # type: ignore[attr-defined]
            outputs=[
                auth_view, workspace_view, user_badge, auth_msg, session_token,
                auth_mode, auth_email, auth_name, auth_password,
                *workspace._pf_outputs,  # type: ignore[attr-defined]
            ],
        ).then(
            # Deletion clears the session — the js drops the persisted cookie.
            js=_SESSION_COOKIE_JS,
            inputs=[session_token],
            outputs=[session_token],
        )
        app.load(
            fn=_on_page_load,
            inputs=[session_token],
            outputs=[
                auth_view, workspace_view, auth_msg, user_badge,
                session_token,
                *workspace._auth_outputs,  # type: ignore[attr-defined]
            ],
        ).then(
            # Restored (cookie or Google callback) tokens are mirrored back
            # into the cookie; a failed restore clears a stale cookie.
            js=_SESSION_COOKIE_JS,
            inputs=[session_token],
            outputs=[session_token],
        )

    return app


demo = build_app()
