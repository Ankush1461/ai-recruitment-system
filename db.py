# ================================================================
# 🗄️ SQLite Persistence — Jobs, Candidates, Screenings, Interviews
# ================================================================

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any

import config

_DB_PATH = str(config.DB_PATH)  # default, overridable via RECRUITER_DB_PATH
# Per-thread override: set by auth.set_active_user / auth.user_scope so two
# concurrent requests (Gradio runs handlers in a thread pool) can never see
# each other's database. Falls back to _DB_PATH when unset.
_thread = threading.local()


def _active_db_path() -> str:
    return getattr(_thread, "db_path", None) or _DB_PATH


def set_active_db(db_path: str | None) -> None:
    """Point THIS THREAD at a specific database file (per-user isolation).

    The default is the global recruiter.db; after login each account switches
    to its own file and the schema is ensured to exist there. Passing None
    clears the thread override, falling back to the module default (_DB_PATH).
    """
    if db_path is None:
        _thread.db_path = None
    else:
        path = str(db_path)
        _thread.db_path = path
        init_db(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


@contextmanager
def connect(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or _active_db_path()
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    # WAL lets concurrent readers/writers coexist (Gradio runs handlers in a
    # thread pool; two instances can even share the file) and busy_timeout
    # turns lock contention into a wait instead of an immediate error.
    conn.execute("PRAGMA busy_timeout = 30000")
    if config.SQLITE_WAL:
        # read-only filesystem — the default journal mode still works
        with suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    """Create tables if they do not exist."""
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                req_id TEXT DEFAULT '',
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                requirements_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY,
                job_id TEXT DEFAULT '',
                name TEXT,
                resume_text TEXT NOT NULL,
                source TEXT DEFAULT 'upload',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS screenings (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                score INTEGER NOT NULL DEFAULT 0,
                verdict TEXT NOT NULL DEFAULT 'FAIL',
                rubric_json TEXT DEFAULT '{}',
                report_json TEXT DEFAULT '{}',
                summary TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            );

            CREATE TABLE IF NOT EXISTS interviews (
                id TEXT PRIMARY KEY,
                screening_id TEXT,
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                messages_json TEXT DEFAULT '[]',
                questions_json TEXT DEFAULT '[]',
                answers_json TEXT DEFAULT '[]',
                eval_json TEXT DEFAULT '{}',
                average_score REAL DEFAULT 0,
                verdict TEXT DEFAULT '',
                status TEXT DEFAULT 'in_progress',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (screening_id) REFERENCES screenings(id),
                FOREIGN KEY (job_id) REFERENCES jobs(id),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            );

            CREATE TABLE IF NOT EXISTS shortlists (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                results_json TEXT NOT NULL DEFAULT '[]',
                top_n INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE TABLE IF NOT EXISTS video_interviews (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                video_path TEXT DEFAULT '',
                transcript TEXT DEFAULT '',
                qa_json TEXT DEFAULT '[]',
                eval_json TEXT DEFAULT '{}',
                average_score REAL DEFAULT 0,
                verdict TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            );

            CREATE TABLE IF NOT EXISTS email_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                host TEXT NOT NULL DEFAULT '',
                port INTEGER NOT NULL DEFAULT 587,
                mail_from TEXT NOT NULL DEFAULT '',
                mail_from_name TEXT NOT NULL DEFAULT '',
                user TEXT NOT NULL DEFAULT '',
                password TEXT NOT NULL DEFAULT '',
                starttls INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS job_candidates (
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                status TEXT DEFAULT 'shortlisted',
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                PRIMARY KEY (job_id, candidate_id),
                FOREIGN KEY (job_id) REFERENCES jobs(id),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                detail TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        _migrate_job_candidates(conn)
        _migrate_job_req_id(conn)
        _migrate_candidate_job_id(conn)


def _migrate_job_req_id(conn: sqlite3.Connection) -> None:
    """Add req_id to jobs (older builds lack it) and backfill human-readable
    requisition IDs for every existing job row."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    except sqlite3.OperationalError:
        return  # table unavailable — skip the migration entirely (fail-open)
    if "req_id" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN req_id TEXT DEFAULT ''")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "req_id" not in cols:
        return
    rows = conn.execute(
        "SELECT id FROM jobs WHERE req_id IS NULL OR req_id = '' ORDER BY created_at ASC"
    ).fetchall()
    n = 1000 + conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    for row in rows:
        candidate = f"REQ-{n}"
        while conn.execute(
            "SELECT 1 FROM jobs WHERE req_id = ?", (candidate,)
        ).fetchone():
            n += 1
            candidate = f"REQ-{n}"
        conn.execute("UPDATE jobs SET req_id = ? WHERE id = ?", (candidate, row["id"]))
        n += 1


def _migrate_candidate_job_id(conn: sqlite3.Connection) -> None:
    """Add job_id to candidates (older builds had a global pool) and backfill
    each candidate's owning job from its job_candidates link."""
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(candidates)").fetchall()}
    except sqlite3.OperationalError:
        return
    if "job_id" not in cols:
        conn.execute("ALTER TABLE candidates ADD COLUMN job_id TEXT DEFAULT ''")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(candidates)").fetchall()}
    if "job_id" not in cols:
        return
    conn.execute(
        """
        UPDATE candidates SET job_id = (
            SELECT jc.job_id FROM job_candidates jc
            WHERE jc.candidate_id = candidates.id
            ORDER BY jc.created_at DESC LIMIT 1
        ) WHERE job_id IS NULL OR job_id = ''
        """
    )


def _next_req_id(conn: sqlite3.Connection) -> str:
    """Generate the next free REQ-XXXX id."""
    n = 1000 + int(conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] or 0)
    while conn.execute(
        "SELECT 1 FROM jobs WHERE req_id = ?", (f"REQ-{n}",)
    ).fetchone():
        n += 1
    return f"REQ-{n}"


def _migrate_job_candidates(conn: sqlite3.Connection) -> None:
    """Normalize legacy job_candidates schemas onto the current shape.

    Older builds used `added_at` instead of `created_at` and omitted
    `notes`; this renames / adds columns in place (preserving rows) and
    backfills created_at for rows that predate the column.
    """
    try:
        cols = {
            r[1] for r in conn.execute("PRAGMA table_info(job_candidates)").fetchall()
        }
    except sqlite3.OperationalError:
        return
    if "created_at" not in cols and "added_at" in cols:
        conn.execute(
            "ALTER TABLE job_candidates RENAME COLUMN added_at TO created_at"
        )
        cols = {
            r[1] for r in conn.execute("PRAGMA table_info(job_candidates)").fetchall()
        }
    if "notes" not in cols:
        conn.execute("ALTER TABLE job_candidates ADD COLUMN notes TEXT DEFAULT ''")
        cols = {
            r[1] for r in conn.execute("PRAGMA table_info(job_candidates)").fetchall()
        }
    if "created_at" in cols:
        conn.execute(
            "UPDATE job_candidates SET created_at = COALESCE(created_at, '')"
        )


def audit(action: str, entity_type: str = "", entity_id: str = "", detail: str = "",
          db_path: str | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO audit_log (action, entity_type, entity_id, detail, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (action, entity_type, entity_id, detail, _utc_now()),
        )


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


# ---- Jobs -----------------------------------------------------------------

def create_job(title: str, description: str, requirements: list | None = None,
               req_id: str | None = None, db_path: str | None = None) -> dict:
    """Create a job with a human-readable requisition ID (REQ-XXXX).

    A custom req_id is honored only when unused; otherwise a unique one is
    auto-generated.
    """
    job_id = _new_id("job_")
    now = _utc_now()
    reqs = json.dumps(requirements or [])
    with connect(db_path) as conn:
        wanted = (req_id or "").strip().upper()
        if wanted and conn.execute(
            "SELECT 1 FROM jobs WHERE req_id = ?", (wanted,)
        ).fetchone():
            wanted = ""
        req = wanted or _next_req_id(conn)
        conn.execute(
            "INSERT INTO jobs (id, req_id, title, description, requirements_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, req, title.strip(), description.strip(), reqs, now),
        )
    audit("create_job", "job", job_id, f"{req} {title}", db_path)
    return get_job(job_id, db_path)  # type: ignore[return-value]


def list_jobs(db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_job(job_id: str, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_dict(row)


def delete_job(job_id: str, db_path: str | None = None) -> None:
    """Delete a job listing and everything tied to it — its pipeline, shortlists,
    screenings, interviews, and the candidates that belong to it (each candidate
    is specific to one job)."""
    with connect(db_path) as conn:
        conn.execute("DELETE FROM job_candidates WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM shortlists WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM interviews WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM video_interviews WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM screenings WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM candidates WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    audit("delete_job", "job", job_id, db_path=db_path)


# ---- Candidates -----------------------------------------------------------

def upsert_candidate(
    candidate_id: str,
    resume_text: str,
    name: str | None = None,
    source: str = "upload",
    job_id: str | None = None,
    db_path: str | None = None,
) -> dict:
    """Create or update a candidate. Every candidate belongs to exactly one job
    listing; when job_id is provided the ownership is (re)assigned.

    Name handling: an explicit `name` always wins. When `name` is omitted, an
    existing row keeps its stored name (a fresh guess from the raw text would
    clobber the display name set at ingest — e.g. a resume whose first line is
    a "SUMMARY" heading); only brand-new rows get a guessed name."""
    now = _utc_now()
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id, name FROM candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
        if existing:
            display_name = (
                name or (existing["name"] or "").strip()
                or _guess_name(resume_text) or "Unknown"
            ).strip()
            fields = ["name = ?", "resume_text = ?", "source = ?"]
            values: list[Any] = [display_name, resume_text, source]
            if job_id is not None:
                fields.append("job_id = ?")
                values.append(job_id)
            values.append(candidate_id)
            conn.execute(
                f"UPDATE candidates SET {', '.join(fields)} WHERE id = ?", values
            )
        else:
            display_name = (name or _guess_name(resume_text) or "Unknown").strip()
            conn.execute(
                "INSERT INTO candidates (id, job_id, name, resume_text, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (candidate_id, job_id or "", display_name, resume_text, source, now),
            )
    audit("upsert_candidate", "candidate", candidate_id, display_name, db_path)
    return get_candidate(candidate_id, db_path)  # type: ignore[return-value]


def list_candidates(
    job_id: str | None = None,
    db_path: str | None = None,
) -> list[dict]:
    """Candidates — optionally filtered to a single job listing."""
    select = (
        "SELECT c.id, c.job_id, c.name, c.source, c.created_at, "
        "j.title AS job_title, "
        "substr(c.resume_text, 1, 120) AS resume_preview "
        "FROM candidates c LEFT JOIN jobs j ON j.id = c.job_id "
    )
    with connect(db_path) as conn:
        if job_id:
            rows = conn.execute(
                select + "WHERE c.job_id = ? ORDER BY c.created_at DESC", (job_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                select + "ORDER BY c.created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def get_candidate(candidate_id: str, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
    return _row_to_dict(row)


def update_candidate(
    candidate_id: str,
    *,
    name: str | None = None,
    resume_text: str | None = None,
    source: str | None = None,
    db_path: str | None = None,
) -> dict | None:
    """Update candidate profile fields. Keeps the same candidate ID."""
    existing = get_candidate(candidate_id, db_path)
    if not existing:
        return None
    new_name = (name if name is not None else existing.get("name") or "Unknown").strip()
    new_text = resume_text if resume_text is not None else existing["resume_text"]
    new_source = source if source is not None else existing.get("source", "upload")
    if not new_text or not str(new_text).strip():
        raise ValueError("Resume text cannot be empty")
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE candidates SET name = ?, resume_text = ?, source = ? WHERE id = ?",
            (new_name, new_text.strip(), new_source, candidate_id),
        )
    audit("update_candidate", "candidate", candidate_id, new_name, db_path)
    return get_candidate(candidate_id, db_path)


def delete_candidate(candidate_id: str, db_path: str | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM job_candidates WHERE candidate_id = ?", (candidate_id,))
        conn.execute("DELETE FROM interviews WHERE candidate_id = ?", (candidate_id,))
        conn.execute("DELETE FROM video_interviews WHERE candidate_id = ?", (candidate_id,))
        conn.execute("DELETE FROM screenings WHERE candidate_id = ?", (candidate_id,))
        conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
    audit("delete_candidate", "candidate", candidate_id, db_path=db_path)


# ---- Per-account email settings -------------------------------------------
# Each account stores its own SMTP sender in its private recruiter.db (single
# row, id=1). Email sends read these with .env fallback (see emailer.py) — so
# every account can use its own free SMTP (Gmail app password, Outlook, …)
# instead of the shared .env config. The SMTP password is encrypted at rest
# with a per-account key that auth.py derives from the account's password hash
# (see set_active_encryption_key) — a leaked .db file alone cannot recover it.

# On-thread encryption key for the SMTP password field, set by
# auth.set_active_user from the account's password hash. None (logged-out
# thread, tests, Google-only accounts) → plaintext, fail-open.
_ENC_PREFIX = "enc$v1$"


def set_active_encryption_key(key: bytes | None) -> None:
    """Point THIS THREAD at the at-rest SMTP-password key for the active user
    (None clears it — e.g. logout, or accounts without a password)."""
    _thread.enc_key = key


def _encrypt_password(plaintext: str) -> str:
    """Encrypt the SMTP password for storage.

    Stream cipher with a random 16-byte nonce seeding a PBKDF2-SHA256
    keystream (one iteration — the key material is already a slow-KDF output)
    XOR'd over the bytes, plus an HMAC-SHA256 tag over nonce||ciphertext so a
    wrong key (e.g. the account password changed) or tampering is detected on
    read. No key on the thread → plaintext, fail-open.
    """
    if not plaintext:
        return ""
    key = getattr(_thread, "enc_key", None)
    if not key:
        return plaintext
    data = plaintext.encode("utf-8")
    nonce = secrets.token_bytes(16)
    ks = hashlib.pbkdf2_hmac("sha256", key, nonce, 1, dklen=len(data))
    ct = bytes(a ^ b for a, b in zip(data, ks, strict=True))
    tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()
    return (
        _ENC_PREFIX
        + base64.urlsafe_b64encode(nonce).decode()
        + "$"
        + base64.urlsafe_b64encode(tag).decode()
        + "$"
        + base64.urlsafe_b64encode(ct).decode()
    )


def _decrypt_password(stored: str) -> str:
    """Reverse _encrypt_password. Legacy plaintext rows (saved before this
    feature, or written without a key) pass through unchanged; a value that
    fails the HMAC tag (wrong/absent key, tampering) yields '' so a send
    never uses garbage credentials."""
    if not stored or not stored.startswith(_ENC_PREFIX):
        return stored
    key = getattr(_thread, "enc_key", None)
    if not key:
        return ""
    try:
        _, _v, nonce_b64, tag_b64, ct_b64 = stored.split("$")
        nonce = base64.urlsafe_b64decode(nonce_b64)
        tag = base64.urlsafe_b64decode(tag_b64)
        ct = base64.urlsafe_b64decode(ct_b64)
        if not hmac.compare_digest(
            hmac.new(key, nonce + ct, hashlib.sha256).digest(), tag
        ):
            return ""
        ks = hashlib.pbkdf2_hmac("sha256", key, nonce, 1, dklen=len(ct))
        return bytes(a ^ b for a, b in zip(ct, ks, strict=True)).decode("utf-8", "replace")
    except (ValueError, TypeError):
        return ""


def save_email_settings(
    *,
    host: str = "",
    port: int = 587,
    mail_from: str = "",
    mail_from_name: str = "",
    user: str = "",
    password: str = "",
    starttls: bool = True,
    db_path: str | None = None,
) -> dict:
    """Upsert this account's SMTP settings (single row, id=1)."""
    now = _utc_now()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO email_settings "
            "(id, host, port, mail_from, mail_from_name, user, password, starttls, updated_at) "
            "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "host = excluded.host, port = excluded.port, mail_from = excluded.mail_from, "
            "mail_from_name = excluded.mail_from_name, user = excluded.user, "
            "password = excluded.password, starttls = excluded.starttls, "
            "updated_at = excluded.updated_at",
            (
                (host or "").strip(),
                int(port or 587),
                (mail_from or "").strip(),
                (mail_from_name or "").strip(),
                (user or "").strip(),
                _encrypt_password(password or ""),
                1 if starttls else 0,
                now,
            ),
        )
    audit("save_email_settings", "email_settings", "account", (host or "").strip(), db_path)
    return get_email_settings(db_path)  # type: ignore[return-value]


def get_email_settings(db_path: str | None = None) -> dict | None:
    """This account's saved SMTP settings (None when never saved).

    Wrapped so a DB that predates the table fails open (returns None) instead
    of raising — the emailer then simply falls back to the .env config.
    """
    try:
        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM email_settings WHERE id = 1"
            ).fetchone()
        cfg = _row_to_dict(row)
        if cfg and cfg.get("password"):
            raw = cfg["password"]
            was_encrypted = raw.startswith(_ENC_PREFIX)
            cfg["password"] = _decrypt_password(raw)
            # An encrypted value that no longer decrypts (account password
            # changed → different key, or tampering) — surface it so the UI
            # can ask the recruiter to re-enter the SMTP password.
            cfg["password_unreadable"] = was_encrypted and not cfg["password"]
        return cfg
    except sqlite3.OperationalError:
        return None


def clear_email_settings(db_path: str | None = None) -> None:
    """Drop the account's SMTP settings — sends revert to the .env config."""
    try:
        with connect(db_path) as conn:
            conn.execute("DELETE FROM email_settings WHERE id = 1")
    except sqlite3.OperationalError:
        return
    audit("clear_email_settings", "email_settings", "account", db_path=db_path)


# ---- Job ↔ Candidate pipeline (per-job candidate lists) ------------------

def add_candidate_to_job(
    job_id: str,
    candidate_id: str,
    status: str = "shortlisted",
    notes: str = "",
    db_path: str | None = None,
) -> None:
    """Assign a candidate to a job's pipeline. No global enrollment: linking a
    candidate to a job moves it there (any previous job link is removed)."""
    with connect(db_path) as conn:
        conn.execute(
            "DELETE FROM job_candidates WHERE candidate_id = ? AND job_id != ?",
            (candidate_id, job_id),
        )
        conn.execute(
            "INSERT OR IGNORE INTO job_candidates "
            "(job_id, candidate_id, status, notes, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, candidate_id, status, notes, _utc_now()),
        )
        conn.execute(
            "UPDATE candidates SET job_id = ? WHERE id = ?", (job_id, candidate_id)
        )
    audit(
        "add_candidate_to_job",
        "job_candidate",
        f"{job_id}:{candidate_id}",
        db_path=db_path,
    )


def remove_candidate_from_job(
    job_id: str, candidate_id: str, db_path: str | None = None
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "DELETE FROM job_candidates WHERE job_id = ? AND candidate_id = ?",
            (job_id, candidate_id),
        )
        # Detach the candidate from the job so it leaves this job's list
        # entirely (candidates belong to exactly one job at a time).
        conn.execute(
            "UPDATE candidates SET job_id = '' WHERE id = ? AND job_id = ?",
            (candidate_id, job_id),
        )
    audit(
        "remove_candidate_from_job",
        "job_candidate",
        f"{job_id}:{candidate_id}",
        db_path=db_path,
    )


def list_job_candidates(job_id: str, db_path: str | None = None) -> list[dict]:
    """Candidates linked to one job (joined with candidate metadata)."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT jc.*, c.name AS candidate_name, c.source AS candidate_source,
                   c.created_at AS candidate_created
            FROM job_candidates jc
            JOIN candidates c ON c.id = jc.candidate_id
            WHERE jc.job_id = ?
            ORDER BY jc.created_at ASC
            """,
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def job_candidate_count(job_id: str, db_path: str | None = None) -> int:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM job_candidates WHERE job_id = ?", (job_id,)
        ).fetchone()
    return int(row["n"] or 0) if row else 0


# Common resume section headings — never a candidate name. In all-caps, Title
# Case or lowercase ("SUMMARY", "Professional Summary", "skills", ...).
_SECTION_HEADINGS = {
    "summary", "professional summary", "executive summary", "career summary",
    "technical summary", "objective", "career objective", "profile",
    "professional profile", "about", "about me", "personal details",
    "personal information", "work experience", "professional experience",
    "experience", "employment history", "work history", "history",
    "professional background", "background", "education", "academic background",
    "skills", "technical skills", "core competencies", "key skills",
    "competencies", "projects", "key projects", "professional projects",
    "project experience", "certifications", "certificates", "licenses",
    "achievements", "awards", "honors", "publications", "papers",
    "interests", "hobbies", "references", "contact", "contact information",
    "languages", "volunteering", "volunteer experience",
    "additional information", "highlights", "qualifications", "resume",
    "curriculum vitae", "cv", "work", "employment", "technical experience",
    "additional skills",
}


def _guess_name(resume_text: str) -> str:
    for line in resume_text.strip().splitlines()[:8]:
        cleaned = line.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower().rstrip(".:|- ").strip()
        if lowered in _SECTION_HEADINGS:
            continue
        # "Name: Maya Chen"-style label lines — take the value after the colon.
        if ":" in cleaned:
            key, _, value = cleaned.partition(":")
            value = value.strip()
            if (
                key.strip().lower() in ("name", "full name", "candidate")
                and value
                and len(value.split()) <= 6
                and len(value) < 80
            ):
                return value
        if "@" in cleaned or cleaned.lower().startswith(("email", "phone", "http")):
            continue
        # Comma-separated lists ("Python, PyTorch, Docker") are skill lines,
        # never names. A single comma is fine — "Chen, Maya" is a real format.
        if cleaned.count(",") >= 2:
            continue
        if len(cleaned.split()) <= 6 and len(cleaned) < 80:
            return cleaned
    return "Unknown"


# ---- Screenings -----------------------------------------------------------

def save_screening(
    job_id: str,
    candidate_id: str,
    score: int,
    verdict: str,
    rubric: dict | None = None,
    report: dict | None = None,
    summary: str = "",
    db_path: str | None = None,
) -> dict:
    sid = _new_id("scr_")
    now = _utc_now()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO screenings "
            "(id, job_id, candidate_id, score, verdict, rubric_json, report_json, summary, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sid,
                job_id,
                candidate_id,
                int(score),
                verdict,
                json.dumps(rubric or {}),
                json.dumps(report or {}),
                summary,
                now,
            ),
        )
    audit("save_screening", "screening", sid, f"score={score} verdict={verdict}", db_path)
    return get_screening(sid, db_path)  # type: ignore[return-value]


def get_screening(screening_id: str, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM screenings WHERE id = ?", (screening_id,)
        ).fetchone()
    return _row_to_dict(row)


def list_screenings(job_id: str | None = None, db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        if job_id:
            rows = conn.execute(
                """
                SELECT s.*, c.name AS candidate_name, j.title AS job_title
                FROM screenings s
                JOIN candidates c ON c.id = s.candidate_id
                JOIN jobs j ON j.id = s.job_id
                WHERE s.job_id = ?
                ORDER BY s.score DESC, s.created_at DESC
                """,
                (job_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT s.*, c.name AS candidate_name, j.title AS job_title
                FROM screenings s
                JOIN candidates c ON c.id = s.candidate_id
                JOIN jobs j ON j.id = s.job_id
                ORDER BY s.created_at DESC
                """
            ).fetchall()
    return [dict(r) for r in rows]


def latest_screening(job_id: str, candidate_id: str, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM screenings WHERE job_id = ? AND candidate_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (job_id, candidate_id),
        ).fetchone()
    return _row_to_dict(row)


def list_qualified_for_job(job_id: str, db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.*, c.name AS candidate_name
            FROM screenings s
            JOIN candidates c ON c.id = s.candidate_id
            WHERE s.job_id = ? AND s.verdict = 'PASS'
            ORDER BY s.score DESC, s.created_at DESC
            """,
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---- Interviews -----------------------------------------------------------

def create_interview(
    job_id: str,
    candidate_id: str,
    screening_id: str | None,
    questions: list[str],
    db_path: str | None = None,
) -> dict:
    iid = _new_id("int_")
    now = _utc_now()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO interviews "
            "(id, screening_id, job_id, candidate_id, messages_json, questions_json, "
            " answers_json, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'in_progress', ?, ?)",
            (
                iid,
                screening_id,
                job_id,
                candidate_id,
                "[]",
                json.dumps(questions),
                "[]",
                now,
                now,
            ),
        )
    audit("create_interview", "interview", iid, db_path=db_path)
    return get_interview(iid, db_path)  # type: ignore[return-value]


def update_interview(
    interview_id: str,
    *,
    messages: list | None = None,
    answers: list | None = None,
    eval_data: dict | None = None,
    average_score: float | None = None,
    verdict: str | None = None,
    status: str | None = None,
    db_path: str | None = None,
) -> dict | None:
    fields: list[str] = ["updated_at = ?"]
    values: list[Any] = [_utc_now()]
    if messages is not None:
        fields.append("messages_json = ?")
        values.append(json.dumps(messages))
    if answers is not None:
        fields.append("answers_json = ?")
        values.append(json.dumps(answers))
    if eval_data is not None:
        fields.append("eval_json = ?")
        values.append(json.dumps(eval_data))
    if average_score is not None:
        fields.append("average_score = ?")
        values.append(average_score)
    if verdict is not None:
        fields.append("verdict = ?")
        values.append(verdict)
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    values.append(interview_id)
    with connect(db_path) as conn:
        conn.execute(
            f"UPDATE interviews SET {', '.join(fields)} WHERE id = ?",
            values,
        )
    return get_interview(interview_id, db_path)


def get_interview(interview_id: str, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM interviews WHERE id = ?", (interview_id,)
        ).fetchone()
    return _row_to_dict(row)


def list_interviews(job_id: str | None = None, db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        if job_id:
            rows = conn.execute(
                """
                SELECT i.*, c.name AS candidate_name, j.title AS job_title
                FROM interviews i
                JOIN candidates c ON c.id = i.candidate_id
                JOIN jobs j ON j.id = i.job_id
                WHERE i.job_id = ?
                ORDER BY i.updated_at DESC
                """,
                (job_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT i.*, c.name AS candidate_name, j.title AS job_title
                FROM interviews i
                JOIN candidates c ON c.id = i.candidate_id
                JOIN jobs j ON j.id = i.job_id
                ORDER BY i.updated_at DESC
                """
            ).fetchall()
    return [dict(r) for r in rows]


# ---- Video interviews (uploaded recordings) --------------------------------

def save_video_interview(
    job_id: str,
    candidate_id: str,
    transcript: str,
    qa_pairs: list | None = None,
    eval_data: dict | None = None,
    video_path: str = "",
    average_score: float = 0.0,
    verdict: str = "",
    db_path: str | None = None,
) -> dict:
    """Persist an analyzed uploaded-video interview session."""
    vid = _new_id("vid_")
    now = _utc_now()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO video_interviews "
            "(id, job_id, candidate_id, video_path, transcript, qa_json, eval_json, "
            " average_score, verdict, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                vid, job_id, candidate_id, video_path or "", transcript or "",
                json.dumps(qa_pairs or []), json.dumps(eval_data or {}),
                float(average_score or 0), verdict or "", now,
            ),
        )
    audit("save_video_interview", "video_interview", vid, f"{job_id}:{candidate_id}", db_path)
    return get_video_interview(vid, db_path)  # type: ignore[return-value]


def get_video_interview(video_id: str, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM video_interviews WHERE id = ?", (video_id,)
        ).fetchone()
    return _row_to_dict(row)


def list_video_interviews(job_id: str | None = None, db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        if job_id:
            rows = conn.execute(
                "SELECT v.*, c.name AS candidate_name FROM video_interviews v "
                "LEFT JOIN candidates c ON c.id = v.candidate_id "
                "WHERE v.job_id = ? ORDER BY v.created_at DESC",
                (job_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT v.*, c.name AS candidate_name FROM video_interviews v "
                "LEFT JOIN candidates c ON c.id = v.candidate_id "
                "ORDER BY v.created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def list_audit(
    action: str | None = None,
    limit: int = 20,
    db_path: str | None = None,
) -> list[dict]:
    """Recent audit log entries, optionally filtered by action."""
    with connect(db_path) as conn:
        if action:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE action = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (action, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def parse_json_field(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default if default is not None else {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default if default is not None else {}


# ---- Shortlists (per-job ranking snapshots) -------------------------------

def save_shortlist(
    job_id: str,
    results: list[dict],
    top_n: int | None = None,
    db_path: str | None = None,
) -> dict:
    """Persist a ranked shortlist snapshot for a job (replaces prior snapshot)."""
    sid = _new_id("sl_")
    now = _utc_now()
    payload = json.dumps(results)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM shortlists WHERE job_id = ?", (job_id,))
        conn.execute(
            "INSERT INTO shortlists (id, job_id, results_json, top_n, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, job_id, payload, top_n, now),
        )
    audit(
        "save_shortlist",
        "shortlist",
        sid,
        f"job={job_id} n={len(results)}",
        db_path,
    )
    return get_shortlist(sid, db_path)  # type: ignore[return-value]


def get_shortlist(shortlist_id: str, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM shortlists WHERE id = ?", (shortlist_id,)
        ).fetchone()
    return _row_to_dict(row)


def get_latest_shortlist(job_id: str, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT s.*, j.title AS job_title FROM shortlists s "
            "JOIN jobs j ON j.id = s.job_id "
            "WHERE s.job_id = ? ORDER BY s.created_at DESC LIMIT 1",
            (job_id,),
        ).fetchone()
    return _row_to_dict(row)


def jobs_table_rows(db_path: str | None = None) -> list[list]:
    rows = []
    for j in list_jobs(db_path):
        n_screen = len(list_screenings(j["id"], db_path))
        n_cands = job_candidate_count(j["id"], db_path)
        sl = get_latest_shortlist(j["id"], db_path)
        sl_n = len(parse_json_field(sl.get("results_json"), [])) if sl else 0
        rows.append([
            j.get("req_id") or "",
            j["title"],
            n_cands,
            sl_n,
            n_screen,
            j["created_at"][:19],
        ])
    return rows
