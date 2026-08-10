# ================================================================
# ⚙️ Central Configuration — env-driven paths & model settings
# ================================================================
# Single source of truth for file locations and LLM settings.
# Every path can be overridden with an environment variable so a Hugging
# Face Space can point storage at a persistent /data mount, and so tests
# can redirect the DB / vector store to a temp directory.

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # make .env values visible before the first read below

_BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def _env_path(name: str, default: Path) -> Path:
    """Resolve an env var to an absolute path ('' falls back to default)."""
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser() if raw else default


# ---- Paths ---------------------------------------------------------------
DB_PATH: Path = _env_path("RECRUITER_DB_PATH", _BASE_DIR / "recruiter.db")
CHROMA_DIR: Path = _env_path("CHROMA_DIR", _BASE_DIR / "chroma_db")
EXPORT_DIR: Path = _env_path("EXPORT_DIR", _BASE_DIR / "exports")

# ---- Multi-user accounts -----------------------------------------------
# Global identity store (users + password hashes) — shared by everyone.
USERS_DB_PATH: Path = _env_path("USERS_DB_PATH", _BASE_DIR / "users.db")
# Per-user data root: each account gets its own recruiter.db, chroma dir and
# exports folder under here, so jobs/candidates/screenings stay isolated.
USER_DATA_DIR: Path = _env_path("USER_DATA_DIR", _BASE_DIR / "data" / "users")
# Google sign-in via the standard OAuth 2.0 authorization-code redirect flow
# with PKCE (browser -> Google consent -> back to the app with a one-time
# code, exchanged server-side for a verified id_token). Empty client id
# disables the Google button with a friendly hint.
GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
# Clock-skew tolerance (seconds) for Google id_token `iat`/`exp` validation.
# Google tokens carry a ~1-minute lifetime cushion, but the verifier rejects a
# token whose `iat` is even a second in the future — which fails every Google
# login whenever this machine's clock drifts a little behind Google's servers
# ("Token used too early"). 60s is the standard industry tolerance.
GOOGLE_CLOCK_SKEW_SECONDS: int = int(os.getenv("GOOGLE_CLOCK_SKEW_SECONDS", "60"))

# ---- Embeddings / rerank ----------------------------------------------------
# Multilingual paraphrase MiniLM-L12 — 384-dim, 50+ languages (EN + DE), CPU-
# friendly. Switching models invalidates existing vectors; vectorstore rebuilds
# them automatically on boot (see vectorstore.maybe_reindex_all).
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
# Multilingual cross-encoder for reranking EN + DE retrieval hits.
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# ---- Fine-tuned skill classifier ---------------------------------------------
# Optional tiny BERT fine-tuned on labeled resume skill phrases (skill_model.py;
# train with `python skill_model.py`). When a trained model exists at
# SKILL_MODEL_DIR, ranking falls back to skill-category matching when literal
# keyword overlap is zero (e.g. JD says "Amazon Web Services", resume says
# "AWS"). SKILL_CLASSIFIER_ENABLED=0 disables it; without a trained model the
# pipeline behaves exactly as before (fail-open).
SKILL_MODEL_DIR: Path = _env_path("SKILL_MODEL_DIR", _BASE_DIR / "data" / "skill_model")
SKILL_CLASSIFIER_ENABLED: bool = os.getenv("SKILL_CLASSIFIER_ENABLED", "1").lower() not in ("0", "false", "no", "off")

# ---- LLM provider settings -------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip().strip(chr(39) + chr(34))
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Fast/cheap model for low-stakes calls (follow-up detection). Hybrid routing
# keeps the strong model for scoring while the fast one absorbs volume.
GROQ_FAST_MODEL: str = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ---- Resilience (LLM retry / backoff) --------------------------------------
LLM_MAX_ATTEMPTS: int = int(os.getenv("LLM_MAX_ATTEMPTS", "4"))
LLM_BASE_DELAY: float = float(os.getenv("LLM_BASE_DELAY", "1.0"))

# ---- Email (SMTP) ----------------------------------------------------------
# Email is configured PER ACCOUNT from the Email tab → ⚙️ Email settings (each
# account brings its own free SMTP, e.g. a Gmail app password). The shared
# .env values are intentionally NOT used — see emailer.resolved_settings().

# ---- Video interviews (uploaded files are kept under this dir) ----------------
MEDIA_DIR: Path = _env_path("MEDIA_DIR", _BASE_DIR / "media")

# ---- Auth security (brute-force protection / session expiry) -------------------
# Failed sign-in attempts per email within AUTH_LOCKOUT_WINDOW trigger a lockout
# (AUTH_MAX_FAILED_ATTEMPTS); per-IP failures use AUTH_IP_MAX_FAILED_ATTEMPTS.
# Registrations are capped per IP within AUTH_REGISTER_WINDOW.
AUTH_MAX_FAILED_ATTEMPTS: int = int(os.getenv("AUTH_MAX_FAILED_ATTEMPTS", "5"))
AUTH_IP_MAX_FAILED_ATTEMPTS: int = int(os.getenv("AUTH_IP_MAX_FAILED_ATTEMPTS", "20"))
AUTH_LOCKOUT_WINDOW: int = int(os.getenv("AUTH_LOCKOUT_WINDOW", "900"))  # seconds
AUTH_MAX_REGISTRATIONS_PER_IP: int = int(os.getenv("AUTH_MAX_REGISTRATIONS_PER_IP", "5"))
AUTH_REGISTER_WINDOW: int = int(os.getenv("AUTH_REGISTER_WINDOW", "3600"))  # seconds
# Per-IP caps assume every client has a real, distinct IP. They are disabled by
# default on Hugging Face Spaces (SPACE_ID) because all users share the Space's
# proxy IP — the email-scoped limits still apply. Override with
# AUTH_ENFORCE_IP_LIMITS=0/1.
AUTH_ENFORCE_IP_LIMITS: bool = os.getenv(
    "AUTH_ENFORCE_IP_LIMITS", "0" if os.getenv("SPACE_ID") else "1"
).lower() not in ("0", "false", "no", "off")
# Persistent login sessions expire after this many days (fixed TTL from creation,
# enforced on every token resolution; expired sessions are pruned lazily).
SESSION_TTL_DAYS: int = int(os.getenv("SESSION_TTL_DAYS", "30"))

# ---- Upload limits (storage-DoS guard) -------------------------------------
# Uploaded PDFs are rejected above this size (MB). Gradio's file_types filter
# only checks extensions — the size cap is enforced by the handler itself.
MAX_PDF_UPLOAD_MB: int = int(os.getenv("MAX_PDF_UPLOAD_MB", "15"))
# Uploaded email logos (company branding) are rejected above this size (MB).
MAX_LOGO_MB: int = int(os.getenv("MAX_LOGO_MB", "2"))

# ---- SQLite journal mode -----------------------------------------------------
# WAL (default) lets the thread pool / two instances share the DB file. Set
# SQLITE_WAL=0 on cloud-synced filesystems (e.g. OneDrive) where the WAL
# sidecar files can cause intermittent locks, to fall back to the default
# journal mode.
SQLITE_WAL: bool = os.getenv("SQLITE_WAL", "1").lower() not in ("0", "false", "no", "off")

# ---- Hugging Face backup (free Spaces data survival) -------------------------
# Free Spaces have ephemeral disks — accounts (users.db) and per-user data
# (data/users/*) are wiped whenever the Space sleeps or rebuilds. backup.py
# tars the important files into a PRIVATE dataset repo on a timer and
# restores them on boot when the local disk is empty, so a rebuilt Space
# comes back with every account, job and candidate intact. Backup is a
# silent no-op unless HF_TOKEN + HF_BACKUP_REPO are both set.
HF_TOKEN: str = os.getenv("HF_TOKEN", "").strip()
HF_BACKUP_REPO: str = os.getenv("HF_BACKUP_REPO", "").strip()  # e.g. "user/talentiq-backup"
HF_BACKUP_INTERVAL_MIN: int = int(os.getenv("HF_BACKUP_INTERVAL_MIN", "30"))
HF_BACKUP_FIRST_DELAY_MIN: int = int(os.getenv("HF_BACKUP_FIRST_DELAY_MIN", "2"))
# Media recordings (live-interview uploads, up to 500 MB each) are excluded
# from backups by default to keep the periodic push fast; transcripts and
# evaluations live in the DBs regardless. Set 1 to archive the raw files too.
HF_BACKUP_INCLUDE_MEDIA: bool = os.getenv(
    "HF_BACKUP_INCLUDE_MEDIA", "0"
).lower() not in ("0", "false", "no", "off")
HF_BACKUP_ENABLED: bool = os.getenv(
    "HF_BACKUP_ENABLED", "1"
).lower() not in ("0", "false", "no", "off")
