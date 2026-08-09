# 🧭 TalentIQ — Complete Project Manual

**Smart AI Recruiter System (RAG Edition) · Phase 2 — Mini-ATS**

This manual is the single source of truth for the project: what every file does, how
the AI pipeline works, how a recruiter uses the app, and how a developer extends it.

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure — Every File](#3-project-structure--every-file)
4. [Data Model (SQLite)](#4-data-model-sqlite)
5. [The AI Pipeline, Step by Step](#5-the-ai-pipeline-step-by-step)
6. [User Manual — End-to-End Flow](#6-user-manual--end-to-end-flow)
7. [Developer Guide — Module Reference](#7-developer-guide--module-reference)
8. [UI, Theme & Styling Guide](#8-ui-theme--styling-guide)
9. [Configuration & Deployment](#9-configuration--deployment)
10. [Troubleshooting & Known Quirks](#10-troubleshooting--known-quirks)
11. [Roadmap (Phase 3)](#11-roadmap-phase-3)

---

## 1. Overview

TalentIQ is an end-to-end recruitment automation platform built in Python with:

| Layer | Technology |
| :--- | :--- |
| UI | **Gradio 6** (custom theme + CSS) |
| Persistence | **SQLite** (`recruiter.db`) |
| Vector search | **ChromaDB** (`chroma_db/`) |
| Embeddings | **Sentence-Transformers** `paraphrase-multilingual-MiniLM-L12-v2` (local, CPU) |
| Reranking | **Cross-Encoder** `mmarco-mMiniLMv2-L12-H384-v1` (optional) |
| LLM | **Groq** `llama-3.3-70b-versatile` with **Ollama** local fallback |

It implements a complete hiring workflow: create requisitions → ingest resumes per job →
hybrid semantic+keyword ranking → LLM rubric deep-screening → multi-turn technical
interviews → per-job CSV hiring reports.

**Design invariants (enforced throughout):**

- **Candidates are job-specific.** There is no global candidate pool — every candidate
  is owned by exactly one job listing and appears only in that job's pipeline, shortlist,
  and reports.
- **Raw internal IDs are never shown in the UI.** Jobs show a human **Req. ID**
  (e.g. `REQ-1001`); tables show names, scores, and timestamps only.
- **Sample data never pollutes real data.** Sample JDs only fill the *create-job form*;
  nothing is seeded into the job list automatically.
- **Instant, non-blocking sync.** Candidate ingest writes the SQLite row first and
  returns to the UI immediately (the embedding model loads + indexes on a background
  thread — `ingest_candidate_deferred`), so adding a resume or uploading PDFs never
  freezes the app, even on a cold model after boot. Ranking and deep-screening
  automatically refresh the Interview tab's dropdowns, so the tabs stay in sync
  without manual "Sync" clicks or page refreshes.
- **Local-first.** No resume leaves the machine except the LLM API call (Groq), which
  receives only the retrieved evidence + rubric prompt.

---

## 2. System Architecture

```mermaid
flowchart TD
    A[📄 PDF / Pasted Resume] --> B[🧩 chunking.py — section-aware split]
    B --> C[🧬 embeddings.py — multilingual MiniLM-L12]
    C --> D[🗄️ vectorstore.py — ChromaDB per candidate]
    D --> E[💾 db.py — SQLite: jobs · candidates · screenings · interviews]
    E --> F[📊 ranking.py — hybrid shortlist 0.7·semantic + 0.3·keyword]
    F --> G[🧠 screening.py — RAG evidence → LLM rubric JSON → score/verdict]
    G --> H{verdict == PASS?}
    H -- No --> I[📋 Stored FAIL + history]
    H -- Yes --> J[🎤 interview.py — multi-turn Q&A with follow-ups]
    J --> K[📤 reports.py — per-job CSV hiring report]
```

Data flows **left to right**, and every stage persists to SQLite so the UI's History tab
and CSV export can reconstruct the whole journey of a candidate.

---

## 3. Project Structure — Every File

```
AI_Recruiter/
├── app.py            # Entry point — launches the Gradio app
├── ui.py             # The entire UI: layout, handlers, theme, custom CSS
├── auth.py           # Accounts: email/password + Google login, per-user storage
├── db.py             # SQLite persistence layer (all tables + queries)
├── config.py         # Central config: env-driven paths + LLM settings
├── prompts.py        # Versioned LLM prompt registry
├── sample_data.py    # Sample JD templates (fill the create-job form only)
├── ranking.py        # Per-job hybrid shortlist (semantic + keyword)
├── screening.py      # RAG deep-screen: evidence → LLM rubric → verdict
├── interview.py      # Multi-turn interview session manager
├── emailer.py        # SMTP email: shortlist notifications + interview invites
├── live_interview.py # Free meeting links + browser-mic live transcript → Q&A → evaluation
├── video_interview.py# Shared transcript engine (Groq Whisper + Q&A parser + recording storage)
├── rubric.py         # Weighted rubric math, labels, prompts, badges
├── llm.py            # Unified LLM client (Groq + Ollama fallback)
├── vectorstore.py    # ChromaDB wrapper (index + search)
├── embeddings.py     # Local sentence-transformer wrapper
├── chunking.py       # Section-aware resume chunking + JD requirement split
├── rerank.py         # Optional cross-encoder reranker
├── pdf.py            # PDF → text extraction
├── reports.py        # Per-job CSV export
├── requirements.txt  # Python dependencies
├── requirements-dev.txt  # Dev/test dependencies (pytest)
├── tests/            # pytest suite: auth, db, ranking, screening, interview, live interview, emailer, …
├── .env.example      # Template for .env (git-ignored secrets)
├── .gitignore        # Secrets, runtime data, caches, tooling
├── README.md         # Quickstart & deployment
├── MANUAL.md         # This complete manual
├── exports/          # Generated CSV reports (regenerable)
├── media/            # Recorded live interviews (regenerable)
├── chroma_db/        # ChromaDB vector persistence (generated)
├── recruiter.db      # SQLite database (generated)
├── users.db          # Global identity store — accounts (generated)
└── data/users/       # Per-account private storage: recruiter.db, chroma/, exports/ (generated)
```

### File-by-file summary

| File | Role | Key exports |
| :--- | :--- | :--- |
| `app.py` | Entry point. Reads `PORT`, `GRADIO_SERVER_NAME`, `SPACE_ID` so the same file runs locally and on Hugging Face Spaces; falls back to the next free port when the default is busy. Initializes the identity store (`auth.init_db()`) at boot; restores data from the HF backup dataset before the DBs open, then starts the backup timer. | — |
| `backup.py` | **Free HF Spaces data survival**: tars `users.db` + `data/users/` (+ legacy `recruiter.db`/`chroma_db`) into `talentiq-backup.tar.gz` and pushes it to a **private dataset repo** on a timer; restores on boot when the local disk is empty (a wiped Space). Fail-open everywhere — a silent no-op without `HF_TOKEN` + `HF_BACKUP_REPO`. | `enabled`, `build_archive`, `push_backup`, `restore_if_needed`, `local_has_data`, `start_backup_timer` |
| `.github/workflows/keepalive.yml` | **Free keepalive**: GitHub Actions cron that pings the deployed Space URL every 30 min so it never idles out (needs the `HF_SPACE_URL` repo variable; UptimeRobot alternative documented in §9). | — |
| `start.bat` | One-click Windows launcher: venv check, first-run `.env` creation, boots `app.py`, auto-opens the browser (parses the real URL from the boot log, so port fallback works), live-tails the log. Closing its window stops the server. | — |
| `start-tunnel.bat` | Optional: exposes the running local app via a free Cloudflare quick tunnel (`cloudflared`) — detects the app's port, downloads cloudflared on first use, prints a `trycloudflare.com` link. | — |
| `auth.py` | Accounts: register/authenticate (PBKDF2-HMAC-SHA256 hashing, stdlib only), Google OAuth 2.0 **redirect flow with PKCE** (browser → Google → back to the app — no code entry), and per-user isolation. `set_active_user()` switches `db`/`vectorstore`/`reports` to the account's private storage; the first account automatically inherits the pre-upgrade legacy data. **Profile**: rename (`update_user_name`) and permanent account deletion (`delete_user` — user row, sessions, lockout history and the whole data dir). | `init_db`, `register_user`, `authenticate`, `get_or_create_google_user`, `set_active_user`, `update_user_name`, `delete_user`, `build_google_auth_url`, `exchange_google_code`, `new_pkce_pair` |
| `ui.py` | Everything the user sees: login gate + 6 workspace tabs, and a 👤 profile icon in the session bar that opens a floating **Profile** bubble (rename, log out, delete account), ~50 event handlers, the Gradio theme, and ~700 lines of custom CSS. Builds the workspace once (`demo = build_app()`, which nests `build_demo()` behind the login gate). | `build_app()`, `build_demo()`, `demo`, `custom_css`, handler functions |
| `db.py` | All SQLite access: schema init + migrations, jobs, candidates, job_candidates links, shortlists, screenings, interviews, **per-account `email_settings`** (single row, id=1), audit log. | `init_db`, `create_job`, `upsert_candidate`, `list_candidates`, `delete_job`, `save_screening`, `create_interview`, `save_email_settings`, `get_email_settings`, `clear_email_settings`, `jobs_table_rows`, … |
| `sample_data.py` | 3 sample JD titles + descriptions used to pre-fill the create form. No candidates. | `SAMPLE_JOB_DESCRIPTIONS` |
| `ranking.py` | Ranks one job's candidates with hybrid retrieval (no LLM). Persists shortlist snapshots; links ranked candidates to the job pipeline. | `ingest_candidate_deferred`, `rank_candidates_for_job`, `rank_and_save_shortlist`, `rank_jobs_batch`, `load_shortlist_results`, `format_ranking_*` |
| `screening.py` | Retrieves per-requirement evidence from Chroma, asks the LLM for a weighted rubric, computes score/verdict, persists the screening row. Also evaluates interview answers and suggests the 10 interview questions — both language-aware (English/German). | `ScreeningResult`, `screen_candidate`, `deep_screen_candidate`, `generate_interview_questions`, `suggest_interview_questions`, `evaluate_answers`, `format_screening_markdown` |
| `interview.py` | Session dataclass + start/answer/submit logic, follow-up detection, auto-screening on start, evaluation at the end. Language-aware: the session carries the chosen English/German setting. | `InterviewSession`, `start_interview`, `submit_answer` |
| `rubric.py` | The weighted dimensions, PASS/FAIL rules, and markdown/badge rendering. | `RUBRIC_WEIGHTS`, `compute_weighted_score`, `apply_verdict`, `verdict_badge`, `rubric_prompt_block` |
| `llm.py` | Unified chat client: Groq (primary) → Ollama (fallback). JSON mode + fence-stripping. | `LLMClient`, `get_llm_client` |
| `vectorstore.py` | ChromaDB collection per resume; index/clear/search with cosine similarity. | `index_resume`, `search_resume`, `clear_candidate` |
| `embeddings.py` | Lazy-loaded `paraphrase-multilingual-MiniLM-L12-v2`. | `embed_texts`, `model_name` |
| `chunking.py` | Splits resumes by sections (Experience, Skills, Education…) with sliding windows; splits JDs into requirement chunks. | `chunk_resume`, `split_jd_requirements` |
| `rerank.py` | Optional `mmarco-mMiniLMv2-L12-H384-v1` cross-encoder rerank of top hits. | `rerank`, `get_model` |
| `eval_retrieval.py` | **Offline retrieval evaluation**: runs the production chunker + embedder + cross-encoder over a small labeled resume↔JD dataset and reports `recall@k` / `MRR` / `NDCG@k` with and without the reranker (`python eval_retrieval.py`). No LLM calls — local models only. | `run_eval`, `recall_at_k`, `mean_reciprocal_rank`, `ndcg_at_k`, `main` |
| `skill_model.py` | **Fine-tuned skill classifier**: tiny BERT (`bert-mini`, 11M params) trained on ~450 hand-labeled resume skill phrases across 10 categories with a from-scratch PyTorch loop (warmup, grad clip, label smoothing, best-epoch macro-F1). Train with `python skill_model.py` → `data/skill_model`. Ranking falls back to skill-category matching when literal keyword overlap is zero (e.g. JD "Amazon Web Services" ↔ resume "AWS"). Fail-open when no model is present. | `train`, `load`, `available`, `classify_tokens`, `classify_token`, `main` |
| `pdf.py` | Extracts text from uploaded PDFs (pypdf). | `extract_pdf_text` |
| `reports.py` | Per-job CSV report with the full pipeline columns. | `export_job_csv` |
| `emailer.py` | Free SMTP email (bring your own server, e.g. Gmail app password). Builds branded HTML templates and sends via `smtplib`; every successful send is recorded in the audit log (`email_sent`). **Per-account SMTP config** — each account saves its own sender from Email tab → ⚙️ Email settings (bubble); there is **no `.env` fallback**. Fails gracefully when SMTP is unset. | `resolved_settings`, `is_configured`, `extract_email`, `build_shortlist_email`, `build_invite_email`, `send_email` |
| `live_interview.py` | **Live meeting interviews**: free **Jitsi** meeting links (rooms created on demand — no account, no time limit), browser-mic capture streamed in rolling chunks to Groq Whisper (`whisper-large-v3`), a rolling live transcript, and finish-and-evaluate through the shared RAG pipeline → persisted to `video_interviews`. Per-call WAV recording. | `LiveSession`, `LiveInterviewResult`, `start_live_session`, `generate_meeting_link`, `append_audio_chunk`, `stop_live_session`, `finish_live_interview` |
| `video_interview.py` | Shared live-interview engine: LLM Q&A splitter (speaker-labelled turns + pairs), WAV recording persistence, and media-dir plumbing used by the live interview flow (`live_interview.py`). | `parse_qa_pairs`, `format_qa_markdown`, `save_live_recording`, `write_wav` |

---

### Evaluating retrieval quality

`eval_retrieval.py` measures how well the retrieval pipeline finds the *right*
resumes for a job description — the question behind every shortlist. It chunks
the built-in labeled corpus (8 resumes incl. one German, 8 JD queries) with the
production chunker, embeds with the production model, and scores the rankings
with and without the cross-encoder reranker:

```powershell
python eval_retrieval.py            # recall@k / MRR / NDCG@k comparison table
python eval_retrieval.py --verbose  # per-query breakdown
```

Metrics are document-level (chunk hits collapse to the candidate, best score
wins) and the rerank path mirrors `search_resume`'s fetch-then-rerank pool.
No LLM API calls; the only network use is the one-time model download. The
harness takes injectable embedder/reranker callables, so the test suite covers
it fully offline with deterministic fakes (`tests/test_eval_retrieval.py`).

### Fine-tuned skill classifier

`skill_model.py` is the project's trained component: a `bert-mini` model
fine-tuned on ~450 labeled skill phrases (10 categories) with a hand-written
PyTorch training loop. Run `python skill_model.py` to (re)train it into
`data/skill_model` (gitignored — (re)train on deploy); `--check` classifies
example phrases. When a trained model exists, `ranking._keyword_overlap`
credits a partial keyword match when literal overlap is zero but the
classifier sees both sides in the same skill category, so JD requirements in
different wording still match ("Amazon Web Services" ↔ "AWS"). Everything is
fail-open: no model → identical behavior to before. Tests cover the loop,
roundtrip and integration fully offline (`tests/test_skill_model.py`).

## 4. Data Model (SQLite)

Schema lives in `db.py` → `init_db()` with automatic migrations for older databases.

| Table | Purpose | Notable columns |
| :--- | :--- | :--- |
| `jobs` | Job requisitions | `id`, `req_id` (REQ-1001…), `title`, `description`, `requirements_json`, `created_at` |
| `candidates` | Resume records | `id` (stable hash of resume), `job_id` (owner listing), `name`, `resume_text`, `source` (pdf/paste/screen), `created_at` |
| `job_candidates` | Link table: candidate ↔ job pipeline with status | `job_id`, `candidate_id`, `status` (shortlisted…), `notes` |
| `shortlists` | Saved ranked snapshots per job | `job_id`, `results_json` (list of `RankResult`), `top_n`, `created_at` |
| `screenings` | Deep-screen results | `job_id`, `candidate_id`, `score`, `verdict`, `summary`, `report_json` (rubric), `evidence_json` |
| `interviews` | Interview sessions | `job_id`, `candidate_id`, `screening_id`, `questions`, `answers`, `messages`, `status`, `average_score`, `verdict`, `eval_json` |
| `video_interviews` | Live-interview records (previously uploaded recordings) | `job_id`, `candidate_id`, `video_path`, `transcript`, `qa_json` (Q&A pairs), `eval_json`, `average_score`, `verdict` |
| `audit_log` | Append-only action log | `action`, `entity_type`, `entity_id`, `detail` |

**Key semantics:**

- **SQLite runs in WAL journal mode with a 30 s busy timeout** — concurrent
  readers/writers from Gradio's thread pool (or two app instances sharing the file)
  coexist without "database is locked" errors; the `-wal`/`-shm` sidecar files are
  git-ignored.
- **Candidate ownership** — `candidates.job_id` is the single owner. `add_candidate_to_job`
  "moves" a candidate to a job (clears other job links, sets `job_id`). This is what makes
  "each candidate belongs to exactly one job listing" true.
- **Deleting a job** (`delete_job`) also deletes its candidates, links, shortlists,
  screenings, interviews, and live interviews — a full cascade.
- **Stable candidate IDs** — `screening.stable_candidate_id(resume_text)` hashes the resume,
  so re-ingesting the same resume updates the same candidate instead of duplicating it.
- **Req. IDs** — auto-generated `REQ-1001`, `REQ-1002`, … (`_next_req_id`), or a custom ID
  typed by the recruiter (validated for uniqueness).

---

## 5. The AI Pipeline, Step by Step

### 5.1 Ingest (`ranking.ingest_candidate_deferred`)

1. Resume text (from PDF or paste) → stable candidate ID (hash).
2. `chunking.chunk_resume` splits text into section-aware chunks (Experience, Skills,
   Education, Projects…), with a sliding window so nothing is lost between sections.
3. `vectorstore.index_resume` embeds each chunk (local model) and stores it in ChromaDB
   under the candidate ID.
4. `db.upsert_candidate` saves the raw text to SQLite.
5. The candidate is immediately linked to its owning job's pipeline
   (`add_candidate_to_job`, status `shortlisted`).

### 5.2 Ranking (`ranking.rank_candidates_for_job`)

No LLM — fast and free.

- JD → requirements via `split_jd_requirements`.
- For each of the job's candidates:
  - **Semantic score (0–100):** for every requirement, retrieve top-2 chunks from the
    candidate's Chroma collection and take the best cosine similarity. Average across
    requirements; blend 60/40 with a global JD similarity.
  - **Keyword score (0–100):** token-overlap ratio between each requirement and the raw
    resume (stop-words filtered).
  - **Hybrid = 0.7 × semantic + 0.3 × keyword.**
- Sort descending, truncate to `top_n`, persist the shortlist snapshot, and link ranked
  candidates to the job pipeline.

### 5.3 Deep Screening (`screening.screen_candidate` / `deep_screen_candidate`)

1. For each JD requirement, retrieve the top-3 most relevant resume chunks
   (reranker enabled by default) → builds an *evidence block* per requirement.
2. The LLM (Groq or Ollama) receives the evidence and is asked to score four weighted
   dimensions on 0–10 with quoted evidence, returning strict JSON:
   - must_have_skills (40%) · experience (25%) · projects (20%) · education_extras (15%)
3. `rubric.compute_weighted_score` → overall /100.
4. **PASS requires overall ≥ 55 AND must-have skills ≥ 4/10** (`apply_verdict`).
5. The result (score, verdict, rubric, evidence, interview focus) is persisted.

### 5.4 Interview (`interview.start_interview` / `submit_answer`)

- Starting an interview auto-screens the candidate if they aren't PASS yet (one LLM call).
  Explicit FAIL candidates are blocked.
- **The AI suggests exactly 10 tailored questions** (`screening.suggest_interview_questions`)
  from the JD + resume + screening focus/gaps (one LLM call); a rule-based fallback
  guarantees 10 even without the LLM.
- Each answer: the LLM decides whether a short follow-up is warranted (vague/short
  answers get one follow-up per core question).
- After all questions, `evaluate_answers` scores the transcript and produces a verdict
  + per-question feedback, persisted to the `interviews` table.

**Interview language (English/German):** the Interview tab has a language selector. The
questions (`suggest_interview_questions`), follow-up probes, and the final evaluation are
generated in the selected language, and the rule-based fallback question set is localized
too — so a German interview stays German even when the LLM is unreachable. The choice is
stored on the `InterviewSession` and drives the whole session.

### 5.5 Export (`reports.export_job_csv`)

One CSV per job with columns: rank, candidate_id, name, hybrid/semantic/keyword scores,
screening score+verdict+summary, interview status+avg+verdict. UTF-8 with BOM for Excel.

### 5.6 Email (`emailer.send_email`)

1. The Email tab builds an HTML template (shortlist notification or interview invite)
   from the job + candidate and extracts the recipient from the resume
   (`extract_email`) when the box isn't typed manually.
2. **Per-account SMTP (no `.env`)** — `resolved_settings()` reads ONLY the signed-in
   account's own `email_settings` row, saved from **Email tab → ⚙️ Email settings**
   (a bubble opened from the button in the tab header). There is deliberately NO
   `.env` SMTP fallback: each account brings its own free SMTP (e.g. its own Gmail
   app password), and an account that hasn't saved a sender gets a clear "not
   configured" message plus a warning banner on the Email tab.

   The SMTP password is **encrypted at rest**: `auth.set_active_user` derives a
   per-account key from the account's stored PBKDF2 password hash (domain-separated
   via a fixed tag) and `db` encrypts the field with a nonce-seeded stream cipher
   plus an HMAC tag — a leaked per-user `.db` file cannot recover it without the
   account password hash. Google-only accounts (no password) fall back to plaintext.
   Changing the account password invalidates the stored ciphertext (the tag fails,
   the field reads empty) until the recruiter re-enters the SMTP password.
3. `send_email` connects to the resolved SMTP server via `smtplib` (STARTTLS + login
   when configured) and sends a multipart text/HTML message.
4. Success is recorded via `db.audit("email_sent", …)`; failures return a clear
   message instead of crashing the app. Costs nothing — bring your own free SMTP.

### 5.7 Live meeting interview (`live_interview.finish_live_interview`)

1. **Free meeting link** — `generate_meeting_link` creates a **Jitsi** room on demand
   (`meet.jit.si/<talentiq-…>`, no account, no time limit) — the only provider the app
   generates links for, and the link opens the meeting directly. It is sent via
   **Email → Interview invite**.
2. **Live capture** — the browser microphone (Gradio streaming audio) appends int16 PCM
   chunks to the in-memory session; a background thread transcribes every ~10 s of NEW
   audio to Groq **Whisper** (`whisper-large-v3`) and stitches a rolling transcript into
   the UI. Each chunk carries the tail of the prior transcript as a continuation hint and
   the selected language, so Whisper keeps context across chunk boundaries. Nothing is written to disk until the interview ends.
3. **Finish & evaluate** — `finish_live_interview` transcribes the remainder, saves the
   call as a WAV in the user's media folder (`data/users/<id>/media/live_*.wav`, or
   `MEDIA_DIR` globally), then `parse_qa_pairs` asks the LLM (strict JSON) to
   **auto-detect the full transcript** as speaker-labelled turns (`turns`: Interviewer/
   Candidate per utterance) and to extract the interviewer **question** / candidate
   **answer** pairs; blank entries are dropped.
4. The answers are scored by the *same* `evaluate_answers` RAG pipeline used for typed
   interviews (evidence-grounded per-question scores + overall verdict). The evaluation is
   written in the selected interview language; the transcript itself stays in its original
   spoken language.
5. Everything — recording path, transcript, turns, Q&A pairs, evaluation, score, verdict —
   is persisted to the `video_interviews` table and shown in **Live interview history**.

The typed-interview and live-interview paths share the evaluator, so scores are
comparable across both modes.

---

## 6. User Manual — End-to-End Flow

Launch: `python app.py` → open `http://127.0.0.1:7861` (auto-falls back to the next
free port if busy — the terminal prints which port was chosen).

### Signing in
- The app opens on a **login screen** with two modes: **Sign in** or **Create account**
  (email + password, min. 6 characters). Create an account to get started.
- **Continue with Google** is available once `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET`
  are set (standard redirect flow: click the button → approve on Google → the app signs
  you in automatically — no code to enter). Create a **Web application** OAuth client in
  Google Cloud Console and add your app's URL as an **Authorized redirect URI**
  (e.g. `http://localhost:7861/` locally, or your Space's `https://<user>-<space>.hf.space/`
  when deployed).
- **Data is isolated per account.** Each user sees only their own jobs, candidates,
  screenings and interviews — switching accounts shows a fresh workspace. The first
  account created automatically inherits any pre-upgrade local data.
- **Log out** (inside the 👤 profile bubble, top-right of the session bar) returns to the
  gate; signing back in restores your data. Both floating bubbles (Profile and
  Email settings) dismiss on **✕ Close** or a click anywhere outside them.
- **Profile bubble** (👤 icon in the session bar) shows your **full name** (editable — type a new name
  and hit **Save name**), the account **email**, **sign-in method** (Email & password or
  Google), and **member since**. It also has a **Log out** button and a **danger zone**:
  type **DELETE** into the confirmation box and click **Delete my account permanently**
  to erase the account and ALL of its data (user row, sessions, jobs, candidates,
  interviews, email settings, exports, media). There is no undo.
- **The session is remembered.** Your login survives page reloads, closing and reopening
  the tab, and even server restarts — a session token lives in the browser
  (`gr.BrowserState`/localStorage) and is resolved against `users.db` on page load, so
  you don't have to sign in again. Sessions **expire after `SESSION_TTL_DAYS` (default
  30 days)** — after that you simply sign in again. Logging out invalidates the token
  immediately.

### Tab 1 — Jobs

1. **Open a requisition**: type a title + paste the JD, or pick a **sample JD** (it only
   fills the form — nothing is auto-added to your lists). Optionally type a custom
   **Req. ID**; leave blank for auto-generation.
2. Click **Create job** → it appears in the **Open roles** table with Req. ID, candidate /
   shortlist / screened counts, and creation date.
3. **Focus job** dropdown drives the rest of the workspace.
4. **Delete job listings**: tick the **Select** boxes in the **Open roles** list →
   **Delete selected listings** (red button) removes them *and everything tied to them*
   — candidates, shortlists, screenings, interviews. Deletion is only ever done from
   the list — there is no dropdown-based delete.
5. **Candidate pipeline** for the focus job: tick **Select** boxes → **Remove selected
   from this job** (detaches them) or **Delete selected entirely** (permanent).

### Tab 2 — Talent pool

1. Pick the **Job listing** first — candidates ingested now belong to *that job only*.
2. **Ingest PDFs** (batch) or paste **Resume text** (+ optional name override) → ingest.
3. Edit candidates via the dropdown + editor (**Load into editor** → **Save changes**).
   Delete candidates only from the **Candidate list** below: tick the **Select** boxes →
   **Delete selected candidates** (red). The list shows only that job's candidates
   (Select · # · Name · Source · Created — no raw IDs).

### Tab 3 — Shortlist

1. **Rank selected jobs**: pick one or more jobs (multi-select) → **Rank selected jobs**
   builds an independent ranked shortlist per job (hybrid scores).
2. **Focus job shortlist & deep screen**: pick a job + candidate; **Deep-screen selected**
   runs the LLM rubric and shows a full evidence-backed report (score, verdict badges,
   per-dimension scores, gaps, interview focus).
3. **Deep-screen top N** batch-screens the top N.
4. **Top N for interview** (1–20) controls how many shortlisted candidates flow to the
   Interview tab; the Interview list syncs live.

### Tab 4 — Email (shortlist notifications & interview invites)

1. Pick a **job** + **candidate** — the recipient box auto-fills with the email found
   in the candidate's resume (editable).
2. Choose **Email type**: *Shortlist notification* or *Interview invite* (invites list
   up to 5 sample questions). Optionally add a **personal message** and an **Interview
   invite link** — when set, the invite email shows a prominent "Join the interview"
   button pointing at your meeting URL.
3. **Send email** → sent over your own SMTP server; successful sends are recorded in
   the **Recently sent emails** table and the audit log. Without SMTP configured, the
   app stays functional and shows a clear setup message.

   **Before you click Send**, a warning banner at the top of the tab tells you when
   sends would fail: a saved-but-incomplete account config (missing host or
   from-address — judged on the **saved** values, not a merged/fallback config), an
   undecryptable saved password (account password changed), or no SMTP sender at all.
4. **Email settings (per account)** — click the **⚙️ Email settings** button
   (top-right of the tab header) to open a floating settings bubble: configure
   **your own SMTP** (host — pick a common provider from the dropdown (Gmail,
   Outlook, Yahoo, iCloud, Zoho, GMX, SendGrid, Mailgun) and the port
   auto-fills, or type any custom host; then port, from address/name, username,
   password, STARTTLS)
   and **Save settings** — stored for *this account only* (each account keeps its
   own; the shared `.env` is never used). **Send test email** verifies the config
   (defaults to your account email); **Clear settings** drops it. The bubble closes
   with **✕ Close**. The saved password is encrypted at rest with a key derived
   from your account password, so it stays out of the browser and out of a leaked
   `.db` file.

### Tab 5 — Interview (Chat **or** Live Meeting)

The tab has an **Interview mode** selector (Chat vs. Live meeting) and an **Interview
language** selector (**English** / **German**) — everything below reuses the same job +
candidate pickers, and **no API key is ever asked in the UI** (the `.env` key is used).
Questions, follow-ups and evaluations are written in the selected language.

**Chat interview:**
1. Choose job + candidate (pre-filled from the top-N shortlist with PASS/FAIL/not-screened
   badges; non-PASS candidates auto-screen on start) and the **Interview language**.
2. **Start chat interview (AI suggests 10 questions)** → the AI generates **10 tailored
   questions** (shown in a panel, grounded in the JD + resume + screening gaps) and asks Q1.
3. Type each answer → **Send** (or Enter). Follow-ups appear when answers are thin. After
   Q10, a full evaluation (per-question + overall verdict) is shown and saved. The
   evaluation is built from the structured question/answer model (each answer mapped to
   its question, follow-up replies included) — no transcript re-parsing.

**Live meeting interview (free — replaces the old upload flow):**
1. Click **Generate free Jitsi meeting link** — a **Jitsi** room is created instantly
   (free, no account, no time limit) and the link opens the meeting directly. Send it via
   **Email → Interview invite**, or share it with the candidate.
2. Optionally show the **AI-suggested questions**, then click **Start live
   transcription**.
3. Press **record** on the microphone and run the call in the meeting app. The app
   captures the call audio from the browser mic and streams the transcript live (every
   ~10 s a chunk is sent to Groq Whisper — the same free engine as before).
4. **Stop & transcribe remainder** (optional) — the full transcript appears in the
   **Review & fix** box, where you can correct any words Whisper misheard.
5. **Finish & evaluate** — if you edited the transcript, the corrected text is what gets
   analyzed (no re-transcription); otherwise Groq Whisper re-transcribes the **entire
   call in one full-context pass**, which is markedly more accurate than stitching live
   chunks. The AI **auto-detects the full transcript and separates interviewer from
   candidate** (speaker-labelled turns), extracts the Q&A pairs, and the RAG-grounded
   evaluator scores each answer + gives an overall verdict. The call recording is saved
   as a WAV in the user's media folder.
6. Results: live transcript + speaker-separated transcript with Q&A + evaluation, saved to
   **Live interview history (this job)** — per-candidate, never global. The evaluation
   follows the selected **Interview language**; the transcript keeps its original spoken
   language.

### Tab 6 — History & export

1. **Filter by job** → screening + interview history tables with PASS/FAIL badges.
2. **Generate CSV report** (per selected job) → **Download CSV**. The file lands in
   `exports/` and is Excel-ready.

---

## 7. Developer Guide — Module Reference

### `app.py`
```python
port = find_free_port(int(os.getenv("PORT", "7861")))  # skips busy ports
server_name = os.getenv("GRADIO_SERVER_NAME",
    "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1")
demo.launch(server_name=server_name, server_port=port, share=...)
```
Runs `db.init_db()` then serves the pre-built `demo`. `find_free_port()` probes the
requested port with the same bind-check Gradio performs and falls back to 7862, 7863, …
if it's taken — a leftover instance no longer crashes startup, it just shifts the port.
Spaces sets `SPACE_ID`/`PORT` automatically.

### `ui.py` — the wiring hub
- **Module-level helpers** (`_job_choices`, `_candidate_choices`, `_interview_choices`,
  `_candidates_table`, `_jobs_table`, `_stats_markdown`, …) turn DB rows into Gradio
  choices/values. No raw IDs ever reach the UI.
- **`refresh_workspace(job_id)`** is the master refresh: it returns **17 outputs**
  (all shared dropdowns, tables, status markdown, KPIs). Keep the count in sync with
  `_ws_outputs` in `build_demo()`.
- **Handlers** (`on_create_job`, `on_delete_jobs`, `on_delete_candidates`,
  `on_add_resume_text`, `on_upload_pdfs`, `on_rank_multi`, `on_deep_screen`,
  `on_start_interview`, `on_chat_submit`, `on_export_csv`, …) — every one returns a
  tuple matching its `outputs=` list exactly.
- **`_JC_ACTION_OUTPUTS = [*_ws_outputs, em_cand_dd, vi_history]`** — the full workspace
  sweep refreshed by the per-job pipeline action buttons (a removed/deleted candidate
  disappears from every tab immediately). The pipeline handlers return
  `job_cand_status` + these outputs. If you add an output to `_ws_outputs`, keep this
  list — and the `_auth_outputs` wiring in `build_demo()` — in sync (both are
  count-asserted at build time).
- **`custom_css`** — the full design system (see §8).
- **`build_demo()`** — constructs the Blocks, wires every event, returns `demo`.
  `demo = build_demo()` runs at import time so `app.py` just launches it.

> ⚠️ **Counting rule of thumb:** after changing any handler's return values, run the app —
> Gradio raises at startup if `outputs=` count ≠ returned tuple length.

### `db.py`
Everything SQLite. Use the context manager pattern:
```python
with db.connect() as conn:      # rows are sqlite3.Row → dict via _row_to_dict
    ...
```
The DB file path comes from `config.DB_PATH` (env override `RECRUITER_DB_PATH`), so tests
and Spaces deployments can point it elsewhere. Migrations (`_migrate_job_req_id`,
`_migrate_candidate_job_id`, `_migrate_job_candidates`) run inside `init_db()` and are
idempotent — older databases are upgraded in place on next launch.

### `config.py`
Single source of truth for paths and LLM settings. Every path is env-overridable
(`RECRUITER_DB_PATH`, `CHROMA_DIR`, `EXPORT_DIR`, `USERS_DB_PATH`, `USER_DATA_DIR`) so a
Hugging Face Space can point storage at a persistent `/data` mount, and tests redirect
them to temp dirs. Also owns `GROQ_API_KEY`/`GROQ_MODEL`/`OLLAMA_MODEL`, the LLM retry
knobs, and the Google OAuth client id/secret (`GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET`).

### `auth.py` — accounts & per-user isolation
- **Identity** lives in one global `users.db` (emails, PBKDF2-SHA256 hashes, Google ids).
  Passwords are never stored in plain text and never logged.
- **Brute-force protection**: failed sign-ins are recorded per email and per IP in the
  `auth_attempts` table (lazily pruned; failures for non-existent emails are counted
  too, which prevents username enumeration — at the cost that someone who knows a
  victim's email can lock it for the window; the per-IP cap bounds bulk attacks). More
  than `AUTH_MAX_FAILED_ATTEMPTS` failures per email (or `AUTH_IP_MAX_FAILED_ATTEMPTS`
  per IP) within `AUTH_LOCKOUT_WINDOW` locks the account/device; a successful login
  resets the per-email counter. Registrations are capped per IP
  (`AUTH_MAX_REGISTRATIONS_PER_IP` per `AUTH_REGISTER_WINDOW`). **Per-IP caps assume
  every client has a real, distinct IP and are disabled automatically on HF Spaces**
  (`SPACE_ID` set — all users share the Space's proxy IP); the email-scoped limits
  always apply. Override with `AUTH_ENFORCE_IP_LIMITS=0/1`.
- **Sessions expire**: every session token carries an `expires_at` (`SESSION_TTL_DAYS`,)
  enforced on every resolution; expired tokens are rejected and pruned lazily.
- **Per-user data**: each account's `recruiter.db`, Chroma vectors, CSV exports and
  recorded-interview media live under `data/users/<user_id>/`. `set_active_user(user)`
  calls `db.set_active_db`, `vectorstore.set_active_chroma`, `reports.set_export_dir`
  and `video_interview.set_active_media_dir` to switch the whole app to that folder;
  `set_active_user(None)` returns to the global paths. Each account's vector store is
  **checked for an embedding-model change lazily on first access** (once per boot per
  user) — so an `EMBEDDING_MODEL` switch rebuilds every account's index, not just the
  legacy default store.
- **Google sign-in** uses the standard OAuth 2.0 *authorization-code redirect flow with
  PKCE*: clicking **Continue with Google** redirects the browser to Google's consent
  page, then back to the app (`?code=...&state=...`), where the code is exchanged for
  the id_token and the user is signed in — no code entry. The redirect lands on the
  app's page-load handler (`_on_page_load`), which mints the browser session token at
  that point, so the workspace's API calls authenticate normally. The id_token is **verified**
  (RSA signature against Google's certificates, `aud` = our client id, issuer, expiry)
  via `google-auth` before any account is created or linked. The redirect URI is derived
  automatically from the page origin, so it works locally (`http://localhost:7861/`)
  and on a deployed Space (`https://<user>-<space>.hf.space/`); **add that exact URL as
  an Authorized redirect URI** on the OAuth client. The redirect URI is **decided once at
  start time and stored with the OAuth attempt**, then reused verbatim by the token
  exchange — the callback request's own headers (Referer = accounts.google.com, often no
  usable Origin) can never produce a mismatching URI that Google would reject with an
  `invalid_grant` (seen as a false *"Google sign-in failed"*). Referer origins are also
  sanitized to scheme+host, so a leftover `?code=...&state=...` in the page URL can never
  leak into the redirect URI. Google rejection reasons (`error`/`error_description`) are
  printed to the server log for diagnosis. id_token `iat`/`exp` validation tolerates
  clock skew (default 60s via `GOOGLE_CLOCK_SKEW_SECONDS` in `.env`) so a machine clock a
  couple of seconds behind Google's servers can no longer fail every Google login with
  *"Token used too early"*. Requires a **Web application** client
  with `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` in `.env`; without them the button
  shows a setup hint instead of failing. An existing email/password account is
  automatically upgraded to also accept Google.
- **Legacy migration**: when the first account signs in after the upgrade, the existing
  single-user data (`recruiter.db` + `chroma_db/`) is copied into that account's private
  storage and a `.legacy_migrated` marker prevents any later account from inheriting it.

### `prompts.py`
All LLM prompts live here as data — the
screening, interview-evaluation, and follow-up builders — so prompt tuning never touches
pipeline code. `build_screening_user_prompt` embeds the rubric schema from `rubric.py`.
The interview builders (`build_suggest_questions_prompt`, `build_evaluation_user_prompt`,
`build_followup_user_prompt`) take a `language` parameter ("English"/"German", normalized
to those two values) and instruct the model to write questions / follow-ups / feedback in
that language.

### `llm.py`
`get_llm_client(api_key, model=None)` returns `(client, error)`. Priority: user/Groq key
→ `GROQ_API_KEY` → Ollama (if reachable). `LLMClient.chat_json` handles JSON mode and
strips ``` fences. Model override via `GROQ_MODEL` (read from `config.py`). Groq calls
retry with exponential backoff on transient errors (429/5xx) via `_retry_call`
(`LLM_MAX_ATTEMPTS` / `LLM_BASE_DELAY`), and every successful call logs token usage to
`audit_log` (`action="llm_call"`) for cost visibility.

**Hybrid routing:** high-stakes scoring (deep screening, interview evaluation) uses the
strong `GROQ_MODEL`; trivial decisions (interview follow-up detection) use
`get_fast_llm_client` → `GROQ_FAST_MODEL` — the fast model absorbs volume while the
strong model keeps final scores calibrated. Every audit `llm_call` row records which
model actually ran.

### `screening.py` / `rubric.py`
`ScreeningResult` carries score, verdict, summary, strengths/gaps/interview_focus,
evidence snippets. Rubric weights live in one place (`rubric.py`) so changing hiring
policy is a one-line edit (weights must sum to 1.0; `PASS_THRESHOLD`/`MUST_HAVE_MIN`).

### `interview.py`
`InterviewSession` is a dataclass that serializes to/from dict (Gradio `gr.State`).
`submit_answer` mutates and returns the session — the UI re-serializes it into state.
Follow-up detection is fail-open (no LLM → heuristic for very short answers). Each session
carries a `language` (English/German) chosen at start; it drives question generation,
follow-up probes, and the final evaluation, and the no-LLM fallback text is localized too.
Answers are stored as **one bucket per question** (`answers[q]` — a follow-up reply shares
its question's bucket), and the evaluation input is built from that structured model via
`_combine_answers_model` — the chat transcript is only for display, never re-parsed for
scoring.

### `vectorstore.py` / `embeddings.py` / `chunking.py` / `rerank.py` / `pdf.py`
Infrastructure wrappers. `embeddings.py` uses the multilingual
`paraphrase-multilingual-MiniLM-L12-v2` (384-dim, English + German) — switch via
`EMBEDDING_MODEL`, and `vectorstore.maybe_reindex_all()` automatically rebuilds the
index on boot when the model changed (resume_text in SQLite is the source of truth).
The boot reindex covers the legacy default store; per-account stores are checked
lazily on first login (see `auth.set_active_user`), so a model switch never leaves any
account on stale vectors.
`rerank.py` uses a multilingual cross-encoder (`RERANK_MODEL`); `RERANK_ENABLED`
(default on) is checked inside `rerank()` so free-tier Spaces can disable it to save
RAM/CPU. `chunking.py` detects German section headers too (KENNTNISSE,
BERUFSERFAHRUNG, AUSBILDUNG, …) and maps them to the canonical English sections.

### `reports.py`
`export_job_csv` writes to `exports/` with a slugified filename
(`hiring_report_<job-slug>_<timestamp>.csv`). Reads rank, screenings, and interviews to
build one row per candidate.

---

## 8. UI, Theme & Styling Guide

### Theme (`_theme()` in `ui.py`)
`gr.themes.Soft` with a custom **teal** primary palette (c50→c950), slate neutrals, and
DM Sans. Core colors are mirrored as CSS variables in `custom_css` (`:root`):

| Token | Value | Use |
| :--- | :--- | :--- |
| `--brand` / `--brand-dark` | `#0f766e` / `#115e59` | primary buttons, focus rings, active nav pill |
| `--danger` / `--danger-dark` | `#dc2626` / `#b91c1c` | delete / destructive buttons |
| `--ink` / `--ink-soft` / `--muted` | `#0f172a` / `#1e293b` / `#475569` | text hierarchy |
| `--surface` / `--line` / `--wash` | `#ffffff` / `#e2e8f0` / `#f0fdfa` | panels, borders, tinted chips |
| `--pass-*` / `--fail-*` | green / red | verdict badges |

### Button system — one consistent scheme
Every button shares the same shape (10px radius, 38px min-height, weight 600) and differs
only by *semantic* color:

- **`primary`** (solid teal) — the main action of each panel: Create job, Ingest PDFs,
  Save changes, Rank selected jobs, Deep-screen selected, Sync top N, Start interview,
  Send, Generate CSV report.
- **`secondary`** (white + hairline border) — supporting actions: Load/refresh,
  Load into editor, Deep-screen top N, Remove selected from this job, Refresh.
- **`stop`** (solid red) — destructive only: Delete selected listings, Delete selected
  entirely, Delete selected candidates.

All variants have hover lift (`translateY(-1px)`) + soft shadow, and an active press
scale. The **nav pills** reuse the brand teal for the selected tab.

### Dropdown styling (Gradio 6 specifics)
Gradio 6 renders a dropdown as `.container > .wrap > .wrap-inner` (the closed box). When
opened, the **same `.wrap`** becomes the popup backdrop around `ul.options`. In dark mode
that backdrop is solid slate — which read as a "dark navy box" around the list. The CSS
pins it white, rounds it, adds the shadow, hides the built-in `✓` span
(`.container .options .inner-item { display: none }`), tints the selected row with
`--wash`, and gives the focused box a teal ring via `.container:focus-within > .wrap`.

### Tables
- `th`/`td` are `white-space: nowrap`; each `gr.Dataframe` declares explicit
  `column_widths=` so headers like "Candidates"/"Shortlist" never wrap into
  "Can/dida/tes".
- `.table-wrap { overflow-x: auto }` — narrow screens scroll horizontally instead of
  squashing.

### Responsive breakpoints
`1100px` (container goes full-width), `900px` (panels stack, nav pills flex), `640px`
(forms go full-width, rows wrap, KPI grid → 2 columns, smaller fonts, shorter chatbot),
`480px` (tighter padding).

> ⚠️ **Critical Gradio quirk (why the mobile CSS broke once):** Gradio *prefixes* every
> custom-CSS rule with `.gradio-container.gradio-container-<ver> .contain`, and for rules
> inside `@media` blocks it keeps **only the prefixed copy**. Therefore media-query
> selectors must **not start with `.gradio-container` or `.contain`** (they'd be double-
> prefixed and never match). Prefer bare selectors (`.row`, `.form`, `.app-row`).
> The container itself is fluid via the base rule `width:100%; max-width:1180px`.

### Dark-mode handling
Gradio inherits the OS dark-mode preference and tags `<body class="dark">`, which flips
surfaces to dark navy. The CSS pins every theme surface to light values inside
`.gradio-container` (see the "Force light surfaces" block) so the app is always light
with readable text.

---

## 9. Configuration & Deployment

### Local — one click

**Windows:** double-click **`start.bat`** — on first run it creates `.env` from the
template (fill in `GROQ_API_KEY`), then it boots the app, opens the browser for you,
and live-tails the server log in its window. Close that window (or Ctrl+C) to stop the
server; your data stays on disk in the project folder (`users.db`, `chroma_db/`, `data/`).

Manual equivalent:
```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
# optional: create .env with GROQ_API_KEY="gsk_..."
python app.py        # → http://127.0.0.1:7861 (falls back to the next free port if busy)
```

> 💡 Running inside OneDrive / Drive-synced folders? If you hit intermittent
> “database is locked” errors, set `SQLITE_WAL=0` in `.env`.

### Environment variables

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | — | LLM provider (fallback: local Ollama) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Strong model for scoring (screening, evaluation) |
| `GROQ_FAST_MODEL` | `llama-3.1-8b-instant` | Cheap model for low-stakes calls (follow-up detection) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Preferred local Ollama model |
| `RERANK_ENABLED` | `1` | `0` disables the cross-encoder (saves ~470 MB RAM/CPU) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embeddings (EN + DE); switching rebuilds the vector index on boot |
| `RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual cross-encoder for reranking retrieval hits |
| `RECRUITER_DB_PATH` | `recruiter.db` (project dir) | Override the SQLite file location |
| `CHROMA_DIR` | `chroma_db/` | Override the Chroma persistence directory |
| `EXPORT_DIR` | `exports/` | Override the CSV export directory |
| `LLM_MAX_ATTEMPTS` | `4` | Groq retry count on transient errors |
| `LLM_BASE_DELAY` | `1.0` | Groq backoff base delay (seconds, doubles per retry) |
| `MEDIA_DIR` | `media/` | Folder for recorded live interviews |
| `USERS_DB_PATH` | `users.db` | Global identity store (accounts, hashes, Google ids) |
| `USER_DATA_DIR` | `data/users/` | Per-account private storage (`recruiter.db`, `chroma/`, `exports/`, `media/` per user) |
| `GOOGLE_CLIENT_ID` | — | Enables **Continue with Google** (OAuth 2.0 redirect flow with PKCE) — create a **Web application** client |
| `GOOGLE_CLIENT_SECRET` | — | Secret of the Web application Google client (required for the redirect flow) |
| `AUTH_MAX_FAILED_ATTEMPTS` | `5` | Failed sign-ins per email within the window before a lockout |
| `AUTH_IP_MAX_FAILED_ATTEMPTS` | `20` | Failed sign-ins per IP within the window before a device lockout |
| `AUTH_LOCKOUT_WINDOW` | `900` | Lockout window (seconds) for failed sign-ins |
| `AUTH_MAX_REGISTRATIONS_PER_IP` | `5` | Account creations allowed per IP per window (raise it — e.g. `200` — when running the E2E scripts repeatedly from one machine) |
| `AUTH_REGISTER_WINDOW` | `3600` | Registration window (seconds) per IP |
| `AUTH_ENFORCE_IP_LIMITS` | `1` (auto-`0` on HF Spaces) | `0` disables the per-IP caps (shared proxies); email-scoped limits always apply |
| `SESSION_TTL_DAYS` | `30` | How long a persistent login session stays valid |
| `MAX_PDF_UPLOAD_MB` | `15` | Largest accepted resume PDF (MB) — larger uploads are rejected with a clear message |
| `SQLITE_WAL` | `1` | `0` disables WAL journal mode (escape hatch for cloud-synced folders like OneDrive) |
| `PORT` | `7861` | Bind port (falls back to the next free port if taken) |
| `GRADIO_SERVER_NAME` | `127.0.0.1` (or `0.0.0.0` on Spaces) | Bind address |
| `GRADIO_SHARE` | `0` | `1` creates a temporary public share link |
| `HF_TOKEN` | — | HF write token (huggingface.co/settings/tokens) — enables the free Space backup |
| `HF_BACKUP_REPO` | — | Private dataset repo for backups, e.g. `user/talentiq-backup` (created automatically) |
| `HF_BACKUP_INTERVAL_MIN` | `30` | Minutes between backup pushes |
| `HF_BACKUP_FIRST_DELAY_MIN` | `2` | Delay after boot before the first push (lets models load first) |
| `HF_BACKUP_INCLUDE_MEDIA` | `0` | `1` also archives raw interview recordings (larger, slower pushes) |
| `HF_BACKUP_ENABLED` | `1` | `0` disables the backup feature entirely |

> **Email (SMTP) is NOT configured via `.env`** — each account saves its own sender
> from the Email tab → ⚙️ Email settings (e.g. a free Gmail app password). The
> per-account password is encrypted at rest in the per-user `recruiter.db`.

### Hugging Face Spaces (free — Gradio SDK + ZeroGPU)

The app is a plain **Gradio** app (`ui.py` builds `gr.Blocks`; `app.py` calls
`demo.launch()`), so it runs natively on HF's **Gradio SDK** — no Dockerfile
ships with the project (the Gradio SDK needs none).

> ⚠️ **2026 free-tier reality:** free personal accounts can no longer create
> **CPU Basic** Gradio Spaces — the hardware selector greys it out. The only
> free hardware for a new Gradio Space is **ZeroGPU** (up to 2 Spaces per free
> account; the account must be 30+ days old with a verified email). ZeroGPU is
> built for GPU demos, and running a CPU-only app there is *allowed but
> discouraged* — your app consumes **0 GPU quota** (quota only ticks inside
> `@spaces.GPU` functions, which this app never uses), but you wait in a GPU
> queue for no benefit. The **Docker** SDK is `Paid` — which is why no
> Dockerfile ships with this project.

**One-time setup (click-by-click):**

1. **Create the backup dataset repo** (free, private, 100 GB):
   https://huggingface.co/new-dataset → name it `talentiq-backup` → set **Private**
   → Create. (It is created automatically on first push if you skip this, but
   creating it manually avoids a permission surprise.)
2. **Create a write token:** https://huggingface.co/settings/tokens → **New token**
   → type **Write** → copy the `hf_...` value.
3. **Create the Space** at https://huggingface.co/new-space → Short description:
   `AI recruiter workspace` → SDK: **Gradio** → hardware: **ZeroGPU** → Create Space.
4. **Push the repo** to the Space (or upload the files directly):
   ```bash
   git init && git add -A && git commit -m "TalentIQ"
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
   (HF accepts either a full clone of this repo or a plain file upload; the
   Gradio runtime installs `requirements.txt` and runs `app.py` automatically.)
5. **Add Secrets** (Space Settings → Variables and secrets):
   - `GROQ_API_KEY` — your Groq key (required).
   - `HF_TOKEN` — the write token from step 2 (enables the backup).
   - `HF_BACKUP_REPO` — `<your-username>/talentiq-backup`.
6. **Optional variables** (same panel): `RERANK_ENABLED=0` (skip the ~470 MB
   cross-encoder on cold boot), `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` for
   **Continue with Google** (add `https://<your-username>-<space-name>.hf.space/`
   as an **Authorized redirect URI** on the OAuth client).

**How data survives (free):** `backup.py` runs automatically on the Space —
accounts (`users.db`) and per-account data (`data/users/`, incl. each user's
`recruiter.db`, chroma index, exports) are tarballed into
`talentiq-backup.tar.gz` and pushed to your **private dataset repo** every
`HF_BACKUP_INTERVAL_MIN` minutes (default 30; first push ~2 min after boot).
On the next boot, if the local disk is empty (a wiped/rebuilding Space), the
latest archive is downloaded and extracted BEFORE the DBs open — so accounts
and jobs come back intact. **Local data is never overwritten**: restore only
runs on a fresh disk, and existing files are never clobbered. Media
recordings are excluded by default to keep pushes fast (transcripts + evals
live in the DBs); set `HF_BACKUP_INCLUDE_MEDIA=1` to archive them too.

**Free-tier expectations (honest):** the Space sleeps after ~2 days idle; the
first visitor pays a cold boot (~1–2+ min, longer the first time while models
download to the ephemeral disk). Storage is ephemeral — but with the backup
above, a wipe is no longer data loss: the next boot restores it. If you want
24/7 with zero cold starts, run locally (`start.bat`) + Cloudflare Tunnel
instead.

### Keep the Space awake (free)

The Space only sleeps after **~2 days** idle, so a tiny ping every 30–60 min is
all it takes to never see a cold start. Two free options:

**Option A — GitHub Actions (recommended, ships with the repo).**
`.github/workflows/keepalive.yml` is a cron that GETs your Space URL every 30
minutes (`*/30 * * * *`, UTC) and treats `200`/`503` (waking/queued) as
healthy. Setup is two steps:

1. Push the repo to GitHub (see the GitHub section above/README).
2. Set **one repo variable**: GitHub → repo → **Settings → Secrets and
   variables → Actions → New repository variable** → `HF_SPACE_URL` =
   `https://<your-username>-<space-name>.hf.space` (the public URL of your
   Space). Alternatively edit the placeholder URL at the top of the workflow
   file — the run logs a warning if it is unset.

The workflow appears under the repo's **Actions** tab ("Keep Space awake") and
runs every 30 min; use **Run workflow** (manual dispatch) to test it instantly.
Cost: a few seconds per run — unlimited on public repos, ~30–50 of the 2000
free minutes/month on private ones. GitHub may delay a cron minute or two under
load; that is harmless because the Space's idle threshold is ~48h.

**Option B — UptimeRobot (no repo, no code).** A classic uptime monitor is the
same thing as a keepalive:

1. Sign up free at https://uptimerobot.com (up to 50 monitors on the free tier).
2. **Add New Monitor** → type **HTTP(s)** → Friendly Name: `TalentIQ Space` →
   URL: `https://<your-username>-<space-name>.hf.space` → select the 5-minute
   interval (free tier default) → **Create Monitor**.

It now pings the Space every 5 minutes (even more aggressive than the cron)
and emails you on downtime — a bonus alert channel on top of the keepalive
itself. The two options can run together with no conflict.

### Cloudflare Tunnel (optional — share your local instance)

Want to open the app to a phone or another person without deploying? **`start-tunnel.bat`**
wraps [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)'s
free quick tunnel — no Cloudflare account or credit card needed:

1. Double-click **`start.bat`** and wait for the app to open in the browser.
2. Double-click **`start-tunnel.bat`** — it detects the running app's port, finds
   cloudflared (downloading it once if missing), and prints a link like
   `https://<random>.trycloudflare.com`.
3. Share that link. Press **Ctrl+C** in the tunnel window to close it (the app keeps running).

Facts to know:

- **Tunnel ≠ storage.** Cloudflare only forwards requests — your data never leaves your
  machine and is **not** deleted when you shut down. Powering off just takes the site
  offline until the next `start.bat`.
- The link is **new each time** (quick tunnels are random). For a fixed URL, set up a
  named tunnel + your own domain on Cloudflare's free plan.
- Anyone with the link sees the **sign-in page** — accounts are still protected by the
  auth gate and IP rate-limits, but don't post the link publicly.

---

## 10. Troubleshooting & Known Quirks

| Symptom | Cause / Fix |
| :--- | :--- |
| "No LLM available" | Set `GROQ_API_KEY` in `.env` or start Ollama (`ollama serve`). |
| Slow first screen | The embedding model (~80 MB) loads lazily on first ingest/rank. |
| Tables empty / 0-height | An old CSS rule hid the "Drop CSV" button that wraps Dataframes — removed. Restart the server to pick up the current `ui.py`. |
| Dark navy dropdown box | OS dark mode + Gradio's `.wrap` backdrop — handled by the dropdown CSS block (§8). |
| My CSS media rules "don't work" | See the CSS-prefixing quirk in §8 — media selectors must not start with `.gradio-container`. |
| Intermittent "database is locked" on a cloud-synced folder (OneDrive) | WAL's `-wal`/`-shm` sidecars can conflict with sync. Set `SQLITE_WAL=0` in `.env` to fall back to the default journal mode (a little concurrency headroom is lost). |
| Duplicate candidates on re-ingest | Stable IDs hash the resume — same resume updates the same candidate. |
| Req. IDs look wrong | Custom IDs must be unique; auto IDs continue from the highest existing. |
| Env var seems ignored | `config.py` reads env vars at startup — restart the app after changing `RECRUITER_DB_PATH`, `CHROMA_DIR`, `GROQ_MODEL`, etc. |
| Google button is disabled | Add `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` (a **Web application** client) to `.env` and restart — see §9. |
| Google shows “redirect_uri_mismatch” | Add the app's exact URL as an **Authorized redirect URI** on the OAuth client (e.g. `http://localhost:7861/` locally, `https://<user>-<space>.hf.space/` deployed), then restart. |
| Google shows “client type” error | The OAuth client must be **Web application**, not “TVs and Limited Input devices” — create a Web application client and update `.env`. |
| I see another account's data | Impossible by design — each account uses its own DB/vectors. If it looks wrong, log out and sign in again; the workspace fully refreshes per user. |
| Where did my old data go? | The first account to sign in after the upgrade inherits the legacy `recruiter.db`/`chroma_db/` data automatically (one-time). Create your account before any other user to claim it. |
| Port already in use | `app.py` now auto-falls back to the next free port (7862, 7863, …) and prints which one it picked. To force a port: `$env:PORT="7862"` then `python app.py`. Leftover `py app.py` processes: `taskkill /F /IM python.exe`. |
| Two instances running | The first instance takes 7861 and a second one auto-falls back to 7862; both run side-by-side and share `recruiter.db` — SQLite handles it, but restart both after code changes. |

---

## 11. Status & Next Steps

Everything in the original roadmap is now **shipped**: FastAPI-style bootstrapping
was replaced by a production-friendly Gradio app with accounts, per-user data
isolation, persistent sessions, Docker packaging, and one-click Hugging Face
Spaces deployment. The architecture remains ready for a REST layer — all logic
is decoupled from the UI via `db.py`/`ranking.py`/`screening.py`.

**Automated end-to-end verification** — two scripts drive a booted app the way a
browser does and print a PASS/FAIL summary:

- `tests/e2e_runtime_verify.py` — login gate → API auth gate (anonymous calls
  rejected) → create job → ingest a resume → rank the shortlist → the live-interview
  pipeline with REAL Groq Whisper + LLM evaluation → media persistence → WAL-mode
  check (27 checks). Needs `GROQ_API_KEY` and a booted app (point a non-default port
  with `E2E_APP=http://127.0.0.1:<port>`).
- `tests/e2e_google_flow.py` — login gate + Google OAuth start/callback probes
  (302 redirect, PKCE params, bad-callback handling) + account isolation between two
  users (18 checks). Needs only a booted app.

Both create throwaway `e2e-…@example.com` accounts; the runtime script also
synthesizes a real speech WAV (`_interview_e2e.wav`) — all safe to delete afterwards.

> ⚠️ **Registration cap:** account creation is rate-limited per IP
> (`AUTH_MAX_REGISTRATIONS_PER_IP`, default 5/hour). Running the E2E scripts
> repeatedly from one machine quickly exhausts that quota (sign-in returns *"Too
> many accounts created from this device"*). Boot the app with
> `AUTH_MAX_REGISTRATIONS_PER_IP=200` (or `AUTH_ENFORCE_IP_LIMITS=0`) for a test
> session — and restart between code changes (a stale instance keeps the old cap).

Ideas for the next iteration:
- REST API over `db.py`/`ranking.py`/`screening.py` (FastAPI) with token auth.
- Role-based access (recruiter / hiring-manager) and PII redaction on exports.
- An evaluation harness that replays recorded interviews against grading rubrics.
- CI: run `pytest` + `pyright` on every push (see `requirements-dev.txt`).

---

*Generated with the project — keep this manual updated whenever the pipeline or UI changes.*
