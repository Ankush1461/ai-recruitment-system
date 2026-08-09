# ================================================================
# 💾 Hugging Face backup — free data survival for Spaces
# ================================================================
# Free Hugging Face Spaces have ephemeral disks: accounts (users.db) and
# per-user data (data/users/*) are wiped whenever the Space sleeps or is
# rebuilt. This module tars the important files and pushes them to a PRIVATE
# dataset repo (free, 100 GB for everyone) on a timer, and restores them on
# boot when the local disk is empty (a wiped Space) — so a rebuilt Space
# comes back with every account, job, candidate and interview intact.
#
# Enable with two env vars on the Space (Settings → Variables and secrets):
#   HF_TOKEN          — a Hugging Face write token (huggingface.co/settings/tokens)
#   HF_BACKUP_REPO    — "your-username/talentiq-backup" (created automatically)
# Backups run every HF_BACKUP_INTERVAL_MIN minutes (default 30).
#
# Everything fails open: without the vars, or on any error, the app runs
# exactly as before — backup is best-effort and never blocks a request.

from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import config

_BASE = Path(__file__).resolve().parent
_ARCHIVE_NAME = "talentiq-backup.tar.gz"
# A path -> original-location map stored inside the archive so a restore
# puts every file back where the app actually reads it — even when storage
# was env-overridden to a non-default location (e.g. a /data mount).
_MANIFEST_NAME = ".talentiq-backup-manifest.json"


def enabled() -> bool:
    """True when the backup env vars are configured (HF_TOKEN + repo + flag)."""
    return bool(
        config.HF_BACKUP_ENABLED
        and config.HF_TOKEN
        and config.HF_BACKUP_REPO
    )


def _include_paths() -> list[Path]:
    """The files/folders that make up the whole app state.

    users.db (global identity store) + data/users/ (per-account recruiter.db,
    chroma, exports, media) + the legacy single-user recruiter.db and root
    chroma_db/ (migrated into the first account on sign-in). Missing paths
    are skipped during archiving.
    """
    return [
        config.USERS_DB_PATH,
        config.USER_DATA_DIR,
        config.DB_PATH,
        config.CHROMA_DIR,
    ]


def _skip_file(path: Path) -> bool:
    """True for files that must never be archived.

    SQLite WAL/shm sidecars (the main .db is checkpointed first), lock files
    and Python caches are noise. Media recordings (up to 500 MB each) are
    excluded by default to keep the periodic push fast — transcripts and
    evaluations live in the DBs, so nothing of analytical value is lost;
    set HF_BACKUP_INCLUDE_MEDIA=1 to store the raw recordings too.
    """
    name = path.name
    return name.endswith(("-wal", "-shm", ".lock")) or (
        not config.HF_BACKUP_INCLUDE_MEDIA and "media" in path.parts
    )


def _iter_files(paths: list[Path]):
    """Yield every archive-eligible file under the given paths."""
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            if not _skip_file(root):
                yield root
        else:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for name in filenames:
                    fp = Path(dirpath) / name
                    if not _skip_file(fp):
                        yield fp


_SQLITE_MAGIC = b"SQLite format 3\x00"


def _checkpoint(path: Path) -> None:
    """Flush WAL sidecars into the main .db before it is archived.

    Best-effort: a checkpoint just makes the snapshot safer; on failure the
    main file is archived as-is (SQLite still recovers the WAL on open).
    Only real SQLite files are touched (magic header) — and the connection
    is closed explicitly so Windows never keeps a stale handle on the file
    (which would block a later delete/rename).
    """
    try:
        with path.open("rb") as f:
            if f.read(16) != _SQLITE_MAGIC:
                return
        conn = sqlite3.connect(str(path), timeout=5)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except Exception:
        pass


def _planned_files() -> list[tuple[str, Path]]:
    """(arcname, source) pairs for every archive-eligible file.

    Files under the project root use a project-relative arcname; files at
    env-overridden locations (e.g. /data/users.db) get a unique flattened
    name. The original absolute path is recorded in the manifest so restore
    can put each file back exactly where the app reads it.
    """
    plans: list[tuple[str, Path]] = []
    used: set[str] = set()
    for fp in _iter_files(_include_paths()):
        try:
            arcname = fp.relative_to(_BASE)
        except ValueError:
            flattened = "__" + str(fp).replace("\\", "/").replace("/", "_").replace(":", "_")
            arcname = Path(flattened)
        name = str(arcname)
        if name in used:  # two files flattened to the same name — disambiguate
            name = f"{name[:200]}_{abs(hash(str(fp)))}"
        used.add(name)
        plans.append((name, fp))
    return plans


def _manifest_payload(plans: list[tuple[str, Path]]) -> bytes:
    """JSON bytes: arcname -> original absolute path, for the archive."""
    return json.dumps({name: str(src) for name, src in plans}, indent=0).encode("utf-8")


def build_archive(dest: Path | None = None) -> Path:
    """Tar + gzip all app state into `dest` (a temp file when omitted).

    Each .db is WAL-checkpointed first so the archived main file is current.
    A JSON manifest (arcname → original path) is embedded so a restore can
    place every file back at its configured location, even when storage was
    env-overridden away from the project root.
    """
    dest = dest or Path(tempfile.mkdtemp(prefix="talentiq-bk-")) / _ARCHIVE_NAME
    paths = _include_paths()
    # Checkpoint every sqlite file (files named *.db, wherever they live).
    for root in paths:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".db":
            _checkpoint(root)
        elif root.is_dir():
            for db in root.rglob("*.db"):
                _checkpoint(db)
    plans = _planned_files()
    manifest = _manifest_payload(plans)
    try:
        with tarfile.open(dest, "w:gz") as tar:
            info = tarfile.TarInfo(_MANIFEST_NAME)
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest))
            for name, fp in plans:
                tar.add(str(fp), arcname=name, recursive=False)
    except Exception:
        # Never leave a partial archive where a later restore could trust it.
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise
    return dest


def _upload_archive(archive: Path) -> str:
    """Push the tarball to the private dataset repo (created on demand)."""
    from huggingface_hub import HfApi

    api = HfApi(token=config.HF_TOKEN)
    api.create_repo(
        repo_id=config.HF_BACKUP_REPO,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=str(archive),
        path_in_repo=_ARCHIVE_NAME,
        repo_id=config.HF_BACKUP_REPO,
        repo_type="dataset",
    )
    return config.HF_BACKUP_REPO


def push_backup() -> str | None:
    """Build and upload a snapshot. Returns the repo id, or None when
    disabled or on failure (fail-open — never raise into the caller)."""
    if not enabled():
        return None
    try:
        archive = build_archive()
        try:
            repo = _upload_archive(archive)
            print(
                f"[backup] pushed {_ARCHIVE_NAME} ({archive.stat().st_size / 1e6:.1f} MB) "
                f"→ {repo}",
                flush=True,
            )
            return repo
        finally:
            archive.unlink(missing_ok=True)
    except Exception as e:
        print(f"[backup] push failed (will retry): {type(e).__name__}: {e}", flush=True)
        return None


def local_has_data() -> bool:
    """True when this disk already holds app state.

    Used to decide whether a restore is safe: we NEVER overwrite existing
    local data with an older backup — restore only happens on a fresh/wiped
    disk (a rebuilt Space), where these files are absent.
    """
    for path in (config.USERS_DB_PATH, config.DB_PATH):
        if path.exists() and path.stat().st_size > 0:
            return True
    ud = config.USER_DATA_DIR
    return bool(ud.is_dir() and any(ud.iterdir()))


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Archive members that are safe to extract.

    Rejects absolute/`..` traversal paths, symlinks and hardlinks (a link
    could point anywhere, and writing through it would escape the restore
    root), the embedded manifest (handled separately), and any member whose
    destination already exists with content (never clobber local data).
    """
    keep: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        name = member.name
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            continue
        if member.issym() or member.islnk():
            continue
        if name == _MANIFEST_NAME:
            continue
        dest = _BASE / name
        if dest.exists() and dest.stat().st_size > 0:
            continue  # never clobber existing local data
        keep.append(member)
    return keep


def _read_manifest(tar: tarfile.TarFile) -> dict[str, str]:
    """arcname -> original absolute path, from the embedded manifest.

    Returns {} when the archive predates manifests (legacy backups restore
    under the project root, which matches their original layout).
    """
    try:
        member = tar.getmember(_MANIFEST_NAME)
        raw = tar.extractfile(member)
        if raw is None:
            return {}
        data = json.loads(raw.read().decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except (KeyError, ValueError, json.JSONDecodeError):
        return {}


def _restore_destination(arcname: str, manifest: dict[str, str]) -> Path:
    """Where a restored file must land.

    Prefers the manifest's recorded original path — honored only when it is
    the project root or one of the configured data locations, so a tampered
    manifest can never write outside the app's own data areas. Falls back to
    the project-relative path (legacy archives without a manifest, or a
    backup built on a different machine), which matches the default layout.
    """
    allowed_roots = [_BASE, *_include_paths()]
    if arcname in manifest:
        src_path = Path(manifest[arcname])
        # Compare RESOLVED paths: a lexical check would let `..` in the
        # recorded path (e.g. <root>/../../etc/pwned) sneak inside the
        # allowed area even though it resolves far outside it.
        if src_path.is_absolute():
            resolved = src_path.resolve()
            if any(
                resolved == root.resolve() or root.resolve() in resolved.parents
                for root in allowed_roots
            ):
                return src_path
    return _BASE / arcname


def _extract_members(staging: Path, tar: tarfile.TarFile, members: list[tarfile.TarInfo]) -> None:
    """Extract vetted members into a staging dir (never directly into the
    project), honoring the `filter=` API where the Python version has it."""
    try:
        tar.extractall(staging, members=members, filter="data")
    except TypeError:
        # Python < 3.12 has no `filter` parameter — members are already
        # vetted by _safe_members (no traversal, no links, no clobber).
        tar.extractall(staging, members=members)


def restore_if_needed() -> bool:
    """Download the latest backup and restore it when the local disk is empty.

    Files are first extracted to a staging dir and vetted against the
    embedded path manifest; each file then moves to its original configured
    location (or the project root for legacy archives). Returns True when
    data was restored. Disabled (no env vars), a non-empty local disk, or
    any failure returns False — the app boots normally either way.
    """
    if not enabled():
        return False
    if local_has_data():
        print("[backup] local data present — skipping restore", flush=True)
        return False
    staging = Path(tempfile.mkdtemp(prefix="talentiq-restore-"))
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=config.HF_BACKUP_REPO,
            filename=_ARCHIVE_NAME,
            repo_type="dataset",
            token=config.HF_TOKEN,
        )
        restored = 0
        with tarfile.open(local, "r:gz") as tar:
            members = _safe_members(tar)
            manifest = _read_manifest(tar)
            if not members:
                print("[backup] backup archive is empty — nothing to restore", flush=True)
                return False
            _extract_members(staging, tar, members)
        for src in staging.rglob("*"):
            if not src.is_file() or src.name == _MANIFEST_NAME:
                continue
            arcname = str(src.relative_to(staging))
            dest = _restore_destination(arcname, manifest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size > 0:
                continue  # never clobber existing local data
            shutil.move(str(src), str(dest))
            restored += 1
        print(f"[backup] restored {restored} file(s) from {config.HF_BACKUP_REPO}", flush=True)
        return restored > 0
    except Exception as e:
        print(f"[backup] restore skipped: {type(e).__name__}: {e}", flush=True)
        return False
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _backup_loop() -> None:
    """Periodic push: once shortly after boot, then every interval."""
    time.sleep(max(1, config.HF_BACKUP_FIRST_DELAY_MIN) * 60)
    push_backup()
    while True:
        time.sleep(max(1, config.HF_BACKUP_INTERVAL_MIN) * 60)
        push_backup()


def start_backup_timer() -> threading.Thread | None:
    """Start the background backup loop (no-op when disabled)."""
    if not enabled():
        return None
    thread = threading.Thread(target=_backup_loop, name="hf-backup", daemon=True)
    thread.start()
    print(
        f"[backup] timer started — pushing to {config.HF_BACKUP_REPO} every "
        f"{config.HF_BACKUP_INTERVAL_MIN} min",
        flush=True,
    )
    return thread
