# ================================================================
# 🗄️ ChromaDB Vector Store — Persist & Search (+ optional rerank)
# ================================================================

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress

import chromadb

import config
import embeddings as emb
import rerank as rr

# Default persistent storage directory (overridable via CHROMA_DIR). The
# ACTIVE dir is per-thread so concurrent requests can't swap each other's
# vector store (see auth.user_scope).
_CHROMA_DIR = str(config.CHROMA_DIR)
_thread = threading.local()

# Serializes Chroma client/collection operations. Clients are per-thread
# (each thread owns a PersistentClient on its own user's dir), so two clients
# can otherwise write the same Chroma SQLite concurrently and hit
# "database is locked" — background ingest threads write while request
# threads search. Embeddings are computed OUTSIDE this lock; only the quick
# client ops (get/add/delete/query/metadata) are serialized.
_chroma_lock = threading.RLock()

_RESUME_COLLECTION = "resumes"


def _active_chroma_dir() -> str:
    return getattr(_thread, "chroma_dir", None) or _CHROMA_DIR


def set_active_chroma(chroma_dir: str | None) -> None:
    """Point THIS THREAD's vector store at a specific Chroma directory
    (per-user isolation). Closes any cached client first (Chroma keeps its
    sqlite file open, so switching users or deleting a user's data directory
    would otherwise leave the old dir locked on Windows), then resets the
    thread's cache so the next call opens the new dir. Passing None clears
    the override (module default).
    """
    close_active_chroma()
    _thread.chroma_dir = str(chroma_dir) if chroma_dir else None
    _thread.client = None


def close_active_chroma() -> None:
    """Close THIS THREAD's Chroma client, releasing its file handles.

    Used before deleting a user's data directory: Chroma's persistent client
    keeps chroma.sqlite3 open on Windows, which would make the directory
    undeletable. After closing, the client is dropped so the next access
    reopens it against the current active dir.
    """
    client = getattr(_thread, "client", None)
    if client is not None:
        with suppress(Exception):
            client.close()
        _thread.client = None

# Fetch extra candidates from the vector store, then rerank down to top_k
_RETRIEVE_MULTIPLIER = 3
_RETRIEVE_MIN = 8


def _get_client() -> chromadb.ClientAPI:
    """Lazy ChromaDB persistent client — one per thread (its own dir)."""
    client = getattr(_thread, "client", None)
    if client is None:
        client = chromadb.PersistentClient(path=_active_chroma_dir())
        _thread.client = client
    return client


def _get_resume_collection() -> chromadb.Collection:
    """Get or create the resumes collection.

    The collection records which embedding model produced its vectors so a
    model switch can be detected and the index rebuilt (maybe_reindex_all).
    """
    with _chroma_lock:
        client = _get_client()
        return client.get_or_create_collection(
            name=_RESUME_COLLECTION,
            metadata={"hnsw:space": "cosine", "embedding_model": emb.model_name()},
        )


def index_resume(candidate_id: str, chunks: list[dict], force: bool = False) -> int:
    """Upsert resume chunks into ChromaDB.

    Skips re-embedding when the candidate already has the same number of
    chunks indexed (stable content-hash IDs make re-screens cheap).

    Args:
        candidate_id: Unique candidate identifier (prefer content hash).
        chunks: List of dicts from chunking.chunk_resume().
        force: If True, always re-index even when counts match.

    Returns:
        Number of chunks indexed (0 if skipped as already present).
    """
    if not chunks:
        return 0

    if not force and get_candidate_count(candidate_id) == len(chunks):
        return 0

    collection = _get_resume_collection()

    # Clear existing data for this candidate (re-index on change)
    clear_candidate(candidate_id)

    ids = [f"{candidate_id}_{c['index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "candidate_id": candidate_id,
            "section": c["section"],
            "index": c["index"],
        }
        for c in chunks
    ]

    # Embedding is the slow step — compute it OUTSIDE the Chroma lock so a
    # background ingest never blocks searches or other writes.
    vectors = emb.embed_texts(documents)

    with _chroma_lock:
        collection.add(
            ids=ids,
            # chromadb 1.5's stubs type these as numpy/typed arrays while our
            # model returns plain list[list[float]] — the runtime accepts lists.
            embeddings=vectors,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    return len(chunks)


def search_resume(
    query: str,
    candidate_id: str,
    top_k: int = 5,
    use_rerank: bool = True,
) -> list[dict]:
    """Vector search resume chunks for a specific candidate.

    Retrieves a wider candidate set from Chroma, then optionally
    reranks with a cross-encoder before returning top_k.

    Returns list of dicts: {"text", "section", "score"}
    """
    collection = _get_resume_collection()
    query_vec = emb.embed_single(query)

    fetch_k = max(top_k * _RETRIEVE_MULTIPLIER, _RETRIEVE_MIN) if use_rerank else top_k

    # Cap fetch_k by how many chunks this candidate actually has
    available = get_candidate_count(candidate_id)
    if available > 0:
        fetch_k = min(fetch_k, available)

    with _chroma_lock:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=max(fetch_k, 1),
            where={"candidate_id": candidate_id},
        )

    hits: list[dict] = []
    if results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists, strict=False):
            hits.append(
                {
                    "text": doc,
                    "section": meta.get("section", "UNKNOWN"),
                    "score": round(1.0 - dist, 4),  # cosine distance → similarity
                }
            )

    if use_rerank and hits:
        return rr.rerank(query, hits, top_k=top_k)

    return hits[:top_k]


def clear_candidate(candidate_id: str) -> None:
    """Remove all chunks for a candidate from the store."""
    with _chroma_lock:
        collection = _get_resume_collection()
        try:
            existing = collection.get(where={"candidate_id": candidate_id})
            if existing and existing["ids"]:
                collection.delete(ids=existing["ids"])
        except Exception:
            pass


def get_candidate_count(candidate_id: str) -> int:
    """Return number of indexed chunks for a candidate."""
    with _chroma_lock:
        collection = _get_resume_collection()
        try:
            existing = collection.get(where={"candidate_id": candidate_id})
            return len(existing["ids"]) if existing and existing["ids"] else 0
        except Exception:
            return 0


def maybe_reindex_all(db_path: str | None = None, force: bool = False) -> int:
    """Re-embed every indexed candidate when the embedding model changed.

    Embeddings from a different model live in a different vector space and are
    meaningless for similarity search, so after an EMBEDDING_MODEL switch the
    whole index must be rebuilt from the stored resume_text (SQLite is the
    source of truth — the vector index is fully regenerable).

    Returns the number of candidates re-indexed (0 = already in sync).
    """
    try:
        collection = _get_resume_collection()
    except Exception:
        return 0

    tag = emb.model_name()
    current = (collection.metadata or {}).get("embedding_model") if collection.metadata else None
    if not force and current == tag:
        return 0

    # One-migrator-at-a-time guard: two instances booting together (e.g. the
    # 7860 + 7861 dev setup) would race on the Chroma store. A stale lock
    # (>15 min, e.g. from a crash) is stolen so recovery stays possible.
    lock_path = os.path.join(_active_chroma_dir(), ".reindex.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        try:
            stale = (time.time() - os.path.getmtime(lock_path)) > 900
        except OSError:
            stale = False
        if not stale:
            return 0
        try:
            os.remove(lock_path)
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except OSError:
            return 0

    import chunking
    import db as _db

    count = 0
    try:
        for c in _db.list_candidates(db_path=db_path):
            full = _db.get_candidate(c["id"], db_path)
            if not full or not (full.get("resume_text") or "").strip():
                continue
            try:
                chunks = chunking.chunk_resume(full["resume_text"], full["id"])
                if chunks:
                    index_resume(full["id"], chunks, force=True)
                    count += 1
            except Exception:
                continue

        # hnsw:space was set at creation and cannot be changed afterwards —
        # Chroma raises if it appears in a later modify() call.
        with _chroma_lock, suppress(Exception):
            collection.modify(metadata={"embedding_model": tag})
    finally:
        with suppress(OSError):
            os.remove(lock_path)
    return count
