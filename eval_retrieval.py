# ================================================================
# 🎯 Offline Retrieval Evaluation — measure search quality (no LLM)
# ================================================================
"""Offline retrieval evaluation harness for TalentIQ.

Answers the question every ML interview asks: *"how do you know your
retrieval works?"* Instead of anecdote, this measures it.

It runs the production retrieval components — section-aware chunking,
sentence-transformers embeddings, cosine search and the cross-encoder
reranker — over a small labeled dataset of resumes ↔ job-description
queries, and reports standard information-retrieval metrics **with and
without** the reranker so you can see exactly what it buys.

    python eval_retrieval.py                 # both variants, k=5
    python eval_retrieval.py --k 3           # top-3 metrics
    python eval_retrieval.py --no-rerank     # vector baseline only
    python eval_retrieval.py --verbose       # per-query breakdown

No LLM API calls and no Chroma persistence — purely local models.
The first run downloads the embedding model (~120 MB) and, unless
``--no-rerank`` is passed, the cross-encoder (~470 MB); both are cached
in ``~/.cache/huggingface`` afterwards. Set ``RERANK_ENABLED=0`` to see
what the app looks like with reranking disabled.

Design notes:
    * Document-level metrics: chunk hits are deduplicated to the resume
      (best chunk score wins) before recall@k / MRR / NDCG@k are scored,
      because the product ranks *candidates*, not chunks.
    * The rerank path mirrors production ``vectorstore.search_resume``:
      fetch ``max(k * 3, 8)`` chunks by cosine, rerank down to ``k``.
    * ``run_eval`` takes injectable ``embedder``/``reranker`` callables
      so tests run fully offline with deterministic fakes.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from collections.abc import Callable

import numpy as np

import chunking
import config
import embeddings as emb
import rerank as rr

Embedder = Callable[[list[str]], list[list[float]]]
Reranker = Callable[[str, list[dict], int], list[dict]]

# Mirrors production's fetch-then-rerank pool (vectorstore._RETRIEVE_MULTIPLIER
# and _RETRIEVE_MIN) so the evaluated path matches the shipped one.
_POOL_MULTIPLIER = 3
_POOL_MIN = 8


# ---------------------------------------------------------------------------
# 📚 Labeled evaluation dataset
# ---------------------------------------------------------------------------
# Small hand-labeled corpus: each query lists the resume ids that genuinely
# match the role. Realistic resumes (proper section headers so chunking
# behaves as in production), one German resume to cover the multilingual path.

EVAL_CASES: dict = {
    "resumes": {
        "alice_ds": """SUMMARY
Senior data scientist with 6 years of experience in statistical modeling, experimentation and product analytics. Built and shipped models that improved conversion and retention.

SKILLS
Python, pandas, scikit-learn, statistics, A/B testing, SQL, Tableau, causal inference, experimentation platforms, dashboarding

EXPERIENCE
Led experimentation for a subscription platform: designed and analyzed A/B tests, powered decisions with causal inference and Bayesian modeling. Built a churn prediction model in scikit-learn with feature engineering in pandas, lifting retention by 11%. Wrote production SQL queries against a data warehouse and shipped weekly dashboards in Tableau for leadership.

EDUCATION
M.S. Statistics, 2017""",
        "bob_ml": """SUMMARY
Machine learning engineer focused on taking models from notebooks to production: training pipelines, deployment, monitoring and the infrastructure that surrounds them.

SKILLS
PyTorch, TensorFlow, Docker, Kubernetes, MLOps, CI/CD, feature stores, model monitoring, model registry, GPU training

EXPERIENCE
Scaled a recommender system to serve 2M daily users: built the training pipeline in PyTorch, deployed behind a Kubernetes inference service and added drift monitoring. Introduced a feature store shared between training and serving, cutting training time 40%. Owned the CI/CD pipeline for model releases, including automated validation and rollback.

PROJECTS
Serverless model serving on AWS Lambda with autoscaling and cold-start tuning.""",
        "carol_frontend": """SUMMARY
Frontend engineer who turns complex products into fast, accessible interfaces. 5 years building and maintaining design systems and web applications.

SKILLS
React, TypeScript, Next.js, CSS, accessibility, web performance, component testing, design systems

EXPERIENCE
Built and maintained a design system used by 12 teams, cutting UI development time by a third. Rebuilt the company storefront in Next.js, improving Lighthouse performance scores from 41 to 92. Introduced component testing with automated accessibility checks into the CI pipeline.

EDUCATION
B.S. Computer Science, 2019""",
        "dave_backend": """SUMMARY
Backend engineer with 7 years designing reliable, scalable services in Python and Go. Comfortable across databases, message queues and distributed systems.

SKILLS
Python, FastAPI, PostgreSQL, Redis, Kafka, microservices, REST, gRPC, distributed systems

EXPERIENCE
Designed the payments service handling 40k requests per minute, built with FastAPI and PostgreSQL with optimistic locking. Built an event-driven order pipeline on Kafka, decoupling inventory and fulfillment. Reduced p99 latency from 900ms to 120ms by profiling and indexing.

EDUCATION
B.Tech Computer Engineering, 2016""",
        "eve_devops": """SUMMARY
Platform and DevOps engineer automating infrastructure for reliable, observable systems. 6 years with Kubernetes, cloud and CI/CD at scale.

SKILLS
Kubernetes, Terraform, AWS, CI/CD, Prometheus, Grafana, observability, incident response, Docker

EXPERIENCE
Operated a multi-cluster Kubernetes platform for 60+ services, standardizing on Terraform-managed infrastructure in AWS. Built centralized observability with Prometheus and Grafana, reducing mean time to detection by 70%. Led incident response and on-call rotations; wrote postmortems and automation to prevent recurrence.""",
        "frank_nlp": """SUMMARY
NLP researcher and engineer specializing in large language models, retrieval-augmented generation and information extraction. Publications in top venues.

SKILLS
transformers, BERT, fine-tuning, RAG, embeddings, spaCy, NER, large language models, LLM evaluation, vector databases

EXPERIENCE
Built a retrieval-augmented question answering system over 2M internal documents: fine-tuned a BERT reader, embedded chunks and served them from a vector database. Shipped an NER pipeline in spaCy extracting entities from legal contracts. Evaluated LLM output with human-labeled test sets and reported precision and recall.

PUBLICATIONS
Fine-grained entity extraction with transfer learning (ACL 2023)""",
        "greta_data_eng": """SUMMARY
Data engineer building reliable, scalable data pipelines and warehouses. 5 years across streaming, batch and analytics infrastructure.

SKILLS
Spark, Airflow, dbt, BigQuery, ETL, data warehouse, Delta Lake, SQL, data quality

EXPERIENCE
Migrated a legacy warehouse to BigQuery with dbt, modeling 400+ tables used by the analytics team. Built streaming ingestion with Spark and orchestrated 80+ daily jobs in Airflow with full lineage and alerting. Introduced data quality tests that catch schema drift before it reaches dashboards.""",
        "hans_mlops_de": """PROFIL
Maschinenlern-Ingenieur mit 6 Jahren Erfahrung in der Entwicklung und dem Betrieb von ML-Systemen in Produktion. Schwerpunkt auf MLOps, Modell-Deployment und skalierbaren Trainings-Pipelines.

KENNTNISSE
PyTorch, TensorFlow, Kubernetes, Docker, CI/CD, AWS, ML-Pipelines, Feature-Store, Modell-Monitoring

BERUFSERFAHRUNG
Trainierte und deployte KI-Modelle für Empfehlungssysteme auf einem GPU-Cluster. Aufbau eines Feature-Stores für Training und Serving. Automatisierte Modell-Releases mit CI/CD inklusive Validierung und Rollback. Monitoring und Drift-Erkennung in Kubernetes.

PROJEKTE
Serverloses Modell-Serving auf AWS mit Autoscaling.

AUSBILDUNG
M.Sc. Informatik, 2018""",
    },
    "queries": [
        {
            "id": "q1",
            "text": "Senior Data Scientist — Python, scikit-learn, statistical analysis, A/B testing, SQL, dashboarding",
            "relevant": ["alice_ds"],
        },
        {
            "id": "q2",
            "text": "Machine Learning Engineer — PyTorch, model deployment, MLOps, Docker, Kubernetes, CI/CD pipelines",
            "relevant": ["bob_ml", "hans_mlops_de"],
        },
        {
            "id": "q3",
            "text": "Frontend Engineer — React, TypeScript, Next.js, web performance, accessibility, component testing",
            "relevant": ["carol_frontend"],
        },
        {
            "id": "q4",
            "text": "Backend Engineer — Python, FastAPI, PostgreSQL, microservices, message queues (Kafka)",
            "relevant": ["dave_backend"],
        },
        {
            "id": "q5",
            "text": "DevOps / Platform Engineer — Kubernetes, Terraform, AWS, observability, Prometheus, incident response",
            "relevant": ["eve_devops", "hans_mlops_de"],
        },
        {
            "id": "q6",
            "text": "NLP / LLM Engineer — transformers, fine-tuning, retrieval-augmented generation, information extraction",
            "relevant": ["frank_nlp"],
        },
        {
            "id": "q7",
            "text": "Data Engineer — Spark, Airflow, dbt, ETL pipelines, data warehousing, BigQuery",
            "relevant": ["greta_data_eng"],
        },
        {
            "id": "q8",
            "text": "Data Analyst — Excel, Tableau, SQL, dashboards, stakeholder reporting",
            "relevant": ["alice_ds"],
        },
    ],
}


# ---------------------------------------------------------------------------
# 🏗️ Index & ranking (mirrors production retrieval)
# ---------------------------------------------------------------------------


def _build_index(resumes: dict[str, str], embedder) -> dict:
    """Chunk every resume with the production chunker and embed all chunks."""
    chunks: list[dict] = []
    texts: list[str] = []
    for resume_id, text in resumes.items():
        for c in chunking.chunk_resume(text, resume_id):
            chunks.append(c)
            texts.append(c["text"])
    if not chunks:
        raise ValueError("dataset produced no chunks (empty resumes?)")
    matrix = np.asarray(embedder(texts), dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    return {"chunks": chunks, "matrix": matrix, "norms": norms}


def _cosine_sims(query_vec, index: dict) -> np.ndarray:
    q = np.asarray(query_vec, dtype=float)
    denom = np.linalg.norm(q) * index["norms"] + 1e-9
    return (index["matrix"] @ q) / denom


def _rank_chunks(query_text: str, index: dict, embedder) -> list[dict]:
    """Rank every chunk by cosine similarity (the vector-search baseline)."""
    qv = embedder([query_text])[0]
    sims = _cosine_sims(qv, index)
    order = np.argsort(-sims, kind="stable")
    return [
        {
            "resume_id": index["chunks"][i]["candidate_id"],
            "section": index["chunks"][i]["section"],
            "text": index["chunks"][i]["text"],
            "score": round(float(sims[i]), 4),
        }
        for i in order
    ]


def _dedupe_docs(ranked_chunks: list[dict]) -> list[dict]:
    """Collapse chunks to documents (best chunk score wins per resume)."""
    seen: set[str] = set()
    docs: list[dict] = []
    for c in ranked_chunks:
        cid = c["resume_id"]
        if cid in seen:
            continue
        seen.add(cid)
        docs.append({"resume_id": cid, "score": c["score"]})
    return docs


# ---------------------------------------------------------------------------
# 📐 Information-retrieval metrics (document-level, binary relevance)
# ---------------------------------------------------------------------------


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k."""
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return sum(1 for r in relevant if r in top) / len(relevant)


def mean_reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document; 0.0 if none is retrieved."""
    for i, cid in enumerate(ranked_ids, 1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """NDCG@k with binary relevance (discount 1/log2(rank+1))."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, cid in enumerate(ranked_ids[:k], 1)
        if cid in relevant
    )
    ideal = sum(
        1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1)
    )
    return dcg / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# ▶️ Evaluation runner
# ---------------------------------------------------------------------------


def _score_query(query: dict, docs: list[dict], k: int) -> dict:
    ranked_ids = [d["resume_id"] for d in docs]
    relevant = set(query["relevant"])
    return {
        "query_id": query["id"],
        "query_text": query["text"],
        "relevant": sorted(relevant),
        "docs": docs,
        "recall": recall_at_k(ranked_ids, relevant, k),
        "mrr": mean_reciprocal_rank(ranked_ids, relevant),
        "ndcg": ndcg_at_k(ranked_ids, relevant, k),
    }


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "recall_at_k": sum(r["recall"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "ndcg_at_k": sum(r["ndcg"] for r in rows) / n,
        "per_query": rows,
    }


def run_eval(
    k: int = 5,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    use_rerank: bool = True,
    cases: dict | None = None,
) -> dict:
    """Run the full evaluation and return structured results.

    Args:
        k: Top-k for recall@k / NDCG@k.
        embedder: ``callable(list[str]) -> list[list[float]]``. Defaults to the
            production sentence-transformers embedder.
        reranker: ``callable(query, hits, top_k) -> list[dict]`` (same contract
            as ``rerank.rerank``). Defaults to the production cross-encoder.
        use_rerank: Whether to compute the with-rerank column. Set ``False``
            for a pure vector-baseline run (nothing reranker-related runs).
        cases: Dataset dict; defaults to the built-in ``EVAL_CASES``.

    Returns:
        A dict with corpus stats plus ``no_rerank`` and (when ``use_rerank``)
        ``with_rerank`` aggregates, each carrying ``recall_at_k``, ``mrr``,
        ``ndcg_at_k`` and a ``per_query`` breakdown.
    """
    cases = cases if cases is not None else EVAL_CASES
    if embedder is None:
        embedder = emb.embed_texts
    if reranker is None:
        reranker = rr.rerank

    resumes: dict[str, str] = cases["resumes"]
    queries: list[dict] = cases["queries"]
    index = _build_index(resumes, embedder)
    pool_k = max(k * _POOL_MULTIPLIER, _POOL_MIN)

    no_rows: list[dict] = []
    with_rows: list[dict] = []
    for q in queries:
        ranked = _rank_chunks(q["text"], index, embedder)
        no_docs = _dedupe_docs(ranked)[:k]
        no_rows.append(_score_query(q, no_docs, k))

        if use_rerank and reranker is not None:
            pool = ranked[:pool_k]
            reranked_chunks = reranker(q["text"], pool, k)
            with_docs = _dedupe_docs(reranked_chunks)[:k]
            with_rows.append(_score_query(q, with_docs, k))

    return {
        "k": k,
        "n_resumes": len(resumes),
        "n_chunks": len(index["chunks"]),
        "n_queries": len(queries),
        "no_rerank": _aggregate(no_rows),
        "with_rerank": _aggregate(with_rows) if use_rerank else None,
    }


# ---------------------------------------------------------------------------
# 🖨️ CLI
# ---------------------------------------------------------------------------


def _load_embedder():
    """Real production embedder (sentence-transformers).

    Warms the model up-front so download/load failures surface inside
    ``main``'s friendly error handler instead of a raw traceback later.
    """
    emb.get_model()
    return emb.embed_texts


def _load_reranker():
    """Real production reranker (cross-encoder). Warms the model up-front."""
    rr.get_model()
    return rr.rerank


def _print_comparison(results: dict, verbose: bool = False) -> None:
    k = results["k"]
    with_ = results["with_rerank"]
    no = results["no_rerank"]

    print()
    print("=" * 64)
    print(" TalentIQ — Offline Retrieval Evaluation")
    print("=" * 64)
    print(f" Corpus      : {results['n_resumes']} resumes · {results['n_chunks']} chunks · {results['n_queries']} queries")
    print(f" Top k       : {k}")
    print(f" Embeddings  : {config.EMBEDDING_MODEL}")
    print(f" Reranker    : {config.RERANK_MODEL if with_ is not None else 'off (--no-rerank)'}")
    print("-" * 64)

    if with_ is None:
        print(f" {'Metric':<14}{'no-rerank':>12}")
        for name, val in (
            (f"recall@{k}", no["recall_at_k"]),
            ("MRR", no["mrr"]),
            (f"NDCG@{k}", no["ndcg_at_k"]),
        ):
            print(f" {name:<14}{val:>12.3f}")
    else:
        print(f" {'Metric':<14}{'no-rerank':>12}{'with-rerank':>14}{'Δ':>9}")
        for name, a, b in (
            (f"recall@{k}", no["recall_at_k"], with_["recall_at_k"]),
            ("MRR", no["mrr"], with_["mrr"]),
            (f"NDCG@{k}", no["ndcg_at_k"], with_["ndcg_at_k"]),
        ):
            print(f" {name:<14}{a:>12.3f}{b:>14.3f}{b - a:>+9.3f}")
        wins = ties = losses = 0
        for nr, wr in zip(no["per_query"], with_["per_query"], strict=False):
            if wr["mrr"] > nr["mrr"] + 1e-9:
                wins += 1
            elif abs(wr["mrr"] - nr["mrr"]) <= 1e-9:
                ties += 1
            else:
                losses += 1
        print("-" * 64)
        print(f" MRR verdict : reranking improved {wins}/{len(no['per_query'])} queries, tied {ties}, hurt {losses}")
    print("=" * 64)

    if verbose:
        _print_per_query(no, with_, k)


def _print_per_query(no: dict, with_: dict | None, k: int) -> None:
    print()
    if with_ is None:
        print(f" {'query':<8}{'recall':>8}{'MRR':>8}{'NDCG':>8}")
    else:
        print(f" {'query':<8}{'recall':>8}{'MRR':>8}{'NDCG':>8}   |   {'recall':>8}{'MRR':>8}{'NDCG':>8}   (with-rerank)")
    for i, nr in enumerate(no["per_query"]):
        wr = with_["per_query"][i] if with_ is not None else None
        left = f" {nr['query_id']:<8}{nr['recall']:>8.3f}{nr['mrr']:>8.3f}{nr['ndcg']:>8.3f}"
        if wr is not None:
            left += f"   |   {wr['recall']:>8.3f}{wr['mrr']:>8.3f}{wr['ndcg']:>8.3f}"
        print(left)
        print(f"   relevant           : {nr['relevant']}")
        print(f"   top-{k} no-rerank    : {[d['resume_id'] for d in nr['docs']]}")
        if wr is not None:
            print(f"   top-{k} with-rerank  : {[d['resume_id'] for d in wr['docs']]}")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252 and raise UnicodeEncodeError on some
    # characters (e.g. Δ) — reconfigure to UTF-8 with lossy fallback so the
    # table always prints instead of crashing. Harmless elsewhere.
    for stream in (sys.stdout, sys.stderr):
        # ``reconfigure`` exists on real console streams (3.7+) but not on
        # e.g. StringIO in tests — resolve dynamically so both type-check and
        # run cleanly.
        with contextlib.suppress(ValueError):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="eval_retrieval",
        description=(
            "Measure retrieval quality (recall@k, MRR, NDCG@k) with and without "
            "the reranker over the labeled resume↔JD eval set."
        ),
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="top-k for recall@k / NDCG@k (default: 5)",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="run the vector baseline only (skips the cross-encoder download)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print a per-query breakdown",
    )
    args = parser.parse_args(argv)

    if args.k < 1:
        parser.error("--k must be >= 1")

    try:
        embedder = _load_embedder()
    except Exception as exc:  # model download / import failure — fail with a hint
        print(f"Could not load the embedding model: {exc}", file=sys.stderr)
        print(
            "Check your network connection (models download on first run) and try again.",
            file=sys.stderr,
        )
        return 2

    reranker = None
    if not args.no_rerank:
        try:
            reranker = _load_reranker()
        except Exception as exc:
            print(f"Could not load the reranker model: {exc}", file=sys.stderr)
            return 2

    results = run_eval(
        k=args.k,
        embedder=embedder,
        reranker=reranker,
        use_rerank=not args.no_rerank,
    )
    _print_comparison(results, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
