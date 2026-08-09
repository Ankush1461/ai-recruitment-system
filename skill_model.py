# ================================================================
# 🎓 Fine-tuned Skill Classifier — tiny BERT trained on resume skills
# ================================================================
"""Fine-tuned skill-category classifier for TalentIQ.

This is the project's *trained* ML component: a small BERT model
(default ``prajjwal1/bert-tiny``) fine-tuned on a labeled dataset of
~300 resume skill phrases across 10 categories (languages, ML frameworks,
NLP, data science, data engineering, databases, cloud/DevOps, frontend,
backend, soft skills).

    python skill_model.py                      # train + save (data/skill_model)
    python skill_model.py --epochs 6           # longer training
    python skill_model.py --model <hf-id>      # different backbone
    python skill_model.py --check              # classify a few phrases with a
                                               # trained model (no training)

Training is a from-scratch PyTorch loop (no ``Trainer``/``accelerate``/``datasets``
deps): warmup schedule, gradient clipping, per-epoch evaluation with
macro-F1, best-checkpoint selection, and HF ``save_pretrained`` output.

Integration: when a trained model exists, ``ranking._keyword_overlap``
falls back to *category* matching when literal keyword overlap is zero —
so a JD that says "Amazon Web Services" still credits a resume that says
"AWS", and "PostgreSQL" credits "postgres". Purely additive and fail-open:
without a trained model every call returns immediately and the pipeline
behaves exactly as before.
"""

from __future__ import annotations

try:
    import spaces  # type: ignore # noqa: F401
except Exception:
    pass

import argparse
import contextlib
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

import config

DEFAULT_MODEL = "prajjwal1/bert-mini"  # 11M params — better capacity than bert-tiny
# Phrases are 1-4 words; longer inputs add padding noise for a tiny model.
_MAX_LEN = 24

# ---------------------------------------------------------------------------
# 📚 Labeled dataset — resume skill phrases → category (~300 hand-labeled)
# ---------------------------------------------------------------------------

TRAIN_DATA: list[tuple[str, str]] = [
    # ── languages ──────────────────────────────────────────────────────────
    ("Python", "languages"), ("Java", "languages"), ("C++", "languages"),
    ("C", "languages"), ("C#", "languages"), ("Go", "languages"),
    ("Golang", "languages"), ("Rust", "languages"), ("Swift", "languages"),
    ("Kotlin", "languages"), ("Ruby", "languages"), ("PHP", "languages"),
    ("Scala", "languages"), ("R", "languages"), ("MATLAB", "languages"),
    ("Bash", "languages"), ("Shell scripting", "languages"),
    ("Perl", "languages"), ("Groovy", "languages"), ("Julia", "languages"),
    ("Lua", "languages"), ("Objective-C", "languages"), ("VBA", "languages"),
    ("Elixir", "languages"), ("Erlang", "languages"), ("Haskell", "languages"),
    ("Clojure", "languages"), ("Dart", "languages"),
    ("PowerShell", "languages"), ("Java programming", "languages"),
    # ── ML frameworks ───────────────────────────────────────────────────────
    ("PyTorch", "ml_frameworks"), ("TensorFlow", "ml_frameworks"),
    ("Keras", "ml_frameworks"), ("scikit-learn", "ml_frameworks"),
    ("XGBoost", "ml_frameworks"), ("LightGBM", "ml_frameworks"),
    ("CatBoost", "ml_frameworks"), ("JAX", "ml_frameworks"),
    ("PyTorch Lightning", "ml_frameworks"), ("Hugging Face", "ml_frameworks"),
    ("fastai", "ml_frameworks"), ("MLflow", "ml_frameworks"),
    ("ONNX", "ml_frameworks"), ("OpenCV", "ml_frameworks"),
    ("CUDA", "ml_frameworks"), ("Weights and Biases", "ml_frameworks"),
    ("SHAP", "ml_frameworks"), ("LangChain", "ml_frameworks"),
    ("LlamaIndex", "ml_frameworks"), ("Ray", "ml_frameworks"),
    ("Optuna", "ml_frameworks"), ("DVC", "ml_frameworks"),
    ("Numba", "ml_frameworks"), ("Caffe", "ml_frameworks"),
    ("scikit-image", "ml_frameworks"), ("model training", "ml_frameworks"),
    ("neural networks", "ml_frameworks"), ("deep learning", "ml_frameworks"),
    ("reinforcement learning", "ml_frameworks"),
    ("maschinelles Lernen", "ml_frameworks"), ("ML-Pipelines", "ml_frameworks"),
    # ── NLP ─────────────────────────────────────────────────────────────────
    ("spaCy", "nlp"), ("NLTK", "nlp"), ("BERT", "nlp"), ("GPT", "nlp"),
    ("Llama", "nlp"), ("T5", "nlp"), ("RoBERTa", "nlp"),
    ("DistilBERT", "nlp"), ("transformers", "nlp"),
    ("tokenization", "nlp"), ("word embeddings", "nlp"),
    ("Named Entity Recognition", "nlp"), ("NER", "nlp"),
    ("text classification", "nlp"), ("sentiment analysis", "nlp"),
    ("machine translation", "nlp"), ("summarization", "nlp"),
    ("question answering", "nlp"), ("RAG", "nlp"),
    ("retrieval-augmented generation", "nlp"), ("sentence embeddings", "nlp"),
    ("fine-tuning", "nlp"), ("prompt engineering", "nlp"),
    ("vector search", "nlp"), ("semantic search", "nlp"),
    ("large language models", "nlp"), ("LLM evaluation", "nlp"),
    ("few-shot learning", "nlp"), ("part-of-speech tagging", "nlp"),
    ("topic modeling", "nlp"), ("TF-IDF", "nlp"), ("Word2Vec", "nlp"),
    ("GloVe", "nlp"), ("FastText", "nlp"), ("ELMo", "nlp"),
    ("XLNet", "nlp"), ("ALBERT", "nlp"), ("ELECTRA", "nlp"),
    ("DeBERTa", "nlp"), ("BART", "nlp"), ("Pegasus", "nlp"),
    # ── data science ────────────────────────────────────────────────────────
    ("pandas", "data_science"), ("numpy", "data_science"),
    ("scipy", "data_science"), ("statsmodels", "data_science"),
    ("matplotlib", "data_science"), ("seaborn", "data_science"),
    ("plotly", "data_science"), ("Tableau", "data_science"),
    ("Power BI", "data_science"), ("A/B testing", "data_science"),
    ("causal inference", "data_science"), ("hypothesis testing", "data_science"),
    ("regression analysis", "data_science"), ("classification", "data_science"),
    ("clustering", "data_science"), ("time series", "data_science"),
    ("forecasting", "data_science"), ("feature engineering", "data_science"),
    ("exploratory data analysis", "data_science"), ("EDA", "data_science"),
    ("data visualization", "data_science"), ("dashboarding", "data_science"),
    ("statistical modeling", "data_science"), ("Bayesian statistics", "data_science"),
    ("experiment design", "data_science"), ("data cleaning", "data_science"),
    ("outlier detection", "data_science"), ("dimension reduction", "data_science"),
    ("PCA", "data_science"), ("t-SNE", "data_science"),
    ("product analytics", "data_science"), ("conversion analysis", "data_science"),
    # ── data engineering ─────────────────────────────────────────────────────
    ("Spark", "data_eng"), ("PySpark", "data_eng"), ("Airflow", "data_eng"),
    ("dbt", "data_eng"), ("Kafka", "data_eng"), ("Hadoop", "data_eng"),
    ("Hive", "data_eng"), ("Flink", "data_eng"), ("Beam", "data_eng"),
    ("Snowflake", "data_eng"), ("BigQuery", "data_eng"),
    ("Redshift", "data_eng"), ("Databricks", "data_eng"),
    ("Delta Lake", "data_eng"), ("ETL", "data_eng"), ("ELT", "data_eng"),
    ("data pipelines", "data_eng"), ("data warehouse", "data_eng"),
    ("data lake", "data_eng"), ("streaming", "data_eng"),
    ("Kinesis", "data_eng"), ("Glue", "data_eng"), ("EMR", "data_eng"),
    ("Trino", "data_eng"), ("ClickHouse", "data_eng"), ("parquet", "data_eng"),
    ("data lineage", "data_eng"), ("data quality", "data_eng"),
    ("CDC", "data_eng"), ("datenbank-Pipelines", "data_eng"),
    # ── databases ───────────────────────────────────────────────────────────
    ("SQL", "databases"), ("PostgreSQL", "databases"), ("MySQL", "databases"),
    ("SQLite", "databases"), ("MongoDB", "databases"), ("Redis", "databases"),
    ("Elasticsearch", "databases"), ("DynamoDB", "databases"),
    ("Cassandra", "databases"), ("Neo4j", "databases"),
    ("Couchbase", "databases"), ("MariaDB", "databases"),
    ("Oracle", "databases"), ("InfluxDB", "databases"),
    ("TimescaleDB", "databases"), ("CockroachDB", "databases"),
    ("SQL Server", "databases"), ("database design", "databases"),
    ("indexing", "databases"), ("query optimization", "databases"),
    ("normalization", "databases"), ("NoSQL", "databases"),
    ("sharding", "databases"), ("replication", "databases"),
    ("transactions", "databases"), ("stored procedures", "databases"),
    ("SQLAlchemy", "databases"), ("Prisma", "databases"),
    ("Datenbanken", "databases"),
    # ── cloud / DevOps ───────────────────────────────────────────────────────
    ("AWS", "cloud_devops"), ("Azure", "cloud_devops"),
    ("GCP", "cloud_devops"), ("Kubernetes", "cloud_devops"),
    ("Docker", "cloud_devops"), ("Terraform", "cloud_devops"),
    ("Ansible", "cloud_devops"), ("Jenkins", "cloud_devops"),
    ("GitLab CI", "cloud_devops"), ("GitHub Actions", "cloud_devops"),
    ("CI/CD", "cloud_devops"), ("Prometheus", "cloud_devops"),
    ("Grafana", "cloud_devops"), ("Datadog", "cloud_devops"),
    ("CloudFormation", "cloud_devops"), ("Helm", "cloud_devops"),
    ("ArgoCD", "cloud_devops"), ("OpenShift", "cloud_devops"),
    ("ECS", "cloud_devops"), ("EKS", "cloud_devops"), ("Lambda", "cloud_devops"),
    ("S3", "cloud_devops"), ("EC2", "cloud_devops"), ("IAM", "cloud_devops"),
    ("serverless", "cloud_devops"), ("observability", "cloud_devops"),
    ("monitoring", "cloud_devops"), ("OpenTelemetry", "cloud_devops"),
    ("service mesh", "cloud_devops"), ("incident response", "cloud_devops"),
    ("SRE", "cloud_devops"), ("devops", "cloud_devops"), ("kubectl", "cloud_devops"),
    ("cloud infrastructure", "cloud_devops"), ("Modell-Monitoring", "cloud_devops"),
    # ── frontend ─────────────────────────────────────────────────────────────
    ("React", "frontend"), ("React Native", "frontend"),
    ("TypeScript", "frontend"), ("Next.js", "frontend"), ("Vue", "frontend"),
    ("Angular", "frontend"), ("Svelte", "frontend"), ("Redux", "frontend"),
    ("CSS", "frontend"), ("HTML", "frontend"), ("Sass", "frontend"),
    ("Tailwind", "frontend"), ("Webpack", "frontend"), ("Vite", "frontend"),
    ("Jest", "frontend"), ("Cypress", "frontend"), ("Playwright", "frontend"),
    ("accessibility", "frontend"), ("responsive design", "frontend"),
    ("web performance", "frontend"), ("design systems", "frontend"),
    ("Storybook", "frontend"), ("server-side rendering", "frontend"),
    ("SSR", "frontend"), ("PWA", "frontend"), ("UI development", "frontend"),
    ("component testing", "frontend"), ("Frontend-Entwicklung", "frontend"),
    # ── backend ──────────────────────────────────────────────────────────────
    ("Django", "backend"), ("Flask", "backend"), ("FastAPI", "backend"),
    ("Node.js", "backend"), ("Express", "backend"), ("Spring Boot", "backend"),
    (".NET", "backend"), ("ASP.NET", "backend"), ("Ruby on Rails", "backend"),
    ("Laravel", "backend"), ("Symfony", "backend"), ("Gin", "backend"),
    ("REST", "backend"), ("RESTful APIs", "backend"), ("gRPC", "backend"),
    ("GraphQL", "backend"), ("microservices", "backend"),
    ("message queues", "backend"), ("RabbitMQ", "backend"),
    ("Celery", "backend"), ("WebSockets", "backend"), ("API design", "backend"),
    ("authentication", "backend"), ("authorization", "backend"),
    ("caching", "backend"), ("rate limiting", "backend"),
    ("load balancing", "backend"), ("nginx", "backend"),
    ("distributed systems", "backend"), ("Backend-Entwicklung", "backend"),
    # ── soft skills ──────────────────────────────────────────────────────────
    ("communication", "soft_skills"), ("teamwork", "soft_skills"),
    ("leadership", "soft_skills"), ("mentoring", "soft_skills"),
    ("agile", "soft_skills"), ("scrum", "soft_skills"), ("kanban", "soft_skills"),
    ("project management", "soft_skills"),
    ("stakeholder management", "soft_skills"),
    ("problem solving", "soft_skills"), ("critical thinking", "soft_skills"),
    ("time management", "soft_skills"), ("adaptability", "soft_skills"),
    ("ownership", "soft_skills"), ("collaboration", "soft_skills"),
    ("documentation", "soft_skills"), ("code review", "soft_skills"),
    ("pair programming", "soft_skills"), ("cross-functional", "soft_skills"),
    ("negotiation", "soft_skills"), ("presentation", "soft_skills"),
    ("public speaking", "soft_skills"), ("empathy", "soft_skills"),
    ("conflict resolution", "soft_skills"), ("decision making", "soft_skills"),
    ("prioritization", "soft_skills"), ("on-call", "soft_skills"),
    # ── realistic resume variants (give the tiny model enough signal) ───────
    ("Python programming", "languages"), ("Java development", "languages"),
    ("Go programming", "languages"), ("C++ development", "languages"),
    ("scripting languages", "languages"),
    ("PyTorch development", "ml_frameworks"), ("TensorFlow models", "ml_frameworks"),
    ("training machine learning models", "ml_frameworks"),
    ("building deep learning models", "ml_frameworks"),
    ("natural language processing models", "nlp"),
    ("text generation systems", "nlp"), ("language model applications", "nlp"),
    ("statistical analysis", "data_science"), ("data analysis", "data_science"),
    ("building dashboards", "data_science"), ("analytical modeling", "data_science"),
    ("ETL pipelines", "data_eng"), ("data pipeline development", "data_eng"),
    ("warehouse modeling", "data_eng"), ("streaming pipelines", "data_eng"),
    ("SQL queries", "databases"), ("database administration", "databases"),
    ("relational databases", "databases"), ("database tuning", "databases"),
    ("AWS cloud", "cloud_devops"), ("cloud infrastructure", "cloud_devops"),
    ("container orchestration", "cloud_devops"), ("deployment pipelines", "cloud_devops"),
    ("React components", "frontend"), ("web application development", "frontend"),
    ("frontend frameworks", "frontend"), ("user interfaces", "frontend"),
    ("backend services", "backend"), ("API development", "backend"),
    ("building microservices", "backend"), ("server applications", "backend"),
    ("team leadership", "soft_skills"), ("agile development", "soft_skills"),
    ("stakeholder communication", "soft_skills"), ("cross-team collaboration", "soft_skills"),
    # ── second expansion round: verb+object / adjective+noun variants ───────
    ("Python scripting", "languages"), ("automation scripts", "languages"),
    ("systems programming", "languages"), ("embedded C", "languages"),
    ("R programming", "languages"), ("Ruby scripting", "languages"),
    ("JavaScript development", "languages"), ("Java programming", "languages"),
    ("C programming", "languages"), ("shell automation", "languages"),
    ("PyTorch models", "ml_frameworks"), ("TensorFlow training", "ml_frameworks"),
    ("Keras models", "ml_frameworks"), ("model development", "ml_frameworks"),
    ("ML training", "ml_frameworks"), ("gradient boosting", "ml_frameworks"),
    ("model optimization", "ml_frameworks"), ("hyperparameter tuning", "ml_frameworks"),
    ("model evaluation", "ml_frameworks"), ("training pipelines", "ml_frameworks"),
    ("text analysis", "nlp"), ("NLP pipelines", "nlp"),
    ("information extraction", "nlp"), ("entity extraction", "nlp"),
    ("text embeddings", "nlp"), ("semantic similarity", "nlp"),
    ("document retrieval", "nlp"), ("chatbot development", "nlp"),
    ("conversational AI", "nlp"), ("LLM applications", "nlp"),
    ("predictive modeling", "data_science"), ("churn analysis", "data_science"),
    ("customer segmentation", "data_science"), ("cohort analysis", "data_science"),
    ("experimentation", "data_science"), ("multivariate testing", "data_science"),
    ("business intelligence", "data_science"), ("KPI analysis", "data_science"),
    ("statistical testing", "data_science"), ("model interpretation", "data_science"),
    ("Spark jobs", "data_eng"), ("Airflow DAGs", "data_eng"),
    ("dbt models", "data_eng"), ("Kafka pipelines", "data_eng"),
    ("batch processing", "data_eng"), ("stream processing", "data_eng"),
    ("data ingestion", "data_eng"), ("schema management", "data_eng"),
    ("lakehouse", "data_eng"), ("data mesh", "data_eng"),
    ("Postgres", "databases"), ("MySQL queries", "databases"),
    ("Mongo collections", "databases"), ("Redis caching", "databases"),
    ("Elasticsearch search", "databases"), ("database performance", "databases"),
    ("data modeling", "databases"), ("schema design", "databases"),
    ("ACID transactions", "databases"), ("read replicas", "databases"),
    ("Kubernetes clusters", "cloud_devops"), ("Docker containers", "cloud_devops"),
    ("AWS services", "cloud_devops"), ("Azure services", "cloud_devops"),
    ("infrastructure as code", "cloud_devops"), ("release automation", "cloud_devops"),
    ("blue-green deployment", "cloud_devops"), ("canary releases", "cloud_devops"),
    ("site reliability", "cloud_devops"), ("SLOs", "cloud_devops"),
    ("React development", "frontend"), ("Vue components", "frontend"),
    ("Angular apps", "frontend"), ("CSS styling", "frontend"),
    ("responsive layouts", "frontend"), ("cross-browser compatibility", "frontend"),
    ("state management", "frontend"), ("UI components", "frontend"),
    ("REST endpoints", "backend"), ("API gateway", "backend"),
    ("FastAPI services", "backend"), ("Django apps", "backend"),
    ("Node services", "backend"), ("event-driven architecture", "backend"),
    ("background jobs", "backend"), ("task queues", "backend"),
    ("OAuth", "backend"), ("API versioning", "backend"),
    ("stakeholder alignment", "soft_skills"), ("requirements gathering", "soft_skills"),
    ("technical writing", "soft_skills"), ("knowledge sharing", "soft_skills"),
    ("workshop facilitation", "soft_skills"), ("remote collaboration", "soft_skills"),
    ("coaching", "soft_skills"), ("roadmap planning", "soft_skills"),
]


def categories() -> list[str]:
    """Sorted category taxonomy (deterministic label ids)."""
    return sorted({c for _, c in TRAIN_DATA})


def _label2id() -> dict[str, int]:
    return {c: i for i, c in enumerate(categories())}


# ---------------------------------------------------------------------------
# 🏋️ Training (from-scratch PyTorch loop — no Trainer/accelerate/datasets)
# ---------------------------------------------------------------------------


def _macro_f1(preds: list[int], labels: list[int], num_labels: int) -> float:
    """Macro-averaged F1 over all classes (classes with no support count 0)."""
    f1s = []
    for c in range(num_labels):
        tp = sum(1 for p, lbl in zip(preds, labels, strict=False) if p == c and lbl == c)
        fp = sum(1 for p, lbl in zip(preds, labels, strict=False) if p == c and lbl != c)
        fn = sum(1 for p, lbl in zip(preds, labels, strict=False) if lbl == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _tokenizer(source: str | Path) -> BertTokenizer:
    """HF tokenizer from a model id, or from a local vocab.txt (hermetic tests)."""
    if isinstance(source, (str, Path)) and Path(source).is_file():
        # transformers 5.x builds the slow tokenizer from a vocab dict, not a file
        tokens = Path(source).read_text(encoding="utf-8").splitlines()
        return BertTokenizer(vocab={t: i for i, t in enumerate(tokens)})
    return BertTokenizer.from_pretrained(str(source))


def _model_for(tokenizer: BertTokenizer, model_name: str | Path, num_labels: int):
    """Model from an HF id, or a random-init tiny config (hermetic tests)."""
    if isinstance(model_name, (str, Path)) and Path(model_name).is_file():
        cfg = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
        )
        cfg.num_labels = num_labels
        return BertForSequenceClassification(cfg)
    cfg = BertConfig.from_pretrained(str(model_name))
    cfg.num_labels = num_labels
    return BertForSequenceClassification(cfg)


def _evaluate(model, tokenizer, texts: list[str], labels: list[int], max_len: int):
    model.eval()
    if not labels:  # a tiny custom dataset can yield an empty val split
        return 0.0, 0.0
    preds: list[int] = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tokenizer(
                texts[i : i + 32], padding=True, truncation=True,
                max_length=max_len, return_tensors="pt",
            )
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).tolist())
    num_labels = len(categories())
    acc = sum(1 for p, lbl in zip(preds, labels, strict=False) if p == lbl) / len(labels)
    return acc, _macro_f1(preds, labels, num_labels)


def train(
    save_dir: str | Path | None = None,
    model_name: str | Path | None = None,
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 5e-4,
    val_frac: float = 0.2,
    seed: int = 42,
    max_len: int = _MAX_LEN,
    data: list[tuple[str, str]] | None = None,
) -> dict:
    """Fine-tune the skill classifier and save the best checkpoint.

    Args:
        save_dir: Output dir (defaults to ``config.SKILL_MODEL_DIR``).
        model_name: HF model id (default ``prajjwal1/bert-tiny``). A path to a
            local ``vocab.txt`` switches to a random-init tiny BERT — used by
            the hermetic offline tests.
        epochs / batch_size / lr / val_frac / seed / max_len: hyperparameters.
        data: (phrase, category) pairs; defaults to ``TRAIN_DATA``.

    Returns:
        Dict with val accuracy / macro-F1 / best epoch / sizes.
    """
    save_dir = Path(save_dir or config.SKILL_MODEL_DIR)
    model_name = model_name if model_name is not None else DEFAULT_MODEL
    data = data if data is not None else TRAIN_DATA
    if not data:
        raise ValueError("empty training data")

    random.seed(seed)
    torch.manual_seed(seed)
    l2i = _label2id()
    texts = [p for p, _ in data]
    labels = [l2i[c] for _, c in data]

    # Stratified split: keep every category represented in both folds.
    by_label: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        by_label.setdefault(lbl, []).append(i)
    val_indices: list[int] = []
    for idxs in by_label.values():
        random.shuffle(idxs)
        take = max(1, round(len(idxs) * val_frac)) if len(idxs) > 2 else 0
        val_indices.extend(idxs[:take])
    val_set = set(val_indices)
    train_idx = [i for i in range(len(texts)) if i not in val_set]
    random.shuffle(train_idx)
    val_idx = sorted(val_set)

    tr_texts = [texts[i] for i in train_idx]
    tr_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    tokenizer = _tokenizer(model_name)
    model = _model_for(tokenizer, model_name, len(l2i))
    print(f"[skill_model] backbone: {model_name} | n={len(texts)} "
          f"(train {len(tr_texts)} / val {len(val_texts)}) | epochs={epochs}", flush=True)

    encoded = tokenizer(
        tr_texts, padding=True, truncation=True, max_length=max_len,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        encoded["input_ids"], encoded["attention_mask"],
        encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])),
        torch.tensor(tr_labels),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Label smoothing + weight decay fight the memorization a tiny model shows
    # on a few-hundred-example dataset; best-epoch selection keeps the model
    # with the best held-out macro-F1 regardless.
    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total_steps = max(1, len(loader) * epochs)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(1, total_steps // 10)
    )

    best_f1, best_acc, best_epoch, best_state = -1.0, 0.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            input_ids, attn, tti, lbls = batch
            optimizer.zero_grad()
            logits = model(
                input_ids=input_ids, attention_mask=attn, token_type_ids=tti,
            ).logits
            loss = loss_fct(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            steps += 1
        acc, f1 = _evaluate(model, tokenizer, val_texts, val_labels, max_len)
        print(f"[skill_model] epoch {epoch}/{epochs} — train loss "
              f"{epoch_loss / max(steps, 1):.4f} · val accuracy {acc:.3f} "
              f"· macro-F1 {f1:.3f}", flush=True)
        if f1 > best_f1:
            best_f1, best_acc, best_epoch = f1, acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    (save_dir / "labels.json").write_text(
        json.dumps({"label2id": l2i, "id2label": {str(v): k for k, v in l2i.items()}}),
        encoding="utf-8",
    )
    metrics = {
        "accuracy": round(best_acc, 4),
        "macro_f1": round(best_f1, 4),
        "best_epoch": best_epoch,
        "n_train": len(tr_texts),
        "n_val": len(val_texts),
        "epochs": epochs,
        "model": str(model_name),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Drop any previously loaded model so a long-lived process serves the NEW
    # weights on the next available()/classify call (CLI is a fresh process).
    _reset()
    print(f"[skill_model] saved to {save_dir} — val acc {best_acc:.3f} · macro-F1 {best_f1:.3f}")
    return metrics


# ---------------------------------------------------------------------------
# 🔮 Inference (lazy singleton, thread-safe, fail-open)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {"model": None, "tokenizer": None, "id2label": None, "dir": None}
# RLock (reentrant): load() resets the cache inline on failure, so the lock
# must be re-acquirable from the same thread — a plain Lock deadlocks there.
_state_lock = threading.RLock()


def _clear_cache_locked() -> None:
    """Drop the cached model (caller must hold ``_state_lock``)."""
    _state["model"] = _state["tokenizer"] = _state["id2label"] = None
    _state["dir"] = None


def _reset() -> None:
    """Drop the cached model (used by tests and after (re)training)."""
    with _state_lock:
        _clear_cache_locked()


def _model_dir() -> Path:
    return Path(config.SKILL_MODEL_DIR)


def _has_artifact() -> bool:
    d = _model_dir()
    return (d / "config.json").exists() and (d / "labels.json").exists()


def load() -> bool:
    """Load the trained model once. Returns False (fail-open) on any problem."""
    with _state_lock:
        if _state["model"] is not None and Path(_state["dir"]) == _model_dir():
            return True
        if not _has_artifact():
            _clear_cache_locked()
            return False
        try:
            tokenizer = BertTokenizer.from_pretrained(str(_model_dir()))
            model = BertForSequenceClassification.from_pretrained(str(_model_dir()))
            labels = json.loads((_model_dir() / "labels.json").read_text(encoding="utf-8"))
            id2label = {int(k): v for k, v in labels["id2label"].items()}
            model.eval()
            _state.update(model=model, tokenizer=tokenizer, id2label=id2label,
                          dir=str(_model_dir()))
            return True
        except Exception as exc:  # corrupt/partial artifact — fail open
            print(f"[skill_model] load failed, classifier disabled: {exc}", flush=True)
            _clear_cache_locked()
            return False


def available() -> bool:
    """True when the env flag is on AND a trained model loaded successfully."""
    if not config.SKILL_CLASSIFIER_ENABLED:
        return False
    return load()


def _predict_batch(texts: list[str]) -> list[tuple[str, float]]:
    """Batch single-label prediction → [(label, confidence), ...].

    Model is read-only in eval mode after load(), so concurrent inference from
    Gradio worker threads is safe (the load itself is lock-protected).
    """
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    id2label = _state["id2label"]
    if model is None or tokenizer is None:
        return []
    out: list[tuple[str, float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tokenizer(
                texts[i : i + 32], padding=True, truncation=True,
                max_length=_MAX_LEN, return_tensors="pt",
            )
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)
            out.extend((id2label[int(i2)], round(float(c), 4))
                       for i2, c in zip(idx.tolist(), conf.tolist(), strict=False))
    return out


def classify_tokens(tokens: list[str]) -> dict[str, tuple[str, float]]:
    """Classify each unique token (deduped, one batched forward pass).

    Returns ``{token: (label, confidence)}`` — empty dict when unavailable.
    """
    if not tokens or not available():
        return {}
    unique = list(dict.fromkeys(tokens))  # dedupe, keep first-seen order
    preds = _predict_batch(unique)
    return dict(zip(unique, preds, strict=False))


def classify_token(token: str) -> tuple[str, float] | None:
    """Convenience for a single token; None when the classifier is off."""
    if not token or not available():
        return None
    return classify_tokens([token]).get(token)


# ---------------------------------------------------------------------------
# 🖨️ CLI
# ---------------------------------------------------------------------------

_CHECK_PHRASES = [
    "PyTorch", "AWS", "spaCy", "PostgreSQL", "React", "FastAPI",
    "A/B testing", "Kubernetes", "Spark", "communication",
]


def _run_check() -> int:
    if not available():
        print("No trained model found — train one first:  python skill_model.py")
        return 1
    print(f"Classifier: {_model_dir()} (labels: {len(categories())})")
    for phrase in _CHECK_PHRASES:
        pair = classify_token(phrase)
        label = pair[0] if pair else "?"
        conf = pair[1] if pair else 0.0
        print(f"  {phrase:<18} → {label:<16} ({conf:.2f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252 and can't print e.g. ↔ — reconfigure
    # to UTF-8 with lossy fallback so the CLI never crashes on a glyph.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(ValueError):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="skill_model",
        description=(
            "Fine-tune the tiny-BERT skill classifier on labeled resume skill "
            "phrases, or classify a few phrases with a trained model."
        ),
    )
    parser.add_argument("--epochs", type=int, default=12, help="training epochs (default: 12)")
    parser.add_argument("--model", default=None,
                        help=f"HF model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--save-dir", default=None,
                        help="output dir (default: config.SKILL_MODEL_DIR)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true",
                        help="classify example phrases with a trained model (no training)")
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.check:
        return _run_check()

    train(
        save_dir=args.save_dir, model_name=args.model,
        epochs=args.epochs, seed=args.seed,
    )
    print()
    print("Done. Ranking now falls back to skill-category matching when literal")
    print("keyword overlap is zero (e.g. JD 'Amazon Web Services' ↔ resume 'AWS').")
    print(f"Saved: {Path(args.save_dir or config.SKILL_MODEL_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
