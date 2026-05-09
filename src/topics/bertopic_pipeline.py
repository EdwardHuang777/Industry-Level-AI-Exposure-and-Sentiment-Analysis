"""Stage 03 — BERTopic clustering + topic→industry mapping.

Pipeline (per cleaned ~180K corpus):
    SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
    → UMAP (20 components, cosine)
    → HDBSCAN (min_cluster_size=200)
    → CountVectorizer + ClassTfidfTransformer
    → BERTopic.fit_transform()

Then map each discovered topic to one of 22 anchor-defined industries via
mean-anchor cosine similarity over the SentenceTransformer space.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.config import (
    BERTOPIC_DIR,
    BERTOPIC_MODEL_DIR,
    INDUSTRY_ANCHORS,
    RANDOM_SEED,
)


# ── Defaults ──────────────────────────────────────────────────────────────────
@dataclass
class BERTopicConfig:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    umap_n_components: int = 20
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 200
    hdbscan_min_samples: int = 15
    top_n_words: int = 15
    min_topic_size: int = 100
    vectorizer_min_df: int = 15
    vectorizer_max_df: float = 0.90
    vectorizer_ngram_range: tuple[int, int] = (1, 2)
    random_state: int = RANDOM_SEED


# ── Text normalization for topic modeling ─────────────────────────────────────
def normalize_for_topic_modeling(text) -> str:
    """Strip URLs/HTML/punctuation, collapse whitespace. Keep words intact."""
    value = "" if pd.isna(text) else str(text)
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^A-Za-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# ── Model construction ────────────────────────────────────────────────────────
def build_model(cfg: BERTopicConfig | None = None) -> BERTopic:
    cfg = cfg or BERTopicConfig()
    embedding_model = SentenceTransformer(cfg.embedding_model_name, trust_remote_code=True)
    umap_model = UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric="cosine",
        random_state=cfg.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=cfg.vectorizer_ngram_range,
        min_df=cfg.vectorizer_min_df,
        max_df=cfg.vectorizer_max_df,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        top_n_words=cfg.top_n_words,
        min_topic_size=cfg.min_topic_size,
        verbose=True,
    )


# ── Fit ───────────────────────────────────────────────────────────────────────
def fit_topics(clean_df: pd.DataFrame, cfg: BERTopicConfig | None = None
               ) -> tuple[BERTopic, pd.DataFrame, np.ndarray]:
    """Fit BERTopic on `clean_df`. Returns (model, df_with_assignments, probs)."""
    cfg = cfg or BERTopicConfig()
    work = clean_df.copy()
    work["model_text"] = work["text_clean_light"].fillna("").map(normalize_for_topic_modeling)
    work = work.loc[work["model_text"].str.len() > 0].copy()

    model = build_model(cfg)
    topics, probs = model.fit_transform(work["model_text"].tolist())

    work["topic_id"] = topics
    probs_arr = np.asarray(probs)
    if probs_arr.ndim == 1:
        work["topic_confidence"] = probs_arr
    elif probs_arr.ndim == 2:
        work["topic_confidence"] = probs_arr.max(axis=1)
    else:
        work["topic_confidence"] = np.nan
    return model, work, probs_arr


# ── Persistence ───────────────────────────────────────────────────────────────
def save_model(model: BERTopic, name: str = "bertopic_model"):
    out = BERTOPIC_MODEL_DIR / name
    try:
        model.save(str(out), serialization="safetensors", save_ctfidf=True)
    except Exception as exc:
        print(f"safetensors save failed ({exc}); falling back to pickle.")
        model.save(str(out), serialization="pickle", save_ctfidf=True)
    print(f"Saved BERTopic model → {out}")


def load_model(name: str = "bertopic_model",
               embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
               ) -> BERTopic:
    embedding_model = SentenceTransformer(embedding_model_name)
    return BERTopic.load(str(BERTOPIC_MODEL_DIR / name), embedding_model=embedding_model)


# ── Topic → Industry mapping ──────────────────────────────────────────────────
def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_topic_embeddings(model: BERTopic, work_df: pd.DataFrame,
                             embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
                             ) -> pd.DataFrame:
    """Mean-pool the model's representative docs per topic into a unit vector."""
    embedder = SentenceTransformer(embedding_model_name, device=_device())
    docs = work_df["model_text"].astype(str).tolist()
    doc_topics = np.array(model.topics_)
    doc_emb = embedder.encode(
        docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True,
    )

    rep_docs_map = model.get_representative_docs()
    topic_ids = sorted(set(doc_topics) - {-1})

    rows = []
    for tid in topic_ids:
        rep_docs = rep_docs_map.get(tid, [])[:10]
        if rep_docs:
            vec = embedder.encode(rep_docs, normalize_embeddings=True).mean(axis=0)
        else:
            idx = np.where(doc_topics == tid)[0]
            vec = doc_emb[idx].mean(axis=0)
        vec /= (np.linalg.norm(vec) + 1e-12)
        rows.append({
            "topic_id": int(tid),
            "topic_emb": vec,
            "count": int((doc_topics == tid).sum()),
        })
    return pd.DataFrame(rows)


def map_topics_to_industries(topic_emb_df: pd.DataFrame,
                             embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                             topk: int = 3) -> pd.DataFrame:
    """For each topic, find the top-K industries by cosine similarity to anchor mean."""
    embedder = SentenceTransformer(embedding_model_name, device=_device())
    industries = list(INDUSTRY_ANCHORS.keys())
    industry_emb = np.stack([
        embedder.encode(INDUSTRY_ANCHORS[ind], normalize_embeddings=True).mean(axis=0)
        for ind in industries
    ])

    topic_mat = np.stack(topic_emb_df["topic_emb"].values)
    sim = topic_mat @ industry_emb.T  # both are unit-norm → cosine
    top_idx = np.argsort(-sim, axis=1)[:, :topk]
    top_scores = np.take_along_axis(sim, top_idx, axis=1)

    out = []
    for i, tid in enumerate(topic_emb_df["topic_id"].values):
        row = {
            "topic_id": int(tid),
            "count": int(topic_emb_df.iloc[i]["count"]),
        }
        for k in range(topk):
            row[f"cand{k+1}"] = industries[top_idx[i, k]]
            row[f"score{k+1}"] = float(top_scores[i, k])
        out.append(row)
    return pd.DataFrame(out).sort_values("count", ascending=False)


def assign_industry(work_df: pd.DataFrame, topic_to_industry: pd.DataFrame) -> pd.DataFrame:
    """Merge top-1 industry candidate onto the document-level frame."""
    mapping = topic_to_industry[["topic_id", "cand1"]].rename(columns={"cand1": "industry"})
    out = work_df.merge(mapping, on="topic_id", how="left")
    out.loc[out["topic_id"] == -1, "industry"] = "Other/Unclustered"
    return out


# ── End-to-end ────────────────────────────────────────────────────────────────
def run(clean_df: pd.DataFrame,
        cfg: BERTopicConfig | None = None) -> tuple[BERTopic, pd.DataFrame]:
    """Fit → embed topics → map to industries → return assignments_df."""
    model, work, _ = fit_topics(clean_df, cfg)
    save_model(model)

    topic_emb_df = compute_topic_embeddings(model, work)
    candidates = map_topics_to_industries(topic_emb_df)
    assignments = assign_industry(work, candidates)

    assignments.to_parquet(BERTOPIC_DIR / "bertopic_assigned.parquet", index=False)
    candidates.to_parquet(BERTOPIC_DIR / "topic_industry_candidates.parquet", index=False)
    print(f"Wrote {len(assignments):,} document assignments → {BERTOPIC_DIR}")
    return model, assignments


def load_assignments() -> pd.DataFrame:
    return pd.read_parquet(BERTOPIC_DIR / "bertopic_assigned.parquet")
