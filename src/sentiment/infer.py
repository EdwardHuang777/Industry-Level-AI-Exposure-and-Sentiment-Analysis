"""Stage 06 — batched RoBERTa inference + multi-level aggregation.

Produces:
  - Document-level sentiment (label, confidence, continuous P(pos)−P(neg))
  - Industry-level composite impact score
  - Company / technology / driver summaries with sentiment composition
  - Monthly time-series of sentiment per industry
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    ADOPTION_DRIVER_PATTERNS,
    ENTITY_DIR,
    SENTIMENT_DIR,
    SENTIMENT_MODEL_DIR,
)

DEFAULT_MODEL_DIR = SENTIMENT_MODEL_DIR / "roberta-base-news-sentiment"
BAD_VALS = {"", "nan", "none", "null"}
SENT_LABELS = ["positive", "neutral", "negative", "unclear"]


# ── Inference ─────────────────────────────────────────────────────────────────
def load_classifier(model_dir: Path = DEFAULT_MODEL_DIR):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device


def batched_predict(texts: list[str], tokenizer, model, device: str,
                    batch_size: int = 64, max_length: int = 256
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pred_id, confidence, continuous_score = P(pos)−P(neg))."""
    id2label = model.config.id2label
    pos_id = next(i for i, n in id2label.items() if n == "positive")
    neg_id = next(i for i, n in id2label.items() if n == "negative")

    pred_chunks, conf_chunks, score_chunks = [], [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa inference"):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = F.softmax(model(**enc).logits, dim=-1).detach().cpu().numpy()
        pred_chunks.append(probs.argmax(axis=1))
        conf_chunks.append(probs.max(axis=1))
        score_chunks.append(probs[:, pos_id] - probs[:, neg_id])

    return (
        np.concatenate(pred_chunks),
        np.concatenate(conf_chunks),
        np.concatenate(score_chunks),
    )


def annotate_with_sentiment(context_df: pd.DataFrame,
                            model_dir: Path = DEFAULT_MODEL_DIR
                            ) -> pd.DataFrame:
    """Run inference on `org_entity_context` and append sent_label/conf/score."""
    tokenizer, model, device = load_classifier(model_dir)
    texts = context_df["org_entity_context"].fillna("").astype(str).tolist()
    pred_id, conf, score = batched_predict(texts, tokenizer, model, device)
    out = context_df.copy()
    out["sent_label"] = [model.config.id2label[int(i)] for i in pred_id]
    out["sent_conf"] = conf
    out["sent_score"] = score
    return out


# ── Cleaning helper ───────────────────────────────────────────────────────────
def _clean_for_analysis(df: pd.DataFrame, cols_to_strip: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols_to_strip:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    if "industry" in out.columns:
        out = out[~out["industry"].str.contains("Other/Unclustered", case=False, na=False)]
        out = out[~out["industry"].str.lower().isin(BAD_VALS)]
    return out


# ── Industry-level summary + composite impact score ───────────────────────────
def _min_max(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def industry_impact_summary(sent_df: pd.DataFrame, ent_df: pd.DataFrame,
                            weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)
                            ) -> pd.DataFrame:
    """Composite impact score = w_doc·doc + w_ctx·context + w_org·orgs + w_tech·tech."""
    org_clean = _clean_for_analysis(sent_df, ["industry", "org_entity"])
    ent_clean = _clean_for_analysis(ent_df, ["industry", "org_entity", "tech_entity"])
    ent_clean.loc[ent_clean["tech_entity"].astype(str).str.lower().isin(BAD_VALS), "tech_entity"] = np.nan

    docs = (
        ent_clean.groupby("industry", as_index=False)
        .agg(n_docs=("row_id", "nunique"))
    )
    contexts = (
        org_clean.groupby("industry", as_index=False)
        .agg(n_contexts=("row_id", "size"), n_orgs=("org_entity", "nunique"))
    )
    techs = (
        ent_clean.dropna(subset=["tech_entity"]).groupby("industry", as_index=False)
        .agg(n_techs=("tech_entity", "nunique"))
    )

    summary = (
        docs.merge(contexts, on="industry", how="outer")
        .merge(techs, on="industry", how="left")
        .fillna(0)
    )
    for c in ("n_docs", "n_contexts", "n_orgs", "n_techs"):
        summary[c] = summary[c].astype(int)

    summary["doc_score"] = _min_max(summary["n_docs"])
    summary["context_score"] = _min_max(summary["n_contexts"])
    summary["org_score"] = _min_max(summary["n_orgs"])
    summary["tech_score"] = _min_max(summary["n_techs"])

    w_doc, w_ctx, w_org, w_tech = weights
    summary["impact_score"] = (
        w_doc * summary["doc_score"]
        + w_ctx * summary["context_score"]
        + w_org * summary["org_score"]
        + w_tech * summary["tech_score"]
    )
    return summary.sort_values("impact_score", ascending=False).reset_index(drop=True)


def industry_sentiment_composition(sent_df: pd.DataFrame) -> pd.DataFrame:
    """Per-industry share of positive / neutral / negative / unclear contexts."""
    df = _clean_for_analysis(sent_df, ["industry", "sent_label"])
    df["sent_label"] = df["sent_label"].str.lower()
    df = df[df["sent_label"].isin(SENT_LABELS)]

    pivot = (
        df.groupby(["industry", "sent_label"]).size()
        .unstack(fill_value=0).reindex(columns=SENT_LABELS, fill_value=0)
    )
    return pivot.div(pivot.sum(axis=1), axis=0)


# ── Company-level ─────────────────────────────────────────────────────────────
def company_summary(sent_df: pd.DataFrame, top_per_industry: int = 20) -> pd.DataFrame:
    df = _clean_for_analysis(sent_df, ["industry", "org_entity"])
    df = df[~df["org_entity"].astype(str).str.lower().isin(BAD_VALS)]
    grouped = (
        df.groupby(["industry", "org_entity"], as_index=False)
        .agg(n_docs=("row_id", "nunique"), n_contexts=("row_id", "size"),
             mean_sent_score=("sent_score", "mean"))
        .sort_values(["industry", "n_docs", "n_contexts"], ascending=[True, False, False])
    )
    return grouped.groupby("industry", group_keys=False).head(top_per_industry).reset_index(drop=True)


# ── Technology-level ──────────────────────────────────────────────────────────
def technology_summary(sent_df: pd.DataFrame, ent_long_df: pd.DataFrame,
                       min_docs: int = 20) -> pd.DataFrame:
    """Mean sentiment score and label-share per technology."""
    tech_long = ent_long_df.query("entity_type == 'TECH'")[["row_id", "industry", "entity_text"]] \
        .rename(columns={"entity_text": "tech_entity"})
    tech_long = _clean_for_analysis(tech_long, ["industry", "tech_entity"])

    s = _clean_for_analysis(sent_df, ["industry"])[["row_id", "industry",
                                                    "sent_label", "sent_score", "sent_conf"]]
    s["sent_label"] = s["sent_label"].str.lower()
    s = s[s["sent_label"].isin(SENT_LABELS)]

    merged = tech_long.merge(s, on=["row_id", "industry"], how="inner")
    base = (
        merged.groupby("tech_entity", as_index=False)
        .agg(n_docs=("row_id", "nunique"), n_contexts=("row_id", "size"),
             mean_sent_score=("sent_score", "mean"), mean_conf=("sent_conf", "mean"))
    )
    pivot = (
        merged.groupby(["tech_entity", "sent_label"]).size().unstack(fill_value=0)
        .reindex(columns=SENT_LABELS, fill_value=0)
    )
    shares = pivot.div(pivot.sum(axis=1), axis=0).add_suffix("_share").reset_index()
    return base.merge(shares, on="tech_entity").query("n_docs >= @min_docs").reset_index(drop=True)


# ── Adoption drivers (rule-based) ─────────────────────────────────────────────
_COMPILED_DRIVERS = {
    name: [re.compile(p, flags=re.I) for p in pats]
    for name, pats in ADOPTION_DRIVER_PATTERNS.items()
}


def match_drivers(text) -> list[str]:
    text = "" if pd.isna(text) else str(text)
    return [name for name, pats in _COMPILED_DRIVERS.items() if any(p.search(text) for p in pats)]


def driver_summary(sent_df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver document/context counts, sentiment composition, mean score."""
    df = _clean_for_analysis(sent_df, ["industry", "org_entity", "sent_label", "org_entity_context"])
    df["sent_label"] = df["sent_label"].str.lower()
    df = df[df["sent_label"].isin(SENT_LABELS)]
    df["driver_list"] = df["org_entity_context"].apply(match_drivers)
    long = df.explode("driver_list").dropna(subset=["driver_list"])

    base = (
        long.groupby("driver_list", as_index=False)
        .agg(n_contexts=("row_id", "size"), n_docs=("row_id", "nunique"),
             mean_sent_score=("sent_score", "mean"))
    )
    pivot = (
        long.groupby(["driver_list", "sent_label"]).size().unstack(fill_value=0)
        .reindex(columns=SENT_LABELS, fill_value=0)
    )
    shares = pivot.div(pivot.sum(axis=1), axis=0).add_suffix("_share").reset_index()
    return base.merge(shares, on="driver_list") \
        .sort_values("n_docs", ascending=False).reset_index(drop=True)


# ── Time-series ───────────────────────────────────────────────────────────────
def industry_monthly_sentiment(sent_df: pd.DataFrame,
                               top_n_industries: int = 6) -> pd.DataFrame:
    """Monthly mean sentiment score for the top-N industries by volume."""
    df = _clean_for_analysis(sent_df, ["industry", "sent_label"])
    df["sent_label"] = df["sent_label"].str.lower()
    df["sent_score"] = pd.to_numeric(df["sent_score"], errors="coerce")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date_parsed"].dt.to_period("M").astype(str)

    top = (df.groupby("industry").size().sort_values(ascending=False)
           .head(top_n_industries).index)
    df = df[df["industry"].isin(top)]
    return (
        df.groupby(["month", "industry"], as_index=False)
        .agg(n_contexts=("row_id", "size"), mean_sent_score=("sent_score", "mean"))
        .sort_values(["industry", "month"]).reset_index(drop=True)
    )


# ── End-to-end orchestration ──────────────────────────────────────────────────
def run(model_dir: Path = DEFAULT_MODEL_DIR) -> dict:
    """Run inference and produce all aggregation tables. Returns dict of DataFrames."""
    context_df = pd.read_parquet(SENTIMENT_DIR / "org_entity_context.parquet")
    ent_wide = pd.read_parquet(ENTITY_DIR / "entity_extract.parquet")
    ent_long = pd.read_parquet(ENTITY_DIR / "doc_entities.parquet")

    annotated = annotate_with_sentiment(context_df, model_dir=model_dir)
    annotated.to_parquet(SENTIMENT_DIR / "org_entity_context_with_sentiment.parquet", index=False)

    tables = {
        "industry_impact": industry_impact_summary(annotated, ent_wide),
        "industry_sentiment": industry_sentiment_composition(annotated).reset_index(),
        "company": company_summary(annotated),
        "technology": technology_summary(annotated, ent_long),
        "drivers": driver_summary(annotated),
        "monthly": industry_monthly_sentiment(annotated),
    }
    for name, frame in tables.items():
        frame.to_parquet(SENTIMENT_DIR / f"summary_{name}.parquet", index=False)
        print(f"Wrote summary_{name}.parquet ({len(frame)} rows)")
    return tables
