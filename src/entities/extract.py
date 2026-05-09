"""Stage 04 — entity extraction (organizations + technologies).

Hybrid approach:
  - Organizations: `dslim/distilbert-NER` over 2K-char overlapping chunks,
    aggregation_strategy='first', confidence ≥ 0.75.
  - Technologies: spaCy `PhraseMatcher` over a 65-term hand-curated lexicon
    centered on AI outcomes & adoption modes.

Outputs two parquets in `data/entity/`:
  - `entity_extract.parquet`: original docs + per-row org/tech lists
  - `doc_entities.parquet`:   long-format (one row per entity mention)
"""
from __future__ import annotations

import re

import pandas as pd
import spacy
import torch
from datasets import Dataset
from spacy.matcher import PhraseMatcher
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset

from src.config import BERTOPIC_DIR, ENTITY_DIR, TECH_LEXICON


# ── Defaults ──────────────────────────────────────────────────────────────────
NER_MODEL_NAME = "dslim/distilbert-NER"
CHUNK_SIZE = 2000
OVERLAP = 120
BATCH_SIZE = 64
SCORE_THRESHOLD = 0.75


# ── Chunking ──────────────────────────────────────────────────────────────────
def segment_text_by_chars(text, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """Char-based sliding-window chunking with overlap to preserve span context."""
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    segments, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        segments.append({"chunk_text": text[start:end], "chunk_start": start, "chunk_end": end})
        if end == n:
            break
        start = end - overlap
    return segments


# ── Org NER post-processing ───────────────────────────────────────────────────
def _clean_entity_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _is_valid_org_text(text: str) -> bool:
    text = _clean_entity_text(text)
    if len(text) <= 1 or not re.search(r"[A-Za-z]", text):
        return False
    return text.lower() not in {"the", "a", "an"}


def _extract_orgs_from_chunk(ner_result, chunk_start: int,
                             score_threshold: float) -> list[dict]:
    out = []
    if not isinstance(ner_result, list):
        return out
    for ent in ner_result:
        label = ent.get("entity_group", ent.get("entity", ""))
        score = float(ent.get("score", 0.0))
        text = _clean_entity_text(ent.get("word", ""))
        if label == "ORG" and score >= score_threshold and _is_valid_org_text(text):
            out.append({
                "entity_text": text,
                "start": int(ent["start"]) + chunk_start,
                "end": int(ent["end"]) + chunk_start,
                "entity_type": "ORG",
                "score": score,
                "source": "distilbert_ner_segmented_first",
            })
    return out


def _dedupe_org_records(records: list[dict]) -> list[dict]:
    seen, out = set(), []
    for rec in records:
        key = (rec["entity_text"].strip().lower(), int(rec["start"]), int(rec["end"]))
        if key not in seen:
            seen.add(key)
            out.append(rec)
    return sorted(out, key=lambda r: (r["start"], r["end"], r["entity_text"].lower()))


# ── Org extraction pipeline ───────────────────────────────────────────────────
def extract_organizations(df: pd.DataFrame, text_col: str = "text_clean_light",
                          score_threshold: float = SCORE_THRESHOLD,
                          batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """Returns the input frame with `org_entity` and `org_entity_spans` columns."""
    out = df.copy()
    out[text_col] = (
        out[text_col].fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )

    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    print(f"NER device: {'GPU' if device == 0 else 'CPU'}")

    ner_pipe = pipeline(
        task="ner", model=model, tokenizer=tokenizer,
        aggregation_strategy="first", device=device,
    )

    # 1) Build chunks
    chunk_rows = []
    for row_id, text in enumerate(tqdm(out[text_col].tolist(), desc="Chunking")):
        for seg_id, seg in enumerate(segment_text_by_chars(text)):
            chunk_rows.append({
                "row_id": row_id, "seg_id": seg_id,
                "chunk_text": seg["chunk_text"],
                "chunk_start": seg["chunk_start"], "chunk_end": seg["chunk_end"],
            })
    chunks = pd.DataFrame(chunk_rows)

    # 2) Run NER batched over the chunk dataset
    ds = Dataset.from_pandas(chunks[["chunk_text"]], preserve_index=False)
    chunks["ner_result"] = list(tqdm(
        ner_pipe(KeyDataset(ds, "chunk_text"), batch_size=batch_size),
        total=len(ds), desc="NER",
    ))

    # 3) Aggregate to document level
    chunks["org_records"] = chunks.apply(
        lambda r: _extract_orgs_from_chunk(
            list(r["ner_result"]) if not isinstance(r["ner_result"], list) else r["ner_result"],
            r["chunk_start"], score_threshold,
        ),
        axis=1,
    )

    doc_org_map: dict[int, list[dict]] = {}
    for _, r in chunks.iterrows():
        doc_org_map.setdefault(int(r["row_id"]), []).extend(r["org_records"])

    org_lists, span_lists = [], []
    for row_id in range(len(out)):
        recs = _dedupe_org_records(doc_org_map.get(row_id, []))
        org_lists.append([r["entity_text"] for r in recs])
        span_lists.append([(r["start"], r["end"]) for r in recs])

    out["org_entity"] = org_lists
    out["org_entity_spans"] = span_lists
    return out


# ── Tech extraction (spaCy PhraseMatcher) ─────────────────────────────────────
def _build_tech_matcher() -> tuple[spacy.language.Language, PhraseMatcher]:
    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("TECH", [nlp.make_doc(t) for t in TECH_LEXICON])
    return nlp, matcher


def _extract_tech(text, nlp, matcher) -> dict:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return {"tech_entity": [], "tech_entity_span": []}
    doc = nlp(text)
    seen, names, spans = set(), [], []
    for _, start, end in matcher(doc):
        span = doc[start:end]
        key = (span.text.strip().lower(), int(span.start_char), int(span.end_char))
        if key in seen:
            continue
        seen.add(key)
        names.append(span.text)
        spans.append((int(span.start_char), int(span.end_char)))
    return {"tech_entity": names, "tech_entity_span": spans}


def extract_technologies(df: pd.DataFrame, text_col: str = "model_text") -> pd.DataFrame:
    nlp, matcher = _build_tech_matcher()
    extractions = df[text_col].apply(lambda t: _extract_tech(t, nlp, matcher))
    out = df.copy()
    out["tech_entity"] = extractions.map(lambda d: d["tech_entity"])
    out["tech_entity_span"] = extractions.map(lambda d: d["tech_entity_span"])
    return out


# ── Long-form / persistence ───────────────────────────────────────────────────
def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Explode org/tech lists into one-row-per-mention long format."""
    rows = []
    for _, row in df.iterrows():
        rid = row["row_id"]
        for ents, spans, etype, source in [
            (row.get("org_entity", []), row.get("org_entity_spans", []), "ORG", "distilbert_ner_segmented_first"),
            (row.get("tech_entity", []), row.get("tech_entity_span", []), "TECH", "lexicon"),
        ]:
            if not isinstance(ents, list):
                continue
            for ent_text, span in zip(ents, spans):
                rows.append({
                    "row_id": rid, "entity_text": ent_text,
                    "start": int(span[0]), "end": int(span[1]),
                    "entity_type": etype, "source": source,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["row_id", "start", "end", "entity_type", "entity_text"]
        ).reset_index(drop=True)
    return out


def run(assignments_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full extraction: orgs + tech → wide and long parquets."""
    if assignments_df is None:
        assignments_df = pd.read_parquet(BERTOPIC_DIR / "bertopic_assigned.parquet")

    wide = extract_organizations(assignments_df)
    wide = extract_technologies(wide)
    long = to_long_format(wide)

    wide.to_parquet(ENTITY_DIR / "entity_extract.parquet", index=False)
    long.to_parquet(ENTITY_DIR / "doc_entities.parquet", index=False)
    print(f"Wrote {len(wide):,} docs and {len(long):,} entity mentions → {ENTITY_DIR}")
    return wide, long


def load_entities() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(ENTITY_DIR / "entity_extract.parquet"),
        pd.read_parquet(ENTITY_DIR / "doc_entities.parquet"),
    )
