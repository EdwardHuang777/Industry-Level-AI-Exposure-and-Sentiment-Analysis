"""LLM-driven weak labeling for entity-level sentiment.

Calls the OpenAI Responses API per entity-context window with a strict
4-class rubric (positive / neutral / negative / unclear) and a thread-pooled
worker. Labels are cached to JSONL so repeated runs short-circuit on cache
hits — important because the labeling pass is O(50K) requests.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import openai
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

from src.config import ENTITY_DIR, SENTIMENT_DIR, SENTIMENT_MODEL_DIR
from src.sentiment.context_builder import build_org_context_dataset

ALLOWED = {"positive", "neutral", "negative", "unclear"}
DEFAULT_LABEL_MODEL = "gpt-5-nano"
LABEL_CACHE_PATH = SENTIMENT_MODEL_DIR / "llm_doc_sentiment_cache.jsonl"
SAMPLE_SIZE = 50_000


# ── Prompting / parsing ──────────────────────────────────────────────────────
LABEL_PROMPT = (
    "You are a helpful and precise assistant for analyzing the sentiment of "
    "sentence(s) extracted from AI-related News articles.\n"
    "Your job is to create a sentiment label (tone) to this sentence. Especially "
    "the attitude towards the impact of AI on the industry and organization.\n"
    "The organization in the sentence is marked with <ENT> and </ENT> tags. "
    "Focus on the sentiment towards the entity.\n"
    "Rules:\n"
    "- Judge overall tone of the coverage.\n"
    "- If the article is mostly factual or mixed without a clear tilt, choose neutral.\n"
    "- If sentiment cannot be inferred, choose unclear.\n"
    "- Output exactly ONE word from: positive, neutral, negative, unclear.\n"
    "- No punctuation. No explanation.\n\n"
    "Text:\n{text}"
)


def _normalize_label(raw: str) -> str:
    raw = (raw or "").strip().lower()
    if not raw:
        return "unclear"
    token = raw.split()[0].strip(" .,:;!\"'`()[]{}<>")
    if token in ALLOWED:
        return token
    for w in raw.replace("\n", " ").split():
        w = w.strip(" .,:;!\"'`()[]{}<>").lower()
        if w in ALLOWED:
            return w
    return "unclear"


def _parse_retry_wait(msg: str, default: float = 2.0) -> float:
    if not msg:
        return default
    m = re.search(r"try again in\s+(\d+(?:\.\d+)?)ms", msg, flags=re.I)
    if m:
        return max(float(m.group(1)) / 1000.0, 0.2)
    m = re.search(r"try again in\s+(\d+(?:\.\d+)?)s", msg, flags=re.I)
    if m:
        return max(float(m.group(1)), 0.2)
    return default


# ── Cache I/O ────────────────────────────────────────────────────────────────
def _make_key(row_id: int, text: str) -> str:
    raw = (str(row_id) + "||" + str(text)).encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()


def _load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cache[obj["key"]] = obj["label_obj"]
    return cache


# ── Single-call labeler ──────────────────────────────────────────────────────
def label_one(client: OpenAI, text: str,
              model: str = DEFAULT_LABEL_MODEL,
              max_attempts: int = 8) -> str:
    prompt = LABEL_PROMPT.format(text=text)
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.responses.create(model=model, input=prompt)
            return _normalize_label(resp.output_text)
        except openai.RateLimitError as exc:
            wait = _parse_retry_wait(str(exc), default=min(2 ** min(attempt, 6), 20))
            time.sleep(wait + random.uniform(0, 0.3))
        except Exception:
            if attempt == max_attempts:
                return "unclear"
            time.sleep(min(2 ** min(attempt, 5), 10) + random.uniform(0, 0.3))
    return "unclear"


# ── Parallel labeling pass ───────────────────────────────────────────────────
def label_pool(pool_df: pd.DataFrame, *, model: str = DEFAULT_LABEL_MODEL,
               max_workers: int = 10, flush_every: int = 100,
               cache_path: Path = LABEL_CACHE_PATH) -> pd.DataFrame:
    """Label every row in `pool_df`. Resumable via the JSONL cache on disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)
    cache_lock, file_lock = Lock(), Lock()
    buffer: list[dict] = []

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var is required for weak labeling.")
    client = OpenAI(timeout=30.0, max_retries=2)

    def flush():
        nonlocal buffer
        if not buffer:
            return
        with file_lock, open(cache_path, "a", encoding="utf-8") as f:
            for obj in buffer:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        buffer = []

    def worker(idx_row):
        idx, r = idx_row
        text, row_id = str(r.org_entity_context), int(r.row_id)
        k = _make_key(row_id, text)

        with cache_lock:
            if k in cache:
                return idx, _build_record(r, cache[k])

        lab = label_one(client, text, model=model)
        with cache_lock:
            if k in cache:
                lab = cache[k]
            else:
                cache[k] = lab
                buffer.append({"key": k, "label_obj": lab})
                if len(buffer) >= flush_every:
                    flush()
        return idx, _build_record(r, lab)

    rows = list(pool_df.itertuples(index=False))
    results: list = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, (i, row)) for i, row in enumerate(rows)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM labeling"):
            idx, rec = fut.result()
            results[idx] = rec

    with cache_lock:
        flush()
    return pd.DataFrame(results)


def _build_record(row, label: str) -> dict:
    return {
        "row_id": int(row.row_id),
        "industry": getattr(row, "industry", None),
        "org_entity_text": str(row.org_entity_context),
        "labels": label,
    }


# ── End-to-end ────────────────────────────────────────────────────────────────
def run(entity_extract_df: pd.DataFrame | None = None,
        sample_size: int = SAMPLE_SIZE,
        random_state: int = 42) -> pd.DataFrame:
    """Full pipeline: build org-contexts → sample → label → save."""
    if entity_extract_df is None:
        entity_extract_df = pd.read_parquet(ENTITY_DIR / "entity_extract.parquet")

    contexts = build_org_context_dataset(entity_extract_df)
    contexts.to_parquet(SENTIMENT_DIR / "org_entity_context.parquet", index=False)
    print(f"Built {len(contexts):,} entity contexts → {SENTIMENT_DIR}/org_entity_context.parquet")

    pool = (
        contexts.dropna(subset=["org_entity_context"])
        .sample(n=min(sample_size, len(contexts)), random_state=random_state)
        .reset_index(drop=True)
    )
    print(f"Sampling {len(pool):,} contexts for labeling...")

    labeled = label_pool(pool)
    out_path = SENTIMENT_DIR / "labeled_org_entity_context.parquet"
    labeled.to_parquet(out_path, index=False)
    print(f"Wrote {len(labeled):,} labeled rows → {out_path}")
    print(labeled["labels"].value_counts())
    return labeled
