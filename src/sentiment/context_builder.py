"""Build entity-centered context windows from extracted ORG entities.

For each (document, ORG-mention) pair, we extract a sentence-aware window
(previous + target + next sentence), capped at ~1200 chars, with the
organization mention surrounded by `<ENT>...</ENT>` tags. This becomes the
input unit for both LLM weak labeling and the trained sentiment model.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import spacy

_NLP_SENT = None  # lazy-loaded sentencizer


def _get_sentencizer():
    global _NLP_SENT
    if _NLP_SENT is None:
        _NLP_SENT = spacy.blank("en")
        _NLP_SENT.add_pipe("sentencizer")
    return _NLP_SENT


# ── Helpers for span column normalization ─────────────────────────────────────
def _force_pylist(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _ensure_tuple_span(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return (int(x[0]), int(x[1]))
    if isinstance(x, str):
        m = re.findall(r"-?\d+", x)
        if len(m) >= 2:
            return (int(m[0]), int(m[1]))
    return None


def _normalize_spans_list(spans_cell):
    return [t for t in (_ensure_tuple_span(s) for s in _force_pylist(spans_cell)) if t]


def _get_sent_bounds(text: str) -> list[tuple[int, int]]:
    if not text or not str(text).strip():
        return []
    return [(s.start_char, s.end_char) for s in _get_sentencizer()(str(text)).sents]


# ── Single-context construction ───────────────────────────────────────────────
def build_entity_context(text: str, span: tuple[int, int],
                         sent_bounds: list[tuple[int, int]],
                         tag_open: str = "<ENT>",
                         tag_close: str = "</ENT>",
                         max_chars: int | None = 1200) -> dict:
    """Wrap the entity span in <ENT> tags and pull a sentence-aware window."""
    if not text or not str(text).strip():
        return {"context": None, "status": "empty_text"}
    s = str(text)
    span = _ensure_tuple_span(span)
    if span is None:
        return {"context": None, "status": "bad_span"}
    start, end = span
    if start < 0 or end < 0 or start >= len(s) or end > len(s) or start >= end:
        return {"context": None, "status": "span_out_of_range"}

    def _fallback_window(reason: str) -> dict:
        left = max(0, start - 200)
        right = min(len(s), end + 200)
        frag = s[left:right]
        rs, re_ = start - left, end - left
        return {
            "context": frag[:rs] + tag_open + frag[rs:re_] + tag_close + frag[re_:],
            "status": reason,
        }

    if not sent_bounds:
        return _fallback_window("no_sents_fallback_window")

    sent_idx = next(
        (i for i, (a, b) in enumerate(sent_bounds) if a <= start < b),
        None,
    )
    if sent_idx is None:
        return _fallback_window("span_not_in_sentence_fallback_window")

    prev_idx = max(0, sent_idx - 1)
    next_idx = min(len(sent_bounds) - 1, sent_idx + 1)
    ctx_start, ctx_end = sent_bounds[prev_idx][0], sent_bounds[next_idx][1]
    frag = s[ctx_start:ctx_end]
    rs, re_ = start - ctx_start, end - ctx_start
    tagged = frag[:rs] + tag_open + frag[rs:re_] + tag_close + frag[re_:]

    if max_chars and len(tagged) > max_chars:
        center = (rs + re_) // 2
        half = max_chars // 2
        tagged = tagged[max(0, center - half): min(len(tagged), center + half)]

    return {"context": tagged, "status": "ok"}


# ── Dataset-level construction ────────────────────────────────────────────────
def build_org_context_dataset(df: pd.DataFrame,
                              entity_col: str = "org_entity",
                              span_col: str = "org_entity_spans",
                              text_col: str = "text_clean_light",
                              keep_cols: list[str] | None = None,
                              max_chars: int = 1200,
                              dedup: bool = True) -> pd.DataFrame:
    """Explode (entity, span) pairs and attach a context window to each one."""
    keep_cols = keep_cols or ["row_id", "date", "language", "url", "industry"]

    base = df[keep_cols + [text_col, entity_col, span_col]].copy()
    base[entity_col] = base[entity_col].apply(_force_pylist)
    base[span_col] = base[span_col].apply(_normalize_spans_list)
    base["pairs"] = [
        list(zip(e, s))[: min(len(e), len(s))]
        for e, s in zip(base[entity_col], base[span_col])
    ]
    base = base.drop(columns=[entity_col, span_col])

    long = base.explode("pairs", ignore_index=True).dropna(subset=["pairs"]).copy()
    pair_df = pd.DataFrame(long["pairs"].tolist(), index=long.index,
                           columns=[entity_col, span_col])
    long = pd.concat([long.drop(columns=["pairs"]), pair_df], axis=1)
    long[entity_col] = long[entity_col].astype(str).str.strip()
    long = long[long[entity_col].ne("")].copy()
    long[span_col] = long[span_col].apply(_ensure_tuple_span)
    long = long.dropna(subset=[span_col]).copy()

    if dedup:
        long["_norm"] = long[entity_col].str.lower()
        long = long.drop_duplicates(["row_id", "_norm", span_col]).drop(columns=["_norm"])

    def _per_doc(g: pd.DataFrame) -> pd.DataFrame:
        s = g[text_col].iloc[0]
        bounds = _get_sent_bounds(s)
        contexts, statuses = [], []
        for sp in g[span_col]:
            r = build_entity_context(s, sp, bounds, max_chars=max_chars)
            contexts.append(r["context"])
            statuses.append(r["status"])
        g = g.copy()
        g["org_entity_context"] = contexts
        g["status"] = statuses
        return g

    out = long.groupby("row_id", group_keys=False, sort=False).apply(_per_doc)
    return out[[
        "row_id", "industry", entity_col, span_col,
        "org_entity_context", "status", "date", "url", "language",
    ]].rename(columns={entity_col: "org_entity", span_col: "org_entity_spans"})
