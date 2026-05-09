"""CLI: clean and filter the raw corpus to AI-relevant documents.

Reads:  data/raw/news_final_project.parquet  (downloaded if missing)
Writes: data/cleaned/news_clean_filtered.parquet

Usage: python scripts/02_clean.py
"""
from src.data.audit import load_raw
from src.data.clean_filter import clean_and_filter, save_clean


def main():
    raw = load_raw()
    print(f"Loaded raw corpus: {len(raw):,} rows")
    clean = clean_and_filter(raw, drop_non_english=True)
    save_clean(clean)


if __name__ == "__main__":
    main()
