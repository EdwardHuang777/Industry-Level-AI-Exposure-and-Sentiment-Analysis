"""CLI: fit BERTopic and map topics to industries.

Reads:  data/cleaned/news_clean_filtered.parquet
Writes: outputs/bertopic_model/  +  data/bertopic/bertopic_assigned.parquet

Usage: python scripts/03_topics.py
"""
from src.data.clean_filter import load_clean
from src.topics.bertopic_pipeline import run


def main():
    clean = load_clean()
    print(f"Loaded {len(clean):,} cleaned rows")
    run(clean)


if __name__ == "__main__":
    main()
