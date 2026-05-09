"""CLI: extract organizations (DistilBERT-NER) and technologies (PhraseMatcher).

Reads:  data/bertopic/bertopic_assigned.parquet
Writes: data/entity/entity_extract.parquet  (wide, one row per doc)
        data/entity/doc_entities.parquet     (long, one row per mention)

Usage: python scripts/04_entities.py
"""
from src.entities.extract import run


def main():
    run()


if __name__ == "__main__":
    main()
