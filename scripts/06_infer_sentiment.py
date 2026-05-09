"""CLI: run sentiment inference and produce industry/company/tech/driver summaries.

Reads:  data/sentiment/org_entity_context.parquet
        data/entity/entity_extract.parquet
        data/entity/doc_entities.parquet
        outputs/sentiment/roberta-base-news-sentiment/
Writes: data/sentiment/org_entity_context_with_sentiment.parquet
        data/sentiment/summary_*.parquet  (industry/company/tech/drivers/monthly)

Usage: python scripts/06_infer_sentiment.py
"""
from src.sentiment.infer import run


def main():
    tables = run()
    print("\n=== Top 10 industries by composite impact ===")
    print(tables["industry_impact"].head(10).to_string())
    print("\n=== Top 10 adoption drivers ===")
    print(tables["drivers"].head(10).to_string())


if __name__ == "__main__":
    main()
