"""CLI: weak-label entity contexts and fine-tune RoBERTa for 4-class sentiment.

Two phases:
  1. Build entity-context windows and label them via the OpenAI API
     (requires OPENAI_API_KEY to be set; cached to JSONL for resume).
  2. Fine-tune `roberta-base` with class-weighted CE loss on the labels.

Reads:  data/entity/entity_extract.parquet
Writes: data/sentiment/labeled_org_entity_context.parquet
        outputs/sentiment/roberta-base-news-sentiment/  (fine-tuned model)

Usage: python scripts/05_train_sentiment.py [--skip-label]
"""
import argparse

from src.sentiment import train, weak_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-label", action="store_true",
        help="Skip the (slow + paid) LLM labeling step; assumes labels already exist.",
    )
    args = parser.parse_args()

    if not args.skip_label:
        weak_label.run()

    metrics = train.train()
    print(f"\nFinal val metrics: {metrics}")


if __name__ == "__main__":
    main()
