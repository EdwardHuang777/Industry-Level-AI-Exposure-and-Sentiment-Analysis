"""CLI: profile the raw corpus and print summary statistics.

Usage: python scripts/01_audit.py
"""
from src.data.audit import run_audit


def main():
    result = run_audit()
    print(f"\n=== Shape: {result['shape']} ===\n")
    print("Missingness (top 10):")
    print(result["missingness"].head(10).to_string())
    print("\nText profile (top 10):")
    print(result["text_profile"].head(10).to_string())
    print("\nTop domains:")
    print(result["top_domains"].head(10).to_string())
    print(f"\nDate quality: {result['date_quality']}")
    print(f"\nDuplicates: {result['duplicates']}")


if __name__ == "__main__":
    main()
