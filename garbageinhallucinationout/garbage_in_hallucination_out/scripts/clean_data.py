#!/usr/bin/env python3
"""
scripts/clean_data.py
---------------------
Runs the data cleaning pipeline and prints a quality report.

Usage:
    python scripts/clean_data.py
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from src.data_processing.data_cleaner import (
    load_raw_data,
    clean_dataframe,
    save_clean_data,
    get_quality_report,
)


def main():
    print("=" * 60)
    print("  Data Cleaning Pipeline")
    print("=" * 60)

    print("\n[1/3] Loading raw data...")
    raw_df = load_raw_data()
    print(raw_df.to_string())

    print("\n[2/3] Applying cleaning transformations...")
    clean_df = clean_dataframe(raw_df)

    print("\n[3/3] Saving clean data and quality report...")
    save_clean_data(clean_df)

    report = get_quality_report(raw_df, clean_df)
    print("\n── Quality Report ──────────────────────────────────────")
    for key, val in report.items():
        print(f"  {key:<35} {val}")

    print("\n── Clean (Golden) Records ──────────────────────────────")
    print(clean_df.to_string())
    print("\n✅ Data cleaning complete.")


if __name__ == "__main__":
    main()
