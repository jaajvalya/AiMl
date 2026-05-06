"""
data_processing/data_cleaner.py
--------------------------------
Data Engineering discipline module.

Applies the transformations that turn "bad data" into "clean data":
  - Deduplication
  - Schema standardization (column names, types, casing)
  - Null handling
  - Conflict resolution via golden record logic (median price, majority values)
  - Metadata tagging (source, verified flag)

This module is the core of Phase 2's data quality improvement.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.config_loader import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names."""
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _standardize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Title-case product names; lowercase categories."""
    if "product_name" in df.columns:
        df["product_name"] = (
            df["product_name"]
            .str.strip()
            .str.replace(r"[-_]", " ", regex=True)
            .str.title()
        )
    if "category" in df.columns:
        df["category"] = df["category"].str.strip().str.title()
    if "description" in df.columns:
        df["description"] = df["description"].str.strip()
    return df


def _handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Replace empty strings / 'null' strings with actual NaN."""
    df.replace(["null", "NULL", "None", "", " "], np.nan, inplace=True)
    return df


def _resolve_conflicts(group: pd.DataFrame) -> pd.Series:
    """
    Golden record logic for a group of rows sharing the same product_id.

    - price: median of non-null values (robust to outliers)
    - description: longest non-null string (most complete)
    - stock: max non-null value (most conservative — don't undersell)
    - supplier: mode (most frequently cited)
    - category: mode
    - last_updated: most recent date
    - product_name: mode (most common name)
    """
    record = {}
    # group.name == the product_id value (pandas excludes the groupby key from the group DataFrame)
    record["product_id"] = group.name

    # product_name — most common (after standardization)
    record["product_name"] = (
        group["product_name"].dropna().mode().iloc[0]
        if not group["product_name"].dropna().empty
        else np.nan
    )

    # price — median
    prices = pd.to_numeric(group["price"], errors="coerce").dropna()
    record["price"] = round(prices.median(), 2) if not prices.empty else np.nan

    # category — mode
    record["category"] = (
        group["category"].dropna().mode().iloc[0]
        if not group["category"].dropna().empty
        else np.nan
    )

    # description — longest non-null
    descs = group["description"].dropna()
    record["description"] = descs.loc[descs.str.len().idxmax()] if not descs.empty else np.nan

    # stock — max non-null
    stocks = pd.to_numeric(group["stock"], errors="coerce").dropna()
    record["stock"] = int(stocks.max()) if not stocks.empty else np.nan

    # supplier — mode
    record["supplier"] = (
        group["supplier"].dropna().mode().iloc[0]
        if not group["supplier"].dropna().empty
        else np.nan
    )

    # last_updated — most recent
    dates = pd.to_datetime(group["last_updated"], errors="coerce").dropna()
    record["last_updated"] = dates.max().strftime("%Y-%m-%d") if not dates.empty else np.nan

    # Governance metadata
    record["source"] = "golden_record"
    record["verified"] = True

    return pd.Series(record)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on a raw product DataFrame.

    Steps:
        1. Standardize column names
        2. Standardize text fields
        3. Handle nulls
        4. Drop fully empty rows
        5. Resolve conflicts → golden records

    Args:
        df: Raw product DataFrame.

    Returns:
        Cleaned DataFrame with one golden record per product_id.
    """
    logger.info(f"Starting data cleaning. Input shape: {df.shape}")

    df = _standardize_columns(df)
    df = _standardize_text_fields(df)
    df = _handle_nulls(df)

    # Drop rows with no product_id at all
    before = len(df)
    df = df.dropna(subset=["product_id"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with null product_id.")

    # Deduplicate + conflict resolution via groupby → golden record
    clean_df = (
        df.groupby("product_id", sort=False)
        .apply(_resolve_conflicts)
        .reset_index(drop=True)
    )

    logger.info(f"Cleaning complete. Output shape: {clean_df.shape}")
    return clean_df


def load_raw_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load raw CSV data from path or config default."""
    if path is None:
        path = get_config()["data"]["raw_csv"]
    logger.info(f"Loading raw data from: {path}")
    return pd.read_csv(path, dtype=str)


def save_clean_data(df: pd.DataFrame, path: Optional[str] = None) -> None:
    """Save cleaned DataFrame to CSV."""
    if path is None:
        path = get_config()["data"]["clean_csv"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Clean data saved to: {path}")


def get_quality_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """
    Generate a before/after data quality summary.

    Args:
        raw_df: Original raw DataFrame.
        clean_df: Cleaned DataFrame.

    Returns:
        dict with quality metrics.
    """
    raw_nulls = raw_df.isnull().sum().sum()
    clean_nulls = clean_df.isnull().sum().sum()

    return {
        "raw_rows": len(raw_df),
        "clean_rows": len(clean_df),
        "duplicates_removed": len(raw_df) - len(raw_df.drop_duplicates()),
        "raw_null_count": int(raw_nulls),
        "clean_null_count": int(clean_nulls),
        "null_reduction_pct": round((1 - clean_nulls / max(raw_nulls, 1)) * 100, 1),
        "unique_products_before": raw_df["product_id"].nunique() if "product_id" in raw_df.columns else "N/A",
        "unique_products_after": len(clean_df),
    }
