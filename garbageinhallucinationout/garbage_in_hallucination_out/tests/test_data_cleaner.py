"""
tests/test_data_cleaner.py
---------------------------
Unit tests for the data_processing.data_cleaner module.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.data_processing.data_cleaner import (
    clean_dataframe,
    get_quality_report,
    _standardize_text_fields,
    _handle_nulls,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "product_id": ["P001", "P001", "P002", "P003", None],
        "product_name": ["Product A", "product-a", "Product B", "Product C", None],
        "price": ["100", "120", "200", "75", None],
        "category": ["Electronics", "electronics", "Premium Devices", "accessories", None],
        "description": [None, "Smart device", "High end gadget", "Portable charger", None],
        "stock": ["50", "null", "30", "200", None],
        "supplier": ["SupplierX", "SupplierX", "SupplierZ", "SupplierX", None],
        "last_updated": ["2024-01-01", "2024-01-10", "2024-01-03", "2024-01-08", None],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHandleNulls:
    def test_null_strings_replaced(self):
        df = pd.DataFrame({"a": ["null", "NULL", "", " ", "real"]})
        result = _handle_nulls(df)
        assert result["a"].iloc[0] is np.nan or pd.isna(result["a"].iloc[0])
        assert result["a"].iloc[4] == "real"


class TestStandardizeText:
    def test_product_name_title_case(self):
        df = pd.DataFrame({"product_name": ["product-a", "product_b"]})
        result = _standardize_text_fields(df)
        assert result["product_name"].iloc[0] == "Product A"
        assert result["product_name"].iloc[1] == "Product B"

    def test_category_title_case(self):
        df = pd.DataFrame({"category": ["electronics", "BUDGET"]})
        result = _standardize_text_fields(df)
        assert result["category"].iloc[0] == "Electronics"
        assert result["category"].iloc[1] == "Budget"


class TestCleanDataframe:
    def test_output_rows_less_than_input(self, raw_df):
        clean = clean_dataframe(raw_df.copy())
        assert len(clean) < len(raw_df)

    def test_null_product_id_dropped(self, raw_df):
        clean = clean_dataframe(raw_df.copy())
        assert clean["product_id"].notna().all()

    def test_price_conflict_resolved_as_median(self, raw_df):
        clean = clean_dataframe(raw_df.copy())
        p001 = clean[clean["product_id"] == "P001"]
        assert len(p001) == 1
        assert p001.iloc[0]["price"] == 110.0  # median(100, 120)

    def test_one_golden_record_per_product(self, raw_df):
        clean = clean_dataframe(raw_df.copy())
        assert clean["product_id"].nunique() == len(clean)

    def test_verified_flag_set(self, raw_df):
        clean = clean_dataframe(raw_df.copy())
        assert (clean["verified"] == True).all()

    def test_source_is_golden_record(self, raw_df):
        clean = clean_dataframe(raw_df.copy())
        assert (clean["source"] == "golden_record").all()


class TestQualityReport:
    def test_report_keys_present(self, raw_df):
        clean = clean_dataframe(raw_df.copy())
        report = get_quality_report(raw_df, clean)
        required_keys = [
            "raw_rows", "clean_rows", "raw_null_count",
            "clean_null_count", "null_reduction_pct",
        ]
        for key in required_keys:
            assert key in report

    def test_clean_rows_leq_raw(self, raw_df):
        clean = clean_dataframe(raw_df.copy())
        report = get_quality_report(raw_df, clean)
        assert report["clean_rows"] <= report["raw_rows"]
