"""
tests/test_document_loader.py
------------------------------
Unit tests for the data_processing.document_loader module.
"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.data_processing.document_loader import load_text_documents, load_csv_documents


@pytest.fixture
def tmp_text_file(tmp_path):
    p = tmp_path / "knowledge.txt"
    p.write_text(
        "Product A costs $110. It is a smart device.\n\nProduct B costs $200. It is premium."
    )
    return str(p)


@pytest.fixture
def tmp_csv_file(tmp_path):
    p = tmp_path / "products.csv"
    p.write_text(
        "product_id,product_name,price\nP001,Product A,110\nP002,Product B,200\n"
    )
    return str(p)


class TestLoadTextDocuments:
    def test_returns_list_of_documents(self, tmp_text_file):
        docs = load_text_documents(tmp_text_file)
        assert isinstance(docs, list)
        assert len(docs) == 2

    def test_document_has_content(self, tmp_text_file):
        docs = load_text_documents(tmp_text_file)
        assert "Product A" in docs[0].page_content

    def test_metadata_contains_source(self, tmp_text_file):
        docs = load_text_documents(tmp_text_file)
        assert "source" in docs[0].metadata


class TestLoadCsvDocuments:
    def test_returns_list_of_documents(self, tmp_csv_file):
        docs = load_csv_documents(tmp_csv_file)
        assert len(docs) == 2

    def test_document_contains_column_values(self, tmp_csv_file):
        docs = load_csv_documents(tmp_csv_file)
        assert "P001" in docs[0].page_content or "Product A" in docs[0].page_content

    def test_metadata_contains_product_id(self, tmp_csv_file):
        docs = load_csv_documents(tmp_csv_file)
        assert docs[0].metadata.get("product_id") == "P001"
