"""
data_processing/document_loader.py
------------------------------------
Converts raw and clean data files into LangChain Document objects
ready for embedding and vector store ingestion.

Supports:
  - Plain text knowledge files (one paragraph per product)
  - CSV product data (each row → a Document with metadata)
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List
from langchain_core.documents import Document

from src.config_loader import get_config

logger = logging.getLogger(__name__)


def load_text_documents(path: str) -> List[Document]:
    """
    Load a plain-text knowledge file.
    Each non-empty paragraph becomes one Document.

    Args:
        path: Path to the .txt knowledge file.

    Returns:
        List of LangChain Document objects.
    """
    logger.info(f"Loading text documents from: {path}")
    text = Path(path).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    docs = [
        Document(page_content=para, metadata={"source": str(path), "index": i})
        for i, para in enumerate(paragraphs)
    ]
    logger.info(f"Loaded {len(docs)} text documents.")
    return docs


def load_csv_documents(path: str) -> List[Document]:
    """
    Load a product CSV file.
    Each row becomes one Document; all columns are included in page_content.

    Args:
        path: Path to the CSV file.

    Returns:
        List of LangChain Document objects.
    """
    logger.info(f"Loading CSV documents from: {path}")
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        content_parts = []
        for col, val in row.items():
            if pd.notna(val):
                content_parts.append(f"{col}: {val}")
        content = ". ".join(content_parts) + "."
        metadata = {"source": str(path), "product_id": str(row.get("product_id", "unknown"))}
        docs.append(Document(page_content=content, metadata=metadata))
    logger.info(f"Loaded {len(docs)} CSV documents.")
    return docs


def load_all_clean_documents() -> List[Document]:
    """
    Convenience loader: loads both clean knowledge text and clean CSV.

    Returns:
        Combined list of Documents from all clean sources.
    """
    cfg = get_config()["data"]
    docs = []
    docs.extend(load_text_documents(cfg["clean_knowledge"]))
    docs.extend(load_csv_documents(cfg["clean_csv"]))
    logger.info(f"Total clean documents loaded: {len(docs)}")
    return docs


def load_all_raw_documents() -> List[Document]:
    """
    Convenience loader: loads both raw (bad) knowledge text and raw CSV.

    Returns:
        Combined list of Documents from all raw sources.
    """
    cfg = get_config()["data"]
    docs = []
    docs.extend(load_text_documents(cfg["raw_knowledge"]))
    docs.extend(load_csv_documents(cfg["raw_csv"]))
    logger.info(f"Total raw documents loaded: {len(docs)}")
    return docs
