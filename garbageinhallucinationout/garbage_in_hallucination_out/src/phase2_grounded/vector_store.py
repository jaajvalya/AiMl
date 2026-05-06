"""
phase2_grounded/vector_store.py
--------------------------------
Builds and manages the FAISS vector store for RAG retrieval.

Provides:
  - build_vector_store(): embed documents and save index to disk
  - load_vector_store(): load a pre-built index from disk
"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.llm_factory import get_embeddings
from src.config_loader import get_config

logger = logging.getLogger(__name__)


def build_vector_store(documents: List[Document], save: bool = True) -> FAISS:
    """
    Embed documents and build a FAISS vector store.

    Args:
        documents: List of LangChain Document objects to index.
        save: Whether to persist the index to disk.

    Returns:
        A FAISS vector store instance.
    """
    logger.info(f"Building vector store from {len(documents)} documents...")
    embeddings = get_embeddings()
    db = FAISS.from_documents(documents, embeddings)

    if save:
        index_path = get_config()["vector_store"]["index_path"]
        Path(index_path).mkdir(parents=True, exist_ok=True)
        db.save_local(index_path)
        logger.info(f"Vector store saved to: {index_path}")

    return db


def load_vector_store() -> FAISS:
    """
    Load a persisted FAISS vector store from disk.

    Returns:
        A FAISS vector store instance.

    Raises:
        FileNotFoundError: If the index path does not exist.
    """
    index_path = get_config()["vector_store"]["index_path"]
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"No vector store found at '{index_path}'. "
            "Run `python scripts/build_index.py` first."
        )
    embeddings = get_embeddings()
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    logger.info(f"Vector store loaded from: {index_path}")
    return db


def get_or_build_vector_store(documents: List[Document] = None) -> FAISS:
    """
    Load from disk if available, otherwise build from provided documents.

    Args:
        documents: Documents to use if building is required.

    Returns:
        A ready FAISS vector store.
    """
    index_path = get_config()["vector_store"]["index_path"]
    if Path(index_path).exists():
        return load_vector_store()
    if documents is None:
        raise ValueError("No index found and no documents provided to build one.")
    return build_vector_store(documents)
