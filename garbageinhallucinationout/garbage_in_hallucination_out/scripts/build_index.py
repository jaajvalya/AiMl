#!/usr/bin/env python3
"""
scripts/build_index.py
-----------------------
One-time script to build the FAISS vector store from clean data.

Usage:
    python scripts/build_index.py
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from src.data_processing.document_loader import load_all_clean_documents
from src.phase2_grounded.vector_store import build_vector_store


def main():
    print("=" * 60)
    print("  Building FAISS Vector Store from Clean Data")
    print("=" * 60)

    print("\n[1/2] Loading clean documents...")
    docs = load_all_clean_documents()
    print(f"      Loaded {len(docs)} documents.")

    print("\n[2/2] Embedding and building FAISS index...")
    build_vector_store(docs, save=True)
    print("      Index saved successfully.\n")

    print("✅ Vector store build complete.")
    print("   You can now run: streamlit run src/ui/app.py")


if __name__ == "__main__":
    main()
