#!/usr/bin/env python3
"""
scripts/run_evaluation.py
--------------------------
Runs both bots against all test queries and prints a comparison report.

Usage:
    python scripts/run_evaluation.py
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

from src.phase1_hallucination.hallucination_bot import HallucinationBot
from src.phase2_grounded.grounded_bot import GroundedBot
from src.phase2_grounded.vector_store import get_or_build_vector_store
from src.data_processing.document_loader import load_all_clean_documents
from src.evaluation.evaluator import run_evaluation, print_report, save_report


def main():
    print("Initializing bots...")

    bot1 = HallucinationBot()

    docs = load_all_clean_documents()
    vs = get_or_build_vector_store(docs)
    bot2 = GroundedBot(vs)

    print("Running evaluation...")
    results = run_evaluation(bot1, bot2)

    print_report(results)
    save_report(results)


if __name__ == "__main__":
    main()
