"""
evaluation/evaluator.py
------------------------
Side-by-side evaluation engine.

Runs both the HallucinationBot and the GroundedBot against the same set
of test queries and produces a comparison report showing:
  - Phase 1 answer (no grounding)
  - Phase 2 answer (RAG grounded)
  - Ground truth (from clean data)
  - A simple heuristic accuracy flag
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from src.config_loader import get_config

logger = logging.getLogger(__name__)


# Lightweight ground truth built from clean data for scoring
GROUND_TRUTH: Dict[str, str] = {
    "what is the price of product a?": "$110",
    "tell me about product b.": "high-end professional gadget",
    "is product g available?": "i don't know",
    "what category is product c in?": "accessories",
    "which products are in the budget category?": "product e, product h",
    "who supplies product d?": "supplierw",
    "what is the stock level of product e?": "150",
}


def _heuristic_correct(answer: str, truth: str) -> bool:
    """
    Very simple heuristic: check if the key fact appears in the answer.
    Not a substitute for human evaluation or LLM-as-judge.
    """
    answer_lower = answer.lower()
    for fact in truth.lower().split(","):
        if fact.strip() in answer_lower:
            return True
    return False


def run_evaluation(
    hallucination_bot,
    grounded_bot,
    queries: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Run both bots on all queries and return comparison results.

    Args:
        hallucination_bot: HallucinationBot instance.
        grounded_bot: GroundedBot instance.
        queries: List of questions. Defaults to config test_queries.

    Returns:
        List of result dicts, one per query.
    """
    if queries is None:
        queries = get_config()["evaluation"]["test_queries"]

    results = []
    for i, query in enumerate(queries, 1):
        logger.info(f"[Eval {i}/{len(queries)}] Query: {query!r}")

        phase1_result = hallucination_bot.ask(query)
        phase2_result = grounded_bot.ask(query)

        truth = GROUND_TRUTH.get(query.lower().strip(), "N/A")
        p1_correct = _heuristic_correct(phase1_result["answer"], truth) if truth != "N/A" else None
        p2_correct = _heuristic_correct(phase2_result["answer"], truth) if truth != "N/A" else None

        results.append({
            "query": query,
            "phase1_answer": phase1_result["answer"],
            "phase2_answer": phase2_result["answer"],
            "ground_truth": truth,
            "phase1_correct": p1_correct,
            "phase2_correct": p2_correct,
            "phase2_sources": phase2_result.get("sources", []),
        })

    return results


def print_report(results: List[Dict]) -> None:
    """Print a human-readable comparison report to stdout."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  GARBAGE IN → HALLUCINATION OUT  |  EVALUATION REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    p1_score = sum(1 for r in results if r["phase1_correct"] is True)
    p2_score = sum(1 for r in results if r["phase2_correct"] is True)
    scored = sum(1 for r in results if r["phase1_correct"] is not None)

    for i, r in enumerate(results, 1):
        print(f"\n[Query {i}] {r['query']}")
        print(f"  Ground Truth : {r['ground_truth']}")
        print(f"  Phase 1 (❌ No Grounding) : {r['phase1_answer'][:200]}")
        p1_flag = "✅" if r["phase1_correct"] else ("❌" if r["phase1_correct"] is False else "–")
        print(f"    → Correct? {p1_flag}")
        print(f"  Phase 2 (✅ RAG Grounded) : {r['phase2_answer'][:200]}")
        p2_flag = "✅" if r["phase2_correct"] else ("❌" if r["phase2_correct"] is False else "–")
        print(f"    → Correct? {p2_flag}")
        if r["phase2_sources"]:
            print(f"    → Sources: {r['phase2_sources']}")

    print(f"\n{sep}")
    print(f"  SUMMARY  |  Evaluated {scored}/{len(results)} queries with ground truth")
    print(f"  Phase 1 (Hallucination) Accuracy : {p1_score}/{scored}")
    print(f"  Phase 2 (Grounded RAG)  Accuracy : {p2_score}/{scored}")
    print(sep)


def save_report(results: List[Dict], output_path: str = "logs/eval_report.json") -> None:
    """Save evaluation results as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Evaluation report saved to: {output_path}")
