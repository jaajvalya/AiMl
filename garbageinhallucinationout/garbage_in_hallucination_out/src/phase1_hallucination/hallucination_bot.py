"""
phase1_hallucination/hallucination_bot.py
-----------------------------------------
Phase 1: The Hallucination Bot.

Demonstrates what happens when an LLM is fed bad or no context:
- No retrieval grounding
- Loose prompt — model fills gaps from its internal (often wrong) knowledge
- Expected outcome: confident but incorrect or invented answers
"""

import logging
from src.llm_factory import get_llm

logger = logging.getLogger(__name__)


HALLUCINATION_PROMPT_TEMPLATE = """Answer the following question based on your general knowledge.
Be confident and provide as much detail as you can.

Question: {query}

Answer:"""


class HallucinationBot:
    """
    Phase 1 bot: No RAG, no grounding.
    Uses only the LLM's parametric memory → high hallucination risk.
    """

    def __init__(self):
        self.llm = get_llm()
        logger.info("HallucinationBot initialized (no retrieval, no grounding).")

    def ask(self, query: str) -> dict:
        """
        Ask the bot a question with no grounding.

        Args:
            query: The user question.

        Returns:
            dict with keys: query, answer, mode, warning
        """
        prompt = HALLUCINATION_PROMPT_TEMPLATE.format(query=query)
        logger.debug(f"[Phase 1] Prompt: {prompt}")

        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"[Phase 1] LLM error: {e}")
            answer = f"Error: {e}"

        result = {
            "query": query,
            "answer": answer,
            "mode": "Phase 1 — No Grounding (Hallucination Risk)",
            "warning": "⚠️  Answer is NOT grounded. The model may be hallucinating.",
        }

        logger.info(f"[Phase 1] Q: {query!r} | A: {answer[:80]}...")
        return result
