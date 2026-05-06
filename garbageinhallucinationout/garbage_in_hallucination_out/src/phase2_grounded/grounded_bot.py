"""
phase2_grounded/grounded_bot.py
--------------------------------
Phase 2: The Grounded Bot (RAG).

Demonstrates how clean data + retrieval grounding eliminates hallucination:
  - FAISS vector store retrieves the most relevant clean documents
  - Strict prompt: "Answer ONLY from context; say 'I don't know' otherwise"
  - Expected outcome: accurate, verifiable, source-cited answers
"""

import logging
from typing import List

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from src.llm_factory import get_llm
from src.config_loader import get_config

logger = logging.getLogger(__name__)


GROUNDED_PROMPT_TEMPLATE = """You are a precise product information assistant.

STRICT RULES:
1. Answer ONLY using the information provided in the context below.
2. If the answer is not found in the context, say exactly: "I don't know — this information is not in my knowledge base."
3. Do NOT make up prices, descriptions, or any other facts.
4. Be concise and factual.

Context:
{context}

Question: {query}

Answer:"""


class GroundedBot:
    """
    Phase 2 bot: RAG-powered, grounded on clean data.
    Uses retrieval from a FAISS vector store + strict prompt guardrails.
    """

    def __init__(self, vector_store: FAISS):
        """
        Args:
            vector_store: Pre-built FAISS vector store loaded with clean documents.
        """
        self.llm = get_llm()
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": get_config().get("rag", {}).get("top_k", 3)}
        )
        logger.info("GroundedBot initialized (RAG + strict grounding).")

    def _retrieve_context(self, query: str) -> tuple[str, List[Document]]:
        """
        Retrieve top-k relevant documents for the query.

        Returns:
            Tuple of (formatted_context_string, list_of_documents)
        """
        docs = self.retriever.invoke(query)
        context_parts = [f"[Doc {i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
        context = "\n\n".join(context_parts)
        return context, docs

    def ask(self, query: str) -> dict:
        """
        Ask the bot a question with full RAG grounding.

        Args:
            query: The user question.

        Returns:
            dict with keys: query, answer, mode, sources, context
        """
        context, source_docs = self._retrieve_context(query)
        prompt = GROUNDED_PROMPT_TEMPLATE.format(context=context, query=query)

        logger.debug(f"[Phase 2] Retrieved {len(source_docs)} docs for: {query!r}")

        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"[Phase 2] LLM error: {e}")
            answer = f"Error: {e}"

        sources = [doc.metadata.get("source", "unknown") for doc in source_docs]

        result = {
            "query": query,
            "answer": answer,
            "mode": "Phase 2 — RAG Grounded (Clean Data)",
            "sources": list(set(sources)),
            "context": context,
            "retrieved_docs": len(source_docs),
        }

        logger.info(f"[Phase 2] Q: {query!r} | A: {answer[:80]}...")
        return result
