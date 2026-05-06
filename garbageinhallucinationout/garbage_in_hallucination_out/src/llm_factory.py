"""
llm_factory.py
--------------
Factory module for creating LLM and Embeddings instances based on config.
Supports OpenAI (hosted) and Ollama (local) providers.
"""

import os
from typing import Any

from src.config_loader import get_config


def get_llm() -> Any:
    """Return a configured LLM instance based on config.

    Raises:
        ValueError: If the provider in config is not supported.

    Returns:
        An LLM instance (OpenAI or Ollama).
    """
    cfg = get_config()["llm"]
    provider = cfg["provider"].lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 512),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    elif provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=cfg["model"],
            base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
            temperature=cfg.get("temperature", 0.0),
        )

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Use 'openai' or 'ollama'.")


def get_embeddings() -> Any:
    """Return a configured Embeddings instance based on config.

    Raises:
        ValueError: If the provider in config is not supported.

    Returns:
        An Embeddings instance (OpenAI or Ollama).
    """
    cfg = get_config()["embeddings"]
    provider = cfg["provider"].lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=cfg["model"],
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    elif provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(
            model=cfg.get("ollama_model", "nomic-embed-text"),
            base_url=get_config()["llm"].get("ollama_base_url", "http://localhost:11434"),
        )

    else:
        raise ValueError(f"Unsupported embeddings provider: '{provider}'. Use 'openai' or 'ollama'.")
