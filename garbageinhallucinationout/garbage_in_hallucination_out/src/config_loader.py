"""
config_loader.py
----------------
Loads and provides access to project configuration from config/config.yaml.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Optional path override. Defaults to config/config.yaml
                     relative to the project root.

    Returns:
        dict: Parsed configuration dictionary.
    """
    if config_path is None:
        root = Path(__file__).resolve().parent.parent
        config_path = root / "config" / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Singleton config loaded once at import time
_config: dict = None


def get_config() -> dict:
    """Return the singleton config, loading it on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
