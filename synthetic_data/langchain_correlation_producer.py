"""LangChain-assisted correlation producer."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_correlation_producer import BaseCorrelationProducer
from .langchain_bridge import render_prompt


logger = logging.getLogger(__name__)


def _resolve_external_python() -> Optional[str]:
    """Locate a Python 3.12 interpreter for the LangChain bridge."""

    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent

    override = os.getenv("SYNTHETIC_DATA_LANGCHAIN_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    repo_python = (repo_root / ".venv-3.12/bin/python").expanduser()
    if repo_python.exists() and os.access(repo_python, os.X_OK):
        return str(repo_python)
"""Legacy LangChain correlation producer removed."""

from .base_correlation_producer import BaseCorrelationProducer


class LangChainCorrelationProducer(BaseCorrelationProducer):
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "LangChainCorrelationProducer has been removed; legacy agentic path is unsupported."
        )
