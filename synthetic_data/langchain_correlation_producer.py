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

from base_correlation_producer import BaseCorrelationProducer
from langchain_bridge import render_prompt


logger = logging.getLogger(__name__)


def _resolve_external_python() -> Optional[str]:
    """Locate a Python 3.12 interpreter for the LangChain bridge."""

    candidates: List[Optional[Path]] = []

    override = os.getenv("SYNTHETIC_DATA_LANGCHAIN_PYTHON")
    if override:
        candidates.append(Path(override))

    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent
    candidates.append(repo_root / ".venv-3.12/bin/python")

    python_path = shutil.which("python3.12")
    if python_path:
        candidates.append(Path(python_path))

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except FileNotFoundError:
            continue
        if os.access(resolved, os.X_OK):
            return str(resolved)

    return None


ChatOpenAI = None  # type: ignore[assignment]
_NATIVE_LANGCHAIN_AVAILABLE = False

if sys.version_info < (3, 14):  # Native imports only when interpreter is supported
    try:  # pragma: no cover - import path verified during integration
        from langchain_openai import ChatOpenAI  # type: ignore
        _NATIVE_LANGCHAIN_AVAILABLE = True
    except Exception as import_error:  # pragma: no cover - optional dependency missing
        logger.debug("Native LangChain imports unavailable: %s", import_error)


def get_langchain_bridge_runtime() -> Optional[str]:
    """Return the Python executable that will execute LangChain prompts."""

    return _resolve_external_python()


LANGCHAIN_CORRELATION_AVAILABLE = True  # Availability validated at instantiation time


class LangChainCorrelationProducer(BaseCorrelationProducer):
    """Generates correlations by synthesizing cross-signal insights via LangChain."""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.1) -> None:
        super().__init__("langchain_correlation")
        self.model_name = model
        self.temperature = temperature
        self.logger = logging.getLogger("synthetic_data.correlation.langchain")
        self.external_python = get_langchain_bridge_runtime()
        bridge_available = self.external_python is not None
        try:
            self.bridge_timeout = int(os.getenv("SYNTHETIC_DATA_LANGCHAIN_TIMEOUT", "30"))
        except ValueError:
            self.bridge_timeout = 30
            self.logger.warning(
                "Invalid SYNTHETIC_DATA_LANGCHAIN_TIMEOUT; defaulting to %s seconds",
                self.bridge_timeout,
            )

        if _NATIVE_LANGCHAIN_AVAILABLE:
            self.mode = "native"
            self.llm = self._initialise_native_llm()
        elif bridge_available:
            self.mode = "bridge"
            self.llm = None
            self.logger.info(
                "LangChain correlations proxied via interpreter: %s",
                self.external_python,
            )
        else:
            self.mode = "disabled"
            self.llm = None
            self.logger.warning(
                "LangChain runtime unavailable; falling back to deterministic correlations."
            )

    def _initialise_native_llm(self):  # pragma: no cover - requires external service
        if ChatOpenAI is None:
            return None
        try:
            return ChatOpenAI(model=self.model_name, temperature=self.temperature, max_tokens=600)
        except Exception as exc:
            self.logger.warning("Failed to initialise native LangChain LLM: %s", exc)
            return None

    def analyze_correlations(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not findings:
            return []

        candidate_refs = self._collect_candidate_refs(findings)
        if len(candidate_refs) < 2:
            return []

        correlation_payload = self._attempt_langchain_correlation(candidate_refs)
        if correlation_payload is None:
            correlation_payload = self._fallback_correlation(candidate_refs)

        correlation = self._create_correlation_finding(
            title=correlation_payload["title"],
            description=correlation_payload["description"],
            severity=correlation_payload["severity"],
            risk_score=int(correlation_payload["risk_score"]),
            related_findings=correlation_payload["related_ids"],
            correlation_type=correlation_payload.get("correlation_type", "langchain_inferred"),
            metadata={
                "insight": correlation_payload.get("insight", ""),
                "generator": "langchain",
            },
        )

        return [correlation]

    def _collect_candidate_refs(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for category, items in findings.items():
            for item in items:
                enriched.append({
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "severity": item.get("severity", "info"),
                    "risk_score": item.get("risk_score", 0),
                    "category": category,
                    "description": item.get("description", ""),
                })
        enriched.sort(key=lambda item: item.get("risk_score", 0), reverse=True)
        return enriched[:6]

    def _attempt_langchain_correlation(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self.mode == "native" and self.llm:  # pragma: no cover - external service
            prompt_text = render_prompt(candidates)
            try:
                completion = self.llm.invoke(prompt_text)
                raw_text = completion.content if hasattr(completion, "content") else str(completion)
                payload = json.loads(raw_text)
            except Exception as exc:
                self.logger.warning("Native LangChain call failed: %s", exc)
                return None

            required_fields = {"title", "description", "severity", "risk_score", "related_ids"}
            if not required_fields.issubset(payload) or len(payload.get("related_ids", [])) < 2:
                self.logger.debug("Native LangChain payload missing required fields")
                return None
            return payload

        if self.mode == "bridge" and self.external_python:
            return self._invoke_external_langchain(candidates)

        return None

    def _invoke_external_langchain(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        payload = {
            "candidates": candidates,
            "model": self.model_name,
            "temperature": self.temperature,
        }

        command = [self.external_python, "-m", "synthetic_data.langchain_bridge"]

        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.bridge_timeout,
                env=os.environ.copy(),
                check=False,
            )
        except Exception as exc:  # pragma: no cover - subprocess errors are rare
            self.logger.warning("Failed to invoke LangChain bridge: %s", exc)
            return None

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            self.logger.warning(
                "LangChain bridge exited with code %s: %s",
                completed.returncode,
                stderr,
            )
            return None

        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            self.logger.warning("LangChain bridge returned invalid JSON payload")
            return None

    def _fallback_correlation(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        top = candidates[:3]
        related_ids = [item["id"] for item in top if item.get("id")]
        severity = max((item.get("severity", "medium") for item in top), key=self._severity_rank)
        risk_score = max(item.get("risk_score", 60) for item in top)

        insight = "; ".join(item.get("title", "") for item in top)

        return {
            "title": "Multi-vector campaign detected",
            "description": "Automated correlation grouping top risk findings across scanners.",
            "severity": severity,
            "risk_score": risk_score,
            "correlation_type": "aggregate_high_risk",
            "related_ids": related_ids,
            "insight": insight,
        }

    @staticmethod
    def _severity_rank(severity: str) -> int:
        ranking = {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}
        return ranking.get(severity, 3)
