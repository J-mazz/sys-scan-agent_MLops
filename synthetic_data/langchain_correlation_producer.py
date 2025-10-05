"""LangChain-assisted correlation producer."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from base_correlation_producer import BaseCorrelationProducer

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    LANGCHAIN_CORRELATION_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PromptTemplate = None  # type: ignore
    ChatOpenAI = None  # type: ignore
    LANGCHAIN_CORRELATION_AVAILABLE = False


class LangChainCorrelationProducer(BaseCorrelationProducer):
    """Generates correlations by synthesizing cross-signal insights via LangChain."""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.1) -> None:
        super().__init__("langchain_correlation")
        self.model_name = model
        self.temperature = temperature
        self.llm = self._initialise_llm() if LANGCHAIN_CORRELATION_AVAILABLE else None
        self.prompt = self._build_prompt() if LANGCHAIN_CORRELATION_AVAILABLE else None

    def _initialise_llm(self):  # pragma: no cover - requires external service
        if ChatOpenAI is None:
            return None
        try:
            return ChatOpenAI(model=self.model_name, temperature=self.temperature, max_tokens=600)
        except Exception:
            return None

    def _build_prompt(self):  # pragma: no cover - requires langchain
        if PromptTemplate is None:
            return None
        template = (
            "You are a SOC analyst creating a high-value correlation. "
            "Use the provided findings to write ONE correlation in strict JSON with fields: "
            "title (string), description (string), severity (info|low|medium|high|critical), "
            "risk_score (integer 0-100), correlation_type (string), insight (string summarising the link). "
            "Ensure you reference at least two finding IDs in a field named related_ids (array of strings)."
            "\n\nContext:\n{context}\n\nJSON:"
        )
        return PromptTemplate.from_template(template)

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
        if not self.llm or not self.prompt:  # LangChain not ready
            return None

        context_lines = [
            f"- {cand['id']} ({cand['severity']}, {cand['risk_score']}): {cand['title']}"
            for cand in candidates
        ]
        context = "\n".join(context_lines)

        try:  # pragma: no cover - depends on external LLM
            completion = self.llm.invoke(self.prompt.format(context=context))
            if hasattr(completion, "content"):
                raw_text = completion.content  # chat model result
            else:
                raw_text = str(completion)
            payload = json.loads(raw_text)
        except Exception:
            return None

        # Basic validation
        required_fields = {"title", "description", "severity", "risk_score", "related_ids"}
        if not required_fields.issubset(payload):
            return None
        if len(payload.get("related_ids", [])) < 2:
            return None
        return payload

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
