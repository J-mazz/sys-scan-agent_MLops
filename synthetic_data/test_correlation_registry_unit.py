from typing import Any, Dict, List

import pytest

from synthetic_data.base_correlation_producer import BaseCorrelationProducer
import synthetic_data.correlation_registry as registry_module
from synthetic_data.correlation_registry import CorrelationRegistry


class StubCorrelationProducer(BaseCorrelationProducer):
    def __init__(self) -> None:
        super().__init__("stub")
        self.invocations = 0

    def analyze_correlations(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        self.invocations += 1
        return [
            {
                "id": "stub-corr",
                "title": "Stub correlation",
                "severity": "medium",
                "risk_score": 40,
                "description": "Stubbed correlation output",
                "metadata": {},
                "category": "correlation",
                "tags": ["correlation"],
                "correlation_refs": [item["id"] for items in findings.values() for item in items],
                "correlation_type": "stub",
            }
        ]


@pytest.fixture
def minimal_findings() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "process": [
            {
                "id": "proc-1",
                "title": "Process",
                "severity": "high",
                "risk_score": 80,
                "description": "",
                "metadata": {},
                "category": "process",
                "tags": ["process"],
            }
        ]
    }


def test_analyze_all_correlations_uses_sequential_fallback(monkeypatch, minimal_findings):
    registry = CorrelationRegistry()
    stub = StubCorrelationProducer()
    registry.correlation_producers = {"stub": stub}

    monkeypatch.setattr(registry_module, "PARALLEL_AVAILABLE", False)

    results = registry.analyze_all_correlations(minimal_findings)

    assert stub.invocations == 1
    assert results[0]["correlation_type"] == "stub"


def test_enable_langchain_is_noop_after_removal():
    """LangChain path is deprecated; enabling should be a no-op."""

    registry = CorrelationRegistry()
    before = set(registry.correlation_producers.keys())

    registry.enable_langchain(True)
    after_enable = set(registry.correlation_producers.keys())

    registry.enable_langchain(False)
    after_disable = set(registry.correlation_producers.keys())

    assert before == after_enable == after_disable


def test_register_and_get_correlation_producer_handles_missing(monkeypatch):
    registry = CorrelationRegistry()

    class TempProducer(BaseCorrelationProducer):
        def __init__(self) -> None:
            super().__init__("temp")

        def analyze_correlations(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:  # pragma: no cover - not used
            return []

    temp = TempProducer()
    registry.register_correlation_producer("temp", temp)
    assert registry.get_correlation_producer("temp") is temp

    producers = registry.list_correlation_producers()
    assert "temp" in producers

    with pytest.raises(ValueError):
        registry.get_correlation_producer("missing")


def test_correlation_summary_counts_by_type_and_severity():
    registry = CorrelationRegistry()

    correlations = [
        {
            "title": "High priority",
            "severity": "high",
            "risk_score": 88,
            "correlation_type": "process_network",
        },
        {
            "title": "Medium priority",
            "severity": "medium",
            "risk_score": 45,
            "correlation_type": "filesystem",
        },
    ]

    summary = registry.get_correlation_summary(correlations)

    assert summary["total_correlations"] == 2
    assert summary["severity_distribution"]["high"] == 1
    assert summary["correlation_types"]["filesystem"] == 1
    top_entry = summary["top_correlations"][0]
    assert top_entry["risk_score"] == 88
