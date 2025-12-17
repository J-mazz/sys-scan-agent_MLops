import random
from datetime import datetime
from typing import Any, Dict, List

import pytest

from synthetic_data.base_correlation_producer import BaseCorrelationProducer


class DummyCorrelationProducer(BaseCorrelationProducer):
    def __init__(self) -> None:
        super().__init__("dummy")

    def analyze_correlations(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return []


def sample_findings() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "process": [
            {
                "id": "proc-1",
                "title": "Suspicious process",
                "severity": "high",
                "risk_score": 80,
                "description": "",
                "metadata": {"command": "/bin/bash"},
                "category": "process",
                "tags": ["process"],
            },
            {
                "id": "proc-2",
                "title": "Benign process",
                "severity": "low",
                "risk_score": 20,
                "description": "",
                "metadata": {"command": "/usr/bin/cron"},
                "category": "process",
                "tags": ["process"],
            },
        ],
        "network": [
            {
                "id": "net-1",
                "title": "Outbound connection",
                "severity": "medium",
                "risk_score": 55,
                "description": "",
                "metadata": {"destination": "10.0.0.5"},
                "category": "network",
                "tags": ["network"],
            }
        ],
    }


def test_matches_criteria_supports_nested_dictionaries():
    producer = DummyCorrelationProducer()
    finding = sample_findings()["process"][0]

    criteria = {
        "metadata": {"command": "/bin/bash"},
        "severity": "high",
    }

    assert producer._matches_criteria(finding, criteria) is True

    negative = {
        "metadata": {"command": "/bin/sh"},
    }

    assert producer._matches_criteria(finding, negative) is False

    # Lists require the finding value to match exactly one of the criteria entries.
    # Because tags are stored as a list, the criteria below should evaluate to False.
    assert producer._matches_criteria(finding, {"tags": ["process"]}) is False


def test_find_related_findings_filters_by_criteria():
    producer = DummyCorrelationProducer()
    findings = sample_findings()

    criteria = {"metadata": {"command": "/usr/bin/cron"}}
    related = producer._find_related_findings(findings, criteria)

    assert related == ["proc-2"]


def test_create_correlation_finding_injects_expected_fields():
    random.seed(1337)
    producer = DummyCorrelationProducer()

    correlation = producer._create_correlation_finding(
        title="Linked behaviour",
        description="Process and network overlap",
        severity="high",
        risk_score=75,
        related_findings=["proc-1", "net-1"],
        correlation_type="process_network",
        metadata={"source": "unit-test"},
    )

    assert correlation["title"] == "Linked behaviour"
    assert correlation["risk_total"] == 75
    assert correlation["correlation_strength"] > 0.1
    assert correlation["metadata"]["source"] == "unit-test"

    timestamp = datetime.fromisoformat(correlation["timestamp"])
    assert isinstance(timestamp, datetime)

    subscores = correlation["risk_subscores"]
    assert 0.0 <= subscores["impact"] <= 1.0
    assert correlation["graph_degree"] == 2


def test_calculate_correlation_strength_scales_with_related_ids():
    random.seed(2024)
    producer = DummyCorrelationProducer()

    strength_two = producer._calculate_correlation_strength(["a", "b"])
    random.seed(2024)
    strength_three = producer._calculate_correlation_strength(["a", "b", "c"])

    assert strength_three >= strength_two


def test_base_class_analyze_correlations_not_implemented():
    producer = BaseCorrelationProducer("base")

    with pytest.raises(NotImplementedError):
        producer.analyze_correlations({})
