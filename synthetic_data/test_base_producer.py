import random
from typing import Any, Dict, List

import pytest

from synthetic_data.base_producer import BaseProducer


class SimpleProducer(BaseProducer):
    def __init__(self) -> None:
        super().__init__("simple")

    def generate_findings(self, count: int = 1) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for index in range(count):
            finding_id = f"simple-{index}"
            findings.append(
                self._generate_base_finding(
                    finding_id=finding_id,
                    title="Test finding",
                    severity="high" if index % 2 == 0 else "low",
                    risk_score=80 if index % 2 == 0 else 25,
                    base_severity_score=80 if index % 2 == 0 else 25,
                    description="Generated for unit testing",
                    metadata={"source": "unit"},
                )
            )
        return findings


def test_generate_base_finding_populates_expected_fields():
    random.seed(42)
    producer = SimpleProducer()
    finding = producer.generate_findings(1)[0]

    assert finding["category"] == "simple"
    assert finding["probability_actionable"] <= 1.0
    assert finding["risk_subscores"]["confidence"] <= 0.95
    assert finding["baseline_status"] in {"new", "existing", "unknown"}


def test_generate_tags_and_risk_subscores_vary_by_severity():
    random.seed(123)
    producer = SimpleProducer()

    high_tags = producer._generate_tags("high")
    low_tags = producer._generate_tags("low")

    assert "high_priority" in high_tags
    assert "low_priority" in low_tags

    high_scores = producer._generate_risk_subscores("high")
    low_scores = producer._generate_risk_subscores("low")

    assert high_scores["impact"] >= low_scores["impact"]
    assert 0.0 <= low_scores["impact"] <= 0.4


def test_calculate_probability_actionable_respects_bounds():
    random.seed(31)
    producer = SimpleProducer()

    low_prob = producer._calculate_probability_actionable(10)
    high_prob = producer._calculate_probability_actionable(90)

    assert 0.0 <= low_prob <= 0.3
    assert 0.8 <= high_prob <= 1.0


def test_choose_scenario_obeys_weighting(monkeypatch):
    producer = SimpleProducer()

    monkeypatch.setattr(random, "random", lambda: 0.01)
    assert producer._choose_scenario() == "normal"

    monkeypatch.setattr(random, "random", lambda: 0.75)
    assert producer._choose_scenario() == "suspicious"

    monkeypatch.setattr(random, "random", lambda: 0.92)
    assert producer._choose_scenario() == "malicious"

    monkeypatch.setattr(random, "random", lambda: 0.999)
    assert producer._choose_scenario() == "edge_case"
