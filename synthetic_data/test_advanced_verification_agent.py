import copy
from typing import Any, Dict, List

import pytest

from advanced_verification_agent import AdvancedVerificationAgent


@pytest.fixture
def baseline_findings() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "process": [
            {
                "id": "proc-1",
                "title": "Suspicious binary execution",
                "severity": "high",
                "risk_score": 78,
                "description": "Process spawned uncommon child",
                "metadata": {"timestamp": "2025-01-01T00:00:00", "host": "alpha"},
                "category": "process",
                "tags": ["process", "suspicious"],
                "risk_subscores": {"impact": 0.8, "exposure": 0.7, "anomaly": 0.75, "confidence": 0.9},
            },
            {
                "id": "proc-2",
                "title": "Routine service",
                "severity": "info",
                "risk_score": 10,
                "description": "Expected maintenance task",
                "metadata": {"timestamp": "2025-01-01T01:00:00", "host": "alpha"},
                "category": "process",
                "tags": ["process"],
                "risk_subscores": {"impact": 0.1, "exposure": 0.05, "anomaly": 0.2, "confidence": 0.95},
            },
        ],
        "network": [
            {
                "id": "net-1",
                "title": "Outbound connection",
                "severity": "medium",
                "risk_score": 55,
                "description": "Beacon to rare domain",
                "metadata": {"timestamp": "2025-01-01T01:05:00", "host": "alpha"},
                "category": "network",
                "tags": ["network"],
                "risk_subscores": {"impact": 0.5, "exposure": 0.4, "anomaly": 0.6, "confidence": 0.82},
            }
        ],
    }


@pytest.fixture
def baseline_correlations() -> List[Dict[str, Any]]:
    return [
        {
            "id": "corr-1",
            "title": "Process and network alignment",
            "severity": "high",
            "risk_score": 88,
            "description": "High-risk process with matching outbound network activity",
            "metadata": {"source": "correlation"},
            "category": "correlation",
            "tags": ["correlation"],
            "risk_subscores": {"impact": 0.8, "exposure": 0.7, "anomaly": 0.9, "confidence": 0.85},
            "correlation_refs": ["proc-1", "net-1"],
            "correlation_type": "process_network",
        }
    ]


def test_verify_dataset_success_path(baseline_findings, baseline_correlations):
    agent = AdvancedVerificationAgent()
    report = agent.verify_dataset(baseline_findings, baseline_correlations)

    assert report["overall_status"] in {"passed", "warning"}
    assert report["summary"]["total_findings"] == 3
    assert "quality_scoring" in report["stages"]

    recommendations = report["recommendations"]
    assert isinstance(recommendations, list)
    assert recommendations  # At least one recommendation string


def test_verify_dataset_handles_stage_errors(monkeypatch, baseline_findings, baseline_correlations):
    agent = AdvancedVerificationAgent()

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent, "_verify_realism_assessment", boom)

    report = agent.verify_dataset(baseline_findings, baseline_correlations)
    realism_stage = report["stages"]["realism_assessment"]

    assert realism_stage["status"] == "error"
    assert realism_stage["passed"] is False


def test_verification_detects_schema_and_consistency_issues(baseline_findings, baseline_correlations):
    agent = AdvancedVerificationAgent()

    findings = copy.deepcopy(baseline_findings)
    correlations = copy.deepcopy(baseline_correlations)

    # Introduce missing fields
    del findings["process"][0]["title"]
    correlations[0].pop("correlation_type")

    # Create duplicate IDs and inconsistent severity
    findings["process"].append({
        "id": "proc-1",
        "title": "Duplicate",
        "severity": "low",
        "risk_score": 99,
        "description": "",
        "metadata": {},
        "category": "process",
        "tags": [],
        "risk_subscores": {},
    })

    correlations[0]["correlation_refs"].append("missing-id")

    report = agent.verify_dataset(findings, correlations)

    schema_stage = report["stages"]["schema_validation"]
    assert schema_stage["status"] == "failed"
    assert schema_stage["invalid_findings"] >= 1

    consistency_stage = report["stages"]["consistency_check"]
    assert consistency_stage["issues_found"] >= 1
    assert "Duplicate" in " ".join(consistency_stage["issues"])


def test_realism_assessment_handles_extreme_distributions():
    agent = AdvancedVerificationAgent()

    findings = {
        "process": [
            {
                "id": "proc-crit",
                "title": "Critical event",
                "severity": "critical",
                "risk_score": 95,
                "description": "",
                "metadata": {},
                "category": "process",
                "tags": [],
                "risk_subscores": {},
            }
            for _ in range(12)
        ]
    }

    assessment = agent._verify_realism_assessment(findings, [])
    assert assessment["status"] in {"warning", "failed"}
    assert assessment["realism_score"] <= 0.7

    empty_assessment = agent._verify_realism_assessment({}, [])
    assert empty_assessment["passed"] is False
    assert empty_assessment["issues"]


def test_correlation_validation_scores_quality():
    agent = AdvancedVerificationAgent()

    findings = {
        "process": [
            {
                "id": "proc-1",
                "title": "",
                "severity": "low",
                "risk_score": 15,
                "description": "",
                "metadata": {},
                "category": "process",
                "tags": [],
                "risk_subscores": {},
            }
        ]
    }

    correlations = [
        {
            "id": "corr-1",
            "title": "",
            "severity": "medium",
            "risk_score": 40,
            "description": "",
            "metadata": {},
            "category": "correlation",
            "tags": [],
            "correlation_refs": ["proc-1", "missing"],
            "correlation_type": "demo",
        },
        {
            "id": "corr-2",
            "title": "",
            "severity": "medium",
            "risk_score": 30,
            "description": "",
            "metadata": {},
            "category": "correlation",
            "tags": [],
            "correlation_refs": [],
            "correlation_type": "demo",
        },
    ]

    validation = agent._verify_correlation_validation(findings, correlations)
    assert validation["issues"]
    assert validation["correlation_quality_score"] <= 1.0

    quality = agent._assess_correlation_quality(correlations[0], findings)
    assert 0.0 < quality <= 1.0

    no_ref_quality = agent._assess_correlation_quality(correlations[1], findings)
    assert no_ref_quality == 0.0


def test_quality_scoring_metrics_cover_multiple_branches(baseline_findings, baseline_correlations):
    agent = AdvancedVerificationAgent()

    results = agent._verify_quality_scoring(baseline_findings, baseline_correlations)
    assert results["quality_metrics"]["diversity_score"] > 0
    assert results["status"] in {"passed", "warning"}

    results_empty = agent._verify_quality_scoring({}, [])
    assert results_empty["quality_metrics"]["diversity_score"] == 0.0
    assert results_empty["status"] == "failed"


def test_generate_recommendations_aggregates_stage_feedback():
    agent = AdvancedVerificationAgent()

    stages = {
        "schema_validation": {"passed": False, "invalid_findings": 2},
        "consistency_check": {"passed": False, "issues": ["duplicate ids", "severity mismatch"]},
        "realism_assessment": {"passed": True, "realism_score": 0.7},
        "correlation_validation": {"passed": False, "correlation_quality_score": 0.55},
        "quality_scoring": {"passed": False, "overall_quality_score": 0.6},
    }

    recommendations = agent._generate_recommendations(stages)
    assert any(rec.startswith("Fix schema validation") for rec in recommendations)
    assert any("duplicate ids" in rec for rec in recommendations)
    assert ".2f" in " ".join(recommendations)
