import json
import math
from pathlib import Path
from typing import Dict, List

import pytest

from synthetic_data.data_transformation_pipeline import DataTransformationPipeline


@pytest.fixture
def sample_findings() -> Dict[str, List[Dict[str, object]]]:
    return {
        "processes": [
            {
                "id": "proc-001",
                "title": "Suspicious process tree",
                "severity": "high",
                "risk_score": 82,
                "description": "Process exhibiting lateral movement behaviour",
                "metadata": {"observed_on": "host-17"},
                "category": "process",
                "tags": ["process", "lateral-movement"],
                "risk_subscores": {
                    "impact": 0.7,
                    "exposure": 0.4,
                    "anomaly": 0.85,
                    "confidence": 0.9,
                },
            },
            {
                "id": "proc-002",
                "title": "Benign background task",
                "severity": "low",
                "risk_score": 20,
                "description": "Routine maintenance daemon",
                "metadata": {"observed_on": "host-17"},
                "category": "process",
                "tags": ["process"],
                "risk_subscores": {
                    "impact": 0.1,
                    "exposure": 0.05,
                    "anomaly": 0.12,
                    "confidence": 0.95,
                },
            },
        ],
        "network": [
            {
                "id": "net-001",
                "title": "Outbound beacon",
                "severity": "medium",
                "risk_score": 55,
                "description": "Regular beacon observed to rare domain",
                "metadata": {"destination": "example.net"},
                "category": "network",
                "tags": ["network", "beacon"],
                "risk_subscores": {
                    "impact": 0.45,
                    "exposure": 0.32,
                    "anomaly": 0.6,
                    "confidence": 0.82,
                },
            }
        ],
    }


@pytest.fixture
def sample_correlations() -> List[Dict[str, object]]:
    return [
        {
            "id": "corr-001",
            "title": "Process / network overlap",
            "severity": "medium",
            "risk_score": 57,
            "description": "Correlation between suspicious process and outbound beacon",
            "metadata": {"technique": "lateral movement"},
            "category": "correlation",
            "tags": ["correlation"],
            "correlation_refs": ["proc-001", "net-001"],
            "correlation_type": "process_network",
            "risk_subscores": {
                "impact": 0.6,
                "exposure": 0.5,
                "anomaly": 0.65,
                "confidence": 0.78,
            },
        }
    ]


@pytest.fixture
def verification_report() -> Dict[str, object]:
    return {
        "overall_status": "pass",
        "summary": {"stages_passed": 4},
        "stages": {
            "quality_scoring": {
                "overall_quality_score": 0.94,
            }
        },
    }


def test_transform_dataset_without_langchain(sample_findings, sample_correlations, verification_report):
    pipeline = DataTransformationPipeline(use_langchain=False, fast_mode=False)

    transformed = pipeline.transform_dataset(
        findings=sample_findings,
        correlations=sample_correlations,
        verification_report=verification_report,
        output_format="optimized_json",
        compress=False,
    )

    # Metadata expectations
    metadata = transformed["metadata"]
    assert metadata["langchain_enriched"] is False
    assert metadata["verification_summary"]["overall_status"] == "pass"
    assert metadata["data_characteristics"]["total_findings"] == 3

    # Optimized data expectations
    optimized = transformed["data"]
    assert optimized["statistics"]["total_correlations"] == 1
    assert optimized["statistics"]["total_findings"] == 3

    # Findings grouped by severity for each scanner
    process_findings = optimized["findings"]["processes"]
    assert set(process_findings.keys()) == {"high", "low"}
    assert process_findings["high"][0]["title"] == "Suspicious process tree"

    # Index checks
    indexes = optimized["indexes"]
    assert indexes["findings_by_id"]["proc-001"]["scanner"] == "processes"
    assert "proc-001" in indexes["findings_by_severity"]["high"]
    assert indexes["correlations_by_finding"]["proc-001"] == ["corr-001"]

    # Ensure normalization added processing metadata
    high_entry = process_findings["high"][0]
    assert "_processed_at" in high_entry
    assert "_data_quality" in high_entry


def test_transform_dataset_with_compression(sample_findings, sample_correlations, verification_report):
    pipeline = DataTransformationPipeline(use_langchain=False)

    compressed = pipeline.transform_dataset(
        findings=sample_findings,
        correlations=sample_correlations,
        verification_report=verification_report,
        compress=True,
    )

    assert compressed["compressed"] is True
    assert compressed["compression_method"] == "gzip"
    assert compressed["original_size"] > compressed["compressed_size"] > 0

    # Round-trip decode to verify gzip payload is valid JSON when decompressed
    binary = bytes.fromhex(compressed["data"])
    import gzip

    decoded = json.loads(gzip.decompress(binary).decode("utf-8"))
    assert decoded["metadata"]["data_characteristics"]["total_findings"] == 3


def test_save_dataset_writes_json(tmp_path, sample_findings, sample_correlations, verification_report):
    pipeline = DataTransformationPipeline(use_langchain=False)
    dataset = pipeline.transform_dataset(
        findings=sample_findings,
        correlations=sample_correlations,
        verification_report=verification_report,
        compress=False,
    )

    output_path = tmp_path / "dataset.json"
    saved_path = pipeline.save_dataset(dataset, output_path)

    assert Path(saved_path).exists()
    loaded = json.loads(Path(saved_path).read_text("utf-8"))
    assert loaded["metadata"]["langchain_enriched"] is False
    assert loaded["data"]["statistics"]["total_findings"] == 3


def test_save_dataset_handles_binary_payload(tmp_path):
    pipeline = DataTransformationPipeline(use_langchain=False)
    binary_dataset = {"data": b"\x00\x01", "compressed": True}

    output = tmp_path / "binary.dat"
    path = pipeline.save_dataset(binary_dataset, output, compress=True)

    assert Path(path).read_bytes() == b"\x00\x01"


def test_normalization_adds_defaults_and_handles_invalid_types():
    pipeline = DataTransformationPipeline(use_langchain=False)

    raw_item = {
        "description": "   Mixed CASE title   ",
        "metadata": None,
        "tags": "single-tag",
        "risk_score": "not-a-number",
        "risk_subscores": {"impact": "nan", "exposure": None},
    }

    ensured = pipeline._ensure_required_fields(raw_item)
    assert ensured["id"].startswith("unknown_")
    assert ensured["severity"] == "info"

    normalized = pipeline._normalize_data_types(ensured)
    assert normalized["risk_score"] == 10.0
    assert normalized["tags"] == ["single-tag"]
    assert math.isnan(normalized["risk_subscores"]["impact"])

    cleaned = pipeline._clean_text_fields({"title": "  example  ", "description": "Line\nbreak"})
    assert cleaned["title"] == "example"
    assert cleaned["description"] == "Line break"
