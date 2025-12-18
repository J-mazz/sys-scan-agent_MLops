"""Tests for the BatchEnrichmentAgent request construction and schema.

These tests are offline; they do not hit the OpenAI API. They verify that
requests built for client.beta.chat.completions.parse conform to the expected
JSON schema and that both findings and correlations are included.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repository root on path so we can import the tool module
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.batch_enrichment_agent import BatchEnrichmentAgent, SecurityEnrichment


def test_security_enrichment_schema_fields():
    schema = SecurityEnrichment.model_json_schema()
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    expected_props = {
        "rationale",
        "risk_score",
        "probability_actionable",
        "mitigation_steps",
        "attack_vectors",
        "business_impact",
        "technical_details",
    }
    assert expected_props.issubset(props.keys())
    assert expected_props.issubset(required)


def test_build_request_shapes_for_parse_endpoint_no_max_tokens():
    agent = BatchEnrichmentAgent(model="gpt-5-mini", max_tokens=None)
    sample = {
        "id": "finding-123",
        "title": "Test Vuln",
        "description": "Sample description",
        "severity": "high",
        "category": "network",
        "metadata": {"port": 443},
    }

    req = agent._build_request(sample, kind="finding")
    body = req["body"]

    assert req["method"] == "POST"
    assert req["url"] == "/v1/chat/completions"
    assert req["custom_id"] == "finding-123"

    # Messages present and well-formed
    assert isinstance(body.get("messages"), list) and len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"

    # Response format targets json_schema with our Pydantic schema and strict flag
    rf = body.get("response_format", {})
    assert rf.get("type") == "json_schema"
    js = rf.get("json_schema", {})
    assert js.get("name") == "security_enrichment"
    assert js.get("schema") == SecurityEnrichment.model_json_schema()
    assert js.get("strict") is True

    assert "max_tokens" not in body


def test_build_request_shapes_with_max_tokens():
    agent = BatchEnrichmentAgent(model="gpt-5-mini", max_tokens=1024)
    sample = {
        "id": "finding-123",
        "title": "Test Vuln",
        "description": "Sample description",
        "severity": "high",
        "category": "network",
        "metadata": {"port": 443},
    }

    req = agent._build_request(sample, kind="finding")
    body = req["body"]
    assert body.get("max_tokens") == 1024


def test_create_batch_file_includes_correlations(tmp_path: Path):
    agent = BatchEnrichmentAgent(model="gpt-5-mini", max_tokens=256, skip_low_value=False)

    findings = [
        {
            "id": "finding-A",
            "title": "A",
            "description": "d1",
            "severity": "medium",
            "category": "process",
            "metadata": {},
        }
    ]
    correlations = [
        {
            "id": "corr-1",
            "title": "C1",
            "description": "related",
            "correlation_type": "process_network",
        }
    ]

    out_file = tmp_path / "batch.jsonl"
    agent.create_batch_file(findings, correlations, filename=str(out_file))

    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    parsed = [json.loads(line) for line in lines]
    custom_ids = {p["custom_id"] for p in parsed}
    assert "finding-A" in custom_ids
    assert "corr-1" in custom_ids

    # Ensure both entries point to chat completions and include response_format
    for p in parsed:
        assert p["url"] == "/v1/chat/completions"
        assert p["method"] == "POST"
        assert p["body"]["response_format"]["type"] == "json_schema"


def test_skip_low_value_filters_info_findings(tmp_path: Path):
    agent = BatchEnrichmentAgent(model="gpt-5-mini", max_tokens=256, skip_low_value=True)

    findings = [
        {"id": "keep-1", "title": "High", "description": "d", "severity": "high", "risk_score": 50},
        {"id": "skip-1", "title": "Info", "description": "d", "severity": "info", "risk_score": 5},
    ]
    correlations = []

    out_file = tmp_path / "batch_skip.jsonl"
    agent.create_batch_file(findings, correlations, filename=str(out_file))

    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    parsed = [json.loads(line) for line in lines]
    ids = {p["custom_id"] for p in parsed}
    assert "keep-1" in ids
    assert "skip-1" not in ids


def test_correlation_missing_id_gets_generated(tmp_path: Path):
    agent = BatchEnrichmentAgent(model="gpt-5-mini", max_tokens=256, skip_low_value=False)

    findings = []
    correlations = [
        {"title": "C1", "description": "related", "correlation_type": "process_network"},
    ]

    out_file = tmp_path / "batch_corr.jsonl"
    agent.create_batch_file(findings, correlations, filename=str(out_file))

    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    parsed = [json.loads(line) for line in lines]
    cid = parsed[0]["custom_id"]
    assert cid.startswith("correlation-")
    assert parsed[0]["body"]["response_format"]["json_schema"]["schema"] == SecurityEnrichment.model_json_schema()


def test_jsonl_line_count_and_no_blanks(tmp_path: Path):
    agent = BatchEnrichmentAgent(model="gpt-5-mini", max_tokens=256, skip_low_value=False)

    findings = [
        {"id": "f1", "title": "t1", "description": "d1", "severity": "medium", "risk_score": 20},
        {"id": "f2", "title": "t2", "description": "d2", "severity": "low", "risk_score": 5},
    ]
    correlations = [
        {"id": "c1", "title": "c1", "description": "cdesc", "correlation_type": "net"},
    ]

    out_file = tmp_path / "batch_count.jsonl"
    agent.create_batch_file(findings, correlations, filename=str(out_file))

    text = out_file.read_text(encoding="utf-8")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    assert len(lines) == 3
