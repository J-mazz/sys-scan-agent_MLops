import os
import random
from copy import deepcopy
from types import SimpleNamespace

import pytest

from synthetic_data.base_producer import BaseProducer, AggregatingProducer
from synthetic_data.base_verifier import BaseVerifier
from synthetic_data.synthetic_data_pipeline import SyntheticDataPipeline, _env_flag


@pytest.fixture
def anyio_backend():
    """Force anyio tests to use asyncio backend to avoid optional trio dependency."""
    return "asyncio"


class DummyVerifier(BaseVerifier):
    def verify(self, data):
        return True, []


def test_base_verifier_helpers_cover_all_paths():
    verifier = DummyVerifier("dummy")

    required = verifier._validate_required_fields({"foo": 1}, ["foo", "bar"], "obj")
    assert required == ["[dummy] obj: Missing required field 'bar'"]

    sev_issues = verifier._validate_severity_and_scores(
        {"severity": "bad", "risk_score": 120, "base_severity_score": "oops"},
        "item",
        score_fields=["risk_score", "base_severity_score"],
    )
    assert any("Invalid severity" in i for i in sev_issues)
    assert any("out of range" in i for i in sev_issues)
    assert any("must be numeric" in i for i in sev_issues)

    prob_issues = verifier._validate_probability({"prob": "x"}, "prob", "item")
    assert prob_issues and "must be numeric" in prob_issues[0]
    prob_issues_high = verifier._validate_probability({"prob": 2}, "prob", "item")
    assert any("between 0 and 1" in i for i in prob_issues_high)

    risk_issues = verifier._validate_risk_subscores({"impact": -1, "exposure": 0.5}, "risk")
    assert len(risk_issues) >= 2  # missing keys and out-of-range numeric

    risk_not_dict = verifier._validate_risk_subscores("oops", "risk")
    assert any("must be a dict" in i for i in risk_not_dict)

    meta_issues = verifier._validate_host_metadata({"distro": "", "kb_refs": []}, "meta")
    assert any("metadata missing" in i for i in meta_issues)
    assert any("kb_refs must be a non-empty list" in i for i in meta_issues)

    meta_not_dict = verifier._validate_host_metadata("oops", "meta")
    assert any("metadata must be a dict" in i for i in meta_not_dict)


class DummyProducer(BaseProducer):
    def generate_findings(self, count: int = 3):
        findings = []
        for i in range(count):
            sev = random.choice(["info", "low", "medium", "high", "critical"])
            base = self._generate_base_finding(
                finding_id=f"id-{i}",
                title=f"Finding {i}",
                severity=sev,
                risk_score=50,
                base_severity_score=40,
                description="desc",
                metadata={"pattern": "p"},
            )
            findings.append(base)
        return findings


def test_base_producer_generation_and_tags(monkeypatch):
    prod = DummyProducer("scanner")

    # Force scenario selection path
    monkeypatch.setattr(random, "random", lambda: 0.0)
    assert prod._choose_scenario() == "normal"
    monkeypatch.setattr(random, "random", lambda: 0.99)
    assert prod._choose_scenario() in prod.scenarios

    tags_high = prod._generate_tags("high")
    assert {"high_priority", "needs_attention"}.issubset(set(tags_high))
    tags_medium = prod._generate_tags("medium")
    assert "moderate_risk" in tags_medium
    tags_low = prod._generate_tags("low")
    assert "low_priority" in tags_low

    subscores = prod._generate_risk_subscores("medium")
    assert all(0.0 <= v <= 1.0 for v in subscores.values())

    prob = prod._calculate_probability_actionable(80)
    assert 0.0 <= prob <= 1.0

    findings = prod.generate_findings(count=2)
    assert len(findings) == 2
    assert all(f["category"] == "scanner" for f in findings)


def _make_sample_findings(category: str):
    return [
        {
            "id": f"{category}-h",
            "category": category,
            "severity": "high",
            "metadata": {"pattern": "alpha"},
            "probability_actionable": 0.9,
        },
        {
            "id": f"{category}-m",
            "category": category,
            "severity": "medium",
            "metadata": {"pattern": "alpha"},
            "probability_actionable": 0.4,
        },
        {
            "id": f"{category}-l",
            "category": category,
            "severity": "low",
            "metadata": {"pattern": "beta"},
            "probability_actionable": 0.1,
        },
    ]


class DummyAggregatingProducer(AggregatingProducer):
    def generate_findings(self, count: int = 1):
        return _make_sample_findings("agg")


def test_aggregating_producer_respects_env(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_DISABLE_AGGREGATION", "false")
    prod = DummyAggregatingProducer("agg")
    findings = _make_sample_findings("agg")

    # Enabled aggregation by default
    aggregated = prod.aggregate_findings(deepcopy(findings))
    agg_counts = [f["metadata"].get("aggregated_count", 0) for f in aggregated]
    assert aggregated != findings
    assert all(c >= 1 for c in agg_counts)
    assert sum(agg_counts) == len(findings)

    # Disable via env
    monkeypatch.setenv("SYNTHETIC_DISABLE_AGGREGATION", "true")
    untouched = prod.aggregate_findings(deepcopy(findings))
    assert untouched == findings


def test_pipeline_sampling_and_extraction(monkeypatch):
    # Ensure deterministic sampling
    random.seed(123)
    monkeypatch.delenv("SYNTHETIC_SAMPLING_RATIOS", raising=False)

    pipeline = SyntheticDataPipeline(use_langchain=False, sampling_config={"medium_ratio": 0.6, "low_ratio": 0.2})

    empty_sample = pipeline._apply_intelligent_sampling({}, target_count=5)
    assert empty_sample == {}

    findings = {"proc": _make_sample_findings("proc")}
    total = sum(len(v) for v in findings.values())

    # target higher than total should short-circuit
    same = pipeline._apply_intelligent_sampling(findings, target_count=total + 10)
    assert same is findings

    sampled = pipeline._apply_intelligent_sampling(findings, target_count=2)
    assert len(sampled["proc"]) >= 2
    assert any(f["severity"] == "high" for f in sampled["proc"])

    report = {
        "stages": {
            "schema_validation": {
                "invalid_details": [
                    {"scanner": "proc"},
                    {"scanner": "network"},
                ]
            }
        },
        "overall_status": "failed",
    }
    categories = pipeline._extract_problem_categories(report, findings)
    assert categories == {"proc", "network"}

    # Consistency-only path should fall back to current findings keys
    report = {
        "stages": {"consistency_check": {"status": "warning"}},
        "overall_status": "failed",
    }
    categories = pipeline._extract_problem_categories(report, {"a": [], "b": []})
    assert categories == {"a", "b"}


def test_pipeline_report_and_env_flag(monkeypatch):
    pipeline = SyntheticDataPipeline(use_langchain=False)

    pipeline.execution_state["start_time"] = "2024-01-01T00:00:00"
    pipeline.execution_state["end_time"] = "2024-01-01T00:00:10"
    pipeline.execution_state["findings_generated"] = 2

    findings = {"proc": _make_sample_findings("proc")}
    correlations = [{"correlation_type": "simple"}]
    transformed = {"metadata": {"version": "1.0", "format": "optimized_json", "compression": False, "langchain_enriched": False}}

    report = pipeline._generate_pipeline_report(findings, correlations, {"overall_status": "passed", "summary": {"stages_passed": 1}, "stages": {}}, transformed)

    assert report["performance_metrics"]["execution_time"] == 10.0
    assert report["data_summary"]["total_findings"] == len(findings["proc"])
    assert report["data_summary"]["total_correlations"] == len(correlations)
    large_payload = {"foo": "x" * 10000}
    assert pipeline._estimate_dataset_size(large_payload) > 0

    monkeypatch.setenv("TEST_FLAG", "true")
    assert _env_flag("TEST_FLAG") is True
    monkeypatch.setenv("TEST_FLAG", "false")
    assert _env_flag("TEST_FLAG") is False


def test_execute_pipeline_with_mocked_stages(monkeypatch, tmp_path):
    pipeline = SyntheticDataPipeline(use_langchain=False)

    monkeypatch.setattr(pipeline, "_execute_finding_generation", lambda counts=None: {"proc": _make_sample_findings("proc")})
    monkeypatch.setattr(pipeline, "_execute_correlation_analysis", lambda findings: [{"correlation_type": "simple"}])
    monkeypatch.setattr(
        pipeline,
        "_execute_verification",
        lambda findings, corrs: {"overall_status": "passed", "summary": {"stages_passed": 1}, "stages": {}},
    )
    monkeypatch.setattr(
        pipeline,
        "_execute_transformation",
        lambda findings, corrs, report, fmt, compress: {
            "metadata": {"version": "1.0", "format": fmt, "compression": compress, "langchain_enriched": False}
        },
    )
    monkeypatch.setattr(pipeline, "_save_final_dataset", lambda dataset, path, compress: str(path))

    report = pipeline.execute_pipeline(output_path=tmp_path / "out.json", save_intermediate=True, max_iterations=1)

    assert report["quality_metrics"]["verification_status"] == "passed"
    assert report["data_summary"]["total_findings"] > 0


def test_execute_pipeline_triggers_refinement(monkeypatch):
    pipeline = SyntheticDataPipeline(use_langchain=False)

    monkeypatch.setattr(pipeline, "_execute_finding_generation", lambda counts=None: {"proc": _make_sample_findings("proc")})
    monkeypatch.setattr(pipeline, "_execute_correlation_analysis", lambda findings: [])

    verify_calls = []

    def _verify(findings, corrs):
        verify_calls.append("called")
        return {"overall_status": "failed", "summary": {}, "stages": {}}

    monkeypatch.setattr(pipeline, "_execute_verification", _verify)

    refined = []
    monkeypatch.setattr(pipeline, "_refine_findings", lambda cur, report, producer_counts=None: refined.append(True) or cur)
    monkeypatch.setattr(pipeline, "_execute_transformation", lambda f, c, v, fmt, compress: {"metadata": {}})

    pipeline.execute_pipeline(max_iterations=1)

    assert verify_calls
    assert refined


def test_refine_findings_regenerates(monkeypatch):
    pipeline = SyntheticDataPipeline(use_langchain=False)

    class RegeneratingProducer(AggregatingProducer):
        def __init__(self):
            super().__init__("proc")

        def generate_findings(self, count: int = 1):  # pragma: no cover - exercised via _refine_findings
            return _make_sample_findings("proc")

    producer = RegeneratingProducer()
    pipeline.producer_registry = SimpleNamespace(get_producer=lambda name: producer)
    pipeline.dedup_agent.deduplicate = lambda items: items
    pipeline.disable_sampling = False

    current = {"proc": [{"category": "proc", "severity": "low", "metadata": {}}]}
    verification_report = {
        "overall_status": "failed",
        "stages": {"schema_validation": {"invalid_details": [{"scanner": "proc"}]}},
    }

    refined = pipeline._refine_findings(current, verification_report, producer_counts={"proc": 2})

    assert "proc" in refined
    assert len(refined["proc"]) >= 1


@pytest.mark.anyio("asyncio")
async def test_execute_pipeline_async_with_stubs(monkeypatch, tmp_path):
    pipeline = SyntheticDataPipeline(use_langchain=False)

    monkeypatch.setattr(pipeline, "_execute_finding_generation", lambda counts=None: {"proc": _make_sample_findings("proc")})
    monkeypatch.setattr(pipeline, "_execute_correlation_analysis", lambda findings: [{"correlation_type": "simple"}])
    monkeypatch.setattr(
        pipeline,
        "_execute_verification",
        lambda findings, corrs: {"overall_status": "passed", "summary": {"stages_passed": 1}, "stages": {}},
    )
    monkeypatch.setattr(pipeline, "_execute_transformation", lambda f, c, v, fmt, compress: {"metadata": {"format": fmt}})
    monkeypatch.setattr(pipeline, "_save_final_dataset", lambda dataset, path, compress: str(path))

    report = await pipeline.execute_pipeline_async(output_path=tmp_path / "async.json", max_iterations=1)

    assert report["execution_state"]["stage"] == "completed"
    assert report["execution_state"].get("output_path") is None or isinstance(report["execution_state"].get("output_path"), str)
