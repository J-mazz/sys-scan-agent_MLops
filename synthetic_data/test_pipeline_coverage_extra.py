import logging
import os
from types import SimpleNamespace

import pytest

from synthetic_data.synthetic_data_pipeline import (
    _ensure_logging_config,
    _load_env_files,
    SyntheticDataPipeline,
    run_synthetic_data_pipeline,
)
import synthetic_data.producer_registry as pr


def test_load_env_files_branches(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\nINVALID_LINE\nBAZ:qux\n", encoding="utf-8")

    # Force duplicate candidate path (custom + cwd) and ensure fresh environment
    monkeypatch.setenv("SYNTHETIC_DATA_ENV", str(env_file))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    for handler in original_handlers:
        root.removeHandler(handler)
    try:
        _ensure_logging_config()  # should add a handler when none are present
        loaded = _load_env_files()
    finally:
        for handler in original_handlers:
            root.addHandler(handler)
        monkeypatch.delenv("SYNTHETIC_DATA_ENV", raising=False)

    assert loaded >= 2
    assert os.getenv("FOO") == "bar"
    assert os.getenv("BAZ") == "qux"


def test_pipeline_init_handles_invalid_sampling_and_cpu(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_SAMPLING_RATIOS", "{bad")
    monkeypatch.setattr(os, "cpu_count", lambda: 3)

    pipeline = SyntheticDataPipeline()
    assert pipeline.max_workers == 1  # branch when cpu_count == 3

    monkeypatch.delenv("SYNTHETIC_SAMPLING_RATIOS", raising=False)
    pipeline2 = SyntheticDataPipeline(max_workers=0)
    assert pipeline2.max_workers == 1  # max_workers capped at minimum


def test_pipeline_init_handles_low_cpu(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    pipeline = SyntheticDataPipeline(max_workers=None)
    assert pipeline.max_workers == 1


def test_execute_pipeline_failure_sets_state(monkeypatch):
    pipeline = SyntheticDataPipeline()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "_execute_finding_generation", boom)

    with pytest.raises(RuntimeError):
        pipeline.execute_pipeline()

    assert pipeline.execution_state["stage"] == "failed"
    assert pipeline.execution_state["error"] == "boom"


@pytest.mark.asyncio
async def test_execute_pipeline_async_failure(monkeypatch):
    pipeline = SyntheticDataPipeline()

    def boom(*args, **kwargs):
        raise RuntimeError("async boom")

    monkeypatch.setattr(pipeline, "_execute_finding_generation", boom)

    with pytest.raises(RuntimeError):
        await pipeline.execute_pipeline_async()

    assert pipeline.execution_state["stage"] == "failed"
    assert pipeline.execution_state["error"] == "async boom"


@pytest.mark.asyncio
async def test_execute_pipeline_async_refines_and_saves(monkeypatch, tmp_path):
    pipeline = SyntheticDataPipeline()

    monkeypatch.setattr(
        pipeline,
        "_execute_finding_generation",
        lambda counts=None: {"proc": [{"category": "proc", "severity": "high", "metadata": {}}]},
    )
    monkeypatch.setattr(pipeline, "_execute_correlation_analysis", lambda findings: [])
    monkeypatch.setattr(
        pipeline,
        "_execute_verification",
        lambda findings, corrs: {"overall_status": "failed", "summary": {}, "stages": {}},
    )

    refined = []
    monkeypatch.setattr(
        pipeline,
        "_refine_findings",
        lambda f, report, producer_counts=None: refined.append(True) or f,
    )
    monkeypatch.setattr(pipeline, "_execute_transformation", lambda f, c, v, fmt, compress: {"metadata": {}})
    monkeypatch.setattr(pipeline, "_save_intermediate", lambda name, data: tmp_path / name)

    result = await pipeline.execute_pipeline_async(save_intermediate=True, max_iterations=1)

    assert refined  # refinement branch executed
    assert result["execution_state"]["stage"] == "completed"


def test_execute_finding_generation_disable_flags(monkeypatch):
    pipeline = SyntheticDataPipeline()
    pipeline.producer_registry = SimpleNamespace(
        list_producers=lambda: ["proc"],
        generate_all_findings=lambda counts, cp, gpu, maxw, density_mode="high": {
            "proc": [{"category": "proc", "severity": "low", "metadata": {}}]
        },
    )
    pipeline.dedup_agent = SimpleNamespace(deduplicate=lambda items: items)
    pipeline.disable_dedup = True
    pipeline.disable_sampling = True

    sampled = pipeline._execute_finding_generation()
    assert sampled["proc"]


def test_apply_intelligent_sampling_handles_empty_category():
    pipeline = SyntheticDataPipeline()
    pipeline.sampling_target = 1

    sampled = pipeline._apply_intelligent_sampling({
        "empty": [],
        "proc": [
            {"severity": "medium", "category": "proc"},
            {"severity": "low", "category": "proc"},
        ],
    })
    assert sampled["empty"] == []


def test_refine_findings_handles_missing_producer(monkeypatch):
    pipeline = SyntheticDataPipeline()
    pipeline.producer_registry = SimpleNamespace(
        get_producer=lambda name: (_ for _ in ()).throw(ValueError("missing"))
    )
    pipeline._execute_finding_generation = lambda counts=None: {"regen": []}

    verification_report = {"overall_status": "failed", "stages": {"schema_validation": {"invalid_details": [{"scanner": "missing"}]}}}

    result = pipeline._refine_findings({}, verification_report, producer_counts={})
    assert result == {"regen": []}


def test_refine_findings_noop_when_passed():
    pipeline = SyntheticDataPipeline()
    findings = {"a": [{"category": "a"}]}
    report = {"overall_status": "passed", "stages": {}}
    assert pipeline._refine_findings(findings, report) == findings


def test_refine_findings_regenerates_when_no_categories(monkeypatch):
    pipeline = SyntheticDataPipeline()
    pipeline._execute_finding_generation = lambda counts=None: {"regen": [{"category": "regen"}]}
    report = {"overall_status": "failed", "stages": {}}

    result = pipeline._refine_findings({"a": []}, report, producer_counts={})
    assert "regen" in result


def test_refine_findings_disable_sampling_and_default_count(monkeypatch):
    pipeline = SyntheticDataPipeline()
    pipeline.disable_sampling = True
    pipeline.dedup_agent = SimpleNamespace(deduplicate=lambda items: items)

    generated_counts = {}

    class StubProducer:
        def generate_findings(self, count):
            generated_counts["count"] = count
            return [{"category": "proc", "severity": "info", "metadata": {}}]

    pipeline.producer_registry = SimpleNamespace(get_producer=lambda name: StubProducer())

    verification_report = {"overall_status": "failed", "stages": {"schema_validation": {"invalid_details": [{"scanner": "proc"}]}}}

    result = pipeline._refine_findings({"proc": []}, verification_report, producer_counts={})
    assert generated_counts.get("count") == 100  # default fallback
    assert "proc" in result


def test_metrics_helpers_and_status(monkeypatch):
    pipeline = SyntheticDataPipeline()
    pipeline.execution_state = {"start_time": None, "end_time": None, "findings_generated": 0}

    assert pipeline._calculate_execution_time() == 0.0
    assert pipeline._calculate_findings_per_second() == 0.0

    class Unserializable:
        pass

    assert pipeline._estimate_dataset_size({"bad": Unserializable()}) == 0.0
    assert isinstance(pipeline.get_pipeline_status(), dict)


def test_run_synthetic_data_pipeline_wrapper(monkeypatch):
    calls = {}

    def fake_execute(self, **kwargs):
        calls["called"] = True
        return {"ok": True}

    monkeypatch.setattr(SyntheticDataPipeline, "execute_pipeline", fake_execute)

    result = run_synthetic_data_pipeline(output_path="ignored.json", producer_counts={"proc": 1}, conservative_parallel=False)
    assert result == {"ok": True}
    assert calls["called"]


def test_get_producer_missing_raises():
    registry = pr.ProducerRegistry()
    with pytest.raises(ValueError):
        registry.get_producer("missing")


def test_generate_all_findings_density_low_with_error(monkeypatch):
    registry = pr.ProducerRegistry()
    registry.producers = {
        "ok": SimpleNamespace(generate_findings=lambda count: [{"id": 1}]),
        "bad": SimpleNamespace(generate_findings=lambda count: (_ for _ in ()).throw(RuntimeError("fail"))),
    }

    monkeypatch.setattr(pr, "PARALLEL_AVAILABLE", False)

    results = registry.generate_all_findings(counts=None, density_mode="low")
    assert results["ok"]
    assert results["bad"] == []
