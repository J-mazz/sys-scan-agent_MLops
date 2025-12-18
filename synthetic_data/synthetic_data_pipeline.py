"""
End-to-end data pipeline for synthetic security data generation and processing.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import time
import os
import sys
import logging
import warnings
import random
import asyncio

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

from .producer_registry import registry
from .base_producer import AggregatingProducer
from .correlation_registry import correlation_registry
from .advanced_verification_agent import AdvancedVerificationAgent
from .data_transformation_pipeline import DataTransformationPipeline
from .deduplication_agent import DeduplicationAgent
from .justification import build_rationale, ensure_kb_refs


def _env_flag(name: str) -> bool:
    """Return True if the environment variable is set to a truthy value."""

    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _ensure_logging_config():
    """Ensure a consistent logging configuration for the pipeline."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)


def _load_env_files():
    """Load environment variables from available .env files."""

    logger = logging.getLogger(__name__)

    # Candidate locations in priority order
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent
    candidates = []

    custom_env = os.getenv("SYNTHETIC_DATA_ENV")
    if custom_env:
        candidates.append(Path(custom_env))

    candidates.extend([
        Path.cwd() / ".env",
        module_dir / ".env",
        repo_root / ".env",
    ])

    loaded_total = 0
    seen_paths = set()

    for path in candidates:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            continue

        if resolved in seen_paths:
            continue

        seen_paths.add(resolved)
        loaded = 0

        for line in resolved.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            delimiter = "=" if "=" in stripped else ":" if ":" in stripped else None
            if delimiter is None:
                continue

            key, value = stripped.split(delimiter, 1)
            key = key.strip()
            value = value.strip()

            if key and key not in os.environ:
                os.environ[key] = value
                loaded += 1

        if loaded:
            logger.info("Loaded %d environment variable(s) from %s", loaded, resolved)
            loaded_total += loaded

    return loaded_total


warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)
warnings.filterwarnings(
    "ignore",
    message="The global interpreter lock (GIL) has been enabled to load module 'orjson.orjson'",
    category=RuntimeWarning,
)

_ensure_logging_config()
_load_env_files()

class SyntheticDataPipeline:
    """Complete pipeline for generating, correlating, verifying, and transforming synthetic security data."""

    def __init__(
        self,
        conservative_parallel: bool = False,
        gpu_optimized: Optional[bool] = None,
        fast_mode: bool = False,
        max_workers: Optional[int] = None,
        sampling_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the synthetic data pipeline.

        Args:
            conservative_parallel: Whether to use conservative parallel processing
            gpu_optimized: Whether to use GPU-optimized parallel processing (auto-detect if None)
            fast_mode: Whether to use fast mode (skip heavy enrichment for massive datasets)
            max_workers: Maximum number of parallel workers
        """
        # Set up logging
        self.logger = logging.getLogger("synthetic_data.pipeline")
        self.logger.setLevel(logging.INFO)

        self.conservative_parallel = conservative_parallel
        self.gpu_optimized = gpu_optimized
        self.fast_mode = fast_mode
        # Default high to avoid unintended downsampling; still overridable via env
        self.sampling_target = int(os.getenv("SYNTHETIC_SAMPLING_TARGET", "2500"))
        self.disable_dedup = os.getenv("SYNTHETIC_DISABLE_DEDUP", "false").lower() in {"1", "true", "yes", "on"}
        self.disable_sampling = os.getenv("SYNTHETIC_DISABLE_SAMPLING", "false").lower() in {"1", "true", "yes", "on"}

        # Severity mix for sampling — defaults preserve current behavior but allow overrides
        default_sampling = {"medium_ratio": 0.5, "low_ratio": 0.25}
        env_sampling = os.getenv("SYNTHETIC_SAMPLING_RATIOS")
        parsed_env_sampling: Dict[str, float] = {}
        if env_sampling:
            try:
                parsed_env_sampling = json.loads(env_sampling)
            except json.JSONDecodeError:
                self.logger.warning("Invalid JSON for SYNTHETIC_SAMPLING_RATIOS; using defaults")

        self.sampling_ratios = {**default_sampling, **parsed_env_sampling, **(sampling_config or {})}

        if max_workers is not None:
            self.max_workers = max(1, max_workers)
        else:
            cpu_count = os.cpu_count() or 8
            if cpu_count <= 2:
                self.max_workers = 1
            elif cpu_count == 3:
                self.max_workers = 1
            else:
                self.max_workers = max(2, min(6, cpu_count - 2))

        self.logger.info(
            "Initializing pipeline: conservative_parallel=%s, max_workers=%s",
            conservative_parallel,
            self.max_workers,
        )

        self.producer_registry = registry
        self.correlation_registry = correlation_registry
        self.dedup_agent = DeduplicationAgent()
        self.verification_agent = AdvancedVerificationAgent()
        self.transformation_pipeline = DataTransformationPipeline(fast_mode=fast_mode)

        # Pipeline execution state
        self.execution_state = {
            "stage": "initialized",
            "start_time": None,
            "end_time": None,
            "findings_generated": 0,
            "correlations_generated": 0,
            "verification_passed": False,
            "transformation_completed": False,
            "stage_timings": {},
            "active_producers": self.producer_registry.list_producers()
        }

    def execute_pipeline(
        self,
        producer_counts: Optional[Dict[str, int]] = None,
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "optimized_json",
        compress: bool = False,
        save_intermediate: bool = False,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Execute the complete synthetic data pipeline with iterative verification for robustness.

        Args:
            producer_counts: Number of findings to generate per producer
            output_path: Path to save the final dataset
            output_format: Format for the output dataset
            compress: Whether to compress the output
            save_intermediate: Whether to save intermediate results
            max_iterations: Maximum iterations for verification loop

        Returns:
            Complete pipeline execution results
        """
        self.execution_state["stage"] = "running"
        self.execution_state["start_time"] = datetime.now().isoformat()

        self.logger.info("🚀 Starting Synthetic Data Pipeline Execution")
        self.logger.info("=" * 60)

        try:
            # Stage 1: Generate findings from all producers (done once)
            self.logger.info("📊 Stage 1: Generating Findings")
            findings = self._time_stage("finding_generation", self._execute_finding_generation, producer_counts)
            self.execution_state["findings_generated"] = sum(len(f) for f in findings.values())

            if save_intermediate:
                self._save_intermediate("raw_findings.json", findings)

            # Stage 2: Generate correlations (done once, based on findings)
            self.logger.info("🔗 Stage 2: Analyzing Correlations")
            correlations = self._time_stage("correlation_analysis", self._execute_correlation_analysis, findings)
            self.execution_state["correlations_generated"] = len(correlations)

            if save_intermediate:
                self._save_intermediate("correlations.json", correlations)

            # Stage 3: Iterative Verification and Refinement
            self.logger.info("✅ Stage 3: Verifying and Refining Data Quality")
            verification_report = None
            for iteration in range(max_iterations):
                self.logger.info(f"  Iteration {iteration + 1}/{max_iterations}: Running verification...")
                verification_report = self._time_stage("verification", self._execute_verification, findings, correlations)
                status = verification_report.get("overall_status", "unknown")
                self.logger.info(f"    Status: {status.upper()}")

                if status == "passed":
                    self.execution_state["verification_passed"] = True
                    break
                else:
                    # Attempt to refine: regenerate problematic producers based on verifier feedback
                    findings = self._refine_findings(findings, verification_report, producer_counts)
                    # Re-run correlations on refined findings
                    correlations = self._time_stage("correlation_analysis", self._execute_correlation_analysis, findings)
                    self.execution_state["correlations_generated"] = len(correlations)

            if not self.execution_state["verification_passed"]:
                self.logger.warning("  ⚠️  Maximum iterations reached. Proceeding with best available data.")

            if save_intermediate:
                self._save_intermediate("verification_report.json", verification_report)

            # Stage 4: Transform and optimize dataset
            self.logger.info("🔄 Stage 4: Transforming Dataset")
            transformed_dataset = self._time_stage(
                "transformation",
                self._execute_transformation,
                findings,
                correlations,
                verification_report,
                output_format,
                compress
            )
            self.execution_state["transformation_completed"] = True

            # Stage 5: Save final dataset (if output path provided)
            if output_path:
                self.logger.info("💾 Stage 5: Saving Dataset")
                saved_path = self._save_final_dataset(transformed_dataset, output_path, compress)
                self.execution_state["output_path"] = saved_path

            # Update execution state
            self.execution_state["stage"] = "completed"
            self.execution_state["end_time"] = datetime.now().isoformat()

            # Generate final report
            final_report = self._generate_pipeline_report(
                findings, correlations, verification_report, transformed_dataset
            )

            self.logger.info("🎉 Pipeline Execution Completed Successfully!")
            self.logger.info(f"📈 Generated {self.execution_state['findings_generated']} findings")
            self.logger.info(f"🔗 Generated {self.execution_state['correlations_generated']} correlations")
            self.logger.info(f"✅ Verification: {verification_report.get('overall_status', 'unknown').upper()}")

            return final_report

        except Exception as e:
            self.execution_state["stage"] = "failed"
            self.execution_state["error"] = str(e)
            self.execution_state["end_time"] = datetime.now().isoformat()

            self.logger.error("❌ Pipeline Execution Failed: %s", e)
            raise

    async def execute_pipeline_async(
        self,
        producer_counts: Optional[Dict[str, int]] = None,
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "optimized_json",
        compress: bool = False,
        save_intermediate: bool = False,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Async execution with asyncio-friendly to_thread bridges per stage."""

        self.execution_state["stage"] = "running"
        self.execution_state["start_time"] = datetime.now().isoformat()

        self.logger.info("🚀 Starting Synthetic Data Pipeline Execution (async)")
        self.logger.info("=" * 60)

        try:
            # Stage 1: Generate findings
            self.logger.info("📊 Stage 1: Generating Findings (async)")
            findings = await asyncio.to_thread(self._execute_finding_generation, producer_counts)
            self.execution_state["findings_generated"] = sum(len(f) for f in findings.values())

            if save_intermediate:
                await asyncio.to_thread(self._save_intermediate, "raw_findings.json", findings)

            # Stage 2: Correlations
            self.logger.info("🔗 Stage 2: Analyzing Correlations (async)")
            correlations = await asyncio.to_thread(self._execute_correlation_analysis, findings)
            self.execution_state["correlations_generated"] = len(correlations)

            if save_intermediate:
                await asyncio.to_thread(self._save_intermediate, "correlations.json", correlations)

            # Stage 3: Verification (iterative)
            self.logger.info("✅ Stage 3: Verifying and Refining Data Quality (async)")
            verification_report = None
            for iteration in range(max_iterations):
                self.logger.info(f"  Iteration {iteration + 1}/{max_iterations}: Running verification...")
                verification_report = await asyncio.to_thread(self._execute_verification, findings, correlations)
                status = verification_report.get("overall_status", "unknown")
                self.logger.info(f"    Status: {status.upper()}")

                if status == "passed":
                    self.execution_state["verification_passed"] = True
                    break
                else:
                    findings = await asyncio.to_thread(self._refine_findings, findings, verification_report, producer_counts)
                    correlations = await asyncio.to_thread(self._execute_correlation_analysis, findings)

            # Stage 4: Transformation
            self.logger.info("📦 Stage 4: Transforming Dataset (async)")
            transformed_dataset = await asyncio.to_thread(
                self._execute_transformation,
                findings,
                correlations,
                verification_report,
                output_format,
                compress,
            )

            final_report = {
                "findings": findings,
                "correlations": correlations,
                "verification_report": verification_report,
                "transformed_dataset": transformed_dataset,
                "execution_state": self.execution_state
            }

            # Optional save
            if output_path:
                await asyncio.to_thread(self._save_final_dataset, transformed_dataset, output_path, compress)

            self.execution_state["stage"] = "completed"
            self.execution_state["end_time"] = datetime.now().isoformat()

            self.logger.info(f"📈 Generated {self.execution_state['findings_generated']} findings (async)")
            self.logger.info(f"🔗 Generated {self.execution_state['correlations_generated']} correlations (async)")
            self.logger.info(f"✅ Verification: {verification_report.get('overall_status', 'unknown').upper() if verification_report else 'UNKNOWN'}")

            return final_report

        except Exception as e:
            self.execution_state["stage"] = "failed"
            self.execution_state["error"] = str(e)
            self.execution_state["end_time"] = datetime.now().isoformat()

            self.logger.error("❌ Pipeline Execution Failed (async): %s", e)
            raise

    def _execute_finding_generation(self, producer_counts: Optional[Dict[str, int]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Execute finding generation from all producers."""
        self.logger.info(f"  Generating findings from {len(self.producer_registry.list_producers())} producers...")

        findings = self.producer_registry.generate_all_findings(
            producer_counts,
            self.conservative_parallel,
            self.gpu_optimized,
            self.max_workers,
            density_mode="high",
        )

        # Inject rationales and KB references to improve cohesion and reduce null fields
        for group in findings.values():
            for finding in group:
                ensure_kb_refs(finding)
                if not finding.get("rationale"):
                    finding["rationale"] = build_rationale(finding)

        flat_findings: List[Dict[str, Any]] = [f for group in findings.values() for f in group]
        if self.disable_dedup:
            deduped = flat_findings
        else:
            deduped = self.dedup_agent.deduplicate(flat_findings)

        # Re-bucket by category after deduplication
        rebucketed: Dict[str, List[Dict[str, Any]]] = {}
        for finding in deduped:
            category = finding.get("category", "unknown")
            rebucketed.setdefault(category, []).append(finding)

        if self.disable_sampling:
            sampled = rebucketed
        else:
            sampled = self._apply_intelligent_sampling(rebucketed, target_count=self.sampling_target)

        total_findings = sum(len(f) for f in sampled.values())
        self.logger.info(f"  ✓ Generated {total_findings} total findings after deduplication and sampling")

        return sampled

    def _apply_intelligent_sampling(
        self,
        findings: Dict[str, List[Dict[str, Any]]],
        target_count: Optional[int] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Apply stratified sampling to reduce volume while maintaining diversity.

        Sampling ratios are configurable via:
        - sampling_config argument on pipeline construction
        - SYNTHETIC_SAMPLING_RATIOS env var (JSON, e.g. {"medium_ratio":0.4,"low_ratio":0.2})
        Defaults: medium_ratio=0.5, low_ratio=0.25
        """
        if not findings:
            return findings

        if target_count is None:
            target_count = self.sampling_target

        total_findings = sum(len(f) for f in findings.values()) or 1

        # If target is large enough, skip sampling entirely
        if target_count >= total_findings:
            return findings

        medium_ratio = float(self.sampling_ratios.get("medium_ratio", 0.5))
        low_ratio = float(self.sampling_ratios.get("low_ratio", 0.25))

        sampled: Dict[str, List[Dict[str, Any]]] = {}

        for category, category_findings in findings.items():
            if not category_findings:
                sampled[category] = []
                continue

            high_value = [f for f in category_findings if f.get("severity") in ["high", "critical"]]
            medium_value = [f for f in category_findings if f.get("severity") == "medium"]
            low_value = [f for f in category_findings if f.get("severity") in ["low", "info"]]

            proportion = len(category_findings) / total_findings
            category_target = max(2, int(target_count * proportion))

            sampled_category: List[Dict[str, Any]] = []
            sampled_category.extend(high_value)

            medium_take = max(1, int(category_target * medium_ratio)) if medium_value else 0
            low_take = max(1, int(category_target * low_ratio)) if low_value else 0

            sampled_category.extend(random.sample(medium_value, min(len(medium_value), medium_take)) if medium_value else [])
            sampled_category.extend(random.sample(low_value, min(len(low_value), low_take)) if low_value else [])

            sampled[category] = sampled_category[:category_target]

        return sampled

    def _execute_correlation_analysis(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Execute correlation analysis across all findings."""
        self.logger.info(f"  Analyzing correlations with {len(self.correlation_registry.list_correlation_producers())} correlation producers...")

        correlations = self.correlation_registry.analyze_all_correlations(findings, self.conservative_parallel, self.gpu_optimized, self.max_workers)

        # Get correlation summary
        summary = self.correlation_registry.get_correlation_summary(correlations)
        self.logger.info(f"  ✓ Generated {len(correlations)} correlations")
        self.logger.debug(f"    Top correlation types: {list(summary.get('correlation_types', {}).keys())[:3]}")

        return correlations

    def _execute_verification(
        self,
        findings: Dict[str, List[Dict[str, Any]]],
        correlations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute comprehensive data verification."""
        self.logger.info("  Running multi-stage verification...")

        verification_report = self.verification_agent.verify_dataset(findings, correlations)

        status = verification_report.get("overall_status", "unknown")
        stages_passed = verification_report.get("summary", {}).get("stages_passed", 0)
        total_stages = len(verification_report.get("stages", {}))

        self.logger.info(f"  ✓ Verification completed: {status.upper()}")
        self.logger.info(f"    Stages passed: {stages_passed}/{total_stages}")

        return verification_report

    def _execute_transformation(
        self,
        findings: Dict[str, List[Dict[str, Any]]],
        correlations: List[Dict[str, Any]],
        verification_report: Dict[str, Any],
        output_format: str,
        compress: bool
    ) -> Dict[str, Any]:
        """Execute data transformation and optimization."""
        self.logger.info(f"  Transforming dataset (format: {output_format}, compress: {compress})...")

        transformed_dataset = self.transformation_pipeline.transform_dataset(
            findings=findings,
            correlations=correlations,
            verification_report=verification_report,
            output_format=output_format,
            compress=compress
        )

        self.logger.info("  ✓ Dataset transformation completed")

        return transformed_dataset

    def _refine_findings(
        self,
        current_findings: Dict[str, List[Dict[str, Any]]],
        verification_report: Dict[str, Any],
        producer_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Refine findings based on verification feedback to improve data quality.

        Strategy:
        - Parse verification stages to identify problematic categories/producers.
        - Regenerate only those categories when possible.
        - Fall back to full regeneration if we cannot pinpoint categories.
        """
        self.logger.info("    🔄 Refining findings based on verification feedback...")

        if verification_report.get("overall_status") == "passed":
            return current_findings

        # Determine which categories need regeneration
        problem_categories = self._extract_problem_categories(verification_report, current_findings)

        if not problem_categories:
            self.logger.info("    No specific categories identified; regenerating all findings...")
            return self._execute_finding_generation(producer_counts)

        self.logger.info("    Targeted regeneration for categories: %s", ", ".join(sorted(problem_categories)))

        # Regenerate targeted categories
        regenerated: Dict[str, List[Dict[str, Any]]] = {}
        for category in problem_categories:
            try:
                producer = self.producer_registry.get_producer(category)
            except ValueError:
                self.logger.warning("    No producer found for category '%s'; skipping", category)
                continue

            count = (producer_counts or {}).get(category)
            if count is None:
                count = 100  # reasonable default if not provided

            try:
                new_findings = producer.generate_findings(count)
                if isinstance(producer, AggregatingProducer):
                    new_findings = producer.aggregate_findings(new_findings)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error("    Error regenerating %s: %s", category, exc)
                continue

            for f in new_findings:
                ensure_kb_refs(f)
                if not f.get("rationale"):
                    f["rationale"] = build_rationale(f)

            regenerated[category] = new_findings

        if not regenerated:
            self.logger.info("    Regeneration produced no updates; falling back to full regeneration.")
            return self._execute_finding_generation(producer_counts)

        # Merge regenerated categories back into the current findings, then re-run dedup/sampling to keep distribution sane
        merged: Dict[str, List[Dict[str, Any]]] = {**current_findings, **regenerated}

        flat = [f for group in merged.values() for f in group]
        if not self.disable_dedup:
            flat = self.dedup_agent.deduplicate(flat)

        rebucketed: Dict[str, List[Dict[str, Any]]] = {}
        for finding in flat:
            rebucketed.setdefault(finding.get("category", "unknown"), []).append(finding)

        if self.disable_sampling:
            return rebucketed

        return self._apply_intelligent_sampling(rebucketed, target_count=self.sampling_target)

    def _extract_problem_categories(self, verification_report: Dict[str, Any], current_findings: Dict[str, List[Dict[str, Any]]]) -> set:
        """Inspect verification report for categories/scanners that failed quality checks."""
        categories = set()

        stages = verification_report.get("stages", {})

        schema_stage = stages.get("schema_validation", {})
        for detail in schema_stage.get("invalid_details", []) or []:
            scanner = detail.get("scanner")
            if scanner and scanner != "correlation":
                categories.add(scanner)

        # If consistency issues are present, we conservatively refresh all categories involved in findings
        consistency_stage = stages.get("consistency_check", {})
        if consistency_stage.get("status") in {"warning", "failed"} and not categories:
            categories.update(current_findings.keys())

        return categories

    def _save_final_dataset(
        self,
        dataset: Dict[str, Any],
        output_path: Union[str, Path],
        compress: bool
    ) -> str:
        """Save the final transformed dataset."""
        saved_path = self.transformation_pipeline.save_dataset(
            dataset, output_path, compress
        )

        self.logger.info(f"  ✓ Dataset saved to: {saved_path}")

        return saved_path

    def _save_intermediate(self, filename: str, data: Any):
        """Save intermediate results for debugging/analysis."""
        intermediate_dir = Path("intermediate_results")
        intermediate_dir.mkdir(exist_ok=True)

        output_path = intermediate_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.debug(f"  💾 Intermediate result saved: {output_path}")

    def _generate_pipeline_report(
        self,
        findings: Dict[str, List[Dict[str, Any]]],
        correlations: List[Dict[str, Any]],
        verification_report: Dict[str, Any],
        transformed_dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comprehensive pipeline execution report."""
        report = {
            "pipeline_execution": self.execution_state.copy(),
            "data_summary": {
                "total_findings": sum(len(f) for f in findings.values()),
                "total_correlations": len(correlations),
                "scanner_types": len(findings),
                "correlation_types": len(set(c.get("correlation_type") for c in correlations))
            },
            "quality_metrics": {
                "verification_status": verification_report.get("overall_status"),
                "stages_passed": verification_report.get("summary", {}).get("stages_passed", 0),
                "quality_score": verification_report.get("stages", {}).get("quality_scoring", {}).get("overall_quality_score", 0.0)
            },
            "dataset_characteristics": {
                "version": transformed_dataset.get("metadata", {}).get("version", "unknown"),
                "format": transformed_dataset.get("metadata", {}).get("format", "unknown"),
                "compressed": transformed_dataset.get("metadata", {}).get("compression", False),
                "langchain_enriched": transformed_dataset.get("metadata", {}).get("langchain_enriched", False)
            },
            "performance_metrics": {
                "execution_time": self._calculate_execution_time(),
                "findings_per_second": self._calculate_findings_per_second(),
                "data_size_mb": self._estimate_dataset_size(transformed_dataset),
                "stage_timings": self.execution_state.get("stage_timings", {})
            },
            "recommendations": verification_report.get("recommendations", [])
        }

        return report

    def _time_stage(self, stage_name: str, func, *args, **kwargs):
        """Measure stage execution duration and return the function result."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        self.execution_state.setdefault("stage_timings", {})[stage_name] = round(duration, 4)
        return result

    def _calculate_execution_time(self) -> float:
        """Calculate total pipeline execution time in seconds."""
        if not self.execution_state.get("start_time") or not self.execution_state.get("end_time"):
            return 0.0

        start = datetime.fromisoformat(self.execution_state["start_time"])
        end = datetime.fromisoformat(self.execution_state["end_time"])

        return (end - start).total_seconds()

    def _calculate_findings_per_second(self) -> float:
        """Calculate findings generation rate."""
        execution_time = self._calculate_execution_time()
        if execution_time == 0:
            return 0.0

        return self.execution_state.get("findings_generated", 0) / execution_time

    def _estimate_dataset_size(self, dataset: Dict[str, Any]) -> float:
        """Estimate dataset size in MB."""
        try:
            json_str = json.dumps(dataset, separators=(',', ':'))
            size_bytes = len(json_str.encode('utf-8'))
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return 0.0

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline execution status."""
        return self.execution_state.copy()

    def get_available_producers(self) -> List[str]:
        """Get list of available producers."""
        return self.producer_registry.list_producers()

    def get_available_correlation_producers(self) -> List[str]:
        """Get list of available correlation producers."""
        return self.correlation_registry.list_correlation_producers()

# Convenience function for quick pipeline execution
def run_synthetic_data_pipeline(
    output_path: str = "synthetic_security_dataset.json",
    producer_counts: Optional[Dict[str, int]] = None,
    compress: bool = False,
    conservative_parallel: bool = True,
    gpu_optimized: Optional[bool] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the complete synthetic data pipeline.

    Args:
        output_path: Path to save the final dataset
        producer_counts: Number of findings per producer (default: 10 each)
        compress: Whether to compress the output
        conservative_parallel: Whether to use conservative parallel processing
        gpu_optimized: Whether to use GPU-optimized parallel processing (auto-detect if None)

    Returns:
        Pipeline execution report
    """
    pipeline = SyntheticDataPipeline(
        conservative_parallel=conservative_parallel,
        gpu_optimized=gpu_optimized,
        max_workers=max_workers,
    )

    return pipeline.execute_pipeline(
        producer_counts=producer_counts,
        output_path=output_path,
        compress=compress,
        save_intermediate=True,
        max_iterations=3
    )

if __name__ == "__main__":  # pragma: no cover - example usage
    # Example usage (production-leaning defaults)
    example_logger = logging.getLogger("synthetic_data.pipeline.example")
    example_logger.info("Running synthetic data pipeline...")

    result = run_synthetic_data_pipeline(
        output_path="synthetic_dataset_example.json",
        producer_counts=None,  # use density_mode="high" defaults per producer_registry
        compress=False,
        conservative_parallel=True,
        max_workers=1,
    )

    example_logger.info("Pipeline completed!")
    example_logger.info("Generated %d findings", result['data_summary']['total_findings'])
    example_logger.info("Generated %d correlations", result['data_summary']['total_correlations'])
    example_logger.info("Quality score: %.2f", result['quality_metrics']['quality_score'])