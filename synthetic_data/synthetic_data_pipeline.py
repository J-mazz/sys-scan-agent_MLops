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

from producer_registry import registry
from correlation_registry import correlation_registry
from advanced_verification_agent import AdvancedVerificationAgent
from data_transformation_pipeline import DataTransformationPipeline

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


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
    message="The global interpreter lock \(GIL\) has been enabled to load module 'orjson.orjson'",
    category=RuntimeWarning,
)

_ensure_logging_config()
_load_env_files()

class SyntheticDataPipeline:
    """Complete pipeline for generating, correlating, verifying, and transforming synthetic security data."""

    def __init__(self, use_langchain: bool = True, conservative_parallel: bool = False, gpu_optimized: Optional[bool] = None, fast_mode: bool = False, max_workers: Optional[int] = None):
        """
        Initialize the synthetic data pipeline.

        Args:
            use_langchain: Whether to use LangChain for data enrichment
            conservative_parallel: Whether to use conservative parallel processing
            gpu_optimized: Whether to use GPU-optimized parallel processing (auto-detect if None)
            fast_mode: Whether to use fast mode (skip heavy enrichment for massive datasets)
            max_workers: Maximum number of parallel workers
        """
        # Set up logging
        self.logger = logging.getLogger("synthetic_data.pipeline")
        self.logger.setLevel(logging.INFO)

        self.use_langchain = use_langchain
        self.conservative_parallel = conservative_parallel
        self.gpu_optimized = gpu_optimized
        self.fast_mode = fast_mode

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
            "Initializing pipeline: use_langchain=%s, conservative_parallel=%s, max_workers=%s",
            use_langchain,
            conservative_parallel,
            self.max_workers,
        )

        self.producer_registry = registry
        self.correlation_registry = correlation_registry
        self.correlation_registry.enable_langchain(self.use_langchain)
        if self.use_langchain:  # pragma: no cover - optional LangChain runtime
            try:
                langchain_producer = self.correlation_registry.get_correlation_producer("langchain")
            except ValueError:
                self.logger.warning(
                    "LangChain correlations requested but runtime unavailable. Falling back to deterministic correlations."
                )
            else:
                mode = getattr(langchain_producer, "mode", "native")
                if mode == "bridge":
                    self.logger.info(
                        "LangChain correlations will execute via external Python runtime: %s",
                        getattr(langchain_producer, "external_python", "python3.12"),
                    )
                elif mode == "native":  # pragma: no cover - depends on external runtime
                    self.logger.info("LangChain correlations executing natively in current interpreter")
                else:
                    self.logger.warning("LangChain correlations disabled; using fallback logic only")
        self.verification_agent = AdvancedVerificationAgent()
        self.transformation_pipeline = DataTransformationPipeline(use_langchain=use_langchain, fast_mode=fast_mode)

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
            self.correlation_registry.enable_langchain(self.use_langchain)
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

    def _execute_finding_generation(self, producer_counts: Optional[Dict[str, int]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Execute finding generation from all producers."""
        self.logger.info(f"  Generating findings from {len(self.producer_registry.list_producers())} producers...")

        if producer_counts is None:
            # Default: 10 findings per producer
            producer_counts = {name: 10 for name in self.producer_registry.list_producers()}

        findings = self.producer_registry.generate_all_findings(producer_counts, self.conservative_parallel, self.gpu_optimized, self.max_workers)

        total_findings = sum(len(f) for f in findings.values())
        self.logger.info(f"  ✓ Generated {total_findings} total findings")

        return findings

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
        producer_counts: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Refine findings based on verification feedback to improve data quality.

        Args:
            current_findings: Current findings dictionary
            verification_report: Report from verification stage
            producer_counts: Original producer counts

        Returns:
            Refined findings dictionary
        """
        self.logger.info("    🔄 Refining findings based on verification feedback...")

        # For now, simple approach: regenerate all findings if verification failed
        # TODO: Implement targeted refinement based on specific verifier failures
        if verification_report.get("overall_status") != "passed":
            self.logger.info("    Regenerating all findings to improve quality...")
            return self._execute_finding_generation(producer_counts)
        else:
            return current_findings

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
    use_langchain: bool = True,
    compress: bool = False,
    conservative_parallel: bool = True,
    gpu_optimized: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the complete synthetic data pipeline.

    Args:
        output_path: Path to save the final dataset
        producer_counts: Number of findings per producer (default: 10 each)
        use_langchain: Whether to use LangChain for enrichment
        compress: Whether to compress the output
        conservative_parallel: Whether to use conservative parallel processing
        gpu_optimized: Whether to use GPU-optimized parallel processing (auto-detect if None)

    Returns:
        Pipeline execution report
    """
    pipeline = SyntheticDataPipeline(
        use_langchain=use_langchain,
        conservative_parallel=False,
        gpu_optimized=gpu_optimized
    )

    return pipeline.execute_pipeline(
        producer_counts=producer_counts,
        output_path=output_path,
        compress=compress,
        save_intermediate=True,
        max_iterations=3
    )

if __name__ == "__main__":
    # Example usage
    example_logger = logging.getLogger("synthetic_data.pipeline.example")
    example_logger.info("Running synthetic data pipeline...")

    result = run_synthetic_data_pipeline(
        output_path="synthetic_dataset_example.json",
        producer_counts={"processes": 5, "network": 5, "kernel_params": 3},
        use_langchain=False,  # Set to True if LangChain is available
        compress=False
    )

    example_logger.info("Pipeline completed!")
    example_logger.info("Generated %d findings", result['data_summary']['total_findings'])
    example_logger.info("Generated %d correlations", result['data_summary']['total_correlations'])
    example_logger.info("Quality score: %.2f", result['quality_metrics']['quality_score'])