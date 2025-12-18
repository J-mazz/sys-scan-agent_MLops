"""OpenAI Batch Enrichment Agent
Wraps the SyntheticDataPipeline to generate data locally, then enriches it
using the OpenAI Batch API via strict JSONL encapsulation.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for offline tests
    OpenAI = None  # type: ignore
from pydantic import BaseModel, Field

try:  # Optional Jinja2 templating
    from jinja2 import Template
except ImportError:  # pragma: no cover
    Template = None

# Ensure repository root is on sys.path for local imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from synthetic_data.synthetic_data_pipeline import SyntheticDataPipeline
except ImportError:  # pragma: no cover
    from synthetic_data_pipeline import SyntheticDataPipeline

LOG_FORMAT = "% (asctime)s - %(name)s - %(levelname)s - %(message)s".replace(" ", "")
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("batch_agent")


class SecurityEnrichment(BaseModel):
    """The strict schema we want OpenAI to populate."""
    rationale: str = Field(
        ...,
        description="Step-by-step logical deduction (chain-of-thought) explaining why this is a security risk.",
    )
    risk_score: int = Field(..., description="Calculated risk score (0-100) based on severity and context.")
    probability_actionable: float = Field(..., description="0.0 to 1.0 probability that this requires human intervention.")
    mitigation_steps: List[str] = Field(..., description="Concrete steps to remediate the finding")
    attack_vectors: List[str] = Field(..., description="Potential methods attackers could use")
    business_impact: str = Field(..., description="Short executive summary of business risk")
    technical_details: str = Field(..., description="Technical context or investigation hints")


SYSTEM_PROMPT = (
    "You are an elite Security Operations Center (SOC) Analyst. "
    "Analyze the raw security finding and enrich it with actionable intelligence."
)

USER_TEMPLATE = textwrap.dedent(
    """
    ACADEMIC SECURITY FINDING ANALYSIS REQUEST

    # PRIMARY FINDING
    **Title:** {{ title }}
    **Severity:** {{ severity }}
    **Category:** {{ category }}

    # DESCRIPTION
    {{ description }}

    # TECHNICAL METADATA
    {{ metadata_json }}

    # ANALYSIS INSTRUCTIONS
    Provide mitigation steps, attack vectors, compliance relevance, business impact, and technical nuances.
    """
)


class BatchEnrichmentAgent:
    def __init__(
        self,
        model: str = "gpt-5-mini",
        batch_poll_interval: int = 60,
        max_tokens: Optional[int] = 1024,
        skip_low_value: bool = False,
    ):
        if OpenAI:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = None
            logger.warning("OpenAI SDK not installed; batch enrichment client disabled. Install 'openai' to enable API calls.")
        self.model = model
        self.poll_interval = batch_poll_interval
        self.max_tokens = max_tokens  # None -> no cap
        self.skip_low_value = skip_low_value

        self._prompt_template = Template(USER_TEMPLATE) if Template else None

    def _render_user_content(self, finding: Dict[str, Any]) -> str:
        payload = {
            "title": finding.get("title", "Unknown"),
            "description": finding.get("description", "No description available"),
            "severity": finding.get("severity", "unknown"),
            "category": finding.get("category", "unknown"),
            "metadata_json": json.dumps(finding.get("metadata", {}), ensure_ascii=False),
        }

        if self._prompt_template:
            return self._prompt_template.render(**payload)

        return (
            USER_TEMPLATE.replace("{{ title }}", str(payload["title"]))
            .replace("{{ description }}", str(payload["description"]))
            .replace("{{ severity }}", str(payload["severity"]))
            .replace("{{ category }}", str(payload["category"]))
            .replace("{{ metadata_json }}", payload["metadata_json"])
        )

    def _build_request(self, item: Dict[str, Any], kind: str = "finding") -> Dict[str, Any]:
        user_content = self._render_user_content(item)

        response_format: Dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": "security_enrichment",
                "schema": SecurityEnrichment.model_json_schema(),
                "strict": True,
            },
        }

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "response_format": response_format,
            "temperature": 0.2,
        }

        # Only include max_tokens if explicitly provided (no cap otherwise)
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

        prefix = "finding" if kind == "finding" else "correlation"
        custom_id = str(item.get("id") or f"{prefix}-{os.urandom(8).hex()}")

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def create_batch_file(
        self,
        findings: List[Dict[str, Any]],
        correlations: List[Dict[str, Any]],
        filename: str = "batch_input.jsonl",
    ) -> str:
        count = 0
        skipped = 0
        with open(filename, "w", encoding="utf-8") as f:
            for finding in findings:
                if self.skip_low_value and finding.get("severity") == "info" and finding.get("risk_score", 0) < 10:
                    skipped += 1
                    continue

                record = self._build_request(finding, kind="finding")
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

            for corr in correlations:
                record = self._build_request(corr, kind="correlation")
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        logger.info(
            "Generated batch file '%s' with %d requests (Skipped %d low-value items).",
            filename,
            count,
            skipped,
        )
        return filename

    def run_batch_lifecycle(self, input_filename: str) -> Optional[str]:
        if self.client is None:
            raise ImportError("OpenAI SDK not installed; install 'openai' to run batch enrichment.")

        logger.info("Uploading %s...", input_filename)
        with open(input_filename, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")

        logger.info("File uploaded. ID: %s", batch_input_file.id)

        logger.info("Creating batch job...")
        batch_job = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "synthetic_security_enrichment"},
        )
        logger.info("Batch Job Started! ID: %s", batch_job.id)

        while True:
            job = self.client.batches.retrieve(batch_job.id)
            status = job.status
            counts = job.request_counts or {}
            completed = counts.completed if hasattr(counts, "completed") else 0
            total = counts.total if hasattr(counts, "total") else 0
            failed = counts.failed if hasattr(counts, "failed") else 0

            logger.info("Batch Status: %s | Progress: %s/%s (Failed: %s)", status.upper(), completed, total, failed)

            if status == "completed":
                return job.output_file_id
            if status in {"failed", "cancelled", "expired"}:
                logger.error("Batch ended with status %s; errors: %s", status, job.errors)
                return None

            time.sleep(self.poll_interval)

    def merge_results(
        self,
        output_file_id: str,
        findings_map: Dict[str, Dict[str, Any]],
        correlation_map: Dict[str, Dict[str, Any]],
    ):
        if not output_file_id:
            return

        logger.info("Downloading results file...")
        content = self.client.files.content(output_file_id).text

        success_count = 0
        error_count = 0

        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                custom_id = data.get("custom_id") or ""

                response = data.get("response") or {}
                status_code = response.get("status_code")
                if status_code != 200:
                    logger.warning("Request %s failed with status %s", custom_id, status_code)
                    error_count += 1
                    continue

                message = response.get("body", {}).get("choices", [{}])[0].get("message", {})
                content = message.get("content")
                if not content:
                    logger.warning("Request %s missing content", custom_id)
                    error_count += 1
                    continue

                enrichment = json.loads(content)

                target = None
                if custom_id in findings_map:
                    target = findings_map[custom_id]
                elif custom_id in correlation_map:
                    target = correlation_map[custom_id]

                if target is None:
                    logger.warning("Unknown custom_id in results: %s", custom_id)
                    error_count += 1
                    continue

                target.update(enrichment)
                target["_enrichment_timestamp"] = datetime.now().isoformat()
                success_count += 1
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to merge a result line: %s", exc)
                error_count += 1

        logger.info("Merged %d enriched records. (Errors: %d)", success_count, error_count)

    def run(self, output_path: str = "final_enriched_dataset.json", producer_counts: Optional[Dict[str, int]] = None):
        logger.info("Phase 1: Generating synthetic data...")
        pipeline = SyntheticDataPipeline(conservative_parallel=True)

        # Save the unenriched dataset so we can reload the structure for batch prep
        temp_unenriched = Path("unenriched_for_batch.json")
        report = pipeline.execute_pipeline(
            producer_counts=producer_counts,
            output_path=temp_unenriched,
            output_format="optimized_json",
            save_intermediate=False,
        )

        with temp_unenriched.open("r", encoding="utf-8") as f:
            dataset_obj = json.load(f)

        dataset = dataset_obj.get("data", {})

        findings_map: Dict[str, Dict[str, Any]] = {}
        correlation_map: Dict[str, Dict[str, Any]] = {}
        all_findings: List[Dict[str, Any]] = []

        for severity_dict in dataset.get("findings", {}).values():
            for finding_list in severity_dict.values():
                for f in finding_list:
                    all_findings.append(f)
                    fid = str(f.get("id"))
                    findings_map[fid] = f

        all_correlations: List[Dict[str, Any]] = dataset.get("correlations", []) if isinstance(dataset, dict) else []
        for c in all_correlations:
            cid = str(c.get("id") or f"correlation-{os.urandom(8).hex()}")
            c["id"] = cid  # ensure id exists for mapping
            correlation_map[cid] = c

        logger.info("Prepared %d findings and %d correlations for batch enrichment.", len(all_findings), len(all_correlations))

        logger.info("Phase 2: Preparing and submitting Batch Job...")
        jsonl_file = self.create_batch_file(all_findings, all_correlations)

        if Path(jsonl_file).stat().st_size > 0:
            output_file_id = self.run_batch_lifecycle(jsonl_file)

            if output_file_id:
                logger.info("Phase 3: Merging results...")
                self.merge_results(output_file_id, findings_map, correlation_map)
        else:
            logger.warning("No valid findings to enrich (all skipped or empty).")

        report["transformed_dataset"].setdefault("metadata", {})
        report["transformed_dataset"]["metadata"].update(
            {
                "enrichment_mode": "openai_batch",
                "enrichment_model": self.model,
                "openai_enriched": True,
                "enriched_findings": len(findings_map),
                "enriched_correlations": len(correlation_map),
            }
        )

        logger.info("Phase 4: Saving final dataset to %s", output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.get("transformed_dataset", dataset_obj), f, indent=2)

        for path in (Path(jsonl_file), temp_unenriched):
            try:
                path.unlink()
            except OSError:
                pass

        logger.info("✅ Pipeline Complete.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI Batch enrichment pipeline")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI Model ID")
    parser.add_argument("--output", default="final_enriched_dataset.json", help="Final JSON output path")
    parser.add_argument("--producer-counts", type=str, default=None, help="JSON string for producer counts")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval (seconds)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Optional cap on completion tokens")
    parser.add_argument("--skip-low-value", action="store_true", help="Skip info findings with risk<10 to save batch slots")
    args = parser.parse_args(argv)

    counts = json.loads(args.producer_counts) if args.producer_counts else None

    agent = BatchEnrichmentAgent(
        model=args.model,
        batch_poll_interval=args.poll,
        max_tokens=args.max_tokens,
        skip_low_value=args.skip_low_value,
    )
    agent.run(output_path=args.output, producer_counts=counts)


if __name__ == "__main__":
    main()
