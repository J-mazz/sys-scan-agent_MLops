#!/usr/bin/env python3
"""Utilities for exporting the synthetic dataset to Hugging Face."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from huggingface_hub import HfApi, upload_folder
except Exception:  # pragma: no cover - huggingface_hub is optional at runtime
    HfApi = None  # type: ignore
    upload_folder = None  # type: ignore


DEFAULT_SUMMARY_PATH = Path("massive_datasets/production_run/production_run_summary_large.json")
DEFAULT_OUTPUT_DIR = Path("massive_datasets/hf_export")
README_FILENAME = "README.md"
DATASET_INFOS_FILENAME = "dataset_infos.json"
FINDINGS_SUBDIR = "findings"
CORRELATIONS_SUBDIR = "correlations"


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class ExportStats:
    total_findings: int = 0
    total_correlations: int = 0
    severity_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    correlation_type_counts: Counter[str] = field(default_factory=Counter)
    baseline_counts: Counter[str] = field(default_factory=Counter)
    correlation_baseline_counts: Counter[str] = field(default_factory=Counter)
    quality_scores: List[float] = field(default_factory=list)
    verification_statuses: Counter[str] = field(default_factory=Counter)
    risk_stats: RunningStats = field(default_factory=RunningStats)
    correlation_risk_stats: RunningStats = field(default_factory=RunningStats)


def load_summary(summary_path: Path) -> Dict[str, object]:
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file '{summary_path}' was not found. "
            "Run the production pipeline before exporting to Hugging Face."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def ensure_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists. "
                "Use --force to overwrite or specify a different --output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / FINDINGS_SUBDIR).mkdir(parents=True, exist_ok=True)
    (output_dir / CORRELATIONS_SUBDIR).mkdir(parents=True, exist_ok=True)


def iter_batches(summary: Dict[str, object]) -> Iterable[Dict[str, object]]:
    for batch in summary.get("batch_results", []):
        yield batch


def read_batch(payload_path: Path) -> Dict[str, object]:
    blob = json.loads(payload_path.read_text(encoding="utf-8"))
    raw_bytes = bytes.fromhex(blob["data"])  # type: ignore[index]
    decompressed = gzip.decompress(raw_bytes)
    return json.loads(decompressed.decode("utf-8"))


def export_batches(
    summary: Dict[str, object],
    output_dir: Path,
    stats: ExportStats,
) -> None:
    for index, batch in enumerate(iter_batches(summary), start=1):
        dataset_path = Path(batch["dataset_path"])  # type: ignore[index]
        payload = read_batch(dataset_path)
        batch_metadata = payload["metadata"]  # type: ignore[index]
        batch_data = payload["data"]  # type: ignore[index]
        findings = batch_data["findings"]  # type: ignore[index]
        correlations = batch_data["correlations"]  # type: ignore[index]
        statistics = batch_data.get("statistics", {})  # type: ignore[assignment]

        findings_path = output_dir / FINDINGS_SUBDIR / f"batch_{index:03d}.jsonl.gz"
        correlations_path = output_dir / CORRELATIONS_SUBDIR / f"batch_{index:03d}.jsonl.gz"

        with gzip.open(findings_path, "wt", encoding="utf-8") as findings_file:
            for category, severity_map in findings.items():  # type: ignore[assignment]
                for severity, items in severity_map.items():  # type: ignore[assignment]
                    stats.severity_counts[severity] += len(items)
                    stats.category_counts[category] += len(items)
                    for item in items:
                        record = dict(item)
                        record["category"] = category
                        record["severity"] = severity
                        record["batch_index"] = index
                        record["batch_dataset_id"] = batch_metadata.get("dataset_id")
                        record["batch_created_at"] = batch_metadata.get("creation_timestamp")
                        stats.baseline_counts[record.get("baseline_status", "unknown")] += 1
                        stats.risk_stats.add(float(record.get("risk_score", 0.0)))
                        json.dump(record, findings_file, ensure_ascii=False, separators=(",", ":"))
                        findings_file.write("\n")

        with gzip.open(correlations_path, "wt", encoding="utf-8") as correlations_file:
            for item in correlations:  # type: ignore[assignment]
                record = dict(item)
                record["batch_index"] = index
                record["batch_dataset_id"] = batch_metadata.get("dataset_id")
                record["batch_created_at"] = batch_metadata.get("creation_timestamp")
                corr_type = record.get("correlation_type") or record.get("title")
                if corr_type:
                    stats.correlation_type_counts[str(corr_type)] += 1
                stats.correlation_baseline_counts[record.get("baseline_status", "unknown")] += 1
                stats.correlation_risk_stats.add(float(record.get("risk_score", 0.0)))
                json.dump(record, correlations_file, ensure_ascii=False, separators=(",", ":"))
                correlations_file.write("\n")

        stats.total_findings += int(statistics.get("total_findings", 0))
        stats.total_correlations += int(statistics.get("total_correlations", 0))

        quality_score = batch.get("quality_score")
        if quality_score is not None:
            stats.quality_scores.append(float(quality_score))
        verification_status = batch.get("verification_status")
        if verification_status:
            stats.verification_statuses[str(verification_status)] += 1


def build_dataset_infos(
    output_dir: Path,
    stats: ExportStats,
) -> Dict[str, object]:
    def collect_split_info(subdir: str) -> Dict[str, object]:
        split_dir = output_dir / subdir
        shards = []
        total_bytes = 0
        for path in sorted(split_dir.glob("*.jsonl.gz")):
            data = path.read_bytes()
            checksum = sha256(data).hexdigest()
            size = len(data)
            total_bytes += size
            shards.append({
                "filename": path.name,
                "num_bytes": size,
                "sha256": checksum,
            })
        return {"shards": shards, "num_bytes": total_bytes}

    findings_info = collect_split_info(FINDINGS_SUBDIR)
    correlations_info = collect_split_info(CORRELATIONS_SUBDIR)

    def features_for_record(record_type: str) -> Dict[str, object]:
        base = {
            "category": {"dtype": "string", "_type": "Value"},
            "severity": {"dtype": "string", "_type": "Value"},
            "risk_score": {"dtype": "float64", "_type": "Value"},
            "baseline_status": {"dtype": "string", "_type": "Value"},
            "batch_index": {"dtype": "int32", "_type": "Value"},
            "batch_dataset_id": {"dtype": "string", "_type": "Value"},
            "batch_created_at": {"dtype": "string", "_type": "Value"},
            "json": {"dtype": "string", "_type": "Value"},
        }
        if record_type == "correlations":
            base["correlation_type"] = {"dtype": "string", "_type": "Value"}
        return base

    dataset_infos = {
        "findings": {
            "description": "Flattened host findings produced by the sys-scan synthetic pipeline.",
            "license": "apache-2.0",
            "citation": "",
            "features": features_for_record("findings"),
            "splits": {
                "train": {
                    "name": "train",
                    "num_examples": stats.total_findings,
                    **findings_info,
                }
            },
            "download_checksums": {
                f"findings/{s['filename']}": {"num_bytes": s["num_bytes"], "checksum": s["sha256"]}
                for s in findings_info["shards"]
            },
        },
        "correlations": {
            "description": "Cross-signal correlations derived from the sys-scan synthetic pipeline.",
            "license": "apache-2.0",
            "citation": "",
            "features": features_for_record("correlations"),
            "splits": {
                "train": {
                    "name": "train",
                    "num_examples": stats.total_correlations,
                    **correlations_info,
                }
            },
            "download_checksums": {
                f"correlations/{s['filename']}": {"num_bytes": s["num_bytes"], "checksum": s["sha256"]}
                for s in correlations_info["shards"]
            },
        },
    }
    return dataset_infos


def render_distribution_table(counter: Counter[str]) -> str:
    total = sum(counter.values()) or 1
    header = "| Key | Count | Share |\n| --- | ---: | ---: |"
    rows = [header]
    for key, value in counter.most_common():
        share = value / total * 100
        rows.append(f"| {key} | {value:,} | {share:5.2f}% |")
    return "\n".join(rows)


def render_readme(
    repo_id: Optional[str],
    stats: ExportStats,
    summary: Dict[str, object],
    output_dir: Path,
) -> str:
    dataset_id = repo_id or "<your-username>/sys-scan-linux-synthetic"
    severity_table = render_distribution_table(stats.severity_counts)
    category_table = render_distribution_table(stats.category_counts)
    correlation_table = render_distribution_table(stats.correlation_type_counts)

    findings_files = "\n".join(
        f"  - `findings/{path.name}` ({path.stat().st_size / 1024 / 1024:.1f} MB)"
        for path in sorted((output_dir / FINDINGS_SUBDIR).glob("*.jsonl.gz"))
    )
    correlations_files = "\n".join(
        f"  - `correlations/{path.name}` ({path.stat().st_size / 1024 / 1024:.1f} MB)"
        for path in sorted((output_dir / CORRELATIONS_SUBDIR).glob("*.jsonl.gz"))
    )

    risk_stats = stats.risk_stats
    corr_risk_stats = stats.correlation_risk_stats

    quality_mean = (sum(stats.quality_scores) / len(stats.quality_scores)) if stats.quality_scores else 0.0
    verification_table = render_distribution_table(stats.verification_statuses)

    return f"""---
license: apache-2.0
language:
- en
task_categories:
- structured-data
pretty_name: Sys-Scan Linux Synthetic
annotations_creators:
- machine-generated
---

# Sys-Scan Linux Synthetic

Synthetic security telemetry generated by the `sys-scan-embedded-agent` pipeline. Each record captures host-, kernel-, network-, and filesystem-oriented findings along with cross-signal correlations that highlight higher-risk situations.

## Summary

- Repo: `{dataset_id}`
- Total findings: **{stats.total_findings:,}**
- Total correlations: **{stats.total_correlations:,}**
- Mean verification quality score: **{quality_mean:.6f}**

### Severity distribution
{severity_table}

### Category distribution
{category_table}

### Correlation types
{correlation_table}

### Verification summary
{verification_table}

### Risk statistics
- Findings: min={risk_stats.minimum:.2f}, max={risk_stats.maximum:.2f}, mean={risk_stats.mean:.2f}, stdev={risk_stats.stddev:.2f} (n={risk_stats.count})
- Correlations: min={corr_risk_stats.minimum:.2f}, max={corr_risk_stats.maximum:.2f}, mean={corr_risk_stats.mean:.2f}, stdev={corr_risk_stats.stddev:.2f} (n={corr_risk_stats.count})

## Files

Findings shards:
{findings_files or '  - (none)'}

Correlation shards:
{correlations_files or '  - (none)'}

## Usage

```python
from datasets import load_dataset

repo_id = "{dataset_id}"
findings = load_dataset(
    "json",
    data_files={"train": f"https://huggingface.co/datasets/{{repo_id}}/resolve/main/findings/batch_*.jsonl.gz"},
    split="train",
    streaming=True,
)

correlations = load_dataset(
    "json",
    data_files={"train": f"https://huggingface.co/datasets/{{repo_id}}/resolve/main/correlations/batch_*.jsonl.gz"},
    split="train",
    streaming=True,
)
```

## Generation pipeline

- Deterministic seed: `{summary.get('seed', 'unknown')}`
- Workers used (per batch): {summary.get('workers_used', [])}
- Generation window: {summary.get('start_timestamp')} → {summary.get('end_timestamp')}

## Responsible use

The dataset is synthetic and intended for benchmarking and research. It should never be treated as real telemetry nor used to simulate offensive capabilities. Respect the Apache-2.0 license and attribute the dataset when you build upon it.
"""


def write_dataset_card(output_dir: Path, repo_id: Optional[str], stats: ExportStats, summary: Dict[str, object]) -> None:
    readme = render_readme(repo_id, stats, summary, output_dir)
    (output_dir / README_FILENAME).write_text(readme, encoding="utf-8")


def write_dataset_infos(output_dir: Path, stats: ExportStats) -> None:
    infos = build_dataset_infos(output_dir, stats)
    (output_dir / DATASET_INFOS_FILENAME).write_text(
        json.dumps(infos, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def push_to_hf(output_dir: Path, repo_id: str, token: Optional[str], private: bool) -> None:
    if HfApi is None or upload_folder is None:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with 'pip install huggingface_hub'."
        )

    auth_token = token or os.getenv("HF_TOKEN")
    if not auth_token:
        raise RuntimeError("Provide --token or set the HF_TOKEN environment variable.")

    api = HfApi(token=auth_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=auth_token,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and publish the dataset to Hugging Face.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Path to the production summary JSON (default: {DEFAULT_SUMMARY_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where export artifacts will be written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Target Hugging Face dataset repository (e.g. username/dataset-name).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload the prepared artifacts to Hugging Face (requires authentication).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the dataset repo as private.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token. Defaults to the HF_TOKEN environment variable.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    summary = load_summary(args.summary_path)
    ensure_output_dir(args.output_dir, args.force)

    stats = ExportStats()
    export_batches(summary, args.output_dir, stats)
    write_dataset_infos(args.output_dir, stats)
    write_dataset_card(args.output_dir, args.repo_id, stats, summary)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "total_findings": stats.total_findings,
                "total_correlations": stats.total_correlations,
                "num_findings_files": len(list((args.output_dir / FINDINGS_SUBDIR).glob("*.jsonl.gz"))),
                "num_correlations_files": len(list((args.output_dir / CORRELATIONS_SUBDIR).glob("*.jsonl.gz"))),
            },
            indent=2,
        )
    )

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when using --push")
        push_to_hf(args.output_dir, args.repo_id, args.token, args.private)
        print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
