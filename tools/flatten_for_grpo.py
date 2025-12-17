"""Flatten synthetic findings into jsonl compatible with SFT/GRPO pipeline.

Each output line has the strict schema:
{
  "title": str,
  "description": str,
  "metadata": dict,
  "category": str,
  "risk_score": int,
  "severity": str,
  "rationale": str
}

Supports inputs:
- Structured JSON from synthetic_data_pipeline (with data.findings nested by category/severity)
- Generic JSONL where each line is a finding dict
- Generic JSON list of findings
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List
import random


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        # Handles int-like strings and floats gracefully
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _coerce_metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"raw": value}
        except json.JSONDecodeError:
            return {"raw": value}
    return {"raw": str(value)}


def flatten_finding(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a single finding to the strict GRPO schema."""
    title = raw.get("title") or "Unknown Finding"
    description = raw.get("description") or "No description provided."
    category = raw.get("category") or "unknown"
    metadata = _coerce_metadata(raw.get("metadata"))

    risk_score = _coerce_int(raw.get("risk_score"), default=0)
    severity = str(raw.get("severity", "info")).lower()

    rationale = raw.get("rationale")
    if not rationale:
        # Construct a lightweight rationale from available context
        sev_text = severity if severity else "info"
        rationale = f"Classified as {sev_text} based on observed evidence and risk score {risk_score}."

    return {
        "title": title,
        "description": description,
        "metadata": metadata,
        "category": category,
        "risk_score": risk_score,
        "severity": severity,
        "rationale": rationale,
    }


def _iter_findings_from_structured(dataset: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    data = dataset.get("data", {}) if isinstance(dataset, dict) else {}
    findings = data.get("findings") if isinstance(data, dict) else None
    if not findings:
        return iter(())

    for category, sev_map in findings.items():
        if not isinstance(sev_map, dict):
            continue
        for _severity, items in sev_map.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        yield item


def _iter_findings_from_json(obj: Any) -> Iterator[Dict[str, Any]]:
    """Yield findings from various JSON shapes."""
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(obj, dict):
        # Structured pipeline export
        for item in _iter_findings_from_structured(obj):
            yield item
        # If dict itself looks like a finding, emit it
        if all(k in obj for k in ("title", "description", "severity", "risk_score")):
            yield obj


def process_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    flattened: List[Dict[str, Any]] = []

    if input_path.suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                flattened.append(flatten_finding(raw))
    else:
        # JSON input (structured or list)
        with input_path.open("r", encoding="utf-8") as infile:
            obj = json.load(infile)

        for finding in _iter_findings_from_json(obj):
            flattened.append(flatten_finding(finding))

    # Randomize order to avoid grouped patterns
    random.shuffle(flattened)

    with output_path.open("w", encoding="utf-8") as outfile:
        for flat in flattened:
            outfile.write(json.dumps(flat, ensure_ascii=False) + "\n")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten findings to GRPO-ready JSONL")
    parser.add_argument(
        "input",
        type=Path,
        help="Input file (.json or .jsonl) containing findings or structured dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("findings_ready_for_grpo.jsonl"),
        help="Output JSONL path (default: findings_ready_for_grpo.jsonl)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    process_file(args.input, args.output)
    print(f"Wrote flattened dataset to {args.output}")


if __name__ == "__main__":
    main()
