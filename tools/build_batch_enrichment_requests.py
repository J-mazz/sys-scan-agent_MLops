"""Build OpenAI Batch API JSONL requests for enrichment.

Usage:
    python tools/build_batch_enrichment_requests.py \
        --input unenriched_findings.jsonl \
        --output openai_enrichment_batch.jsonl \
        --model gpt-5-mini

Each output line conforms to the Batch API shape:
{
  "custom_id": "finding-<id>",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": { ... }
}
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


def _schema() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "SecurityEnrichment",
            "schema": {
                "type": "object",
                "properties": {
                    "mitigation_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete steps to remediate the finding",
                    },
                    "attack_vectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Potential attacker methods",
                    },
                    "compliance_impact": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Compliance frameworks affected (PCI, HIPAA, CIS, etc.)",
                    },
                    "business_impact": {
                        "type": "string",
                        "description": "Executive summary of business risk",
                    },
                    "technical_details": {
                        "type": "string",
                        "description": "Technical context or investigation hints",
                    },
                },
                "required": [
                    "mitigation_steps",
                    "attack_vectors",
                    "compliance_impact",
                    "business_impact",
                    "technical_details",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def _build_body(finding: Dict[str, Any], model: str, max_tokens: Optional[int]) -> Dict[str, Any]:
    system_prompt = (
        "You are a senior security engineer. Respond ONLY with JSON matching the provided schema. "
        "Be concise and actionable."
    )

    user_prompt = {
        "title": finding.get("title", "Unknown Finding"),
        "description": finding.get("description", ""),
        "severity": finding.get("severity", "info"),
        "category": finding.get("category", "unknown"),
        "metadata": finding.get("metadata", {}),
        "risk_score": finding.get("risk_score", 0),
    }

    body: Dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "response_format": _schema(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
    }

    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    return body


def build_batch_file(input_path: Path, output_path: Path, model: str, max_tokens: Optional[int]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                finding = json.loads(line)
            except json.JSONDecodeError:
                continue

            custom_id = str(finding.get("id")) if finding.get("id") else f"finding-{uuid.uuid4()}"
            body = _build_body(finding, model=model, max_tokens=max_tokens)

            record = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }

            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenAI Batch API JSONL requests for enrichment")
    parser.add_argument("--input", type=Path, required=True, help="Input findings JSONL (flattened)")
    parser.add_argument("--output", type=Path, default=Path("openai_enrichment_batch.jsonl"), help="Output batch JSONL")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model to use for enrichment")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max tokens. Omit to let the model decide (no cap).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_batch_file(args.input, args.output, model=args.model, max_tokens=args.max_tokens)
    print(f"Wrote {count} batch requests to {args.output}")


if __name__ == "__main__":
    main()
