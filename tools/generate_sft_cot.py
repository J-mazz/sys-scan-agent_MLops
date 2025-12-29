"""Generate ChatML-style SFT dataset with <think> (rationale) + <answer> (verdict JSON).

Reads a flattened JSONL of findings (e.g., `dataset_sft.jsonl`) and writes
` sft_curated_cot.jsonl` where each line is a JSON object with a single key `text`.

Usage:
    python3 tools/generate_sft_cot.py --input dataset_sft.jsonl --output sft_curated_cot.jsonl
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict
import sys, os
# Ensure project root is importable when running from tools/
sys.path.append(os.getcwd())

from synthetic_data.justification import build_rationale

SYSTEM_PROMPT = (
    "You are a Tier 3 SOC Analyst & Linux Systems Specialist.\n"
    "Objective: Analyze Linux system telemetry to identify security incidents with high precision.\n"
    "Cognitive Framework:\n"
    "1. Observation: Extract facts.\n"
    "2. Context: Compare against Linux standards.\n"
    "3. Hypothesis: Map to MITRE ATT&CK.\n"
    "4. Evidence: (+) Anomalies vs (-) Legitimate behavior.\n"
    "5. Verdict: Assign Risk Score (0-100) and Severity.\n"
    "Response Format: <think>...</think><answer>...</answer> (Strict JSON inside answer)."
)

GENERIC_RATIONALE_FALLBACK = "No detailed rationale available; generated summary follows."


def load_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def is_generic_rationale(r: str) -> bool:
    if not r:
        return True
    lowered = r.strip().lower()
    if len(lowered) < 20:
        return True
    generic_phrases = [
        'analysis complete', 'no additional metadata provided', 'n/a', 'none', 'unknown'
    ]
    for gp in generic_phrases:
        if gp in lowered:
            return True
    return False


def make_conversation_text(finding: Dict) -> str:
    # Ensure rationale exists and is substantive
    rationale = finding.get('rationale') or ''
    if is_generic_rationale(rationale):
        rationale = build_rationale(finding) or GENERIC_RATIONALE_FALLBACK

    # Strip existing tags if present
    if rationale.strip().startswith('<think>') and rationale.strip().endswith('</think>'):
        rationale = rationale.strip()[7:-8].strip()

    user_obj = {
        'title': finding.get('title', 'Unknown Finding'),
        'description': finding.get('description', ''),
        'metadata': finding.get('metadata', {}) or {}
    }

    # Build answer JSON
    risk_score = finding.get('risk_score')
    try:
        risk_score = int(round(float(risk_score))) if risk_score is not None else 0
    except Exception:
        risk_score = 0

    severity = str(finding.get('severity') or '').upper() or 'UNKNOWN'
    category = finding.get('category') or 'unknown'

    answer_obj = {
        'risk_score': risk_score,
        'severity': severity,
        'category': category
    }

    # Assemble ChatML text
    text_parts = []
    text_parts.append('<|im_start|>system')
    text_parts.append(SYSTEM_PROMPT)
    text_parts.append('<|im_end|>')

    text_parts.append('<|im_start|>user')
    # Use pretty-printed JSON for readability inside the user role
    text_parts.append(json.dumps(user_obj, ensure_ascii=False, indent=2))
    text_parts.append('<|im_end|>')

    text_parts.append('<|im_start|>assistant')
    text_parts.append('<think>')
    text_parts.append(rationale)
    text_parts.append('</think>')
    text_parts.append('<answer>')
    text_parts.append(json.dumps(answer_obj, ensure_ascii=False))
    text_parts.append('</answer><|im_end|>')

    return '\n'.join(text_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open('w', encoding='utf-8') as outf:
        for f in load_jsonl(inp):
            text = make_conversation_text(f)
            outf.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            count += 1

    print(f'Wrote {count} examples to {out}')


if __name__ == '__main__':
    main()
