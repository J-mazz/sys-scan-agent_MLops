"""Generate GRPO prompt/ground_truth JSONL from validated ChatML curated dataset.

Each output line is a JSON object with two fields:
- prompt: the ChatML string ending exactly at the opening <think> tag (no trailing whitespace)
- ground_truth: JSON object with numeric 'risk_score', uppercased 'severity', and 'category'

Usage:
    python3 tools/generate_grpo_prompt_groundtruth.py --input grpo_curated_cot_validated.jsonl --output grpo_prompt_groundtruth.jsonl
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any

CHATML_USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
CHATML_SYSTEM_RE = re.compile(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", re.S)
CHATML_ASSIST_RE = re.compile(r"<\|im_start\|>assistant\n(.*)$", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
THINK_RE = re.compile(r"<think>", re.I)


def extract_blocks(text: str) -> Dict[str,str]:
    system = CHATML_SYSTEM_RE.search(text)
    user = CHATML_USER_RE.search(text)
    assist = CHATML_ASSIST_RE.search(text)
    return {
        'system': system.group(1).strip() if system else '',
        'user': user.group(1).strip() if user else '',
        'assistant': assist.group(1).strip() if assist else '',
    }


def extract_answer(assistant_text: str) -> Dict[str, Any]:
    m = ANSWER_RE.search(assistant_text)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
        return obj
    except Exception:
        # fallback to empty
        return {}


def build_prompt(system: str, user_block: str, assistant_prefix: str) -> str:
    # Compose the prompt ChatML ending exactly at <think>
    # Structure: <|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n<think>

    parts = []
    parts.append('<|im_start|>system')
    parts.append(system.strip())
    parts.append('<|im_end|>')
    parts.append('<|im_start|>user')
    parts.append(user_block.strip())
    parts.append('<|im_end|>')
    parts.append('<|im_start|>assistant')
    parts.append('<think>')
    # Join with newlines but ensure no trailing whitespace after <think>
    prompt = '\n'.join(parts)
    return prompt


def normalize_ground_truth(answer_obj: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    gt = {}
    # risk_score
    try:
        gt['risk_score'] = int(round(float(answer_obj.get('risk_score'))))
    except Exception:
        gt['risk_score'] = int(round(float(fallback.get('risk_score') or 0)))
    # severity
    sev = (answer_obj.get('severity') or fallback.get('severity') or 'INFO')
    if isinstance(sev, str):
        gt['severity'] = sev.strip().upper()
    else:
        gt['severity'] = str(sev).upper()
    # category
    cat = answer_obj.get('category') or fallback.get('category') or 'unspecified'
    gt['category'] = str(cat)
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with inp.open('r', encoding='utf-8') as inf, out.open('w', encoding='utf-8') as outf:
        for line in inf:
            row = json.loads(line)
            text = row.get('text','')
            blocks = extract_blocks(text)
            if not blocks['user'] or not blocks['assistant']:
                continue
            # find answer object
            answer = extract_answer(blocks['assistant'])
            # fallback values from row top-level if present
            fallback = {
                'risk_score': row.get('risk_score'),
                'severity': row.get('severity'),
                'category': row.get('category')
            }
            gt = normalize_ground_truth(answer, fallback)
            prompt = build_prompt(blocks['system'], blocks['user'], blocks['assistant'])
            # confirm no trailing whitespace after '<think>' token
            if prompt.endswith('\n'):
                prompt = prompt.rstrip()  # remove trailing newline if any
            # ensure <think> is the final substring
            if not prompt.endswith('<think>'):
                # if there is stray content, truncate at first occurrence of '<think>' if exists
                idx = prompt.find('<think>')
                if idx != -1:
                    prompt = prompt[:idx+len('<think>')]
            outf.write(json.dumps({'prompt': prompt, 'ground_truth': gt}, ensure_ascii=False) + '\n')
            count += 1
    print(f'Wrote {count} prompt/ground_truth lines to {out}')

if __name__ == '__main__':
    main()
