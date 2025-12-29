"""Validate ChatML curated datasets and optionally fix placeholder values.

Checks performed:
- Each line is valid JSON with a 'text' field
- ChatML contains system/user/assistant blocks
- Assistant response contains <think> and <answer>
- <answer> contains valid JSON with keys 'risk_score','severity','category'
- No occurrences of placeholder strings: 'unknown', 'none', 'null' (case-insensitive) in user JSON or answer

Fixes applied:
- Replace placeholder strings in user metadata values with 'unspecified'
- Ensure answer JSON keys exist; if missing, fill from source finding when possible or use sane defaults
- Rebuild the 'text' field with cleaned user JSON and answer JSON

Writes a validated file with suffix '_validated.jsonl' and prints a summary.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any

PLACEHOLDERS = {"unknown", "none", "null", "n/a", "na", ""}
PH_REGEX = re.compile(r"^(unknown|none|null|n/?a)$", re.I)


def parse_chatml(text: str) -> Dict[str, str]:
    # naive splits based on markers
    parts = {}
    def extract(role):
        start = f"<|im_start|>{role}"
        end = "<|im_end|>"
        if start not in text:
            return None
        s = text.split(start,1)[1]
        if end in s:
            return s.split(end,1)[0].strip()
        return s.strip()
    parts['system'] = extract('system')
    parts['user'] = extract('user')
    parts['assistant'] = extract('assistant')
    return parts


def find_think_and_answer(assistant_text: str):
    if not assistant_text:
        return None, None
    think_m = re.search(r"<think>(.*?)</think>", assistant_text, re.S | re.I)
    answer_m = re.search(r"<answer>(.*?)</answer>", assistant_text, re.S | re.I)
    think = think_m.group(1).strip() if think_m else None
    answer = answer_m.group(1).strip() if answer_m else None
    return think, answer


def clean_placeholders_in_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k,v in obj.items():
            out[k] = clean_placeholders_in_obj(v)
        return out
    if isinstance(obj, list):
        return [clean_placeholders_in_obj(x) for x in obj]
    if isinstance(obj, str):
        if PH_REGEX.match(obj.strip()):
            return 'unspecified'
        return obj
    if obj is None:
        return 'unspecified'
    return obj


def ensure_answer_keys(answer_obj: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    # risk_score
    try:
        rs = int(round(float(answer_obj.get('risk_score')))) if 'risk_score' in answer_obj and answer_obj.get('risk_score') is not None else None
    except Exception:
        rs = None
    if rs is None:
        rs = int(round(float(fallback.get('risk_score') or 0))) if fallback.get('risk_score') is not None else 0
    answer_obj['risk_score'] = rs

    # severity
    sev = str(answer_obj.get('severity') or fallback.get('severity') or 'INFO').upper()
    if PH_REGEX.match(sev):
        sev = 'INFO'
    answer_obj['severity'] = sev

    # category
    cat = str(answer_obj.get('category') or fallback.get('category') or 'unspecified')
    if PH_REGEX.match(cat):
        cat = 'unspecified'
    answer_obj['category'] = cat

    return answer_obj


def validate_and_fix_file(path: Path) -> Dict[str, Any]:
    out_path = path.with_name(path.stem + '_validated.jsonl')
    total = 0
    errors = []
    fixed_count = 0
    placeholder_count = 0

    with path.open('r', encoding='utf-8') as inf, out_path.open('w', encoding='utf-8') as outf:
        for i,line in enumerate(inf, start=1):
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((i, 'json_parse', str(e)))
                continue
            text = row.get('text')
            if text is None:
                errors.append((i, 'missing_text', 'text field missing'))
                continue

            chat = parse_chatml(text)
            if not chat.get('user') or not chat.get('assistant'):
                errors.append((i, 'chatml_structure', 'missing user or assistant section'))
            user_json = None
            try:
                user_json = json.loads(chat['user'])
            except Exception:
                # try to be forgiving: extract first { ... }
                m = re.search(r"\{.*\}", chat.get('user') or '', re.S)
                if m:
                    try:
                        user_json = json.loads(m.group(0))
                    except Exception as e:
                        errors.append((i, 'user_json_parse', str(e)))
                else:
                    errors.append((i, 'user_json_missing', 'no JSON found in user block'))

            think, answer_text = find_think_and_answer(chat.get('assistant') or '')
            if think is None:
                errors.append((i, 'missing_think', 'assistant missing <think>'))
            if answer_text is None:
                errors.append((i, 'missing_answer', 'assistant missing <answer>'))
                answer_obj = {}
            else:
                try:
                    answer_obj = json.loads(answer_text)
                except Exception as e:
                    errors.append((i, 'answer_json_parse', str(e)))
                    answer_obj = {}

            # Clean placeholders in user_json
            if user_json is not None:
                cleaned_user = clean_placeholders_in_obj(user_json)
                # Count placeholders replaced
                # crudely: compare str representations
                if json.dumps(cleaned_user) != json.dumps(user_json):
                    placeholder_count += 1
                    fixed_count += 1
            else:
                cleaned_user = {}

            # ensure answer keys exist and are valid, fallback to source fields
            fallback = {
                'risk_score': row.get('risk_score') or cleaned_user.get('risk_score') or 0,
                'severity': (row.get('severity') or cleaned_user.get('severity') or cleaned_user.get('severity') or 'INFO'),
                'category': (row.get('category') or cleaned_user.get('category') or 'unspecified')
            }
            answer_obj = ensure_answer_keys(answer_obj, fallback)

            # Rebuild text
            parts = []
            parts.append('<|im_start|>system')
            parts.append(chat.get('system') or '')
            parts.append('<|im_end|>')
            parts.append('<|im_start|>user')
            parts.append(json.dumps(cleaned_user, ensure_ascii=False, indent=2))
            parts.append('<|im_end|>')
            parts.append('<|im_start|>assistant')
            parts.append('<think>')
            parts.append(think or '')
            parts.append('</think>')
            parts.append('<answer>')
            parts.append(json.dumps(answer_obj, ensure_ascii=False))
            parts.append('</answer><|im_end|>')
            new_text = '\n'.join(parts)

            outf.write(json.dumps({'text': new_text}, ensure_ascii=False) + '\n')

    return {'path': str(path), 'out_path': str(out_path), 'total': total, 'errors': errors, 'fixed_count': fixed_count, 'placeholder_count': placeholder_count}


def main():
    import sys
    args = sys.argv[1:]
    if not args:
        print('Usage: validate_and_fix_curated.py file1.jsonl file2.jsonl ...')
        sys.exit(1)
    summary = []
    for f in args:
        res = validate_and_fix_file(Path(f))
        summary.append(res)
        print(f"Validated {f}: total={res['total']}, fixed={res['fixed_count']}, placeholders={res['placeholder_count']}, errors={len(res['errors'])}")
    # Optionally, return nonzero if any errors

if __name__ == '__main__':
    main()
