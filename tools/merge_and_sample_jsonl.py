"""Merge, deduplicate, and stratified-sample multiple JSONL files of findings.

Usage:
    python3 tools/merge_and_sample_jsonl.py --inputs a.jsonl b.jsonl c.jsonl --output dataset.jsonl --target 1200

Deduplication key: (title, description, category, rationale)
Sampling: stratified by `severity` to preserve distribution.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(obj)
    return items


def dedupe_items(items: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for it in items:
        key = (
            it.get("title"),
            it.get("description"),
            it.get("category"),
            it.get("rationale"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def stratified_sample(items: List[Dict], target: int) -> List[Dict]:
    if len(items) <= target:
        return items

    # Group by severity
    buckets: Dict[str, List[Dict]] = {}
    for it in items:
        sev = str(it.get("severity", "info")).lower()
        buckets.setdefault(sev, []).append(it)

    # Calculate proportional allocation
    total = len(items)
    allocation: Dict[str, int] = {}
    for sev, group in buckets.items():
        proportion = len(group) / total
        allocation[sev] = max(1, int(round(proportion * target)))

    # Fix rounding issues to match target exactly
    alloc_sum = sum(allocation.values())
    # Adjust by adding/subtracting from largest groups
    if alloc_sum != target:
        diff = target - alloc_sum
        # Order severities by group size descending
        ordered = sorted(buckets.keys(), key=lambda s: len(buckets[s]), reverse=True)
        idx = 0
        while diff != 0:
            sev = ordered[idx % len(ordered)]
            if diff > 0:
                allocation[sev] += 1
                diff -= 1
            else:
                if allocation[sev] > 1:
                    allocation[sev] -= 1
                    diff += 1
            idx += 1

    sampled = []
    for sev, count in allocation.items():
        group = buckets.get(sev, [])
        if len(group) <= count:
            sampled.extend(group)
        else:
            sampled.extend(random.sample(group, count))

    # If due to rounding we have slightly off count, trim or pad
    if len(sampled) > target:
        sampled = random.sample(sampled, target)
    elif len(sampled) < target:
        # pad from remaining items
        remaining = [it for it in items if it not in sampled]
        need = target - len(sampled)
        sampled.extend(random.sample(remaining, min(len(remaining), need)))
    random.shuffle(sampled)
    return sampled


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs='+', required=True, help="Input JSONL files")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--target", type=int, default=0, help="Target number of final samples (0 = keep all)")
    args = parser.parse_args(argv)

    all_items: List[Dict] = []
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            print(f"Warning: input {p} not found, skipping")
            continue
        items = load_jsonl(path)
        print(f"Loaded {len(items)} from {p}")
        all_items.extend(items)

    print(f"Total loaded before dedupe: {len(all_items)}")
    deduped = dedupe_items(all_items)
    print(f"After dedupe: {len(deduped)}")

    final = deduped
    if args.target and len(deduped) >= args.target:
        final = stratified_sample(deduped, args.target)
        print(f"Stratified sampled to target {args.target}")
    elif args.target and len(deduped) < args.target:
        print(f"Warning: only {len(deduped)} unique samples available (< target {args.target}); outputting all")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for it in final:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final)} samples to {args.output}")

if __name__ == '__main__':
    main()
