"""Split a JSONL dataset into train/val/test with optional stratification by severity.

Usage:
  python3 tools/split_jsonl.py --input dataset.jsonl --train_out train.jsonl --val_out val.jsonl --test_out test.jsonl --ratios 0.3 0.6 0.1 --stratify
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict
import random


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def write_jsonl(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def stratified_split(items: List[Dict], ratios: List[float]) -> Dict[str, List[Dict]]:
    # ratios are [train, val, test]
    assert abs(sum(ratios) - 1.0) < 1e-6
    buckets = {}
    for it in items:
        sev = str(it.get('severity', 'info')).lower()
        buckets.setdefault(sev, []).append(it)

    train, val, test = [], [], []
    for sev, group in buckets.items():
        random.shuffle(group)
        n = len(group)
        nt = int(round(ratios[0] * n))
        nv = int(round(ratios[1] * n))
        # ensure sum doesn't exceed n
        if nt + nv > n:
            nv = max(0, n - nt)
        ns = n - nt - nv
        train.extend(group[:nt])
        val.extend(group[nt:nt+nv])
        test.extend(group[nt+nv:])

    # If rounding caused totals to be off, adjust by moving from val to train/test
    total = len(items)
    desired_train = int(round(ratios[0] * total))
    desired_val = int(round(ratios[1] * total))
    desired_test = total - desired_train - desired_val

    def adjust(dest, src, need):
        while len(dest) < need and src:
            dest.append(src.pop())

    # adjust train
    if len(train) < desired_train:
        adjust(train, val, desired_train)
        adjust(train, test, desired_train)
    # adjust val
    if len(val) < desired_val:
        adjust(val, test, desired_val)
    # if too large, trim
    all_parts = train + val + test
    if len(all_parts) != total:
        # flatten and re-split deterministically
        random.shuffle(items)
        nt = desired_train
        nv = desired_val
        train = items[:nt]
        val = items[nt:nt+nv]
        test = items[nt+nv:]

    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return {'train': train, 'val': val, 'test': test}


def random_split(items: List[Dict], ratios: List[float]) -> Dict[str, List[Dict]]:
    total = len(items)
    indices = list(range(total))
    random.shuffle(indices)
    nt = int(round(ratios[0] * total))
    nv = int(round(ratios[1] * total))
    train = [items[i] for i in indices[:nt]]
    val = [items[i] for i in indices[nt:nt+nv]]
    test = [items[i] for i in indices[nt+nv:]]
    return {'train': train, 'val': val, 'test': test}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--train_out', required=True)
    parser.add_argument('--val_out', required=True)
    parser.add_argument('--test_out', required=True)
    parser.add_argument('--ratios', nargs=3, type=float, default=[0.3, 0.6, 0.1])
    parser.add_argument('--stratify', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    items = load_jsonl(Path(args.input))
    if args.stratify:
        parts = stratified_split(items, args.ratios)
    else:
        parts = random_split(items, args.ratios)

    write_jsonl(Path(args.train_out), parts['train'])
    write_jsonl(Path(args.val_out), parts['val'])
    write_jsonl(Path(args.test_out), parts['test'])

    print(f"Wrote: train={len(parts['train'])}, val={len(parts['val'])}, test={len(parts['test'])}")

if __name__ == '__main__':
    main()
