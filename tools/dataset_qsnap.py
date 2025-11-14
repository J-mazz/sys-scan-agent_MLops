#!/usr/bin/env python3
"""Dataset profiling utility inspired by the user's QSnap script."""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Iterator


def parse_stream(path: str) -> Iterator[Dict[str, Any]]:
    """Stream JSONL or JSON array content from disk.

    NOTE: JSON arrays are fully loaded; convert to JSONL for large datasets.
    """

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        head = fh.read(2048)

    first = next((c for c in head if not c.isspace()), "[")

    if first == "[":
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            data = json.load(fh)
        for row in data:
            yield row
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def is_string(x: Any) -> bool:
    return isinstance(x, str)


def is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def is_list(x: Any) -> bool:
    return isinstance(x, list)


def is_obj(x: Any) -> bool:
    return isinstance(x, dict)


def quantiles_from_sample(sample: Iterable[float]):
    sample = list(sample)
    if not sample:
        return "", "", ""
    sample.sort()

    def q(p: float) -> float:
        idx = int(p * (len(sample) - 1))
        return sample[idx]

    return q(0.50), q(0.95), q(0.99)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to JSON or JSONL (data.jsonl)")
    parser.add_argument("--label", default=None, help="Name of target label column (optional)")
    parser.add_argument("--max_topk", type=int, default=10, help="Top-K values to report per field")
    parser.add_argument(
        "--max_cat_track",
        type=int,
        default=20000,
        help="Stop counting uniques beyond this (per field)",
    )
    parser.add_argument(
        "--max_len_sample",
        type=int,
        default=8000,
        help="Sample cap per field for length stats",
    )
    parser.add_argument(
        "--hash_dupes",
        action="store_true",
        help="Enable duplicate detection (uses memory)",
    )
    args = parser.parse_args()

    rows = 0
    dupes = 0
    seen_hashes = set()
    keys_all = set()

    present: Counter[str] = Counter()
    type_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    num_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": float("inf"), "max": float("-inf")}
    )
    text_len_sample: Dict[str, list] = defaultdict(list)
    list_len_sample: Dict[str, list] = defaultdict(list)
    obj_keycount_sample: Dict[str, list] = defaultdict(list)
    cat_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    distinct_counts: Dict[str, int] = defaultdict(int)
    first_seen_row: Dict[str, int] = {}

    for row in parse_stream(args.path):
        rows += 1

        if args.hash_dupes:
            row_hash = hashlib.sha1(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).digest()
            if row_hash in seen_hashes:
                dupes += 1
            else:
                seen_hashes.add(row_hash)

        row_keys = set(row.keys())
        for key in row_keys:
            if key not in first_seen_row:
                first_seen_row[key] = rows
        keys_all |= row_keys

        for key, value in row.items():
            if value is None:
                continue

            present[key] += 1

            if is_number(value):
                type_counts[key]["number"] += 1
                stats = num_stats[key]
                val = float(value)
                stats["n"] += 1
                stats["sum"] += val
                stats["sumsq"] += val * val
                if val < stats["min"]:
                    stats["min"] = val
                if val > stats["max"]:
                    stats["max"] = val

            elif is_string(value):
                type_counts[key]["string"] += 1
                length = len(value)
                if length < 4096 and len(text_len_sample[key]) < args.max_len_sample:
                    text_len_sample[key].append(length)
                if length <= 120 and value:
                    counts = cat_counts[key]
                    if len(counts) < args.max_cat_track or value in counts:
                        counts[value] += 1
                        distinct_counts[key] = len(counts)

            elif is_bool(value):
                type_counts[key]["bool"] += 1
                counts = cat_counts[key]
                token = str(value)
                if len(counts) < args.max_cat_track or token in counts:
                    counts[token] += 1
                    distinct_counts[key] = len(counts)

            elif is_list(value):
                type_counts[key]["list"] += 1
                length = len(value)
                if len(list_len_sample[key]) < args.max_len_sample:
                    list_len_sample[key].append(length)

            elif is_obj(value):
                type_counts[key]["object"] += 1
                keycount = len(value.keys())
                if len(obj_keycount_sample[key]) < args.max_len_sample:
                    obj_keycount_sample[key].append(keycount)

            else:
                type_counts[key][type(value).__name__] += 1

    missing = {key: rows - present[key] for key in keys_all}

    fields = sorted(keys_all)
    with open("metrics.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "field",
                "non_null",
                "missing",
                "missing_rate",
                "types",
                "mixed_type_ratio",
                "num_min",
                "num_max",
                "num_mean",
                "num_std",
                "text_len_p50",
                "text_len_p95",
                "text_len_p99",
                "list_len_p50",
                "list_len_p95",
                "list_len_p99",
                "obj_keys_p50",
                "obj_keys_p95",
                "obj_keys_p99",
                "cardinality",
                "top_values",
                "first_seen_row",
            ]
        )

        for key in fields:
            non_null = present[key]
            miss_rate = (missing[key] / rows) * 100 if rows else 0.0
            types_repr = ", ".join(f"{t}:{c}" for t, c in type_counts[key].most_common())
            mixed_ratio = ""
            total_types = sum(type_counts[key].values())
            if total_types > 0:
                majority = type_counts[key].most_common(1)[0][1]
                mixed_ratio = f"{(1 - majority / total_types):.3f}"

            stats = num_stats[key]
            mean = std = minv = maxv = ""
            if stats["n"] > 0:
                mean = stats["sum"] / stats["n"]
                variance = max(0.0, (stats["sumsq"] / stats["n"]) - mean * mean)
                std = math.sqrt(variance)
                minv, maxv = stats["min"], stats["max"]

            t50, t95, t99 = quantiles_from_sample(text_len_sample[key])
            l50, l95, l99 = quantiles_from_sample(list_len_sample[key])
            o50, o95, o99 = quantiles_from_sample(obj_keycount_sample[key])

            top_vals = ""
            if cat_counts[key]:
                top_vals = "; ".join(
                    f"{value}:{count}" for value, count in cat_counts[key].most_common(args.max_topk)
                )

            card = distinct_counts[key] if key in distinct_counts else ""

            writer.writerow(
                [
                    key,
                    non_null,
                    missing[key],
                    f"{miss_rate:.2f}%",
                    types_repr,
                    mixed_ratio,
                    minv,
                    maxv,
                    f"{mean:.4g}" if mean != "" else "",
                    f"{std:.4g}" if std != "" else "",
                    t50,
                    t95,
                    t99,
                    l50,
                    l95,
                    l99,
                    o50,
                    o95,
                    o99,
                    card,
                    top_vals,
                    first_seen_row.get(key, ""),
                ]
            )

    missing_rates = [(key, (missing[key] / rows) * 100 if rows else 0.0) for key in fields]
    missing_rates.sort(key=lambda x: x[1], reverse=True)
    worst5 = ", ".join(f"{key}:{rate:.1f}%" for key, rate in missing_rates[:5])

    label_blurb = ""
    if args.label and cat_counts[args.label]:
        total = sum(cat_counts[args.label].values())
        top = ", ".join(
            f"{value}:{count / total:.1%}"
            for value, count in cat_counts[args.label].most_common(min(args.max_topk, 5))
        )
        label_blurb = f"\n- y({args.label}): {top}"

    zero_var = [key for key, stats in num_stats.items() if stats["n"] > 0 and stats["min"] == stats["max"]]

    mixed_type_hot = []
    for key in fields:
        total = sum(type_counts[key].values())
        if total > 0 and len(type_counts[key]) > 1:
            majority = type_counts[key].most_common(1)[0][1]
            mixed_type_hot.append((key, 1 - majority / total))
    mixed_type_hot.sort(key=lambda x: x[1], reverse=True)
    mixed5 = ", ".join(f"{key}:{val:.2f}" for key, val in mixed_type_hot[:5]) if mixed_type_hot else "n/a"

    miss_avg = sum(rate for _, rate in missing_rates) / len(missing_rates) if missing_rates else 0.0
    dup_rate = (dupes / rows * 100) if rows and args.hash_dupes else 0.0
    mt_penalty = (
        sum(val for _, val in mixed_type_hot[:10]) / 10.0 * 100 if mixed_type_hot else 0.0
    )

    score = 100
    score -= min(40, miss_avg * 0.8)
    score -= min(30, dup_rate * 1.5)
    score -= min(20, mt_penalty * 0.5)
    if zero_var:
        score -= 5
    score = max(0, int(round(score)))

    with open("report.md", "w", encoding="utf-8") as fh:
        fh.write(f"# QSnap — {os.path.basename(args.path)}\n")
        fh.write(f"- rows:{rows} fields:{len(fields)} dup%:{dup_rate:.2f}\n")
        fh.write(f"- worst-miss: {worst5 or 'n/a'}\n")
        fh.write(f"- mixed-type: {mixed5}\n")
        fh.write(f"- num-fields: {sum(1 for stats in num_stats.values() if stats['n'] > 0)}\n")
        fh.write(f"- cat-ish fields: {sum(1 for key in fields if cat_counts[key])}\n")
        if zero_var:
            fh.write(f"- zero-var: {', '.join(zero_var[:10])}\n")
        fh.write(label_blurb + "\n")
        fh.write(f"- score:**{score}/100**\n")
        fh.write("See metrics.csv for details.\n")


if __name__ == "__main__":
    main()
