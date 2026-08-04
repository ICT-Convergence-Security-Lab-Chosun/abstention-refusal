#!/usr/bin/env python3
"""Summarize judged v2 generations into paper-friendly CSV tables."""

from __future__ import annotations

import argparse
import glob
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from instruction_common import write_csv, write_manifest
from v2_common import read_jsonl, v2_run_root


GROUP_FIELDS = [
    "model",
    "exp",
    "pool",
    "split",
    "dataset",
    "condition",
    "vector_type",
    "vector_source_dataset",
    "probe_layer",
    "alpha",
    "sample_filter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--judged_glob", required=True)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def optional_bool(row: dict[str, Any], key: str, fallback: str | None = None) -> bool | None:
    value = row.get(key)
    if value is None and fallback:
        value = row.get(fallback)
    if value is None:
        return None
    return bool(value)


def rate(rows: list[dict[str, Any]], key: str, fallback: str | None = None) -> float | None:
    vals = [optional_bool(row, key, fallback) for row in rows]
    vals = [value for value in vals if value is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    rationale_counts = Counter(str(row.get("api_rationale_label", "")) for row in rows)
    out = {
        "n": n,
        "api_abstention_rate": rate(rows, "api_abstained", "model_abstained"),
        "api_invalid_rate": rate(rows, "api_invalid_generation"),
        "api_answer_accuracy": rate(rows, "api_answer_correct", "is_answer_correct"),
        "api_over_refusal_rate": rate(rows, "api_over_refusal"),
        "api_unfaithful_answer_rate": rate(rows, "api_unfaithful_answer"),
        "api_acceptable_decision_rate": rate(rows, "api_acceptable_decision"),
        "api_gave_answer_despite_abstaining_rate": rate(rows, "api_gave_answer_despite_abstaining"),
    }
    for label, count in sorted(rationale_counts.items()):
        if label:
            out[f"rationale_{label}_rate"] = count / n if n else 0.0
    return out


def main() -> None:
    args = parse_args()
    paths = sorted(Path(p) for p in glob.glob(args.judged_glob, recursive=True))
    if not paths:
        raise ValueError(f"No judged files matched: {args.judged_glob}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups[key].append(row)
    summary_rows = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        base = {field: value for field, value in zip(GROUP_FIELDS, key)}
        summary_rows.append({**base, **summarize_group(group)})

    output_dir = Path(args.output_dir) if args.output_dir else v2_run_root(args.run_id) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = sorted({key for row in summary_rows for key in row if key not in GROUP_FIELDS})
    fieldnames = GROUP_FIELDS + metric_fields if summary_rows else GROUP_FIELDS + ["n"]
    summary_csv = output_dir / "summary.csv"
    write_csv(summary_csv, fieldnames, summary_rows)

    headline_rows = [
        row
        for row in summary_rows
        if row.get("split") == "test"
        and row.get("vector_type") == "comp1"
        and str(row.get("alpha")) not in {"0", "0.0", ""}
    ]
    if headline_rows:
        headline_metric_fields = sorted({key for row in headline_rows for key in row if key not in GROUP_FIELDS})
        write_csv(output_dir / "headline_comp1_test.csv", GROUP_FIELDS + headline_metric_fields, headline_rows)
    write_manifest(
        output_dir / "manifest.json",
        {
            "stage": "summarize_v2_results",
            "run_id": args.run_id,
            "judged_files": [str(path) for path in paths],
            "n_rows": len(rows),
            "n_groups": len(summary_rows),
            "summary_csv": str(summary_csv),
        },
    )
    print(f"[DONE] {summary_csv} groups={len(summary_rows)}")


if __name__ == "__main__":
    main()

