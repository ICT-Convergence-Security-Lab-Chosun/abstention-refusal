#!/usr/bin/env python3
"""Prepare locked dev/test/cue splits for the v2 paper experiment."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from instruction_common import load_source_rows, write_manifest
from v2_common import (
    BENCHMARK_DATASETS,
    CONTROL_CONDITIONS,
    CONTROL_SHOULD_ABSTAIN,
    entity_key,
    read_jsonl,
    row_reference_answers,
    stable_order,
    v2_run_root,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--benchmark_jsonl", required=True)
    parser.add_argument("--controlled_dev_jsonl", required=True)
    parser.add_argument("--controlled_test_jsonl", required=True)
    parser.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS), choices=list(BENCHMARK_DATASETS))
    parser.add_argument("--benchmark_dev_n", type=int, default=100)
    parser.add_argument("--benchmark_test_n", type=int, default=100)
    parser.add_argument("--controlled_dev_entities", type=int, default=100)
    parser.add_argument("--controlled_test_entities", type=int, default=100)
    parser.add_argument("--cue_limit_per_dataset", type=int, default=None)
    parser.add_argument("--benchmark_filter", choices=["answerable_only", "all"], default="answerable_only")
    parser.add_argument("--allow_short_control_dev", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_benchmark(rows: list[dict[str, Any]], dataset: str, split: str, n: int, seed: int) -> list[dict[str, Any]]:
    selected = stable_order(rows, seed)[:n]
    return [
        {
            **row,
            "split": split,
            "v2_pool": "benchmark",
            "condition": "",
            "sample_filter_group": "answerable_only" if not row["should_abstain"] else "unanswerable_only",
        }
        for row in selected
    ]


def expand_control_entities(rows: list[dict[str, Any]], split: str, n_entities: int, seed: int, allow_short: bool) -> list[dict[str, Any]]:
    ordered = stable_order(rows, seed)
    picked = ordered[:n_entities]
    if len(picked) < n_entities and not allow_short:
        raise ValueError(f"Controlled {split} requested {n_entities} entities but only {len(picked)} are available.")
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(picked):
        entity = str(row.get("entity", f"entity_{idx}"))
        category = str(row.get("category", ""))
        for condition in CONTROL_CONDITIONS:
            question = str(row.get(condition, "")).strip()
            if not question:
                raise ValueError(f"Controlled row for entity={entity!r} is missing condition={condition}")
            sample_id = f"controlled:{split}:{entity}:{condition}".replace(" ", "_")
            out.append(
                {
                    "sample_id": sample_id,
                    "dataset": "controlled",
                    "condition": condition,
                    "entity": entity,
                    "category": category,
                    "question": question,
                    "prompt": question,
                    "reference_answers": [],
                    "should_abstain": CONTROL_SHOULD_ABSTAIN[condition],
                    "split": split,
                    "v2_pool": "controlled",
                    "sample_filter_group": "answerable_only" if condition == "A" else "unanswerable_only",
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["key", "value"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = v2_run_root(args.run_id)
    split_root = root / "splits"
    benchmark_all_rows = load_source_rows(args.benchmark_jsonl, set(args.datasets))
    benchmark_rows = list(benchmark_all_rows)
    if args.benchmark_filter == "answerable_only":
        benchmark_rows = [row for row in benchmark_rows if not row["should_abstain"]]

    dev_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    cue_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        dataset_rows = [row for row in benchmark_rows if row["dataset"] == dataset]
        ordered = stable_order(dataset_rows, args.seed)
        dev_source = ordered[: args.benchmark_dev_n]
        test_source = ordered[args.benchmark_dev_n : args.benchmark_dev_n + args.benchmark_test_n]
        if len(dev_source) < args.benchmark_dev_n or len(test_source) < args.benchmark_test_n:
            raise ValueError(f"Dataset={dataset} does not have enough benchmark rows for dev/test.")
        dev_rows.extend(select_benchmark(dev_source, dataset, "dev", args.benchmark_dev_n, args.seed))
        test_rows.extend(select_benchmark(test_source, dataset, "test", args.benchmark_test_n, args.seed))
        heldout_ids = {row["sample_id"] for row in dev_source + test_source}
        cue_source = stable_order(
            [row for row in benchmark_all_rows if row["dataset"] == dataset and row["sample_id"] not in heldout_ids],
            args.seed + 17,
        )
        if args.cue_limit_per_dataset is not None:
            cue_source = cue_source[: args.cue_limit_per_dataset]
        cue_rows.extend(
            {
                **row,
                "split": "cue_extract",
                "v2_pool": "cue",
                "condition": "",
                "reference_answers": row_reference_answers(row),
            }
            for row in cue_source
        )

    control_dev = expand_control_entities(
        read_jsonl(Path(args.controlled_dev_jsonl)),
        "dev",
        args.controlled_dev_entities,
        args.seed,
        args.allow_short_control_dev,
    )
    control_test = expand_control_entities(
        read_jsonl(Path(args.controlled_test_jsonl)),
        "test",
        args.controlled_test_entities,
        args.seed,
        False,
    )
    dev_rows.extend(control_dev)
    test_rows.extend(control_test)

    dev_ids = {row["sample_id"] for row in dev_rows}
    test_ids = {row["sample_id"] for row in test_rows}
    if dev_ids.intersection(test_ids):
        raise ValueError("dev/test sample_id overlap detected.")
    dev_entities = {entity_key(row) for row in dev_rows if row.get("v2_pool") == "controlled"}
    test_entities = {entity_key(row) for row in test_rows if row.get("v2_pool") == "controlled"}
    overlap_entities = sorted(x for x in dev_entities.intersection(test_entities) if x)
    if overlap_entities:
        raise ValueError(f"Controlled dev/test entity overlap detected: {overlap_entities[:10]}")

    write_jsonl(split_root / "dev.jsonl", dev_rows)
    write_jsonl(split_root / "test.jsonl", test_rows)
    write_jsonl(split_root / "cue_extract.jsonl", cue_rows)
    write_jsonl(split_root / "all_v2.jsonl", dev_rows + test_rows + cue_rows)

    counts = Counter((row.get("v2_pool"), row.get("dataset"), row.get("split"), row.get("condition", ""), row.get("should_abstain")) for row in dev_rows + test_rows + cue_rows)
    summary_rows = [
        {
            "v2_pool": key[0],
            "dataset": key[1],
            "split": key[2],
            "condition": key[3],
            "should_abstain": key[4],
            "n": value,
        }
        for key, value in sorted(counts.items(), key=lambda item: tuple(str(x) for x in item[0]))
    ]
    write_csv(split_root / "split_summary.csv", summary_rows)
    write_manifest(
        split_root / "manifest.json",
        {
            "stage": "prepare_v2_splits",
            "run_id": args.run_id,
            "benchmark_jsonl": args.benchmark_jsonl,
            "controlled_dev_jsonl": args.controlled_dev_jsonl,
            "controlled_test_jsonl": args.controlled_test_jsonl,
            "datasets": args.datasets,
            "benchmark_filter": args.benchmark_filter,
            "dev_rows": len(dev_rows),
            "test_rows": len(test_rows),
            "cue_rows": len(cue_rows),
            "controlled_dev_entities": len(dev_entities),
            "controlled_test_entities": len(test_entities),
            "controlled_entity_overlap": len(overlap_entities),
            "seed": args.seed,
        },
    )
    print(f"[DONE] {split_root}")


if __name__ == "__main__":
    main()

