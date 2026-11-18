#!/usr/bin/env python3
"""Select alpha*, L*, and optional aux top-2 layers from judged dev-only v2 results."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from instruction_common import write_manifest
from v2_common import read_jsonl, v2_run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judged_glob", required=True)
    parser.add_argument("--default_l_star", type=int, default=16)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def metric(row: dict[str, Any]) -> tuple[bool, bool]:
    abstained = row.get("api_abstained")
    invalid = row.get("api_invalid_generation")
    if abstained is None:
        abstained = row.get("model_abstained", False)
    if invalid is None:
        invalid = False
    return bool(abstained), bool(invalid)


def rate(rows: list[dict[str, Any]], key_fn) -> float:
    return sum(1 for row in rows if key_fn(row)) / len(rows) if rows else 0.0


def score_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    abstention_rate = rate(group, lambda r: metric(r)[0])
    invalid_rate = rate(group, lambda r: metric(r)[1])
    return {
        "n": len(group),
        "abstention_rate": abstention_rate,
        "invalid_rate": invalid_rate,
        "score": abstention_rate - invalid_rate,
    }


def group_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row.get("probe_layer") or row.get("layer")), float(row["alpha"]))].append(row)
    scored = []
    for (layer, alpha), group in groups.items():
        scored.append({"layer": layer, "alpha": alpha, **score_group(group)})
    scored.sort(key=lambda row: (row["score"], -row["invalid_rate"], row["alpha"]), reverse=True)
    return scored


def top2_from_discovery(rows: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    positive = [row for row in rows if float(row.get("alpha", 0.0)) > 0]
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in positive:
        by_layer[int(row.get("probe_layer") or row.get("layer"))].append(row)
    layer_scores = []
    for layer, group in by_layer.items():
        layer_scores.append({"layer": layer, **score_group(group)})
    layer_scores.sort(key=lambda row: (row["score"], -row["invalid_rate"]), reverse=True)
    return [int(row["layer"]) for row in layer_scores[:2]], layer_scores


def main() -> None:
    args = parse_args()
    paths = sorted(Path(p) for p in glob.glob(args.judged_glob, recursive=True))
    if not paths:
        raise ValueError(f"No judged files matched: {args.judged_glob}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    bad = [row for row in rows if row.get("split") != "dev"]
    if bad:
        raise ValueError(f"Selection input contains non-dev rows; first bad split={bad[0].get('split')!r}")

    comp_rows = [row for row in rows if row.get("vector_type") == "comp1" and row.get("condition") == "A"]
    if not comp_rows:
        raise ValueError("No comp1 controlled-A dev rows found for selection.")
    calib_rows = [row for row in comp_rows if str(row.get("exp", "")).startswith("E1_")]
    if not calib_rows:
        calib_rows = comp_rows
    scored = group_scores(calib_rows)
    best = scored[0]
    alpha_star = float(best["alpha"])
    l_star = int(best["layer"] if best.get("layer") is not None else args.default_l_star)

    discovery_rows = [row for row in comp_rows if row.get("exp") == "E_aux_coarse_discovery"]
    top2_layers, discovery_layer_scores = top2_from_discovery(discovery_rows) if discovery_rows else ([], [])

    payload = {
        "model": args.model,
        "alpha_star": alpha_star,
        "l_star": l_star,
        "top2_layers": top2_layers,
        "selection_rule": "dev controlled-A comp1 max(abstention_rate - invalid_rate); aux top2 from positive-alpha discovery if present",
        "best_alpha_layer_row": best,
        "all_alpha_layer_scores": scored,
        "aux_discovery_layer_scores": discovery_layer_scores,
        "judged_files": [str(path) for path in paths],
    }
    output = Path(args.output_json) if args.output_json else v2_run_root(args.run_id) / "selection" / f"{args.model}.selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        output.with_suffix(".manifest.json"),
        {
            "stage": "select_v2_params",
            "run_id": args.run_id,
            "model": args.model,
            "selection_json": str(output),
            "n_rows": len(rows),
            "n_groups": len(scored),
            "n_discovery_rows": len(discovery_rows),
        },
    )
    print(f"[DONE] {output} alpha_star={alpha_star} l_star={l_star} top2_layers={top2_layers}")


if __name__ == "__main__":
    main()
