#!/usr/bin/env python3
"""Build auditable v2 generation job manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from instruction_common import write_manifest
from v2_common import BENCHMARK_DATASETS, V2_ALPHA_GRID, V2_DOSE_GRID, resolve_layer_spec, v2_run_root, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["calibration", "aux_discovery", "eval"], required=True)
    parser.add_argument("--selection_json", default=None)
    parser.add_argument("--layers_full", default="auto")
    parser.add_argument("--layers_lean", default="auto_lean")
    parser.add_argument("--layers_coarse", default="auto_coarse")
    parser.add_argument("--calib_layers", default="auto_calib")
    parser.add_argument("--discovery_layers", default="auto_calib")
    parser.add_argument("--alpha_star", type=float, default=5.0)
    parser.add_argument("--l_star", type=int, default=16)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS), choices=list(BENCHMARK_DATASETS))
    parser.add_argument("--alpha_grid", default=None, help="Comma-separated calibration alphas (model-aware raw α). Defaults to V2_ALPHA_GRID.")
    parser.add_argument("--dose_grid", default=None, help="Comma-separated E4 dose alphas. Defaults to V2_DOSE_GRID.")
    return parser.parse_args()


def _grid(raw: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return default
    return tuple(float(x) for x in str(raw).replace(",", " ").split())


def load_selection(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def base_job(args: argparse.Namespace, **kwargs) -> dict[str, Any]:
    row = {
        "model": args.model,
        "split": "test",
        "dataset": "controlled",
        "condition": "",
        "pool": "controlled",
        "sample_filter": "all",
        "source_dataset": "__all__",
        "base_vector_type": "cue_diff",
        "position": "t_post",
        "token_position": "all",
        "scale_mode": "raw",
        "steering_mode": "all_steps",
        "sample_limit": args.sample_limit,
    }
    row.update(kwargs)
    return row


def calibration_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = []
    for layer in resolve_layer_spec(args.calib_layers, args.model):
        for alpha in _grid(args.alpha_grid, V2_ALPHA_GRID):
            jobs.append(
                base_job(
                    args,
                    exp="E1_alpha_layer_calibration",
                    split="dev",
                    dataset="controlled",
                    condition="A",
                    pool="controlled_neutral_A",
                    sample_filter="answerable_only",
                    arm="comp1",
                    layer=layer,
                    alpha=alpha,
                    source_dataset="__all__",
                )
            )
    return jobs


def aux_discovery_jobs(args: argparse.Namespace, selection: dict[str, Any]) -> list[dict[str, Any]]:
    alpha_star = float(selection.get("alpha_star", args.alpha_star))
    jobs = []
    for layer in resolve_layer_spec(args.discovery_layers, args.model):
        for alpha in (alpha_star, -alpha_star):
            jobs.append(
                base_job(
                    args,
                    exp="E_aux_coarse_discovery",
                    split="dev",
                    dataset="controlled",
                    condition="A",
                    pool="controlled_neutral_A",
                    sample_filter="answerable_only",
                    arm="comp1",
                    layer=layer,
                    alpha=alpha,
                    source_dataset="__all__",
                )
            )
    return jobs


def selected_eval_layers(args: argparse.Namespace, selection: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    top2 = selection.get("top2_layers") or []
    if top2:
        layers = [int(x) for x in top2]
        return layers, layers, layers
    return (
        resolve_layer_spec(args.layers_full, args.model),
        resolve_layer_spec(args.layers_lean, args.model),
        resolve_layer_spec(args.layers_coarse, args.model),
    )


def eval_jobs(args: argparse.Namespace, selection: dict[str, Any]) -> list[dict[str, Any]]:
    alpha_star = float(selection.get("alpha_star", args.alpha_star))
    l_star = int(selection.get("l_star", args.l_star))
    layers_full, layers_lean, layers_coarse = selected_eval_layers(args, selection)
    jobs: list[dict[str, Any]] = []

    for layer in layers_full:
        jobs.append(
            base_job(
                args,
                exp="E2_neutral_layer_sweep",
                dataset="controlled",
                condition="A",
                pool="controlled_neutral_A",
                sample_filter="answerable_only",
                arm="comp1",
                layer=layer,
                alpha=alpha_star,
            )
        )
    for arm in ["comp2", "comp3", "residual_squad2", "residual_umwp", "residual_falseqa", "random_norm_matched_squad2"]:
        source_dataset = arm.replace("residual_", "").replace("random_norm_matched_", "") if "_" in arm else "__all__"
        for layer in layers_lean:
            jobs.append(
                base_job(
                    args,
                    exp="E2_neutral_layer_sweep",
                    dataset="controlled",
                    condition="A",
                    pool="controlled_neutral_A",
                    sample_filter="answerable_only",
                    arm=arm,
                    layer=layer,
                    alpha=alpha_star,
                    source_dataset=source_dataset,
                )
            )

    # Ordinary per-dataset diff-means baseline (the conventional steering vector
    # extracted directly from each existing dataset). cue_diff = comp1 (shared) + residual.
    for ds in args.datasets:
        for layer in layers_lean:
            jobs.append(
                base_job(
                    args,
                    exp="E2_neutral_layer_sweep",
                    dataset="controlled",
                    condition="A",
                    pool="controlled_neutral_A",
                    sample_filter="answerable_only",
                    arm="cue_diff",
                    layer=layer,
                    alpha=alpha_star,
                    source_dataset=ds,
                )
            )

    for condition in ["FP", "NC", "UNK"]:
        for layer in layers_lean:
            for arm, alpha in [("comp1", alpha_star), ("comp1", -alpha_star), ("random_norm_matched_squad2", alpha_star)]:
                jobs.append(
                    base_job(
                        args,
                        exp="E3_unanswerable_necessity",
                        dataset="controlled",
                        condition=condition,
                        pool=f"controlled_{condition}",
                        sample_filter="unanswerable_only",
                        arm=arm,
                        layer=layer,
                        alpha=alpha,
                        source_dataset="squad2" if arm.startswith("random") else "__all__",
                    )
                )

    for condition, sample_filter in [("A", "answerable_only"), ("NC", "unanswerable_only")]:
        for arm in ["comp1", "random_norm_matched_squad2"]:
            for alpha in _grid(args.dose_grid, V2_DOSE_GRID):
                jobs.append(
                    base_job(
                        args,
                        exp="E4_alpha_dose_response",
                        dataset="controlled",
                        condition=condition,
                        pool=f"controlled_{condition}",
                        sample_filter=sample_filter,
                        arm=arm,
                        layer=l_star,
                        alpha=alpha,
                        source_dataset="squad2" if arm.startswith("random") else "__all__",
                    )
                )

    for dataset in args.datasets:
        for layer in layers_coarse:
            for arm, source_dataset in [
                ("comp1", "__all__"),
                ("cue_diff", dataset),
                (f"residual_{dataset}", dataset),
                (f"random_norm_matched_{dataset}", dataset),
            ]:
                jobs.append(
                    base_job(
                        args,
                        exp="E5_benchmark_transfer",
                        dataset=dataset,
                        condition="",
                        pool=f"benchmark_{dataset}",
                        sample_filter="answerable_only",
                        arm=arm,
                        layer=layer,
                        alpha=alpha_star,
                        source_dataset=source_dataset,
                    )
                )

    for arm, source_dataset in [
        ("comp2", "__all__"),
        ("comp3", "__all__"),
        ("rank2_residual_squad2", "squad2"),
        ("rank2_residual_umwp", "umwp"),
        ("rank2_residual_falseqa", "falseqa"),
    ]:
        jobs.append(
            base_job(
                args,
                exp="E6_component_rank_controls",
                dataset="controlled",
                condition="A",
                pool="controlled_neutral_A",
                sample_filter="answerable_only",
                arm=arm,
                layer=l_star,
                alpha=alpha_star,
                source_dataset=source_dataset,
            )
        )

    for dataset, condition, sample_filter, pool, arms in [
        ("controlled", "A", "answerable_only", "controlled_neutral_A", [("comp1", alpha_star, "__all__"), ("refusal_dir", alpha_star, "__refusal__")]),
        ("advbench", "harmful", "unanswerable_only", "advbench_harmful", [("comp1", -alpha_star, "__all__"), ("refusal_dir", -alpha_star, "__refusal__")]),
    ]:
        jobs.append(
            base_job(
                args,
                exp="E7_refusal_dissociation",
                dataset=dataset,
                condition=condition,
                pool=pool,
                sample_filter=sample_filter,
                arm="comp1",
                layer=l_star,
                alpha=0.0,
                source_dataset="__all__",
            )
        )
        for arm, alpha, source_dataset in arms:
            jobs.append(
                base_job(
                    args,
                    exp="E7_refusal_dissociation",
                    dataset=dataset,
                    condition=condition,
                    pool=pool,
                    sample_filter=sample_filter,
                    arm=arm,
                    layer=l_star,
                    alpha=alpha,
                    source_dataset=source_dataset,
                )
            )

    # E8: cross-dataset steering-transfer matrix. Steer with the ordinary diff-means
    # extracted from EACH source dataset (cue_diff) applied to EACH target dataset/pool,
    # plus the shared comp1 row as the generalizing reference. Off-diagonal = cross-dataset
    # transfer (does a vector from dataset A move abstention on dataset B?).
    e8_sources = [("cue_diff", ds) for ds in args.datasets] + [("comp1", "__all__")]
    e8_targets = [(ds, "", f"benchmark_{ds}") for ds in args.datasets] + [("controlled", "A", "controlled_neutral_A")]
    for arm, src in e8_sources:
        for t_dataset, t_condition, t_pool in e8_targets:
            jobs.append(
                base_job(
                    args,
                    exp="E8_cross_dataset_transfer",
                    dataset=t_dataset,
                    condition=t_condition,
                    pool=t_pool,
                    sample_filter="answerable_only",
                    arm=arm,
                    layer=l_star,
                    alpha=alpha_star,
                    source_dataset=src,
                )
            )
    return jobs


def main() -> None:
    args = parse_args()
    root = v2_run_root(args.run_id)
    out_dir = root / "jobs"
    selection = load_selection(args.selection_json)
    if args.mode == "calibration":
        jobs = calibration_jobs(args)
    elif args.mode == "aux_discovery":
        jobs = aux_discovery_jobs(args, selection)
    else:
        jobs = eval_jobs(args, selection)
    output = out_dir / f"{args.model}.{args.mode}.jobs.jsonl"
    write_jsonl(output, jobs)
    write_manifest(
        out_dir / f"{args.model}.{args.mode}.manifest.json",
        {
            "stage": "build_v2_jobs",
            "run_id": args.run_id,
            "model": args.model,
            "mode": args.mode,
            "n_jobs": len(jobs),
            "jobs_jsonl": str(output),
            "selection_json": args.selection_json,
        },
    )
    print(f"[DONE] {output} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
