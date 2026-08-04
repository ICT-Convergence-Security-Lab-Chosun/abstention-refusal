#!/usr/bin/env python3
"""Extract v2 cue directions, SVD components, residuals, and geometry tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from instruction_common import (
    default_cache_env,
    format_prompt,
    hidden_state_count,
    load_model_and_tokenizer,
    load_source_rows,
    model_metadata,
    normalize_layers,
    set_seed,
    write_csv,
    write_manifest,
)
from instruction_config import DEFAULT_MODEL_KEY, MODEL_ID_BY_KEY
from v2_common import BENCHMARK_DATASETS, cosine, resolve_layer_spec, unit, v2_run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=sorted(MODEL_ID_BY_KEY))
    parser.add_argument("--input_jsonl", default=None, help="Defaults to runs/{run_id}/splits/cue_extract.jsonl.")
    parser.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS), choices=list(BENCHMARK_DATASETS))
    parser.add_argument("--layers", default="all", help="'all', '1:32', or a space/comma separated list.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--hidden_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def layer_list(raw: str, model_key: str, model) -> list[int]:
    if raw == "all":
        return normalize_layers(range(1, hidden_state_count(model)), model)
    return normalize_layers(resolve_layer_spec(raw, model_key), model)


def collect_tpost_hidden(model, tokenizer, rows: list[dict[str, Any]], batch_size: int, max_length: int, hidden_dtype: str) -> np.ndarray:
    device = next(model.parameters()).device
    torch_dtype = torch.float16 if hidden_dtype == "float16" else torch.float32
    chunks: list[np.ndarray] = []
    total = (len(rows) + batch_size - 1) // batch_size
    for start in range(0, len(rows), batch_size):
        batch_no = start // batch_size + 1
        if batch_no == 1 or batch_no % 20 == 0 or batch_no == total:
            print(f"[FORWARD] batch={batch_no}/{total}", flush=True)
        batch = rows[start : start + batch_size]
        prompts = [format_prompt(str(row["question"]), tokenizer, prompt_mode="neutral") for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        # Left padding means the final prompt token is at index -1 for every row.
        last_token = torch.stack([hidden[:, -1, :] for hidden in outputs.hidden_states], dim=1)
        chunks.append(last_token.to(torch_dtype).cpu().numpy())
        del outputs, last_token
    return np.concatenate(chunks, axis=0)


def mean_diff(x_layer: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        x_layer[y == 1].astype(np.float32).mean(axis=0)
        - x_layer[y == 0].astype(np.float32).mean(axis=0)
    ).astype(np.float32)


def projection_auc(x_layer: np.ndarray, y: np.ndarray, direction: np.ndarray) -> float:
    scores = x_layer.astype(np.float32) @ unit(direction)
    if len(np.unique(y)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y, scores))
    return max(auc, 1.0 - auc)


def save_npz(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vectors=np.stack([row["vector"] for row in records], axis=0).astype(np.float32),
        models=np.asarray([row["model"] for row in records]),
        arms=np.asarray([row["arm"] for row in records]),
        datasets=np.asarray([row["dataset"] for row in records]),
        source_dataset=np.asarray([row["source_dataset"] for row in records]),
        layers=np.asarray([row["layer"] for row in records], dtype=np.int32),
        positions=np.asarray([row["position"] for row in records]),
        base_vector_type=np.asarray([row["base_vector_type"] for row in records]),
        norm=np.asarray([row["norm"] for row in records], dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    default_cache_env()
    set_seed(args.seed)
    root = v2_run_root(args.run_id)
    input_jsonl = Path(args.input_jsonl) if args.input_jsonl else root / "splits" / "cue_extract.jsonl"
    output_dir = root / "vectors" / args.model
    rows = load_source_rows(str(input_jsonl), set(args.datasets))
    model, tokenizer = load_model_and_tokenizer(args.model)
    layers = layer_list(args.layers, args.model, model)

    rows_by_dataset = {dataset: [row for row in rows if row["dataset"] == dataset] for dataset in args.datasets}
    hidden_by_dataset: dict[str, np.ndarray] = {}
    labels_by_dataset: dict[str, np.ndarray] = {}
    for dataset, dataset_rows in rows_by_dataset.items():
        if not dataset_rows:
            raise ValueError(f"No cue rows for dataset={dataset}")
        labels = np.asarray([1 if row["should_abstain"] else 0 for row in dataset_rows], dtype=np.int64)
        if len(np.unique(labels)) < 2:
            raise ValueError(f"Cue rows for dataset={dataset} need both should_abstain classes.")
        print(f"[DATASET] {dataset} n={len(dataset_rows)} pos={int(labels.sum())}", flush=True)
        hidden_by_dataset[dataset] = collect_tpost_hidden(
            model,
            tokenizer,
            dataset_rows,
            args.batch_size,
            args.max_length,
            args.hidden_dtype,
        )
        labels_by_dataset[dataset] = labels

    vector_records: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    geometry_json: dict[str, Any] = {"layers": {}, "datasets": args.datasets}
    rng = np.random.default_rng(args.seed)
    for layer in layers:
        dirs = []
        raw_dirs: dict[str, np.ndarray] = {}
        for dataset in args.datasets:
            x_layer = hidden_by_dataset[dataset][:, layer, :]
            d_raw = mean_diff(x_layer, labels_by_dataset[dataset])
            d_unit = unit(d_raw)
            raw_dirs[dataset] = d_unit
            dirs.append(d_unit)
            vector_records.append(
                {
                    "model": args.model,
                    "arm": "cue_diff",
                    "dataset": dataset,
                    "source_dataset": dataset,
                    "layer": layer,
                    "position": "t_post",
                    "base_vector_type": "cue_diff",
                    "vector": d_unit,
                    "norm": float(np.linalg.norm(d_unit)),
                }
            )
        matrix = np.stack(dirs, axis=0).astype(np.float32)
        _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        components = [unit(vh[i].astype(np.float32)) for i in range(min(3, vh.shape[0]))]
        for idx, comp in enumerate(components):
            mean_projection = float(np.mean(matrix @ comp))
            if mean_projection < 0:
                components[idx] = -comp
        comp1 = components[0]
        explained = (singular_values ** 2) / max(float(np.sum(singular_values ** 2)), 1e-12)
        layer_key = str(layer)
        geometry_json["layers"][layer_key] = {
            "singular_values": [float(x) for x in singular_values],
            "explained_variance": [float(x) for x in explained],
            "residuals": {},
            "leave_one_out_auc": {},
        }

        for idx, comp in enumerate(components, start=1):
            vector_records.append(
                {
                    "model": args.model,
                    "arm": f"comp{idx}",
                    "dataset": "__common__",
                    "source_dataset": "__all__",
                    "layer": layer,
                    "position": "t_post",
                    "base_vector_type": "cue_diff",
                    "vector": comp.astype(np.float32),
                    "norm": float(np.linalg.norm(comp)),
                }
            )
        for dataset, d_unit in raw_dirs.items():
            proj1 = float(np.dot(d_unit, comp1))
            residual1 = (d_unit - proj1 * comp1).astype(np.float32)
            residual1_norm = float(np.linalg.norm(residual1))
            residual2 = residual1.copy()
            if len(components) >= 2:
                comp2 = components[1]
                residual2 = (residual2 - float(np.dot(residual2, comp2)) * comp2).astype(np.float32)
            residual2_norm = float(np.linalg.norm(residual2))
            random_vec = unit(rng.normal(size=d_unit.shape).astype(np.float32)) * max(residual1_norm, 1e-12)
            for arm, vector, norm in (
                (f"residual_{dataset}", residual1, residual1_norm),
                (f"rank2_residual_{dataset}", residual2, residual2_norm),
                (f"random_norm_matched_{dataset}", random_vec.astype(np.float32), float(np.linalg.norm(random_vec))),
            ):
                vector_records.append(
                    {
                        "model": args.model,
                        "arm": arm,
                        "dataset": dataset,
                        "source_dataset": dataset,
                        "layer": layer,
                        "position": "t_post",
                        "base_vector_type": "cue_diff",
                        "vector": vector.astype(np.float32),
                        "norm": norm,
                    }
                )
            geometry_json["layers"][layer_key]["residuals"][dataset] = {
                "projection_on_comp1": proj1,
                "residual_norm": residual1_norm,
                "rank2_residual_norm": residual2_norm,
            }

        for heldout in args.datasets:
            train = [raw_dirs[d] for d in args.datasets if d != heldout]
            if not train:
                continue
            _, _, vh_loo = np.linalg.svd(np.stack(train, axis=0), full_matrices=False)
            loo_dir = unit(vh_loo[0].astype(np.float32))
            if float(np.mean(np.stack(train, axis=0) @ loo_dir)) < 0:
                loo_dir = -loo_dir
            auc = projection_auc(hidden_by_dataset[heldout][:, layer, :], labels_by_dataset[heldout], loo_dir)
            geometry_json["layers"][layer_key]["leave_one_out_auc"][heldout] = auc

        for i, left in enumerate(args.datasets):
            for right in args.datasets[i + 1 :]:
                geometry_rows.append(
                    {
                        "model": args.model,
                        "layer": layer,
                        "metric": "cue_diff_cosine",
                        "left": left,
                        "right": right,
                        "value": cosine(raw_dirs[left], raw_dirs[right]),
                    }
                )
        for idx, value in enumerate(explained, start=1):
            geometry_rows.append(
                {
                    "model": args.model,
                    "layer": layer,
                    "metric": f"svd_explained_comp{idx}",
                    "left": "__all__",
                    "right": "",
                    "value": float(value),
                }
            )
        for dataset, auc in geometry_json["layers"][layer_key]["leave_one_out_auc"].items():
            geometry_rows.append(
                {
                    "model": args.model,
                    "layer": layer,
                    "metric": "leave_one_out_auc",
                    "left": dataset,
                    "right": "",
                    "value": auc,
                }
            )

    save_npz(output_dir / "v2_vectors.npz", vector_records)
    write_csv(output_dir / "geometry.csv", ["model", "layer", "metric", "left", "right", "value"], geometry_rows)
    (output_dir / "geometry.json").write_text(json.dumps(geometry_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        output_dir / "manifest.json",
        {
            "stage": "extract_v2_geometry_vectors",
            "run_id": args.run_id,
            "model": model_metadata(args.model, model),
            "input_jsonl": str(input_jsonl),
            "datasets": args.datasets,
            "layers": layers,
            "position": "t_post",
            "outputs": {
                "vectors": str(output_dir / "v2_vectors.npz"),
                "geometry_json": str(output_dir / "geometry.json"),
                "geometry_csv": str(output_dir / "geometry.csv"),
            },
        },
    )
    print(f"[DONE] {output_dir / 'v2_vectors.npz'}")


if __name__ == "__main__":
    main()

