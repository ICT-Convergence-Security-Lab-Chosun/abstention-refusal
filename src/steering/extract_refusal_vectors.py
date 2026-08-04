#!/usr/bin/env python3
"""Extract refusal directions and merge them into v2_vectors.npz for E7."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from instruction_common import default_cache_env, format_prompt, load_model_and_tokenizer, model_metadata, set_seed, write_csv, write_manifest
from instruction_config import DEFAULT_MODEL_KEY, MODEL_ID_BY_KEY
from v2_common import cosine, unit, v2_run_root, write_jsonl


HARMFUL_FIELDS = ("bad_q", "goal", "prompt", "instruction", "question")
HARMLESS_FIELDS = ("instruction", "prompt", "question")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=sorted(MODEL_ID_BY_KEY))
    parser.add_argument("--advbench_path", required=True)
    parser.add_argument("--alpaca_path", required=True)
    parser.add_argument("--vectors_npz", default=None, help="Defaults to runs/{run_id}/vectors/{model}/v2_vectors.npz.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--hidden_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--e7_eval_n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
        raise ValueError(f"Unsupported JSON structure: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported table format: {path}")


def prompt_from_row(row: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def prompt_rows(path: Path, fields: tuple[str, ...], limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(load_table(path)):
        prompt = prompt_from_row(row, fields)
        if prompt:
            rows.append({"sample_id": f"{path.stem}:{idx}", "question": prompt})
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def collect_tpost_hidden(model, tokenizer, rows: list[dict[str, Any]], batch_size: int, max_length: int, hidden_dtype: str) -> np.ndarray:
    device = next(model.parameters()).device
    torch_dtype = torch.float16 if hidden_dtype == "float16" else torch.float32
    chunks = []
    total = (len(rows) + batch_size - 1) // batch_size
    for start in range(0, len(rows), batch_size):
        batch_no = start // batch_size + 1
        if batch_no == 1 or batch_no % 10 == 0 or batch_no == total:
            print(f"[FORWARD] batch={batch_no}/{total}", flush=True)
        batch = rows[start : start + batch_size]
        prompts = [format_prompt(row["question"], tokenizer, prompt_mode="neutral") for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        last_token = torch.stack([hidden[:, -1, :] for hidden in outputs.hidden_states], dim=1)
        chunks.append(last_token.to(torch_dtype).cpu().numpy())
        del outputs, last_token
    return np.concatenate(chunks, axis=0)


def existing_records(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    with np.load(path, allow_pickle=False) as data:
        layers = sorted({int(layer) for layer, arm in zip(data["layers"], data["arms"].astype(str)) if arm == "comp1"})
        records = []
        for i, vector in enumerate(data["vectors"]):
            if str(data["arms"][i]) == "refusal_dir":
                continue
            records.append(
                {
                    "vector": vector.astype(np.float32),
                    "model": str(data["models"][i]),
                    "arm": str(data["arms"][i]),
                    "dataset": str(data["datasets"][i]),
                    "source_dataset": str(data["source_dataset"][i]),
                    "layer": int(data["layers"][i]),
                    "position": str(data["positions"][i]),
                    "base_vector_type": str(data["base_vector_type"][i]),
                    "norm": float(data["norm"][i]),
                }
            )
    if not layers:
        raise ValueError(f"No comp1 layers found in {path}")
    return records, layers


def save_records(path: Path, records: list[dict[str, Any]]) -> None:
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


def comp1_lookup(records: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    return {
        int(row["layer"]): row["vector"]
        for row in records
        if row["arm"] == "comp1" and row["source_dataset"] == "__all__"
    }


def write_e7_samples(root: Path, harmful_rows: list[dict[str, Any]], n: int) -> Path:
    out = root / "splits" / "e7_refusal_samples.jsonl"
    rows = []
    for idx, row in enumerate(harmful_rows[:n]):
        rows.append(
            {
                "sample_id": f"advbench:test:{idx}",
                "dataset": "advbench",
                "condition": "harmful",
                "question": row["question"],
                "prompt": row["question"],
                "reference_answers": [],
                "should_abstain": True,
                "split": "test",
                "v2_pool": "e7_refusal",
                "sample_filter_group": "unanswerable_only",
            }
        )
    write_jsonl(out, rows)
    return out


def main() -> None:
    args = parse_args()
    default_cache_env()
    set_seed(args.seed)
    root = v2_run_root(args.run_id)
    vectors_npz = Path(args.vectors_npz) if args.vectors_npz else root / "vectors" / args.model / "v2_vectors.npz"
    out_dir = vectors_npz.parent
    records, layers = existing_records(vectors_npz)
    harmful_rows = prompt_rows(Path(args.advbench_path), HARMFUL_FIELDS, args.limit)
    harmless_rows = prompt_rows(Path(args.alpaca_path), HARMLESS_FIELDS, args.limit)
    print(f"[DATA] harmful={len(harmful_rows)} harmless={len(harmless_rows)}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model)
    harmful_hidden = collect_tpost_hidden(model, tokenizer, harmful_rows, args.batch_size, args.max_length, args.hidden_dtype)
    harmless_hidden = collect_tpost_hidden(model, tokenizer, harmless_rows, args.batch_size, args.max_length, args.hidden_dtype)
    harmful_mean = harmful_hidden.astype(np.float32).mean(axis=0)
    harmless_mean = harmless_hidden.astype(np.float32).mean(axis=0)
    comp1 = comp1_lookup(records)
    cosine_rows = []
    for layer in layers:
        refusal = unit(harmful_mean[layer] - harmless_mean[layer])
        records.append(
            {
                "vector": refusal,
                "model": args.model,
                "arm": "refusal_dir",
                "dataset": "__refusal__",
                "source_dataset": "__refusal__",
                "layer": layer,
                "position": "t_post",
                "base_vector_type": "refusal_diff",
                "norm": float(np.linalg.norm(refusal)),
            }
        )
        cosine_rows.append(
            {
                "model": args.model,
                "layer": layer,
                "metric": "cos_comp1_refusal_dir",
                "value": cosine(comp1[layer], refusal) if layer in comp1 else "",
            }
        )
    save_records(vectors_npz, records)
    write_csv(out_dir / "refusal_geometry.csv", ["model", "layer", "metric", "value"], cosine_rows)
    e7_samples = write_e7_samples(root, harmful_rows, args.e7_eval_n)
    write_manifest(
        out_dir / "refusal_manifest.json",
        {
            "stage": "extract_refusal_vectors",
            "run_id": args.run_id,
            "model": model_metadata(args.model, model),
            "advbench_path": args.advbench_path,
            "alpaca_path": args.alpaca_path,
            "vectors_npz": str(vectors_npz),
            "refusal_geometry_csv": str(out_dir / "refusal_geometry.csv"),
            "e7_samples_jsonl": str(e7_samples),
            "layers": layers,
            "limit": args.limit,
            "e7_eval_n": args.e7_eval_n,
        },
    )
    print(f"[DONE] merged refusal_dir into {vectors_npz}")


if __name__ == "__main__":
    main()
