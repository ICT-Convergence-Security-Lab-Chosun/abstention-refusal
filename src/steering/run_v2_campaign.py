#!/usr/bin/env python3
"""Run v2 steering generation jobs with one model load and tag-based resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from instruction_common import (
    default_cache_env,
    format_prompt,
    keyword_abstained,
    load_model_and_tokenizer,
    model_metadata,
    set_seed,
    write_manifest,
)
from instruction_config import MODEL_ID_BY_KEY
from steering_hooks import apply_steering_hook, probe_layer_to_module_layer
from v2_common import append_jsonl, done_keys, job_tag, read_jsonl, row_reference_answers, v2_run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ID_BY_KEY))
    parser.add_argument("--jobs_jsonl", required=True)
    parser.add_argument("--samples_jsonl", default=None, help="Defaults to runs/{run_id}/splits/all_v2.jsonl.")
    parser.add_argument("--vectors_npz", default=None, help="Defaults to runs/{run_id}/vectors/{model}/v2_vectors.npz.")
    parser.add_argument("--output_dir", default=None, help="Defaults to runs/{run_id}/generations/{model}.")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def simple_answer_correct(generation: str, reference_answers: Any) -> bool:
    if not reference_answers:
        return False
    refs = [reference_answers] if isinstance(reference_answers, str) else list(reference_answers)
    gen = normalize_text(generation)
    return any(normalize_text(str(ref)) and normalize_text(str(ref)) in gen for ref in refs)


def load_v2_vector(npz_path: Path, *, model: str, arm: str, layer: int, source_dataset: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"vectors", "models", "arms", "layers", "source_dataset"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{npz_path} missing arrays: {sorted(missing)}")
        mask = (
            (data["models"].astype(str) == model)
            & (data["arms"].astype(str) == arm)
            & (data["layers"].astype(int) == int(layer))
            & (data["source_dataset"].astype(str) == source_dataset)
        )
        idx = np.flatnonzero(mask)
        if idx.shape[0] != 1:
            raise ValueError(
                f"Expected one vector for model={model} arm={arm} layer={layer} "
                f"source_dataset={source_dataset}; got {idx.shape[0]}"
            )
        return data["vectors"][int(idx[0])].astype(np.float32, copy=True)


def sample_matches(sample: dict[str, Any], job: dict[str, Any]) -> bool:
    if sample.get("split") != job.get("split"):
        return False
    if sample.get("dataset") != job.get("dataset"):
        return False
    condition = str(job.get("condition") or "")
    if condition and sample.get("condition") != condition:
        return False
    sample_filter = job.get("sample_filter", "all")
    should_abstain = bool(sample.get("should_abstain"))
    if sample_filter == "answerable_only" and should_abstain:
        return False
    if sample_filter == "unanswerable_only" and not should_abstain:
        return False
    return True


def select_samples(samples: list[dict[str, Any]], job: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [sample for sample in samples if sample_matches(sample, job)]
    limit = job.get("sample_limit")
    if limit is not None:
        selected = selected[: int(limit)]
    return selected


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    vector: np.ndarray | None,
    job: dict[str, Any],
    max_length: int,
    max_new_tokens: int,
) -> list[str]:
    device = next(model.parameters()).device
    rendered = [format_prompt(prompt, tokenizer, prompt_mode="neutral") for prompt in prompts]
    inputs = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    alpha = float(job["alpha"])
    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if alpha == 0.0 or vector is None:
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)
    else:
        with apply_steering_hook(
            model,
            vector,
            probe_layer=int(job["layer"]),
            alpha=alpha,
            token_position=job.get("token_position", "all"),
            scale_mode=job.get("scale_mode", "raw"),
            steering_mode=job.get("steering_mode", "all_steps"),
        ):
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generate_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True) for i in range(output_ids.shape[0])]


def build_output_row(
    args: argparse.Namespace,
    job: dict[str, Any],
    tag: str,
    sample: dict[str, Any],
    generation: str,
    vectors_npz: Path,
) -> dict[str, Any]:
    model_abstained = keyword_abstained(generation)
    reference_answers = row_reference_answers(sample)
    return {
        **job,
        "tag": tag,
        "sample_id": str(sample["sample_id"]),
        "probe_layer": int(job["layer"]),
        "vector_type": str(job["arm"]),
        "vector_source_model": args.model,
        "vector_source_dataset": str(job.get("source_dataset", "__all__")),
        "vector_position": str(job.get("position", "t_post")),
        "prompt": sample["question"],
        "question": sample["question"],
        "generation": generation,
        "reference_answers": reference_answers,
        "should_abstain": bool(sample["should_abstain"]),
        "model_abstained": model_abstained,
        "is_answer_correct": False
        if bool(sample["should_abstain"]) or model_abstained
        else simple_answer_correct(generation, reference_answers),
        "hook_module_layer": probe_layer_to_module_layer(int(job["layer"])),
        "model_id": MODEL_ID_BY_KEY[args.model],
        "vector_npz": str(vectors_npz),
    }


def load_samples(root: Path, samples_path: Path) -> list[dict[str, Any]]:
    samples = read_jsonl(samples_path)
    e7_samples = root / "splits" / "e7_refusal_samples.jsonl"
    if e7_samples.exists():
        samples.extend(read_jsonl(e7_samples))
    return samples


def main() -> None:
    args = parse_args()
    default_cache_env()
    set_seed(args.seed)
    root = v2_run_root(args.run_id)
    samples_path = Path(args.samples_jsonl) if args.samples_jsonl else root / "splits" / "all_v2.jsonl"
    vectors_npz = Path(args.vectors_npz) if args.vectors_npz else root / "vectors" / args.model / "v2_vectors.npz"
    output_dir = Path(args.output_dir) if args.output_dir else root / "generations" / args.model
    output_path = output_dir / "generations.jsonl"
    jobs = read_jsonl(Path(args.jobs_jsonl))
    samples = load_samples(root, samples_path)
    already_done = done_keys(output_path)
    print(f"[PLAN] jobs={len(jobs)} samples={len(samples)} done={len(already_done)} output={output_path}", flush=True)
    if args.dry_run:
        for job in jobs[:10]:
            print(json.dumps({**job, "n_samples": len(select_samples(samples, job)), "tag": job_tag(job)}, ensure_ascii=False), flush=True)
        return

    model, tokenizer = load_model_and_tokenizer(args.model)
    written = 0
    skipped = 0
    for job_index, job in enumerate(jobs, start=1):
        tag = job_tag(job)
        selected = select_samples(samples, job)
        print(f"[JOB] {job_index}/{len(jobs)} tag={tag} n={len(selected)}", flush=True)
        vector = None
        if float(job["alpha"]) != 0.0:
            vector = load_v2_vector(
                vectors_npz,
                model=args.model,
                arm=str(job["arm"]),
                layer=int(job["layer"]),
                source_dataset=str(job.get("source_dataset", "__all__")),
            )
        todo = []
        for sample in selected:
            if (tag, str(sample["sample_id"])) in already_done:
                skipped += 1
            else:
                todo.append(sample)
        batch_rows = []
        for start in range(0, len(todo), args.generation_batch_size):
            chunk = todo[start : start + args.generation_batch_size]
            if start == 0 or start + len(chunk) >= len(todo) or (start // args.generation_batch_size) % 5 == 0:
                print(
                    f"[GENERATE] job={job_index}/{len(jobs)} todo={start + len(chunk)}/{len(todo)} skipped={skipped}",
                    flush=True,
                )
            generations = generate_batch(
                model,
                tokenizer,
                [str(sample["question"]) for sample in chunk],
                vector,
                job,
                args.max_length,
                args.max_new_tokens,
            )
            for sample, generation in zip(chunk, generations):
                row = build_output_row(args, job, tag, sample, generation, vectors_npz)
                batch_rows.append(row)
                already_done.add((tag, str(sample["sample_id"])))
                written += 1
                if len(batch_rows) >= 25:
                    append_jsonl(output_path, batch_rows)
                    batch_rows = []
        if batch_rows:
            append_jsonl(output_path, batch_rows)

    write_manifest(
        output_dir / "manifest.json",
        {
            "stage": "run_v2_campaign",
            "run_id": args.run_id,
            "model": model_metadata(args.model, model),
            "jobs_jsonl": args.jobs_jsonl,
            "samples_jsonl": str(samples_path),
            "vectors_npz": str(vectors_npz),
            "generations_jsonl": str(output_path),
            "written": written,
            "skipped": skipped,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "generation_batch_size": args.generation_batch_size,
            "seed": args.seed,
        },
    )
    print(f"[DONE] written={written} skipped={skipped} output={output_path}", flush=True)


if __name__ == "__main__":
    main()
