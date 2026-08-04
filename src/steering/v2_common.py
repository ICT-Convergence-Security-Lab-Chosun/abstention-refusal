#!/usr/bin/env python3
"""Shared helpers for the v2 paper-grade abstention steering package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from instruction_common import parse_bool, run_root


CONTROL_CONDITIONS = ("A", "FP", "NC", "UNK")
CONTROL_SHOULD_ABSTAIN = {"A": False, "FP": True, "NC": True, "UNK": True}
BENCHMARK_DATASETS = ("squad2", "umwp", "falseqa")
V2_ALPHA_GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
V2_DOSE_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
MODEL_TOTAL_LAYERS = {
    "llama31_8b_instruct": 32,
    "qwen25_7b_instruct": 28,
    "gemma2_9b_it": 42,
    "mistral7b_instruct_v03": 32,
    "qwen35_9b": 32,
    "gemma3_12b": 48,
}



@dataclass(frozen=True)
class VectorKey:
    model: str
    arm: str
    layer: int
    source_dataset: str = "__all__"
    base_vector_type: str = "cue_diff"
    position: str = "t_post"


def unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return (arr / norm).astype(np.float32, copy=False)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / den) if den > 1e-12 else float("nan")


def stable_int(text: str, seed: int) -> int:
    raw = f"v2:{seed}:{text}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest(), 16)


def stable_order(rows: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_int(str(row.get("sample_id", "")), seed))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_metadata_json(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def entity_key(row: dict[str, Any]) -> str:
    if row.get("entity"):
        return str(row["entity"]).strip().lower()
    meta = parse_metadata_json(row)
    for key in ("entity", "raw_entity", "topic"):
        if meta.get(key):
            return str(meta[key]).strip().lower()
    return ""


def coerce_should_abstain(value: Any) -> bool:
    return parse_bool(value)


def row_reference_answers(row: dict[str, Any]) -> Any:
    if "reference_answers" in row:
        return row["reference_answers"]
    raw = row.get("reference_answers_json")
    if raw is None:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return raw


def v2_run_root(run_id: str) -> Path:
    return run_root(run_id)


def job_tag(job: dict[str, Any]) -> str:
    keys = [
        "exp",
        "pool",
        "split",
        "dataset",
        "condition",
        "arm",
        "layer",
        "alpha",
        "source_dataset",
        "sample_filter",
    ]
    return "|".join(f"{key}={job.get(key, '')}" for key in keys)


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    out: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        tag = str(row.get("tag") or "")
        sample_id = str(row.get("sample_id") or "")
        if tag and sample_id:
            out.add((tag, sample_id))
    return out


def parse_layers(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        text = raw.strip()
        if ":" in text:
            start, end = text.split(":", 1)
            return list(range(int(start), int(end) + 1))
        return [int(part) for part in text.replace(",", " ").split() if part]
    return [int(x) for x in raw]


def model_total_layers(model_key: str) -> int:
    if model_key not in MODEL_TOTAL_LAYERS:
        raise KeyError(f"No layer default for model={model_key!r}; pass explicit layer lists.")
    return MODEL_TOTAL_LAYERS[model_key]


def unique_sorted_layers(values: Iterable[int], total_layers: int) -> list[int]:
    return sorted({max(1, min(total_layers, int(v))) for v in values})


def relative_layers(model_key: str, fractions: Iterable[float]) -> list[int]:
    total = model_total_layers(model_key)
    return unique_sorted_layers((round(total * frac) for frac in fractions), total)


def resolve_layer_spec(raw: str | Iterable[int], model_key: str) -> list[int]:
    if not isinstance(raw, str):
        return unique_sorted_layers(raw, model_total_layers(model_key))
    text = raw.strip().lower()
    total = model_total_layers(model_key)
    if text in {"auto", "auto_full"}:
        return list(range(1, total + 1))
    if text == "auto_lean":
        return relative_layers(model_key, (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0))
    if text == "auto_coarse":
        return relative_layers(model_key, (0.25, 0.5, 0.75, 1.0))
    if text == "auto_calib":
        return relative_layers(model_key, (0.2, 0.35, 0.5, 0.65, 0.8))
    if text == "auto_mid":
        return relative_layers(model_key, (0.5,))
    return unique_sorted_layers(parse_layers(raw), total)

