#!/usr/bin/env python3
"""Shared helpers for instruction-model abstention steering experiments."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import torch

from instruction_config import ABSTAIN_SUBSTRINGS, DATASET_NAME_MAP, MODEL_ID_BY_KEY, MODEL_SPECS, SYSTEM_PROMPT, model_spec_dict


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSONL = PACKAGE_ROOT / "data" / "squad2_umwp_falseqa_rawq_normalized.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_dataset_name(raw_name: Any) -> str | None:
    return DATASET_NAME_MAP.get(str(raw_name).strip().lower())


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return parse_bool(value)
    except ValueError:
        return None


def parse_reference_answers(row: dict):
    value = row.get("reference_answers")
    if value is not None:
        return value
    value = row.get("reference_answers_json")
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def resolve_input_path(raw_path: str | None) -> Path:
    if raw_path is None:
        return DEFAULT_INPUT_JSONL
    path = Path(raw_path)
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        alt = PACKAGE_ROOT / raw_path
        if alt.exists():
            return alt.resolve()
    raise FileNotFoundError(f"Input JSONL not found: {raw_path}")


def load_source_rows(path: str | None, datasets: set[str] | None = None) -> List[dict]:
    input_path = resolve_input_path(path)
    rows: List[dict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            dataset = normalize_dataset_name(raw.get("dataset_class") or raw.get("dataset") or raw.get("dataset_config"))
            if dataset is None:
                continue
            if datasets is not None and dataset not in datasets:
                continue
            question = str(raw.get("question") or raw.get("prompt") or "").strip()
            if not question:
                continue
            rows.append(
                {
                    **raw,
                    "sample_id": str(raw.get("sample_id") or raw.get("pilot_sample_id") or f"{dataset}:{raw.get('source_index', idx)}"),
                    "dataset": dataset,
                    "question": question,
                    "prompt": question,
                    "should_abstain": parse_bool(raw.get("should_abstain")),
                    "reference_answers": parse_reference_answers(raw),
                }
            )
    if not rows:
        raise ValueError(f"No usable rows found in {input_path}")
    return rows


def filter_rows(rows: List[dict], dataset: str, limit: int | None, sample_filter: str, seed: int = 42) -> List[dict]:
    filtered = [r for r in rows if r["dataset"] == dataset]
    if sample_filter == "answerable_only":
        filtered = [r for r in filtered if not r["should_abstain"]]
    elif sample_filter == "unanswerable_only":
        filtered = [r for r in filtered if r["should_abstain"]]
    elif sample_filter != "all":
        raise ValueError(f"Unknown sample_filter={sample_filter!r}")
    rng = random.Random(seed)
    rng.shuffle(filtered)
    return filtered[:limit] if limit is not None else filtered


def keyword_abstained(text: str) -> bool:
    lower = text.strip().lower()
    return any(needle in lower for needle in ABSTAIN_SUBSTRINGS)


def load_model_and_tokenizer(model_key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_key not in MODEL_ID_BY_KEY:
        raise KeyError(f"Unknown model_key={model_key!r}; choices={sorted(MODEL_ID_BY_KEY)}")
    model_id = MODEL_ID_BY_KEY[model_key]
    family = getattr(MODEL_SPECS.get(model_key), "family", "")
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    print(f"[LOAD] {model_key}: {model_id}", flush=True)
    if family == "qwen3_5":
        # Multimodal (vision + text) checkpoint: the text decoder lives under
        # model.model.language_model. Load the full model via the multimodal auto-class
        # (requires transformers >= 5.3; imported lazily so the 4.x pipeline env is unaffected)
        # and operate text-only. get_hook_module() targets the language_model layers.
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cuda:0" if use_cuda else None,
            local_files_only=local_files_only,
        )
    elif family == "gemma4_unified":
        # Audio+vision+text checkpoint (transformers >= 5.15). AutoModelForCausalLM opens it
        # as Gemma4UnifiedForConditionalGeneration with the text decoder at
        # model.language_model.layers (48), which get_hook_module() already steps into. The
        # new transformers takes `dtype` rather than the deprecated `torch_dtype`.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cuda:0" if use_cuda else None,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    elif family == "gemma3":
        # Vision+text, non-reasoning (transformers >= 5.3). Loads as
        # Gemma3ForConditionalGeneration with the text decoder at model.language_model.layers,
        # which get_hook_module() steps into exactly as for gemma4. Uses the 5.3-era
        # `torch_dtype` kwarg since gemma3 runs under the frozen pipeline env.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cuda:0" if use_cuda else None,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="cuda:0" if use_cuda else None,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    if not use_cuda:
        model.to("cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only, trust_remote_code=False)
    # Gemma 4 ships its chat template as a separate chat_template.jinja that the tokenizer
    # does not auto-load; inject it so format_prompt() renders turns correctly.
    if not getattr(tokenizer, "chat_template", None):
        import os as _os
        tmpl = _os.path.join(model_id, "chat_template.jinja")
        if _os.path.exists(tmpl):
            tokenizer.chat_template = open(tmpl).read()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def build_user_content(question: str, prompt_mode: str = "neutral") -> str:
    if prompt_mode == "neutral":
        return question
    if prompt_mode == "prompt_only_abstention":
        return (
            "Answer the following question. If the question cannot be answered as stated, "
            "explain why instead of giving a direct answer.\n\n"
            f"Question:\n{question}\n\nAnswer:"
        )
    raise ValueError(f"Unknown prompt_mode={prompt_mode!r}")


def format_prompt(question: str, tokenizer, prompt_mode: str = "neutral", system_prompt: str = SYSTEM_PROMPT) -> str:
    user_content = build_user_content(question, prompt_mode)
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        # Disable reasoning/thinking traces for models whose template supports the toggle
        # (e.g. Qwen3.x). Harmless no-op for templates that do not use it.
        extra = {"enable_thinking": False} if "enable_thinking" in chat_template else {}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
        except Exception as exc:
            if "System role not supported" not in str(exc):
                raise
            merged_user_content = f"{system_prompt}\n\n{user_content}" if system_prompt else user_content
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": merged_user_content}],
                tokenize=False,
                add_generation_prompt=True,
                **extra,
            )
    return f"System: {system_prompt}\nUser: {user_content}\nAssistant:"


def hidden_state_count(model) -> int:
    cfg = model.config
    n = getattr(cfg, "num_hidden_layers", None)
    if not n:
        # multimodal wrappers (e.g. Qwen3.5) nest the layer count under text_config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            n = getattr(text_cfg, "num_hidden_layers", None)
    if not n:
        # last resort: count decoder layers on the (possibly nested) text backbone
        base = getattr(model, "model", model)
        if not hasattr(base, "layers") and hasattr(base, "language_model"):
            base = base.language_model
        n = len(getattr(base, "layers", []))
    return int(n) + 1


def normalize_layers(raw_layers: Iterable[int], model) -> List[int]:
    n_layers = hidden_state_count(model)
    layers = [int(layer) for layer in raw_layers if 0 <= int(layer) < n_layers]
    if not layers:
        raise ValueError(f"No requested layers are valid for n_hidden_state_layers={n_layers}")
    return layers


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def env_metadata() -> dict:
    import torch as _torch
    try:
        import transformers as _transformers
        transformers_version = _transformers.__version__
    except Exception:
        transformers_version = "unknown"
    return {
        "torch_version": _torch.__version__,
        "transformers_version": transformers_version,
        "cuda_available": bool(_torch.cuda.is_available()),
    }


def write_manifest(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    enriched = {
        "created_at_utc": utc_now(),
        **env_metadata(),
        **payload,
    }
    path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_metadata(model_key: str, model=None) -> dict:
    payload = model_spec_dict(model_key)
    if model is not None:
        payload.update(
            {
                "architectures": getattr(model.config, "architectures", None),
                "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
                "hidden_size": getattr(model.config, "hidden_size", None),
                "model_type": getattr(model.config, "model_type", None),
            }
        )
    return payload


def run_root(run_id: str) -> Path:
    return PACKAGE_ROOT / "runs" / run_id


def default_cache_env() -> None:
    # Portable default: honor $ABST_HF_CACHE, else a cache dir inside the repo
    cache_root = Path(os.environ.get("ABST_HF_CACHE") or (PACKAGE_ROOT / "hf_cache"))
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg_cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "torchinductor"))
    os.environ.setdefault("TMPDIR", str(cache_root / "tmp"))
    for key in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


__all__ = [
    "DEFAULT_INPUT_JSONL",
    "PACKAGE_ROOT",
    "default_cache_env",
    "filter_rows",
    "build_user_content",
    "format_prompt",
    "hidden_state_count",
    "keyword_abstained",
    "load_model_and_tokenizer",
    "load_source_rows",
    "maybe_bool",
    "model_metadata",
    "normalize_layers",
    "run_root",
    "set_seed",
    "write_csv",
    "write_jsonl",
    "write_manifest",
]

