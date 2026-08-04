#!/usr/bin/env python3
"""Configuration for dense text-only instruction-model abstention experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    family: str
    nominal_size: str
    text_only: bool = True
    dense: bool = True
    role: str = "candidate"


MODEL_SPECS: Dict[str, ModelSpec] = {
    "qwen25_7b_instruct": ModelSpec(
        key="qwen25_7b_instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        family="qwen2.5",
        nominal_size="7B",
        role="robustness",
    ),
    "llama31_8b_instruct": ModelSpec(
        key="llama31_8b_instruct",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        family="llama3.1",
        nominal_size="8B",
        role="primary",
    ),
    "gemma2_9b_it": ModelSpec(
        key="gemma2_9b_it",
        model_id="google/gemma-2-9b-it",
        family="gemma2",
        nominal_size="9B",
        role="robustness",
    ),
    "qwen35_9b": ModelSpec(
        key="qwen35_9b",
        model_id="Qwen/Qwen3.5-9B",
        family="qwen3_5",  # multimodal (text backbone under model.language_model); non-reasoning
        nominal_size="9B",
        role="robustness",
    ),
    "gemma3_12b": ModelSpec(
        key="gemma3_12b",
        model_id="google/gemma-3-12b-it",
        family="gemma3",  # vision+text, non-reasoning; text backbone under model.language_model
        nominal_size="12B",
        role="robustness",
    ),
    "gemma4_12b": ModelSpec(
        key="gemma4_12b",
        model_id="google/gemma-4-12b-it",
        family="gemma4_unified",  # audio+vision+text; text backbone under model.language_model
        nominal_size="12B",
        role="robustness",
    ),
    "mistral7b_instruct_v03": ModelSpec(
        key="mistral7b_instruct_v03",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        family="mistral",
        nominal_size="7B",
        role="robustness",
    ),
}

MODEL_ID_BY_KEY = {key: spec.model_id for key, spec in MODEL_SPECS.items()}
INSTRUCT_MODEL_KEYS = set(MODEL_SPECS)
DEFAULT_MODEL_KEY = "llama31_8b_instruct"

DATASET_NAME_MAP = {
    "squad2dataset": "squad2",
    "squad2": "squad2",
    "umwp": "umwp",
    "falseqa": "falseqa",
    "falseqadataset": "falseqa",
}

VECTOR_TYPES = (
    "U_probe",
    "A_probe",
    "U_mean",
    "A_mean",
    "U_gt",
    "A_gt",
    "U_behavior",
    "A_behavior",
    "U_shuffle_mean",
    "A_shuffle_mean",
)
STEERING_VECTOR_TYPES = (
    "U_gt",
    "U_behavior",
    "A_behavior",
    "A_gt",
    "A_mean",
    "U_mean",
    "A_shuffle_mean",
    "U_shuffle_mean",
    "opposite_A_mean",
    "opposite_U_mean",
    "random",
    "permuted",
    "prompt_only",
)

SYSTEM_PROMPT = "You are a helpful assistant."

RATIONALE_LABELS = (
    "missing_information",
    "clarification_needed",
    "context_mismatch",
    "false_premise_or_misunderstanding",
    "math_or_logic_inconsistency",
    "hypothetical_or_conceptual",
    "generic_abstention",
    "substantive_answer",
    "invalid_generation",
)

ABSTAIN_SUBSTRINGS = (
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not enough information",
    "insufficient information",
    "not provided",
    "not described",
    "not mentioned",
    "does not provide",
    "do not have enough",
    "don't have enough",
    "unknown",
    "unanswerable",
    "no answer",
    "does not mention",
    "doesn't mention",
    "not stated",
    "not specified",
    "cannot be determined",
    "can't be determined",
    "cannot determine",
    "can't determine",
    "not possible to determine",
    "the context does not",
    "the passage does not",
)


def repo_root_from_package(package_root: Path) -> Path:
    """Resolve the repository root from this package."""

    return package_root.resolve().parents[2]


def model_spec_dict(model_key: str) -> dict:
    return asdict(MODEL_SPECS[model_key])

