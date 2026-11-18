#!/usr/bin/env python3
"""Forward hooks for activation steering in Hugging Face causal LMs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch


TokenPosition = Literal["last", "all"]
ScaleMode = Literal["unit", "raw", "activation_norm", "hidden_norm"]
SteeringMode = Literal["all_steps", "prefill_only"]


def probe_layer_to_module_layer(probe_layer: int) -> int | None:
    """Map outputs.hidden_states index to the transformer module index."""

    if probe_layer < 0:
        raise ValueError(f"probe_layer must be non-negative, got {probe_layer}")
    if probe_layer == 0:
        return None
    return probe_layer - 1


def get_hook_module(model, probe_layer: int):
    """Return the HF module corresponding to a hidden_states layer index."""

    module_layer = probe_layer_to_module_layer(probe_layer)
    base = getattr(model, "model", None)
    if base is None:
        raise AttributeError("Expected model.model on the Hugging Face causal LM.")
    # Multimodal checkpoints (e.g. Qwen3.5) nest the text decoder under
    # model.model.language_model; step into it when the top-level base has no .layers.
    if not hasattr(base, "layers") and hasattr(base, "language_model"):
        base = base.language_model
    if module_layer is None:
        return base.embed_tokens
    if probe_layer == len(base.layers):
        return base.norm
    if probe_layer > len(base.layers):
        raise IndexError(f"probe_layer={probe_layer} is out of range for {len(base.layers)} transformer layers.")
    return base.layers[module_layer]


@dataclass
class SteeringConfig:
    vector: torch.Tensor
    alpha: float
    probe_layer: int
    token_position: TokenPosition = "last"
    scale_mode: ScaleMode = "raw"
    steering_mode: SteeringMode = "all_steps"


class SteeringHook:
    """Registerable hook object that adds alpha * vector, optionally normalized or activation-scaled."""

    def __init__(self, config: SteeringConfig):
        if config.token_position not in ("last", "all"):
            raise NotImplementedError(f"Unsupported token_position={config.token_position!r}")
        if config.scale_mode not in ("unit", "raw", "activation_norm", "hidden_norm"):
            raise NotImplementedError(f"Unsupported scale_mode={config.scale_mode!r}")
        if config.steering_mode not in ("all_steps", "prefill_only"):
            raise NotImplementedError(f"Unsupported steering_mode={config.steering_mode!r}")

        vector = config.vector.detach().float()
        norm = torch.linalg.vector_norm(vector)
        if not torch.isfinite(norm) or float(norm) == 0.0:
            raise ValueError("Steering vector must have finite non-zero norm.")
        self.config = config
        self.vector_raw = vector
        self.vector_unit = vector / norm
        self.calls = 0

    def steer_hidden(self, hidden: torch.Tensor | None):
        if hidden is None:
            return hidden
        is_prefill = hidden.shape[1] > 1
        if self.config.steering_mode == "prefill_only" and not is_prefill:
            return hidden

        if self.config.scale_mode in {"raw", "hidden_norm"}:
            delta = self.config.alpha * self.vector_raw.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        else:
            w = self.vector_unit.to(device=hidden.device, dtype=hidden.dtype)
            if self.config.scale_mode == "activation_norm":
                if self.config.token_position == "last":
                    scale = hidden[:, -1:, :].float().norm(dim=-1, keepdim=True).to(hidden.dtype)
                else:
                    scale = hidden.float().norm(dim=-1, keepdim=True).to(hidden.dtype)
                delta = self.config.alpha * scale * w.view(1, 1, -1)
            else:
                delta = self.config.alpha * w.view(1, 1, -1)

        steered = hidden.clone()
        if self.config.token_position == "last":
            steered[:, -1:, :] = steered[:, -1:, :] + delta
        elif self.config.token_position == "all":
            steered = steered + delta
        else:
            raise ValueError(f"Unsupported token_position={self.config.token_position!r}")

        return steered

    def __call__(self, module, inputs, output):
        self.calls += 1
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        steered = self.steer_hidden(hidden)
        if steered is hidden:
            return output
        if rest is None:
            return steered
        return (steered, *rest)


@contextmanager
def apply_steering_hook(
    model,
    vector,
    *,
    probe_layer: int,
    alpha: float,
    token_position: TokenPosition = "last",
    scale_mode: ScaleMode = "raw",
    steering_mode: SteeringMode = "all_steps",
) -> Iterator[SteeringHook]:
    """Temporarily attach a steering hook to the requested layer."""

    vector_tensor = torch.as_tensor(vector, dtype=torch.float32)
    hook = SteeringHook(
        SteeringConfig(
            vector=vector_tensor,
            alpha=float(alpha),
            probe_layer=int(probe_layer),
            token_position=token_position,
            scale_mode=scale_mode,
            steering_mode=steering_mode,
        )
    )
    module = get_hook_module(model, probe_layer)
    handle = module.register_forward_hook(hook, prepend=True)
    try:
        yield hook
    finally:
        handle.remove()


__all__ = [
    "SteeringConfig",
    "SteeringHook",
    "apply_steering_hook",
    "get_hook_module",
    "probe_layer_to_module_layer",
]
