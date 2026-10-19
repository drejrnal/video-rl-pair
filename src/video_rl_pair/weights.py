# Copyright (c) 2026 video-rl-pair contributors.
"""Load compatible local state dictionaries without depending on a source repo."""

from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from video_rl_pair.config import WeightConfig

StateDict: TypeAlias = dict[str, Tensor]
CheckpointValue: TypeAlias = (
    Tensor
    | str
    | int
    | float
    | bool
    | list["CheckpointValue"]
    | Mapping[str, "CheckpointValue"]
    | None
)


def read_state_dict(path: Path, entry: str | None) -> StateDict:
    """Select and validate a tensor dictionary inside a local checkpoint."""
    payload: CheckpointValue = (
        load_file(str(path), device="cpu")
        if path.suffix == ".safetensors"
        else torch.load(path, map_location="cpu", weights_only=True)
    )
    if entry is not None:
        for key in entry.split("."):
            if not isinstance(payload, Mapping):
                message = f"Checkpoint key {entry} does not select a nested dictionary."
                raise TypeError(message)
            payload = payload[key]
    else:
        while isinstance(payload, Mapping) and not all(
            isinstance(value, Tensor) for value in payload.values()
        ):
            candidates = [
                key
                for key in ("state_dict", "model_state_dict", "model", "encoder", "backbone")
                if key in payload and isinstance(payload[key], Mapping)
            ]
            if len(candidates) != 1:
                message = "Ambiguous checkpoint layout; set model.ckpt.key to its state dictionary."
                raise ValueError(message)
            payload = payload[candidates[0]]
    if not isinstance(payload, Mapping) or not payload:
        message = "The selected checkpoint entry must be a nonempty tensor state dictionary."
        raise ValueError(message)
    state: StateDict = {}
    for key, value in payload.items():
        if not isinstance(value, Tensor):
            message = f"State dictionary entry {key} is not a tensor."
            raise TypeError(message)
        state[key] = value
    return state


def initialize_weights(module: nn.Module, config: WeightConfig) -> None:
    """Map compatible tensor weights to a model, with strict shape/key checks.

    This initializes parameters only. Optimizer state belongs to training resume.
    """
    if config.path is None:
        return
    state = read_state_dict(config.path, config.key)
    ddp_prefix = "module." if all(key.startswith("module.") for key in state) else ""
    state = {key.removeprefix(ddp_prefix): value for key, value in state.items()}
    prefix = config.prefix
    if (
        prefix is None
        and config.scope == "backbone"
        and any(key.startswith("backbone.") for key in state)
    ):
        prefix = "backbone."
    if prefix is not None:
        state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
    state = {
        key: value for key, value in state.items() if not key.startswith(config.exclude_prefixes)
    }
    try:
        _ = module.load_state_dict(state, strict=True)
    except RuntimeError as error:
        message = (
            f"Checkpoint {config.path} does not match this {config.scope}. "
            "Select the matching backbone and check ckpt.key/prefix/exclude_prefixes. "
            f"Original load error: {error}"
        )
        raise ValueError(message) from error
