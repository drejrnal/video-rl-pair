# Copyright (c) 2026 video-rl-pair contributors.
"""Validate Hydra's composed configuration once, at the application boundary."""

import json
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class Settings(BaseModel):
    """Immutable configuration shared by the CLI and the Python training API."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class WandbConfig(Settings):
    """Optional experiment tracking; authentication uses the W&B CLI or environment."""

    mode: Literal["disabled", "offline", "online"] = "disabled"
    project: str = Field(default="video-rl-pair", min_length=1)
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    dir: Path = Path("runs/default")
    id: str | None = None
    resume: Literal["allow", "must", "never"] | None = None

    @model_validator(mode="after")
    def validate_resume(self) -> "WandbConfig":
        """Require an explicit online run identity when requesting W&B resume."""
        if self.resume is not None and (not self.id or self.mode != "online"):
            message = "wandb.resume requires wandb.id and wandb.mode=online."
            raise ValueError(message)
        return self


class ModelConfig(Settings):
    """CLS-only DINOv3 architecture and which parameters are trained."""

    source: Path = Path("checkpoints/dinov3-vitb16-pretrain-lvd1689m")
    mode: Literal["baseline", "stage1", "stage2"] = "stage1"
    hidden_dim: int = Field(default=512, gt=0)
    embedding_dim: int = Field(default=128, gt=0)
    unfreeze_last_n: int = Field(default=2, ge=1)
    # Resolved into every training checkpoint, enabling offline reconstruction.
    backbone_config: dict[str, JsonValue] | None = None

    @field_validator("backbone_config", mode="before")
    @classmethod
    def json_architecture(cls, value: object) -> object:
        """Normalize Hugging Face integer label keys before checkpoint round trips."""
        return None if value is None else json.loads(json.dumps(value))


class LossConfig(Settings):
    """Temperatures and coefficients in the supplied paper equations."""

    beta: float = Field(default=0.1, gt=0)
    gamma: float = Field(default=0.1, gt=0)
    alpha: float = Field(default=1.0, gt=0)
    cycle_weight: float = Field(default=1.0, ge=0)
    alignment_weight: float = Field(default=0.1, ge=0)


class TrainConfig(Settings):
    """One optimizer step can accumulate multiple local batches on each rank."""

    max_steps: int = Field(default=6_000, gt=0)
    batch_size: int = Field(default=2, ge=2)
    accumulate: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    backbone_learning_rate: float = Field(default=1e-5, gt=0)
    l2_weight: float = Field(default=1e-5, ge=0)
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    grad_clip: float | None = Field(default=1.0, gt=0)
    seed: int = Field(default=42, ge=0)
    log_every: int = Field(default=100, gt=0)
    save_every: int = Field(default=1_200, gt=0)
    output_dir: Path = Path("runs/default")
    resume: Path | None = None
    init_checkpoint: Path | None = None

    @model_validator(mode="after")
    def validate_initialization(self) -> "TrainConfig":
        """Distinguish a weights-only stage transition from a complete resume."""
        if self.resume is not None and self.init_checkpoint is not None:
            message = "Use either train.resume or train.init_checkpoint, not both."
            raise ValueError(message)
        return self


class InputConfig(Settings):
    """Number of sampled frames and square image size; no temporal context head."""

    positions: int = Field(default=32, ge=2)
    image_size: int = Field(default=224, ge=16, multiple_of=16)


class ExperimentConfig(Settings):
    """Typed portion of the Hydra configuration, excluding its batch factory."""

    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    input: InputConfig
