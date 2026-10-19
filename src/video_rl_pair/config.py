# Copyright (c) 2026 video-rl-pair contributors.
"""Validate Hydra's composed configuration once, at the application boundary."""

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    """Immutable configuration shared by the CLI and the Python training API."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class WeightConfig(Settings):
    """Map a compatible checkpoint to a backbone or complete encoder."""

    path: Path | None = None
    scope: Literal["backbone", "encoder"] = "backbone"
    key: str | None = None
    prefix: str | None = None
    exclude_prefixes: tuple[str, ...] = ()


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
    """Encoder architecture and optional initialization, independent of resume."""

    backbone: Literal["resnet50_v2", "resnet50_v1"] = "resnet50_v2"
    ckpt: WeightConfig = Field(default_factory=WeightConfig)
    train_backbone: Literal["all", "only_bn", "frozen"] = "all"
    conv_channels: int = Field(default=512, gt=0)
    fc_channels: int = Field(default=512, gt=0)
    embedding_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)


class LossConfig(Settings):
    """Temperatures and coefficients in the supplied paper equations."""

    beta: float = Field(default=0.1, gt=0)
    gamma: float = Field(default=0.1, gt=0)
    alpha: float = Field(default=1.0, gt=0)
    cycle_weight: float = Field(default=1.0, ge=0)
    alignment_weight: float = Field(default=0.1, ge=0)


class TrainConfig(Settings):
    """One optimizer step can accumulate multiple local batches on each rank."""

    max_steps: int = Field(default=150_000, gt=0)
    batch_size: int = Field(default=4, ge=2)
    accumulate: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    l2_weight: float = Field(default=1e-5, ge=0)
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    sync_batchnorm: bool = True
    grad_clip: float | None = Field(default=None, gt=0)
    seed: int = Field(default=42, ge=0)
    log_every: int = Field(default=100, gt=0)
    save_every: int = Field(default=30_000, gt=0)
    output_dir: Path = Path("runs/default")
    resume: Path | None = None


class InputConfig(Settings):
    """Expected clip shape; a data pipeline must construct this context."""

    positions: int = Field(default=20, ge=2)
    context_frames: int = Field(default=2, gt=0)
    context_stride: int = Field(default=15, gt=0)
    image_size: int = Field(default=224, ge=16)


class ExperimentConfig(Settings):
    """Typed portion of the Hydra configuration, excluding its batch factory."""

    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    input: InputConfig
