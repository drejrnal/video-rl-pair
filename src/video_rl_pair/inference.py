# Copyright (c) 2026 video-rl-pair contributors.
"""Restore a frozen encoder and extract bounded chunks of per-moment embeddings."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, final

import torch
from torch import Tensor

from video_rl_pair.config import ExperimentConfig, WeightConfig
from video_rl_pair.data import PairDataConfig
from video_rl_pair.evaluation_types import EvaluationConfig
from video_rl_pair.model import VideoEncoder
from video_rl_pair.provenance import DataProvenance, check_held_out
from video_rl_pair.video import sample_video
from video_rl_pair.visual_report import thumbnails

if TYPE_CHECKING:
    from video_rl_pair.checkpoint import Checkpoint


@dataclass(frozen=True, slots=True)
class EncodedVideo:
    """Small CPU representation retained after decoded clip tensors are released."""

    path: Path
    embeddings: Tensor
    times: Tensor
    images: tuple[str, ...]


@final
class FrozenEncoder:
    """An inference-only model; checkpoint architecture wins over CLI architecture defaults."""

    def __init__(
        self, settings: ExperimentConfig, data: PairDataConfig, options: EvaluationConfig
    ) -> None:
        """Load training checkpoints or explicitly requested initialization baselines."""
        self.provenance_checked = False
        self.options = options
        self.checkpoint = options.checkpoint
        device = settings.train.device
        self.device = torch.device(
            "cuda"
            if device == "cuda" or (device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        if options.checkpoint is not None:
            checkpoint: Checkpoint = torch.load(
                options.checkpoint, map_location="cpu", weights_only=True
            )
            if checkpoint["version"] != 1:
                message = "Evaluation expects a video-rl-pair training checkpoint (version 1)."
                raise ValueError(message)
            settings = ExperimentConfig.model_validate_json(checkpoint["config"])
            provenance = checkpoint.get("data_provenance")
            if provenance is not None:
                check_held_out(DataProvenance.model_validate_json(provenance), data)
                self.provenance_checked = True
            model_settings = settings.model.model_copy(update={"ckpt": WeightConfig()})
            self.model = VideoEncoder(model_settings, initialize=False)
            _ = self.model.load_state_dict(checkpoint["model"], strict=True)
        else:
            if not options.allow_initialization:
                message = "Set evaluation.checkpoint, or allow_initialization=true for a baseline."
                raise ValueError(message)
            _ = torch.manual_seed(options.seed)
            self.model = VideoEncoder(settings.model)
        self.settings = settings.model_copy(
            update={"input": settings.input.model_copy(update={"positions": options.positions})}
        )
        _ = self.model.to(self.device).eval().requires_grad_(requires_grad=False)

    def encode(self, path: Path) -> EncodedVideo:
        """Uniformly sample a video; retain past context and disable gradients/BN updates."""
        sampled = sample_video(path, self.settings)
        output: list[Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(sampled.clips), self.options.chunk_size):
                clips = sampled.clips[start : start + self.options.chunk_size]
                result: Tensor = self.model(clips.unsqueeze(0).to(self.device))
                output.append(result[0].cpu())
        return EncodedVideo(
            path=path,
            embeddings=torch.cat(output),
            times=sampled.times,
            images=thumbnails(sampled.clips, self.settings.model.backbone, max_side=256),
        )
