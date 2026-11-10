# Copyright (c) 2026 video-rl-pair contributors.
"""Frozen CLS inference from a trained checkpoint or the raw DINOv3 baseline."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, final

import torch
from PIL import Image
from torch import Tensor

from video_rl_pair.config import ExperimentConfig
from video_rl_pair.data import PairDataConfig
from video_rl_pair.evaluation_types import EvaluationConfig
from video_rl_pair.model import VideoEncoder
from video_rl_pair.provenance import DataProvenance, check_held_out
from video_rl_pair.video import sample_video

if TYPE_CHECKING:
    from video_rl_pair.checkpoint import Checkpoint


@dataclass(frozen=True, slots=True)
class EncodedVideo:
    """Per-frame embeddings, original timestamps and unnormalized RGB images."""

    path: Path
    embeddings: Tensor
    times: Tensor
    images: tuple[Image.Image, ...]


@final
class FrozenEncoder:
    """Restore architecture and weights without downloading the initial backbone."""

    def __init__(
        self,
        settings: ExperimentConfig,
        data: PairDataConfig | None,
        options: EvaluationConfig,
    ) -> None:
        """Keep evaluation settings separate from checkpoint architecture settings."""
        self.provenance_check: Literal["not_available", "files", "tasks_and_files"] = (
            "not_available"
        )
        self.options = options
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
            if checkpoint["version"] != 2:
                message = "Evaluation expects a DINOv3 training checkpoint (version 2)."
                raise ValueError(message)
            settings = ExperimentConfig.model_validate_json(checkpoint["config"])
            provenance_json = checkpoint.get("data_provenance")
            if provenance_json is not None:
                provenance = DataProvenance.model_validate_json(provenance_json)
                if options.source is not None and options.target is not None:
                    trained = {str(Path(video.path).resolve()) for video in provenance.videos}
                    selected = {str(options.source.resolve()), str(options.target.resolve())}
                    if trained & selected:
                        message = "Evaluation overlaps the checkpoint's recorded training videos."
                        raise ValueError(message)
                    self.provenance_check = "files"
                elif data is not None:
                    check_held_out(provenance, data)
                    self.provenance_check = "tasks_and_files"
            self.model = VideoEncoder(settings.model, initialize=False)
            _ = self.model.load_state_dict(checkpoint["model"], strict=True)
        else:
            if settings.model.mode != "baseline":
                message = "Without evaluation.checkpoint, set model.mode=baseline for raw CLS."
                raise ValueError(message)
            _ = torch.manual_seed(options.seed)
            self.model = VideoEncoder(settings.model)
            settings = settings.model_copy(update={"model": self.model.resolved_config})
        self.settings = settings.model_copy(
            update={"input": settings.input.model_copy(update={"positions": options.positions})}
        )
        _ = self.model.to(self.device).eval().requires_grad_(requires_grad=False)

    @property
    def provenance_checked(self) -> bool:
        """Whether recorded training inputs were checked against the requested evaluation."""
        return self.provenance_check != "not_available"

    def encode(self, path: Path) -> EncodedVideo:
        """Encode each unique sampled frame, without gradients or repeated padding frames."""
        sampled = sample_video(path, self.settings)
        output: list[Tensor] = []
        with torch.inference_mode():
            for start in range(0, sampled.valid_length, self.options.chunk_size):
                clips = sampled.clips[
                    start : min(start + self.options.chunk_size, sampled.valid_length)
                ]
                result: Tensor = self.model(clips.unsqueeze(0).to(self.device))
                output.append(result[0].cpu())
        return EncodedVideo(
            path=path,
            embeddings=torch.cat(output),
            times=sampled.times[: sampled.valid_length],
            images=sampled.images[: sampled.valid_length],
        )
