# Copyright (c) 2026 video-rl-pair contributors.
"""Shared DINOv3 image encoder: one normalized CLS embedding per video frame."""

from typing import Self, final

import torch
from torch import Tensor, nn
from torch.nn import functional
from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel
from typing_extensions import override

from video_rl_pair.config import ModelConfig


@final
class VideoEncoder(nn.Module):
    """Raw CLS baseline, head-only adaptation, or head plus backbone-suffix tuning."""

    def __init__(self, config: ModelConfig, *, initialize: bool = True) -> None:
        """Load local pretrained weights or reconstruct from checkpoint architecture."""
        super().__init__()
        if initialize:
            if not config.source.is_dir():
                message = (
                    f"Missing local DINOv3 model directory: {config.source}. "
                    "Download the model first; training never downloads weights."
                )
                raise FileNotFoundError(message)
            backbone = AutoModel.from_pretrained(
                str(config.source), local_files_only=True, use_safetensors=True
            )
            if not isinstance(backbone, DINOv3ViTModel):
                message = "model.source must contain a DINOv3 ViT checkpoint."
                raise ValueError(message)
            self.backbone = backbone
        else:
            if config.backbone_config is None:
                message = "Checkpoint reconstruction requires model.backbone_config."
                raise ValueError(message)
            if config.backbone_config.get("model_type") != "dinov3_vit":
                message = "Checkpoint backbone_config must describe a DINOv3 ViT."
                raise ValueError(message)
            backbone_config = DINOv3ViTConfig.from_dict(config.backbone_config)
            self.backbone = DINOv3ViTModel(backbone_config)

        self.resolved_config = ModelConfig.model_validate(
            {**config.model_dump(), "backbone_config": self.backbone.config.to_dict()}
        )
        self.config = self.resolved_config
        feature_dim = self.backbone.config.hidden_size
        self.head = (
            nn.Identity()
            if config.mode == "baseline"
            else nn.Sequential(
                nn.Linear(feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.embedding_dim),
            )
        )
        _ = self.backbone.requires_grad_(requires_grad=False)
        if config.mode == "stage2":
            if config.unfreeze_last_n > len(self.backbone.layer):
                message = "model.unfreeze_last_n exceeds the backbone's block count."
                raise ValueError(message)
            for block in self.backbone.layer[-config.unfreeze_last_n :]:
                _ = block.requires_grad_(requires_grad=True)
            _ = self.backbone.norm.requires_grad_(requires_grad=True)
        # Embeddings (including the unused mask token) remain frozen in every mode.
        _ = self.train()

    @override
    def train(self, mode: bool = True) -> Self:
        """Keep the frozen prefix and positional augmentation deterministic."""
        _ = super().train(mode)
        _ = self.backbone.eval()
        if mode and self.config.mode == "stage2":
            for block in self.backbone.layer[-self.config.unfreeze_last_n :]:
                _ = block.train()
            _ = self.backbone.norm.train()
        return self

    @override
    def forward(self, clips: Tensor) -> Tensor:
        """Map normalized RGB [B,T,3,H,W] to unit embeddings [B,T,D]."""
        if clips.ndim != 5 or clips.shape[2] != 3 or min(clips.shape) < 1:
            message = "Expected nonempty clips shaped [B,T,3,H,W]."
            raise ValueError(message)
        batch, positions, channels, height, width = clips.shape
        patch_size = self.backbone.config.patch_size
        if height % patch_size or width % patch_size:
            message = "Image dimensions must be multiples of the backbone patch size."
            raise ValueError(message)
        frames = clips.reshape(batch * positions, channels, height, width)
        outputs = self.backbone(pixel_values=frames)
        # Token zero is CLS; register and patch tokens are never pooled into it here.
        cls: Tensor = outputs.last_hidden_state[:, 0, :]
        embeddings: Tensor = self.head(cls)
        return functional.normalize(embeddings.float(), dim=-1).reshape(batch, positions, -1)

    def regularization(self) -> Tensor:
        """Squared L2 of projection weights and biases; no penalty on pretrained weights."""
        terms = [parameter.square().sum() for parameter in self.head.parameters()]
        if not terms:
            return next(self.backbone.parameters()).new_zeros(())
        return torch.stack(terms).sum()
