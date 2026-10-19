# Copyright (c) 2026 video-rl-pair contributors.
"""Frames -> Conv4c -> temporal/spatial head -> unit-length embeddings."""

from collections import OrderedDict
from typing import Self, assert_never, final

import torch
from torch import Tensor, nn
from torch.nn import functional
from torchvision.models import resnet50
from typing_extensions import override

from video_rl_pair.backbone import ResNet50V2
from video_rl_pair.config import ModelConfig
from video_rl_pair.weights import initialize_weights


@final
class VideoEncoder(nn.Module):
    """Encode B videos at T moments using K context frames per moment."""

    def __init__(self, config: ModelConfig, *, initialize: bool = True) -> None:
        """Build the encoder, optionally initialize weights, and select trainable layers."""
        super().__init__()
        self.config = config
        match config.backbone:
            case "resnet50_v2":
                self.backbone = ResNet50V2()
            case "resnet50_v1":
                full = resnet50(weights=None)
                self.backbone = nn.Sequential(
                    OrderedDict(
                        (name, layer)
                        for name, layer in full.named_children()
                        if name not in {"layer4", "avgpool", "fc"}
                    )
                )
            case unreachable:
                assert_never(unreachable)

        channels, hidden = config.conv_channels, config.fc_channels
        self.head = nn.Sequential(
            nn.Conv3d(1024, channels, 3, padding=1),
            nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config.embedding_dim),
        )
        for module in self.head.modules():
            if isinstance(module, nn.Conv3d):
                _ = nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    _ = nn.init.zeros_(module.bias)
            if isinstance(module, nn.Linear):
                _ = nn.init.xavier_uniform_(module.weight)
                _ = nn.init.zeros_(module.bias)

        if initialize:
            target = self.backbone if config.ckpt.scope == "backbone" else self
            initialize_weights(target, config.ckpt)
        _ = self.backbone.requires_grad_(config.train_backbone == "all")
        for layer in self.backbone.modules():
            if isinstance(layer, nn.BatchNorm2d):
                _ = layer.requires_grad_(config.train_backbone != "frozen")

    @override
    def train(self, mode: bool = True) -> Self:
        """Keep frozen backbone running statistics fixed as well as its weights."""
        _ = super().train(mode)
        if self.config.train_backbone == "frozen":
            _ = self.backbone.eval()
        return self

    @override
    def forward(self, clips: Tensor) -> Tensor:
        """Map [B,T,K,3,H,W] -> [B,T,D]; pool over context, never over T.

        Context must end at the represented moment (no future frames).
        For K=2 and stride=15, use [max(t-15,0), t].
        """
        if clips.ndim != 6 or clips.shape[3] != 3:
            message = "Expected clips shaped [B,T,K,3,H,W]."
            raise ValueError(message)
        batch, positions, context, channels, height, width = clips.shape
        frames = clips.reshape(batch * positions * context, channels, height, width)
        features: Tensor = self.backbone(frames)
        _, channels, height, width = features.shape
        contexts = features.reshape(batch * positions, context, channels, height, width)
        embedding: Tensor = self.head(contexts.transpose(1, 2))
        return functional.normalize(embedding.float(), dim=-1).reshape(batch, positions, -1)

    def regularization(self) -> Tensor:
        """Apply squared L2 to head kernels and biases, excluding BN."""
        terms = [
            parameter.square().sum()
            for module in self.head
            if isinstance(module, (nn.Conv3d, nn.Linear))
            for parameter in module.parameters()
        ]
        return torch.stack(terms).sum()
