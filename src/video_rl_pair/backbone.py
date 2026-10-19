# Copyright (c) 2026 video-rl-pair contributors.
"""ResNet50-v2 through Conv4c, initialized entirely from scratch.

Block ordering follows keras-applications' preactivation ResNet-v2. This is
not torchvision's ResNet50-v1. Keras puts stage downsampling in the LAST
block; the third stage is deliberately cut after its third of six blocks.
"""

from typing import final

from torch import Tensor, nn
from torch.nn import functional
from typing_extensions import override


@final
class PreActBlock(nn.Module):
    """BN-ReLU before convolutions, with a projected or subsampled shortcut."""

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        """Set bottleneck width; the expanded output has four times this width."""
        super().__init__()
        self.preact_bn = nn.BatchNorm2d(in_channels, eps=1.001e-5, momentum=0.01)
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=1.001e-5, momentum=0.01)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1.001e-5, momentum=0.01)
        self.conv3 = nn.Conv2d(channels, 4 * channels, 1)
        self.project = in_channels != 4 * channels
        self.shortcut = (
            nn.Conv2d(in_channels, 4 * channels, 1)
            if self.project
            else nn.MaxPool2d(1, stride=stride)
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Preserve an identity path before preactivation when widths match."""
        preactivated = functional.relu(self.preact_bn(x))
        shortcut = self.shortcut(preactivated if self.project else x)
        residual = self.conv1(preactivated)
        residual = self.conv2(functional.relu(self.bn1(residual)))
        return shortcut + self.conv3(functional.relu(self.bn2(residual)))


@final
class ResNet50V2(nn.Module):
    """At 224 x 224, produce 1024 x 14 x 14 unpooled Conv4c features."""

    def __init__(self) -> None:
        """Retain stages of 3, 4, and 3 blocks, with no classifier or final pooling."""
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2)
        blocks: list[PreActBlock] = []
        in_channels = 64
        for stage, channels, count in ((2, 64, 3), (3, 128, 4), (4, 256, 3)):
            for index in range(1, count + 1):
                stride = 2 if stage < 4 and index == count else 1
                blocks.append(PreActBlock(in_channels, channels, stride))
                in_channels = 4 * channels
        self.blocks = nn.Sequential(*blocks)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _ = nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    _ = nn.init.zeros_(module.bias)

    @override
    def forward(self, images: Tensor) -> Tensor:
        """Encode RGB frames [N,3,H,W], scaled to [-1,1]."""
        stem: Tensor = self.stem(images)
        features: Tensor = self.pool(functional.pad(stem, (1, 1, 1, 1), value=0))
        result: Tensor = self.blocks(features)
        return result
