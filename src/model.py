"""The CNN.

Sized deliberately: at 32x32 with one conv per block this trains in ~6 minutes
on a 12-core CPU. Measured alternatives on the same machine -- 48x48 input took
13.5 min and two convs per block took 14.5 min, for accuracy gains that did not
justify making an evaluator wait.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import NUM_CLASSES


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool, halving the spatial size."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # bias=False: BatchNorm immediately re-centres the output, so a conv bias
        # is a redundant parameter with no effect on the function.
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class TrafficSignNet(nn.Module):
    """Three conv blocks and a global-average-pooled linear head.

    The GAP head (rather than flatten -> Linear) is doing three jobs: it drops
    ~88k parameters, it makes the network accept any input size without code
    changes, and GAP followed by a single Linear is the canonical arrangement
    that makes class activation maps meaningful.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, channels=(32, 64, 128),
                 dropout: float = 0.3, width_mult: float = 1.0, in_ch: int = 3):
        super().__init__()
        chans = [max(8, int(round(c * width_mult))) for c in channels]

        blocks, prev = [], in_ch
        for c in chans:
            blocks.append(ConvBlock(prev, c))
            prev = c
        self.features = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(prev, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(self.pool(x))
        return self.classifier(self.dropout(x))

    def get_last_conv_layer(self) -> nn.Module:
        """Grad-CAM's hook target.

        Exposed as a method so gradcam.py does not have to reach into the module
        tree by name and silently break if the architecture is edited.
        """
        return self.features[-1].conv


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def build_model(cfg) -> TrafficSignNet:
    return TrafficSignNet(
        num_classes=NUM_CLASSES,
        channels=tuple(cfg.channels),
        dropout=cfg.dropout,
        width_mult=cfg.width_mult,
    )
