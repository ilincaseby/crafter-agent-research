"""IMPALA-style residual encoder inspired by arXiv:2307.03486's released config.

Independent implementation: three downsampling stages (64/128/128), two
residual blocks per stage, GroupNorm for images and LayerNorm for dense inputs.
No pretrained weights, BatchNorm, or value-target normalization.
"""

import torch
from torch import nn


def fan_in_layer(layer, scale=1.0):
    """Give each output filter a fixed norm, following the paper's encoder init."""
    with torch.no_grad():
        weight = layer.weight.flatten(1)
        layer.weight.mul_(
            (scale / weight.norm(dim=1)).reshape((-1,) + (1,) * (layer.weight.ndim - 1))
        )
        if layer.bias is not None:
            layer.bias.zero_()
    return layer


def conv_block(in_channels, out_channels, *, normalize=True, scale=1.0):
    layers = [nn.GroupNorm(1, in_channels)] if normalize else []
    layers.extend(
        (
            fan_in_layer(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not normalize),
                scale,
            ),
            nn.ReLU(),
        )
    )
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Two blocks per stage and three stages share the residual init budget.
        scale = 6**-0.25
        self.branch = nn.Sequential(
            conv_block(channels, channels, scale=scale),
            conv_block(channels, channels, scale=scale),
        )

    def forward(self, inputs):
        return inputs + self.branch(inputs)


class ImpalaEncoder(nn.Module):
    output_size = 1024

    def __init__(self, in_channels=3):
        super().__init__()
        stages = []
        for index, channels in enumerate((64, 128, 128)):
            stages.append(
                nn.Sequential(
                    conv_block(in_channels, channels, normalize=index != 0),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.GroupNorm(1, channels),
                    ResidualBlock(channels),
                    ResidualBlock(channels),
                )
            )
            in_channels = channels
        self.stages = nn.Sequential(*stages)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(128 * 8 * 8),
            fan_in_layer(nn.Linear(128 * 8 * 8, 256, bias=False), scale=1.4),
            nn.ReLU(),
            nn.LayerNorm(256),
            fan_in_layer(nn.Linear(256, self.output_size, bias=False)),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.projection(self.stages(inputs))
