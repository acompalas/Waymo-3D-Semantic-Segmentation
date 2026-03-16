import torch
import torch.nn as nn


class PointMLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(input_dim)),
            nn.SiLU(),
            nn.Linear(int(input_dim), int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RangeConvHead(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 1) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        self.conv = nn.Conv2d(
            int(input_channels),
            int(output_channels),
            kernel_size=int(kernel_size),
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
