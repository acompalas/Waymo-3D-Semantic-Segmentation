import torch
import torch.nn as nn


class PointHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(input_dim), int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RangeHead(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            int(input_channels),
            int(output_channels),
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
