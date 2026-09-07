import math

import torch
import torch.nn as nn


class RoCLU(nn.Module):
    """Robust Cauchy Linear Unit (RoCLU).

    RoCLU is defined as

        f(x) = k * x * ln(1.5 + atan(x) / pi)

    with k = 1.444, as reported in the paper.

    The logarithm argument remains in (1, 2) for finite real-valued x,
    because atan(x) / pi is in (-0.5, 0.5).

    Args:
        k (float): Scaling factor. Default: 1.444.
    """

    def __init__(self, k: float = 1.444):
        super().__init__()
        self.k = float(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.log(1.5 + torch.atan(x) / math.pi)
        return self.k * x * gate

    def extra_repr(self) -> str:
        return f"k={self.k}"
