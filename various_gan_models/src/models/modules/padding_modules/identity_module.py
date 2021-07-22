from typing import Tuple

import torch
import torch.nn as nn


def create_norm_module() -> Tuple[nn.Module, int]:
    return IdentityModule(), 0


class IdentityModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
