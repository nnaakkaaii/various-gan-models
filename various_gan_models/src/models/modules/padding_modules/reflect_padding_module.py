from typing import Tuple

import torch.nn as nn


def create_padding_module() -> Tuple[nn.Module, int]:
    p = 1
    return nn.ReflectionPad2d(p), p
