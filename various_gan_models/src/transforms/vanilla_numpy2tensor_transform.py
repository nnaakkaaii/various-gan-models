import argparse
from typing import Any, Tuple

import numpy as np
import torch


def create_transform(opt: argparse.Namespace) -> Any:
    return vanilla_numpy2tensor_transform


def modify_transform_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


def vanilla_numpy2tensor_transform(a_img: np.ndarray, b_img: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(a_img).unsqueeze(0).float(),
        torch.from_numpy(b_img).unsqueeze(1).float(),
    )
