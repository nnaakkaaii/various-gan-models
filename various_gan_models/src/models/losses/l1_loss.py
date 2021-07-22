import argparse

import torch
import torch.nn as nn


def create_loss(opt: argparse.Namespace) -> nn.Module:
    return L1Loss(opt.discriminator_lambda_l1)


def modify_loss_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--lambda_l1', type=float, default=100.0, help='weight for L1 loss')
    return parser


class L1Loss(nn.L1Loss):
    def __init__(self, lambda_l1: float) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().__call__(prediction, target) * self.lambda_l1
