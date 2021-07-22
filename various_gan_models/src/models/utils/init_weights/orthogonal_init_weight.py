import argparse

import torch
from torch.nn import init


def modify_init_weight_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--init_gain', type=float, default=0.02, help='Scaling Factor')
    return parser


def init_weight(data: torch.Tensor, opt: argparse.Namespace) -> None:
    init.orthogonal_(data, gain=opt.init_gain)
    return
