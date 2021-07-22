import argparse
from typing import List

import torch
import torch.nn as nn

from . import base_module


def create_module(opt: argparse.Namespace) -> nn.Module:
    return VanillaFCModule(opt.out_size, opt.output_nch, opt.discriminator_n_layers)


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_module.modify_module_commandline_options(parser)
    parser.add_argument('--discriminator_n_layers', type=str, default='512,256', help='Discriminatorの各層のパーセプトロン数')
    return parser


class VanillaFCModule(nn.Module):
    """Full Connected Discriminator
    入力 : opt.out_size, opt.out_size, opt.output_nch
    出力 : 1
    """
    def __init__(self,
                 out_size: int,
                 output_nch: int,
                 n_layers: str) -> None:
        """
        :param out_size:
        :param output_nch:
        :param n_layers:
        """
        super().__init__()
        _n_layers = list(map(int, n_layers.split(',')))

        model = []
        for i, n_layer in enumerate(_n_layers):
            if i == 0:
                model += self.__block(out_size * out_size * output_nch, n_layer)
            else:
                model += self.__block(_n_layers[i - 1], n_layer)
        model += [nn.Linear(_n_layers[-1], 1)]

        self.model = nn.Sequential(*model)

    @staticmethod
    def __block(in_size: int, out_size: int) -> List[nn.Module]:
        return [
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.model(x)
