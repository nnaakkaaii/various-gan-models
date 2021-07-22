import argparse
from typing import List

import torch
import torch.nn as nn

from ..norm_modules import norm_modules
from . import base_module


def create_module(opt: argparse.Namespace) -> nn.Module:
    return VanillaFCModule(opt.out_size, opt.generator_n_layers, opt.output_nch, opt.latent_dim, opt.generator_norm_module_name)


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_module.modify_module_commandline_options(parser)
    parser.add_argument('--latent_dim', type=int, default=64, help='Generatorに入力するノイズの次元数')
    parser.add_argument('--generator_norm_module_name', type=str, required=True, choices=norm_modules.keys())
    parser.add_argument('--generator_n_layers', type=str, default='128,256,512,1024', help='Generatorの各層のパーセプトロン数')
    return parser


class VanillaFCModule(nn.Module):
    """Full Connected Generator
    入力 : opt.latent_dim
    出力 : opt.output_nch, opt.out_size, opt.out_size
    """
    def __init__(self,
                 out_size: int,
                 n_layers: str,
                 output_nch: int,
                 latent_dim: int,
                 norm_module_name: str) -> None:
        """
        :param out_size:
        :param n_layers:
        :param output_nch:
        :param latent_dim:
        :param norm_module_name:
        """
        super().__init__()
        self.out_size = out_size
        self.output_nch = output_nch

        _n_layers = list(map(int, n_layers.split(',')))

        model = []
        for i, n_layer in enumerate(_n_layers):
            if i == 0:
                model += self.__block(latent_dim, _n_layers[0], norm_module_name=norm_module_name, normalize=False)
            else:
                model += self.__block(_n_layers[i - 1], n_layer, norm_module_name=norm_module_name)
        model += [
            nn.Linear(_n_layers[-1], out_size * out_size),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    @staticmethod
    def __block(in_size: int, out_size: int, norm_module_name: str, normalize: bool = True) -> List[nn.Module]:
        if normalize:
            return [
                nn.Linear(in_size, out_size),
                norm_modules[norm_module_name](out_size),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            return [
                nn.Linear(in_size, out_size),
                nn.LeakyReLU(0.2, inplace=True),
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = x.view(x.size(0), self.output_nch, self.out_size, self.out_size)
        return x
