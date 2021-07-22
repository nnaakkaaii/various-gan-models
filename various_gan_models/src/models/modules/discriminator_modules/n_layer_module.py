import argparse

import torch
import torch.nn as nn

from ..norm_modules import norm_modules
from . import base_module


def create_module(opt: argparse.Namespace) -> nn.Module:
    return NLayerModule(opt.input_nch, opt.output_nch, opt.discriminator_n_layers, opt.ndf, opt.discriminator_norm_module_name)


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_module.modify_module_commandline_options(parser)
    parser.add_argument('--discriminator_n_layers', type=int, default=3, help='Discriminatorの層数')
    parser.add_argument('--discriminator_norm_module_name', type=str, required=True, choices=norm_modules.keys())
    return parser


class NLayerModule(nn.Module):
    """PatchGAN discriminator
    """
    def __init__(self,
                 input_nch: int,
                 output_nch: int,
                 n_layers: int,
                 ndf: int,
                 norm_module_name: str) -> None:
        """PatchGAN discriminator
        :param input_nch: 入力画像のチャンネル数
        :param output_nch: 出力画像のチャンネル数
        :param n_layers: ネットワークの層数
        :param ndf: 1層目のニューラル数
        :param norm_module_name: 利用するBN層の名前
        """
        super().__init__()

        use_bias = norm_module_name == 'instance_norm'

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels=input_nch + output_nch, out_channels=ndf, kernel_size=(kw, kw), stride=(2, 2), padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(in_channels=ndf * nf_mult_prev, out_channels=ndf * nf_mult, kernel_size=(kw, kw), stride=(2, 2), padding=padw, bias=use_bias),
                norm_modules[norm_module_name](ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(in_channels=ndf * nf_mult_prev, out_channels=ndf * nf_mult, kernel_size=(kw, kw), stride=(1, 1), padding=padw, bias=use_bias),
            norm_modules[norm_module_name](ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(in_channels=ndf * nf_mult, out_channels=1, kernel_size=(kw, kw), stride=(1, 1), padding=padw),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
