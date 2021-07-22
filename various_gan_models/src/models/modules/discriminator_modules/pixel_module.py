import argparse

import torch
import torch.nn as nn

from ..norm_modules import norm_modules
from . import base_module


def create_module(opt: argparse.Namespace) -> nn.Module:
    return PixelModule(opt.input_nch, opt.output_nch, opt.ndf, opt.discriminator_norm_module_name)


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_module.modify_module_commandline_options(parser)
    parser.add_argument('--discriminator_norm_module_name', type=str, required=True, choices=norm_modules.keys())
    return parser


class PixelModule(nn.Module):
    """1x1 PatchGAN discriminator
    入力 : 128 x 128 x (input_nch + output_nch), 256 x 256 x (input_nch + output_nch)
    出力 : 1
    """
    def __init__(self,
                 input_nch: int,
                 output_nch: int,
                 ndf: int,
                 norm_module_name: str) -> None:
        """Construct a 1x1 PatchGAN discriminator
        :param input_nch:
        :param output_nch:
        :param ndf:
        :param norm_module_name:
        """
        super().__init__()

        use_bias = norm_module_name == 'instance_norm'

        model = [
            nn.Conv2d(in_channels=input_nch + output_nch, out_channels=ndf, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=use_bias),
            norm_modules[norm_module_name](ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=ndf * 2, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=use_bias),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
