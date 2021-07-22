import argparse
from typing import List

import torch
import torch.nn as nn

from ..norm_modules import norm_modules
from ..padding_modules import padding_modules
from . import base_module


def create_module(opt: argparse.Namespace) -> nn.Module:
    return ResnetModule(
        opt.out_size, opt.ngf, opt.output_nch, opt.input_nch, opt.no_dropout,
        opt.generator_norm_module_name, opt.generator_padding_module_name,
    )


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_module.modify_module_commandline_options(parser)
    parser.add_argument('--no_dropout', action='store_true', help='do not use dropout for the generator')
    parser.add_argument('--generator_norm_module_name', type=str, required=True, choices=norm_modules.keys())
    parser.add_argument('--generator_padding_module_name', type=str, required=True, choices=padding_modules.keys())
    return parser


class ResnetModule(nn.Module):
    """Resnet-based generator that consistsof Resnet blocks between a few down-sampling/up-sampling operations.
    入力 : 128 x 128 x input_nch, 256 x 256 x input_nch
    出力 : 128 x 128 x output_nch, 256 x 256 x output_nch
    """
    def __init__(self,
                 out_size: int,
                 ngf: int,
                 output_nch: int,
                 input_nch: int,
                 no_dropout: bool,
                 norm_module_name: str,
                 padding_module_name: str) -> None:
        """Construct a Resnet-based generator
        :param out_size:
        :param ngf:
        :param input_nch:
        :param no_dropout:
        :param norm_module_name:
        :param padding_module_name:
        """
        super().__init__()
        if out_size == 128:
            n_blocks = 6
        elif out_size == 256:
            n_blocks = 9
        else:
            raise NotImplementedError

        use_bias = norm_module_name == 'instance_norm'

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=input_nch, out_channels=ngf, kernel_size=(7, 7), padding=0, bias=use_bias),
            norm_modules[norm_module_name](ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(in_channels=ngf * mult, out_channels=ngf * mult * 2, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=use_bias),
                norm_modules[norm_module_name](ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    dim=ngf * mult, use_dropout=not no_dropout, use_bias=use_bias,
                    norm_module_name=norm_module_name, padding_module_name=padding_module_name
                ),
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    in_channels=ngf * mult, out_channels=int(ngf * mult / 2),
                    kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=use_bias
                ),
                norm_modules[norm_module_name](int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=output_nch, kernel_size=(7, 7), padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block
    """
    def __init__(self, dim: int, use_dropout: bool, use_bias: bool, norm_module_name: str, padding_module_name: str) -> None:
        """Initialize the Resnet block
        :param dim:
        :param use_dropout:
        :param use_bias:
        :param norm_module_name:
        :param padding_module_name:
        """
        super().__init__()

        model: List[nn.Module] = []

        padding_layer, p = padding_modules[padding_module_name]()

        model += [padding_layer]
        model += [
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=p, bias=use_bias),
            norm_modules[norm_module_name](dim),
            nn.ReLU(True),
        ]

        if use_dropout:
            model += [nn.Dropout(0.5)]

        model += [padding_layer]

        model += [
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=p, bias=use_bias),
            norm_modules[norm_module_name](dim),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function with skip connections
        :param x:
        :return:
        """
        out = x + self.model(x)
        return out
