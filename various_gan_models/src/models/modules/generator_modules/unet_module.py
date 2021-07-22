import argparse
from typing import Optional

import torch
import torch.nn as nn

from ..norm_modules import norm_modules
from . import base_module


def create_module(opt: argparse.Namespace) -> nn.Module:
    return UnetModule(opt.out_size, opt.ngf, opt.output_nch, opt.input_nch, opt.no_dropout, opt.generator_norm_module_name)


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_module.modify_module_commandline_options(parser)
    parser.add_argument('--no_dropout', action='store_true', help='do not use dropout for the generator')
    parser.add_argument('--generator_norm_module_name', type=str, required=True, choices=norm_modules.keys())
    return parser


class UnetModule(nn.Module):
    """Create a Unet-based generator
    入力 : 128 x 128 x input_nch, 256 x 256 x input_nch
    出力 : 128 x 128 x output_nch, 256 x 256 x output_nch
    """
    def __init__(self,
                 out_size: int,
                 ngf: int,
                 output_nch: int,
                 input_nch: int,
                 no_dropout: bool,
                 norm_module_name: str) -> None:
        """Construct a Unet generator
        :param out_size:
        :param ngf:
        :param output_nch:
        :param input_nch:
        :param no_dropout:
        :param norm_module_name:
        """
        super().__init__()

        if out_size == 128:
            num_downs = 7
        elif out_size == 256:
            num_downs = 8
        else:
            raise NotImplementedError

        unet_block = UnetSkipConnectionBlock(
            outer_nch=ngf * 8, inner_nch=ngf * 8, input_nch=ngf * 8,
            norm_module_name=norm_module_name, submodule=None, innermost=True,
        )
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                outer_nch=ngf * 8, inner_nch=ngf * 8, input_nch=ngf * 8,
                norm_module_name=norm_module_name, submodule=unet_block, use_dropout=not no_dropout,
            )
        unet_block = UnetSkipConnectionBlock(
            outer_nch=ngf * 4, inner_nch=ngf * 8, input_nch=ngf * 4,
            norm_module_name=norm_module_name, submodule=unet_block,
        )
        unet_block = UnetSkipConnectionBlock(
            outer_nch=ngf * 2, inner_nch=ngf * 4, input_nch=ngf * 2,
            norm_module_name=norm_module_name, submodule=unet_block,
        )
        unet_block = UnetSkipConnectionBlock(
            outer_nch=ngf, inner_nch=ngf * 2, input_nch=ngf,
            norm_module_name=norm_module_name, submodule=unet_block,
        )

        self.model = UnetSkipConnectionBlock(
            outer_nch=output_nch, inner_nch=ngf, input_nch=input_nch,
            norm_module_name=norm_module_name, submodule=unet_block, outermost=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    """
    def __init__(self,
                 outer_nch: int,
                 inner_nch: int,
                 input_nch: int,
                 norm_module_name: str,
                 submodule: Optional[nn.Module] = None,
                 outermost: bool = False,
                 innermost: bool = False,
                 use_dropout: bool = False) -> None:
        """
        :param outer_nch:
        :param inner_nch:
        :param input_nch:
        :param norm_module_name:
        :param submodule:
        :param outermost:
        :param innermost:
        :param use_dropout:
        """
        super().__init__()

        self.outermost = outermost
        use_bias = norm_module_name == 'instance_norm'

        down_conv = nn.Conv2d(in_channels=input_nch, out_channels=inner_nch, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=use_bias)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_modules[norm_module_name](inner_nch)
        up_relu = nn.ReLU(True)
        up_norm = norm_modules[norm_module_name](outer_nch)

        if outermost and submodule is not None:
            up_conv = nn.ConvTranspose2d(
                in_channels=inner_nch * 2, out_channels=outer_nch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            model = [down_conv, submodule, up_relu, up_conv, nn.Tanh()]
        elif innermost:
            up_conv = nn.ConvTranspose2d(
                in_channels=inner_nch, out_channels=outer_nch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=use_bias)
            model = [down_relu, down_conv, up_relu, up_conv, up_norm]
        elif submodule is not None:
            up_conv = nn.ConvTranspose2d(
                in_channels=inner_nch * 2, out_channels=outer_nch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=use_bias)
            model = [down_relu, down_conv, down_norm, submodule, up_relu, up_conv, up_norm]

            if use_dropout:
                model += [nn.Dropout(0.5)]
        else:
            raise KeyError

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
