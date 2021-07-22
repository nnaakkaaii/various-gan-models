import argparse
from typing import List

import torch
import torch.nn as nn

from ..norm_modules import norm_modules
from . import base_module


def create_module(opt: argparse.Namespace) -> nn.Module:
    return CNNModule(opt.out_size, opt.discriminator_n_blocks, opt.ndf, opt.output_nch, opt.discriminator_norm_module_name)


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_module.modify_module_commandline_options(parser)
    parser.add_argument('--discriminator_n_blocks', type=int, default=2, help='Discriminatorのブロック数')
    parser.add_argument('--discriminator_norm_module_name', type=str, required=True, choices=norm_modules.keys())
    return parser


class CNNModule(nn.Module):
    """CNN discriminator
    """
    def __init__(self,
                 out_size: int,
                 n_blocks: int,
                 ndf: int,
                 output_nch: int,
                 norm_module_name: str) -> None:
        """Construct a PatchGAN discriminator
        :param out_size: 出力画像サイズ
        :param output_nch: 出力画像のチャンネル
        :param n_blocks: ネットワークの階数
        :param ndf: 1層目のニューラル数
        :param norm_module_name: 利用するBN層の名前
        """
        assert out_size % 2**n_blocks == 0 and n_blocks > 0
        super().__init__()
        mult = ndf
        end_size = int(out_size / 2**n_blocks)

        model: List[nn.Module] = []
        for i in range(n_blocks):
            if i == 0:
                model += self.__block(input_nch=output_nch, output_nch=ndf, norm_module_name=norm_module_name, normalize=False)
            else:
                model += self.__block(input_nch=mult, output_nch=2 * mult, norm_module_name=norm_module_name)
                mult *= 2
        self.model = nn.Sequential(*model)
        self.linear = nn.Linear(mult * end_size * end_size, 1)

    @staticmethod
    def __block(input_nch: int, output_nch: int, norm_module_name: str, normalize: bool = True) -> List[nn.Module]:
        if normalize:
            return [
                nn.Conv2d(input_nch, output_nch, kernel_size=(4, 4), stride=(2, 2), padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.5),
                norm_modules[norm_module_name](output_nch),
            ]
        else:
            return [
                nn.Conv2d(input_nch, output_nch, kernel_size=(4, 4), stride=(2, 2), padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.5),
            ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
