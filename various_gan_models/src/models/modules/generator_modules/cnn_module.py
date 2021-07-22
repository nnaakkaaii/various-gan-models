import argparse
from typing import List

import torch
import torch.nn as nn

from ..norm_modules import norm_modules
from . import base_module


def create_module(opt: argparse.Namespace) -> nn.Module:
    return CNNModule(opt.out_size, opt.generator_n_blocks, opt.ngf, opt.output_nch, opt.latent_dim, opt.generator_norm_module_name)


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_module.modify_module_commandline_options(parser)
    parser.add_argument('--generator_n_blocks', type=int, default=2, help='GeneratorのCNNブロック数, out_size/2**n_blocksが整数になる必要がある, 1以上')
    parser.add_argument('--latent_dim', type=int, default=64, help='Generatorに入力するノイズの次元数')
    parser.add_argument('--generator_norm_module_name', type=str, required=True, choices=norm_modules.keys())
    return parser


class CNNModule(nn.Module):
    """CNN generator
    入力 : opt.latent_dim
    出力 : opt.output_nch, opt.out_size, opt.out_size
    """
    def __init__(self,
                 out_size: int,
                 n_blocks: int,
                 ngf: int,
                 output_nch: int,
                 latent_dim: int,
                 norm_module_name: str) -> None:
        """Construct a PatchGAN discriminator
        :param out_size: 出力画像サイズ
        :param n_blocks: ネットワークの階数
        :param ngf: 最終層手前のニューラル数
        :param output_nch: 出力画像のチャンネル
        :param latent_dim: 入力するノイズの次元
        :param norm_module_name: 利用するBN層の名前
        """
        assert out_size % 2**n_blocks == 0 and n_blocks > 0
        super().__init__()
        self.n_blocks = n_blocks
        self.ngf = ngf

        self.init_size = int(out_size / 2**n_blocks)
        mult = 2**(n_blocks - 1) * ngf
        self.linear = nn.Linear(latent_dim, mult * self.init_size * self.init_size)

        model: List[nn.Module] = [norm_modules[norm_module_name](mult)]
        for i in range(n_blocks - 1):
            model += self.__block(mult, int(mult / 2), norm_module_name)
            mult = int(mult / 2)

        assert mult == ngf
        model += [
            nn.ConvTranspose2d(ngf, output_nch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    @staticmethod
    def __block(input_nch: int, output_nch: int, norm_module_name: str) -> List[nn.Module]:
        return [
            nn.ConvTranspose2d(input_nch, output_nch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            norm_modules[norm_module_name](output_nch),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = x.view(x.size(0), 2**(self.n_blocks - 1) * self.ngf, self.init_size, self.init_size)
        x = self.model(x)
        return x
