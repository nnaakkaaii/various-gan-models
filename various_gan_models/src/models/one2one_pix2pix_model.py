import argparse

import torch

from . import pix2pix_model
from .abstract_model import AbstractModel


def create_model(opt: argparse.Namespace) -> AbstractModel:
    return One2OnePix2PixModel(opt)


def modify_model_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = pix2pix_model.modify_model_commandline_options(parser)
    parser.add_argument('--one2one', action='store_true', help='if white place should be white.')
    return parser


class One2OnePix2PixModel(pix2pix_model.Pix2PixModel):
    """variety of pix2pix supposed by y.nakai.
    pix2pixの派生モデル。使用モジュールはpix2pixと同じ
    """
    def forward(self) -> None:
        if self.opt.one2one:
            fake_b = self._generator_module(self.real_a)
            ones = torch.ones_like(fake_b)
            self.fake_b = torch.where(self.real_a.repeat(1, 3, 1, 1) == 1, ones, fake_b)
        else:
            super().forward()
        return
