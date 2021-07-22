import argparse

import torch.nn as nn

from ._gan_loss import GANLoss


def create_loss(opt: argparse.Namespace) -> nn.Module:
    return GANLoss(gan_mode='vanilla')
