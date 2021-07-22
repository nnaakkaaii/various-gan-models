import argparse
from typing import Any

from torch.optim import lr_scheduler


def modify_discriminator_scheduler_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--discriminator_lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    return parser


def modify_generator_scheduler_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--generator_lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    return parser


def create_discriminator_optimizer(optimizer: Any, opt: argparse.Namespace) -> Any:
    return lr_scheduler.StepLR(optimizer, step_size=opt.discriminator_lr_decay_iters, gamma=0.1)


def create_generator_optimizer(optimizer: Any, opt: argparse.Namespace) -> Any:
    return lr_scheduler.StepLR(optimizer, step_size=opt.generator_lr_decay_iters, gamma=0.1)
