import argparse
from typing import Any

import torch.optim as optim


def modify_discriminator_optimizer_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--discriminator_lr', type=float, default=0.00001, help='Discriminatorのlearning rate')
    parser.add_argument('--discriminator_beta1', type=float, default=0.5, help='Discriminatorのbeta1')
    parser.add_argument('--discriminator_beta2', type=float, default=0.999, help='Discriminatorのbeta2')
    return parser


def modify_generator_optimizer_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--generator_lr', type=float, default=0.001, help='Generatorのlearning rate')
    parser.add_argument('--generator_beta1', type=float, default=0.5, help='Generatorのbeta1')
    parser.add_argument('--generator_beta2', type=float, default=0.999, help='Generatorのbeta2')
    return parser


def create_discriminator_optimizer(params: Any, opt: argparse.Namespace) -> Any:
    return optim.Adam(params, lr=opt.discriminator_lr, betas=(opt.discriminator_beta1, opt.discriminator_beta2))


def create_generator_optimizer(params: Any, opt: argparse.Namespace) -> Any:
    return optim.Adam(params, lr=opt.generator_lr, betas=(opt.generator_beta1, opt.generator_beta2))
