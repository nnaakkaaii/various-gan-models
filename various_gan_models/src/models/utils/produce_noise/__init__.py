import argparse

import torch


def produce_noise(opt: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.randn(opt.batch_size, opt.latent_dim).to(device)
