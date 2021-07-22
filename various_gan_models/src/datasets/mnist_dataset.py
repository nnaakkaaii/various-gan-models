import argparse
import os
from typing import Any, Dict

import torch
from torchvision.datasets import MNIST

from . import base_dataset


def create_dataset(transform: Any, opt: argparse.Namespace) -> base_dataset.BaseDataset:
    return MnistDataset(transform, opt.max_dataset_size, opt.img_dir)


def modify_dataset_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--img_dir', type=str, default=os.path.join('../dataloaders', 'inputs', 'mnist'), help='mnistデータを保存する場所')
    return parser


class MnistDataset(base_dataset.BaseDataset):
    def __init__(self, transform: Any, max_dataset_size: int, img_dir: str) -> None:
        self.dataset = MNIST(img_dir, download=True, transform=transform)
        super().__init__(max_dataset_size, len(self.dataset))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_, label = self.dataset[idx]
        return {'data': data_, 'label': label}
