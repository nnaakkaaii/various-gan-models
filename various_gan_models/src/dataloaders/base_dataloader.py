import argparse

from .datasets import dataset_options, datasets
from .transforms import transform_options, transforms


def modify_dataloader_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--dataset_name', type=str, required=True, choices=datasets.keys())
    opt, _ = parser.parse_known_args()
    dataset_modify_commandline_options = dataset_options[opt.dataset_name]
    parser = dataset_modify_commandline_options(parser)

    parser.add_argument('--transform_name', type=str, required=True, choices=transforms.keys())
    opt, _ = parser.parse_known_args()
    transform_modify_commandline_options = transform_options[opt.transform_name]
    parser = transform_modify_commandline_options(parser)
    return parser
