import argparse
from typing import Any, Callable, Dict

import torch.utils.data as data

from . import simple_dataloader

dataloaders: Dict[str, Callable[[data.Dataset, argparse.Namespace], Any]] = {
    'simple': simple_dataloader.create_dataloader,
}

dataloader_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'simple': simple_dataloader.modify_dataloader_commandline_options,
}
