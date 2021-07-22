import abc
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, List

from ..models.base_model import AbstractModel
from .abstract_logger import AbstractLogger
from .utils.averager import Averager


def modify_logger_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


class BaseLogger(AbstractLogger, metaclass=abc.ABCMeta):
    def __init__(self, model: AbstractModel, opt: argparse.Namespace) -> None:
        super().__init__(model=model, opt=opt)
        self.save_dir = model.save_dir
        self._trained_epoch = opt.epoch
        self._dataset_length = -1
        self.averager = Averager()

        self.history: DefaultDict[str, List[float]] = defaultdict(list)
        os.makedirs(self.save_dir, exist_ok=True)

    def start_epoch(self) -> None:
        self.averager.reset()
        return

    def end_epoch(self) -> None:
        self._increment_epoch()
        return

    def end_iter(self) -> None:
        current_losses = self.model.get_current_losses()
        self.averager.send(current_losses)
        return

    def save_options(self) -> None:
        with open(os.path.join(self.save_dir, 'options.json'), 'w') as f:
            json.dump(self.opt, f)
        return

    def set_dataset_length(self, dataset_length: int) -> None:
        self._dataset_length = dataset_length
        return

    def _increment_epoch(self) -> None:
        self._trained_epoch += 1
        return

    def print_status(self, iterations: int, status_dict: Dict[str, float]) -> None:
        status_str = f'[Epoch {self._trained_epoch}][{iterations:.0f}/{self._dataset_length}] '
        for key, value in status_dict.items():
            status_str += f'{key}: {value:.4f}, '
        sys.stdout.write(f'\r{status_str}')
        sys.stdout.flush()
        return
