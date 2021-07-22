import argparse
import json
import os

from ..models.base_model import AbstractModel
from . import base_logger
from .abstract_logger import AbstractLogger


def create_logger(model: AbstractModel, opt: argparse.Namespace) -> AbstractLogger:
    return SimpleLogger(model, opt)


def modify_logger_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_logger.modify_logger_commandline_options(parser)
    parser.add_argument('--save_freq', type=int, default=5, help='モデルの出力の保存頻度')
    return parser


class SimpleLogger(base_logger.BaseLogger):
    def __init__(self, model: AbstractModel, opt: argparse.Namespace) -> None:
        super().__init__(model=model, opt=opt)

    def end_epoch(self) -> None:
        super().end_epoch()
        for key, value in self.averager.value().items():
            self.history[key].append(value)

        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f)

        if self._trained_epoch % self.opt['save_freq'] == 0:
            self.model.save_networks(self._trained_epoch)
            self.model.save_current_image(self._trained_epoch)
        return

    def end_iter(self) -> None:
        current_losses = self.model.get_current_losses()
        self.averager.send(current_losses)
        self.print_status(
            iterations=self.averager.iterations,
            status_dict=current_losses,
        )
        return

    def end_all_training(self) -> None:
        self.model.save_networks('last')
        return
