import argparse

from .base_option import BaseOption


class TrainOption(BaseOption):
    """This class includes training options.
    """

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--epoch', type=int, default=1, help='事前学習時の最後のepoch (読み込みたい重みのepoch)')
        parser.add_argument('--continue_train', action='store_true', help='前回の学習を続行するか')

        self.is_train = True
        return parser
