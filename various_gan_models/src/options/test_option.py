import argparse

from .base_option import BaseOption


class TestOption(BaseOption):
    """This class includes test options.
    It also includes shared options defined in BaseOption.
    """

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.add_argument('--results_dir', type=str, default='results', help='saves results here.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        self.is_train = False
        return parser
