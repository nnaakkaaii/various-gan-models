import argparse


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the last conv layer')
    return parser
