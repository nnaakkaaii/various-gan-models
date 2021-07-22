import argparse


def modify_module_commandline_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in the first conv layer')
    return parser
