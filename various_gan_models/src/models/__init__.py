import argparse
from typing import Callable, Dict

from . import one2one_pix2pix_model, pix2pix_model, vanilla_gan_model
from .abstract_model import AbstractModel

models: Dict[str, Callable[[argparse.Namespace], AbstractModel]] = {
    'vanilla_gan': vanilla_gan_model.create_model,
    'pix2pix': pix2pix_model.create_model,
    'one2one_pix2pix': one2one_pix2pix_model.create_model,
}

model_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'vanilla_gan': vanilla_gan_model.modify_model_commandline_options,
    'pix2pix': pix2pix_model.modify_model_commandline_options,
    'one2one_pix2pix': one2one_pix2pix_model.modify_model_commandline_options,
}
