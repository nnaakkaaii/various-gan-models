import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import l1_loss, lsgan_loss, vanilla_gan_loss, wgangp_loss

losses: Dict[str, Callable[[argparse.Namespace], nn.Module]] = {
    'l1_loss': l1_loss.create_loss,
    'vanilla_gan_loss': vanilla_gan_loss.create_loss,
    'lsgan_loss': lsgan_loss.create_loss,
    'wgangp_loss': wgangp_loss.create_loss,
}

loss_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'l1_loss': l1_loss.modify_loss_commandline_options,
    'vanilla_gan_loss': lambda x: x,
    'lsgan_loss': lambda x: x,
    'wgangp_loss': lambda x: x,
}
