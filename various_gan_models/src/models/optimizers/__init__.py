import argparse
from typing import Any, Callable, Dict

from . import adam_optimizer

optimizers: Dict[str, Callable[[Any, argparse.Namespace], Any]] = {
    'discriminator_adam': adam_optimizer.create_discriminator_optimizer,
    'generator_adam': adam_optimizer.create_discriminator_optimizer,
}

optimizer_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'discriminator_adam': adam_optimizer.modify_discriminator_optimizer_commandline_options,
    'generator_adam': adam_optimizer.modify_generator_optimizer_commandline_options,
}
