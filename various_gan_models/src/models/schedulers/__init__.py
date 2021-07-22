import argparse
from typing import Any, Callable, Dict

from . import (cosine_scheduler, linear_scheduler, plateau_scheduler,
               step_scheduler)

schedulers: Dict[str, Callable[[Any, argparse.Namespace], Any]] = {
    'discriminator_cosine_scheduler': cosine_scheduler.create_scheduler,
    'discriminator_linear_scheduler': linear_scheduler.create_scheduler,
    'discriminator_plateau_scheduler': plateau_scheduler.create_scheduler,
    'discriminator_step_scheduler': step_scheduler.create_discriminator_optimizer,
    'generator_cosine_scheduler': cosine_scheduler.create_scheduler,
    'generator_linear_scheduler': linear_scheduler.create_scheduler,
    'generator_plateau_scheduler': plateau_scheduler.create_scheduler,
    'generator_step_scheduler': step_scheduler.create_generator_optimizer,
}

scheduler_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'discriminator_cosine_scheduler': lambda x: x,
    'discriminator_linear_scheduler': lambda x: x,
    'discriminator_plateau_scheduler': lambda x: x,
    'discriminator_step_scheduler': step_scheduler.modify_discriminator_scheduler_commandline_options,
    'generator_cosine_scheduler': lambda x: x,
    'generator_linear_scheduler': lambda x: x,
    'generator_plateau_scheduler': lambda x: x,
    'generator_step_scheduler': step_scheduler.modify_generator_scheduler_commandline_options,
}
