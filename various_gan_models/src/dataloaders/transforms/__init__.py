import argparse
from typing import Any, Callable, Dict

from . import (affine_transform, crop_and_random_flip_transform,
               vanilla_numpy2tensor_transform)

transforms: Dict[str, Callable[[argparse.Namespace], Any]] = {
    'affine': affine_transform.create_transform,
    'crop_and_random_flip': crop_and_random_flip_transform.create_transform,
    'vanilla_numpy2tensor': vanilla_numpy2tensor_transform.create_transform,
}

transform_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'affine': affine_transform.modify_transform_commandline_option,
    'crop_and_random_flip': crop_and_random_flip_transform.modify_transform_commandline_options,
    'vanilla_numpy2tensor': vanilla_numpy2tensor_transform.modify_transform_commandline_options,
}
