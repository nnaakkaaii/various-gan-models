import argparse
from typing import Any, Callable, Dict

import torch.utils.data as data

from . import (fillet_shadow2contour_dataset,
               tree_gravity_shadow2contour_dataset)

datasets: Dict[str, Callable[[Any, argparse.Namespace], data.Dataset]] = {
    'fillet_shadow2contour': fillet_shadow2contour_dataset.create_dataset,
    'tree_gravity_shadow2contour': tree_gravity_shadow2contour_dataset.create_dataset,
}

dataset_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'fillet_shadow2contour': fillet_shadow2contour_dataset.modify_dataset_commandline_options,
    'tree_gravity_shadow2contour': tree_gravity_shadow2contour_dataset.modify_dataset_commandline_options,
}
