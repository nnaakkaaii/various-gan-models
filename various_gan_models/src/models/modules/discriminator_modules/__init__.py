import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import cnn_module, n_layer_module, pixel_module, vanilla_fc_module

modules: Dict[str, Callable[[argparse.Namespace], nn.Module]] = {
    'cnn': cnn_module.create_module,
    'n_layer': n_layer_module.create_module,
    'pixel': pixel_module.create_module,
    'vanilla_fc': vanilla_fc_module.create_module,
}

module_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'cnn': cnn_module.modify_module_commandline_options,
    'n_layer': n_layer_module.modify_module_commandline_options,
    'pixel': pixel_module.modify_module_commandline_options,
    'vanilla_fc': vanilla_fc_module.modify_module_commandline_options,
}
