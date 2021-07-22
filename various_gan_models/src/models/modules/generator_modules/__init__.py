import argparse
from typing import Callable, Dict

import torch.nn as nn

from . import cnn_module, resnet_module, unet_module, vanilla_fc_module

modules: Dict[str, Callable[[argparse.Namespace], nn.Module]] = {
    'cnn': cnn_module.create_module,
    'resnet': resnet_module.create_module,
    'unet': unet_module.create_module,
    'vanilla_fc': vanilla_fc_module.create_module,
}

module_options: Dict[str, Callable[[argparse.ArgumentParser], argparse.ArgumentParser]] = {
    'cnn': cnn_module.modify_module_commandline_options,
    'resnet': resnet_module.modify_module_commandline_options,
    'unet': unet_module.modify_module_commandline_options,
    'vanilla_fc': vanilla_fc_module.modify_module_commandline_options,
}
