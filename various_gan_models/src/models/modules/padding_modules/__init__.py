from typing import Callable, Dict, Tuple

import torch.nn as nn

from . import (identity_module, reflect_padding_module,
               replication_padding_module)

padding_modules: Dict[str, Callable[[], Tuple[nn.Module, int]]] = {
    'none': identity_module.create_norm_module,
    'reflect': reflect_padding_module.create_padding_module,
    'replication': replication_padding_module.create_padding_module,
}
