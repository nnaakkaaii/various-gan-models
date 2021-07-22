import argparse
from typing import Any

from torch.optim import lr_scheduler


def create_scheduler(optimizer: Any, opt: argparse.Namespace) -> Any:
    return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
