"""Reproducibility utilities and global seed control."""

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seed across all libraries to ensure reproducible execution.

    Args:
        seed: The integer seed value to enforce.
        deterministic: If True, set PyTorch cuDNN to deterministic mode.

    Returns:
        The integer seed applied.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

    return seed
