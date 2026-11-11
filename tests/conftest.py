"""Pytest fixtures for Computer Vision testing suite."""

import os
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure TMPDIR defaults to local project cache when available
_cache_tmp = Path(__file__).resolve().parent.parent / ".cache" / "tmp"
try:
    _cache_tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(_cache_tmp))
except (PermissionError, OSError):
    pass



@pytest.fixture
def dummy_image_rgb() -> np.ndarray:
    """Fixture providing synthetic 64x64 RGB image with uint8 in [0, 255]."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def dummy_image_bgr() -> np.ndarray:
    """Fixture providing synthetic 64x64 BGR image with uint8 in [0, 255]."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def dummy_batch_tensor() -> torch.Tensor:
    """Fixture providing PyTorch tensor of shape (B=2, C=3, H=32, W=32)."""
    return torch.randn(2, 3, 32, 32, dtype=torch.float32)
