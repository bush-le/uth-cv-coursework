"""Pytest fixtures for Computer Vision testing suite."""

import numpy as np
import pytest
import torch


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
