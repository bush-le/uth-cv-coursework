"""Tests for data transformations and CV normalization."""

import numpy as np
import pytest

from src.data.transforms import normalize_image


@pytest.mark.smoke
def test_normalize_image_shape_hwc() -> None:
    """Smoke test: verify shape preservation on (H, W, C)."""
    img = np.random.rand(32, 32, 3).astype(np.float32)
    norm = normalize_image(img)
    assert norm.shape == (32, 32, 3)
    assert norm.dtype == np.float32


@pytest.mark.smoke
def test_normalize_image_shape_chw() -> None:
    """Smoke test: verify shape preservation on (C, H, W)."""
    img = np.random.rand(3, 32, 32).astype(np.float32)
    norm = normalize_image(img)
    assert norm.shape == (3, 32, 32)
    assert norm.dtype == np.float32
