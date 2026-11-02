"""Tests for neural network architectures and model builders."""

import pytest
import torch

from src.models.builder import build_model
from src.models.deep import SimpleCVCNN


@pytest.mark.smoke
def test_simple_cv_cnn_forward(dummy_batch_tensor: torch.Tensor) -> None:
    """Smoke test: verify forward pass produces expected output shape (B, num_classes)."""
    model = SimpleCVCNN(in_channels=3, num_classes=5)
    model.eval()
    with torch.no_grad():
        out = model(dummy_batch_tensor)

    assert out.shape == (2, 5)
    assert not torch.isnan(out).any()


def test_build_model_factory() -> None:
    """Verify factory returns valid nn.Module instance."""
    model = build_model(architecture="custom_cnn", num_classes=10)
    assert isinstance(model, torch.nn.Module)
