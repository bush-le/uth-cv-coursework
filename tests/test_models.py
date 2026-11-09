"""Tests for neural network architectures and model builders."""

import pytest
import torch

from src.models.builder import build_model
from src.models.deep import SimpleCVCNN
from src.models.detector import DeepfakeImageDetector


@pytest.mark.smoke
def test_simple_cv_cnn_forward(dummy_batch_tensor: torch.Tensor) -> None:
    """Smoke test: verify forward pass produces expected output shape (B, num_classes)."""
    model = SimpleCVCNN(in_channels=3, num_classes=5)
    model.eval()
    with torch.no_grad():
        out = model(dummy_batch_tensor)

    assert out.shape == (2, 5)
    assert not torch.isnan(out).any()


@pytest.mark.smoke
def test_build_model_dinovit_factory(dummy_batch_tensor: torch.Tensor) -> None:
    """Smoke test: verify factory builds DeepfakeImageDetector with dino_vit backbone."""
    detector = build_model(architecture="dino_vit", num_classes=1)
    assert isinstance(detector, DeepfakeImageDetector)
    assert detector.backbone_name == "dino_vit"

    detector.eval()
    with torch.no_grad():
        out = detector(dummy_batch_tensor)
    assert out.shape == (2, 1)


@pytest.mark.smoke
def test_build_model_dinov2_factory(dummy_batch_tensor: torch.Tensor) -> None:
    """Smoke test: verify factory builds DeepfakeImageDetector with dinov2_small backbone."""
    detector = build_model(architecture="dinov2_small", num_classes=1, pretrained=True)
    assert isinstance(detector, DeepfakeImageDetector)
    assert detector.backbone_name == "dinov2_small"


def test_build_model_factory() -> None:
    """Verify factory returns valid nn.Module instance."""
    model = build_model(architecture="custom_cnn", num_classes=10)
    assert isinstance(model, torch.nn.Module)
