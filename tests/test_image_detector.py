"""Unit tests for downstream 2D deepfake image detector and benchmark evaluation."""

import pytest
import torch
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector


def test_deepfake_image_detector_forward_simple():
    """Test DeepfakeImageDetector forward pass with simple CNN backbone."""
    model = DeepfakeImageDetector(backbone="simple_cnn", num_classes=1)
    x = torch.rand(4, 3, 64, 64) # (B, 3, H, W)
    logits = model(x)

    assert logits.shape == (4, 1)
    assert logits.dtype == torch.float32

    probs = model.predict_proba(x)
    assert probs.shape == (4, 1)
    assert 0.0 <= probs.min() and probs.max() <= 1.0


def test_deepfake_image_detector_forward_resnet():
    """Test DeepfakeImageDetector forward pass with ResNet-18 backbone."""
    model = DeepfakeImageDetector(backbone="resnet18", num_classes=1)
    x = torch.rand(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 1)


def test_evaluate_image_detector_benchmark():
    """Test evaluation benchmark calculating accuracy and ROC-AUC metrics."""
    model = DeepfakeImageDetector(backbone="simple_cnn", num_classes=1)
    mock_batch = {
        "images": torch.rand(4, 3, 32, 32),
        "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }

    metrics = evaluate_image_detector(model, [mock_batch], device="cpu")
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
