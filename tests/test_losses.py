"""Unit tests for composite segmentation loss (CrossEntropy + Soft Dice)."""

import pytest
import torch

from src.models.losses import CompositeOcclusionLoss


def test_composite_loss_computation():
    criterion = CompositeOcclusionLoss(num_classes=6, ce_weight=1.0, dice_weight=1.0)

    # Logits: (B, C, H, W)
    logits = torch.randn(2, 6, 64, 64, requires_grad=True)
    # Targets: (B, H, W)
    targets = torch.randint(0, 6, (2, 64, 64))

    loss_dict = criterion(logits, targets)

    assert "loss" in loss_dict
    assert "ce_loss" in loss_dict
    assert "dice_loss" in loss_dict

    loss = loss_dict["loss"]
    assert loss.dim() == 0  # Scalar tensor
    assert loss.item() > 0.0

    # Backpropagation test
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_composite_loss_perfect_prediction():
    criterion = CompositeOcclusionLoss(num_classes=2, ce_weight=0.0, dice_weight=1.0)

    # 100% confident logits for class 1
    logits = torch.zeros(1, 2, 10, 10)
    logits[:, 1, :, :] = 100.0  # High probability for class 1
    targets = torch.ones(1, 10, 10, dtype=torch.int64)

    loss_dict = criterion(logits, targets)
    # Dice loss should approach 0.0
    assert loss_dict["dice_loss"].item() < 0.01
