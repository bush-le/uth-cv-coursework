"""Unit tests for segmentation metrics (mIoU and Boundary IoU)."""

import numpy as np
import pytest

from src.common.metrics import compute_boundary_iou, compute_miou


def test_compute_miou_perfect_match():
    # 2 classes: 0 and 1
    targets = np.array([[0, 0], [1, 1]], dtype=np.int64)
    preds = np.array([[0, 0], [1, 1]], dtype=np.int64)

    metrics = compute_miou(preds, targets, num_classes=2)
    assert "miou" in metrics
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["iou_class_0"] == pytest.approx(1.0)
    assert metrics["iou_class_1"] == pytest.approx(1.0)


def test_compute_miou_partial_overlap():
    targets = np.array([[0, 1], [1, 1]], dtype=np.int64)
    preds = np.array([[0, 0], [1, 1]], dtype=np.int64)

    # Class 0: target has 1 pixel, pred has 2 pixels. Intersection = 1, Union = 2 -> IoU = 0.5
    # Class 1: target has 3 pixels, pred has 2 pixels. Intersection = 2, Union = 3 -> IoU = 2/3
    metrics = compute_miou(preds, targets, num_classes=2)
    assert metrics["iou_class_0"] == pytest.approx(0.5)
    assert metrics["iou_class_1"] == pytest.approx(2.0 / 3.0)
    assert metrics["miou"] == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)


def test_compute_boundary_iou_perfect_match():
    # 10x10 grid with center box as class 1
    targets = np.zeros((10, 10), dtype=np.int64)
    targets[2:8, 2:8] = 1
    preds = targets.copy()

    biou = compute_boundary_iou(preds, targets, num_classes=2, dilation_radius=1)
    assert "mean_biou" in biou
    assert biou["biou_class_1"] == pytest.approx(1.0)
