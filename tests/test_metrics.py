"""Unit tests for binary classification, ROC-AUC, and hard-sample mining metrics."""

import numpy as np
import pytest
import torch

from src.common.metrics import (
    calibrate_threshold,
    compute_accuracy,
    compute_bypass_rate,
    compute_hard_sample_recall,
    compute_iou,
    compute_laplacian_sharpness,
    compute_roc_auc,
    detect_faces,
)


def test_compute_accuracy_perfect_match() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    preds = np.array([0, 0, 1, 1], dtype=np.int64)

    acc = compute_accuracy(preds, targets)
    assert acc == pytest.approx(1.0)


def test_compute_roc_auc_ranking() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    probs = np.array([0.1, 0.2, 0.8, 0.95], dtype=np.float32)

    auc = compute_roc_auc(targets, probs)
    assert auc == pytest.approx(1.0)


def test_compute_bypass_rate() -> None:
    # 4 fake samples: 2 caught (>= 0.5), 2 bypassed (< 0.5)
    fake_probs = np.array([0.85, 0.72, 0.35, 0.15], dtype=np.float32)

    bypass = compute_bypass_rate(fake_probs, threshold=0.5)
    assert bypass == pytest.approx(0.5)


def test_compute_hard_sample_recall() -> None:
    preds = np.array([0, 1, 1, 0], dtype=np.int64)
    targets = np.array([0, 1, 1, 1], dtype=np.int64)
    is_hard = np.array([False, True, True, True], dtype=bool)

    recall = compute_hard_sample_recall(preds, targets, is_hard)
    assert recall == pytest.approx(2.0 / 3.0)


def test_compute_iou_and_curation_metrics() -> None:
    iou = compute_iou([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], box_format="xyxy")
    assert iou == pytest.approx(1.0)

    img = torch.rand(3, 32, 32)
    sharpness = compute_laplacian_sharpness(img)
    assert sharpness >= 0.0


def test_calibrate_threshold_is_quantile() -> None:
    assert calibrate_threshold([1, 2, 3, 4, 5], 0.0) == 1.0
    assert calibrate_threshold([1, 2, 3, 4, 5], 1.0) == 5.0
    assert calibrate_threshold([10, 20, 30, 40, 50], 0.5) == 30.0

    with pytest.raises(ValueError):
        calibrate_threshold([], 0.5)

    with pytest.raises(ValueError):
        calibrate_threshold([1, 2], 1.5)


def test_face_check_none_backend_skips_and_haar_backend_rejects_blank() -> None:
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detect_faces(blank, backend="opencv_haar") == 0
    assert detect_faces(blank, backend="none") == 1

