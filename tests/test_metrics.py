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
    simulate_cascaded_defense,
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


def test_compute_roc_auc_with_tied_scores() -> None:
    """Verify tied probabilities yield unbiased 0.5 ROC-AUC via mid-rank averaging."""
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    # Saturated tied scores
    probs = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    auc = compute_roc_auc(targets, probs)
    assert auc == pytest.approx(0.5, abs=1e-5), f"Expected 0.5 for tied scores, got {auc}"


def test_compute_roc_auc_single_class_returns_nan() -> None:
    """Verify single class targets return NaN rather than false 1.0."""
    import math

    targets = np.array([1, 1, 1, 1], dtype=np.int64)
    probs = np.array([0.8, 0.9, 0.7, 0.95], dtype=np.float32)
    auc = compute_roc_auc(targets, probs)
    assert math.isnan(auc), f"Expected NaN for single-class target, got {auc}"



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


def test_simulate_cascaded_defense() -> None:
    # 4 samples: [Real, Real, Fake, Fake]
    m1_probs = np.array([0.10, 0.40, 0.35, 0.80], dtype=np.float32)
    m2_probs = np.array([0.05, 0.20, 0.60, 0.90], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    # tau1=0.30: samples 1, 2, 3 routed (intercept_rate = 3/4 = 0.75)
    # tau2=0.50: samples 2, 3 confirmed as fake (m2 >= 0.50). Sample 1 cleared as real (m2=0.20 < 0.50)
    res = simulate_cascaded_defense(m1_probs, m2_probs, labels, tau1=0.30, tau2=0.50)

    assert res["total_samples"] == 4.0
    assert res["stage1_intercept_rate"] == 0.75
    assert res["overall_flagged_fake"] == 2.0
    assert res["overall_recall"] == 1.0  # Both fakes (2 and 3) detected
    assert res["overall_precision"] == 1.0  # No real samples flagged as fake
    assert res["overall_accuracy"] == 1.0
    assert res["false_positive_rate"] == 0.0

    # Test edge case: empty inputs
    empty_res = simulate_cascaded_defense([], [])
    assert empty_res["total_samples"] == 0.0


