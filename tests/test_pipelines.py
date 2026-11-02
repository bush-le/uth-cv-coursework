"""Tests for pipeline utilities, metrics, and common helpers."""

import pytest

from src.common.logger import get_logger
from src.common.metrics import compute_accuracy, compute_iou
from src.common.seed import seed_everything


@pytest.mark.smoke
def test_seed_reproducibility() -> None:
    """Smoke test: verify seed_everything returns seed integer."""
    s = seed_everything(42)
    assert s == 42


@pytest.mark.smoke
def test_compute_iou_identical_boxes() -> None:
    """Smoke test: identical bounding boxes must yield IoU 1.0."""
    box = [10.0, 10.0, 50.0, 50.0]
    iou = compute_iou(box, box, box_format="xyxy")
    assert pytest.approx(iou, rel=1e-5) == 1.0


@pytest.mark.smoke
def test_compute_accuracy() -> None:
    """Smoke test: compute accuracy correctly."""
    preds = [0, 1, 2, 1]
    targets = [0, 1, 1, 1]
    acc = compute_accuracy(preds, targets)
    assert pytest.approx(acc, rel=1e-5) == 0.75


def test_logger_instance() -> None:
    """Verify logger setup."""
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
