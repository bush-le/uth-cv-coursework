"""Unit tests for extended mode-aware image quality audit metrics."""

import numpy as np
import pytest
import torch

from src.common.metrics import verify_eye_pair


@pytest.mark.smoke
def test_verify_eye_pair_symmetry_and_schema() -> None:
    """Symmetric synthetic face-like pattern returns full audit schema and high symmetry."""
    img = torch.full((3, 256, 256), 0.7, dtype=torch.float32)
    img[:, 75:95, 70:105] = 0.1
    img[:, 75:95, 151:186] = 0.1
    res = verify_eye_pair(img, face_bbox_xyxy=(32, 32, 224, 224))
    assert set(res.keys()) >= {
        "eye_count",
        "eyes_detected",
        "sunglasses_occluded",
        "bilateral_symmetry",
        "passed",
        "reasons",
    }
    assert res["bilateral_symmetry"] >= 0.90


@pytest.mark.smoke
def test_verify_eye_pair_asymmetric_face_fails_symmetry() -> None:
    """Strongly asymmetric face ROI fails bilateral symmetry threshold."""
    rng = np.random.default_rng(42)
    img = np.full((256, 256, 3), 230, dtype=np.uint8)
    img[30:220, 30:125, :] = rng.integers(5, 30, size=(190, 95, 3), dtype=np.uint8)
    res = verify_eye_pair(img, face_bbox_xyxy=(30, 30, 220, 220), min_symmetry=0.60)
    assert res["bilateral_symmetry"] < 0.60
    assert res["passed"] is False


@pytest.mark.smoke
def test_verify_eye_pair_sunglasses_occlusion_handled_gracefully() -> None:
    """Dark sunglasses band across upper eye region sets sunglasses_occluded without false missing-eye rejection."""
    img = np.full((256, 256, 3), 190, dtype=np.uint8)
    img[80:125, 55:200, :] = 15
    res = verify_eye_pair(img, face_bbox_xyxy=(32, 32, 224, 224))
    assert res["sunglasses_occluded"] is True
    assert not any("No eyes detected" in r for r in res["reasons"])
