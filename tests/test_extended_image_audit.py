"""Unit tests for extended mode-aware image quality audit metrics."""

import numpy as np
import pytest
import torch

from src.common.metrics import compute_fft_high_freq_ratio, verify_eye_pair


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


@pytest.mark.smoke
def test_fft_high_freq_ratio_flat_image_guarded() -> None:
    """Zero-energy flat image must safely return 0.0 without division by zero."""
    flat = torch.zeros((3, 128, 128), dtype=torch.float32)
    res = compute_fft_high_freq_ratio(flat)
    assert res["high_freq_ratio"] == pytest.approx(0.0)
    assert res["texture_flag"] == "waxy_low_freq"
    assert res["passed"] is False


@pytest.mark.smoke
def test_fft_high_freq_ratio_distinguishes_smooth_vs_noisy() -> None:
    """Checkerboard crunchy texture exhibits higher high-frequency ratio than smooth gradient."""
    x = np.linspace(0, 1, 128, dtype=np.float32)[None, :]
    smooth = np.repeat(np.repeat(x, 128, axis=0)[..., None], 3, axis=-1)
    yy, xx = np.indices((128, 128))
    crunchy = (((yy + xx) % 2) * 255).astype(np.uint8)
    crunchy_rgb = np.stack([crunchy, crunchy, crunchy], axis=-1)

    res_smooth = compute_fft_high_freq_ratio(torch.from_numpy(smooth).permute(2, 0, 1))
    res_crunchy = compute_fft_high_freq_ratio(crunchy_rgb)
    assert res_smooth["high_freq_ratio"] < res_crunchy["high_freq_ratio"]
    assert res_crunchy["texture_flag"] == "crunchy_high_freq"
    assert res_crunchy["passed"] is False
