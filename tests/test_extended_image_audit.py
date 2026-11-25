"""Unit tests for extended mode-aware image quality audit metrics."""

import numpy as np
import pytest
import torch

from src.common.metrics import (
    ExtendedImageAuditReport,
    audit_image_extended,
    compute_fft_high_freq_ratio,
    compute_pair_color_and_structure,
    verify_eye_pair,
)


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


@pytest.mark.smoke
def test_pair_color_and_structure_resizes_and_checks_seam() -> None:
    """Paired color/structure metric resizes mismatched source images and detects seam/color drift."""
    src = np.full((512, 512, 3), 150, dtype=np.uint8)
    src[100:400, 150:350, 0] = 200
    src_hi = np.repeat(np.repeat(src, 2, axis=0), 2, axis=1)

    syn_ok = src.copy()
    mask = torch.zeros((1, 512, 512), dtype=torch.float32)
    mask[:, 280:360, 180:330] = 1.0
    res_ok = compute_pair_color_and_structure(syn_ok, src_hi, mask=mask)
    assert res_ok["delta_e_lab"] < 1.0
    assert res_ok["structure_corr"] > 0.90
    assert res_ok["is_seamless"] is True
    assert res_ok["passed"] is True

    syn_bad = src.copy()
    syn_bad[280:360, 180:330, :] = [20, 220, 240]
    res_bad = compute_pair_color_and_structure(syn_bad, src_hi, mask=mask, max_delta_e=10.0)
    assert res_bad["delta_e_lab"] > 10.0
    assert res_bad["is_seamless"] is False
    assert res_bad["passed"] is False


@pytest.mark.smoke
def test_audit_image_extended_handles_all_three_modes() -> None:
    """Unified audit_image_extended dispatches mode-specific checks across text2img, img2img, and inpaint."""
    img = torch.full((3, 512, 512), 0.6, dtype=torch.float32)
    rep_t2i = audit_image_extended(img, mode="sd_text2img", prompt="portrait", clip_scorer=None, image_id="t2i_1")
    assert isinstance(rep_t2i, ExtendedImageAuditReport)
    assert rep_t2i.mode == "sd_text2img"
    assert rep_t2i.clip_score is None
    assert rep_t2i.delta_e_lab is None
    assert rep_t2i.seam_diff is None

    rep_i2i = audit_image_extended(img, mode="sd_img2img", source_image=img, image_id="i2i_1")
    assert rep_i2i.delta_e_lab == pytest.approx(0.0, abs=1e-4)
    assert rep_i2i.seam_diff is None

    mask = torch.zeros((1, 512, 512), dtype=torch.float32)
    mask[:, 150:350, 150:350] = 1.0
    rep_inp = audit_image_extended(img, mode="sd_inpaint", source_image=img, mask=mask, image_id="inp_1")
    assert rep_inp.seam_diff is not None
    assert rep_inp.is_seamless is True
