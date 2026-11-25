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
    """Dark sunglasses band across upper eye region sets sunglasses_occluded without missing-eye rejection."""
    img = np.full((256, 256, 3), 190, dtype=np.uint8)
    img[80:125, 55:200, :] = 15
    res = verify_eye_pair(img, face_bbox_xyxy=(32, 32, 224, 224))
    assert res["sunglasses_occluded"] is True
    assert not any("Expected >= 2 eyes" in r for r in res["reasons"])
    assert res["passed"] is True


@pytest.mark.smoke
def test_verify_eye_pair_merges_eye_and_glasses_cascades(monkeypatch: pytest.MonkeyPatch) -> None:
    """Glasses cascade fallback must merge with eye_cascade detections instead of overwriting them."""
    class _StubCascade:
        def __init__(self, boxes):
            self._boxes = np.array(boxes, dtype=np.int32) if boxes else ()

        def empty(self) -> bool:
            return False

        def detectMultiScale(self, *args, **kwargs):
            return self._boxes

    # Case 1: eye_cascade detects 1 eye, glasses_cascade detects 0 -> must preserve eye_count=1
    monkeypatch.setattr(
        "src.common.metrics._get_eye_classifiers",
        lambda: (_StubCascade([(20, 20, 24, 24)]), _StubCascade([])),
    )
    img = np.full((256, 256, 3), 180, dtype=np.uint8)
    res1 = verify_eye_pair(img, face_bbox_xyxy=(32, 32, 224, 224))
    assert res1["eye_count"] == 1

    # Case 2: eye_cascade detects left eye, glasses_cascade detects right eye -> merged eye_count=2
    monkeypatch.setattr(
        "src.common.metrics._get_eye_classifiers",
        lambda: (_StubCascade([(20, 20, 24, 24)]), _StubCascade([(110, 20, 24, 24)])),
    )
    res2 = verify_eye_pair(img, face_bbox_xyxy=(32, 32, 224, 224))
    assert res2["eye_count"] == 2
    assert res2["eyes_detected"] is True
    assert res2["passed"] is True


@pytest.mark.smoke
def test_verify_eye_pair_no_bbox_detects_face_on_highres_sunglasses() -> None:
    """When face_bbox_xyxy=None on a real 1024x1024 FFHQ sunglasses image or high-res face, relaxed Haar avoids full-canvas dilution."""
    from pathlib import Path
    from PIL import Image

    ffhq_sample = Path("data/raw/faces/ffhq/00000.png")
    if ffhq_sample.is_file():
        arr = np.array(Image.open(ffhq_sample).convert("RGB"), dtype=np.uint8)
        # Paint dark sunglasses band over the aligned eye zone (~y=360..500, x=280..740)
        arr[360:500, 280:740, :] = 15
        res_no_box = verify_eye_pair(arr, face_bbox_xyxy=None)
        assert res_no_box["sunglasses_occluded"] is True
        assert not any("Expected >= 2 eyes" in r for r in res_no_box["reasons"])


@pytest.mark.smoke
def test_audit_image_extended_parity_and_mismatched_source() -> None:
    """audit_image_extended gives identical outputs for uint8 HWC and float32 CHW and handles 1024x1024 source + 4-ch tensor."""
    rng = np.random.default_rng(7)
    arr_u8 = rng.integers(80, 180, size=(512, 512, 3), dtype=np.uint8)
    ten_chw = torch.from_numpy(arr_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
    src_hi = np.repeat(np.repeat(arr_u8, 2, axis=0), 2, axis=1)
    mask_small = np.zeros((3, 3, 1), dtype=np.float32)

    rep_u8 = audit_image_extended(arr_u8, mode="sd_img2img", source_image=src_hi, image_id="p1")
    rep_chw = audit_image_extended(ten_chw, mode="sd_img2img", source_image=src_hi, image_id="p1")
    assert rep_u8.fft_high_freq_ratio == pytest.approx(rep_chw.fft_high_freq_ratio, abs=1e-6)
    assert rep_u8.delta_e_lab == pytest.approx(rep_chw.delta_e_lab, abs=1e-5)

    # Ensure 4-channel CHW and tiny (H<=3, W, 1) mask do not crash
    from src.common.metrics import _to_rgb_uint8
    ten_4ch = torch.zeros((4, 64, 64), dtype=torch.float32)
    assert _to_rgb_uint8(ten_4ch).shape == (64, 64, 3)
    pair_tiny = compute_pair_color_and_structure(ten_4ch, ten_4ch, mask=mask_small)
    assert pair_tiny["delta_e_lab"] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.smoke
def test_fft_high_freq_ratio_passes_real_ffhq_sample() -> None:
    """Calibrated default min_ratio passes natural FFHQ portrait texture without false waxy rejection."""
    from pathlib import Path
    from PIL import Image

    ffhq_sample = Path("data/raw/faces/ffhq/00000.png")
    if ffhq_sample.is_file():
        arr = np.array(Image.open(ffhq_sample).convert("RGB"), dtype=np.uint8)
        fft_res = compute_fft_high_freq_ratio(arr)
        assert fft_res["texture_flag"] == "normal"
        assert fft_res["passed"] is True


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


@pytest.mark.smoke
def test_sunglasses_inpaint_permits_dark_lenses_but_rejects_bg_drift() -> None:
    """Sunglasses inpaint tolerates dark lens delta-E inside mask while catching background drift."""
    src = np.full((256, 256, 3), 180, dtype=np.uint8)
    mask = torch.zeros((1, 256, 256), dtype=torch.float32)
    mask[:, 70:115, 75:181] = 1.0

    # Valid black sunglasses inside mask, feathered seam, preserved background
    syn_ok = src.copy()
    syn_ok[80:105, 85:171, :] = 20
    rep_ok = audit_image_extended(
        syn_ok,
        mode="sd_inpaint",
        prompt="wearing stylish black sunglasses, realistic skin",
        source_image=src,
        mask=mask,
        min_sharpness=0.0,
        run_geometry=False,
    )
    assert not any("CIELab color drift" in r for r in rep_ok.rejection_reasons)

    # Corrupted background outside mask must still fail Background CIELab drift
    syn_bg_bad = syn_ok.copy()
    syn_bg_bad[0:60, :, :] = 40
    rep_bad = audit_image_extended(
        syn_bg_bad,
        mode="sd_inpaint",
        prompt="wearing stylish black sunglasses",
        source_image=src,
        mask=mask,
        min_sharpness=0.0,
        run_geometry=False,
    )
    assert any("Background CIELab drift" in r for r in rep_bad.rejection_reasons)
    assert rep_bad.is_valid is False


@pytest.mark.smoke
def test_verify_eye_pair_clear_eyes_not_flagged_sunglasses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Portraits with 2 detected eyes must not falsely set sunglasses_occluded without glasses prompt."""
    class _StubTwoEyes:
        def empty(self) -> bool:
            return False

        def detectMultiScale(self, *args, **kwargs):
            return np.array([(25, 25, 24, 24), (115, 25, 24, 24)], dtype=np.int32)

    monkeypatch.setattr(
        "src.common.metrics._get_eye_classifiers",
        lambda: (_StubTwoEyes(), _StubTwoEyes()),
    )
    img = np.full((256, 256, 3), 180, dtype=np.uint8)
    # Dark eyebrows/shadow band (~40% dark) with both eyes clearly visible
    img[78:100, 50:205, :] = 20
    res = verify_eye_pair(img, face_bbox_xyxy=(32, 32, 224, 224), prompt="a smiling mouth")
    assert res["eyes_detected"] is True
    assert res["sunglasses_occluded"] is False
    assert res["passed"] is True
