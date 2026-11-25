"""Smoke and unit tests for src.analysis.ui_showcase (overlays, thresholds, DINOv2 heatmap)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.ui_showcase import (
    CurationMetricItem,
    DetectorBundle,
    InspectionResult,
    ShowcaseCatalogEntry,
    apply_interactive_thresholds,
    build_before_after_wipe_html,
    compute_cheek_roi_boxes,
    compute_dinov2_patch_heatmap,
    render_face_audit_overlay,
    render_latent_noise_preview,
)
from src.models.detector.image_detector import DeepfakeImageDetector


@pytest.mark.smoke
def test_render_overlays_handles_yunet_none_and_explicit_rgb_contract() -> None:
    """Verify overlay rendering handles missing YuNet faces and preserves RGB channel ordering."""
    img = np.full((512, 512, 3), 120, dtype=np.uint8)

    out_none = render_face_audit_overlay(img, yunet_face=None, passed=False)
    assert out_none.shape == (512, 512, 3)
    assert out_none.dtype == np.uint8

    yunet_face = {
        "bbox_xyxy": (100, 80, 412, 430),
        "score": 0.99,
        "significant_face_count": 1,
        "area_ratio": 0.42,
        "landmarks": {
            "right_eye": (190.0, 200.0),
            "left_eye": (320.0, 200.0),
            "nose": (256.0, 270.0),
            "mouth_right": (205.0, 345.0),
            "mouth_left": (305.0, 345.0),
        },
    }
    cheek_boxes = compute_cheek_roi_boxes(yunet_face, width=512, height=512)
    assert len(cheek_boxes) == 2
    for cx1, cy1, cx2, cy2 in cheek_boxes:
        assert 0 <= cx1 < cx2 <= 512
        assert 0 <= cy1 < cy2 <= 512

    mask = np.zeros((512, 512), dtype=np.float32)
    mask[320:380, 190:320] = 1.0

    out_pass = render_face_audit_overlay(
        img,
        yunet_face=yunet_face,
        passed=True,
        cheek_boxes=cheek_boxes,
        mask=mask,
    )
    out_fail = render_face_audit_overlay(
        img,
        yunet_face=yunet_face,
        passed=False,
        cheek_boxes=cheek_boxes,
        mask=mask,
    )

    assert out_pass.shape == (512, 512, 3) and out_pass.dtype == np.uint8
    assert out_fail.shape == (512, 512, 3) and out_fail.dtype == np.uint8

    # Top horizontal edge of bbox (y=80, x=256) must be green when passed=True and red when passed=False.
    pass_px = out_pass[80, 256]
    fail_px = out_fail[80, 256]
    assert int(pass_px[1]) > int(pass_px[0]) + 40 and int(pass_px[1]) > int(pass_px[2]) + 40
    assert int(fail_px[0]) > int(fail_px[1]) + 40 and int(fail_px[0]) > int(fail_px[2]) + 40

    # Eye landmarks (blue in RGB: B > R) and mouth landmarks (red/orange in RGB: R > B) must not invert.
    eye_px = out_pass[200, 190]
    mouth_px = out_pass[345, 205]
    assert int(eye_px[2]) > int(eye_px[0]) + 40
    assert int(mouth_px[0]) > int(mouth_px[2]) + 40


@pytest.mark.smoke
def test_render_noise_and_wipe_html_determinism() -> None:
    """Verify latent noise preview determinism and HTML before/after wipe structure."""
    noise_a = render_latent_noise_preview(42)
    noise_b = render_latent_noise_preview(42)
    noise_c = render_latent_noise_preview(99)

    assert noise_a.shape == (256, 256, 3)
    assert noise_a.dtype == np.uint8
    assert np.array_equal(noise_a, noise_b)
    assert not np.array_equal(noise_a, noise_c)

    before = np.full((128, 128, 3), 40, dtype=np.uint8)
    after = np.full((128, 128, 3), 210, dtype=np.uint8)
    html_str = build_before_after_wipe_html(
        before,
        after,
        left_label="INPUT FFHQ",
        right_label="OUTPUT <TEST>",
        height_px=300,
    )

    assert html_str.count("data:image/jpeg;base64,") == 2
    assert "clip-path" in html_str
    assert "INPUT FFHQ" in html_str
    assert "OUTPUT &lt;TEST&gt;" in html_str
    assert "<TEST>" not in html_str


@pytest.mark.smoke
def test_apply_interactive_thresholds_separates_cheek_fft_and_clip_by_mode() -> None:
    """Verify cheek_fft_ratio is gated independently, missing pair metrics are skipped, and min_clip applies directly."""
    # Case 1: whole-face FFT is high (0.04) but localized cheek_fft_ratio is waxy (0.0048 < 0.0050)
    waxy_measured = {
        "sharpness_score": 12.0,
        "geometry_valid": True,
        "eye_passed": True,
        "fft_high_freq_ratio": 0.04,
        "texture_flag": "normal",
        "cheek_fft_ratio": 0.0048,
        "clip_score": 0.28,
        "delta_e_lab": None,
        "seam_diff": None,
    }
    passed_waxy, reasons_waxy = apply_interactive_thresholds(
        waxy_measured,
        mode="sd_text2img",
        min_sharpness=1.3,
        min_cheek_fft=0.0050,
        min_clip=0.20,
    )
    assert passed_waxy is False
    assert any("cheek" in r.lower() for r in reasons_waxy)

    # Case 2: unpaired mode (rv51_ood / sd_text2img) with None delta_e_lab & seam_diff passes cleanly
    unpaired_ok = dict(waxy_measured, cheek_fft_ratio=0.0065)
    passed_ood, reasons_ood = apply_interactive_thresholds(
        unpaired_ok,
        mode="rv51_ood",
        min_sharpness=1.3,
        min_cheek_fft=0.0050,
        min_clip=0.20,
    )
    assert passed_ood is True
    assert reasons_ood == []

    # Case 3: caller's min_clip applies directly even on sd_img2img (no silent clamping to 0.12)
    img2img_measured = {
        "sharpness_score": 15.0,
        "geometry_valid": True,
        "eye_passed": True,
        "fft_high_freq_ratio": 0.05,
        "texture_flag": "normal",
        "cheek_fft_ratio": 0.0080,
        "clip_score": 0.18,
        "delta_e_lab": 8.5,
        "seam_diff": None,
    }
    passed_strict, reasons_strict = apply_interactive_thresholds(
        img2img_measured,
        mode="sd_img2img",
        min_sharpness=1.3,
        min_cheek_fft=0.0050,
        min_clip=0.20,
    )
    assert passed_strict is False
    assert any("clip" in r.lower() for r in reasons_strict)

    passed_relaxed, reasons_relaxed = apply_interactive_thresholds(
        img2img_measured,
        mode="sd_img2img",
        min_sharpness=1.3,
        min_cheek_fft=0.0050,
        min_clip=0.12,
    )
    assert passed_relaxed is True
    assert reasons_relaxed == []


@pytest.mark.smoke
def test_compute_dinov2_patch_heatmap_produces_valid_overlay_map() -> None:
    """Verify DINOv2 patch sensitivity map returns valid probability, raw map, RGB overlay, and eval state."""
    torch.manual_seed(42)
    model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
    model.freeze_backbone()
    model.eval()

    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)

    result = compute_dinov2_patch_heatmap(model, img, device=torch.device("cpu"))

    assert 0.0 <= result["prob"] <= 1.0
    assert result["heatmap_rgb"].shape == (512, 512, 3)
    assert result["heatmap_rgb"].dtype == np.uint8
    assert result["raw_map"].shape == (512, 512)
    assert float(result["raw_map"].min()) >= 0.0
    assert float(result["raw_map"].max()) <= 1.0
    assert float(result["raw_map"].max()) > 0.0
    assert model.training is False

    # Smoke-check exported dataclasses instantiate cleanly
    item = CurationMetricItem(name="sharpness", label="Sharpness", value=10.0, threshold=1.3, passed=True)
    entry = ShowcaseCatalogEntry(sample_id="s1", mode="sd_text2img", tab="sd_text2img", image_path="a.png")
    bundle = DetectorBundle(models={"M0": model}, device=torch.device("cpu"))
    insp = InspectionResult(mode="sd_text2img", image_512_rgb=img, metrics=[item])
    assert entry.sample_id == "s1"
    assert bundle.device.type == "cpu"
    assert insp.metrics[0].passed is True
