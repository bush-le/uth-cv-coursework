"""Smoke and unit tests for src.analysis.ui_showcase (overlays, thresholds, DINOv2 heatmap)."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
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
    inspect_pipeline_sample,
    launch_live_sd_worker,
    load_detector_bundle,
    render_face_audit_overlay,
    render_latent_noise_preview,
    score_image_with_bundle,
    unload_detector_bundle,
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


def test_load_real_detector_bundle_and_inspect_real_ffhq_and_hard_inpaint() -> None:
    """Verify real M0/M2*/M3 checkpoints inspect real FFHQ and paired hard inpaint sample end-to-end."""
    ckpts = [
        Path("experiments/checkpoints/baseline_d0_best.pt"),
        Path("experiments/checkpoints/active_learning_m2_best.pt"),
        Path("experiments/checkpoints/active_learning_m3_best.pt"),
    ]
    real_path = Path("data/raw/faces/ffhq/00820.png")
    fake_path = Path("data/synthetic/master_test_images/inpainting/fake_ffhq_00820_3.png")
    if not all(p.is_file() for p in ckpts) or not real_path.is_file() or not fake_path.is_file():
        pytest.skip("Required real checkpoints or FFHQ/inpaint sample files missing locally.")

    bundle = load_detector_bundle()
    try:
        assert set(bundle.models.keys()) == {"M0", "M2*", "M3"}

        real_res = inspect_pipeline_sample(
            real_path,
            bundle=bundle,
            mode="real_ffhq",
            ground_truth_label=0,
        )
        assert real_res.gate_passed is True
        assert real_res.measured["cheek_fft_ratio"] is not None
        assert real_res.before_after_html is None
        assert real_res.detector_scores["M3"] < 0.50
        assert real_res.heatmaps["M0"].shape == (512, 512, 3)
        assert real_res.heatmaps["M0"].dtype == np.uint8
        assert real_res.heatmaps["M3"].shape == (512, 512, 3)
        assert real_res.heatmaps["M3"].dtype == np.uint8
        assert "total" in real_res.latencies_ms and real_res.latencies_ms["total"] > 0.0

        inpaint_prompt = (
            "seamless natural facial features, authentic skin texture, man face, masculine contours, "
            "adult portrait, natural skin, dark sunglasses with subtle lens glare, candid neutral glance, "
            "contemplative look, natural expression, dramatic cinematic side lighting, chiaroscuro shadows, "
            "sculpted cheekbones"
        )
        fake_res = inspect_pipeline_sample(
            fake_path,
            bundle=bundle,
            source_image=real_path,
            mask_region="lips",
            mode="sd_inpaint",
            prompt=inpaint_prompt,
            ground_truth_label=1,
        )
        assert fake_res.gate_passed is True
        assert fake_res.measured["cheek_fft_ratio"] is not None
        assert "M3" in fake_res.detector_scores
        assert 0.0 <= fake_res.detector_scores["M3"] <= 1.0
        assert fake_res.detector_scores["M3"] >= 0.50
        assert fake_res.heatmaps["M0"].shape == (512, 512, 3)
        assert fake_res.heatmaps["M0"].dtype == np.uint8
        assert fake_res.heatmaps["M3"].shape == (512, 512, 3)
        assert fake_res.heatmaps["M3"].dtype == np.uint8
        assert fake_res.before_after_html is not None
        assert "detector" in fake_res.latencies_ms and "heatmaps" in fake_res.latencies_ms
    finally:
        unload_detector_bundle(bundle)


def test_inspect_non_face_rgba_upload_fails_gate_and_honors_force_flag(tmp_path: Path) -> None:
    """Verify random RGBA non-face upload fails curation gate and respects run_detector_if_rejected."""
    ckpts = [
        Path("experiments/checkpoints/baseline_d0_best.pt"),
        Path("experiments/checkpoints/active_learning_m2_best.pt"),
        Path("experiments/checkpoints/active_learning_m3_best.pt"),
    ]
    if not all(p.is_file() for p in ckpts):
        pytest.skip("Required real checkpoints missing locally.")

    rng = np.random.default_rng(123)
    rgba_arr = rng.integers(0, 256, size=(160, 240, 4), dtype=np.uint8)
    rgba_path = tmp_path / "non_face_rgba.png"
    Image.fromarray(rgba_arr, mode="RGBA").save(rgba_path)

    bundle = load_detector_bundle()
    try:
        blocked = inspect_pipeline_sample(
            rgba_path,
            bundle=bundle,
            mode="custom_upload",
            prompt="portrait",
            run_detector_if_rejected=False,
        )
        assert blocked.image_512_rgb.shape == (512, 512, 3)
        assert blocked.image_512_rgb.dtype == np.uint8
        assert blocked.gate_passed is False
        assert any(
            ("No valid face" in r or "No face" in r or "geometry" in r.lower() or "aspect" in r.lower())
            for r in blocked.rejection_reasons
        )
        assert blocked.detector_scores == {}

        forced = inspect_pipeline_sample(
            rgba_arr,
            bundle=bundle,
            mode="custom_upload",
            prompt="portrait",
            run_detector_if_rejected=True,
            compute_heatmaps=False,
        )
        assert forced.gate_passed is False
        assert set(forced.detector_scores.keys()) == {"M0", "M2*", "M3"}
        assert all(0.0 <= val <= 1.0 for val in forced.detector_scores.values())
    finally:
        unload_detector_bundle(bundle)


def test_inspect_offline_clip_and_unpaired_rv51_graceful_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLIP Hub/cache failure degrades to clip_score=None and unpaired rv51_ood skips pair metrics."""
    import src.analysis.ui_showcase as ui_mod
    import src.common.metrics as metrics_mod

    class FailingClipScorer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def score(self, *args: object, **kwargs: object) -> float:
            raise OSError("Simulated offline HF Hub failure")

        def close(self) -> None:
            pass

    monkeypatch.setattr(ui_mod, "ClipScorer", FailingClipScorer)
    monkeypatch.setattr(metrics_mod, "ClipScorer", FailingClipScorer)

    img = np.full((512, 512, 3), 128, dtype=np.uint8)
    res = inspect_pipeline_sample(
        img,
        bundle=None,
        source_image=None,
        mode="rv51_ood",
        prompt="studio headshot portrait",
    )
    assert res.measured["clip_score"] is None
    assert res.measured["delta_e_lab"] is None
    assert res.measured["seam_diff"] is None
    assert res.before_after_html is None
    assert len(res.warnings) >= 1


def test_launch_live_sd_worker_raises_contextual_error_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify launch_live_sd_worker raises RuntimeError with stderr context and cleans failed output dir."""

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="RuntimeError: CUDA out of memory while allocating VAE",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        launch_live_sd_worker(
            mode="text2img",
            prompt="a studio portrait of a person",
            output_root=tmp_path,
        )

    assert list(tmp_path.iterdir()) == []
