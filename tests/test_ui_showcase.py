"""Smoke and unit tests for src.analysis.ui_showcase (overlays, thresholds, DINOv2 heatmap)."""

from __future__ import annotations

import json
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
    get_golden_presets,
    inspect_pipeline_sample,
    launch_live_sd_batch_worker,
    launch_live_sd_worker,
    load_curated_showcase_catalog,
    load_detector_bundle,
    render_face_audit_overlay,
    render_latent_noise_preview,
    sample_random_batch_configs,
    sample_random_live_config,
    score_image_with_bundle,
    unload_detector_bundle,
    evaluate_batch_with_bundle,
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


def test_catalog_entries_passed_are_strictly_held_out_and_cover_all_tabs() -> None:
    """Verify UI catalog covers all 4 tabs with strictly held-out passed items, failed items, and cheek_fft_ratio."""
    catalog_path = Path("experiments/demo_runs/ui_showcase_catalog.json")
    if not catalog_path.is_file() or not Path("data/raw/faces/ffhq").is_dir():
        pytest.skip("Required UI showcase catalog or local data/ image directories missing.")
    catalog = load_curated_showcase_catalog(catalog_path)
    assert set(catalog.keys()) == {"sd_text2img", "sd_img2img", "sd_inpaint", "ood_and_real"}

    for tab in ("sd_text2img", "sd_img2img", "sd_inpaint"):
        entries = catalog[tab]
        passed = [e for e in entries if e.gate_passed]
        failed = [e for e in entries if not e.gate_passed]
        assert len(passed) >= 1, f"Expected passed entries in {tab}"
        assert len(failed) >= 1, f"Expected failed entries in {tab}"

        for entry in passed:
            assert entry.split_provenance in {"master_frozen_test", "cumulative_challenge_test"}
            assert "d0_train_images" not in entry.image_path

        for entry in entries:
            assert set(entry.scores.keys()) >= {"M0", "M2*", "M3"}
            assert all(0.0 <= float(v) <= 1.0 for v in entry.scores.values())
            assert Path(entry.image_path).is_file(), f"Missing image: {entry.image_path}"
            if entry.source_image_path is not None:
                assert Path(entry.source_image_path).is_file(), f"Missing source: {entry.source_image_path}"
            if entry.measured.get("face_detected") is True:
                assert entry.cheek_fft_ratio is not None

    ood_real_entries = catalog["ood_and_real"]
    ood = [e for e in ood_real_entries if e.mode == "rv51_ood"]
    real = [e for e in ood_real_entries if e.label == 0]
    assert len(ood) >= 1
    assert len(real) >= 1
    assert all(e.split_provenance == "unseen_rv51_test" for e in ood)
    assert all(
        e.mode == "real_ffhq" and e.split_provenance in {"master_frozen_test", "cumulative_challenge_test"}
        for e in real
    )
    for entry in ood_real_entries:
        assert set(entry.scores.keys()) >= {"M0", "M2*", "M3"}
        assert all(0.0 <= float(v) <= 1.0 for v in entry.scores.values())
        assert Path(entry.image_path).is_file(), f"Missing image: {entry.image_path}"
        if entry.measured.get("face_detected") is True:
            assert entry.cheek_fft_ratio is not None


def test_golden_presets_cover_all_four_presentation_scenarios() -> None:
    """Verify get_golden_presets returns 4 scenario presets backed by existing catalog entries and live scores."""
    catalog = load_curated_showcase_catalog()
    by_id = {e.sample_id: e for entries in catalog.values() for e in entries}

    presets = get_golden_presets(catalog)
    assert len(presets) == 4
    tabs = {p["tab"] for p in presets}
    assert tabs == {"sd_text2img", "sd_img2img", "sd_inpaint", "ood_and_real"}

    labels_joined = " | ".join(str(p.get("label", "")) for p in presets)
    for expected_kw in ("Text2Img", "Img2Img", "Inpaint", "OOD RV5.1 & Real FFHQ"):
        assert expected_kw in labels_joined

    for preset in presets:
        primary_id = preset["primary_sample_id"]
        secondary_id = preset["secondary_sample_id"]
        assert primary_id in by_id
        assert secondary_id in by_id
        assert by_id[primary_id].tab == preset["tab"]
        assert by_id[secondary_id].tab == preset["tab"]

        metrics = preset["checkpoint_metrics"]
        assert set(metrics.keys()) >= {"M0", "M2*", "M3"}
        assert all(0.0 <= float(v) <= 1.0 for v in metrics.values())


def test_sd15_pregen_200_scores_match_live_checkpoints() -> None:
    """Verify sd15_pregen_200/scored_manifest.jsonl contains all 200 items and matches live M0/M2*/M3 scores."""
    scored_path = Path("experiments/demo_runs/sd15_pregen_200/scored_manifest.jsonl")
    assert scored_path.is_file()

    rows = [json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 200
    by_id = {r["image_id"]: r for r in rows}

    assert "fake_ffhq_00500_4" in by_id
    assert "fake_ffhq_00032_1" in by_id
    assert by_id["fake_ffhq_00500_4"]["is_curated"] is True
    assert by_id["fake_ffhq_00032_1"]["is_curated"] is False

    target_rec = by_id["fake_ffhq_00500_4"]
    img_path = Path(target_rec["file_path"])
    assert img_path.is_file()

    ckpts = [
        Path("experiments/checkpoints/baseline_d0_best.pt"),
        Path("experiments/checkpoints/active_learning_m2_best.pt"),
        Path("experiments/checkpoints/active_learning_m3_best.pt"),
    ]
    if not all(p.is_file() for p in ckpts):
        pytest.skip("Required real checkpoints missing locally.")

    bundle = load_detector_bundle()
    try:
        with Image.open(img_path) as raw_im:
            rgb_512 = np.asarray(
                raw_im.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
        live_scores = score_image_with_bundle(bundle, rgb_512)
    finally:
        unload_detector_bundle(bundle)

    rec_scores = target_rec["detector_scores"]
    key_map = {
        "M0": "M0 (Baseline)",
        "M2*": "M2* (Active R2)",
        "M3": "M3 (Champion)",
    }
    for short_key, full_key in key_map.items():
        recorded = float(rec_scores.get(short_key, rec_scores.get(full_key)))
        assert abs(recorded - live_scores[short_key]) <= 1e-3


@pytest.mark.smoke
def test_streamlit_app_boots_renders_4_modes_and_handles_sliders() -> None:
    """Verify Streamlit app boots, renders 4 modes, Golden Presets, global Scorecard, and handles sliders."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "scripts" / "app_streamlit.py"
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised exception on initial boot: {at.exception}"

    button_labels = [str(b.label) for b in at.button]
    for mode_kw in ("Text2Img", "Img2Img", "Inpaint", "OOD RV5.1 & Real FFHQ"):
        assert any(mode_kw in lbl for lbl in button_labels), f"Missing mode button '{mode_kw}' in {button_labels}"

    for preset_kw in ("Ca 1:", "Ca 2:", "Ca 3:", "Ca 4:"):
        assert any(preset_kw in lbl for lbl in button_labels), (
            f"Missing Golden Preset button '{preset_kw}' in {button_labels}"
        )

    expander_labels = [str(e.label) for e in at.expander]
    assert any("Scorecard" in lbl or "Proof of Utility" in lbl for lbl in expander_labels), (
        f"Missing global Scorecard expander in {expander_labels}"
    )

    sliders_by_label = {str(s.label): s for s in at.slider}
    sharp_key = next((k for k in sliders_by_label if "Min Sharpness" in k), None)
    fft_key = next((k for k in sliders_by_label if "Min Cheek FFT" in k), None)
    tau_key = next((k for k in sliders_by_label if "τ" in k or "Ngưỡng" in k), None)
    assert sharp_key is not None, f"Missing Min Sharpness slider in {list(sliders_by_label)}"
    assert fft_key is not None, f"Missing Min Cheek FFT slider in {list(sliders_by_label)}"
    assert tau_key is not None, f"Missing tau slider in {list(sliders_by_label)}"

    sliders_by_label[sharp_key].set_value(2.0)
    sliders_by_label[fft_key].set_value(0.008)
    sliders_by_label[tau_key].set_value(0.60)
    at.run()
    assert not at.exception, f"AppTest raised exception after slider updates: {at.exception}"

    img2img_btn = next((b for b in at.button if "Img2Img" in str(b.label) and "Ca " not in str(b.label)), None)
    assert img2img_btn is not None, f"Missing Img2Img mode button in {[str(b.label) for b in at.button]}"
    img2img_btn.click()
    at.run()
    assert not at.exception, f"AppTest raised exception after switching to Img2Img: {at.exception}"
    sliders_after = {str(s.label): s for s in at.slider}
    clip_key = next((k for k in sliders_after if "Min CLIP" in k), None)
    assert clip_key is not None, f"Missing Min CLIP slider in {list(sliders_after)}"
    assert float(sliders_after[clip_key].value) == pytest.approx(0.12, abs=1e-6)


@pytest.mark.smoke
def test_metric_progress_ratio_matches_curation_metric_names() -> None:
    """Verify _metric_progress_ratio handles all 6 CurationMetricItem names emitted by _build_curation_metric_items."""
    from scripts.app_streamlit import _metric_progress_ratio

    items = [
        CurationMetricItem(name="sharpness", label="Sharpness", value=5.2, threshold=1.3, passed=True),
        CurationMetricItem(name="cheek_fft", label="Cheek FFT", value=0.0125, threshold=0.005, passed=True),
        CurationMetricItem(name="clip_score", label="CLIP", value=0.20, threshold=0.20, passed=True),
        CurationMetricItem(name="geometry", label="Geometry", value=0.95, threshold=0.60, passed=True),
        CurationMetricItem(name="geometry", label="Geometry Fail", value=0.0, threshold=0.60, passed=False),
        CurationMetricItem(name="delta_e_lab", label="Delta E", value=9.0, threshold=18.0, passed=True),
        CurationMetricItem(name="seam_diff", label="Seam", value=17.5, threshold=35.0, passed=True),
    ]
    ratios = [_metric_progress_ratio(it) for it in items]
    assert ratios[0] == pytest.approx(1.0)
    assert ratios[1] == pytest.approx(1.0)
    assert ratios[2] == pytest.approx(0.5)
    assert ratios[3] == pytest.approx(1.0)
    assert ratios[4] == pytest.approx(0.25)
    assert ratios[5] == pytest.approx(0.75)
    assert ratios[6] == pytest.approx(0.75)


@pytest.mark.smoke
def test_bg_delta_e_and_paired_upload_and_weights_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify bg_delta_e gating, paired upload metrics, weights_only=True checkpoint loading, and empty-pool guard."""
    # 1. bg_delta_e gating in apply_interactive_thresholds
    measured_bg_fail = {
        "sharpness_score": 15.0,
        "geometry_valid": True,
        "eye_passed": True,
        "fft_high_freq_ratio": 0.04,
        "texture_flag": "normal",
        "cheek_fft_ratio": 0.008,
        "clip_score": 0.25,
        "delta_e_lab": 5.0,
        "bg_delta_e": 14.5,
        "seam_diff": 10.0,
    }
    ok_bg, reasons_bg = apply_interactive_thresholds(
        measured_bg_fail,
        mode="sd_inpaint",
        min_sharpness=1.3,
        min_cheek_fft=0.005,
        min_clip=0.12,
    )
    assert ok_bg is False
    assert any("Background CIELab" in r for r in reasons_bg)

    # 2. Paired upload (sd_img2img with source_image) computes delta_e_lab & structure_corr
    src_img = np.full((512, 512, 3), 100, dtype=np.uint8)
    tgt_img = np.full((512, 512, 3), 115, dtype=np.uint8)
    paired_res = inspect_pipeline_sample(
        tgt_img,
        bundle=None,
        source_image=src_img,
        mode="sd_img2img",
        prompt="",
        compute_heatmaps=False,
    )
    assert paired_res.measured["delta_e_lab"] is not None
    assert paired_res.measured["structure_corr"] is not None

    # 3. load_detector_bundle uses torch.load(..., weights_only=True)
    dummy_model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
    ckpt_path = tmp_path / "dummy.pt"
    torch.save({"model_state_dict": dummy_model.state_dict()}, ckpt_path)

    seen_weights_only: list[object] = []
    orig_load = torch.load

    def spy_load(*args: object, **kwargs: object) -> object:
        seen_weights_only.append(kwargs.get("weights_only"))
        return orig_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)
    loaded_bundle = load_detector_bundle(
        checkpoint_map={"M0": str(ckpt_path)},
        preferred_device="cpu",
    )
    try:
        assert seen_weights_only == [True]
    finally:
        unload_detector_bundle(loaded_bundle)

    # 4. get_golden_presets raises ValueError on empty pool
    with pytest.raises(ValueError, match="empty"):
        get_golden_presets({"sd_text2img": [], "sd_img2img": [], "sd_inpaint": [], "ood_and_real": []})


def test_streamlit_live_sd_submit_updates_min_clip_without_widget_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Submitting Live SD 1.5 form must not raise StreamlitAPIException on min_clip."""
    from streamlit.testing.v1 import AppTest
    import src.analysis.ui_showcase as ui_showcase

    out_png = tmp_path / "live_out.png"
    Image.new("RGB", (512, 512), color=(128, 128, 128)).save(out_png)

    def fake_launch_worker(**kwargs: object) -> dict[str, object]:
        return {
            "mode": kwargs.get("mode", "text2img"),
            "prompt": kwargs.get("prompt", "test prompt"),
            "seed": kwargs.get("seed", 42),
            "steps": 20,
            "guidance_scale": 6.2,
            "strength": 0.30,
            "mask_region": kwargs.get("mask_region", "lips"),
            "output_path": str(out_png),
            "latency_ms": 123.4,
        }

    monkeypatch.setattr(ui_showcase, "launch_live_sd_worker", fake_launch_worker)

    app_path = Path(__file__).resolve().parents[1] / "scripts" / "app_streamlit.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception

    submit_btn = next(
        btn for btn in at.button if "Sinh thử 1 ảnh Live (SD 1.5)" in str(btn.label)
    )
    submit_btn.click().run()
    assert not at.exception
    assert not any("Sinh ảnh Live SD 1.5 thất bại" in str(e.value) for e in at.error)
    assert at.session_state["custom_target"]["kind"] == "live_sd"
    assert at.session_state["min_clip"] == pytest.approx(0.20)


@pytest.mark.smoke
def test_sample_random_live_and_batch_configs_uses_prompt_grammar_and_handles_missing_ffhq(
    tmp_path: Path,
) -> None:
    """Verify random single/batch samplers use PromptGrammarEngine and handle missing FFHQ gracefully."""
    ffhq_png = tmp_path / "ffhq_sample.png"
    Image.new("RGB", (64, 64), color=(140, 120, 110)).save(ffhq_png)

    catalog = {
        "sd_text2img": [],
        "sd_img2img": [],
        "sd_inpaint": [],
        "ood_and_real": [
            ShowcaseCatalogEntry(
                sample_id="real_001",
                mode="real_ffhq",
                tab="ood_and_real",
                image_path=str(ffhq_png),
                label=0,
            )
        ],
    }

    cfg = sample_random_live_config(catalog, mode="text2img", seed=123)
    assert set(cfg.keys()) == {"mode", "prompt", "seed", "source_image", "mask_region"}
    assert cfg["mode"] == "text2img"
    assert isinstance(cfg["prompt"], str) and len(cfg["prompt"].strip()) > 0
    assert cfg["seed"] == 123
    assert cfg["source_image"] is None

    batch = sample_random_batch_configs(catalog, count=5, mode="mixed", base_seed=77)
    assert len(batch) == 5
    assert [item["mode"] for item in batch] == ["text2img", "img2img", "inpaint", "text2img", "img2img"]
    assert [item["seed"] for item in batch] == [77, 78, 79, 80, 81]
    assert batch[0]["source_image"] is None
    assert batch[1]["source_image"] == str(ffhq_png)
    assert batch[2]["source_image"] == str(ffhq_png)

    empty_t2i = sample_random_live_config({}, mode="text2img", seed=11)
    assert empty_t2i["mode"] == "text2img"
    assert empty_t2i["source_image"] is None

    empty_batch = sample_random_batch_configs({}, count=3, mode="mixed", base_seed=20)
    assert len(empty_batch) == 3
    assert all(item["mode"] == "text2img" for item in empty_batch)

    with pytest.raises(ValueError):
        sample_random_live_config({}, mode="img2img")


@pytest.mark.smoke
def test_launch_live_sd_batch_worker_mock_and_error_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify launch_live_sd_batch_worker invokes --batch-manifest and cleans run_dir on failure."""
    seen_cmds: list[list[str]] = []

    def fake_run_ok(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(list(cmd))
        out_idx = cmd.index("--output-dir") + 1
        out_dir = Path(cmd[out_idx])
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "output_dir": str(out_dir),
            "total_items": 1,
            "items": [
                {
                    "item_index": 0,
                    "mode": "text2img",
                    "prompt": "test prompt",
                    "output_path": str(out_dir / "item_000" / "output.png"),
                }
            ],
        }
        (out_dir / "batch_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run_ok)
    configs = [{"mode": "text2img", "prompt": "test prompt", "seed": 42}]
    res = launch_live_sd_batch_worker(configs, output_root=tmp_path)
    assert len(seen_cmds) == 1
    assert "--batch-manifest" in seen_cmds[0]
    assert res["total_items"] == 1
    assert len(res["items"]) == 1

    def fake_run_fail(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        out_idx = cmd.index("--output-dir") + 1
        out_dir = Path(cmd[out_idx])
        out_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run_fail)
    fail_root = tmp_path / "fail_root"
    with pytest.raises(RuntimeError, match="exit code 1"):
        launch_live_sd_batch_worker(configs, output_root=fail_root)
    assert not fail_root.exists() or list(fail_root.iterdir()) == []


@pytest.mark.smoke
def test_evaluate_batch_with_bundle_computes_funnel_counts_and_precomputed_cache_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify evaluate_batch_with_bundle funnel counts and inspect_pipeline_sample precomputed hook."""
    img1_path = tmp_path / "batch_1.png"
    img2_path = tmp_path / "batch_2.png"
    rng = np.random.default_rng(1234)
    arr1 = rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8)
    arr2 = np.full((512, 512, 3), 128, dtype=np.uint8)
    Image.fromarray(arr1).save(img1_path)
    Image.fromarray(arr2).save(img2_path)

    items = [
        {"mode": "text2img", "prompt": "", "seed": 1, "output_path": str(img1_path)},
        {"mode": "text2img", "prompt": "", "seed": 2, "output_path": str(img2_path)},
    ]
    batch_res = evaluate_batch_with_bundle(
        items,
        bundle=None,
        min_sharpness=1.3,
        min_cheek_fft=0.005,
        min_clip=0.12,
        tau=0.50,
    )
    funnel = batch_res["funnel"]
    assert set(funnel.keys()) == {"generated", "gate_passed", "gate_rejected", "hard_samples", "m3_rescued"}
    assert funnel["generated"] == 2
    assert funnel["gate_passed"] + funnel["gate_rejected"] == 2

    # Now test precomputed cache hook on inspect_pipeline_sample
    import src.analysis.ui_showcase as ui_showcase

    fake_scores = {"M0": 0.25, "M2*": 0.65, "M3": 0.85}
    scorer_calls = 0

    def spy_score_image(bundle: object, image_512_rgb: np.ndarray) -> dict[str, float]:
        nonlocal scorer_calls
        scorer_calls += 1
        return dict(fake_scores)

    monkeypatch.setattr(ui_showcase, "score_image_with_bundle", spy_score_image)

    dummy_model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
    dummy_bundle = DetectorBundle(models={"M0": dummy_model}, device=torch.device("cpu"))

    first_res = inspect_pipeline_sample(
        arr1,
        bundle=dummy_bundle,
        mode="sd_text2img",
        prompt="",
        min_sharpness=1.3,
        min_cheek_fft=0.005,
        min_clip=0.12,
        tau=0.50,
        run_detector_if_rejected=True,
        compute_heatmaps=False,
    )
    assert scorer_calls == 1
    assert first_res.detector_scores == fake_scores

    precomputed_payload = {
        "rgb_512": first_res.image_512_rgb,
        "source_512": first_res.source_512_rgb,
        "mask_512": first_res.mask_512,
        "yunet_face": first_res.yunet_face,
        "cheek_boxes": first_res.cheek_boxes,
        "biometrics": first_res.biometrics,
        "measured_base": dict(first_res.measured),
        "before_after_html": first_res.before_after_html,
        "detector_scores": dict(first_res.detector_scores),
        "heatmaps": dict(first_res.heatmaps),
        "warnings": list(first_res.warnings),
    }

    cached_res = inspect_pipeline_sample(
        arr1,
        bundle=dummy_bundle,
        mode="sd_text2img",
        prompt="",
        min_sharpness=999.0,
        min_cheek_fft=0.005,
        min_clip=0.12,
        tau=0.50,
        run_detector_if_rejected=False,
        compute_heatmaps=False,
        precomputed=precomputed_payload,
    )
    assert cached_res.gate_passed is False
    sharp_metric = next(m for m in cached_res.metrics if m.name == "sharpness")
    assert sharp_metric.threshold == pytest.approx(999.0)
    assert sharp_metric.passed is False
    assert cached_res.detector_scores == fake_scores
    assert scorer_calls == 1
    assert cached_res.is_hard_sample is True


@pytest.mark.smoke
def test_inspect_pipeline_sample_precomputed_empty_scores_runs_on_demand_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify precomputed with empty detector_scores/heatmaps runs on-demand scoring+heatmaps when forced."""
    import src.analysis.ui_showcase as ui_showcase

    rng = np.random.default_rng(77)
    arr = rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8)

    scorer_calls = 0
    heatmap_calls: list[str] = []

    def spy_score_image(bundle: object, image_512_rgb: np.ndarray) -> dict[str, float]:
        nonlocal scorer_calls
        scorer_calls += 1
        return {"M0": 0.22, "M2*": 0.61, "M3": 0.88}

    def spy_heatmap(model: object, image_512_rgb: np.ndarray, device: object) -> dict[str, object]:
        heatmap_calls.append("called")
        return {"heatmap_rgb": np.zeros((512, 512, 3), dtype=np.uint8)}

    monkeypatch.setattr(ui_showcase, "score_image_with_bundle", spy_score_image)
    monkeypatch.setattr(ui_showcase, "compute_dinov2_patch_heatmap", spy_heatmap)

    dummy_model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
    dummy_bundle = DetectorBundle(models={"M0": dummy_model, "M3": dummy_model}, device=torch.device("cpu"))

    base_res = inspect_pipeline_sample(
        arr,
        bundle=dummy_bundle,
        mode="sd_text2img",
        prompt="",
        min_sharpness=999.0,
        min_cheek_fft=0.005,
        min_clip=0.12,
        tau=0.50,
        run_detector_if_rejected=False,
        compute_heatmaps=True,
    )
    assert base_res.gate_passed is False
    assert base_res.detector_scores == {}
    assert base_res.heatmaps == {}
    assert scorer_calls == 0

    precomputed_empty = {
        "rgb_512": base_res.image_512_rgb,
        "source_512": base_res.source_512_rgb,
        "mask_512": base_res.mask_512,
        "yunet_face": base_res.yunet_face,
        "cheek_boxes": base_res.cheek_boxes,
        "biometrics": base_res.biometrics,
        "measured_base": dict(base_res.measured),
        "before_after_html": base_res.before_after_html,
        "detector_scores": {},
        "heatmaps": {},
        "warnings": list(base_res.warnings),
    }

    forced_res = inspect_pipeline_sample(
        arr,
        bundle=dummy_bundle,
        mode="sd_text2img",
        prompt="",
        min_sharpness=999.0,
        min_cheek_fft=0.005,
        min_clip=0.12,
        tau=0.50,
        run_detector_if_rejected=True,
        compute_heatmaps=True,
        precomputed=precomputed_empty,
    )
    assert scorer_calls == 1
    assert len(heatmap_calls) == 2
    assert forced_res.detector_scores == {"M0": 0.22, "M2*": 0.61, "M3": 0.88}
    assert set(forced_res.heatmaps.keys()) == {"M0", "M3"}
    assert forced_res.is_hard_sample is True


@pytest.mark.smoke
def test_streamlit_studio_v2_random_config_auto_batch_and_pipeline_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify Studio v2 semantic bars, Random config button, Auto Batch submission, and 4-stage pipeline ribbon."""
    from streamlit.testing.v1 import AppTest
    import src.analysis.ui_showcase as ui_showcase
    from scripts.app_streamlit import _render_semantic_bar_html

    # 1. Unit check for _render_semantic_bar_html colors: green (#16a34a), red (#dc2626), gray (#64748b)
    pass_html = _render_semantic_bar_html(ratio=0.80, color="#16a34a", label_left="Sharpness", label_right="PASS")
    fail_html = _render_semantic_bar_html(ratio=0.25, color="#dc2626", label_left="Cheek FFT", label_right="FAIL")
    na_html = _render_semantic_bar_html(ratio=0.0, color="#64748b", label_left="Seam", label_right="N/A")
    assert "#16a34a" in pass_html and "80%" in pass_html
    assert "#dc2626" in fail_html and "25%" in fail_html
    assert "#64748b" in na_html and "N/A" in na_html

    # 2. Mock launch_live_sd_batch_worker returning 2 synthetic 512x512 items
    src_png = tmp_path / "batch_src.png"
    out0_png = tmp_path / "item_000_out.png"
    out1_png = tmp_path / "item_001_out.png"
    Image.new("RGB", (512, 512), color=(120, 110, 100)).save(src_png)
    Image.new("RGB", (512, 512), color=(135, 125, 115)).save(out0_png)
    Image.new("RGB", (512, 512), color=(145, 130, 120)).save(out1_png)

    def fake_launch_batch_worker(configs: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
        return {
            "output_dir": str(tmp_path),
            "total_items": 2,
            "items": [
                {
                    "item_index": 0,
                    "mode": "text2img",
                    "prompt": str(configs[0].get("prompt", "random t2i prompt")),
                    "seed": int(configs[0].get("seed", 101)),
                    "steps": 20,
                    "guidance_scale": 6.2,
                    "strength": 0.30,
                    "mask_region": "lips",
                    "output_path": str(out0_png),
                    "latency_ms": 110.0,
                },
                {
                    "item_index": 1,
                    "mode": "img2img",
                    "prompt": str(configs[-1].get("prompt", "random i2i prompt")),
                    "seed": int(configs[-1].get("seed", 102)),
                    "steps": 20,
                    "guidance_scale": 6.2,
                    "strength": 0.30,
                    "mask_region": "lips",
                    "input_path": str(src_png),
                    "output_path": str(out1_png),
                    "latency_ms": 130.0,
                },
            ],
        }

    monkeypatch.setattr(ui_showcase, "launch_live_sd_batch_worker", fake_launch_batch_worker)

    app_path = Path(__file__).resolve().parents[1] / "scripts" / "app_streamlit.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"Initial boot failed: {at.exception}"

    # Verify 4-stage pipeline flow ribbon labels appear in rendered markdown
    all_md = "\n".join(str(m.value) for m in at.markdown)
    for stage_kw in ("Input", "Data Gate", "M0", "M3"):
        assert stage_kw in all_md, f"Missing pipeline flow label '{stage_kw}' in rendered markdown"

    # Click Random config button
    init_prompt = str(at.session_state["live_sd_prompt"])
    init_seed = int(at.session_state["live_sd_seed"])
    random_btn = next(b for b in at.button if "Random cấu hình" in str(b.label))
    random_btn.click().run()
    assert not at.exception, f"Random config click raised: {at.exception}"
    assert (
        str(at.session_state["live_sd_prompt"]) != init_prompt
        or int(at.session_state["live_sd_seed"]) != init_seed
    )

    # Click Auto Batch submit button
    batch_btn = next(b for b in at.button if "Auto Sinh & Kiểm Định Lô" in str(b.label))
    batch_btn.click().run()
    assert not at.exception, f"Auto Batch click raised: {at.exception}"
    batch_res = at.session_state["batch_results"]
    assert batch_res is not None
    assert len(batch_res["items"]) == 2
    assert at.session_state["custom_target"] is not None
    assert at.session_state["custom_target"]["kind"] == "batch_item"


