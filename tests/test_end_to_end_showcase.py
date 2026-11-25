"""Unit tests for ImageCurationGate extended audit integration and unified end-to-end showcase."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.pipelines.curate_image_dataset import ImageCurationGate, curate_image_manifest


@pytest.mark.smoke
def test_curation_gate_extended_audit_attaches_metrics(tmp_path: Path) -> None:
    """ImageCurationGate(extended_audit=True) populates extended fields and verifies mask seam on inpaint."""
    src_path = tmp_path / "src.png"
    syn_path = tmp_path / "syn.png"
    arr = np.full((256, 256, 3), 160, dtype=np.uint8)
    Image.fromarray(arr).save(src_path)
    syn_arr = arr.copy()
    syn_arr[90:160, 90:160, :] = [30, 210, 220]
    Image.fromarray(syn_arr).save(syn_path)

    raw_manifest = tmp_path / "raw.jsonl"
    raw_manifest.write_text(
        json.dumps({"image_id": "real_0", "label": 0, "file_path": str(src_path)}) + "\n",
        encoding="utf-8",
    )
    fake_manifest = tmp_path / "fake.jsonl"
    fake_manifest.write_text(
        json.dumps({
            "image_id": "fake_0",
            "label": 1,
            "manipulation_type": "sd_inpaint",
            "source_real_id": "real_0",
            "file_path": str(syn_path),
            "generation": {"inpaint_mask_box": [0.35, 0.35, 0.63, 0.63]},
        }) + "\n",
        encoding="utf-8",
    )

    out_manifest = tmp_path / "curated.jsonl"
    rep_path = tmp_path / "report.jsonl"
    gate = ImageCurationGate(
        min_sharpness=0.0,
        clip_enabled=False,
        face_backend="none",
        dedup=True,
        extended_audit=True,
    )
    summary = curate_image_manifest(
        input_manifest=fake_manifest,
        output_manifest=out_manifest,
        gate=gate,
        curation_report=rep_path,
        raw_manifest=raw_manifest,
    )
    assert summary["total"] == 1
    rep = json.loads(rep_path.read_text(encoding="utf-8").strip())
    assert rep["fft_high_freq_ratio"] is not None
    assert rep["bilateral_symmetry"] is not None
    assert rep["delta_e_lab"] is not None and rep["delta_e_lab"] > 18.0
    assert rep["seam_diff"] is not None
    assert rep["is_valid"] is False


@pytest.mark.smoke
def test_compute_mode_quotas_balances_three_sd15_modes() -> None:
    """Mode quota splitter divides N images evenly across sd_img2img, sd_inpaint, and sd_text2img."""
    from src.pipelines.run_end_to_end_showcase import compute_mode_quotas

    q200 = compute_mode_quotas(200)
    assert sum(q200.values()) == 200
    assert set(q200.keys()) == {"sd_img2img", "sd_inpaint", "sd_text2img"}
    assert max(q200.values()) - min(q200.values()) <= 1

    q3 = compute_mode_quotas(3)
    assert q3 == {"sd_img2img": 1, "sd_inpaint": 1, "sd_text2img": 1}

    with pytest.raises(ValueError):
        compute_mode_quotas(0)


@pytest.mark.smoke
def test_run_end_to_end_showcase_pipeline_orchestration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unified showcase runs generate -> curate(extended_audit) -> multi-model score -> summary JSON."""
    from src.pipelines.run_end_to_end_showcase import run_end_to_end_showcase

    # Create tiny raw real images
    raw_dir = tmp_path / "raw_imgs"
    raw_dir.mkdir()
    raw_records = []
    for idx in range(3):
        p = raw_dir / f"real_{idx}.png"
        arr = np.full((128, 128, 3), 150 + idx * 12, dtype=np.uint8)
        arr[40:90, 40:90, 0] = 210
        Image.fromarray(arr).save(p)
        raw_records.append({"image_id": f"real_{idx}", "label": 0, "file_path": str(p)})
    raw_manifest = tmp_path / "raw_manifest.jsonl"
    raw_manifest.write_text("".join(json.dumps(r) + "\n" for r in raw_records), encoding="utf-8")

    # Stub run_image_generation and detector scoring so unit test runs offline in <1s
    def _stub_generate(**kwargs):
        out_m = Path(kwargs["output_manifest"])
        out_d = Path(kwargs["output_dir"])
        out_d.mkdir(parents=True, exist_ok=True)
        quotas = kwargs["mode_quotas"]
        fake_recs = []
        i = 0
        for mode_name, count in quotas.items():
            for _ in range(count):
                src_rec = raw_records[i % len(raw_records)]
                img_p = out_d / f"fake_{i}_{mode_name}.png"
                arr = np.array(Image.open(src_rec["file_path"]).convert("RGB"), dtype=np.uint8)
                Image.fromarray(arr).save(img_p)
                fake_recs.append({
                    "image_id": f"fake_{i}",
                    "label": 1,
                    "manipulation_type": mode_name,
                    "source_real_id": src_rec["image_id"],
                    "file_path": str(img_p),
                    "prompt": "portrait",
                    "generation": {"mode": mode_name, "inpaint_mask_box": [0.3, 0.25, 0.7, 0.75]},
                })
                i += 1
        out_m.write_text("".join(json.dumps(r) + "\n" for r in fake_recs), encoding="utf-8")
        return len(fake_recs)

    monkeypatch.setattr("src.pipelines.run_end_to_end_showcase.run_image_generation", _stub_generate)
    monkeypatch.setattr(
        "src.pipelines.run_end_to_end_showcase._score_manifest_with_checkpoints",
        lambda items, checkpoints, config_path, batch_size=16: {
            ckpt_label: [0.25 if idx == 0 else 0.92 for idx, _ in enumerate(items)]
            if ckpt_label == "M0 (Baseline)"
            else [0.95 for _ in items]
            for ckpt_label in checkpoints
        },
    )

    out_dir = tmp_path / "showcase_out"
    summary = run_end_to_end_showcase(
        total_images=3,
        output_dir=out_dir,
        raw_manifest=raw_manifest,
        min_sharpness=0.0,
        face_backend="none",
        extended_audit=True,
    )
    assert summary["requested_images"] == 3
    assert summary["generated_images"] == 3
    assert summary["curated_images"] == 3
    assert summary["hard_samples_count"] == 1
    assert summary["model_metrics"]["M0 (Baseline)"]["recall"] == pytest.approx(2 / 3)
    assert summary["model_metrics"]["M3 (Champion)"]["recall"] == pytest.approx(1.0)
    assert (out_dir / "e2e_summary.json").is_file()
    assert (out_dir / "hard_samples_manifest.jsonl").is_file()


@pytest.mark.smoke
def test_run_end_to_end_showcase_with_pregenerated_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When pregenerated_manifest is supplied, showcase samples N balanced records without calling diffusion."""
    from src.pipelines.run_end_to_end_showcase import run_end_to_end_showcase

    src_dir = tmp_path / "src_imgs"
    src_dir.mkdir()
    raw_recs, pregen_recs = [], []
    for idx, mode_name in enumerate(["sd_img2img", "sd_inpaint", "sd_text2img", "sd_img2img", "sd_inpaint", "sd_text2img"]):
        rp = src_dir / f"real_{idx}.png"
        fp = src_dir / f"fake_{idx}.png"
        arr = np.full((128, 128, 3), 120 + idx * 15, dtype=np.uint8)
        arr[40:90, 40:90, 0] = 210
        Image.fromarray(arr).save(rp)
        Image.fromarray(arr).save(fp)
        raw_recs.append({"image_id": f"real_{idx}", "label": 0, "file_path": str(rp)})
        pregen_recs.append({
            "image_id": f"pre_{idx}",
            "label": 1,
            "manipulation_type": mode_name,
            "source_real_id": f"real_{idx}",
            "file_path": str(fp),
            "prompt": "portrait",
            "generation": {"mode": mode_name, "inpaint_mask_box": [0.3, 0.25, 0.7, 0.75]},
        })

    raw_m = tmp_path / "raw.jsonl"
    raw_m.write_text("".join(json.dumps(r) + "\n" for r in raw_recs), encoding="utf-8")
    pre_m = tmp_path / "pregen.jsonl"
    pre_m.write_text("".join(json.dumps(r) + "\n" for r in pregen_recs), encoding="utf-8")

    # Fail test immediately if diffusion generation were invoked
    monkeypatch.setattr(
        "src.pipelines.run_end_to_end_showcase.run_image_generation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Should not generate when pregenerated_manifest is set")),
    )
    monkeypatch.setattr(
        "src.pipelines.run_end_to_end_showcase._score_manifest_with_checkpoints",
        lambda items, checkpoints, config_path, batch_size=16: {k: [0.9 for _ in items] for k in checkpoints},
    )

    out_dir = tmp_path / "pregen_out"
    summary = run_end_to_end_showcase(
        total_images=3,
        output_dir=out_dir,
        raw_manifest=raw_m,
        pregenerated_manifest=pre_m,
        min_sharpness=0.0,
        face_backend="none",
        extended_audit=True,
    )
    assert summary["source_mode"] == "pregenerated"
    assert summary["generated_images"] == 3
    assert summary["curated_images"] == 3


@pytest.mark.smoke
def test_curate_dataset_forwards_extended_audit_and_groups_reasons(tmp_path: Path) -> None:
    """curate_dataset forwards extended_audit/raw_manifest and _aggregate_quality_stats groups numeric reasons."""
    from src.pipelines.curate_image_dataset import curate_dataset
    from src.pipelines.run_end_to_end_showcase import _aggregate_quality_stats

    src_p = tmp_path / "src.png"
    syn_p = tmp_path / "syn.png"
    arr = np.full((128, 128, 3), 160, dtype=np.uint8)
    arr[40:90, 40:90, 0] = 210
    Image.fromarray(arr).save(src_p)
    Image.fromarray(arr).save(syn_p)

    raw_m = tmp_path / "raw.jsonl"
    raw_m.write_text(json.dumps({"image_id": "r0", "label": 0, "file_path": str(src_p)}) + "\n", encoding="utf-8")
    fake_m = tmp_path / "fake.jsonl"
    fake_m.write_text(
        json.dumps({"image_id": "f0", "label": 1, "manipulation_type": "sd_img2img", "source_real_id": "r0", "file_path": str(syn_p)}) + "\n",
        encoding="utf-8",
    )
    out_m = tmp_path / "curated.jsonl"
    rep_m = tmp_path / "rep.jsonl"
    summary = curate_dataset(
        fake_m,
        out_m,
        report_path=rep_m,
        min_sharpness=0.0,
        face_backend="none",
        extended_audit=True,
        raw_manifest=raw_m,
    )
    assert summary["kept"] == 1
    rep = json.loads(rep_m.read_text(encoding="utf-8").strip())
    assert rep["delta_e_lab"] is not None

    agg = _aggregate_quality_stats(
        [
            {"image_id": "a", "manipulation_type": "sd_inpaint"},
            {"image_id": "b", "manipulation_type": "sd_inpaint"},
            {"image_id": "c", "manipulation_type": "sd_text2img"},
        ],
        [
            {
                "image_id": "a",
                "is_valid": False,
                "clip_score": 0.28,
                "texture_flag": "waxy_plastic_skin",
                "rejection_reasons": ["CIELab color drift 20.12 exceeds 18.00"],
            },
            {
                "image_id": "b",
                "is_valid": False,
                "clip_score": 0.32,
                "texture_flag": "normal",
                "rejection_reasons": [
                    "CIELab color drift 25.44 exceeds 18.00",
                    "Semantic-structure correlation 0.350 below 0.550",
                ],
            },
            {
                "image_id": "c",
                "is_valid": False,
                "clip_score": 0.15,
                "texture_flag": "normal",
                "rejection_reasons": ["CLIP score 0.150 below threshold 0.200"],
            },
        ],
    )
    assert agg["rejection_counts"] == {
        "CIELab color drift threshold": 2,
        "Semantic-structure correlation threshold": 1,
        "CLIP score threshold": 1,
    }
    assert agg["overall"]["plastic_skin_count"] == 1
    assert agg["overall"]["clip_score_mean"] == pytest.approx(0.25)


@pytest.mark.smoke
def test_curate_manifest_supports_rv51_schema_celeba_source_and_dhash_dedup(tmp_path: Path) -> None:
    """curate_image_manifest handles RV5.1 keys (image_path, task, sub_region, source_real_image) and dHash dedup."""
    from src.pipelines.curate_image_dataset import ImageCurationGate, curate_image_manifest

    celeba_dir = tmp_path / "celeba_hq"
    celeba_dir.mkdir()
    src_img = celeba_dir / "0.jpg"
    yy, xx = np.indices((256, 256))
    base_arr = np.stack([
        (110 + (xx // 4) % 70).astype(np.uint8),
        (100 + (yy // 4) % 60).astype(np.uint8),
        np.full((256, 256), 120, dtype=np.uint8),
    ], axis=-1)
    base_arr[90:150, 90:166, :] = 175
    Image.fromarray(base_arr).save(src_img)

    # First RV5.1 inpaint image, distinct same-source img2img variant (shares background -> close dHash!),
    # and an actual near-duplicate re-save of syn1 (1-pixel tweak -> SHA-256 differs, thumbdiff < 2.5)
    syn1 = tmp_path / "0_inpaint.png"
    syn_img2img = tmp_path / "0_img2img.png"
    syn2 = tmp_path / "1_inpaint.png"
    syn1_arr = base_arr.copy()
    syn1_arr[160:190, 90:166, :] = np.clip(syn1_arr[160:190, 90:166, :].astype(int) + 5, 0, 255).astype(np.uint8)
    Image.fromarray(syn1_arr).save(syn1)

    syn_var_arr = base_arr.copy()
    syn_var_arr[110:190, 100:156, :] = np.clip(syn_var_arr[110:190, 100:156, :].astype(int) + 35, 0, 255).astype(np.uint8)
    Image.fromarray(syn_var_arr).save(syn_img2img)

    syn2_arr = syn1_arr.copy()
    syn2_arr[0, 0, 0] = (int(syn2_arr[0, 0, 0]) + 1) % 255  # 1-pixel tweak defeats SHA-256, preserves dHash & thumb
    Image.fromarray(syn2_arr).save(syn2)

    rv_manifest = tmp_path / "rv51_fake.jsonl"
    rv_manifest.write_text(
        json.dumps({"image_path": str(syn1), "task": "inpaint", "sub_region": "mouth", "source_real_image": "0.jpg", "prompt": "smiling"}) + "\n"
        + json.dumps({"image_path": str(syn_img2img), "task": "img2img", "source_real_image": "0.jpg", "prompt": "natural smile"}) + "\n"
        + json.dumps({"image_path": str(syn2), "task": "inpaint", "sub_region": "mouth", "source_real_image": "0.jpg", "prompt": "smiling"}) + "\n",
        encoding="utf-8",
    )

    out_m = tmp_path / "rv51_curated.jsonl"
    rep_m = tmp_path / "rv51_report.jsonl"
    gate = ImageCurationGate(
        min_sharpness=0.0,
        clip_enabled=False,
        face_backend="none",
        dedup=True,
        perceptual_dedup_threshold=4,
        extended_audit=True,
    )
    summary = curate_image_manifest(
        input_manifest=rv_manifest,
        output_manifest=out_m,
        gate=gate,
        curation_report=rep_m,
        source_dirs=[celeba_dir],
    )
    assert summary["total"] == 3
    assert summary["kept"] == 2
    assert summary["rejected"] == 1
    reports = [json.loads(l) for l in rep_m.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert reports[0]["image_id"] == "0_inpaint"
    assert reports[0]["seam_diff"] is not None
    assert reports[1]["image_id"] == "0_img2img"
    assert reports[1]["is_valid"] is True
    assert "duplicate_perceptual_dhash" in reports[2]["rejection_reasons"]


@pytest.mark.smoke
def test_curate_manifest_ignores_static_config_inpaint_box_and_uses_source_mask(tmp_path: Path) -> None:
    """When provenance stores dummy static inpaint_mask_box=[0.3, 0.25, 0.7, 0.75], curation uses source-detected mask."""
    from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask
    from src.pipelines.curate_image_dataset import ImageCurationGate, curate_image_manifest

    ffhq_sample = Path("data/raw/faces/ffhq/00004.png")
    if not ffhq_sample.is_file():
        pytest.skip("FFHQ 00004 sample not available")

    src_u8 = np.array(Image.open(ffhq_sample).convert("RGB").resize((512, 512)), dtype=np.uint8)
    src_t = torch.from_numpy(src_u8.astype(np.float32) / 255.0).permute(2, 0, 1)
    true_mask = generate_dynamic_face_mask(src_t, bbox=None, blur_radius=18, target_region="full")
    # Paint an in-mask modification with feathered blend matching generation (no boundary seam!)
    interior = src_t.clone()
    interior[0] = torch.clamp(interior[0] + 0.08, 0.0, 1.0)
    syn_t = true_mask * interior + (1.0 - true_mask) * src_t
    syn_u8 = np.rint(syn_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    raw_p = tmp_path / "00004.png"
    syn_p = tmp_path / "syn_00004.png"
    Image.fromarray(src_u8).save(raw_p)
    Image.fromarray(syn_u8).save(syn_p)

    raw_m = tmp_path / "raw.jsonl"
    raw_m.write_text(json.dumps({"image_id": "ffhq_00004", "file_path": str(raw_p), "bbox": [0, 0, 1024, 1024]}) + "\n", encoding="utf-8")
    fake_m = tmp_path / "fake.jsonl"
    fake_m.write_text(
        json.dumps(
            {
                "image_id": "fake_00004",
                "manipulation_type": "sd_inpaint",
                "source_real_id": "ffhq_00004",
                "file_path": str(syn_p),
                "generation": {"inpaint_mask_box": [0.3, 0.25, 0.7, 0.75], "inpaint_mask_target": "full"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_m = tmp_path / "curated.jsonl"
    rep_m = tmp_path / "rep.jsonl"
    gate = ImageCurationGate(min_sharpness=0.0, clip_enabled=False, face_backend="opencv_haar", dedup=False, extended_audit=True)
    summary = curate_image_manifest(fake_m, out_m, gate, curation_report=rep_m, raw_manifest=raw_m)
    assert summary["kept"] == 1
    rep = json.loads(rep_m.read_text(encoding="utf-8").strip())
    assert rep["seam_diff"] is not None
    assert rep["seam_diff"] < 15.0


@pytest.mark.smoke
def test_curate_manifest_batch_flush_isolates_single_item_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception evaluating one image in a flush batch must not poison remaining items in the batch."""
    from src.pipelines.curate_image_dataset import ImageCurationGate, curate_image_manifest

    p1 = tmp_path / "good1.png"
    p2 = tmp_path / "bad.png"
    p3 = tmp_path / "good2.png"
    arr = np.full((64, 64, 3), 150, dtype=np.uint8)
    for p in (p1, p2, p3):
        Image.fromarray(arr).save(p)

    m_in = tmp_path / "in.jsonl"
    m_in.write_text(
        json.dumps({"image_id": "good1", "file_path": str(p1)}) + "\n"
        + json.dumps({"image_id": "bad", "file_path": str(p2)}) + "\n"
        + json.dumps({"image_id": "good2", "file_path": str(p3)}) + "\n",
        encoding="utf-8",
    )
    gate = ImageCurationGate(min_sharpness=0.0, clip_enabled=False, face_backend="none", dedup=False)
    orig_eval = gate.evaluate_image

    def _flaky_eval(image, **kwargs):
        if kwargs.get("image_id") == "bad":
            raise RuntimeError("simulated single-item failure")
        return orig_eval(image, **kwargs)

    monkeypatch.setattr(gate, "evaluate_image", _flaky_eval)
    out_m = tmp_path / "out.jsonl"
    rep_m = tmp_path / "rep.jsonl"
    summary = curate_image_manifest(m_in, out_m, gate, curation_report=rep_m)
    assert summary["total"] == 3
    assert summary["kept"] == 2
    assert summary["rejected"] == 1
    reps = [json.loads(l) for l in rep_m.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["image_id"] for r in reps] == ["good1", "bad", "good2"]
    assert reps[1]["rejection_reasons"] == ["eval_error: simulated single-item failure"]

    # When clip_enabled=True but prompt is empty "", CLIP is cleanly skipped without penalizing real images
    gate_clip = ImageCurationGate(min_sharpness=0.0, min_clip_score=0.20, clip_enabled=True, face_backend="none")
    rep_empty_prompt = gate_clip.evaluate_image(torch.full((3, 64, 64), 0.5), prompt="", image_id="real_no_prompt")
    assert rep_empty_prompt.clip_score is None
    assert rep_empty_prompt.is_valid is True

    # Repo-relative config paths resolve cleanly even when launched from notebooks/coursework/
    from src.pipelines.run_end_to_end_showcase import _resolve_repo_path

    monkeypatch.chdir(tmp_path)
    resolved_cfg = _resolve_repo_path("configs/config.yaml")
    assert resolved_cfg.is_file() and resolved_cfg.name == "config.yaml"
