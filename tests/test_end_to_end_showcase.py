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
