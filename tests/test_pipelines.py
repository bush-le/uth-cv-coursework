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


def test_hard_sample_mining_flow_dinov2():
    """Verify hard sample mining flow with Meta DINOv2 pretrained baseline."""
    import torch
    from src.data.image_schema import ImageMetadata
    from src.models.detector.image_detector import DeepfakeImageDetector
    from src.pipelines.mine_hard_samples import mine_hard_samples

    detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
    cache = {f"synth_{i}": torch.rand(3, 64, 64) for i in range(4)}
    metas = [
        ImageMetadata(image_id=f"synth_{i}", label=1, manipulation_type="sd_inpaint")
        for i in range(4)
    ]
    report = mine_hard_samples(
        model=detector,
        synthetic_metadata=metas,
        cache=cache,
        bypass_threshold=0.5,
    )
    assert report.total_synthetic == 4
    assert report.rejected_count + report.bypassed_hard_count == 4
    assert 0.0 <= report.bypass_rate <= 1.0


def test_mine_hard_samples_enforces_checkpoint_guardrail(monkeypatch, tmp_path):
    """Zero-Mock Invariant: Running mining pipeline with detector method without --checkpoint or --allow-untrained must raise ValueError."""
    import sys
    from src.pipelines.mine_hard_samples import main as mine_main

    # Provide an empty config to ensure detector_checkpoint is None
    empty_cfg = tmp_path / "empty_factory.yaml"
    empty_cfg.write_text("paths: {}\n")

    monkeypatch.setattr(
        sys,
        "argv",
        ["mine_hard_samples.py", "--config", str(empty_cfg), "--method", "detector"],
    )
    with pytest.raises(ValueError, match="Refusing to mine hard samples with uninitialized/untrained weights"):
        mine_main()


def test_mine_hard_samples_cli_loads_checkpoint(tmp_path, monkeypatch):
    """Verify that mining CLI properly loads weights from checkpoint."""
    import json
    import sys
    import torch
    from PIL import Image
    from src.models.detector.image_detector import DeepfakeImageDetector
    from src.pipelines.mine_hard_samples import main as mine_main

    detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
    ckpt_file = tmp_path / "dummy_baseline.pt"
    torch.save({"model_state_dict": detector.state_dict()}, ckpt_file)

    img_p = tmp_path / "test_fake.png"
    Image.new("RGB", (64, 64), color="red").save(img_p)
    in_manifest = tmp_path / "curated.jsonl"
    in_manifest.write_text(json.dumps({"image_id": "f1", "label": 1, "file_path": str(img_p)}) + "\n")

    out_annotated = tmp_path / "annotated.jsonl"
    out_hard = tmp_path / "hard.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_hard_samples.py",
            "--method",
            "detector",
            "--input-manifest",
            str(in_manifest),
            "--checkpoint",
            str(ckpt_file),
            "--annotated-manifest",
            str(out_annotated),
            "--hard-manifest",
            str(out_hard),
        ],
    )
    mine_main()
    assert out_annotated.exists()


def test_mine_hard_samples_cli_feature_similarity(tmp_path, monkeypatch):
    """Verify that mining CLI works with --method real_manifold."""
    import json
    import sys
    from PIL import Image
    from src.pipelines.mine_hard_samples import main as mine_main

    img_real = tmp_path / "real.png"
    img_fake = tmp_path / "fake.png"
    Image.new("RGB", (64, 64), color="blue").save(img_real)
    Image.new("RGB", (64, 64), color="blue").save(img_fake)

    raw_manifest = tmp_path / "raw.jsonl"
    curated_manifest = tmp_path / "curated.jsonl"
    raw_manifest.write_text(json.dumps({"image_id": "r1", "label": 0, "file_path": str(img_real)}) + "\n")
    curated_manifest.write_text(json.dumps({"image_id": "f1", "label": 1, "file_path": str(img_fake)}) + "\n")

    out_annotated = tmp_path / "annotated_sim.jsonl"
    out_hard = tmp_path / "hard_sim.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_hard_samples.py",
            "--method",
            "real_manifold",
            "--input-manifest",
            str(curated_manifest),
            "--raw-manifest",
            str(raw_manifest),
            "--threshold",
            "0.99",
            "--annotated-manifest",
            str(out_annotated),
            "--hard-manifest",
            str(out_hard),
        ],
    )
    mine_main()
    assert out_annotated.exists()
    assert out_hard.exists()


def test_generate_pipeline_balanced_mode_distribution(tmp_path):
    """Verify that deepfake generator balances modes equally and supports resume."""
    from src.pipelines.generate_deepfake_images import run_image_generation

    in_manifest = tmp_path / "raw_in.jsonl"
    out_manifest = tmp_path / "fake_out.jsonl"
    out_dir = tmp_path / "synthetic"

    # Create 6 dummy real inputs with actual image files on disk
    import json
    from PIL import Image
    real_img_dir = tmp_path / "real_images"
    real_img_dir.mkdir(parents=True, exist_ok=True)
    with open(in_manifest, "w", encoding="utf-8") as f:
        for i in range(6):
            p = real_img_dir / f"real_{i:04d}.png"
            Image.new("RGB", (64, 64), color=(120, 120, 120)).save(p)
            f.write(json.dumps({"image_id": f"real_{i:04d}", "label": 0, "file_path": str(p)}) + "\n")

    count = run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        num_per_real=1,
        max_samples=6,
        use_mock_fallback=True,
        mode="all",
        resume=True,
    )
    assert count == 6

    with open(out_manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    types = [r["manipulation_type"] for r in records]
    assert types.count("sd_img2img") == 2
    assert types.count("sd_inpaint") == 2
    assert types.count("sd_text2img") == 2

    # Second run with resume=True should reuse existing images without errors
    count_resumed = run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        num_per_real=1,
        max_samples=6,
        use_mock_fallback=True,
        mode="all",
        resume=True,
    )
    assert count_resumed == 6




def test_mine_hard_samples_by_feature_similarity():
    """Verify objective hard sample mining via DINOv2 latent feature cosine similarity."""
    import torch
    from src.data.image_schema import ImageMetadata
    from src.models.detector.image_detector import MetaDinoV2Wrapper
    from src.pipelines.mine_hard_samples import mine_hard_samples_by_feature_similarity

    extractor = MetaDinoV2Wrapper(pretrained=True)

    # Identical image for fake_0 -> high cosine similarity (~1.0) -> Hard sample
    # Distinct pattern vs noise for fake_1 -> cosine similarity ~0.65 (< 0.85) -> Rejected
    img_real0 = torch.arange(70 * 70 * 3, dtype=torch.float32).reshape(3, 70, 70) / (70 * 70 * 3)
    img_fake0 = img_real0.clone()
    img_real1 = img_real0.clone()
    img_fake1 = torch.randn(3, 70, 70)

    cache = {
        "real_0": img_real0,
        "fake_0": img_fake0,
        "real_1": img_real1,
        "fake_1": img_fake1,
    }

    real_metas = [
        ImageMetadata(image_id="real_0", label=0, file_path="real_0.png"),
        ImageMetadata(image_id="real_1", label=0, file_path="real_1.png"),
    ]
    syn_metas = [
        ImageMetadata(image_id="fake_0", label=1, source_real_id="real_0", file_path="fake_0.png"),
        ImageMetadata(image_id="fake_1", label=1, source_real_id="real_1", file_path="fake_1.png"),
    ]

    report = mine_hard_samples_by_feature_similarity(
        extractor=extractor,
        synthetic_metadata=syn_metas,
        real_metadata=real_metas,
        cache=cache,
        similarity_threshold=0.85,
    )

    assert report.total_synthetic == 2
    assert "fake_0" in report.hard_sample_ids
    assert report.bypassed_hard_count == 1
    assert report.rejected_count == 1


def test_generator_skips_img2img_when_source_missing(tmp_path):
    """Zero-Mock: Generator must skip img2img when source image file is missing, never use random noise."""
    import json
    from src.pipelines.generate_deepfake_images import run_image_generation

    in_manifest = tmp_path / "in.jsonl"
    out_manifest = tmp_path / "out.jsonl"
    out_dir = tmp_path / "synthetic"
    in_manifest.write_text(
        json.dumps({"image_id": "r1", "label": 0, "file_path": "non_existent.png"}) + "\n",
        encoding="utf-8",
    )
    count = run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        use_mock_fallback=True,
        mode="img2img",
    )
    assert count == 0


def test_mock_refuses_output_inside_data_dir(tmp_path):
    """Zero-Mock Invariant: Mock generator is strictly forbidden from writing to production data/ directory."""
    from pathlib import Path
    from src.pipelines.generate_deepfake_images import run_image_generation

    with pytest.raises(ValueError, match=r"(?i)mock.*data"):
        run_image_generation(
            input_manifest=tmp_path / "dummy.jsonl",
            output_manifest=Path("data/synthetic/fake_manifest.jsonl"),
            output_dir=Path("data/synthetic"),
            model_name="mock_sd",
            use_mock_fallback=True,
        )


def test_miner_raises_when_manifest_missing(monkeypatch, tmp_path):
    """Zero-Mock Invariant: Miner must raise FileNotFoundError when input manifest is missing, never fabricate data."""
    import sys
    from src.pipelines.mine_hard_samples import main as mine_main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_hard_samples.py",
            "--input-manifest",
            str(tmp_path / "missing_curated.jsonl"),
            "--allow-untrained",
        ],
    )
    with pytest.raises(FileNotFoundError):
        mine_main()


def test_load_yaml_with_defaults_resolves_subconfigs():
    """Verify load_yaml_with_defaults resolves Hydra-style defaults entries in configs/config.yaml."""
    from src.common.config import load_yaml_with_defaults

    cfg = load_yaml_with_defaults("configs/config.yaml")
    assert "detection" in cfg and cfg["detection"]["model"]["spatial_backbone"] == "dinov2_small"
    assert "dataset" in cfg and "generation" in cfg


def test_evaluate_pipeline_requires_verified_checkpoint(tmp_path):
    """Zero-Mock Invariant: evaluate_pipeline must reject missing or None checkpoints."""
    from src.pipelines.evaluate import evaluate_pipeline

    with pytest.raises(ValueError, match="checkpoint_path is required"):
        evaluate_pipeline(config_path="configs/config.yaml", checkpoint_path=None)

    with pytest.raises(FileNotFoundError):
        evaluate_pipeline(config_path="configs/config.yaml", checkpoint_path=str(tmp_path / "missing.pt"))


def test_infer_pipeline_runs_with_verified_checkpoint(tmp_path):
    """Verify infer_pipeline rejects missing checkpoints and runs real RGB inference when valid."""
    import torch
    from PIL import Image
    from src.models.detector.image_detector import DeepfakeImageDetector
    from src.pipelines.infer import infer_pipeline

    with pytest.raises(FileNotFoundError):
        infer_pipeline(
            config_path="configs/config.yaml",
            input_path=str(tmp_path),
            checkpoint_path=str(tmp_path / "nonexistent.pt"),
        )

    det = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
    ckpt_path = tmp_path / "m_test.pt"
    torch.save({"model_state_dict": det.state_dict()}, ckpt_path)

    img_path = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=(100, 120, 140)).save(img_path)

    results = infer_pipeline(
        config_path="configs/config.yaml",
        input_path=str(img_path),
        checkpoint_path=str(ckpt_path),
        threshold=0.5,
    )
    assert len(results) == 1
    assert 0.0 <= results[0]["p_fake"] <= 1.0
    assert results[0]["prediction"] in ("real", "fake")


@pytest.mark.smoke
def test_build_d3_clean_dataset_dynamic_curated_rv51(tmp_path: Path) -> None:
    """Verify build_d3_clean_dataset supports arbitrary curated RV5.1 counts without 626/600 asserts."""
    import json
    from pathlib import Path
    from PIL import Image
    from scripts.build_d3_clean_dataset import build_d3_clean_dataset

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    def _save(name: str) -> str:
        p = img_dir / name
        Image.new("RGB", (32, 32), color=(120, 130, 140)).save(p)
        return str(p)

    train_reals_p = tmp_path / "train_reals.jsonl"
    val_reals_p = tmp_path / "val_reals.jsonl"
    test_reals_p = tmp_path / "test_reals.jsonl"
    d2_train_p = tmp_path / "d2_train.jsonl"
    rv51_cur_p = tmp_path / "rv51_curated.jsonl"
    out_dir = tmp_path / "D3_clean"

    train_reals = [{"image_id": f"tr_{i}", "label": 0, "source_real_id": f"tr_{i}", "file_path": _save(f"tr_{i}.jpg")} for i in range(6)]
    val_reals = [{"image_id": f"va_{i}", "label": 0, "source_real_id": f"va_{i}", "file_path": _save(f"va_{i}.jpg")} for i in range(2)]
    test_reals = [{"image_id": f"te_{i}", "label": 0, "source_real_id": f"te_{i}", "file_path": _save(f"te_{i}.jpg")} for i in range(2)]

    train_reals_p.write_text("".join(json.dumps(r) + "\n" for r in train_reals), encoding="utf-8")
    val_reals_p.write_text("".join(json.dumps(r) + "\n" for r in val_reals), encoding="utf-8")
    test_reals_p.write_text("".join(json.dumps(r) + "\n" for r in test_reals), encoding="utf-8")

    # D2 pool: 2 hard + 2 non-hard (not fixed 629 / 800)
    d2_records = train_reals + [
        {"image_id": "h1", "label": 1, "is_hard_sample": True, "manipulation_type": "sd_inpaint", "source_real_id": "tr_0", "file_path": _save("h1.png")},
        {"image_id": "h2", "label": 1, "is_hard_sample": True, "manipulation_type": "sd_img2img", "source_real_id": "tr_1", "file_path": _save("h2.png")},
        {"image_id": "nh1", "label": 1, "is_hard_sample": False, "manipulation_type": "sd_inpaint", "source_real_id": "tr_2", "file_path": _save("nh1.png")},
        {"image_id": "nh2", "label": 1, "is_hard_sample": False, "manipulation_type": "sd_text2img", "source_real_id": "tr_3", "file_path": _save("nh2.png")},
    ]
    d2_train_p.write_text("".join(json.dumps(r) + "\n" for r in d2_records), encoding="utf-8")

    # Curated RV5.1: 1 test-derived (not 626!), 1 val-derived (not 600!), plus clean train & text2img pools
    rv51_entries = [
        {"image_path": _save("rv_te.png"), "task": "inpaint", "source_real_image": "te_0.jpg"},
        {"image_path": _save("rv_va.png"), "task": "img2img", "source_real_image": "va_0.jpg"},
        {"image_path": _save("rv_tr_inp.png"), "task": "inpaint", "source_real_image": "tr_0.jpg"},
        {"image_path": _save("rv_tr_i2i.png"), "task": "img2img", "source_real_image": "tr_1.jpg"},
        {"file_path": _save("rv_t2i_1.png"), "manipulation_type": "rv51_text2img"},
        {"file_path": _save("rv_t2i_2.png"), "manipulation_type": "rv51_text2img"},
    ]
    rv51_cur_p.write_text("".join(json.dumps(r) + "\n" for r in rv51_entries), encoding="utf-8")

    res = build_d3_clean_dataset(
        train_reals_path=train_reals_p,
        val_reals_path=val_reals_p,
        test_reals_path=test_reals_p,
        d2_train_path=d2_train_p,
        rv51_manifest_path=rv51_cur_p,
        output_dir=out_dir,
        sd_nonhard_quota=1,
        seed=42,
    )

    assert res["train_real"] == 6
    assert res["train_fake"] == 6
    assert res["rv51_test_derived"] == 1
    assert res["rv51_val_derived"] == 1
    assert res["unseen_rv51_test_total"] == 3
    assert res["val_total"] == 3
    assert (out_dir / "train.jsonl").is_file()
    assert (out_dir / "val.jsonl").is_file()
    assert (out_dir / "unseen_rv51_test.jsonl").is_file()


@pytest.mark.smoke
def test_populate_hard_samples_vault_dynamic_counts(tmp_path: Path) -> None:
    """Verify populate_hard_samples_vault writes real computed counts instead of legacy constants."""
    import json
    from pathlib import Path
    from PIL import Image
    from scripts.populate_hard_samples_vault import populate_hard_samples_vault

    src_dir = tmp_path / "src_imgs"
    src_dir.mkdir()
    img_paths = []
    for i in range(4):
        p = src_dir / f"h_{i}.png"
        Image.new("RGB", (32, 32), color=(50 * i, 100, 150)).save(p)
        img_paths.append(str(p))

    r1_p = tmp_path / "r1_hard.jsonl"
    r2_p = tmp_path / "r2_hard.jsonl"
    chal_p = tmp_path / "challenge.jsonl"
    d2_p = tmp_path / "d2_train.jsonl"
    target_dir = tmp_path / "hard_samples"

    r1_p.write_text(
        json.dumps({"image_id": "h0", "label": 1, "manipulation_type": "sd_inpaint", "file_path": img_paths[0], "baseline_score": 0.2})
        + "\n"
        + json.dumps({"image_id": "h1", "label": 1, "manipulation_type": "sd_img2img", "file_path": img_paths[1], "baseline_score": 0.4})
        + "\n",
        encoding="utf-8",
    )
    # Include 1 overlapping image_id ("h1") + 2 new hard samples
    r2_p.write_text(
        json.dumps({"image_id": "h1", "label": 1, "manipulation_type": "sd_img2img", "file_path": img_paths[1], "baseline_score": 0.4})
        + "\n"
        + json.dumps({"image_id": "h2", "label": 1, "manipulation_type": "sd_text2img", "file_path": img_paths[2], "baseline_score": 0.3})
        + "\n"
        + json.dumps({"image_id": "h3", "label": 1, "manipulation_type": "sd_inpaint", "file_path": img_paths[3], "baseline_score": 0.5})
        + "\n",
        encoding="utf-8",
    )
    chal_p.write_text(
        json.dumps({"image_id": "h3", "label": 1, "is_hard_sample": True})
        + "\n"
        + json.dumps({"image_id": "real_pair", "label": 0})
        + "\n",
        encoding="utf-8",
    )
    d2_p.write_text(
        "".join(
            json.dumps({"image_id": hid, "label": 1, "is_hard_sample": True}) + "\n"
            for hid in ("h0", "h1", "h2")
        ),
        encoding="utf-8",
    )

    report = populate_hard_samples_vault(
        r1_path=r1_p,
        r2_path=r2_p,
        chal_path=chal_p,
        d2_train_path=d2_p,
        target_dir=target_dir,
    )

    assert report["total_hard_samples"] == 4
    assert report["allocation"]["round_1_mined"] == 2
    assert report["allocation"]["round_2_mined"] == 3
    assert report["allocation"]["retraining_buffer_D2"] == 3
    assert report["allocation"]["challenge_hard_fakes"] == 1
    assert report["allocation"]["cumulative_challenge_test_vault"] == 2

    readme_text = (target_dir / "README.md").read_text(encoding="utf-8")
    assert "629" not in readme_text
    assert "553" not in readme_text
    assert "187" not in readme_text
    assert "2 mẫu khó" in readme_text
    assert "3 mẫu khó" in readme_text
    assert "3 mẫu)" in readme_text


def test_generate_image_preserves_inpaint_target_on_resume_and_guards_empty_manifest(tmp_path: Path) -> None:
    """Verify inpaint_mask_target persists on resume, triggers regen when changed, and empty input raises ValueError."""
    import json
    from PIL import Image
    from src.pipelines.generate_deepfake_images import run_image_generation

    empty_manifest = tmp_path / "empty.jsonl"
    empty_manifest.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no real image records"):
        run_image_generation(
            input_manifest=empty_manifest,
            output_manifest=tmp_path / "out_empty.jsonl",
            output_dir=tmp_path / "syn_empty",
            model_name="mock_sd",
            use_mock_fallback=True,
        )

    in_manifest = tmp_path / "raw_in.jsonl"
    out_manifest = tmp_path / "fake_out.jsonl"
    out_dir = tmp_path / "syn"
    src_p = tmp_path / "real_0.png"
    Image.new("RGB", (64, 64), color=(110, 120, 130)).save(src_p)
    in_manifest.write_text(json.dumps({"image_id": "real_0", "label": 0, "file_path": str(src_p)}) + "\n")

    run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        use_mock_fallback=True,
        mode_quotas={"sd_inpaint": 1},
        mask_targets=["periocular"],
        resume=True,
    )
    rec1 = json.loads(out_manifest.read_text(encoding="utf-8").strip())
    assert rec1["generation"]["inpaint_mask_target"] == "periocular"

    # Resuming with identical mask_targets must preserve inpaint_mask_target
    run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        use_mock_fallback=True,
        mode_quotas={"sd_inpaint": 1},
        mask_targets=["periocular"],
        resume=True,
    )
    rec2 = json.loads(out_manifest.read_text(encoding="utf-8").strip())
    assert rec2["generation"]["inpaint_mask_target"] == "periocular"

    # Changing mask_targets must update provenance and regenerate cleanly
    run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        use_mock_fallback=True,
        mode_quotas={"sd_inpaint": 1},
        mask_targets=["lips"],
        resume=True,
    )
    rec3 = json.loads(out_manifest.read_text(encoding="utf-8").strip())
    assert rec3["generation"]["inpaint_mask_target"] == "lips"


def test_train_detector_handles_nan_auc_single_class_val(tmp_path: Path) -> None:
    """Verify training still saves finite best checkpoint when validation set has a single class (NaN AUC)."""
    import math
    import torch
    from src.data.image_datamodule import DeepfakeImageDataModule
    from src.data.image_schema import ImageMetadata
    from src.models.detector.image_detector import DeepfakeImageDetector
    from src.pipelines.train_image_detector import train_deepfake_image_detector

    train_recs = [
        ImageMetadata(image_id=f"tr_{i}", label=i % 2, manipulation_type="real" if i % 2 == 0 else "sd_text2img")
        for i in range(4)
    ]
    # Single-class val set guarantees compute_roc_auc returns NaN
    val_recs = [
        ImageMetadata(image_id=f"va_{i}", label=0, manipulation_type="real")
        for i in range(2)
    ]
    cache = {m.image_id: torch.rand(3, 64, 64) for m in (train_recs + val_recs)}

    dm = DeepfakeImageDataModule(
        train_metadata=train_recs,
        val_metadata=val_recs,
        test_metadata=[],
        cache=cache,
        batch_size=2,
        seed=42,
    )
    dm.setup()
    model = DeepfakeImageDetector(backbone="dino_vit", num_classes=1)
    ckpt_dir = tmp_path / "ckpts_nan"

    metrics = train_deepfake_image_detector(
        model=model,
        datamodule=dm,
        num_epochs=1,
        learning_rate=1e-3,
        checkpoint_dir=ckpt_dir,
        device="cpu",
    )
    assert math.isnan(metrics["roc_auc"])
    assert (ckpt_dir / "best_model.pt").is_file()
    saved = torch.load(ckpt_dir / "best_model.pt", map_location="cpu", weights_only=False)
    assert math.isfinite(saved["best_metric"])






