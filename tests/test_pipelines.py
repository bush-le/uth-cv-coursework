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


def test_mine_hard_samples_enforces_checkpoint_guardrail(monkeypatch):
    """Zero-Mock Invariant: Running mining pipeline with detector method without --checkpoint or --allow-untrained must raise ValueError."""
    import sys
    from src.pipelines.mine_hard_samples import main as mine_main

    monkeypatch.setattr(sys, "argv", ["mine_hard_samples.py", "--method", "detector"])
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





