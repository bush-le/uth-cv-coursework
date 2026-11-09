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
    """Zero-Mock Invariant: Running mining pipeline without --checkpoint or --allow-untrained must raise ValueError."""
    import sys
    from src.pipelines.mine_hard_samples import main as mine_main

    monkeypatch.setattr(sys, "argv", ["mine_hard_samples.py"])
    with pytest.raises(ValueError, match="Refusing to mine hard samples with uninitialized/untrained weights"):
        mine_main()


def test_mine_hard_samples_cli_loads_checkpoint(tmp_path, monkeypatch):
    """Verify that mining CLI properly loads weights from checkpoint."""
    import sys
    import torch
    from src.models.detector.image_detector import DeepfakeImageDetector
    from src.pipelines.mine_hard_samples import main as mine_main

    detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
    ckpt_file = tmp_path / "dummy_baseline.pt"
    torch.save({"model_state_dict": detector.state_dict()}, ckpt_file)

    out_annotated = tmp_path / "annotated.jsonl"
    out_hard = tmp_path / "hard.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_hard_samples.py",
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
    """Verify that mining CLI works with --method feature_similarity without requiring a checkpoint."""
    import sys
    from src.pipelines.mine_hard_samples import main as mine_main

    out_annotated = tmp_path / "annotated_sim.jsonl"
    out_hard = tmp_path / "hard_sim.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mine_hard_samples.py",
            "--method",
            "feature_similarity",
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




