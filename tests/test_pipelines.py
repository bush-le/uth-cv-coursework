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



