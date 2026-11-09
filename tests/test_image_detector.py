"""Unit tests for dinoViT detector, hard-sample mining, and closed-loop fine-tuning."""

import json
from pathlib import Path
import pytest
import torch

from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import (
    compare_baseline_vs_finetuned,
    evaluate_image_detector,
)
from src.pipelines.mine_hard_samples import mine_hard_samples
from src.pipelines.train_image_detector import (
    combine_training_manifests,
    train_deepfake_image_detector,
)


@pytest.mark.smoke
def test_deepfake_image_detector_forward_dinovit() -> None:
    """Smoke test: verify DeepfakeImageDetector forward pass with dinoViT backbone."""
    model = DeepfakeImageDetector(backbone="dino_vit", num_classes=1)
    model.eval()
    x = torch.rand(3, 3, 64, 64)  # (B, 3, H, W)
    with torch.no_grad():
        logits = model(x)
        feats = model.extract_features(x)
        probs = model.predict_proba(x)

    assert logits.shape == (3, 1)
    assert logits.dtype == torch.float32
    assert feats.shape == (3, 192)
    norms = torch.linalg.norm(feats, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    assert probs.shape == (3, 1)
    assert 0.0 <= float(probs.min().item()) and float(probs.max().item()) <= 1.0


def test_dinovit_freeze_and_partial_unfreeze() -> None:
    """Verify backbone freezing and partial unfreezing of last N transformer blocks."""
    model = DeepfakeImageDetector(backbone="dino_vit", num_classes=1, freeze_extractor=True)
    assert all(not p.requires_grad for p in model.extractor.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())

    model.unfreeze_backbone(unfreeze_last_n_blocks=1)
    assert all(not p.requires_grad for p in model.extractor.blocks[0].parameters())
    assert all(p.requires_grad for p in model.extractor.blocks[-1].parameters())


def test_dinov2_small_freeze_and_partial_unfreeze() -> None:
    """Verify MetaDinoV2Wrapper partial unfreezing enables gradients on last block."""
    model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, freeze_extractor=True, pretrained=False)
    assert all(not p.requires_grad for p in model.extractor.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())

    model.unfreeze_backbone(unfreeze_last_n_blocks=1)
    layers = list(model.extractor.backbone.encoder.layer)
    assert all(not p.requires_grad for p in layers[0].parameters())
    assert any(p.requires_grad for p in layers[-1].parameters())



def test_deepfake_image_detector_forward_simple_and_resnet() -> None:
    """Test DeepfakeImageDetector forward pass with simple CNN and ResNet-18 backbones."""
    for bb in ("simple_cnn", "resnet18"):
        model = DeepfakeImageDetector(backbone=bb, num_classes=1)
        x = torch.rand(2, 3, 64, 64)
        logits = model(x)
        assert logits.shape == (2, 1)


def test_dinov2_pretrained_wrapper() -> None:
    """Verify loading real Meta DINOv2 backbone or factory."""
    model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
    assert model.feature_dim == 384
    assert model.backbone_name == "dinov2_small"


def test_mine_hard_samples_reject_vs_bypass(tmp_path: Path) -> None:
    """Test Phase 2 hard-sample mining separating Rejected vs Bypassed synthetic DF images."""
    model = DeepfakeImageDetector(backbone="dino_vit", num_classes=1)
    synthetic_meta = [
        ImageMetadata(
            image_id=f"syn_{i:03d}",
            label=1,
            manipulation_type="sd_inpaint",
            source_real_id=f"real_{i:03d}",
            data_source="synthetic",
        )
        for i in range(6)
    ]
    cache = {f"syn_{i:03d}": torch.rand(3, 64, 64) for i in range(6)}

    # Zero classifier weights and set negative bias so P(fake) < 0.5 -> Bypassed Hard Samples
    with torch.no_grad():
        for p in model.classifier.parameters():
            p.zero_()
        model.classifier[-1].bias.fill_(-0.5)

    hard_manifest = tmp_path / "hard_samples.jsonl"
    annotated_manifest = tmp_path / "annotated_syn.jsonl"
    report = mine_hard_samples(
        model=model,
        synthetic_metadata=synthetic_meta,
        cache=cache,
        bypass_threshold=0.5,
        batch_size=3,
        device="cpu",
        annotated_manifest_path=annotated_manifest,
        hard_manifest_path=hard_manifest,
    )

    assert report.total_synthetic == 6
    assert report.bypassed_hard_count == 6
    assert report.rejected_count == 0
    assert report.bypass_rate == pytest.approx(1.0)
    assert len(report.hard_sample_ids) == 6
    assert hard_manifest.exists()
    assert annotated_manifest.exists()


def test_closed_loop_combine_and_finetune_dinovit(tmp_path: Path) -> None:
    """Test Phase 3 combining Data Baseline + Synthetic Data + Hard Samples and fine-tuning dinoViT."""
    baseline_path = tmp_path / "baseline.jsonl"
    syn_path = tmp_path / "synthetic.jsonl"
    hard_path = tmp_path / "hard.jsonl"

    baseline_data = [
        ImageMetadata(
            image_id=f"base_{i}",
            label=i % 2,
            manipulation_type="real" if i % 2 == 0 else "deepfake_baseline",
            source_real_id=f"id_{i}",
            data_source="baseline",
        )
        for i in range(8)
    ]
    syn_data = [
        ImageMetadata(
            image_id=f"syn_{i}",
            label=1,
            manipulation_type="sd_text2img",
            source_real_id=f"id_{i}",
            data_source="synthetic",
        )
        for i in range(4)
    ]
    hard_data = [
        ImageMetadata(
            image_id=f"hard_{i}",
            label=1,
            manipulation_type="sd_inpaint",
            source_real_id=f"id_{i}",
            is_hard_sample=True,
            baseline_score=0.25,
            data_source="hard_sample",
        )
        for i in range(4)
    ]

    for path, items in [
        (baseline_path, baseline_data),
        (syn_path, syn_data),
        (hard_path, hard_data),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.to_dict()) + "\n")

    combined = combine_training_manifests(
        baseline_manifest=baseline_path,
        synthetic_manifest=syn_path,
        hard_sample_manifest=hard_path,
    )
    assert len(combined) == 16
    assert sum(1 for m in combined if m.is_hard_sample) == 4

    cache = {m.image_id: torch.rand(3, 64, 64) for m in combined}
    dm = DeepfakeImageDataModule(
        metadata_list=combined,
        cache=cache,
        batch_size=4,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
        hard_sample_oversample_factor=2,
    )
    dm.setup()

    baseline_model = DeepfakeImageDetector(backbone="dino_vit", num_classes=1, freeze_extractor=True)
    finetuned_model = DeepfakeImageDetector(backbone="dino_vit", num_classes=1)

    ckpt_dir = tmp_path / "ckpts"
    val_metrics = train_deepfake_image_detector(
        model=finetuned_model,
        datamodule=dm,
        num_epochs=1,
        learning_rate=1e-3,
        hard_sample_loss_weight=2.0,
        unfreeze_last_n_blocks=1,
        checkpoint_dir=ckpt_dir,
        device="cpu",
    )
    assert "roc_auc" in val_metrics
    assert (ckpt_dir / "best_model.pt").exists()

    comparison = compare_baseline_vs_finetuned(
        baseline_model=baseline_model,
        finetuned_model=finetuned_model,
        dataloader=dm.test_dataloader(),
        device="cpu",
    )
    assert "deltas" in comparison
    assert "roc_auc_gain" in comparison["deltas"]
    assert "hard_sample_recall_gain" in comparison["deltas"]
    assert "bypass_rate_reduction" in comparison["deltas"]
