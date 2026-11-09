"""Unit tests for real-manifold hardness evaluation and verified checkpoint loading."""

from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector
from src.models.evaluators.hardness import (
    HardnessReport,
    build_report,
    load_verified_checkpoint,
    real_manifold_scores,
    real_reference_scores,
)


def test_real_manifold_excludes_own_source():
    """Verify that real-manifold distance excludes own source face index."""
    real = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    fake = real[:1].clone()  # Identical to real face 0
    s = real_manifold_scores(fake, real, k=1, exclude_idx=[0])
    # Real 0 is excluded, so it only compares with Real 1 (dot product = 0.0)
    assert torch.isclose(s[0], torch.tensor(0.0))


def test_reference_scores_leave_one_out():
    """Verify that leave-one-out reference scores mask the self-similarity diagonal."""
    real = F.normalize(torch.eye(3), dim=-1)
    # Orthogonal basis: cosine with any other element is 0.0
    ref_s = real_reference_scores(real, k=1)
    assert torch.allclose(ref_s, torch.zeros(3))


def test_load_verified_checkpoint_rejects_missing_head(tmp_path):
    """Zero-Mock Invariant: Checkpoint missing classification head weights must be rejected."""
    model = DeepfakeImageDetector(backbone="dinov2_small", pretrained=False)
    # Strip classifier keys from state_dict
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("classifier") and not k.startswith("head")}
    ckpt_path = tmp_path / "bad_checkpoint.pt"
    torch.save({"model_state_dict": sd}, ckpt_path)

    with pytest.raises(ValueError, match=r"(classifier|head)"):
        load_verified_checkpoint(model, ckpt_path, "cpu")


def test_report_breaks_down_by_mode():
    """Verify HardnessReport computes accurate breakdown per manipulation mode."""
    metas = [
        ImageMetadata(image_id="f1", label=1, manipulation_type="sd_inpaint"),
        ImageMetadata(image_id="f2", label=1, manipulation_type="sd_inpaint"),
        ImageMetadata(image_id="f3", label=1, manipulation_type="sd_text2img"),
        ImageMetadata(image_id="f4", label=1, manipulation_type="sd_text2img"),
    ]
    # Scores: f1=0.8, f2=0.2, f3=0.9, f4=0.1. Tau = 0.5 (higher is harder)
    scores = torch.tensor([0.8, 0.2, 0.9, 0.1])
    rep = build_report(
        metas=metas,
        scores=scores,
        tau=0.5,
        tau_source="test",
        method="real_manifold",
        higher_is_harder=True,
    )
    assert rep.hard_count == 2
    assert rep.hard_rate_by_mode["sd_inpaint"] == 0.5
    assert rep.hard_rate_by_mode["sd_text2img"] == 0.5
    assert rep.hard_rate == pytest.approx(0.5)
    assert rep.hard_ids == ["f1", "f3"]


def test_hard_sample_requires_face_crop_manifold_proximity():
    """Verify face-crop tensor extractor properly crops and resizes facial region."""
    from src.pipelines.mine_hard_samples import crop_face_tensor

    dummy = torch.randn(3, 256, 256)
    # Specified bounding box
    cropped = crop_face_tensor(dummy, bbox=(0.20, 0.20, 0.80, 0.80), target_size=(224, 224))
    assert cropped.shape == (3, 224, 224)

    # Automatic fallback
    auto_cropped = crop_face_tensor(dummy, bbox=None, target_size=(224, 224))
    assert auto_cropped.shape == (3, 224, 224)


def test_diversity_filter_prunes_near_duplicate_embeddings():
    """Verify DINOv2 embedding diversity filter prunes redundant near-identical faces."""
    from src.pipelines.mine_hard_samples import filter_by_embedding_diversity

    # 4 embeddings: 0 and 1 are almost identical (cos > 0.99), 2 and 3 are distinct
    e0 = torch.tensor([1.0, 0.0, 0.0])
    e1 = torch.tensor([0.999, 0.01, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    e3 = torch.tensor([0.0, 0.0, 1.0])
    embeddings = torch.stack([e0, e1, e2, e3])
    scores = [0.10, 0.12, 0.20, 0.30]  # Lower is harder

    selected_indices = filter_by_embedding_diversity(embeddings, scores, max_cosine_sim=0.95)
    # e1 must be pruned because e0 was already chosen and has lower score
    assert 0 in selected_indices
    assert 1 not in selected_indices
    assert 2 in selected_indices
    assert 3 in selected_indices

