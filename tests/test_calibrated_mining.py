"""Unit tests for temperature calibration and calibrated hard sample mining."""

import numpy as np
import pytest
import torch

from src.data.image_schema import ImageMetadata
from src.models.evaluators.calibrated_mining import CalibratedHardMiner


def test_temperature_calibration():
    """Verify temperature scaling optimizes scalar T to calibrate overconfident logits."""
    miner = CalibratedHardMiner()
    # Logits that are overconfident: true positives have logit=10, true negatives have logit=-10
    logits = torch.tensor([-8.0, -6.0, 6.0, 8.0], dtype=torch.float32)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    temp = miner.calibrate_temperature(logits, labels)
    
    assert temp > 0.01
    assert isinstance(temp, float)
    assert miner.temperature == temp


def test_dual_criterion_hard_sample_mining():
    """Verify miner extracts both False Negatives (bypassed) and Uncertain boundary samples."""
    miner = CalibratedHardMiner(decision_threshold=0.5, uncertainty_window=0.10)
    candidates = [
        ImageMetadata(image_id="f1", label=1, file_path="f1.png"),  # FN: P=0.20
        ImageMetadata(image_id="f2", label=1, file_path="f2.png"),  # Boundary uncertain: P=0.45
        ImageMetadata(image_id="f3", label=1, file_path="f3.png"),  # Boundary uncertain: P=0.55
        ImageMetadata(image_id="f4", label=1, file_path="f4.png"),  # Easy fake: P=0.95
        ImageMetadata(image_id="r1", label=0, file_path="r1.png"),  # Real face: P=0.10 (not a fake)
    ]
    raw_probs = [0.20, 0.45, 0.55, 0.95, 0.10]
    
    hard_samples, stats = miner.mine_hard_samples(candidates, raw_probs)
    
    hard_ids = {h.image_id for h in hard_samples}
    assert "f1" in hard_ids, "False negative f1 must be mined"
    assert "f2" in hard_ids, "Uncertain sample f2 must be mined"
    assert "f3" in hard_ids, "Uncertain sample f3 must be mined"
    assert "f4" not in hard_ids, "Easy fake f4 must NOT be mined"
    assert "r1" not in hard_ids, "Real image r1 must NOT be mined as hard fake"
    
    assert all(h.is_hard_sample for h in hard_samples)
    assert stats["false_negative_count"] >= 1
    assert stats["uncertain_count"] >= 1


def test_perceptual_diversity_deduplication():
    """Verify that near-identical hard sample candidates in embedding space are deduplicated."""
    miner = CalibratedHardMiner(min_cosine_distance=0.08)
    candidates = [
        ImageMetadata(image_id="f1", label=1, file_path="f1.png"),
        ImageMetadata(image_id="f2", label=1, file_path="f2.png"),  # Duplicate of f1
        ImageMetadata(image_id="f3", label=1, file_path="f3.png"),  # Distinct
    ]
    raw_probs = [0.20, 0.22, 0.25]
    
    # f1 and f2 have cosine similarity 0.999 (almost identical)
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.999, 0.001, 0.0], dtype=np.float32)
    emb3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    embeddings = np.stack([emb1, emb2, emb3])
    
    hard_samples, stats = miner.mine_hard_samples(candidates, raw_probs, embeddings=embeddings)
    hard_ids = [h.image_id for h in hard_samples]
    
    assert len(hard_samples) == 2, "Duplicate f2 should have been filtered out"
    assert "f1" in hard_ids
    assert "f3" in hard_ids
    assert stats["perceptual_duplicates_filtered"] == 1
