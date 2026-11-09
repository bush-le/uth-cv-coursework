"""Orchestration pipeline for the closed-loop, 2-round iterative evolutionary framework.

Coordinates:
- Strict Identity-Disjoint Partitioning & Frozen Test Vault (100% Zero Leakage)
- Round 0: Baseline D0 Training -> Temperature Calibration (M0)
- Round 1: Mining H1 (tau1=0.50, Uncertainty, DINOv2 Diversity) + Random Control R -> Train M1 & M_rand
- Round 2: Adversarial Feedback Synthesis (strength=0.28, micro-inpaint) -> Mining H2 (tau2=0.35 on M1) -> Train M2
- Replay Buffer accumulation: D0 subset of D1 subset of D2 (Prevents catastrophic forgetting)
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.common.config import cfg_get, load_config
from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_schema import ImageMetadata
from src.data.splits import create_identity_partitions
from src.models.detector.calibration import TemperatureScaler

logger = get_logger(__name__)


def assemble_evolutionary_stage(
    base_records: List[ImageMetadata],
    incremental_rounds: List[List[ImageMetadata]],
) -> List[ImageMetadata]:
    """Assemble evolutionary training stage preserving base data and appending hard samples.

    Implements a strict Replay Buffer to eliminate catastrophic forgetting.

    Args:
        base_records: Foundational D0 dataset records (Real + Base Fake).
        incremental_rounds: List of hard sample lists from each round [H1, H2, ...].

    Returns:
        Consolidated list of ImageMetadata for current stage.
    """
    seen_ids = set()
    consolidated: List[ImageMetadata] = []

    # 1. Always retain 100% of base records
    for m in base_records:
        if m.image_id not in seen_ids:
            consolidated.append(deepcopy(m))
            seen_ids.add(m.image_id)

    # 2. Incrementally accumulate hard samples from all completed rounds
    for round_samples in incremental_rounds:
        for m in round_samples:
            if m.image_id not in seen_ids:
                annotated = deepcopy(m)
                annotated.is_hard_sample = True
                annotated.data_source = "hard_sample"
                consolidated.append(annotated)
                seen_ids.add(m.image_id)

    return consolidated


def sample_random_control_subset(
    candidate_pool: List[ImageMetadata],
    sample_size: int,
    seed: int = 42,
) -> List[ImageMetadata]:
    """Uniformly sample random fakes to match hard sample set size for fair ablation.

    Args:
        candidate_pool: Available synthetic candidate records.
        sample_size: Desired number of random samples |R| = |H1|.
        seed: Random seed.

    Returns:
        List of randomly selected ImageMetadata.
    """
    rng = random.Random(seed)
    candidates = [m for m in candidate_pool if m.label == 1]
    k = min(sample_size, len(candidates))
    selected = rng.sample(candidates, k=k)
    out: List[ImageMetadata] = []
    for m in selected:
        annotated = deepcopy(m)
        annotated.data_source = "random_augmented"
        annotated.is_hard_sample = False
        out.append(annotated)
    return out
