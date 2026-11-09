"""Identity-disjoint dataset splitting, benchmark stage construction, and dataset analytics.

Ensures zero identity leakage across train/val/test splits and consistent, reproducible
evaluation across D0 (baseline), D1 (augmented), and D2 (hard-boosted) stages.
"""

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.common.logger import get_logger
from src.data.image_schema import ImageMetadata

logger = get_logger(__name__)


def get_group_key(meta: ImageMetadata) -> str:
    """Extract identity grouping key ensuring parent real faces and synthetic variants share group."""
    if meta.source_real_id:
        return str(meta.source_real_id)
    if meta.demographics and meta.demographics.get("identity_id"):
        return str(meta.demographics["identity_id"])
    return str(meta.image_id)


def assign_identity_splits(
    records: List[ImageMetadata],
    ratios: Dict[str, float],
    seed: int = 42,
) -> Dict[str, str]:
    """Compute identity-disjoint partition assignments (group_key -> split_name) once for all records.

    Stratifies by paired groups (containing synthetic deepfake faces) vs real-only identity groups.

    Args:
        records: Full pool of candidate image metadata records.
        ratios: Split proportion mapping, e.g. {'train': 0.70, 'val': 0.15, 'test': 0.15}.
        seed: Random seed for deterministic shuffling.

    Returns:
        Dictionary mapping each group_key to its assigned split ('train', 'val', or 'test').
    """
    groups: Dict[str, List[ImageMetadata]] = defaultdict(list)
    for m in records:
        groups[get_group_key(m)].append(m)

    paired_keys: List[str] = []
    real_only_keys: List[str] = []

    for k, metas in groups.items():
        if any(m.label == 1 or m.source_real_id is not None for m in metas):
            paired_keys.append(k)
        else:
            real_only_keys.append(k)

    # Sort first for deterministic ordering before RNG shuffle
    paired_keys.sort()
    real_only_keys.sort()

    rng = random.Random(seed)
    rng.shuffle(paired_keys)
    rng.shuffle(real_only_keys)

    train_r = ratios.get("train", 0.70)
    val_r = ratios.get("val", 0.15)
    test_r = ratios.get("test", 0.15)
    total_r = train_r + val_r + test_r
    train_r = train_r / total_r
    val_r = val_r / total_r

    assignment: Dict[str, str] = {}

    for k_list in (paired_keys, real_only_keys):
        n = len(k_list)
        if n == 0:
            continue
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))

        # Ensure non-empty test if requested and enough samples exist
        if n >= 3:
            if n_train >= n:
                n_train = n - 2
            if n_train + n_val >= n:
                n_val = n - n_train - 1
            if n_train < 1:
                n_train = 1
            if n_val < 1:
                n_val = 1

        for k in k_list[:n_train]:
            assignment[k] = "train"
        for k in k_list[n_train : n_train + n_val]:
            assignment[k] = "val"
        for k in k_list[n_train + n_val :]:
            assignment[k] = "test"

    return assignment


def build_stage(
    records: List[ImageMetadata],
    assignment: Dict[str, str],
    stage_spec: Dict[str, Any],
) -> Dict[str, List[ImageMetadata]]:
    """Build train, val, and test splits for a specific benchmark stage.

    Guarantees:
    - Test and validation splits are fixed and identical across stages.
    - Preserves exact data_source provenance without relabeling.
    - Applies oversampling for hard samples in train split when configured.
    """
    train_sources: Set[str] = set(stage_spec.get("train_sources", ["baseline", "synthetic", "hard_sample"]))
    train_modes: List[str] = stage_spec.get("train_modes", ["all"])
    oversample_hard = int(stage_spec.get("oversample_hard", 1))

    train_list: List[ImageMetadata] = []
    val_list: List[ImageMetadata] = []
    test_list: List[ImageMetadata] = []

    for m in records:
        k = get_group_key(m)
        split = assignment.get(k, "train")

        if split == "test":
            test_list.append(m)
        elif split == "val":
            val_list.append(m)
        else:
            # Training split: filter by allowed data sources and modes
            if m.data_source not in train_sources:
                continue

            if "all" not in train_modes and m.label == 1:
                if m.manipulation_type and m.manipulation_type not in train_modes:
                    continue

            if m.is_hard_sample and oversample_hard > 1:
                for _ in range(oversample_hard):
                    train_list.append(m)
            else:
                train_list.append(m)

    labels = set(m.label for m in train_list)
    if len(labels) < 2 and len(train_list) > 0:
        logger.warning(
            "Stage %s train split contains only one class (%s). Check train_sources configuration.",
            stage_spec.get("name"),
            labels,
        )

    return {
        "train": train_list,
        "val": val_list,
        "test": test_list,
    }


def compute_dataset_stats(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate and export comprehensive dataset statistics to paths.stats_file."""
    paths_cfg = cfg.get("paths", {})
    raw_path = Path(paths_cfg.get("raw_manifest", "data/raw/manifest.jsonl"))
    fake_path = Path(paths_cfg.get("fake_manifest", "data/synthetic/fake_manifest.jsonl"))
    curated_path = Path(paths_cfg.get("curated_manifest", "data/synthetic/curated_manifest.jsonl"))
    hard_path = Path(paths_cfg.get("hard_manifest", "data/hard_samples/hard_samples_manifest.jsonl"))
    report_path = Path(paths_cfg.get("curation_report", "data/synthetic/curation_report.jsonl"))
    stats_file = Path(paths_cfg.get("stats_file", "data/dataset_stats.json"))

    def _read_manifest(p: Path) -> List[Dict[str, Any]]:
        if not p.is_file():
            return []
        items = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    try:
                        items.append(json.loads(line_s))
                    except Exception:
                        pass
        return items

    raw_items = _read_manifest(raw_path)
    fake_items = _read_manifest(fake_path)
    curated_items = _read_manifest(curated_path)
    hard_items = _read_manifest(hard_path)

    # Counts by source
    counts_by_source = {
        "raw_real": len(raw_items),
        "synthetic_total": len(fake_items),
        "curated_synthetic": len(curated_items),
        "hard_samples": len(hard_items),
    }

    # Counts by manipulation mode
    counts_by_mode: Dict[str, int] = defaultdict(int)
    for it in curated_items:
        m = it.get("manipulation_type", "unknown")
        counts_by_mode[m] += 1

    # Demographics breakdown
    age_dist: Dict[str, int] = defaultdict(int)
    gender_dist: Dict[str, int] = defaultdict(int)
    for it in raw_items + curated_items:
        demo = it.get("demographics") or {}
        a = demo.get("age_group", "unknown")
        g = demo.get("gender", "unknown")
        age_dist[a] += 1
        gender_dist[g] += 1

    # Hard samples by mode
    hard_by_mode: Dict[str, int] = defaultdict(int)
    for it in hard_items:
        m = it.get("manipulation_type", "unknown")
        hard_by_mode[m] += 1

    hard_rate_by_mode = {
        m: float(hard_by_mode[m] / max(1, counts_by_mode.get(m, 0)))
        for m in ("sd_text2img", "sd_inpaint", "sd_img2img")
    }

    curation_pass_rate = (
        float(len(curated_items) / max(1, len(fake_items))) if fake_items else 0.0
    )

    stats = {
        "counts_by_source": counts_by_source,
        "counts_by_mode": dict(counts_by_mode),
        "demographics": {
            "age_distribution": dict(age_dist),
            "gender_distribution": dict(gender_dist),
        },
        "curation_pass_rate": curation_pass_rate,
        "hard_rate_by_mode": hard_rate_by_mode,
        "tau": cfg.get("hardness", {}).get("tau"),
        "hardness_config": cfg.get("hardness", {}),
        "generation_config": cfg.get("generation", {}),
    }

    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved dataset statistics to %s", stats_file)
    return stats
