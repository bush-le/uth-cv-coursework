"""Unit and integration tests for Block 1 baseline pipeline."""

import json
from pathlib import Path
import pytest

from src.data.image_schema import ImageMetadata
from src.pipelines.run_block1_baseline import (
    create_identity_partitions,
    package_benchmark_d0,
    run_block1_pipeline,
)


@pytest.fixture
def sample_raw_manifest(tmp_path):
    """Create a temporary raw manifest with 30 distinct identity records."""
    raw_file = tmp_path / "raw_manifest.jsonl"
    records = []
    for i in range(30):
        records.append(
            ImageMetadata(
                image_id=f"real_{i:03d}",
                label=0,
                manipulation_type="real",
                source_real_id=f"id_{i // 2:02d}",  # 15 distinct identities
                file_path=str(tmp_path / f"real_{i:03d}.png"),
                data_source="baseline",
            )
        )
    with open(raw_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    return raw_file


def test_create_identity_partitions_disjointness(sample_raw_manifest, tmp_path):
    """Verify identity partitions are 100% disjoint across train, val, and test."""
    part_dir = tmp_path / "partitions"
    splits = create_identity_partitions(sample_raw_manifest, part_dir, seed=42)

    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    train_ids = {m.source_real_id for m in splits["train"]}
    val_ids = {m.source_real_id for m in splits["val"]}
    test_ids = {m.source_real_id for m in splits["test"]}

    # Identity disjointness
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # All records accounted for
    total_reals = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_reals == 30


def test_package_benchmark_d0(tmp_path):
    """Verify D0 baseline packaging creates train, val, and test manifests."""
    benchmarks_dir = tmp_path / "benchmarks"
    partitions = {
        "train": [ImageMetadata(image_id=f"r_tr_{i}", label=0, source_real_id=f"tr_{i}") for i in range(10)],
        "val": [ImageMetadata(image_id=f"r_va_{i}", label=0, source_real_id=f"va_{i}") for i in range(3)],
        "test": [ImageMetadata(image_id=f"r_te_{i}", label=0, source_real_id=f"te_{i}") for i in range(3)],
    }
    curated_fakes = {
        "train": [ImageMetadata(image_id=f"f_tr_{i}", label=1, source_real_id=f"tr_{i}") for i in range(8)],
        "val": [ImageMetadata(image_id=f"f_va_{i}", label=1, source_real_id=f"va_{i}") for i in range(2)],
        "test": [ImageMetadata(image_id=f"f_te_{i}", label=1, source_real_id=f"te_{i}") for i in range(2)],
    }

    manifest_paths = package_benchmark_d0(partitions, curated_fakes, benchmarks_dir)

    assert manifest_paths["master_test"].exists()
    assert manifest_paths["d0_train"].exists()
    assert manifest_paths["d0_val"].exists()
    assert manifest_paths["d0_test"].exists()

    # Verify counts
    with open(manifest_paths["master_test"], "r", encoding="utf-8") as f:
        master_lines = [json.loads(l) for l in f if l.strip()]
    assert len(master_lines) == 5  # 3 reals + 2 fakes


def test_block1_pipeline_dry_run(sample_raw_manifest, tmp_path):
    """Verify Block 1 pipeline executes cleanly in dry-run mode."""
    metrics = run_block1_pipeline(
        raw_manifest_path=sample_raw_manifest,
        test_fake_count=4,
        train_fake_count=6,
        val_fake_count=2,
        dry_run=True,
        output_base_dir=tmp_path / "block1_out",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    assert "baseline_test_metrics" in metrics
    assert "d0_counts" in metrics
    assert metrics["d0_counts"]["test_total"] > 0
    assert metrics["baseline_test_metrics"]["roc_auc"] > 0
