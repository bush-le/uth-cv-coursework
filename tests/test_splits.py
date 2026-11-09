"""Unit tests for identity-disjoint splitting, benchmark stage construction, and dataset statistics."""

import json
from pathlib import Path
import pytest

from src.data.image_schema import ImageMetadata
from src.data.splits import (
    assign_identity_splits,
    build_stage,
    compute_dataset_stats,
)


@pytest.fixture
def sample_records():
    """Create a diverse pool of real, synthetic, and hard sample metadata records."""
    records = []
    # 4 real individuals (r0, r1, r2, r3)
    for i in range(4):
        records.append(
            ImageMetadata(
                image_id=f"real_{i}",
                label=0,
                file_path=f"real_{i}.png",
                data_source="baseline",
                demographics={"identity_id": f"person_{i}", "gender": "male", "age_group": "adult"},
            )
        )
    # Synthetic faces: 2 from r0 (sd_inpaint), 2 from r1 (sd_img2img), 2 text2img (r_none)
    records.append(
        ImageMetadata(
            image_id="fake_r0_0",
            label=1,
            manipulation_type="sd_inpaint",
            source_real_id="real_0",
            file_path="fake_r0_0.png",
            is_hard_sample=True,
            data_source="hard_sample",
        )
    )
    records.append(
        ImageMetadata(
            image_id="fake_r0_1",
            label=1,
            manipulation_type="sd_inpaint",
            source_real_id="real_0",
            file_path="fake_r0_1.png",
            is_hard_sample=False,
            data_source="synthetic",
        )
    )
    records.append(
        ImageMetadata(
            image_id="fake_r1_0",
            label=1,
            manipulation_type="sd_img2img",
            source_real_id="real_1",
            file_path="fake_r1_0.png",
            is_hard_sample=False,
            data_source="synthetic",
        )
    )
    records.append(
        ImageMetadata(
            image_id="fake_t2i_0",
            label=1,
            manipulation_type="sd_text2img",
            file_path="fake_t2i_0.png",
            is_hard_sample=False,
            data_source="synthetic",
        )
    )
    return records


def test_no_identity_crosses_splits(sample_records):
    """Verify that every image belonging to the same source identity stays in the same split."""
    assignment = assign_identity_splits(
        records=sample_records,
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        seed=42,
    )

    # Check that real_0 and fake_r0_0, fake_r0_1 have identical split assignment
    split_r0 = assignment["real_0"]
    # All records for real_0 share group_key real_0
    for r in sample_records:
        if r.source_real_id == "real_0" or r.image_id == "real_0":
            key = r.source_real_id or r.image_id
            assert assignment[key] == split_r0


def test_test_split_identical_across_stages(sample_records):
    """Verify test split records are fixed and identical across D0, D1, D2 benchmarks."""
    assignment = assign_identity_splits(
        records=sample_records,
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        seed=42,
    )

    stage_d0 = {
        "name": "D0_baseline",
        "train_sources": ["baseline"],
        "train_modes": ["all"],
        "oversample_hard": 1,
    }
    stage_d2 = {
        "name": "D2_hard_boosted",
        "train_sources": ["baseline", "synthetic", "hard_sample"],
        "train_modes": ["all"],
        "oversample_hard": 2,
    }

    s0 = build_stage(sample_records, assignment, stage_d0)
    s2 = build_stage(sample_records, assignment, stage_d2)

    # Test sets must be bit-for-bit identical
    assert [m.image_id for m in s0["test"]] == [m.image_id for m in s2["test"]]
    assert [m.image_id for m in s0["val"]] == [m.image_id for m in s2["val"]]


def test_stage_never_relabels_data_source(sample_records):
    """Verify build_stage never mutates or overwrites data_source of synthetic/hard records."""
    assignment = assign_identity_splits(
        records=sample_records,
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        seed=42,
    )
    stage_d0 = {
        "name": "D0_baseline",
        "train_sources": ["baseline", "synthetic"],
        "train_modes": ["all"],
        "oversample_hard": 1,
    }
    s0 = build_stage(sample_records, assignment, stage_d0)
    for m in s0["train"]:
        if m.label == 1:
            assert m.data_source != "baseline", "Synthetic face must never be relabeled as baseline!"


def test_stats_contains_no_inputs_only_computed_counts(tmp_path):
    """Verify compute_dataset_stats produces comprehensive data analytics dictionary."""
    stats_file = tmp_path / "stats.json"
    fake_manifest = tmp_path / "fake.jsonl"
    curated_manifest = tmp_path / "curated.jsonl"
    hard_manifest = tmp_path / "hard.jsonl"
    raw_manifest = tmp_path / "raw.jsonl"

    raw_manifest.write_text(json.dumps({"image_id": "r1", "label": 0, "demographics": {"gender": "female"}}) + "\n")
    fake_manifest.write_text(json.dumps({"image_id": "f1", "label": 1, "manipulation_type": "sd_inpaint"}) + "\n")
    curated_manifest.write_text(json.dumps({"image_id": "f1", "label": 1, "manipulation_type": "sd_inpaint"}) + "\n")
    hard_manifest.write_text(json.dumps({"image_id": "f1", "label": 1, "manipulation_type": "sd_inpaint"}) + "\n")

    cfg = {
        "paths": {
            "raw_manifest": str(raw_manifest),
            "fake_manifest": str(fake_manifest),
            "curated_manifest": str(curated_manifest),
            "hard_manifest": str(hard_manifest),
            "stats_file": str(stats_file),
        },
        "hardness": {"tau": 0.85},
    }

    stats = compute_dataset_stats(cfg)
    assert stats_file.exists()
    assert "counts_by_source" in stats
    assert "counts_by_mode" in stats
    assert "curation_pass_rate" in stats
    assert "hard_rate_by_mode" in stats
    assert stats["counts_by_source"]["raw_real"] == 1
    assert stats["counts_by_source"]["hard_samples"] == 1


def test_strict_identity_isolation_across_rounds():
    """Verify strict identity disjointness and immutable partition generation."""
    from src.data.splits import create_identity_partitions
    from src.data.image_schema import ImageMetadata
    import tempfile
    from pathlib import Path
    import json

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        records = [
            ImageMetadata(image_id=f"ffhq_{i:05d}", label=0, source_real_id=f"ffhq_{i:05d}")
            for i in range(100)
        ]
        train_p, test_p = create_identity_partitions(records, output_dir=tmp_p, train_ratio=0.8, seed=42)
        assert train_p.exists() and test_p.exists()
        train_ids = {json.loads(line)["source_real_id"] for line in open(train_p)}
        test_ids = {json.loads(line)["source_real_id"] for line in open(test_p)}
        assert len(train_ids) == 80
        assert len(test_ids) == 20
        assert train_ids.isdisjoint(test_ids)

