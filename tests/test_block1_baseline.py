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


@pytest.mark.smoke
def test_package_benchmark_d0_vault_protection_blocks_and_allows_overwrite(tmp_path: Path) -> None:
    """Sealed master_frozen_test.jsonl (>=1178) stays protected unless allow_overwrite_master_test=True."""
    benchmarks_dir = tmp_path / "benchmarks"
    d0_dir = benchmarks_dir / "D0_baseline"
    d0_dir.mkdir(parents=True)

    sealed_vault = benchmarks_dir / "master_frozen_test.jsonl"
    sealed_records = [
        ImageMetadata(image_id=f"sealed_{i}", label=i % 2, source_real_id=f"id_{i}")
        for i in range(1178)
    ]
    sealed_payload = "".join(json.dumps(r.to_dict()) + "\n" for r in sealed_records)
    sealed_vault.write_text(sealed_payload, encoding="utf-8")

    # Even with >=1178 new candidates (e.g. 600+600=1200), default must protect sealed vault
    partitions = {
        "train": [ImageMetadata(image_id="r_tr_0", label=0, source_real_id="tr_0")],
        "val": [ImageMetadata(image_id="r_va_0", label=0, source_real_id="va_0")],
        "test": [ImageMetadata(image_id=f"new_r_{i}", label=0, source_real_id=f"nr_{i}") for i in range(600)],
    }
    curated_fakes = {
        "train": [ImageMetadata(image_id="f_tr_0", label=1, source_real_id="tr_0")],
        "val": [ImageMetadata(image_id="f_va_0", label=1, source_real_id="va_0")],
        "test": [ImageMetadata(image_id=f"new_f_{i}", label=1, source_real_id=f"nr_{i}") for i in range(600)],
    }

    paths = package_benchmark_d0(partitions, curated_fakes, benchmarks_dir, allow_overwrite_master_test=False)
    assert paths["master_test"].read_text(encoding="utf-8") == sealed_payload
    assert paths["d0_test"].read_text(encoding="utf-8") == sealed_payload

    package_benchmark_d0(partitions, curated_fakes, benchmarks_dir, allow_overwrite_master_test=True)
    updated_lines = [line for line in paths["master_test"].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(updated_lines) == 1200


@pytest.mark.smoke
def test_synthesize_and_curate_quota_curate_only_and_shortfall_refill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify curate_only avoids generation and --no-resume re-curates existing raw before refilling shortfall."""
    from PIL import Image
    from src.models.diffusion.prompt_engine import PromptGrammarEngine
    from src.pipelines import run_block1_baseline as rb1

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    real_m = tmp_path / "train_reals.jsonl"
    real_p = img_dir / "real_0.png"
    Image.new("RGB", (64, 64), color=(140, 140, 140)).save(real_p)
    real_m.write_text(json.dumps({"image_id": "real_0", "label": 0, "file_path": str(real_p)}) + "\n", encoding="utf-8")

    raw_base = tmp_path / "d0_raw.jsonl"
    raw_att0 = tmp_path / "d0_raw_att0.jsonl"
    cur_m = tmp_path / "d0_curated.jsonl"
    rep_p = tmp_path / "d0_report.json"

    f0_p = img_dir / "f0.png"
    f1_p = img_dir / "f1.png"
    f2_p = img_dir / "f2.png"
    Image.new("RGB", (64, 64), color=(120, 140, 160)).save(f0_p)
    Image.new("RGB", (64, 64), color=(150, 130, 110)).save(f1_p)
    Image.new("RGB", (64, 64), color=(180, 100, 120)).save(f2_p)

    # Pre-existing raw attempt0 (2 samples), stale curated (3 samples would falsely short-circuit if resume=True)
    raw_att0.write_text(
        json.dumps({"image_id": "f0", "label": 1, "manipulation_type": "sd_text2img", "file_path": str(f0_p)})
        + "\n"
        + json.dumps({"image_id": "f1", "label": 1, "manipulation_type": "sd_text2img", "file_path": str(f1_p)})
        + "\n",
        encoding="utf-8",
    )
    cur_m.write_text(
        "".join(
            json.dumps({"image_id": f"stale_{i}", "label": 1, "file_path": str(f0_p)}) + "\n"
            for i in range(3)
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rb1,
        "curate_dataset",
        lambda **kwargs: Path(kwargs["output_manifest"]).write_text(
            Path(kwargs["input_manifest"]).read_text(encoding="utf-8"), encoding="utf-8"
        ),
    )

    def _forbid_gen(**kwargs):
        raise AssertionError("run_image_generation must not be called in curate_only mode")

    monkeypatch.setattr(rb1, "run_image_generation", _forbid_gen)
    prompt_eng = PromptGrammarEngine(seed=42)

    kept_only = rb1.synthesize_and_curate_quota(
        input_manifest=real_m,
        output_raw_manifest=raw_base,
        output_curated_manifest=cur_m,
        output_dir=img_dir,
        target_count=3,
        prompt_engine=prompt_eng,
        report_path=rep_p,
        curate_only=True,
        resume=False,
    )
    assert [m.image_id for m in kept_only] == ["f0", "f1"]
    assert raw_base.exists()

    # Now allow shortfall refill with resume=False, curate_only=False
    gen_calls = []

    def _mock_gen(**kwargs):
        gen_calls.append(kwargs["output_manifest"].name)
        Path(kwargs["output_manifest"]).write_text(
            json.dumps({"image_id": "f2", "label": 1, "manipulation_type": "sd_text2img", "file_path": str(f2_p)})
            + "\n",
            encoding="utf-8",
        )
        return 1

    monkeypatch.setattr(rb1, "run_image_generation", _mock_gen)
    refilled = rb1.synthesize_and_curate_quota(
        input_manifest=real_m,
        output_raw_manifest=raw_base,
        output_curated_manifest=cur_m,
        output_dir=img_dir,
        target_count=3,
        prompt_engine=prompt_eng,
        report_path=rep_p,
        curate_only=False,
        resume=False,
    )
    assert gen_calls == ["d0_raw_att1.jsonl"]
    assert [m.image_id for m in refilled] == ["f0", "f1", "f2"]


@pytest.mark.smoke
def test_block1_parse_args_supports_no_resume_curate_only_and_overwrite_vault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Block 1 CLI flags and clean 10k defaults (589/1411/200)."""
    import sys
    from src.pipelines.run_block1_baseline import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_block1_baseline.py", "--no-resume", "--curate-only", "--overwrite-master-test"],
    )
    args = parse_args()
    assert args.test_fake_count == 589
    assert args.train_fake_count == 1411
    assert args.val_fake_count == 200
    assert args.resume is False
    assert args.curate_only is True
    assert args.overwrite_master_test is True
