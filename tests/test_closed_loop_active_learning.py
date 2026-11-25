"""Integration tests for the closed-loop active learning orchestrator."""

from pathlib import Path
import json
import pytest
import torch

from src.data.image_schema import ImageMetadata
from src.pipelines.run_closed_loop_active_learning import (
    extract_error_attribution_from_hard_samples,
    run_active_learning_loop,
    split_hard_samples,
)


def test_extract_error_attribution():
    """Verify error attribution extracts failure rates across modes, regions, and prompt keywords."""
    hard_samples = [
        ImageMetadata(
            image_id="h1",
            label=1,
            manipulation_type="sd_inpaint",
            prompt="RAW photo, elderly face, direct harsh camera flash, wearing eyeglasses",
            generation={"inpaint_mask_target": "periocular"},
        ),
        ImageMetadata(
            image_id="h2",
            label=1,
            manipulation_type="sd_inpaint",
            prompt="RAW photo, young woman, soft studio key light",
            generation={"inpaint_mask_target": "periocular"},
        ),
        ImageMetadata(
            image_id="h3",
            label=1,
            manipulation_type="sd_text2img",
            prompt="RAW photo, man face, direct harsh camera flash",
        ),
    ]
    all_evaluated = hard_samples + [
        ImageMetadata(
            image_id="e1",
            label=1,
            manipulation_type="sd_img2img",
            prompt="standard portrait",
        )
    ]

    attribution = extract_error_attribution_from_hard_samples(hard_samples, all_evaluated)

    # Check mode errors
    assert "mode_errors" in attribution
    assert attribution["mode_errors"]["sd_inpaint"] == 1.0  # 2 inpaint candidates, both hard
    assert attribution["mode_errors"]["sd_img2img"] == 0.0  # 1 img2img candidate, none hard

    # Check region errors
    assert "region_errors" in attribution
    assert attribution["region_errors"]["periocular"] == 1.0

    # Check keyword attribution
    assert "keyword_errors" in attribution
    assert attribution["keyword_errors"]["lighting:direct harsh camera flash"] > 0.0


def test_closed_loop_dry_run_scaffolding(tmp_path):
    """Verify closed loop orchestrator runs a dry-run iteration cycle with mock detector."""
    out_dir = tmp_path / "closed_loop"
    metrics = run_active_learning_loop(
        num_rounds=1,
        candidates_per_round=6,
        dry_run=True,
        output_base_dir=out_dir,
    )

    assert "rounds" in metrics
    assert len(metrics["rounds"]) == 1
    r0 = metrics["rounds"][0]
    assert "test_metrics" in r0
    assert "hard_sample_count" in r0
    assert "attribution" in r0


def test_stateful_round_resume_detection(tmp_path):
    """Verify orchestrator detects pre-existing completed rounds and resumes from next round."""
    out_dir = tmp_path / "closed_loop"
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    # Simulate completed Round 1
    r1_dir = out_dir / "round_1"
    r1_dir.mkdir(parents=True)
    h1_file = r1_dir / "hard_samples.jsonl"
    h1_file.write_text(json.dumps({"image_id": "h1_sample", "label": 1, "is_hard_sample": True}) + "\n")
    
    m1_ckpt = ckpt_dir / "active_learning_m1_best.pt"
    torch.save({"epoch": 2, "best_metric": 0.75}, m1_ckpt)

    metrics = run_active_learning_loop(
        num_rounds=1,
        candidates_per_round=6,
        dry_run=True,
        output_base_dir=out_dir,
        checkpoints_dir=ckpt_dir,
    )

    # Should have executed Round 2 since Round 1 was detected as completed
    assert metrics["rounds"][-1]["round"] == 2


def test_split_hard_samples_deterministic():
    """Verify hard samples are deterministically split into 85% train and 15% challenge test without overlap."""
    hard_pool = [
        ImageMetadata(image_id=f"h_{i}", label=1, is_hard_sample=True)
        for i in range(20)
    ]
    train_h1, test_h1 = split_hard_samples(hard_pool, test_ratio=0.15, seed=42)
    train_h2, test_h2 = split_hard_samples(hard_pool, test_ratio=0.15, seed=42)

    # Determinism
    assert [m.image_id for m in train_h1] == [m.image_id for m in train_h2]
    assert [m.image_id for m in test_h1] == [m.image_id for m in test_h2]

    # No overlap & exact union
    train_ids = {m.image_id for m in train_h1}
    test_ids = {m.image_id for m in test_h1}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.union(test_ids) == {f"h_{i}" for i in range(20)}
    assert len(test_h1) == 3  # round(20 * 0.15) = 3


def test_dual_benchmark_dry_run_records_both_metrics(tmp_path):
    """Verify orchestrator dry-run outputs both master_test_metrics and challenge_test_metrics."""
    out_dir = tmp_path / "closed_loop"
    metrics = run_active_learning_loop(
        num_rounds=2,
        candidates_per_round=10,
        dry_run=True,
        output_base_dir=out_dir,
    )

    assert "rounds" in metrics
    assert len(metrics["rounds"]) == 2
    for r_entry in metrics["rounds"]:
        assert "master_test_metrics" in r_entry
        assert "challenge_test_metrics" in r_entry
        assert "hard_train_count" in r_entry
        assert "hard_test_count" in r_entry
        assert r_entry["master_test_metrics"]["roc_auc"] > 0
        assert r_entry["challenge_test_metrics"]["roc_auc"] > 0


def test_active_learning_asserts_zero_leakage_with_test_reals(tmp_path: Path) -> None:
    """Verify active learning detects identity leakage if train and test partitions overlap."""
    from src.pipelines.run_closed_loop_active_learning import verify_partition_disjointness

    train_file = tmp_path / "train.jsonl"
    test_file = tmp_path / "test.jsonl"

    train_file.write_text('{"image_id": "real_1", "identity": "id_100"}\n')
    test_file.write_text('{"image_id": "real_2", "identity": "id_100"}\n')

    with pytest.raises(AssertionError, match="FATAL IDENTITY LEAKAGE"):
        verify_partition_disjointness(train_file, test_file)


@pytest.mark.smoke
def test_active_learning_parse_args_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify active learning CLI supports --no-resume, --curate-only, and --force-regenerate."""
    import sys
    from src.pipelines.run_closed_loop_active_learning import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_closed_loop_active_learning.py", "--no-resume", "--curate-only", "--force-regenerate"],
    )
    args = parse_args()
    assert args.resume is False
    assert args.curate_only is True
    assert args.force_regenerate is True


def test_active_learning_curate_only_rebuilds_from_existing_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify curate_only re-curates existing round_1 raw pool without generation and resets stale challenge vault."""
    import numpy as np
    from PIL import Image
    import torch.nn as nn
    from src.pipelines import run_closed_loop_active_learning as al_mod

    monkeypatch.chdir(tmp_path)

    cfg_path = tmp_path / "factory.yaml"
    cfg_path.write_text(
        "curation:\n  clip:\n    enabled: false\n  face_check:\n    backend: none\n  dedup: true\n  perceptual_dedup_threshold: 2\n  extended_audit: false\n",
        encoding="utf-8",
    )

    data_dir = tmp_path / "data"
    part_dir = data_dir / "partitions"
    d0_dir = data_dir / "benchmarks" / "D0_baseline"
    al_dir = data_dir / "active_learning"
    r1_dir = al_dir / "round_1"
    ckpt_dir = tmp_path / "experiments" / "checkpoints"
    img_dir = tmp_path / "imgs"
    for d in (part_dir, d0_dir, r1_dir, ckpt_dir, img_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Create real & candidate images on disk
    r_tr_p = img_dir / "r_tr.png"
    r_va_p = img_dir / "r_va.png"
    r_te_p = img_dir / "r_te.png"
    c1_p = img_dir / "c1.png"
    c2_p = img_dir / "c2.png"
    c3_p = img_dir / "c3.png"
    Image.new("RGB", (64, 64), color=(100, 120, 140)).save(r_tr_p)
    Image.new("RGB", (64, 64), color=(110, 130, 150)).save(r_va_p)
    Image.new("RGB", (64, 64), color=(120, 140, 160)).save(r_te_p)

    yy, xx = np.indices((64, 64))
    pat1 = (((yy // 4 + xx // 4) % 2) * 180 + 40).astype(np.uint8)
    arr1 = np.stack([pat1, np.roll(pat1, 2, axis=1), pat1], axis=-1)
    Image.fromarray(arr1).save(c1_p)
    arr2 = arr1.copy()
    arr2[0, 0, 0] = min(255, int(arr2[0, 0, 0]) + 1)  # Near-duplicate of c1 rejected by dHash threshold=2
    Image.fromarray(arr2).save(c2_p)
    pat3 = (128.0 + 70.0 * np.sin(xx / 4.0) - 50.0 * np.cos(yy / 6.0)).clip(10, 245).astype(np.uint8)
    arr3 = np.stack([pat3, np.roll(pat3, 8, axis=0), 255 - pat3], axis=-1)
    Image.fromarray(arr3).save(c3_p)

    (part_dir / "train_reals.jsonl").write_text(
        json.dumps({"image_id": "r_tr", "label": 0, "file_path": str(r_tr_p)}) + "\n", encoding="utf-8"
    )
    (part_dir / "test_reals.jsonl").write_text(
        json.dumps({"image_id": "r_te", "label": 0, "file_path": str(r_te_p)}) + "\n", encoding="utf-8"
    )
    (d0_dir / "train.jsonl").write_text(
        json.dumps({"image_id": "r_tr", "label": 0, "file_path": str(r_tr_p)}) + "\n", encoding="utf-8"
    )
    (d0_dir / "val.jsonl").write_text(
        json.dumps({"image_id": "r_va", "label": 0, "file_path": str(r_va_p)}) + "\n", encoding="utf-8"
    )
    (d0_dir / "test.jsonl").write_text(
        json.dumps({"image_id": "r_te", "label": 0, "file_path": str(r_te_p)}) + "\n", encoding="utf-8"
    )

    # Pre-existing completed Round 1 artifacts & stale challenge vault to ensure curate_only resets instead of skipping
    (r1_dir / "candidates_raw.jsonl").write_text(
        json.dumps({"image_id": "c1", "label": 1, "manipulation_type": "sd_text2img", "file_path": str(c1_p)})
        + "\n"
        + json.dumps({"image_id": "c2", "label": 1, "manipulation_type": "sd_text2img", "file_path": str(c2_p)})
        + "\n"
        + json.dumps({"image_id": "c3", "label": 1, "manipulation_type": "sd_text2img", "file_path": str(c3_p)})
        + "\n",
        encoding="utf-8",
    )
    (r1_dir / "hard_samples.jsonl").write_text(json.dumps({"image_id": "old_hard", "label": 1}) + "\n", encoding="utf-8")
    (al_dir / "cumulative_challenge_test.jsonl").write_text(
        json.dumps({"image_id": "stale_chal", "label": 1, "file_path": str(r_te_p)}) + "\n", encoding="utf-8"
    )

    class _StubDetector(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))

        def unfreeze_backbone(self, *args, **kwargs) -> None:
            return None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bias.expand(x.shape[0], 1)

    stub = _StubDetector()
    torch.save({"model_state_dict": stub.state_dict()}, ckpt_dir / "baseline_d0_best.pt")
    torch.save({"model_state_dict": stub.state_dict()}, ckpt_dir / "active_learning_m1_best.pt")

    monkeypatch.setattr(al_mod, "DeepfakeImageDetector", _StubDetector)
    monkeypatch.setattr(
        al_mod,
        "run_image_generation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Generation must be skipped in curate_only mode")),
    )
    monkeypatch.setattr(
        al_mod,
        "train_deepfake_image_detector",
        lambda **kwargs: torch.save(
            {"model_state_dict": kwargs["model"].state_dict()},
            Path(kwargs["checkpoint_dir"]) / f"{kwargs['checkpoint_prefix']}_best.pt",
        ),
    )
    monkeypatch.setattr(
        al_mod,
        "evaluate_image_detector",
        lambda model, loader, device="cpu": {"accuracy": 0.9, "roc_auc": 0.95},
    )

    summary = al_mod.run_active_learning_loop(
        num_rounds=1,
        candidates_per_round=2,
        config_path=str(cfg_path),
        dry_run=False,
        output_base_dir=al_dir,
        checkpoints_dir=ckpt_dir,
        resume=True,
        curate_only=True,
        device="cpu",
        epochs_per_round=1,
    )

    assert len(summary["rounds"]) == 1
    curated_lines = [
        json.loads(line) for line in (r1_dir / "candidates_curated.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert [r["image_id"] for r in curated_lines] == ["c1", "c3"]
    chal_ids = {
        json.loads(line)["image_id"]
        for line in (al_dir / "cumulative_challenge_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "stale_chal" not in chal_ids
    assert len(chal_ids) == 2  # 1 hard test fake + 1 paired test real


def test_verify_partition_disjointness_checks_demographics_and_source_real_id(tmp_path: Path) -> None:
    """Verify identity leakage check inspects demographics.identity_id and source_real_id schemas."""
    from src.pipelines.run_closed_loop_active_learning import verify_partition_disjointness

    train_file = tmp_path / "train_reals.jsonl"
    test_file = tmp_path / "test_reals.jsonl"

    train_file.write_text(
        json.dumps({"image_id": "img_tr_1", "source_real_id": "celeba_id_545", "demographics": {"identity_id": 545}}) + "\n",
        encoding="utf-8",
    )
    test_file.write_text(
        json.dumps({"image_id": "img_te_9", "source_real_id": "celeba_id_545"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="FATAL IDENTITY LEAKAGE"):
        verify_partition_disjointness(train_file, test_file)



