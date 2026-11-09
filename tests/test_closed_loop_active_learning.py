"""Integration tests for the closed-loop active learning orchestrator."""

from pathlib import Path
import json
import pytest
import torch

from src.data.image_schema import ImageMetadata
from src.pipelines.run_closed_loop_active_learning import (
    extract_error_attribution_from_hard_samples,
    run_active_learning_loop,
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

