"""Tests for provenance tracking, per-sample seed derivation, and reproducibility."""

import json
from pathlib import Path
import pytest
from PIL import Image

from src.data.image_schema import ImageMetadata
from src.data.provenance import file_sha256, generation_params, sample_seed
from src.pipelines.generate_deepfake_images import run_image_generation


def test_sample_seed_differs_per_image_and_is_stable():
    """Verify that sample_seed is deterministic, stable, and unique per image_id / variant."""
    s1 = sample_seed(42, "img_0001", 0)
    s2 = sample_seed(42, "img_0001", 0)
    s3 = sample_seed(42, "img_0002", 0)
    s4 = sample_seed(42, "img_0001", 1)

    assert s1 == s2, "Same inputs must produce identical seed."
    assert s1 != s3, "Different image_id must produce distinct seed."
    assert s1 != s4, "Different variant must produce distinct seed."
    assert isinstance(s1, int) and s1 >= 0


def test_metadata_roundtrip_keeps_generation_and_scores():
    """Verify ImageMetadata serializes and deserializes generation and scores fields."""
    meta = ImageMetadata(
        image_id="syn_test_01",
        label=1,
        manipulation_type="sd_text2img",
        generation={"model_id": "test_sd", "seed": 12345, "steps": 25},
        scores={"clip": 0.82, "sharpness": 145.2},
    )
    d = meta.to_dict()
    assert d["generation"]["seed"] == 12345
    assert d["scores"]["clip"] == 0.82

    recovered = ImageMetadata.from_dict(d)
    assert recovered == meta
    assert recovered.generation == {"model_id": "test_sd", "seed": 12345, "steps": 25}
    assert recovered.scores == {"clip": 0.82, "sharpness": 145.2}


def test_text2img_same_prompt_yields_distinct_images(tmp_path):
    """Verify that text2img with identical prompts produces distinct images via per-sample seeds."""
    in_manifest = tmp_path / "raw_in.jsonl"
    out_manifest = tmp_path / "fake_out.jsonl"
    out_dir = tmp_path / "synthetic"

    # 3 real records with identical demographics
    records = [
        {"image_id": f"real_{i:04d}", "label": 0, "demographics": {"gender": "female", "age_group": "young"}}
        for i in range(3)
    ]
    with open(in_manifest, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    count = run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        num_per_real=1,
        max_samples=3,
        use_mock_fallback=True,
        mode="text2img",
    )
    assert count == 3

    generated_pngs = list((out_dir / "text2image").glob("*.png"))
    assert len(generated_pngs) == 3

    hashes = {file_sha256(p) for p in generated_pngs}
    assert len(hashes) == 3, f"Expected 3 distinct file hashes, got {len(hashes)}"


def test_resume_regenerates_when_generation_params_change(tmp_path):
    """Verify resume skips only when generation parameters match, regenerates if params change."""
    in_manifest = tmp_path / "raw_in.jsonl"
    out_manifest = tmp_path / "fake_out.jsonl"
    out_dir = tmp_path / "synthetic"

    record = {"image_id": "real_0001", "label": 0, "demographics": {"gender": "male", "age_group": "adult"}}
    with open(in_manifest, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Run 1: steps=20
    run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        num_per_real=1,
        max_samples=1,
        use_mock_fallback=True,
        mode="text2img",
        steps=20,
    )
    rec1 = json.loads(out_manifest.read_text().strip())
    assert rec1["generation"]["steps"] == 20

    # Run 2: resume=True, same steps=20 -> should reuse
    run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        num_per_real=1,
        max_samples=1,
        use_mock_fallback=True,
        mode="text2img",
        steps=20,
        resume=True,
    )
    rec2 = json.loads(out_manifest.read_text().strip())
    assert rec2["generation"]["steps"] == 20

    # Run 3: resume=True, steps changed to 40 -> must regenerate and update manifest
    run_image_generation(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        output_dir=out_dir,
        model_name="mock_sd",
        num_per_real=1,
        max_samples=1,
        use_mock_fallback=True,
        mode="text2img",
        steps=40,
        resume=True,
    )
    rec3 = json.loads(out_manifest.read_text().strip())
    assert rec3["generation"]["steps"] == 40
