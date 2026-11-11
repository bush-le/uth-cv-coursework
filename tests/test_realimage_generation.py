"""Unit tests for RealImage v5.1 generation math and 6k dataset reproducibility."""

import json
from pathlib import Path
import pytest
from PIL import Image

from scripts.run_generation import create_mask, generate_prompts


def test_generate_prompts_exact_reproducibility():
    """Verify that generate_prompts with seed 42 matches the exact prompts in the 6k fake manifest."""
    manifest_path = Path("data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip("6k benchmark manifest not found on disk.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        text2img_records = [json.loads(line) for line in f if '"task": "text2img"' in line]

    assert len(text2img_records) == 2000, f"Expected 2000 text2img records, found {len(text2img_records)}"

    generated = generate_prompts(num_prompts=2000, seed=42)
    assert len(generated) == 2000

    # Test exact equality for all 2000 records and seeds
    for idx, (gen_prompt, rec) in enumerate(zip(generated, text2img_records)):
        assert gen_prompt == rec["prompt"], f"Prompt mismatch at index {idx}"
        assert (42 + idx) == rec["seed"], f"Seed mismatch at index {idx}"


def test_create_mask_geometry():
    """Verify mouth and eyes elliptical mask coordinates and format."""
    size = (512, 512)

    mouth_mask = create_mask(size, "mouth")
    assert mouth_mask.size == (512, 512)
    assert mouth_mask.mode == "L"
    # Point at center of mouth: (256, 370) should be white (255)
    assert mouth_mask.getpixel((256, 370)) == 255
    # Point at top corner: (10, 10) should be black (0)
    assert mouth_mask.getpixel((10, 10)) == 0

    eyes_mask = create_mask(size, "eyes")
    assert eyes_mask.size == (512, 512)
    assert eyes_mask.mode == "L"
    # Point at center of eye strip: (256, 200) should be white (255)
    assert eyes_mask.getpixel((256, 200)) == 255
    # Point at bottom center: (256, 450) should be black (0)
    assert eyes_mask.getpixel((256, 450)) == 0


def test_inpaint_alternation_consistency():
    """Verify that inpaint records alternate between mouth and eyes and match seeds."""
    manifest_path = Path("data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl")
    if not manifest_path.exists():
        pytest.skip("6k benchmark manifest not found on disk.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        inpaint_records = [json.loads(line) for line in f if '"task": "inpaint"' in line]

    assert len(inpaint_records) == 2000
    for idx, rec in enumerate(inpaint_records):
        expected_region = "mouth" if (idx % 2 == 0) else "eyes"
        assert rec["sub_region"] == expected_region
        assert rec["seed"] == (42 + idx)
        if expected_region == "mouth":
            assert "smiling mouth" in rec["prompt"]
        else:
            assert "sunglasses" in rec["prompt"]
