"""Unit tests for StableDiffusionEngine and synthetic generation pipeline."""

import json
from pathlib import Path
import pytest
import torch
from src.models.diffusion.stable_diffusion_engine import StableDiffusionEngine
from src.pipelines.generate_deepfake_images import run_image_generation


def test_stable_diffusion_engine_text2img():
    """Test text2img mode returns valid (3, H, W) float tensor in [0.0, 1.0]."""
    engine = StableDiffusionEngine(model_name="mock_sd")
    img = engine.generate_text2img(prompt="a photorealistic face", height=128, width=128)

    assert img.shape == (3, 128, 128)
    assert img.dtype == torch.float32
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_real_engine_configuration():
    """Test real engine configuration attributes and pipeline mode factories."""
    engine = StableDiffusionEngine(
        model_name="runwayml/stable-diffusion-v1-5",
        use_mock_fallback=False,
    )
    assert engine.model_name == "runwayml/stable-diffusion-v1-5"
    assert engine.use_mock_fallback is False
    assert "deformed" in engine.default_negative_prompt
    assert hasattr(engine, "get_text2img_pipeline")
    assert hasattr(engine, "get_inpaint_pipeline")
    assert hasattr(engine, "get_img2img_pipeline")


def test_stable_diffusion_engine_inpaint():
    """Test inpaint mode modifies only within masked region."""
    engine = StableDiffusionEngine(model_name="mock_sd")
    source_img = torch.rand(3, 64, 64)
    # Mask center region (1, 64, 64)
    mask = torch.zeros(1, 64, 64)
    mask[:, 16:48, 16:48] = 1.0

    inpainted = engine.generate_inpaint(source_img, mask, prompt="smiling mouth")
    assert inpainted.shape == (3, 64, 64)
    # Outside mask must be preserved
    assert torch.allclose(inpainted[:, :16, :], source_img[:, :16, :], atol=1e-5)


def test_stable_diffusion_engine_img2img():
    """Test img2img mode alters features with controllable strength."""
    engine = StableDiffusionEngine(model_name="mock_sd")
    source_img = torch.rand(3, 64, 64)

    altered = engine.generate_img2img(source_img, prompt="alteration", strength=0.5)
    assert altered.shape == (3, 64, 64)
    assert not torch.allclose(altered, source_img)


def test_run_image_generation_pipeline(tmp_path):
    """Test CLI batch generation pipeline writing images and JSONL manifest."""
    input_manifest = tmp_path / "real_manifest.jsonl"
    output_manifest = tmp_path / "fake_manifest.jsonl"
    output_dir = tmp_path / "synthetic_faces"

    real_entry = {
        "image_id": "real_001",
        "label": 0,
        "file_path": None,
    }
    input_manifest.write_text(json.dumps(real_entry) + "\n")

    count = run_image_generation(
        input_manifest=input_manifest,
        output_manifest=output_manifest,
        output_dir=output_dir,
        model_name="mock_sd",
        mock_cache={"real_001": torch.rand(3, 64, 64)},
        num_per_real=2,
    )
    assert count == 2
    assert output_manifest.exists()
    records = [json.loads(line) for line in output_manifest.read_text().splitlines() if line.strip()]
    assert len(records) == 2
    assert Path(records[0]["file_path"]).exists()
