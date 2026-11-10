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
        inpaint_model_name="stable-diffusion-v1-5/stable-diffusion-inpainting",
        use_mock_fallback=False,
    )
    assert engine.model_name == "runwayml/stable-diffusion-v1-5"
    assert engine.inpaint_model_name == "stable-diffusion-v1-5/stable-diffusion-inpainting"
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


def test_calibrated_negative_prompt_within_clip_budget():
    """Verify calibrated negative prompt string stays within 68 tokens and has key triggers."""
    from src.models.diffusion.stable_diffusion_engine import CALIBRATED_NEGATIVE_PROMPT

    words = CALIBRATED_NEGATIVE_PROMPT.split()
    assert len(words) <= 25, f"Negative prompt too long ({len(words)} words): {CALIBRATED_NEGATIVE_PROMPT}"
    assert "bad anatomy" in CALIBRATED_NEGATIVE_PROMPT
    assert "distorted face" in CALIBRATED_NEGATIVE_PROMPT
    assert "blurry" in CALIBRATED_NEGATIVE_PROMPT



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


def test_dynamic_face_mask_pixel_coordinates():
    """Verify pixel coordinates on 512x512 canvas are normalized against canvas size, not hardcoded 1024."""
    from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask

    img = torch.zeros(3, 512, 512)
    # BBox centered on 512x512 image
    pixel_bbox = (100.0, 100.0, 400.0, 400.0)
    mask = generate_dynamic_face_mask(img, bbox=pixel_bbox, target_region="full")
    assert mask.shape == (1, 512, 512)
    assert mask.min() >= 0.0 and mask.max() <= 1.0

    # Center pixel (256, 256) must be active inside face mask
    assert mask[0, 256, 256].item() > 0.8, "Center of face mask must be active"
    # Far corner must be 0
    assert mask[0, 10, 10].item() == 0.0, "Top-left outer corner must be 0"


def test_dynamic_face_mask_target_regions():
    """Verify target regions produce valid (1, H, W) masks in [0.0, 1.0]."""
    from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask

    img = torch.rand(3, 256, 256)
    for region in ["full", "periocular", "lips"]:
        mask = generate_dynamic_face_mask(img, bbox=(0.2, 0.2, 0.8, 0.8), target_region=region)
        assert mask.shape == (1, 256, 256)
        assert mask.dtype == torch.float32
        assert 0.0 <= mask.min().item() <= mask.max().item() <= 1.0


def test_engine_no_grad_execution():
    """Verify that generation methods do not construct or retain autograd graphs."""
    engine = StableDiffusionEngine(model_name="mock_sd")
    src = torch.rand(3, 64, 64, requires_grad=True)
    mask = torch.ones(1, 64, 64)

    t2i = engine.generate_text2img(prompt="test", height=64, width=64)
    assert not t2i.requires_grad
    assert t2i.grad_fn is None

    inp = engine.generate_inpaint(source_image=src, mask=mask, prompt="test")
    assert not inp.requires_grad
    assert inp.grad_fn is None

    i2i = engine.generate_img2img(source_image=src, prompt="test")
    assert not i2i.requires_grad
    assert i2i.grad_fn is None
