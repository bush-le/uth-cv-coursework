"""Tests for dynamic anatomical face masking in Stable Diffusion inpainting."""

import pytest
import torch
import numpy as np


def test_dynamic_oval_mask_covers_full_jaw():
    """Verify dynamic oval mask covers lower jaw and expands past mouth."""
    from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask

    dummy_img = torch.zeros(3, 256, 256)
    # Face bbox in upper-middle: [ymin=0.20, xmin=0.25, ymax=0.65, xmax=0.75]
    bbox = (0.20, 0.25, 0.65, 0.75)
    mask = generate_dynamic_face_mask(dummy_img, bbox=bbox, blur_radius=12)

    assert mask.shape == (1, 256, 256)
    assert mask.dtype == torch.float32
    # Ensure mask values are normalized in [0, 1]
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0

    # Ensure center of face is covered
    assert mask[:, int(0.45 * 256), int(0.50 * 256)].item() > 0.8

    # Ensure mask expanded downwards past y=0.65 to cover the jawline
    assert mask[:, int(0.70 * 256), int(0.50 * 256)].item() > 0.5

    # Ensure background outside face area is close to zero
    assert mask[:, 10, 10].item() < 0.05
    assert mask[:, 250, 10].item() < 0.05


def test_dynamic_oval_mask_auto_detects_or_fallbacks():
    """Verify fallback when bbox is None or no face detected."""
    from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask

    dummy_img = torch.zeros(3, 256, 256)
    mask = generate_dynamic_face_mask(dummy_img, bbox=None, blur_radius=12)
    assert mask.shape == (1, 256, 256)
    assert mask[:, 128, 128].item() > 0.5
