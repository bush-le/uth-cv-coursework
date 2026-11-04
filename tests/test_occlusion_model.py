"""Unit tests for facial occlusion segmentation baseline model and builder."""

import pytest
import torch

from src.models.builder import build_model
from src.models.deep.segmentation import FacialOcclusionSegmenter


def test_segmenter_forward_pass():
    model = FacialOcclusionSegmenter(in_channels=3, num_classes=6)
    model.eval()

    # Input: (B, 3, 512, 512)
    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)

    # Output: (B, num_classes, 512, 512)
    assert output.shape == (2, 6, 512, 512)
    assert output.dtype == torch.float32


def test_model_builder_wiring():
    model = build_model(architecture="occlusion_segmenter", num_classes=6, in_channels=3)
    assert isinstance(model, FacialOcclusionSegmenter)
