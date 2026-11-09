"""Unit tests for image quality and curation gates."""

import json
import pytest
import torch
from src.common.metrics import (
    compute_laplacian_sharpness,
    compute_clip_similarity_proxy,
)
from src.pipelines.curate_image_dataset import ImageCurationGate, ImageCurationReport, curate_image_manifest


def test_laplacian_sharpness_contrast():
    """Test that a uniform flat image has lower sharpness than an edge image."""
    flat_img = torch.ones(3, 64, 64) * 0.5
    sharp_img = torch.zeros(3, 64, 64)
    sharp_img[:, :, :32] = 1.0 # High contrast sharp edge

    flat_score = compute_laplacian_sharpness(flat_img)
    sharp_score = compute_laplacian_sharpness(sharp_img)

    assert isinstance(flat_score, float)
    assert isinstance(sharp_score, float)
    assert sharp_score > flat_score


def test_clip_similarity_proxy():
    """Test CLIP semantic alignment proxy score range [0.0, 1.0]."""
    img = torch.rand(3, 64, 64)
    score = compute_clip_similarity_proxy(img, "a realistic human face")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_image_curation_gate_evaluation():
    """Test automated decision of ImageCurationGate."""
    gate = ImageCurationGate(min_sharpness=5.0, min_clip_score=0.10)
    img = torch.rand(3, 64, 64)
    report = gate.evaluate_image(img, prompt="test portrait", face_detected=True, image_id="face_01")

    assert isinstance(report, ImageCurationReport)
    assert report.image_id == "face_01"
    assert report.face_detected is True
    assert isinstance(report.is_valid, bool)


def test_curate_image_manifest_pipeline(tmp_path):
    """Test curation manifest filtering with real image files on disk."""
    from PIL import Image
    input_manifest = tmp_path / "raw.jsonl"
    output_manifest = tmp_path / "curated.jsonl"

    img_p1 = tmp_path / "face_1.png"
    img_p2 = tmp_path / "face_2.png"
    Image.new("RGB", (64, 64), color="blue").save(img_p1)
    Image.new("RGB", (64, 64), color="green").save(img_p2)

    items = [
        {"image_id": "face_1", "label": 1, "file_path": str(img_p1)},
        {"image_id": "face_2", "label": 1, "file_path": str(img_p2)},
    ]
    with open(input_manifest, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    gate = ImageCurationGate(min_sharpness=0.0, min_clip_score=0.0) # Permissive gate
    summary = curate_image_manifest(input_manifest, output_manifest, gate)

    assert summary["kept"] == 2
    assert output_manifest.exists()


def test_curation_rejects_missing_file(tmp_path):
    """Zero-Mock Invariant: Manifest entry pointing to missing file must be rejected, not kept."""
    m_in = tmp_path / "in.jsonl"
    m_out = tmp_path / "out.jsonl"
    m_in.write_text(json.dumps({"image_id": "a", "label": 1, "file_path": "non_existent.png"}) + "\n")
    gate = ImageCurationGate(min_sharpness=0.0, min_clip_score=0.0)
    summary = curate_image_manifest(m_in, m_out, gate)
    assert summary["kept"] == 0
    assert summary["rejected"] == 1

