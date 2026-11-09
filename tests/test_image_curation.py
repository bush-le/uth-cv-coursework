"""Unit tests for image quality and curation gates."""

import json
from pathlib import Path
import pytest
import torch
from PIL import Image

from src.common.metrics import (
    compute_laplacian_sharpness,
)
from src.pipelines.curate_image_dataset import (
    ImageCurationGate,
    ImageCurationReport,
    curate_image_manifest,
)


def test_laplacian_sharpness_contrast():
    """Test that a uniform flat image has lower sharpness than an edge image."""
    flat_img = torch.ones(3, 64, 64) * 0.5
    sharp_img = torch.zeros(3, 64, 64)
    sharp_img[:, :, :32] = 1.0  # High contrast sharp edge

    flat_score = compute_laplacian_sharpness(flat_img)
    sharp_score = compute_laplacian_sharpness(sharp_img)

    assert isinstance(flat_score, float)
    assert isinstance(sharp_score, float)
    assert sharp_score > flat_score


def test_clip_disabled_records_none_not_fake_score():
    """Verify that when CLIP is disabled, score is None and no fake scores or rejections are emitted."""
    gate = ImageCurationGate(clip_enabled=False, min_sharpness=0.0)
    report = gate.evaluate_image(
        torch.ones(3, 64, 64),
        prompt="a portrait of a person",
        face_detected=True,
        image_id="img1",
    )
    assert report.clip_score is None
    assert not any("clip" in r.lower() for r in report.rejection_reasons)


def test_image_curation_gate_evaluation():
    """Test automated decision of ImageCurationGate."""
    gate = ImageCurationGate(min_sharpness=5.0, clip_enabled=False)
    img = torch.rand(3, 64, 64)
    report = gate.evaluate_image(img, prompt="test portrait", face_detected=True, image_id="face_01")

    assert isinstance(report, ImageCurationReport)
    assert report.image_id == "face_01"
    assert report.face_detected is True
    assert isinstance(report.is_valid, bool)


def test_curate_image_manifest_pipeline(tmp_path):
    """Test curation manifest filtering with real image files on disk."""
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

    gate = ImageCurationGate(min_sharpness=0.0, clip_enabled=False)  # Permissive gate
    summary = curate_image_manifest(input_manifest, output_manifest, gate)

    assert summary["kept"] == 2
    assert output_manifest.exists()


def test_duplicate_content_rejected(tmp_path):
    """Verify that duplicate image files are identified and rejected by hash."""
    in_m = tmp_path / "in.jsonl"
    out_m = tmp_path / "out.jsonl"
    p1 = tmp_path / "dup1.png"
    p2 = tmp_path / "dup2.png"

    img = Image.new("RGB", (64, 64), color="blue")
    img.save(p1)
    img.save(p2)  # Identical image content

    with open(in_m, "w") as f:
        f.write(json.dumps({"image_id": "dup1", "label": 1, "file_path": str(p1)}) + "\n")
        f.write(json.dumps({"image_id": "dup2", "label": 1, "file_path": str(p2)}) + "\n")

    gate = ImageCurationGate(min_sharpness=0.0, clip_enabled=False, dedup=True)
    summary = curate_image_manifest(in_m, out_m, gate)

    assert summary["kept"] == 1
    assert summary["rejected"] == 1


def test_curation_rejects_missing_file(tmp_path):
    """Zero-Mock Invariant: Manifest entry pointing to missing file must be rejected, not kept."""
    m_in = tmp_path / "in.jsonl"
    m_out = tmp_path / "out.jsonl"
    m_in.write_text(json.dumps({"image_id": "a", "label": 1, "file_path": "non_existent.png"}) + "\n")
    gate = ImageCurationGate(min_sharpness=0.0, clip_enabled=False)
    summary = curate_image_manifest(m_in, m_out, gate)
    assert summary["kept"] == 0
    assert summary["rejected"] == 1


def test_curation_report_output_file(tmp_path):
    """Verify curation report JSONL is written with quality inspection audit."""
    in_m = tmp_path / "in.jsonl"
    out_m = tmp_path / "out.jsonl"
    rep_m = tmp_path / "report.jsonl"
    p = tmp_path / "valid.png"
    Image.new("RGB", (64, 64), color="cyan").save(p)

    with open(in_m, "w") as f:
        f.write(json.dumps({"image_id": "v1", "label": 1, "file_path": str(p)}) + "\n")

    gate = ImageCurationGate(min_sharpness=0.0, clip_enabled=False)
    curate_image_manifest(in_m, out_m, gate, curation_report=rep_m)

    assert rep_m.exists()
    lines = [json.loads(line) for line in rep_m.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["image_id"] == "v1"
    assert lines[0]["is_valid"] is True
