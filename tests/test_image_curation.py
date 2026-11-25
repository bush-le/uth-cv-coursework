"""Unit tests for image quality and curation gates."""

import json
from pathlib import Path
import numpy as np
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


def test_haar_face_classifier_is_singleton():
    """Verify Haar cascade classifier is loaded once as a singleton to avoid 22ms per-image penalty."""
    from src.common.metrics import get_haar_face_classifier

    c1 = get_haar_face_classifier()
    c2 = get_haar_face_classifier()
    assert c1 is c2, "get_haar_face_classifier must return the identical singleton instance"


def test_laplacian_sharpness_on_face_roi():
    """Verify sharpness can be computed specifically on face ROI rather than blurred bokeh background."""
    # Create image with smooth black background (256x256) but noisy/sharp face in [100, 100, 150, 150]
    img = torch.zeros(3, 256, 256)
    img[:, 100:150, 100:150] = torch.rand(3, 50, 50)

    # Sharpness on full image is heavily diluted by 96% black background
    full_sharpness = compute_laplacian_sharpness(img)
    # Sharpness on ROI focuses on the face
    roi_sharpness = compute_laplacian_sharpness(img, bbox_xyxy=[100, 100, 150, 150])

    assert roi_sharpness > full_sharpness * 3.0, f"Expected ROI sharpness > 3x full, got {roi_sharpness} vs {full_sharpness}"



def test_curation_rejects_dual_mouth_anomaly():
    """Verify facial geometry check detects anatomical defects like dual mouth anomalies or seam spikes."""
    from src.common.metrics import verify_facial_geometry

    # Find any existing face image to inject an anatomical boundary seam spike
    face_candidates = list(Path("data/raw").glob("*.png")) + list(Path("data/raw").glob("*.jpg"))
    if face_candidates:
        arr = np.array(Image.open(face_candidates[0]).convert("RGB").resize((256, 256)))
        # Inject an abrupt horizontal seam line across lower face (simulating dual mouth / cutoff anomaly)
        arr[140:144, 50:200, :] = 255  # sharp seam line
        arr[145:149, 50:200, :] = 0    # high contrast edge
        is_valid, reasons = verify_facial_geometry(arr)
        # Should be flagged as invalid due to dual mouth anomaly or abnormal boundary discontinuity
        assert not is_valid
        assert len(reasons) > 0
        assert any("boundary" in r.lower() or "discontinuity" in r.lower() or "aspect ratio" in r.lower() for r in reasons)


def test_boundary_seam_verification():
    """Verify boundary seam detector flags sharp edge mismatches."""
    from src.common.metrics import verify_boundary_seam

    src = torch.zeros(3, 64, 64)
    # Synthetic with high-contrast sharp seam
    fake = torch.ones(3, 64, 64)
    mask = torch.zeros(1, 64, 64)
    mask[:, 16:48, 16:48] = 1.0

    is_seamless, score = verify_boundary_seam(fake, src, mask, max_gradient_diff=10.0)
    assert not is_seamless
    assert score > 10.0


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


def test_curation_rejects_faceless_artifacts_with_opencv_haar(tmp_path):
    """Verify curation gate rejects non-face artifacts (e.g. scanlines, barcode) when opencv_haar is enabled."""
    in_m = tmp_path / "in.jsonl"
    out_m = tmp_path / "out.jsonl"
    rep_m = tmp_path / "report.jsonl"

    # Create an image with horizontal stripes (like blinds artifact, no face)
    stripes = np.zeros((128, 128, 3), dtype=np.uint8)
    stripes[::8, :, :] = 255
    blinds_path = tmp_path / "blinds.png"
    Image.fromarray(stripes).save(blinds_path)

    with open(in_m, "w") as f:
        f.write(json.dumps({"image_id": "blinds_01", "label": 1, "file_path": str(blinds_path)}) + "\n")

    gate = ImageCurationGate(min_sharpness=0.0, clip_enabled=False, face_backend="opencv_haar")
    summary = curate_image_manifest(in_m, out_m, gate, curation_report=rep_m)

    assert summary["kept"] == 0
    assert summary["rejected"] == 1
    report = json.loads(rep_m.read_text().strip())
    assert report["is_valid"] is False
    assert any("face" in r.lower() for r in report["rejection_reasons"])


@pytest.mark.smoke
def test_curate_dataset_enables_config_perceptual_dedup_source_dirs_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify curate_dataset resolves perceptual_dedup_threshold=2, forwards source_dirs, and closes gate."""
    from src.pipelines import curate_image_dataset as cid

    src_dir = tmp_path / "custom_sources"
    src_dir.mkdir()
    yy, xx = np.indices((128, 128))
    base = (128.0 + 45.0 * np.sin(xx / 10.0) + 35.0 * np.cos(yy / 14.0)).clip(20, 235).astype(np.uint8)
    arr1 = np.stack([base, np.roll(base, 6, axis=1), np.roll(base, 12, axis=0)], axis=-1)

    src_img = src_dir / "ref_01.jpg"
    Image.fromarray(arr1).save(src_img)

    p1 = tmp_path / "syn1.png"
    p2 = tmp_path / "syn2.png"
    Image.fromarray(arr1).save(p1)
    arr2 = arr1.copy()
    arr2[0, 0, 0] = min(255, int(arr2[0, 0, 0]) + 1)  # Different SHA-256, identical dHash
    Image.fromarray(arr2).save(p2)

    in_m = tmp_path / "in.jsonl"
    out_m = tmp_path / "out.jsonl"
    rep_m = tmp_path / "rep.jsonl"
    in_m.write_text(
        json.dumps(
            {
                "image_id": "f1",
                "label": 1,
                "manipulation_type": "sd_img2img",
                "source_real_image": "ref_01.jpg",
                "file_path": str(p1),
            }
        )
        + "\n"
        + json.dumps(
            {
                "image_id": "f2",
                "label": 1,
                "manipulation_type": "sd_img2img",
                "source_real_image": "ref_01.jpg",
                "file_path": str(p2),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    closed_flags = []
    orig_close = cid.ImageCurationGate.close

    def _spy_close(self) -> None:
        closed_flags.append(True)
        orig_close(self)

    monkeypatch.setattr(cid.ImageCurationGate, "close", _spy_close)

    summary = cid.curate_dataset(
        input_manifest=in_m,
        output_manifest=out_m,
        report_path=rep_m,
        min_sharpness=0.0,
        face_backend="none",
        clip_enabled=False,
        extended_audit=True,
        source_dirs=[src_dir],
    )

    assert summary["kept"] == 1
    assert summary["rejected"] == 1
    assert closed_flags == [True]
    reports = [json.loads(line) for line in rep_m.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert reports[0]["delta_e_lab"] is not None
    assert "duplicate_perceptual_dhash" in reports[1]["rejection_reasons"]


def test_rejected_sample_does_not_poison_perceptual_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify a rejected first candidate does not cause a valid near-duplicate second candidate to be dropped."""
    yy, xx = np.indices((64, 64))
    pat = (((yy // 4 + xx // 4) % 2) * 180 + 40).astype(np.uint8)
    arr = np.stack([pat, np.roll(pat, 2, axis=1), pat], axis=-1)
    p1 = tmp_path / "bad_first.png"
    p2 = tmp_path / "good_second.png"
    Image.fromarray(arr).save(p1)
    Image.fromarray(arr).save(p2)

    in_m = tmp_path / "in.jsonl"
    out_m = tmp_path / "out.jsonl"
    rep_m = tmp_path / "rep.jsonl"
    in_m.write_text(
        json.dumps({"image_id": "bad_1", "label": 1, "file_path": str(p1)})
        + "\n"
        + json.dumps({"image_id": "good_2", "label": 1, "file_path": str(p2)})
        + "\n",
        encoding="utf-8",
    )

    gate = ImageCurationGate(min_sharpness=0.0, clip_enabled=False, dedup=True, perceptual_dedup_threshold=2)

    def _selective_eval(image, prompt="", image_id="", **kwargs):
        if image_id == "bad_1":
            return ImageCurationReport(
                image_id=image_id,
                is_valid=False,
                sharpness_score=0.0,
                clip_score=None,
                face_detected=False,
                rejection_reasons=["simulated_quality_reject"],
            )
        return ImageCurationReport(
            image_id=image_id,
            is_valid=True,
            sharpness_score=50.0,
            clip_score=None,
            face_detected=True,
            rejection_reasons=[],
        )

    monkeypatch.setattr(gate, "evaluate_image", _selective_eval)
    summary = curate_image_manifest(in_m, out_m, gate, curation_report=rep_m)

    assert summary["kept"] == 1
    assert summary["rejected"] == 1
    kept_items = [json.loads(line)["image_id"] for line in out_m.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert kept_items == ["good_2"]


@pytest.mark.smoke
def test_detect_faces_accepts_float16_and_bfloat16_tensors() -> None:
    """Verify detect_faces safely converts float16 and bfloat16 CHW tensors without OpenCV depth errors."""
    from src.common.metrics import detect_faces

    for dt in (torch.float16, torch.bfloat16):
        img = torch.zeros(3, 64, 64, dtype=dt)
        count = detect_faces(img, backend="opencv_haar")
        assert isinstance(count, int)
        assert count >= 0
