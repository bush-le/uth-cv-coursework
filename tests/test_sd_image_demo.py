"""Regression tests for real pipeline routing, source integrity and resume."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.sd_demo_data import load_edit_mask, read_manifest
from src.models.diffusion.stable_diffusion_engine import StableDiffusionEngine
from src.pipelines.generate_deepfake_images import run_image_generation
from src.pipelines.sd_image_demo import build_demo_gallery


@pytest.mark.smoke
def test_missing_input_never_creates_synthetic_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_image_generation(
            tmp_path / "absent.jsonl",
            tmp_path / "out.jsonl",
            tmp_path / "images",
            "mock_sd",
        )


def test_real_img2img_receives_source_and_settings(monkeypatch):
    calls = []
    engine = StableDiffusionEngine(
        model_name="real-model",
        device="cpu",
        num_inference_steps=8,
        guidance_scale=3.0,
        negative_prompt="bad",
    )

    def pipeline(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            images=[Image.new("RGB", (64, 64), (204, 204, 204))],
            nsfw_content_detected=[False],
        )

    monkeypatch.setattr(
        engine, "_load_pipeline", lambda mode: pipeline if mode == "img2img" else None
    )
    source = torch.zeros(3, 64, 64)
    result = engine.generate_img2img(source, "warm light", strength=0.5, seed=9)
    assert result.mean().item() == pytest.approx(0.8)  # not 0.4 from old pixel blending
    assert calls[0]["image"].getpixel((0, 0)) == (0, 0, 0)
    assert calls[0]["num_inference_steps"] == 8
    assert calls[0]["negative_prompt"] == "bad"
    assert calls[0]["guidance_scale"] == 3.0
    assert calls[0]["strength"] == 0.5


def test_inpaint_conditions_on_image_and_mask(monkeypatch):
    calls = []
    engine = StableDiffusionEngine(model_name="real-model", device="cpu")

    def pipeline(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (64, 64), "white")])

    monkeypatch.setattr(engine, "_load_pipeline", lambda _: pipeline)
    source = torch.zeros(3, 64, 64)
    mask = torch.zeros(1, 64, 64)
    mask[:, 20:40, 20:40] = 1
    output = engine.generate_inpaint(source, mask)
    assert calls[0]["mask_image"].getpixel((30, 30)) == 255
    assert calls[0]["mask_image"].getpixel((0, 0)) == 0
    assert "image" in calls[0]
    assert output[:, 30, 30].eq(1).all()
    assert output[:, 0, 0].eq(0).all()


def test_real_failure_and_safety_rejection_never_mock(monkeypatch):
    engine = StableDiffusionEngine(model_name="real-model", device="cpu")

    def fail(_: str):
        raise RuntimeError("weights unavailable")

    monkeypatch.setattr(engine, "_load_pipeline", fail)
    with pytest.raises(RuntimeError, match="weights unavailable"):
        engine.generate_text2img(height=64, width=64)
    monkeypatch.setattr(
        engine,
        "_load_pipeline",
        lambda _: (
            lambda **kw: SimpleNamespace(
                images=[Image.new("RGB", (64, 64))], nsfw_content_detected=[True]
            )
        ),
    )
    with pytest.raises(RuntimeError, match="Safety checker"):
        engine.generate_text2img(height=64, width=64)


def test_sources_three_modes_resume_and_provenance(tmp_path, monkeypatch):
    image = tmp_path / "source.png"
    Image.fromarray(np.full((64, 64, 3), [0, 100, 200], dtype=np.uint8)).save(image)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_id": "real_01",
                "label": 0,
                "file_path": "source.png",
                "split": "demo_only",
            }
        )
        + "\n"
    )
    out = tmp_path / "output.jsonl"
    kwargs = {
        "input_manifest": manifest,
        "output_manifest": out,
        "output_dir": tmp_path / "images",
        "model_name": "mock_sd",
        "height": 64,
        "width": 64,
        "device": "cpu",
    }
    assert run_image_generation(**kwargs) == 3
    records = read_manifest(out)
    assert {r["manipulation_type"] for r in records} == {
        "sd_text2img",
        "sd_img2img",
        "sd_inpaint",
    }
    assert all(r["is_mock"] and r["source_sha256"] for r in records)
    t2i = next(r for r in records if r["manipulation_type"] == "sd_text2img")
    assert t2i["source_real_id"] is None  # text-only is not identity-conditioned
    assert np.array(Image.open(records[0]["source_snapshot"]))[0, 0].tolist() == [
        0,
        100,
        200,
    ]

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A completed, hash-verified job must resume without inference"
        )

    monkeypatch.setattr(StableDiffusionEngine, "generate_text2img", forbidden)
    monkeypatch.setattr(StableDiffusionEngine, "generate_img2img", forbidden)
    monkeypatch.setattr(StableDiffusionEngine, "generate_inpaint", forbidden)
    assert run_image_generation(**kwargs) == 3
    assert read_manifest(out) == records
    summary = build_demo_gallery(out, tmp_path / "gallery.html")
    assert summary["mock_outputs"] == 3
    assert summary["manual_qa"] == "pending"


def test_bad_masks_and_duplicate_inputs_fail(tmp_path):
    with pytest.raises(ValueError, match="normalized"):
        load_edit_mask((64, 64), box=(0.8, 0.2, 0.1, 0.5))
    record = {"image_id": "dup", "label": 0}
    path = tmp_path / "duplicates.jsonl"
    path.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)


def test_corrupt_output_is_regenerated(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "blue").save(source)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps({"image_id": "a", "label": 0, "file_path": "source.png"}) + "\n"
    )
    output = tmp_path / "out.jsonl"
    args = (manifest, output, tmp_path / "images", "mock_sd")
    kwargs = {"modes": ("img2img",), "height": 64, "width": 64, "device": "cpu"}
    run_image_generation(*args, **kwargs)
    record = read_manifest(output)[0]
    from pathlib import Path

    image = Path(record["file_path"])
    expected = image.read_bytes()
    image.write_bytes(b"corrupt")
    run_image_generation(*args, **kwargs)
    assert image.read_bytes() == expected


def test_seed_change_after_input_reordering_has_new_job_hash(tmp_path):
    entries = []
    for name in ("a", "b"):
        Image.new("RGB", (64, 64), "blue").save(tmp_path / f"{name}.png")
        entries.append({"image_id": name, "label": 0, "file_path": f"{name}.png"})
    manifest, output = tmp_path / "input.jsonl", tmp_path / "out.jsonl"
    kwargs = {"modes": ("text2img",), "height": 64, "width": 64, "device": "cpu"}
    manifest.write_text("".join(json.dumps(r) + "\n" for r in entries))
    run_image_generation(manifest, output, tmp_path / "images", "mock_sd", **kwargs)
    old = {r["anchor_source_id"]: r for r in read_manifest(output)}
    manifest.write_text("".join(json.dumps(r) + "\n" for r in reversed(entries)))
    run_image_generation(manifest, output, tmp_path / "images", "mock_sd", **kwargs)
    for record in read_manifest(output):
        previous = old[record["anchor_source_id"]]
        assert record["generation_seed"] != previous["generation_seed"]
        assert record["job_hash"] != previous["job_hash"]
