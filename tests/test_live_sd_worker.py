"""Unit tests for offline-safe StableDiffusionEngine fallback and live_sd_worker."""

import json
from pathlib import Path
from typing import Any

from PIL import Image
import pytest
import torch

from src.models.diffusion.stable_diffusion_engine import StableDiffusionEngine


def test_load_pipeline_passes_local_files_only_and_handles_cpu_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify local_files_only is forwarded, CPU never crashes on cpu_offload, and CUDA uses enable_model_cpu_offload."""
    import diffusers

    calls: list[dict[str, Any]] = []

    class DummyPipe:
        def __init__(self) -> None:
            self.to_calls: list[Any] = []
            self.offload_calls: int = 0
            self.scheduler = type("Sched", (), {"config": {}})()

        def to(self, device: Any) -> "DummyPipe":
            self.to_calls.append(device)
            return self

        def enable_model_cpu_offload(self) -> None:
            self.offload_calls += 1

        def enable_attention_slicing(self, *args: Any, **kwargs: Any) -> None:
            pass

        def enable_vae_slicing(self) -> None:
            pass

    created_pipes: list[DummyPipe] = []

    def fake_from_pretrained(model_id: str, **kwargs: Any) -> DummyPipe:
        calls.append({"model_id": model_id, **kwargs})
        pipe = DummyPipe()
        created_pipes.append(pipe)
        return pipe

    monkeypatch.setattr(diffusers.StableDiffusionPipeline, "from_pretrained", staticmethod(fake_from_pretrained))
    monkeypatch.setattr(
        diffusers.DPMSolverMultistepScheduler,
        "from_config",
        staticmethod(lambda cfg, **kw: cfg),
    )

    cpu_engine = StableDiffusionEngine(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device="cpu",
        local_files_only=True,
        cpu_offload=True,
    )
    cpu_engine._load_pipeline("text2img")
    assert len(calls) == 1
    assert calls[0]["model_id"] == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert calls[0]["local_files_only"] is True
    assert calls[0]["safety_checker"] is None
    assert created_pipes[0].to_calls == [torch.device("cpu")]
    assert created_pipes[0].offload_calls == 0

    cuda_engine = StableDiffusionEngine(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device="cuda",
        torch_dtype=torch.float32,
        local_files_only=True,
        cpu_offload=True,
    )
    cuda_engine._load_pipeline("text2img")
    assert len(calls) == 2
    assert calls[1]["local_files_only"] is True
    assert created_pipes[1].offload_calls == 1
    assert created_pipes[1].to_calls == []


def test_from_pretrained_safe_retries_offline_alias_on_hub_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Hub download fails with local_files_only=False, retry local_files_only=True across SD 1.5 alias."""
    import diffusers

    attempts: list[tuple[str, bool]] = []

    class DummyPipe:
        scheduler = type("Sched", (), {"config": {}})()

        def to(self, device: Any) -> "DummyPipe":
            return self

    def fake_from_pretrained(model_id: str, **kwargs: Any) -> DummyPipe:
        local_only = bool(kwargs.get("local_files_only", False))
        attempts.append((model_id, local_only))
        if model_id == "runwayml/stable-diffusion-v1-5" and local_only is True:
            return DummyPipe()
        raise OSError("Simulated offline Hub failure")

    monkeypatch.setattr(diffusers.StableDiffusionPipeline, "from_pretrained", staticmethod(fake_from_pretrained))
    monkeypatch.setattr(
        diffusers.DPMSolverMultistepScheduler,
        "from_config",
        staticmethod(lambda cfg, **kw: cfg),
    )

    engine = StableDiffusionEngine(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device="cpu",
        local_files_only=False,
        cpu_offload=False,
    )
    pipe = engine._load_pipeline("text2img")
    assert pipe is not None
    assert ("stable-diffusion-v1-5/stable-diffusion-v1-5", False) in attempts
    assert ("runwayml/stable-diffusion-v1-5", True) in attempts


def test_resolve_cached_sd_model_id_falls_back_to_runwayml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolve_cached_sd_model_id picks runwayml/stable-diffusion-v1-5 when only that cache dir exists."""
    from src.pipelines.live_sd_worker import resolve_cached_sd_model_id

    hub_dir = tmp_path / "hf_home" / "hub"
    (hub_dir / "models--runwayml--stable-diffusion-v1-5").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    resolved = resolve_cached_sd_model_id("stable-diffusion-v1-5/stable-diffusion-v1-5", "text2img")
    assert resolved == "runwayml/stable-diffusion-v1-5"


def test_live_worker_text2img_img2img_inpaint_mock_artifacts(tmp_path: Path) -> None:
    """Verify live worker writes normalized 512x512 RGB artifacts, mask.png, and metadata.json across modes."""
    from src.pipelines.live_sd_worker import run_single_live_generation

    src_path = tmp_path / "source_rgba.png"
    rgba = Image.new("RGBA", (640, 480), color=(180, 140, 120, 255))
    rgba.save(src_path)

    expected_keys = {
        "mode",
        "model_id",
        "prompt",
        "seed",
        "steps",
        "guidance_scale",
        "strength",
        "mask_region",
        "latency_ms",
    }

    # 1. text2img
    t2i_dir = tmp_path / "out_text2img"
    meta_t2i = run_single_live_generation(
        mode="text2img",
        prompt="a photorealistic portrait of a person",
        output_dir=t2i_dir,
        model_id="mock_sd",
        steps=20,
        guidance_scale=6.2,
        strength=0.30,
        seed=42,
    )
    assert (t2i_dir / "output.png").is_file()
    assert not (t2i_dir / "input.png").exists()
    assert not (t2i_dir / "mask.png").exists()
    assert (t2i_dir / "metadata.json").is_file()
    with Image.open(t2i_dir / "output.png") as im:
        assert im.size == (512, 512)
        assert im.mode == "RGB"
    disk_meta_t2i = json.loads((t2i_dir / "metadata.json").read_text(encoding="utf-8"))
    assert expected_keys.issubset(disk_meta_t2i.keys())
    assert disk_meta_t2i == meta_t2i
    assert disk_meta_t2i["mode"] == "text2img"

    # 2. img2img
    i2i_dir = tmp_path / "out_img2img"
    run_single_live_generation(
        mode="img2img",
        prompt="warm studio lighting portrait",
        output_dir=i2i_dir,
        source_image=src_path,
        model_id="mock_sd",
        steps=20,
        guidance_scale=6.2,
        strength=0.30,
        seed=43,
    )
    assert (i2i_dir / "input.png").is_file()
    assert (i2i_dir / "output.png").is_file()
    assert not (i2i_dir / "mask.png").exists()
    with Image.open(i2i_dir / "input.png") as im_in, Image.open(i2i_dir / "output.png") as im_out:
        assert im_in.size == (512, 512) and im_in.mode == "RGB"
        assert im_out.size == (512, 512) and im_out.mode == "RGB"
    disk_meta_i2i = json.loads((i2i_dir / "metadata.json").read_text(encoding="utf-8"))
    assert expected_keys.issubset(disk_meta_i2i.keys())
    assert disk_meta_i2i["mode"] == "img2img"

    # 3. inpaint
    inp_dir = tmp_path / "out_inpaint"
    run_single_live_generation(
        mode="inpaint",
        prompt="natural smiling lips",
        output_dir=inp_dir,
        source_image=src_path,
        mask_region="lips",
        model_id="mock_sd",
        steps=20,
        guidance_scale=6.2,
        strength=0.30,
        seed=44,
    )
    assert (inp_dir / "input.png").is_file()
    assert (inp_dir / "mask.png").is_file()
    assert (inp_dir / "output.png").is_file()
    with Image.open(inp_dir / "mask.png") as im_mask, Image.open(inp_dir / "output.png") as im_out:
        assert im_mask.size == (512, 512) and im_mask.mode == "L"
        assert im_out.size == (512, 512) and im_out.mode == "RGB"
    disk_meta_inp = json.loads((inp_dir / "metadata.json").read_text(encoding="utf-8"))
    assert expected_keys.issubset(disk_meta_inp.keys())
    assert disk_meta_inp["mode"] == "inpaint"
    assert disk_meta_inp["mask_region"] == "lips"


def test_live_worker_cli_requires_source_for_paired_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI parse_args / main fails cleanly with non-empty error when --mode img2img omits --source-image."""
    from src.pipelines.live_sd_worker import main, parse_args

    argv = [
        "--mode",
        "img2img",
        "--prompt",
        "test portrait",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit) as exc_info:
        parse_args(argv)
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "--source-image" in (captured.err or captured.out)

    with pytest.raises(SystemExit) as main_exc:
        main(argv)
    assert main_exc.value.code != 0


def test_run_batch_live_generation_mixed_modes_mock_sd(tmp_path: Path) -> None:
    """Verify batch generation handles mixed mode order, preserves item indices, and writes batch_manifest.json."""
    from src.pipelines.live_sd_worker import run_batch_live_generation

    src_path = tmp_path / "source_rgba.png"
    rgba = Image.new("RGBA", (640, 480), color=(180, 140, 120, 255))
    rgba.save(src_path)

    out_dir = tmp_path / "batch_out"
    configs = [
        {
            "mode": "inpaint",
            "prompt": "natural smiling lips",
            "source_image": str(src_path),
            "mask_region": "lips",
            "model_id": "mock_sd",
            "steps": 10,
            "guidance_scale": 6.0,
            "seed": 100,
        },
        {
            "mode": "text2img",
            "prompt": "a photorealistic portrait",
            "model_id": "mock_sd",
            "steps": 10,
            "guidance_scale": 6.0,
            "seed": 101,
        },
        {
            "mode": "img2img",
            "prompt": "warm studio lighting portrait",
            "source_image": str(src_path),
            "model_id": "mock_sd",
            "steps": 10,
            "guidance_scale": 6.0,
            "strength": 0.3,
            "seed": 102,
        },
    ]

    result = run_batch_live_generation(configs, out_dir)

    manifest_path = out_dir / "batch_manifest.json"
    assert manifest_path.is_file()
    assert len(result["items"]) == 3
    assert [item["item_index"] for item in result["items"]] == [0, 1, 2]
    assert (out_dir / "item_000" / "mask.png").is_file()
    assert (out_dir / "item_002" / "input.png").is_file()

    for idx in range(3):
        out_img_path = out_dir / f"item_{idx:03d}" / "output.png"
        assert out_img_path.is_file()
        assert result["items"][idx]["output_path"] == str(out_img_path)
        with Image.open(out_img_path) as im:
            assert im.size == (512, 512)
            assert im.mode == "RGB"

    assert result["items"][0]["input_path"] == str(out_dir / "item_000" / "input.png")
    assert result["items"][0]["mask_path"] == str(out_dir / "item_000" / "mask.png")
    assert "input_path" not in result["items"][1]
    assert "mask_path" not in result["items"][1]
    assert result["items"][2]["input_path"] == str(out_dir / "item_002" / "input.png")
    assert "mask_path" not in result["items"][2]

    disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["item_index"] for item in disk_manifest["items"]] == [0, 1, 2]
    assert disk_manifest["items"][0]["output_path"] == str(out_dir / "item_000" / "output.png")
    assert disk_manifest["items"][0]["mask_path"] == str(out_dir / "item_000" / "mask.png")


@pytest.mark.smoke
def test_run_batch_live_generation_mock_sd_e2e_evaluate_batch(tmp_path: Path) -> None:
    """End-to-end integration test chaining run_batch_live_generation(mock_sd) -> evaluate_batch_with_bundle."""
    from src.analysis.ui_showcase import evaluate_batch_with_bundle
    from src.pipelines.live_sd_worker import run_batch_live_generation

    src_path = tmp_path / "source_face.png"
    Image.new("RGB", (512, 512), color=(175, 135, 115)).save(src_path)

    out_dir = tmp_path / "e2e_batch"
    configs = [
        {
            "mode": "text2img",
            "prompt": "",
            "model_id": "mock_sd",
            "steps": 4,
            "guidance_scale": 6.0,
            "seed": 11,
        },
        {
            "mode": "img2img",
            "prompt": "",
            "source_image": str(src_path),
            "model_id": "mock_sd",
            "steps": 4,
            "guidance_scale": 6.0,
            "strength": 0.3,
            "seed": 12,
        },
        {
            "mode": "inpaint",
            "prompt": "",
            "source_image": str(src_path),
            "mask_region": "lips",
            "model_id": "mock_sd",
            "steps": 4,
            "guidance_scale": 6.0,
            "seed": 13,
        },
    ]

    manifest = run_batch_live_generation(configs, out_dir)
    eval_res = evaluate_batch_with_bundle(
        manifest["items"],
        bundle=None,
        min_sharpness=1.3,
        min_cheek_fft=0.005,
        min_clip=None,
        tau=0.50,
    )

    funnel = eval_res["funnel"]
    assert funnel["generated"] == 3
    assert funnel["gate_passed"] + funnel["gate_rejected"] == 3
    assert len(eval_res["items"]) == 3
    assert eval_res["items"][0]["inspection"].measured["effective_min_clip"] == pytest.approx(0.20)
    assert eval_res["items"][1]["inspection"].measured["effective_min_clip"] == pytest.approx(0.12)
    assert eval_res["items"][2]["inspection"].measured["effective_min_clip"] == pytest.approx(0.12)
    assert eval_res["items"][2]["inspection"].mask_512 is not None


def test_run_batch_live_generation_validates_empty_and_missing_source(tmp_path: Path) -> None:
    """Verify ValueError is raised for empty configs or img2img/inpaint configs missing source_image."""
    from src.pipelines.live_sd_worker import run_batch_live_generation

    with pytest.raises(ValueError):
        run_batch_live_generation([], tmp_path / "empty_out")

    with pytest.raises(ValueError):
        run_batch_live_generation(
            [{"mode": "img2img", "prompt": "missing source"}],
            tmp_path / "missing_src_out",
        )


def test_live_worker_cli_batch_manifest_executes(tmp_path: Path) -> None:
    """Verify CLI --batch-manifest executes batch generation and writes batch_manifest.json."""
    from src.pipelines.live_sd_worker import main

    manifest_path = tmp_path / "manifest.json"
    manifest_payload = [
        {"mode": "text2img", "prompt": "portrait one", "model_id": "mock_sd", "steps": 5, "seed": 11},
        {"mode": "text2img", "prompt": "portrait two", "model_id": "mock_sd", "steps": 5, "seed": 12},
    ]
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_dir = tmp_path / "cli_batch_out"
    rc = main(["--batch-manifest", str(manifest_path), "--output-dir", str(out_dir)])
    assert rc == 0
    assert (out_dir / "batch_manifest.json").is_file()
