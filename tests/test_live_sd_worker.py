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
