"""Isolated single-image Stable Diffusion 1.5 worker for interactive live generation."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import torch

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.config import cfg_get, load_config
from src.common.logger import get_logger
from src.common.metrics import detect_face_yunet
from src.common.seed import seed_everything
from src.models.diffusion.stable_diffusion_engine import (
    NATIVE_SD_SIZE,
    StableDiffusionEngine,
    generate_dynamic_face_mask,
    tensor_to_pil,
)

logger = get_logger(__name__)

VALID_MODES: Tuple[str, ...] = ("text2img", "img2img", "inpaint")
DEFAULT_CONFIG_PATH: Path = _ROOT / "configs" / "dataset_factory.yaml"


def _get_hf_hub_cache_dir() -> Path:
    """Resolve the local Hugging Face hub cache directory from environment or default home."""
    hub_override = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_override:
        return Path(hub_override)
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    return hf_home / "hub"


def _is_repo_cached(cache_dir: Path, repo_id: str) -> bool:
    """Return True if a Hugging Face repo snapshot directory exists locally."""
    return (cache_dir / f"models--{repo_id.replace('/', '--')}").is_dir()


def resolve_cached_sd_model_id(configured_id: str, mode: str) -> str:
    """Resolve configured SD 1.5 model ID against local Hugging Face cache aliases.

    Falls back to ``"runwayml/stable-diffusion-v1-5"`` when ``configured_id`` is absent
    from the local cache and ``models--runwayml--stable-diffusion-v1-5`` is cached.
    """
    if configured_id.startswith("mock") or Path(configured_id).exists():
        return configured_id

    cache_dir = _get_hf_hub_cache_dir()
    if _is_repo_cached(cache_dir, configured_id):
        return configured_id

    if (
        configured_id == "stable-diffusion-v1-5/stable-diffusion-v1-5"
        and _is_repo_cached(cache_dir, "runwayml/stable-diffusion-v1-5")
    ):
        logger.info(
            "Configured model '%s' not in local cache; falling back to cached 'runwayml/stable-diffusion-v1-5'.",
            configured_id,
        )
        return "runwayml/stable-diffusion-v1-5"

    if (
        mode == "inpaint"
        and configured_id == "stable-diffusion-v1-5/stable-diffusion-inpainting"
        and _is_repo_cached(cache_dir, "runwayml/stable-diffusion-inpainting")
    ):
        logger.info(
            "Configured inpaint model '%s' not in local cache; falling back to cached 'runwayml/stable-diffusion-inpainting'.",
            configured_id,
        )
        return "runwayml/stable-diffusion-inpainting"

    return configured_id


def _load_generation_defaults(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load generation defaults from ``configs/dataset_factory.yaml`` with safe fallbacks."""
    defaults: Dict[str, Any] = {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "inpaint_model_id": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "steps": 20,
        "guidance_scale": 6.2,
        "img2img_strength": 0.30,
        "seed": 42,
    }
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        return defaults

    cfg = load_config(cfg_file)
    for key, dotted in (
        ("model_id", "generation.model_id"),
        ("inpaint_model_id", "generation.inpaint_model_id"),
        ("steps", "generation.steps"),
        ("guidance_scale", "generation.guidance_scale"),
        ("img2img_strength", "generation.img2img_strength"),
        ("seed", "seed"),
    ):
        try:
            defaults[key] = cfg_get(cfg, dotted)
        except KeyError:
            pass
    return defaults


def _normalize_source_image(source_path: Path, size: int = NATIVE_SD_SIZE) -> Tuple[Image.Image, torch.Tensor, np.ndarray]:
    """Load, EXIF-orient, convert to RGB, and resize source image to ``(size, size)``.

    Args:
        source_path: Path to source portrait image (RGB, RGBA, grayscale, or EXIF-rotated).
        size: Target square resolution (default 512).

    Returns:
        Tuple of:
        - PIL Image in RGB mode of size ``(size, size)``.
        - Float32 RGB PyTorch tensor of shape ``(C=3, H=size, W=size)`` in ``[0.0, 1.0]``.
        - Uint8 RGB NumPy array of shape ``(H=size, W=size, C=3)`` in ``[0, 255]``
          (note: ``detect_face_yunet`` handles internal RGB -> BGR conversion for OpenCV).
    """
    with Image.open(source_path) as raw_im:
        pil_rgb = (
            ImageOps.exif_transpose(raw_im)
            .convert("RGB")
            .resize((size, size), Image.Resampling.LANCZOS)
        )
    rgb_u8 = np.asarray(pil_rgb, dtype=np.uint8)
    tensor_chw = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return pil_rgb, tensor_chw, rgb_u8


def run_single_live_generation(
    mode: str,
    prompt: str,
    output_dir: Path,
    *,
    source_image: Optional[Path] = None,
    mask_region: str = "lips",
    model_id: Optional[str] = None,
    inpaint_model_id: Optional[str] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    strength: Optional[float] = None,
    seed: Optional[int] = None,
    local_files_only: bool = False,
    cpu_offload: bool = True,
    device: Optional[str] = None,
    config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    """Run an isolated single-image Stable Diffusion generation and write artifacts to disk.

    Processes RGB tensors of shape ``(3, 512, 512)`` in ``[0.0, 1.0]`` (converted from
    ``(512, 512, 3)`` uint8 RGB images; YuNet receives RGB and converts internally to BGR)
    and produces ``output.png``, ``input.png`` (for ``img2img``/``inpaint``), ``mask.png``
    (for ``inpaint``, grayscale ``(512, 512)``), and ``metadata.json``.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {VALID_MODES}.")

    if mode in ("img2img", "inpaint"):
        if source_image is None:
            raise ValueError(f"Mode '{mode}' requires source_image.")
        source_path = Path(source_image)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source image not found: {source_path}")
    else:
        source_path = None

    defaults = _load_generation_defaults(config_path)
    resolved_steps = int(steps if steps is not None else defaults["steps"])
    resolved_guidance = float(guidance_scale if guidance_scale is not None else defaults["guidance_scale"])
    resolved_strength = float(strength if strength is not None else defaults["img2img_strength"])
    resolved_seed = int(seed if seed is not None else defaults["seed"])

    raw_model_id = str(model_id if model_id is not None else defaults["model_id"])
    resolved_model_id = resolve_cached_sd_model_id(raw_model_id, mode)

    if resolved_model_id.startswith("mock"):
        resolved_inpaint_id: Optional[str] = resolved_model_id
    else:
        raw_inpaint_id = str(inpaint_model_id if inpaint_model_id is not None else defaults["inpaint_model_id"])
        resolved_inpaint_id = resolve_cached_sd_model_id(raw_inpaint_id, "inpaint")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(resolved_seed)
    start_time = time.perf_counter()

    engine = StableDiffusionEngine(
        model_name=resolved_model_id,
        inpaint_model_name=resolved_inpaint_id,
        device=device,
        num_inference_steps=resolved_steps,
        guidance_scale=resolved_guidance,
        cpu_offload=cpu_offload,
        local_files_only=local_files_only,
    )
    try:
        if mode == "text2img":
            # Output tensor shape: (C=3, H=512, W=512) float32 RGB in [0.0, 1.0]
            out_tensor = engine.generate_text2img(
                prompt=prompt,
                height=NATIVE_SD_SIZE,
                width=NATIVE_SD_SIZE,
                seed=resolved_seed,
                num_inference_steps=resolved_steps,
                guidance_scale=resolved_guidance,
            )
        else:
            assert source_path is not None
            pil_input, source_tensor, source_rgb = _normalize_source_image(source_path, size=NATIVE_SD_SIZE)
            pil_input.save(out_dir / "input.png")

            if mode == "img2img":
                # source_tensor shape: (C=3, H=512, W=512) float32 RGB in [0.0, 1.0]
                out_tensor = engine.generate_img2img(
                    source_image=source_tensor,
                    prompt=prompt,
                    strength=resolved_strength,
                    seed=resolved_seed,
                    num_inference_steps=resolved_steps,
                    guidance_scale=resolved_guidance,
                )
            else:
                # detect_face_yunet takes (H, W, 3) uint8 RGB and converts RGB -> BGR internally
                yunet_res = detect_face_yunet(source_rgb)
                bbox = yunet_res["bbox_xyxy"] if yunet_res is not None else None
                # mask_tensor shape: (1, H=512, W=512) float32 in [0.0, 1.0]
                mask_tensor = generate_dynamic_face_mask(
                    source_tensor,
                    bbox=bbox,
                    target_region=mask_region,
                )
                mask_u8 = np.rint(
                    mask_tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).numpy() * 255.0
                ).astype(np.uint8)
                Image.fromarray(mask_u8, mode="L").save(out_dir / "mask.png")

                out_tensor = engine.generate_inpaint(
                    source_image=source_tensor,
                    mask=mask_tensor,
                    prompt=prompt,
                    seed=resolved_seed,
                    num_inference_steps=resolved_steps,
                    guidance_scale=resolved_guidance,
                )
    finally:
        engine.close()

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    out_pil = tensor_to_pil(out_tensor.detach().cpu().clamp(0.0, 1.0))
    out_pil.save(out_dir / "output.png")

    metadata: Dict[str, Any] = {
        "mode": mode,
        "model_id": resolved_model_id,
        "prompt": prompt,
        "seed": resolved_seed,
        "steps": resolved_steps,
        "guidance_scale": resolved_guidance,
        "strength": resolved_strength,
        "mask_region": mask_region,
        "latency_ms": round(latency_ms, 2),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Live SD generation finished (mode=%s, model=%s, latency_ms=%.2f, output_dir=%s)",
        mode,
        resolved_model_id,
        metadata["latency_ms"],
        out_dir,
    )
    return metadata


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and validate CLI arguments for the isolated live SD worker."""
    parser = argparse.ArgumentParser(description="Isolated single-image Stable Diffusion 1.5 worker.")
    parser.add_argument("--mode", choices=VALID_MODES, required=True, help="Generation mode.")
    parser.add_argument("--prompt", type=str, required=True, help="Text conditioning prompt.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output PNGs and metadata.json.")
    parser.add_argument("--source-image", type=Path, default=None, help="Source image path (required for img2img/inpaint).")
    parser.add_argument(
        "--mask-region",
        type=str,
        default="lips",
        choices=("full", "periocular", "lips", "nose"),
        help="Target facial region for inpainting mask.",
    )
    parser.add_argument("--model-id", type=str, default=None, help="Override SD 1.5 model ID (or 'mock_sd').")
    parser.add_argument("--inpaint-model-id", type=str, default=None, help="Override SD 1.5 inpainting model ID.")
    parser.add_argument("--steps", type=int, default=None, help="Number of denoising inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--strength", type=float, default=None, help="Img2Img denoising strength in (0, 1].")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible synthesis.")
    parser.add_argument("--local-files-only", action="store_true", help="Restrict Diffusers loading to local HF cache.")
    parser.add_argument("--no-cpu-offload", action="store_true", help="Disable Diffusers model CPU offload on CUDA.")
    parser.add_argument("--device", type=str, default=None, help="Target torch device override ('cpu', 'cuda').")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to dataset_factory.yaml defaults.",
    )

    args = parser.parse_args(argv)
    if args.mode in ("img2img", "inpaint") and args.source_image is None:
        parser.error(f"--source-image is required when --mode is '{args.mode}'.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint executing one isolated Stable Diffusion generation."""
    args = parse_args(argv)
    run_single_live_generation(
        mode=args.mode,
        prompt=args.prompt,
        output_dir=args.output_dir,
        source_image=args.source_image,
        mask_region=args.mask_region,
        model_id=args.model_id,
        inpaint_model_id=args.inpaint_model_id,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        seed=args.seed,
        local_files_only=args.local_files_only,
        cpu_offload=not args.no_cpu_offload,
        device=args.device,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
