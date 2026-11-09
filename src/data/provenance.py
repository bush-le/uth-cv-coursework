"""Provenance tracking, deterministic seed derivation, and file hash utilities.

Ensures zero duplicate outputs, full traceability of generation parameters, and
robust resume detection across the synthetic dataset generation pipeline.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import zlib


def sample_seed(base_seed: int, image_id: str, variant: int = 0) -> int:
    """Derive a deterministic, stable 32-bit pseudo-random seed per sample.

    Uses zlib.crc32 instead of Python's hash() to guarantee identical results across
    different Python processes, runs, and operating systems.

    Args:
        base_seed: Global seed configured in dataset_factory.yaml.
        image_id: Unique identifier of the real or synthetic face.
        variant: Index of the variant (e.g. 0..num_per_real-1).

    Returns:
        Deterministic non-negative 32-bit integer seed.
    """
    key = f"{base_seed}:{image_id}:{variant}".encode("utf-8")
    return zlib.crc32(key)


def file_sha256(path: Union[str, Path]) -> str:
    """Compute the SHA-256 cryptographic digest of a file.

    Args:
        path: Path to the target file.

    Returns:
        Hexadecimal string digest.

    Raises:
        FileNotFoundError: If the target file does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for sha256 calculation: {p}")

    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generation_params(
    cfg: Dict[str, Any],
    mode: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract standard generation parameter dictionary from configuration.

    Args:
        cfg: Configuration dictionary (typically from dataset_factory.yaml).
        mode: Manipulation mode ('sd_text2img', 'sd_inpaint', 'sd_img2img').
        overrides: Optional dictionary of CLI or runtime overrides.

    Returns:
        Dictionary of standardized generation parameters.
    """
    gen_cfg = cfg.get("generation", {})
    overrides = overrides or {}

    model_key = "inpaint_model_id" if mode == "sd_inpaint" else "model_id"
    default_model = gen_cfg.get(model_key, gen_cfg.get("model_id", "runwayml/stable-diffusion-v1-5"))

    model_id = overrides.get("model_id", default_model)
    steps = int(overrides.get("steps", gen_cfg.get("steps", 30)))
    guidance_scale = float(overrides.get("guidance_scale", gen_cfg.get("guidance_scale", 7.5)))
    img2img_strength = float(overrides.get("img2img_strength", gen_cfg.get("img2img_strength", 0.55)))
    image_size = list(overrides.get("image_size", gen_cfg.get("image_size", [512, 512])))
    negative_prompt = str(overrides.get("negative_prompt", gen_cfg.get("negative_prompt", "")))
    inpaint_mask_box = list(overrides.get("inpaint_mask_box", gen_cfg.get("inpaint_mask_box", [0.25, 0.25, 0.75, 0.75])))

    return {
        "model_id": model_id,
        "mode": mode,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "img2img_strength": img2img_strength,
        "image_size": image_size,
        "negative_prompt": negative_prompt,
        "inpaint_mask_box": inpaint_mask_box,
    }
