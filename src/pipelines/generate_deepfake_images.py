"""Pipeline entrypoint: Phase 1 batch generation of synthetic deepfake facial images via Stable Diffusion."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from PIL import Image
import torch

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_schema import ImageMetadata
from src.models.diffusion.stable_diffusion_engine import StableDiffusionEngine

logger = get_logger(__name__)


def _build_demographic_prompt(demographics: Optional[Dict[str, Any]], base_prompt: str) -> str:
    """Enrich generation prompt with demographic context (age, occlusion, glasses)."""
    if not demographics:
        return base_prompt
    tags = []
    age = demographics.get("age_group")
    if age in ["0-2", "3-6", "7-9", "10-14"]:
        tags.append("young child portrait, soft natural skin")
    elif age in ["50-69", "70-120"]:
        tags.append("elderly person face, visible skin wrinkles, natural aging")
    elif demographics.get("gray_hair") is True:
        tags.append("elderly face, natural gray hair")

    glasses = demographics.get("glasses")
    if glasses == "Dark":
        tags.append("wearing dark sunglasses with subtle glare")
    elif glasses == "Normal" or demographics.get("eyeglasses") is True:
        tags.append("wearing eyeglasses with realistic frame reflections")

    if tags:
        return f"{base_prompt}, {', '.join(tags)}"
    return base_prompt


def run_image_generation(
    input_manifest: Path,
    output_manifest: Path,
    output_dir: Path,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    mock_cache: Optional[Dict[str, torch.Tensor]] = None,
    num_per_real: int = 1,
    max_samples: Optional[int] = None,
    steps: int = 20,
    use_mock_fallback: bool = False,
    seed: int = 42,
) -> int:
    """Generate synthetic deepfake faces (DF img) from real image manifest.

    Args:
        input_manifest: Path to real faces manifest.jsonl.
        output_manifest: Path to output synthetic faces manifest.jsonl.
        output_dir: Directory where generated image files will be saved.
        model_name: Name of the Stable Diffusion model or mock.
        mock_cache: Optional dictionary of tensors for headless testing.
        num_per_real: Number of synthetic variants to generate per real face.
        max_samples: Optional maximum number of real samples to synthesize from.
        steps: Number of diffusion inference steps.
        use_mock_fallback: Whether to allow fallback to mock generator on failure.
        seed: Random seed for reproducibility.

    Returns:
        Number of synthetic images generated.
    """
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    engine = StableDiffusionEngine(
        model_name=model_name,
        use_mock_fallback=use_mock_fallback,
    )
    generated_count = 0
    fake_records = []

    if not input_manifest.exists():
        logger.warning("Input manifest %s not found, generating minimal sample.", input_manifest)
        real_items = [{"image_id": "real_sample_01", "label": 0}]
    else:
        with open(input_manifest, "r", encoding="utf-8") as f:
            real_items = [json.loads(line) for line in f if line.strip()]

    if max_samples is not None and max_samples > 0:
        real_items = real_items[:max_samples]

    for item_idx, item in enumerate(real_items):
        real_id = item["image_id"]
        demographics = item.get("demographics")

        if mock_cache and real_id in mock_cache:
            source_img = mock_cache[real_id]
        elif "file_path" in item and item["file_path"] and Path(item["file_path"]).exists():
            with Image.open(item["file_path"]) as img:
                img_rgb = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                np_img = np.array(img_rgb, dtype=np.float32) / 255.0
                source_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
        else:
            source_img = torch.rand(3, 256, 256)

        for k in range(num_per_real):
            fake_id = f"fake_{real_id}_{k}"

            mode_idx = (item_idx + k) % 3
            if mode_idx == 0:
                base_p = "subtle facial manipulation, realistic skin texture"
                prompt_used = _build_demographic_prompt(demographics, base_p)
                fake_tensor = engine.generate_img2img(
                    source_image=source_img,
                    prompt=prompt_used,
                    strength=0.5,
                    num_inference_steps=steps,
                    seed=seed + k,
                )
                manip_type = "sd_img2img"
                sub_folder = "img2img"
            elif mode_idx == 1:
                h, w = source_img.shape[1], source_img.shape[2]
                mask = torch.zeros(1, h, w)
                mask[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
                base_p = "altered facial expression and eyes"
                prompt_used = _build_demographic_prompt(demographics, base_p)
                fake_tensor = engine.generate_inpaint(
                    source_image=source_img,
                    mask=mask,
                    prompt=prompt_used,
                    num_inference_steps=steps,
                    seed=seed + k,
                )
                manip_type = "sd_inpaint"
                sub_folder = "inpainting"
            else:
                h, w = source_img.shape[1], source_img.shape[2]
                base_p = "photorealistic portrait of a synthetic person"
                prompt_used = _build_demographic_prompt(demographics, base_p)
                fake_tensor = engine.generate_text2img(
                    prompt=prompt_used,
                    height=h,
                    width=w,
                    num_inference_steps=steps,
                    seed=seed + k,
                )
                manip_type = "sd_text2img"
                sub_folder = "text2image"

            # Save PNG into dedicated mode folder
            sub_dir = output_dir / sub_folder
            sub_dir.mkdir(parents=True, exist_ok=True)
            out_file = sub_dir / f"{fake_id}.png"
            np_img = (fake_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(np_img).save(out_file)

            # Also save flat copy in output_dir root for flat lookup
            flat_file = output_dir / f"{fake_id}.png"
            if flat_file != out_file:
                Image.fromarray(np_img).save(flat_file)

            meta = ImageMetadata(
                image_id=fake_id,
                label=1,
                manipulation_type=manip_type,
                prompt=prompt_used,
                source_real_id=real_id,
                file_path=str(out_file),
                is_hard_sample=False,
                data_source="synthetic",
                demographics=demographics,
            )
            fake_records.append(meta.to_dict())
            generated_count += 1

    with open(output_manifest, "w", encoding="utf-8") as f:
        for r in fake_records:
            f.write(json.dumps(r) + "\n")

    # Mirror to data/processed for backwards compatibility
    if "synthetic" in str(output_manifest):
        legacy_manifest = Path("data/processed/fake_manifest.jsonl")
        legacy_manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(legacy_manifest, "w", encoding="utf-8") as f:
            for r in fake_records:
                f.write(json.dumps(r) + "\n")

    logger.info(
        "Generation complete: %d synthetic faces saved to %s.", generated_count, output_manifest
    )
    return generated_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic deepfake faces via Stable Diffusion."
    )
    parser.add_argument("--input-manifest", type=str, default="data/raw/manifest.jsonl")
    parser.add_argument("--output-manifest", type=str, default="data/synthetic/fake_manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Model checkpoint or ID (e.g. runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock generator mode for offline testing",
    )
    parser.add_argument("--num-per-real", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None, help="Max real samples to synthesize from")
    parser.add_argument("--steps", type=int, default=20, help="Number of diffusion inference steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_to_use = "mock_sd" if args.mock else args.model
    run_image_generation(
        input_manifest=Path(args.input_manifest),
        output_manifest=Path(args.output_manifest),
        output_dir=Path(args.output_dir),
        model_name=model_to_use,
        num_per_real=args.num_per_real,
        max_samples=args.max_samples,
        steps=args.steps,
        use_mock_fallback=args.mock,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
