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


def run_image_generation(
    input_manifest: Path,
    output_manifest: Path,
    output_dir: Path,
    model_name: str = "mock_sd",
    mock_cache: Optional[Dict[str, torch.Tensor]] = None,
    num_per_real: int = 1,
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
        seed: Random seed for reproducibility.

    Returns:
        Number of synthetic images generated.
    """
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    engine = StableDiffusionEngine(model_name=model_name)
    generated_count = 0
    fake_records = []

    if not input_manifest.exists():
        logger.warning("Input manifest %s not found, generating minimal sample.", input_manifest)
        real_items = [{"image_id": "real_sample_01", "label": 0}]
    else:
        with open(input_manifest, "r", encoding="utf-8") as f:
            real_items = [json.loads(line) for line in f if line.strip()]

    for item in real_items:
        real_id = item["image_id"]

        if mock_cache and real_id in mock_cache:
            source_img = mock_cache[real_id]
        else:
            source_img = torch.rand(3, 256, 256)

        for k in range(num_per_real):
            fake_id = f"fake_{real_id}_{k}"
            out_file = output_dir / f"{fake_id}.png"

            mode_idx = k % 3
            if mode_idx == 0:
                fake_tensor = engine.generate_img2img(
                    source_image=source_img,
                    prompt="subtle facial manipulation, realistic skin texture",
                    strength=0.5,
                    seed=seed + k,
                )
                manip_type = "sd_img2img"
                prompt_used = "subtle facial manipulation, realistic skin texture"
            elif mode_idx == 1:
                h, w = source_img.shape[1], source_img.shape[2]
                mask = torch.zeros(1, h, w)
                mask[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
                fake_tensor = engine.generate_inpaint(
                    source_image=source_img,
                    mask=mask,
                    prompt="altered facial expression and eyes",
                    seed=seed + k,
                )
                manip_type = "sd_inpaint"
                prompt_used = "altered facial expression and eyes"
            else:
                h, w = source_img.shape[1], source_img.shape[2]
                fake_tensor = engine.generate_text2img(
                    prompt="photorealistic portrait of a synthetic person",
                    height=h,
                    width=w,
                    seed=seed + k,
                )
                manip_type = "sd_text2img"
                prompt_used = "photorealistic portrait of a synthetic person"

            # Save PNG
            np_img = (fake_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(np_img).save(out_file)

            meta = ImageMetadata(
                image_id=fake_id,
                label=1,
                manipulation_type=manip_type,
                prompt=prompt_used,
                source_real_id=real_id,
                file_path=str(out_file),
                is_hard_sample=False,
                data_source="synthetic",
            )
            fake_records.append(meta.to_dict())
            generated_count += 1

    with open(output_manifest, "w", encoding="utf-8") as f:
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
    parser.add_argument("--output-manifest", type=str, default="data/processed/fake_manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/processed/synthetic_faces")
    parser.add_argument("--model", type=str, default="mock_sd")
    parser.add_argument("--num-per-real", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_image_generation(
        input_manifest=Path(args.input_manifest),
        output_manifest=Path(args.output_manifest),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        num_per_real=args.num_per_real,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
