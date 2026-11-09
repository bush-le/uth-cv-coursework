"""Pipeline entrypoint: Phase 1 batch generation of synthetic deepfake facial images via Stable Diffusion."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image
import torch

from src.common.config import cfg_get, load_config, resolve
from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_schema import ImageMetadata
from src.data.provenance import generation_params, sample_seed
from src.models.diffusion.stable_diffusion_engine import StableDiffusionEngine, generate_dynamic_face_mask

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
    input_manifest: Optional[Union[str, Path]] = None,
    output_manifest: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    mock_cache: Optional[Dict[str, torch.Tensor]] = None,
    num_per_real: Optional[int] = None,
    max_samples: Optional[int] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    img2img_strength: Optional[float] = None,
    use_mock_fallback: bool = False,
    mode: Optional[str] = None,
    resume: bool = True,
    seed: Optional[int] = None,
    config: Optional[Union[Dict[str, Any], Path, str]] = None,
) -> int:
    """Generate synthetic deepfake faces (DF img) from real image manifest.

    Zero-Mock & Zero-Hardcode compliant:
    - Automatically loads default configurations from configs/dataset_factory.yaml.
    - CLI/explicit arguments override config settings.
    - Derives deterministic per-sample seeds via sample_seed() preventing duplicate outputs.
    - Records full generation provenance in ImageMetadata.generation.
    - Resume regenerates when generation parameters change.
    """
    # Load configuration
    cfg: Dict[str, Any] = {}
    if isinstance(config, (str, Path)) and Path(config).is_file():
        cfg = load_config(Path(config))
    elif isinstance(config, dict):
        cfg = config
    elif Path("configs/dataset_factory.yaml").is_file():
        try:
            cfg = load_config(Path("configs/dataset_factory.yaml"))
        except Exception:
            cfg = {}

    def _res(cli_val: Any, dotted_key: str, default_val: Any) -> Any:
        if cli_val is not None:
            return cli_val
        try:
            return cfg_get(cfg, dotted_key)
        except (KeyError, TypeError):
            return default_val

    in_manifest_path = Path(_res(input_manifest, "paths.raw_manifest", "data/raw/manifest.jsonl"))
    out_manifest_path = Path(_res(output_manifest, "paths.fake_manifest", "data/synthetic/fake_manifest.jsonl"))
    out_dir_path = Path(_res(output_dir, "paths.synthetic_dir", "data/synthetic"))
    model_name_val = str(_res(model_name, "generation.model_id", "runwayml/stable-diffusion-v1-5"))
    num_per_real_val = int(_res(num_per_real, "generation.num_per_real", 1))
    max_samples_val = _res(max_samples, "generation.max_real_samples", None)
    if max_samples_val is not None:
        max_samples_val = int(max_samples_val)
    steps_val = int(_res(steps, "generation.steps", 20))
    guidance_scale_val = float(_res(guidance_scale, "generation.guidance_scale", 7.5))
    img2img_strength_val = float(_res(img2img_strength, "generation.img2img_strength", 0.55))
    seed_val = int(_res(seed, "seed", 42))
    mode_val = str(mode or "all")

    seed_everything(seed_val)

    if use_mock_fallback:
        out_res = out_dir_path.resolve()
        data_res = Path("data").resolve()
        if out_res == data_res or data_res in out_res.parents:
            raise ValueError(
                "Mock generator mode is strictly forbidden from writing to production data/ directory! "
                "See .agents/rules/ZERO_MOCK_DATA_RULE.md"
            )

    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing records for provenance-aware resume
    existing_records: Dict[str, Dict[str, Any]] = {}
    if out_manifest_path.exists():
        with open(out_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    try:
                        rec = json.loads(line_s)
                        if "image_id" in rec:
                            existing_records[rec["image_id"]] = rec
                    except Exception:
                        pass

    engine = StableDiffusionEngine(
        model_name=model_name_val,
        use_mock_fallback=use_mock_fallback,
    )
    generated_count = 0
    fake_records = []

    if not in_manifest_path.exists():
        logger.warning("Input manifest %s not found, generating minimal sample.", in_manifest_path)
        real_items = [{"image_id": "real_sample_01", "label": 0}]
    else:
        with open(in_manifest_path, "r", encoding="utf-8") as f:
            real_items = [json.loads(line) for line in f if line.strip()]

    if max_samples_val is not None and max_samples_val > 0:
        real_items = real_items[:max_samples_val]

    # Pre-plan generation tasks and group by mode to load each pipeline ONCE
    tasks = []
    for item_idx, item in enumerate(real_items):
        real_id = item["image_id"]
        demographics = item.get("demographics")
        for k in range(num_per_real_val):
            fake_id = f"fake_{real_id}_{k}"
            if mode_val == "all":
                mode_idx = (item_idx + k) % 3
            elif mode_val == "img2img":
                mode_idx = 0
            elif mode_val == "inpaint":
                mode_idx = 1
            elif mode_val == "text2img":
                mode_idx = 2
            else:
                mode_idx = (item_idx + k) % 3

            tasks.append({
                "item_idx": item_idx,
                "item": item,
                "real_id": real_id,
                "k": k,
                "fake_id": fake_id,
                "mode_idx": mode_idx,
                "demographics": demographics,
            })

    # Group by mode_idx (img2img, inpaint, text2img)
    tasks.sort(key=lambda t: t["mode_idx"])

    for task in tasks:
        item = task["item"]
        real_id = task["real_id"]
        demographics = task["demographics"]
        k = task["k"]
        fake_id = task["fake_id"]
        mode_idx = task["mode_idx"]
        source_img = None

        if mode_idx == 0:
            manip_type = "sd_img2img"
            sub_folder = "img2img"
            base_p = "subtle facial manipulation, realistic skin texture"
        elif mode_idx == 1:
            manip_type = "sd_inpaint"
            sub_folder = "inpainting"
            base_p = "altered facial expression and eyes"
        else:
            manip_type = "sd_text2img"
            sub_folder = "text2image"
            base_p = "photorealistic portrait of a synthetic person"

        prompt_used = _build_demographic_prompt(demographics, base_p)
        sub_dir = out_dir_path / sub_folder
        sub_dir.mkdir(parents=True, exist_ok=True)
        out_file = sub_dir / f"{fake_id}.png"

        # Derive per-sample deterministic seed
        curr_seed = sample_seed(seed_val, fake_id, k)

        curr_gen = generation_params(
            cfg,
            manip_type,
            overrides={
                "model_id": model_name_val,
                "steps": steps_val,
                "guidance_scale": guidance_scale_val,
                "img2img_strength": img2img_strength_val,
            },
        )
        curr_gen["seed"] = curr_seed

        # Check provenance-aware resume
        if resume and out_file.exists() and out_file.stat().st_size > 0:
            old_rec = existing_records.get(fake_id)
            old_gen = old_rec.get("generation") if old_rec else None

            # Must match key parameters to safely reuse
            if old_gen and (
                old_gen.get("steps") == curr_gen["steps"]
                and old_gen.get("model_id") == curr_gen["model_id"]
                and old_gen.get("mode") == curr_gen["mode"]
                and old_gen.get("seed") == curr_gen["seed"]
            ):
                logger.info("Resuming %s: identical generation parameters, skipping generation.", out_file.name)
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
                    generation=curr_gen,
                    scores=old_rec.get("scores"),
                )
                fake_records.append(meta.to_dict())
                generated_count += 1
                continue

        # Need source image for img2img / inpaint
        if mode_idx in (0, 1):
            if source_img is None:
                if mock_cache and real_id in mock_cache:
                    source_img = mock_cache[real_id]
                elif "file_path" in item and item["file_path"] and Path(item["file_path"]).exists():
                    with Image.open(item["file_path"]) as img:
                        img_rgb = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                        np_img = np.array(img_rgb, dtype=np.float32) / 255.0
                        source_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
                else:
                    logger.warning("Skipping %s: source image missing for mode %s", real_id, manip_type)
                    continue
            h, w = source_img.shape[1], source_img.shape[2]
        else:
            h, w = 256, 256

        if mode_idx == 0:
            fake_tensor = engine.generate_img2img(
                source_image=source_img,
                prompt=prompt_used,
                strength=img2img_strength_val,
                num_inference_steps=steps_val,
                seed=curr_seed,
                guidance_scale=guidance_scale_val,
            )
        elif mode_idx == 1:
            raw_bbox = item.get("bbox")
            mask = generate_dynamic_face_mask(source_img, bbox=raw_bbox, blur_radius=16)

            fake_tensor = engine.generate_inpaint(
                source_image=source_img,
                mask=mask,
                prompt=prompt_used,
                num_inference_steps=steps_val,
                seed=curr_seed,
                guidance_scale=guidance_scale_val,
            )
        else:
            fake_tensor = engine.generate_text2img(
                prompt=prompt_used,
                height=h,
                width=w,
                num_inference_steps=steps_val,
                seed=curr_seed,
                guidance_scale=guidance_scale_val,
            )

        # Save PNG into dedicated mode folder
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
            demographics=demographics,
            generation=curr_gen,
        )
        fake_records.append(meta.to_dict())
        generated_count += 1

    fake_records.sort(key=lambda r: r["image_id"])
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        for r in fake_records:
            f.write(json.dumps(r) + "\n")

    logger.info(
        "Generation complete: %d synthetic faces saved to %s.", generated_count, out_manifest_path
    )
    return generated_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic deepfake faces via Stable Diffusion."
    )
    parser.add_argument("--config", type=str, default="configs/dataset_factory.yaml", help="Path to YAML config")
    parser.add_argument("--input-manifest", type=str, default=None)
    parser.add_argument("--output-manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model checkpoint or ID (e.g. runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock generator mode for offline testing",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "img2img", "inpaint", "text2img"],
        default=None,
        help="Generation mode (all, img2img, inpaint, text2img).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip generation if image file already exists on disk (default True).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Force overwrite/regenerate all images even if they already exist.",
    )
    parser.add_argument("--num-per-real", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Max real samples to synthesize from")
    parser.add_argument("--steps", type=int, default=None, help="Number of diffusion inference steps")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model_to_use = "mock_sd" if args.mock else args.model
    run_image_generation(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        output_dir=args.output_dir,
        model_name=model_to_use,
        num_per_real=args.num_per_real,
        max_samples=args.max_samples,
        steps=args.steps,
        use_mock_fallback=args.mock,
        mode=args.mode,
        resume=args.resume,
        seed=args.seed,
        config=args.config,
    )


if __name__ == "__main__":
    main()
