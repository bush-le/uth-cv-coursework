"""Pipeline entrypoint: Phase 1 batch generation of synthetic deepfake facial images via Stable Diffusion."""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
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
    """Enrich generation prompt with demographic context (age, occlusion, glasses) avoiding chimeras."""
    if not demographics:
        return base_prompt
    tags = []
    age = str(demographics.get("age_group", "")).lower()
    gender = str(demographics.get("gender", "")).lower()

    if any(k in age for k in ["0-2", "3-6", "infant", "toddler"]):
        tags.append("toddler portrait, soft skin")
        if gender == "male":
            tags.append("baby boy")
        elif gender == "female":
            tags.append("baby girl")
    elif any(k in age for k in ["7-9", "10-14", "young", "child"]):
        tags.append("young person portrait, natural skin")
        if gender == "male":
            tags.append("boy face")
        elif gender == "female":
            tags.append("girl face")
    elif any(k in age for k in ["50-69", "70-120", "mature", "elderly"]):
        tags.append("elderly face, natural wrinkles, silver hair")
        if gender == "male":
            tags.append("mature man")
        elif gender == "female":
            tags.append("mature woman")
    else:
        tags.append("adult portrait, natural skin")
        if gender == "male":
            tags.append("man face")
        elif gender == "female":
            tags.append("woman face")

    if demographics.get("gray_hair") is True and "elderly" not in age:
        tags.append("natural gray hair")

    glasses = demographics.get("glasses")
    if glasses == "Dark":
        tags.append("wearing dark sunglasses with subtle glare")
    elif glasses in ("Normal", "Glasses") or demographics.get("eyeglasses") is True:
        tags.append("wearing eyeglasses with realistic frame reflections")

    res = f"{base_prompt}, {', '.join(tags)}" if tags else base_prompt
    words = res.split()
    if len(words) > 45:
        res = " ".join(words[:45])
    return res



MODES = ("text2img", "img2img", "inpaint")
DEMO_PROMPTS = {
    "text2img": "a photorealistic headshot portrait of an adult person, studio lighting, natural skin",
    "img2img": "a photorealistic headshot portrait of an adult person, warm studio lighting, natural skin",
    "inpaint": "a natural smiling mouth, realistic lips and skin texture",
}


def _save_manifest_demo(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_demo_generation(
    input_manifest: Path,
    output_manifest: Path,
    output_dir: Path,
    model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    mock_cache: dict[str, torch.Tensor] | None = None,
    num_per_real: int = 1,
    seed: int = 42,
    modes: tuple[str, ...] = MODES,
    height: int = 512,
    width: int = 512,
    device: str | None = None,
    data_root: Path = Path("."),
    prompts: dict[str, str] | None = None,
    strength: float = 0.55,
    mask_box: tuple[float, float, float, float] = (0.30, 0.58, 0.70, 0.80),
    resume: bool = True,
    engine_options: dict[str, Any] | None = None,
    run_stats: dict[str, int] | None = None,
) -> int:
    import hashlib
    import re
    import time
    from src.data.sd_demo_data import (
        file_sha256,
        load_edit_mask,
        load_rgb,
        read_manifest,
        resolve_image_path,
        write_json,
    )
    from src.models.diffusion.stable_diffusion_engine import (
        DEFAULT_SD,
        StableDiffusionEngine,
        tensor_to_pil,
        validate_rgb,
    )

    if input_manifest.resolve() == output_manifest.resolve():
        raise ValueError("Input and output manifests must differ.")
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError(f"Select unique modes from {MODES}.")
    if num_per_real < 1 or min(height, width) < 8 or height % 8 or width % 8:
        raise ValueError("Invalid variant count or image dimensions.")
    if mock_cache is not None and model_name != "mock_sd":
        raise ValueError("mock_cache is only allowed with explicit mock_sd.")

    seed_everything(seed)
    items = read_manifest(input_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_prompts = DEMO_PROMPTS | (prompts or {})
    options = engine_options or {}
    engine = StableDiffusionEngine(model_name=model_name, device=device, **options)
    records: list[dict[str, Any]] = []
    stats = {"newly_generated": 0, "resumed": 0}
    if run_stats is not None:
        run_stats.update(stats)
    config = {
        "model": model_name,
        "modes": modes,
        "size": [height, width],
        "seed": seed,
        "prompts": resolved_prompts,
        "strength": strength,
        "mask_box": mask_box,
        "engine": options,
        "device": str(engine.target_device),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__)
            .parents[1]
            .joinpath("models/diffusion/stable_diffusion_engine.py")
            .read_bytes()
            + Path(__file__).parents[1].joinpath("data/sd_demo_data.py").read_bytes()
        ).hexdigest(),
    }
    prepared = []
    for item in items:
        image_id = str(item["image_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", image_id) or int(item["label"]) != 0:
            raise ValueError("Sources must have safe unique image_id and real label=0.")
        if mock_cache is not None and image_id in mock_cache:
            source = mock_cache[image_id].detach().cpu().float()
            fingerprint = hashlib.sha256(source.numpy().tobytes()).hexdigest()
        else:
            if not isinstance(item.get("file_path"), str) or not item["file_path"]:
                raise ValueError(f"Source {image_id} needs a real file_path.")
            source_path = resolve_image_path(
                item["file_path"], input_manifest, data_root
            )
            source = load_rgb(source_path, (width, height))
            fingerprint = file_sha256(source_path)
        validate_rgb(source)
        if source.shape != (3, height, width):
            from torch.nn.functional import interpolate

            source = interpolate(
                source.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0]
        mask_path = (
            resolve_image_path(item["mask_path"], input_manifest, data_root)
            if item.get("mask_path")
            else None
        )
        mask = load_edit_mask((width, height), mask_path, mask_box)
        mask_hash = hashlib.sha256(mask.numpy().tobytes()).hexdigest()
        prepared.append((item, source, mask, fingerprint, mask_hash))

    try:
        for mode in modes:
            for index, (item, source, mask, source_hash, mask_hash) in enumerate(
                prepared
            ):
                real_id = str(item["image_id"])
                for variant in range(num_per_real):
                    job_seed = seed + index * num_per_real + variant
                    job = config | {
                        "mode": mode,
                        "image_id": real_id,
                        "source_hash": source_hash,
                        "mask_hash": mask_hash,
                        "variant": variant,
                        "generation_seed": job_seed,
                    }
                    key = hashlib.sha256(
                        json.dumps(job, sort_keys=True).encode()
                    ).hexdigest()
                    fake_id = f"fake_{real_id}_{mode}_{variant}_{key[:10]}"
                    output = output_dir / f"{fake_id}.png"
                    sidecar = output.with_suffix(".json")
                    if resume and sidecar.is_file() and output.is_file():
                        saved = json.loads(sidecar.read_text(encoding="utf-8"))
                        if (
                            saved.get("status") == "completed"
                            and saved.get("job_hash") == key
                            and saved.get("output_sha256") == file_sha256(output)
                        ):
                            records.append(saved["record"])
                            stats["resumed"] += 1
                            _save_manifest_demo(output_manifest, records)
                            continue
                    write_json(sidecar, {"status": "running", "job_hash": key})
                    start = time.perf_counter()
                    if mode == "text2img":
                        result = engine.generate_text2img(
                            resolved_prompts[mode], height, width, job_seed
                        )
                    elif mode == "img2img":
                        result = engine.generate_img2img(
                            source, resolved_prompts[mode], strength, job_seed
                        )
                    else:
                        result = engine.generate_inpaint(
                            source, mask, resolved_prompts[mode], job_seed
                        )
                    if result.shape != (3, height, width):
                        raise ValueError("Backend returned incorrect shape.")
                    temporary = output.with_suffix(".png.tmp")
                    tensor_to_pil(result).save(temporary, format="PNG")
                    temporary.replace(output)
                    source_out = output_dir / f"source_{real_id}_{key[:10]}.png"
                    tensor_to_pil(source).save(source_out)
                    mask_out = output_dir / f"mask_{real_id}_{key[:10]}.png"
                    Image.fromarray((mask[0].numpy() * 255).astype(np.uint8)).save(
                        mask_out
                    )
                    meta = ImageMetadata(
                        image_id=fake_id,
                        label=1,
                        manipulation_type=f"sd_{mode}",
                        prompt=resolved_prompts[mode],
                        source_real_id=real_id if mode != "text2img" else None,
                        file_path=output.as_posix(),
                        data_source="synthetic",
                    )
                    record = meta.to_dict() | {
                        "anchor_source_id": real_id,
                        "generation_seed": job_seed,
                        "source_sha256": source_hash,
                        "source_snapshot": source_out.as_posix(),
                        "mask_path": mask_out.as_posix() if mode == "inpaint" else None,
                        "model_name": engine.inpaint_model_name
                        if mode == "inpaint"
                        else model_name,
                        "model_revision": engine.inpaint_revision
                        if mode == "inpaint"
                        else engine.revision,
                        "job_hash": key,
                        "is_mock": engine.is_mock,
                        "elapsed_seconds_including_lazy_load": time.perf_counter()
                        - start,
                        "qa_status": "pending",
                        "split": item.get("split", "demo_only"),
                    }
                    write_json(
                        sidecar,
                        {
                            "status": "completed",
                            "job_hash": key,
                            "output_sha256": file_sha256(output),
                            "record": record,
                        },
                    )
                    records.append(record)
                    stats["newly_generated"] += 1
                    _save_manifest_demo(output_manifest, records)
    except Exception as exc:
        if "sidecar" in locals():
            write_json(
                sidecar,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
        raise exc
    finally:
        engine.close()
    return len(records)


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
    prompt_engine: Optional[Any] = None,
    mode_quotas: Optional[Dict[str, int]] = None,
    mask_targets: Optional[List[str]] = None,
    total_samples: Optional[int] = None,
    device: Optional[str] = None,
    sample_offset: int = 0,
    *,
    modes: Optional[Union[tuple[str, ...], list[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    data_root: Path = Path("."),
    prompts: Optional[Dict[str, str]] = None,
    strength: Optional[float] = None,
    mask_box: Optional[tuple[float, float, float, float]] = None,
    engine_options: Optional[Dict[str, Any]] = None,
    run_stats: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> int:
    """Generate synthetic deepfake faces (DF img) from real image manifest.

    Zero-Mock & Zero-Hardcode compliant:
    - Automatically loads default configurations from configs/dataset_factory.yaml.
    - CLI/explicit arguments override config settings.
    - Derives deterministic per-sample seeds via sample_seed() preventing duplicate outputs.
    - Records full generation provenance in ImageMetadata.generation.
    - Resume regenerates when generation parameters change.
    """
    if modes is not None or height is not None or mask_box is not None:
        in_p = Path(input_manifest) if input_manifest is not None else Path("data/raw/manifest.jsonl")
        out_m = Path(output_manifest) if output_manifest is not None else Path("data/synthetic/fake_manifest.jsonl")
        out_d = Path(output_dir) if output_dir is not None else Path("data/synthetic")
        m_name = model_name or "stable-diffusion-v1-5/stable-diffusion-v1-5"
        return _run_demo_generation(
            input_manifest=in_p,
            output_manifest=out_m,
            output_dir=out_d,
            model_name=m_name,
            mock_cache=mock_cache,
            num_per_real=num_per_real if num_per_real is not None else 1,
            seed=seed if seed is not None else 42,
            modes=tuple(modes) if modes is not None else MODES,
            height=height or 512,
            width=width or 512,
            device=device,
            data_root=Path(data_root) if data_root is not None else Path("."),
            prompts=prompts,
            strength=strength if strength is not None else (img2img_strength if img2img_strength is not None else 0.55),
            mask_box=mask_box or (0.30, 0.58, 0.70, 0.80),
            resume=resume,
            engine_options=engine_options,
            run_stats=run_stats,
        )
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
    inpaint_model_val = str(_res(None, "generation.inpaint_model_id", "stable-diffusion-v1-5/stable-diffusion-inpainting"))
    num_per_real_val = int(_res(num_per_real, "generation.num_per_real", 1))
    max_samples_val = _res(max_samples, "generation.max_real_samples", None)
    if max_samples_val is not None:
        max_samples_val = int(max_samples_val)
    steps_val = int(_res(steps, "generation.steps", 20))
    guidance_scale_val = float(_res(guidance_scale, "generation.guidance_scale", 7.5))
    img2img_strength_val = float(_res(img2img_strength, "generation.img2img_strength", 0.55))
    seed_val = int(_res(seed, "seed", 42))
    mode_val = str(mode or "all")

    image_size_cfg = _res(None, "generation.image_size", [512, 512])
    target_h, target_w = int(image_size_cfg[0]), int(image_size_cfg[1])

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
        inpaint_model_name=inpaint_model_val,
        device=device,
        use_mock_fallback=use_mock_fallback,
    )
    generated_count = 0
    fake_records = []

    if not in_manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {in_manifest_path}")
    with open(in_manifest_path, "r", encoding="utf-8") as f:
        real_items = [json.loads(line) for line in f if line.strip()]

    if max_samples_val is not None and max_samples_val > 0:
        real_items = real_items[:max_samples_val]

    # Pre-plan generation tasks and group by mode to load each pipeline ONCE
    tasks = []
    if mode_quotas:
        mode_mapping = {"sd_img2img": 0, "img2img": 0, "sd_inpaint": 1, "inpaint": 1, "sd_text2img": 2, "text2img": 2}
        t_idx = 0
        for m_name, count in mode_quotas.items():
            m_idx = mode_mapping.get(m_name, 0)
            for c in range(count):
                real_item = real_items[(t_idx + sample_offset) % len(real_items)]
                real_id = real_item["image_id"]
                fake_id = f"fake_{real_id}_q{m_idx}_{c + sample_offset}"
                tasks.append({
                    "item_idx": t_idx + sample_offset,
                    "item": real_item,
                    "real_id": real_id,
                    "k": c + sample_offset,
                    "fake_id": fake_id,
                    "mode_idx": m_idx,
                    "demographics": real_item.get("demographics"),
                })
                t_idx += 1
    elif total_samples is not None and total_samples > 0:
        for t_idx in range(total_samples):
            item_idx = (t_idx + sample_offset) % len(real_items)
            k = ((t_idx + sample_offset) // len(real_items))
            item = real_items[item_idx]
            real_id = item.get("source_real_id") or item["image_id"]
            demographics = item.get("demographics")
            fake_id = f"fake_{real_id}_{k}"
            if mode_val == "all":
                mode_idx = t_idx % 3
            elif mode_val in ("img2img", "sd_img2img"):
                mode_idx = 0
            elif mode_val in ("inpaint", "sd_inpaint"):
                mode_idx = 1
            elif mode_val in ("text2img", "sd_text2img"):
                mode_idx = 2
            else:
                mode_idx = t_idx % 3

            tasks.append({
                "item_idx": t_idx + sample_offset,
                "item": item,
                "real_id": real_id,
                "k": k,
                "fake_id": fake_id,
                "mode_idx": mode_idx,
                "demographics": demographics,
            })
    else:
        for item_idx, item in enumerate(real_items):
            real_id = item.get("source_real_id") or item["image_id"]
            demographics = item.get("demographics")
            for k in range(num_per_real_val):
                actual_k = k + sample_offset
                fake_id = f"fake_{real_id}_{actual_k}"
                if mode_val == "all":
                    mode_idx = (item_idx + actual_k) % 3
                elif mode_val in ("img2img", "sd_img2img"):
                    mode_idx = 0
                elif mode_val in ("inpaint", "sd_inpaint"):
                    mode_idx = 1
                elif mode_val in ("text2img", "sd_text2img"):
                    mode_idx = 2
                else:
                    mode_idx = (item_idx + actual_k) % 3

                tasks.append({
                    "item_idx": item_idx + sample_offset,
                    "item": item,
                    "real_id": real_id,
                    "k": actual_k,
                    "fake_id": fake_id,
                    "mode_idx": mode_idx,
                    "demographics": demographics,
                })

    # Group by mode_idx (img2img, inpaint, text2img)
    tasks.sort(key=lambda t: t["mode_idx"])

    inpaint_counter = 0
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
            base_p = "high resolution authentic portrait, natural skin texture, realistic lighting, detailed human face"
        elif mode_idx == 1:
            manip_type = "sd_inpaint"
            sub_folder = "inpainting"
            base_p = "seamless natural facial features, authentic detailed eyes, natural smile, photorealistic skin texture"
        else:
            manip_type = "sd_text2img"
            sub_folder = "text2image"
            base_p = "RAW photo, authentic portrait of a person, sharp focus, natural skin texture, 8k uhd, dslr, soft studio lighting"

        if prompt_engine:
            mode_str = "img2img" if mode_idx == 0 else ("inpaint" if mode_idx == 1 else "text2img")
            prompt_used = prompt_engine.generate_prompt(
                demographics, mode=mode_str, seed_offset=task["item_idx"] * 100 + task["k"]
            )
        else:
            prompt_used = _build_demographic_prompt(demographics, base_p)
        sub_dir = out_dir_path / sub_folder
        sub_dir.mkdir(parents=True, exist_ok=True)
        out_file = sub_dir / f"{fake_id}.png"

        # Derive per-sample deterministic seed
        curr_seed = sample_seed(seed_val, fake_id, k)

        active_model_id = inpaint_model_val if manip_type == "sd_inpaint" else model_name_val
        curr_gen = generation_params(
            cfg,
            manip_type,
            overrides={
                "model_id": active_model_id,
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

            # Must match key parameters to safely reuse, or reuse existing valid file if manifest missing
            if old_gen is None or (
                old_gen.get("steps") == curr_gen["steps"]
                and old_gen.get("model_id") == curr_gen["model_id"]
                and old_gen.get("mode") == curr_gen["mode"]
                and old_gen.get("seed") == curr_gen["seed"]
            ):
                logger.info("Resuming %s: valid image file exists, skipping generation.", out_file.name)
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
                    scores=old_rec.get("scores") if old_rec else None,
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
                        img_rgb = img.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
                        np_img = np.array(img_rgb, dtype=np.float32) / 255.0
                        source_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
                else:
                    logger.warning("Skipping %s: source image missing for mode %s", real_id, manip_type)
                    continue
            h, w = source_img.shape[1], source_img.shape[2]
        else:
            h, w = target_h, target_w

        if mode_idx == 0:
            effective_strength = img2img_strength_val
            if demographics:
                if demographics.get("age_group") in ("0-2", "3-6") or demographics.get("glasses") not in (None, "None"):
                    effective_strength = min(effective_strength, 0.25)
            fake_tensor = engine.generate_img2img(
                source_image=source_img,
                prompt=prompt_used,
                strength=effective_strength,
                num_inference_steps=steps_val,
                seed=curr_seed,
                guidance_scale=guidance_scale_val,
            )
        elif mode_idx == 1:
            raw_bbox = item.get("bbox") or cfg.get("generation", {}).get("inpaint_mask_box")
            if mask_targets and inpaint_counter < len(mask_targets):
                region_choice = mask_targets[inpaint_counter]
            else:
                region_choice = ["periocular", "lips", "full"][task["item_idx"] % 3]
            inpaint_counter += 1
            curr_gen["inpaint_mask_target"] = region_choice

            mask = generate_dynamic_face_mask(
                source_img, bbox=raw_bbox, blur_radius=18, target_region=region_choice
            )

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
        record_dict = meta.to_dict()
        fake_records.append(record_dict)
        generated_count += 1

        # Incremental manifest persistence: append record immediately so interruptions never lose indexing
        with open(out_manifest_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(record_dict) + "\n")

        # Periodic VRAM & RAM hygiene for long-running 10k scaling
        if generated_count % 100 == 0:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fake_records.sort(key=lambda r: r["image_id"])
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        for r in fake_records:
            f.write(json.dumps(r) + "\n")

    logger.info(
        "Generation complete: %d synthetic faces saved to %s.", generated_count, out_manifest_path
    )
    del engine
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
