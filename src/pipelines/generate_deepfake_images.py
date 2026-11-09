"""Phase 1: real source images -> native SD modes -> images and provenance."""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_schema import ImageMetadata
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

logger = get_logger(__name__)
MODES = ("text2img", "img2img", "inpaint")
PROMPTS = {
    "text2img": "a photorealistic headshot portrait of an adult person, studio lighting, natural skin",
    "img2img": "a photorealistic headshot portrait of an adult person, warm studio lighting, natural skin",
    "inpaint": "a natural smiling mouth, realistic lips and skin texture",
}


def _save_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically save completed outputs only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_image_generation(
    input_manifest: Path,
    output_manifest: Path,
    output_dir: Path,
    model_name: str = DEFAULT_SD,
    mock_cache: dict[str, torch.Tensor] | None = None,
    num_per_real: int = 1,
    seed: int = 42,
    *,
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
    """Generate num_per_real outputs per source per selected mode.

    Source tensors are RGB float32 (3,H,W), edit masks (1,H,W), [0,1].
    Real runs require decodable source files. Mock caches work only when the
    caller explicitly selects mock_sd. Outputs and job sidecars use SHA256
    fingerprints for resume; completed records retain ImageMetadata fields.
    """
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
    resolved_prompts = PROMPTS | (prompts or {})
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
    # Validate all inputs before loading any model or starting generation.
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
            # Cached fixtures also obey requested output dimensions.
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
                            _save_manifest(output_manifest, records)
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
                    # Source/mask snapshots make qualitative review unambiguous.
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
                    _save_manifest(output_manifest, records)
    except Exception as exc:
        if "sidecar" in locals():
            write_json(
                sidecar,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    finally:
        engine.close()
        if run_stats is not None:
            run_stats.update(stats)
    logger.info("Saved %d completed outputs to %s", len(records), output_manifest)
    return len(records)


def main() -> None:
    """Run explicit CLI options or the YAML-driven Kaggle demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--input-manifest", type=Path, default=Path("data/raw/manifest.jsonl")
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/fake_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/synthetic_faces")
    )
    parser.add_argument("--model", default=DEFAULT_SD)
    parser.add_argument("--num-per-real", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.config:
        from src.pipelines.sd_image_demo import run_demo

        run_demo(args.config)
    else:
        run_image_generation(
            args.input_manifest,
            args.output_manifest,
            args.output_dir,
            args.model,
            num_per_real=args.num_per_real,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
