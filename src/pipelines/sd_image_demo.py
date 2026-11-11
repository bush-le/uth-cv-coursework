"""YAML-driven SD image demo with provenance and a local qualitative gallery."""

import argparse
import base64
import html
import importlib.metadata
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from src.common.logger import get_logger
from src.data.sd_demo_data import file_sha256, read_manifest, write_json
from src.pipelines.generate_deepfake_images import run_image_generation

logger = get_logger(__name__)


def build_demo_gallery(manifest: Path, destination: Path) -> dict[str, Any]:
    """Decode generated RGB PNGs (H,W,3); export technical QA and HTML.

    Sharpness is actual uint8 Laplacian variance, not CLIP or identity quality.
    No automatic anatomical/semantic acceptance is claimed by this pilot.
    """
    records = read_manifest(manifest)
    sections = []
    qa = []
    for record in records:
        image = Path(record["file_path"])
        with Image.open(image) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        qa.append(
            {
                "image_id": record["image_id"],
                "decode_ok": True,
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "laplacian_variance_uint8": float(
                    cv2.Laplacian(gray, cv2.CV_64F).var()
                ),
                "manual_qa": "pending",
                "face_landmarks": "not_evaluated",
                "clip_similarity": "not_evaluated",
            }
        )

        def figure(path: Path, title: str) -> str:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure>'

        pictures = figure(
            Path(record["source_snapshot"]), "Source (reference only for T2I)"
        )
        if record.get("mask_path"):
            pictures += figure(Path(record["mask_path"]), "Mask: white = edit")
        if record.get("target_snapshot"):
            pictures += figure(
                Path(record["target_snapshot"]), "Target: pose and scene"
            )
        pictures += figure(image, record["manipulation_type"])
        sections.append(
            f'<section><h2>{html.escape(record["image_id"])}</h2><p>{html.escape(record["prompt"])}</p><div class="pictures">{pictures}</div></section>'
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>SD image demo</title><style>body{font:16px/1.5 Segoe UI,Arial;background:#f6f6f1;color:#17242e;margin:35px}h1{font-size:30px}h2{font-size:17px;overflow-wrap:anywhere}section{border-top:1px solid #ccc;padding:20px 0}.pictures{display:flex;gap:20px;flex-wrap:wrap}figure{margin:0}img{width:300px;max-width:100%;height:auto}figcaption{font-size:14px;color:#555}</style><h1>Stable Diffusion / image pilot</h1><p>Real inference only when is_mock=false. T2I is text-only; it does not preserve source identity. Inpainting ROI must be reviewed. Technical QA is not the full curation gate. No detector/hard-sample claims.</p>'
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )
    write_json(destination.with_name("technical_qa.json"), qa)
    return {
        "images": len(records),
        "mock_outputs": sum(r["is_mock"] for r in records),
        "manual_qa": "pending",
    }


def run_demo(config_path: Path) -> dict[str, Any]:
    """Run the three configured modes, never detector inference or training."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data, generation = config["data"], config["generation"]
    engine_options = dict(config["engine"])
    model_name = config["model_name"]
    if config.get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError(
            "This demo requires Kaggle GPU. CPU testing needs an explicit override config."
        )
    if model_name == "mock_sd":
        raise ValueError(
            "The real demo does not accept mock_sd. Tests call the engine directly."
        )
    if config.get("pin_hub_revisions", True):
        from huggingface_hub import model_info

        for key, model in [
            ("revision", model_name),
            ("inpaint_revision", engine_options.get("inpaint_model_name", model_name)),
        ]:
            if not Path(model).exists():
                engine_options[key] = model_info(
                    model, revision=engine_options.get(key)
                ).sha
        if engine_options.get("unet_override"):
            override = dict(engine_options["unet_override"])
            override["revision"] = model_info(
                override["repo"], revision=override.get("revision")
            ).sha
            engine_options["unet_override"] = override
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = config | {"engine": engine_options}
    write_json(run_dir / "resolved_config.json", resolved)
    packages = {}
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "Pillow",
    ):
        packages[name] = importlib.metadata.version(name)
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    write_json(
        run_dir / "environment.json",
        {
            "python": platform.python_version(),
            "packages": packages,
            "git_revision": git.stdout.strip(),
            "source_sha256": {
                path.as_posix(): file_sha256(path)
                for path in sorted(Path("src").rglob("*.py"))
            },
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output_manifest = run_dir / "generated_manifest.jsonl"
    run_stats: dict[str, int] = {}
    count = run_image_generation(
        Path(data["input_manifest"]),
        output_manifest,
        run_dir / "images",
        model_name,
        num_per_real=generation["variants_per_mode"],
        seed=generation["seed"],
        modes=tuple(generation["modes"]),
        height=generation["height"],
        width=generation["width"],
        device=config["device"],
        data_root=Path(data.get("root", ".")),
        strength=generation["strength"],
        mask_box=tuple(generation["mask_box_xyxy_normalized"]),
        prompts=generation["prompts"],
        engine_options=engine_options,
        resume=True,
        run_stats=run_stats,
    )
    summary = build_demo_gallery(output_manifest, run_dir / "gallery.html") | {
        "count": count,
        **run_stats,
        "wall_seconds_including_load_and_qa": time.perf_counter() - start,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
        "detector_evaluated": False,
        "full_curation_evaluated": False,
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("Demo artifacts: %s", run_dir)
    return summary


def main() -> None:
    """CLI for the reproducible Kaggle image pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/generation/kaggle_image_demo.yaml")
    )
    args = parser.parse_args()
    run_demo(args.config)


if __name__ == "__main__":
    main()
