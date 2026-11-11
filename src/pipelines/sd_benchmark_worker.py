"""One isolated benchmark task; stdout/stderr are captured by the queue."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from src.common.seed import seed_everything
from src.data.image_schema import ImageMetadata
from src.data.sd_demo_data import (
    file_sha256,
    load_rgb,
    read_manifest,
    resolve_image_path,
    write_json,
)
from src.models.diffusion.reference_backends import (
    checkout_backend,
    load_reference_backend,
)
from src.models.diffusion.stable_diffusion_engine import tensor_to_pil
from src.pipelines.sd_image_demo import build_demo_gallery, run_demo


def run_reference_case(payload: dict[str, Any]) -> dict[str, Any]:
    """Run real reference conditioning with CPU RGB sources (3,H,W), [0,1]."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Reference benchmark requires Kaggle CUDA; CPU is not a mock substitute."
        )
    case = payload["case"]
    output = Path(payload["run_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
            "packages": {
                dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")
            },
        },
    )
    manifest_path = Path(payload["input_manifest"]).resolve()
    source_items = read_manifest(manifest_path)
    if case["backend"] in {"reface", "face_adapter"} and len(source_items) < 2:
        raise ValueError("Face swapping needs at least two distinct input images.")
    source_paths = [
        resolve_image_path(r["file_path"], manifest_path) for r in source_items
    ]
    # Resolve everything before upstream code changes cwd.
    repo = checkout_backend(
        case["backend"],
        Path(payload["vendor_root"]).resolve(),
        case.get("upstream_revision"),
    )
    revision = (
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if repo
        else None
    )
    runtime = output / "_runtime"
    write_json(
        output / "resolved_backend.json",
        {
            "case": case,
            "upstream_revision": revision,
            "input_manifest_sha256": file_sha256(manifest_path),
        },
    )
    seed_everything(payload["seed"])
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    backend = load_reference_backend(case, repo, runtime)
    records = []
    for i, source_item in enumerate(source_items[: payload["source_limit"]]):
        target_index = (i + 1) % len(source_items)
        source = tensor_to_pil(load_rgb(source_paths[i], (case["size"], case["size"])))
        target = tensor_to_pil(
            load_rgb(source_paths[target_index], (case["size"], case["size"]))
        )
        seed = payload["seed"] + i
        prompt = payload["prompt"]
        if case["backend"] == "photomaker":
            prompt = "a photo of an adult person img, studio lighting, natural skin"
        elif case["backend"] == "fastcomposer":
            prompt = "an adult person <|image|> in studio lighting"
        signature = {
            "case": case,
            "source_sha256": file_sha256(source_paths[i]),
            "target_sha256": file_sha256(source_paths[target_index]),
            "seed": seed,
            "prompt": prompt,
            "upstream_revision": revision,
        }
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        image_id = f"{case['id']}_{source_item['image_id']}_{key[:10]}"
        images = output / "images"
        images.mkdir(exist_ok=True)
        source_snapshot = images / f"{image_id}_source.png"
        target_snapshot = images / f"{image_id}_target.png"
        result_path = images / f"{image_id}.png"
        source.save(source_snapshot)
        target.save(target_snapshot)
        begin = time.perf_counter()
        seed_everything(seed)
        with torch.inference_mode():
            result = backend(source, target, prompt, seed)
        result.convert("RGB").save(result_path)
        with result_path.open("rb") as stream:
            from PIL import Image

            with Image.open(stream) as image:
                image.verify()
        swapping = case["backend"] in {"reface", "face_adapter"}
        record = ImageMetadata(
            image_id=image_id,
            label=1,
            manipulation_type=case["backend"],
            prompt=prompt,
            source_real_id=source_item["image_id"],
            file_path=result_path.as_posix(),
            data_source="synthetic",
        ).to_dict() | {
            "source_snapshot": source_snapshot.as_posix(),
            "target_snapshot": target_snapshot.as_posix() if swapping else None,
            "target_image_id": source_items[target_index]["image_id"]
            if swapping
            else None,
            "mask_path": None,
            "is_mock": False,
            "split": "demo_only",
            "generation_seed": seed,
            "job_hash": key,
            "output_sha256": file_sha256(result_path),
            "model_name": case["model_name"],
            "upstream_revision": revision,
            "elapsed_seconds_including_inference": time.perf_counter() - begin,
            "requested_size": case["size"],
            "native_output_size": list(result.size),
            "qa_status": "pending",
            "task_family": "face_swap" if swapping else "reference_generation",
        }
        records.append(record)
        manifest = output / "generated_manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
    summary = build_demo_gallery(
        output / "generated_manifest.jsonl", output / "gallery.html"
    ) | {
        "newly_generated": len(records),
        "resumed": 0,
        "wall_seconds_including_load_and_qa": time.perf_counter() - start,
        "cuda_peak_allocated_bytes": None
        if case["backend"] in {"face_adapter", "reface", "fastcomposer"}
        else torch.cuda.max_memory_allocated(),
        "cuda_peak_measurement_scope": "worker_process_only; CLI child peaks are not measured",
        "detector_evaluated": False,
        "full_curation_evaluated": False,
    }
    write_json(output / "summary.json", summary)
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    """Execute the task's frozen config, never silently skip a backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    output = Path(payload["run_dir"]).resolve()
    try:
        if payload["case"]["backend"] == "diffusers":
            config_file = output / "config.yaml"
            run_demo(config_file)
        else:
            run_reference_case(payload)
    finally:
        # Upstream code may download additional models internally. Record what
        # actually landed, including revisions, rather than claiming all of
        # those transitive downloads were pinned by this wrapper.
        from huggingface_hub import scan_cache_dir

        cache = (
            Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
            / "hub"
        )
        revisions = []
        if cache.is_dir():
            info = scan_cache_dir(cache)
            revisions = [
                {
                    "repo_id": repo.repo_id,
                    "commits": sorted(rev.commit_hash for rev in repo.revisions),
                }
                for repo in info.repos
            ]
        local_metadata = []
        for path in (output / "_runtime").rglob("*.metadata"):
            if "huggingface" in path.parts:
                local_metadata.append(
                    {
                        "path": path.relative_to(output).as_posix(),
                        "metadata": path.read_text(encoding="utf-8", errors="replace"),
                    }
                )
        write_json(
            output / "download_provenance.json",
            {
                "hub_snapshots": revisions,
                "local_downloads": local_metadata,
                "transitive_downloads_pre_pinned": False,
            },
        )


if __name__ == "__main__":
    main()
