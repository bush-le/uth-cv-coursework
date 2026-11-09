"""Generate a visual comparison gallery: Real Faces vs Synthetic Faces vs Mined Hard Samples."""

import argparse
import json
from pathlib import Path
from typing import List

from src.common.logger import get_logger
from src.common.visualizer import plot_real_vs_synthetic_comparison

logger = get_logger(__name__)


def extract_paths_from_manifest(manifest_path: Path, max_count: int = 4) -> List[str]:
    """Read existing image file paths from a JSONL manifest."""
    paths: List[str] = []
    if not manifest_path.exists():
        return paths

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            fp = item.get("file_path")
            if fp and Path(fp).exists():
                paths.append(fp)
                if len(paths) >= max_count:
                    break
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparison gallery for Topic 11.")
    parser.add_argument("--real-manifest", type=str, default="data/raw/manifest.jsonl")
    parser.add_argument("--synthetic-manifest", type=str, default="data/synthetic/curated_manifest.jsonl")
    parser.add_argument("--hard-manifest", type=str, default="data/hard_samples/hard_samples_manifest.jsonl")
    parser.add_argument(
        "--output-image",
        type=str,
        default="experiments/predictions/real_vs_synthetic_gallery.png",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    real_manifest = Path(args.real_manifest)
    if not real_manifest.exists() and Path("data/01_raw_faces/raw_faces_manifest.jsonl").exists():
        real_manifest = Path("data/01_raw_faces/raw_faces_manifest.jsonl")

    syn_manifest = Path(args.synthetic_manifest)
    if not syn_manifest.exists() and Path("data/02_synthetic_sd/manifests/curated_synthetic.jsonl").exists():
        syn_manifest = Path("data/02_synthetic_sd/manifests/curated_synthetic.jsonl")

    hard_manifest = Path(args.hard_manifest)
    if not hard_manifest.exists() and Path("data/03_hard_samples/hard_samples_manifest.jsonl").exists():
        hard_manifest = Path("data/03_hard_samples/hard_samples_manifest.jsonl")

    real_paths = extract_paths_from_manifest(real_manifest, args.num_samples)
    synthetic_paths = extract_paths_from_manifest(syn_manifest, args.num_samples)
    hard_paths = extract_paths_from_manifest(hard_manifest, args.num_samples)

    logger.info(
        "Extracted paths for gallery: %d real, %d synthetic, %d hard samples",
        len(real_paths),
        len(synthetic_paths),
        len(hard_paths),
    )

    plot_real_vs_synthetic_comparison(
        real_paths=real_paths,
        synthetic_paths=synthetic_paths,
        hard_paths=hard_paths,
        output_path=args.output_image,
        num_samples=args.num_samples,
    )
    logger.info("Comparison gallery successfully saved to %s", args.output_image)


if __name__ == "__main__":
    main()
