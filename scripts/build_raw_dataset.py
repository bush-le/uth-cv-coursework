#!/usr/bin/env python3
"""CLI utility to index and build baseline manifest for real face datasets (FFHQ & CelebA-HQ).

Usage:
    python scripts/build_raw_dataset.py
    python scripts/build_raw_dataset.py --max-ffhq 2000 --max-celeba 2000
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.logger import get_logger
from src.data.raw_ingestion import build_baseline_manifest

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build baseline manifest from raw FFHQ and CelebA-HQ faces."
    )
    parser.add_argument(
        "--ffhq-dir",
        type=Path,
        default=Path("data/raw/faces/ffhq"),
        help="Directory containing FFHQ face images (*.png).",
    )
    parser.add_argument(
        "--celeba-dir",
        type=Path,
        default=Path("data/raw/faces/celeba_hq"),
        help="Directory containing CelebA-HQ face images (*.jpg).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/raw/metadata"),
        help="Directory containing annotation files (CSV, TXT).",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/raw/manifest.jsonl"),
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--max-ffhq",
        type=int,
        default=2000,
        help="Maximum FFHQ images to index (default: 2000).",
    )
    parser.add_argument(
        "--max-celeba",
        type=int,
        default=2000,
        help="Maximum CelebA-HQ images to index (default: 2000).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute raw dataset indexing and manifest assembly."""
    args = parse_args()
    logger.info("Building raw baseline manifest...")
    items = build_baseline_manifest(
        ffhq_dir=args.ffhq_dir,
        celeba_dir=args.celeba_dir,
        metadata_dir=args.metadata_dir,
        output_manifest=args.output_manifest,
        max_ffhq=args.max_ffhq,
        max_celeba=args.max_celeba,
    )
    logger.info("Done. Generated %d items in %s.", len(items), args.output_manifest)


if __name__ == "__main__":
    main()
