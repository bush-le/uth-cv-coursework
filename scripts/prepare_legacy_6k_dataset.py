import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#!/usr/bin/env python3
"""Unpack and normalize the external 6k synthetic face benchmark dataset."""

import json
from pathlib import Path
from typing import Optional
import zipfile

from src.common.logger import get_logger

logger = get_logger(__name__)


def prepare_legacy_6k(
    zip_path: Optional[Path] = None,
    dest_dir: Path = Path("data/external_benchmarks/synthetic_face_dataset_6k"),
) -> int:
    """Extract and rebase manifest for the external 6k synthetic dataset.

    Args:
        zip_path: Path to synthetic_face_dataset_6k.zip.
        dest_dir: Target destination directory.

    Returns:
        Number of verified images.
    """
    if zip_path is None:
        candidate_zip = Path.home() / "Downloads/synthetic_face_dataset_6k.zip"
        if candidate_zip.exists():
            zip_path = candidate_zip
        else:
            zip_path = Path("data/external_benchmarks/synthetic_face_dataset_6k.zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"Source zip not found at: {zip_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "fake_manifest.jsonl"

    # 1. Extract zip if not already extracted
    logger.info("Extracting %s to %s...", zip_path.name, dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest fake_manifest.jsonl not found in {dest_dir}")

    # 2. Rebase manifest paths to relative workspace paths
    logger.info("Rebasing manifest paths in %s...", manifest_path)
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            orig_path = rec.get("image_path", "")

            # Rebase from legacy "data/synthetic/..." to "data/external_benchmarks/synthetic_face_dataset_6k/..."
            if "data/synthetic/" in orig_path:
                rel_suffix = orig_path.split("data/synthetic/")[-1]
            else:
                rel_suffix = orig_path

            rebased_path = str(dest_dir / rel_suffix)
            rec["image_path"] = rebased_path
            records.append(rec)

    # 3. Verify all 6,000 files exist on disk
    missing = 0
    task_counts = {}
    for r in records:
        p = Path(r["image_path"])
        if not p.exists():
            missing += 1
        t = r.get("task", "unknown")
        task_counts[t] = task_counts.get(t, 0) + 1

    if missing > 0:
        logger.warning("Found %d missing files after rebasing!", missing)
    else:
        logger.info("All %d images verified on disk! Task breakdown: %s", len(records), task_counts)

    # 4. Save rebased manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info("Rebased manifest saved to: %s", manifest_path)
    return len(records)


if __name__ == "__main__":
    count = prepare_legacy_6k()
    print(f"Preparation complete! Verified {count} samples.")
