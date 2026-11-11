"""Ingestion utilities for real face datasets (FFHQ and CelebA-HQ).

Constructs unified ImageMetadata entries with demographic annotations and
strict identity-disjoint mapping for deepfake training and evaluation.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.data.image_schema import ImageMetadata

logger = get_logger(__name__)


def parse_ffhq_aging_metadata(
    csv_path: Path,
    max_samples: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Parse FFHQ demographic aging and occlusion metadata.

    Args:
        csv_path: Path to ffhq_aging_labels.csv.
        max_samples: Optional maximum number of images to index.

    Returns:
        Dictionary mapping image_number (int) to demographic dictionary.
    """
    records: Dict[int, Dict[str, Any]] = {}
    if not csv_path.exists():
        logger.warning("FFHQ aging metadata not found at %s", csv_path)
        return records

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                num = int(row["image_number"])
            except (ValueError, KeyError):
                continue

            if max_samples is not None and num >= max_samples:
                break

            records[num] = {
                "dataset": "ffhq",
                "age_group": row.get("age_group", "unknown"),
                "gender": row.get("gender", "unknown"),
                "glasses": row.get("glasses", "None"),
                "yaw": float(row["head_yaw"]) if "head_yaw" in row else 0.0,
                "pitch": float(row["head_pitch"]) if "head_pitch" in row else 0.0,
                "roll": float(row["head_roll"]) if "head_roll" in row else 0.0,
            }

    logger.info("Parsed %d FFHQ demographic records from %s", len(records), csv_path)
    return records


def parse_celeba_metadata(
    mapping_path: Path,
    identity_path: Path,
    attribute_path: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """Parse CelebA-HQ identity mapping and demographic attributes.

    Args:
        mapping_path: Path to CelebA-HQ-to-CelebA-mapping.txt.
        identity_path: Path to identity_CelebA.txt.
        attribute_path: Optional path to CelebAMask-HQ-attribute-anno.txt.
        max_samples: Optional maximum number of images to index.

    Returns:
        Dictionary mapping CelebA-HQ index (int) to metadata dictionary.
    """
    identity_map: Dict[str, str] = {}
    if identity_path.exists():
        with open(identity_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    identity_map[parts[0]] = parts[1]

    # Optional attribute parsing
    attributes_by_file: Dict[str, Dict[str, bool]] = {}
    if attribute_path and attribute_path.exists():
        with open(attribute_path, "r", encoding="utf-8") as f:
            _ = f.readline()  # Skip total count header
            header_line = f.readline().strip().split()
            for line in f:
                row = line.strip().split()
                if not row:
                    continue
                file_key = row[0]
                attrs = {
                    header_line[i]: (row[i + 1] == "1")
                    for i in range(min(len(header_line), len(row) - 1))
                }
                attributes_by_file[file_key] = attrs

    records: Dict[int, Dict[str, Any]] = {}
    if not mapping_path.exists():
        logger.warning("CelebA-HQ mapping not found at %s", mapping_path)
        return records

    with open(mapping_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    # Skip header row (idx orig_idx orig_file)
    for row in lines[1:]:
        try:
            idx = int(row[0])
            orig_file = row[2]
        except (IndexError, ValueError):
            continue

        if max_samples is not None and idx >= max_samples:
            break

        file_name = f"{idx}.jpg"
        attr_dict = attributes_by_file.get(file_name, {})
        ident = identity_map.get(orig_file, f"unknown_id_{idx}")

        records[idx] = {
            "dataset": "celeba_hq",
            "identity_id": ident,
            "orig_file": orig_file,
            "male": attr_dict.get("Male", False),
            "young": attr_dict.get("Young", True),
            "eyeglasses": attr_dict.get("Eyeglasses", False),
            "smiling": attr_dict.get("Smiling", False),
            "gray_hair": attr_dict.get("Gray_Hair", False),
        }

    logger.info("Parsed %d CelebA-HQ identity/attribute records", len(records))
    return records


def build_baseline_manifest(
    ffhq_dir: Path,
    celeba_dir: Path,
    metadata_dir: Path,
    output_manifest: Path,
    max_ffhq: Optional[int] = 2000,
    max_celeba: Optional[int] = 2000,
) -> List[ImageMetadata]:
    """Assemble standardized baseline manifest with Real faces from FFHQ and CelebA-HQ.

    Args:
        ffhq_dir: Directory containing extracted FFHQ images (*.png).
        celeba_dir: Directory containing extracted CelebA-HQ images (*.jpg).
        metadata_dir: Directory containing annotations and identity mappings.
        output_manifest: Path to write manifest.jsonl.
        max_ffhq: Maximum number of FFHQ images to index.
        max_celeba: Maximum number of CelebA-HQ images to index.

    Returns:
        List of assembled ImageMetadata items.
    """
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_items: List[ImageMetadata] = []

    # 1. Parse metadata
    ffhq_aging = parse_ffhq_aging_metadata(
        metadata_dir / "ffhq_aging_labels.csv",
        max_samples=max_ffhq,
    )
    celeba_meta = parse_celeba_metadata(
        mapping_path=metadata_dir / "CelebA-HQ-to-CelebA-mapping.txt",
        identity_path=metadata_dir / "identity_CelebA.txt",
        attribute_path=metadata_dir / "CelebAMask-HQ-attribute-anno.txt",
        max_samples=max_celeba,
    )

    # 2. Ingest FFHQ images
    if ffhq_dir.exists():
        ffhq_files = sorted(ffhq_dir.glob("*.png"))
        if max_ffhq is not None:
            ffhq_files = ffhq_files[:max_ffhq]

        for p in ffhq_files:
            try:
                num = int(p.stem)
            except ValueError:
                num = -1

            demo = ffhq_aging.get(num, {"dataset": "ffhq"})
            # In FFHQ, each image is a distinct person
            source_id = f"ffhq_{p.stem}"
            meta = ImageMetadata(
                image_id=f"ffhq_{p.stem}",
                label=0,
                manipulation_type="real",
                prompt=None,
                source_real_id=source_id,
                file_path=str(p.as_posix()),
                bbox=[0, 0, 1024, 1024],
                is_hard_sample=False,
                baseline_score=None,
                data_source="baseline",
                demographics=demo,
            )
            manifest_items.append(meta)

    # 3. Ingest CelebA-HQ images
    if celeba_dir.exists():
        celeba_files = sorted(
            celeba_dir.glob("*.jpg"),
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem,
        )
        if max_celeba is not None:
            celeba_files = celeba_files[:max_celeba]

        for p in celeba_files:
            try:
                idx = int(p.stem)
            except ValueError:
                idx = -1

            demo = celeba_meta.get(idx, {"dataset": "celeba_hq"})
            ident_id = demo.get("identity_id", f"celeba_{p.stem}")
            source_id = f"celeba_id_{ident_id}"

            meta = ImageMetadata(
                image_id=f"celeba_hq_{p.stem}",
                label=0,
                manipulation_type="real",
                prompt=None,
                source_real_id=source_id,
                file_path=str(p.as_posix()),
                bbox=[0, 0, 1024, 1024],
                is_hard_sample=False,
                baseline_score=None,
                data_source="baseline",
                demographics=demo,
            )
            manifest_items.append(meta)

    # 4. Write manifest.jsonl
    with open(output_manifest, "w", encoding="utf-8") as f:
        for item in manifest_items:
            f.write(json.dumps(item.to_dict()) + "\n")

    logger.info(
        "Successfully built baseline manifest with %d samples at %s",
        len(manifest_items),
        output_manifest,
    )
    return manifest_items
