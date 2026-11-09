"""Organize and standardize dataset structure strictly adhering to Topic 11 lifecycle."""

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_schema import ImageMetadata

logger = get_logger(__name__)


def organize_synthetic_faces(
    source_manifest: Path,
    output_base: Path,
) -> Dict[str, int]:
    """Organize synthetic faces by generation method (text2image, inpainting, img2img)."""
    counts = {"text2image": 0, "inpainting": 0, "img2img": 0, "total": 0}
    if not source_manifest.exists():
        return counts

    t2i_dir = output_base / "text2image"
    inpaint_dir = output_base / "inpainting"
    img2img_dir = output_base / "img2img"
    for d in (t2i_dir, inpaint_dir, img2img_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(source_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            src_path = item.get("file_path")
            if not src_path or not Path(src_path).exists():
                continue

            manip = item.get("manipulation_type", "")
            p = Path(src_path)
            if manip == "sd_text2img":
                dest = t2i_dir / p.name
                counts["text2image"] += 1
            elif manip == "sd_inpaint":
                dest = inpaint_dir / p.name
                counts["inpainting"] += 1
            else:
                dest = img2img_dir / p.name
                counts["img2img"] += 1

            shutil.copy2(p, dest)
            counts["total"] += 1

    logger.info("Organized synthetic faces: %s", counts)
    return counts


def organize_curated_and_hard_samples(
    curated_manifest: Path,
    hard_manifest: Path,
    curated_dir: Path,
    hard_images_dir: Path,
    hard_audit_dir: Path,
) -> Dict[str, Any]:
    """Sort curated images and hard samples with demographic audit."""
    curated_dir.mkdir(parents=True, exist_ok=True)
    hard_images_dir.mkdir(parents=True, exist_ok=True)
    hard_audit_dir.mkdir(parents=True, exist_ok=True)

    curated_count = 0
    if curated_manifest.exists():
        with open(curated_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                fp = item.get("file_path")
                if fp and Path(fp).exists():
                    shutil.copy2(fp, curated_dir / Path(fp).name)
                    curated_count += 1

    hard_items = []
    if hard_manifest.exists():
        with open(hard_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                hard_items.append(item)
                fp = item.get("file_path")
                if fp and Path(fp).exists():
                    shutil.copy2(fp, hard_images_dir / Path(fp).name)

    # Demographic audit of hard samples
    age_dist: Dict[str, int] = {}
    gender_dist: Dict[str, int] = {}
    glasses_dist: Dict[str, int] = {}

    for h in hard_items:
        demo = h.get("demographics") or {}
        age = demo.get("age_group", "unknown")
        gender = demo.get("gender", "unknown")
        glasses = demo.get("glasses", "unknown")

        age_dist[age] = age_dist.get(age, 0) + 1
        gender_dist[gender] = gender_dist.get(gender, 0) + 1
        glasses_dist[glasses] = glasses_dist.get(glasses, 0) + 1

    audit = {
        "total_hard_samples": len(hard_items),
        "age_distribution": age_dist,
        "gender_distribution": gender_dist,
        "glasses_distribution": glasses_dist,
        "mean_baseline_score": float(
            sum(h.get("baseline_score", 0.0) or 0.0 for h in hard_items) / max(1, len(hard_items))
        ),
    }

    audit_path = hard_audit_dir / "hard_sample_demographics.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    logger.info("Organized %d curated and %d hard samples (audit written to %s)", curated_count, len(hard_items), audit_path)
    return audit


def build_benchmark_splits(
    real_manifest: Path,
    curated_manifest: Path,
    hard_manifest: Path,
    output_dir: Path,
    max_real_samples: int = 50,
    seed: int = 42,
) -> None:
    """Generate reproducible train/val/test splits for D0, D1, D2."""
    seed_everything(seed)
    real_records: List[ImageMetadata] = []
    if real_manifest.exists():
        with open(real_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                real_records.append(ImageMetadata.from_dict(json.loads(line)))
                if len(real_records) >= max_real_samples:
                    break

    curated_records: List[ImageMetadata] = []
    if curated_manifest.exists():
        with open(curated_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                curated_records.append(ImageMetadata.from_dict(json.loads(line)))

    hard_records: List[ImageMetadata] = []
    if hard_manifest.exists():
        with open(hard_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                hard_records.append(ImageMetadata.from_dict(json.loads(line)))

    # Stage D0: Real + Generic Fake
    d0_records = real_records + [
        ImageMetadata(
            image_id=c.image_id,
            label=1,
            manipulation_type=c.manipulation_type,
            source_real_id=c.source_real_id,
            file_path=c.file_path,
            is_hard_sample=False,
            data_source="baseline",
            demographics=c.demographics,
        )
        for c in curated_records
    ]

    # Stage D1: Real + Curated Synthetic
    d1_records = real_records + curated_records

    # Stage D2: Real + Curated + Hard
    d2_map = {m.image_id: m for m in real_records}
    for c in curated_records:
        d2_map[c.image_id] = c
    for h in hard_records:
        d2_map[h.image_id] = h
    d2_records = list(d2_map.values())

    stages = [
        ("D0_baseline", d0_records, 1),
        ("D1_augmented", d1_records, 1),
        ("D2_hard_boosted", d2_records, 2),
    ]

    for stage_name, records, oversample in stages:
        stage_dir = output_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        dm = DeepfakeImageDataModule(
            metadata_list=records,
            hard_sample_oversample_factor=oversample,
            identity_disjoint=True,
            seed=seed,
        )
        train_m, val_m, test_m = dm._partition_splits()

        for split_name, split_list in [("train", train_m), ("val", val_m), ("test", test_m)]:
            out_file = stage_dir / f"{split_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for item in split_list:
                    f.write(json.dumps(item.to_dict()) + "\n")

        logger.info(
            "Saved %s splits to %s: %d train, %d val, %d test (identity-disjoint).",
            stage_name,
            stage_dir,
            len(train_m),
            len(val_m),
            len(test_m),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize dataset into Topic 11 standardized structure.")
    parser.add_argument("--raw-manifest", type=str, default="data/raw/manifest.jsonl")
    parser.add_argument("--synthetic-manifest", type=str, default="data/processed/fake_manifest.jsonl")
    parser.add_argument("--curated-manifest", type=str, default="data/processed/curated_manifest.jsonl")
    parser.add_argument("--hard-manifest", type=str, default="data/processed/hard_samples_manifest.jsonl")
    parser.add_argument("--base-dir", type=str, default="data")
    args = parser.parse_args()

    base = Path(args.base_dir)
    logger.info("Beginning dataset reorganization in %s...", base)

    # 1. Organize synthetic faces by mode
    organize_synthetic_faces(
        source_manifest=Path(args.synthetic_manifest),
        output_base=base / "02_synthetic_sd",
    )

    # 2. Curated & Hard samples
    organize_curated_and_hard_samples(
        curated_manifest=Path(args.curated_manifest),
        hard_manifest=Path(args.hard_manifest),
        curated_dir=base / "02_synthetic_sd" / "curated",
        hard_images_dir=base / "03_hard_samples" / "images",
        hard_audit_dir=base / "03_hard_samples" / "audit_reports",
    )

    # Copy manifests
    manifest_dest = base / "02_synthetic_sd" / "manifests"
    manifest_dest.mkdir(parents=True, exist_ok=True)
    if Path(args.synthetic_manifest).exists():
        shutil.copy2(args.synthetic_manifest, manifest_dest / "all_generated.jsonl")
    if Path(args.curated_manifest).exists():
        shutil.copy2(args.curated_manifest, manifest_dest / "curated_synthetic.jsonl")
    if Path(args.hard_manifest).exists():
        shutil.copy2(args.hard_manifest, base / "03_hard_samples" / "hard_samples_manifest.jsonl")

    # 3. Build benchmark splits D0, D1, D2
    build_benchmark_splits(
        real_manifest=Path(args.raw_manifest),
        curated_manifest=Path(args.curated_manifest),
        hard_manifest=Path(args.hard_manifest),
        output_dir=base / "04_benchmarks",
        max_real_samples=50,
    )

    logger.info("Dataset reorganization complete!")


if __name__ == "__main__":
    main()
