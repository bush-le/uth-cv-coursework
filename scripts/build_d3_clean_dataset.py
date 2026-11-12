"""Build clean D3 dataset partition for M3 Champion dual-engine training.

Enforces zero-data-leakage / zero-identity-leakage invariants:
1. Filters out all 626 RealisticVision v5.1 synthetic faces whose parent real face belongs to test_reals.
   These are quarantined into data/benchmarks/D3_clean/unseen_rv51_test.jsonl as an OOD benchmark.
2. Filters out 600 synthetic faces belonging to val_reals into data/benchmarks/D3_clean/val.jsonl.
3. Builds data/benchmarks/D3_clean/train.jsonl with strict 1:1 Real/Fake balance (5,626 images):
   - 2,813 Real images from train_reals.jsonl
   - 2,813 Fake images: 629 D2 hard samples + 800 SD 1.5 non-hard samples + 1,384 clean RV5.1 samples.
"""

from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Set

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_schema import ImageMetadata

logger = get_logger("build_d3_clean_dataset")


def load_metadata_manifest(path: Path) -> List[ImageMetadata]:
    """Load JSONL manifest into list of ImageMetadata."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ImageMetadata.from_dict(json.loads(line)))
    return records


def save_metadata_manifest(records: List[ImageMetadata], path: Path) -> None:
    """Save list of ImageMetadata to JSONL manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), path)


def get_real_filename_set(manifest_path: Path) -> Set[str]:
    """Extract set of image filenames (e.g., '0.jpg') from real manifest."""
    s = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                fp = d.get("file_path", "")
                if fp:
                    s.add(Path(fp).name)
    return s


def main() -> None:
    seed = 42
    seed_everything(seed)
    random.seed(seed)

    logger.info("=== BẮT ĐẦU PHÂN VÙNG D3 SẠCH (ZERO IDENTITY LEAKAGE) ===")

    train_reals_path = Path("data/partitions/train_reals.jsonl")
    val_reals_path = Path("data/partitions/val_reals.jsonl")
    test_reals_path = Path("data/partitions/test_reals.jsonl")
    d2_train_path = Path("data/benchmarks/D2_hard_boosted/train.jsonl")
    rv51_manifest_path = Path("data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl")

    output_dir = Path("data/benchmarks/D3_clean")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Real partition sets
    train_reals_meta = load_metadata_manifest(train_reals_path)
    val_reals_meta = load_metadata_manifest(val_reals_path)
    test_reals_meta = load_metadata_manifest(test_reals_path)

    test_filenames = get_real_filename_set(test_reals_path)
    val_filenames = get_real_filename_set(val_reals_path)
    train_filenames = get_real_filename_set(train_reals_path)

    logger.info(
        "Real partitions: Train=%d (names=%d), Val=%d (names=%d), Test=%d (names=%d)",
        len(train_reals_meta),
        len(train_filenames),
        len(val_reals_meta),
        len(val_filenames),
        len(test_reals_meta),
        len(test_filenames),
    )

    # 2. Parse 6k RealisticVision v5.1 dataset
    rv51_records: List[Dict[str, Any]] = []
    with open(rv51_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rv51_records.append(json.loads(line))

    logger.info("Total RV5.1 6k records loaded: %d", len(rv51_records))

    test_derived_rv51: List[ImageMetadata] = []
    val_derived_rv51: List[ImageMetadata] = []
    clean_train_derived_rv51: List[ImageMetadata] = []
    clean_text2img_rv51: List[ImageMetadata] = []

    for idx, r in enumerate(rv51_records):
        src_real = r.get("source_real_image")
        task = r.get("task", "unknown")
        img_path = r["image_path"]

        # Ensure image exists
        if not Path(img_path).exists():
            logger.error("Missing file: %s", img_path)
            continue

        meta = ImageMetadata(
            image_id=f"rv51_{task}_{idx:05d}",
            label=1,
            manipulation_type=f"rv51_{task}",
            prompt=r.get("prompt"),
            source_real_id=Path(src_real).stem if src_real else None,
            file_path=img_path,
            bbox=[0, 0, 512, 512],
            is_hard_sample=False,
            baseline_score=None,
            data_source="rv51_6k",
        )

        if task == "text2img" or not src_real:
            clean_text2img_rv51.append(meta)
        elif src_real in test_filenames:
            test_derived_rv51.append(meta)
        elif src_real in val_filenames:
            val_derived_rv51.append(meta)
        elif src_real in train_filenames:
            clean_train_derived_rv51.append(meta)
        else:
            # Fallback (non-test)
            clean_text2img_rv51.append(meta)

    logger.info("RV5.1 Categorization:")
    logger.info("  -> Test-derived (QUARANTINED): %d", len(test_derived_rv51))
    logger.info("  -> Val-derived (QUARANTINED): %d", len(val_derived_rv51))
    logger.info("  -> Clean Train-derived: %d", len(clean_train_derived_rv51))
    logger.info("  -> Clean Text2Img: %d", len(clean_text2img_rv51))

    assert len(test_derived_rv51) == 626, f"Expected 626 test-derived, got {len(test_derived_rv51)}"
    assert len(val_derived_rv51) == 600, f"Expected 600 val-derived, got {len(val_derived_rv51)}"

    # 3. Create Unseen RV5.1 OOD Test Vault (626 fake + 589 test reals)
    unseen_rv51_test_records = list(test_reals_meta) + list(test_derived_rv51)
    random.shuffle(unseen_rv51_test_records)
    save_metadata_manifest(unseen_rv51_test_records, output_dir / "unseen_rv51_test.jsonl")

    # 4. Create D3 Clean Validation Set (val reals + val rv51 + some SD val)
    d3_val_records = list(val_reals_meta) + list(val_derived_rv51)
    random.shuffle(d3_val_records)
    save_metadata_manifest(d3_val_records, output_dir / "val.jsonl")

    # 5. Build D3 Clean Training Set
    # Real: 2,813 real images from train_reals
    d3_train_reals = list(train_reals_meta)
    target_real_count = len(d3_train_reals)  # 2,813

    # Fake pool:
    # 5a. D2 Hard Samples (629 samples)
    d2_train_records = load_metadata_manifest(d2_train_path)
    d2_hard_fakes = [r for r in d2_train_records if r.label == 1 and r.is_hard_sample]
    d2_nonhard_fakes = [r for r in d2_train_records if r.label == 1 and not r.is_hard_sample]

    logger.info(
        "D2 Fake pool: Hard=%d, Non-hard=%d", len(d2_hard_fakes), len(d2_nonhard_fakes)
    )

    # Sample SD 1.5 non-hard: exactly 800
    random.shuffle(d2_nonhard_fakes)
    selected_sd_nonhard = d2_nonhard_fakes[:800]

    # Target RV5.1 clean fakes needed: 2813 - (629 + 800) = 1,384
    needed_rv51_count = target_real_count - (len(d2_hard_fakes) + len(selected_sd_nonhard))
    logger.info("Selecting %d clean RV5.1 samples to balance 1:1...", needed_rv51_count)

    # Stratified selection from clean RV5.1 pool:
    # clean_text2img_rv51 (2000 available) -> select 484
    # clean_train_derived_rv51 (inpaint + img2img, 2774 available) -> select 900 (450 each)
    random.shuffle(clean_text2img_rv51)
    random.shuffle(clean_train_derived_rv51)

    rv51_inpaint = [r for r in clean_train_derived_rv51 if "inpaint" in r.manipulation_type]
    rv51_img2img = [r for r in clean_train_derived_rv51 if "img2img" in r.manipulation_type]

    selected_rv51 = (
        clean_text2img_rv51[:484]
        + rv51_inpaint[:450]
        + rv51_img2img[:450]
    )
    if len(selected_rv51) < needed_rv51_count:
        remaining = needed_rv51_count - len(selected_rv51)
        selected_rv51.extend(clean_text2img_rv51[484 : 484 + remaining])
    elif len(selected_rv51) > needed_rv51_count:
        selected_rv51 = selected_rv51[:needed_rv51_count]

    logger.info("Selected RV5.1 clean samples: %d", len(selected_rv51))

    # Combine all fakes
    d3_train_fakes = list(d2_hard_fakes) + list(selected_sd_nonhard) + list(selected_rv51)
    logger.info("Total D3 Fake count: %d", len(d3_train_fakes))

    assert len(d3_train_fakes) == target_real_count, (
        f"Fake ({len(d3_train_fakes)}) != Real ({target_real_count})"
    )

    # Combine Train (5,626 samples)
    d3_train_all = d3_train_reals + d3_train_fakes
    random.shuffle(d3_train_all)

    # 6. Strict Self-Audit
    logger.info("Running strict Self-Audit on D3 Train...")
    leaked_ids = []
    for r in d3_train_all:
        if r.label == 1:
            src = r.source_real_id
            if src and f"{src}.jpg" in test_filenames:
                leaked_ids.append(src)
        else:
            if Path(r.file_path).name in test_filenames:
                leaked_ids.append(r.file_path)

    if leaked_ids:
        raise ValueError(f"CRITICAL FAULT: Found {len(leaked_ids)} leaked test IDs in D3 Train!")

    logger.info("[✓] ZERO IDENTITY LEAKAGE CONFIRMED: 0 leaked IDs found!")

    save_metadata_manifest(d3_train_all, output_dir / "train.jsonl")

    # Summary
    train_dist = Counter((r.label, r.manipulation_type, r.is_hard_sample) for r in d3_train_all)
    logger.info("=== D3 CLEAN TRAIN DISTRIBUTION ===")
    for k, v in sorted(train_dist.items()):
        logger.info("  Label=%d, Type=%s, Hard=%s: %d", k[0], k[1], k[2], v)
    logger.info("Total D3 Clean Train: %d (Real=%d, Fake=%d)", len(d3_train_all), target_real_count, len(d3_train_fakes))
    logger.info("=== PHÂN VÙNG D3 SẠCH HOÀN TẤT THÀNH CÔNG ===")


if __name__ == "__main__":
    main()
