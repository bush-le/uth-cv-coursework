"""Build clean D3 dataset partition for M3 Champion dual-engine training.

Enforces zero-data-leakage / zero-identity-leakage invariants:
1. Quarantines RealisticVision v5.1 synthetic faces whose parent real face belongs to test_reals
   into data/benchmarks/D3_clean/unseen_rv51_test.jsonl as an OOD benchmark.
2. Quarantines RV5.1 synthetic faces belonging to val_reals into data/benchmarks/D3_clean/val.jsonl.
3. Builds data/benchmarks/D3_clean/train.jsonl with strict 1:1 Real/Fake balance using dynamic quotas
   compatible with both filtered (curated) and unfiltered RV5.1 manifests.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Set

# Ensure project root is in sys.path for direct execution without mutating cwd on import
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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


def resolve_rv51_manifest(rv51_manifest_path: Optional[Path] = None) -> Path:
    """Resolve RV5.1 manifest path, preferring curated manifest when available."""
    if rv51_manifest_path is not None:
        return Path(rv51_manifest_path)
    curated_cand = Path("data/external_benchmarks/synthetic_face_dataset_6k/curated_manifest.jsonl")
    if curated_cand.is_file():
        return curated_cand
    return Path("data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl")


def _allocate_stratified_rv51(
    clean_text2img_rv51: Sequence[ImageMetadata],
    rv51_inpaint: Sequence[ImageMetadata],
    rv51_img2img: Sequence[ImageMetadata],
    needed_rv51_count: int,
) -> List[ImageMetadata]:
    """Allocate clean RV5.1 samples across text2img/inpaint/img2img dynamically without fixed counts."""
    if needed_rv51_count <= 0:
        return []

    t2i_target = int(round(needed_rv51_count * 0.35))
    inp_target = int(round(needed_rv51_count * 0.325))
    i2i_target = max(0, needed_rv51_count - t2i_target - inp_target)

    picked_t2i = list(clean_text2img_rv51[:t2i_target])
    picked_inp = list(rv51_inpaint[:inp_target])
    picked_i2i = list(rv51_img2img[:i2i_target])

    selected: List[ImageMetadata] = picked_t2i + picked_inp + picked_i2i
    selected_ids = {m.image_id for m in selected}

    if len(selected) < needed_rv51_count:
        remainder = (
            list(clean_text2img_rv51[len(picked_t2i) :])
            + list(rv51_inpaint[len(picked_inp) :])
            + list(rv51_img2img[len(picked_i2i) :])
        )
        for item in remainder:
            if item.image_id not in selected_ids:
                selected.append(item)
                selected_ids.add(item.image_id)
            if len(selected) >= needed_rv51_count:
                break

    return selected[:needed_rv51_count]


def build_d3_clean_dataset(
    train_reals_path: Path = Path("data/partitions/train_reals.jsonl"),
    val_reals_path: Path = Path("data/partitions/val_reals.jsonl"),
    test_reals_path: Path = Path("data/partitions/test_reals.jsonl"),
    d2_train_path: Path = Path("data/benchmarks/D2_hard_boosted/train.jsonl"),
    rv51_manifest_path: Optional[Path] = None,
    output_dir: Path = Path("data/benchmarks/D3_clean"),
    sd_nonhard_quota: int = 800,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build identity-disjoint D3_clean splits dynamically from curated or raw RV5.1 manifests."""
    seed_everything(seed)
    rng = random.Random(seed)

    logger.info("=== BẮT ĐẦU PHÂN VÙNG D3 SẠCH (ZERO IDENTITY LEAKAGE) ===")

    resolved_rv51_path = resolve_rv51_manifest(rv51_manifest_path)
    if not resolved_rv51_path.is_file():
        raise FileNotFoundError(f"RV5.1 manifest not found: {resolved_rv51_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Real partition sets
    train_reals_meta = load_metadata_manifest(train_reals_path)
    val_reals_meta = load_metadata_manifest(val_reals_path)
    test_reals_meta = load_metadata_manifest(test_reals_path)

    test_filenames = get_real_filename_set(test_reals_path)
    val_filenames = get_real_filename_set(val_reals_path)
    train_filenames = get_real_filename_set(train_reals_path)

    test_real_stems = {Path(fn).stem for fn in test_filenames}
    val_real_stems = {Path(fn).stem for fn in val_filenames}
    train_real_stems = {Path(fn).stem for fn in train_filenames}
    test_real_ids = (
        {m.source_real_id for m in test_reals_meta if m.source_real_id}
        | {m.image_id for m in test_reals_meta}
        | test_real_stems
    )
    val_real_ids = (
        {m.source_real_id for m in val_reals_meta if m.source_real_id}
        | {m.image_id for m in val_reals_meta}
        | val_real_stems
    )

    logger.info(
        "Real partitions: Train=%d (names=%d), Val=%d (names=%d), Test=%d (names=%d)",
        len(train_reals_meta),
        len(train_filenames),
        len(val_reals_meta),
        len(val_filenames),
        len(test_reals_meta),
        len(test_filenames),
    )

    # 2. Parse RealisticVision v5.1 dataset (supports both fake_manifest and curated_manifest schemas)
    rv51_records: List[Dict[str, Any]] = []
    with open(resolved_rv51_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rv51_records.append(json.loads(line))

    if not rv51_records:
        raise ValueError(f"Loaded 0 records from RV5.1 manifest: {resolved_rv51_path}")

    logger.info("Total RV5.1 records loaded from %s: %d", resolved_rv51_path, len(rv51_records))

    test_derived_rv51: List[ImageMetadata] = []
    val_derived_rv51: List[ImageMetadata] = []
    clean_train_derived_rv51: List[ImageMetadata] = []
    clean_text2img_rv51: List[ImageMetadata] = []

    for idx, r in enumerate(rv51_records):
        src_real = r.get("source_real_image")
        src_real_id = Path(str(src_real)).stem if src_real else r.get("source_real_id")
        raw_task = str(r.get("task") or r.get("manipulation_type") or "unknown")
        task = raw_task.replace("rv51_", "").replace("sd_", "")
        img_path = str(r.get("image_path") or r.get("file_path") or "")

        if not img_path or not Path(img_path).exists():
            logger.warning("Missing RV5.1 image file, skipping: %s", img_path)
            continue

        meta = ImageMetadata(
            image_id=str(r.get("image_id") or f"rv51_{task}_{idx:05d}"),
            label=1,
            manipulation_type=f"rv51_{task}",
            prompt=r.get("prompt"),
            source_real_id=str(src_real_id) if src_real_id else None,
            file_path=img_path,
            bbox=r.get("bbox") or [0, 0, 512, 512],
            is_hard_sample=False,
            baseline_score=None,
            data_source=str(r.get("data_source") or "rv51_6k"),
        )

        if task == "text2img" or (not src_real and not src_real_id):
            clean_text2img_rv51.append(meta)
        elif (src_real and src_real in test_filenames) or (src_real_id and str(src_real_id) in test_real_ids):
            test_derived_rv51.append(meta)
        elif (src_real and src_real in val_filenames) or (src_real_id and str(src_real_id) in val_real_ids):
            val_derived_rv51.append(meta)
        elif (src_real and src_real in train_filenames) or (src_real_id and str(src_real_id) in train_real_stems):
            clean_train_derived_rv51.append(meta)
        else:
            clean_text2img_rv51.append(meta)

    logger.info("RV5.1 Categorization:")
    logger.info("  -> Test-derived (QUARANTINED): %d", len(test_derived_rv51))
    logger.info("  -> Val-derived (QUARANTINED): %d", len(val_derived_rv51))
    logger.info("  -> Clean Train-derived: %d", len(clean_train_derived_rv51))
    logger.info("  -> Clean Text2Img: %d", len(clean_text2img_rv51))

    # 3. Create Unseen RV5.1 OOD Test Vault (test_derived_rv51 + test_reals)
    unseen_rv51_test_records = list(test_reals_meta) + list(test_derived_rv51)
    rng.shuffle(unseen_rv51_test_records)
    save_metadata_manifest(unseen_rv51_test_records, output_dir / "unseen_rv51_test.jsonl")

    # 4. Create D3 Clean Validation Set (val reals + val rv51)
    d3_val_records = list(val_reals_meta) + list(val_derived_rv51)
    rng.shuffle(d3_val_records)
    save_metadata_manifest(d3_val_records, output_dir / "val.jsonl")

    # 5. Build D3 Clean Training Set with strict 1:1 Real/Fake balance
    d3_train_reals = list(train_reals_meta)
    target_real_count = len(d3_train_reals)

    d2_train_records = load_metadata_manifest(d2_train_path)
    d2_hard_fakes = [r for r in d2_train_records if r.label == 1 and r.is_hard_sample]
    d2_nonhard_fakes = [r for r in d2_train_records if r.label == 1 and not r.is_hard_sample]

    logger.info("D2 Fake pool: Hard=%d, Non-hard=%d", len(d2_hard_fakes), len(d2_nonhard_fakes))

    rng.shuffle(d2_nonhard_fakes)
    sd_nonhard_cap = max(
        0, min(sd_nonhard_quota, len(d2_nonhard_fakes), target_real_count - len(d2_hard_fakes))
    )
    selected_sd_nonhard = d2_nonhard_fakes[:sd_nonhard_cap]

    needed_rv51_count = max(0, target_real_count - (len(d2_hard_fakes) + len(selected_sd_nonhard)))
    logger.info("Selecting %d clean RV5.1 samples to balance 1:1...", needed_rv51_count)

    rng.shuffle(clean_text2img_rv51)
    rng.shuffle(clean_train_derived_rv51)

    rv51_inpaint = [r for r in clean_train_derived_rv51 if "inpaint" in (r.manipulation_type or "")]
    rv51_img2img = [r for r in clean_train_derived_rv51 if "img2img" in (r.manipulation_type or "")]
    selected_rv51 = _allocate_stratified_rv51(
        clean_text2img_rv51=clean_text2img_rv51,
        rv51_inpaint=rv51_inpaint,
        rv51_img2img=rv51_img2img,
        needed_rv51_count=needed_rv51_count,
    )
    # Include any remaining clean_train_derived items of other subtypes if still short
    if len(selected_rv51) < needed_rv51_count:
        picked_ids = {m.image_id for m in selected_rv51}
        for item in clean_train_derived_rv51:
            if item.image_id not in picked_ids:
                selected_rv51.append(item)
                picked_ids.add(item.image_id)
            if len(selected_rv51) >= needed_rv51_count:
                break

    # Top up from remaining SD non-hard fakes only if curated RV5.1 pool is smaller than needed
    extra_sd_nonhard: List[ImageMetadata] = []
    shortfall = needed_rv51_count - len(selected_rv51)
    if shortfall > 0:
        extra_sd_nonhard = d2_nonhard_fakes[sd_nonhard_cap : sd_nonhard_cap + shortfall]
        logger.warning(
            "Clean RV5.1 pool supplied %d/%d samples; supplemented %d from SD 1.5 non-hard pool.",
            len(selected_rv51),
            needed_rv51_count,
            len(extra_sd_nonhard),
        )

    logger.info("Selected RV5.1 clean samples: %d", len(selected_rv51))

    d3_train_fakes = (
        list(d2_hard_fakes)
        + list(selected_sd_nonhard)
        + list(selected_rv51)
        + list(extra_sd_nonhard)
    )
    logger.info("Total D3 Fake count: %d", len(d3_train_fakes))

    if len(d3_train_fakes) != target_real_count:
        raise ValueError(
            f"Insufficient clean fake candidates to match Real count: Fake={len(d3_train_fakes)} vs Real={target_real_count}"
        )

    d3_train_all = d3_train_reals + d3_train_fakes
    rng.shuffle(d3_train_all)

    # 6. Strict Self-Audit against both test_reals and val_reals
    logger.info("Running strict Self-Audit on D3 Train...")
    leaked_ids = []
    for r in d3_train_all:
        if r.label == 1:
            src = r.source_real_id
            if src and (str(src) in test_real_ids or f"{src}.jpg" in test_filenames or str(src) in val_real_ids):
                leaked_ids.append(src)
        else:
            if Path(r.file_path or "").name in test_filenames:
                leaked_ids.append(r.file_path)

    if leaked_ids:
        raise ValueError(f"CRITICAL FAULT: Found {len(leaked_ids)} leaked non-train IDs in D3 Train!")

    logger.info("[✓] ZERO IDENTITY LEAKAGE CONFIRMED: 0 leaked IDs found!")

    save_metadata_manifest(d3_train_all, output_dir / "train.jsonl")

    train_dist = Counter((r.label, r.manipulation_type, r.is_hard_sample) for r in d3_train_all)
    logger.info("=== D3 CLEAN TRAIN DISTRIBUTION ===")
    for k, v in sorted(train_dist.items()):
        logger.info("  Label=%d, Type=%s, Hard=%s: %d", k[0], k[1], k[2], v)
    logger.info(
        "Total D3 Clean Train: %d (Real=%d, Fake=%d)",
        len(d3_train_all),
        target_real_count,
        len(d3_train_fakes),
    )
    logger.info("=== PHÂN VÙNG D3 SẠCH HOÀN TẤT THÀNH CÔNG ===")

    return {
        "rv51_source": str(resolved_rv51_path),
        "output_dir": str(output_dir),
        "train_total": len(d3_train_all),
        "train_real": target_real_count,
        "train_fake": len(d3_train_fakes),
        "val_total": len(d3_val_records),
        "unseen_rv51_test_total": len(unseen_rv51_test_records),
        "rv51_test_derived": len(test_derived_rv51),
        "rv51_val_derived": len(val_derived_rv51),
        "rv51_train_selected": len(selected_rv51),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for building D3_clean manifests."""
    parser = argparse.ArgumentParser(description="Build clean identity-disjoint D3 benchmark splits.")
    parser.add_argument("--train-reals", type=Path, default=Path("data/partitions/train_reals.jsonl"))
    parser.add_argument("--val-reals", type=Path, default=Path("data/partitions/val_reals.jsonl"))
    parser.add_argument("--test-reals", type=Path, default=Path("data/partitions/test_reals.jsonl"))
    parser.add_argument("--d2-train", type=Path, default=Path("data/benchmarks/D2_hard_boosted/train.jsonl"))
    parser.add_argument("--rv51-manifest", type=Path, default=None, help="Path to curated or raw RV5.1 manifest JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/D3_clean"))
    parser.add_argument("--sd-nonhard-quota", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    os.chdir(root_dir)
    args = parse_args()
    build_d3_clean_dataset(
        train_reals_path=args.train_reals,
        val_reals_path=args.val_reals,
        test_reals_path=args.test_reals,
        d2_train_path=args.d2_train,
        rv51_manifest_path=args.rv51_manifest,
        output_dir=args.output_dir,
        sd_nonhard_quota=args.sd_nonhard_quota,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
