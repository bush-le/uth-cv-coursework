"""Pipeline entrypoint for Block 1: Master Frozen Test Vault & Baseline M0 Initialization.

Lifecycle:
1. Splits raw reference faces (4,000 real images) into 100% identity-disjoint partitions:
   - Train Reals (~2,800 images, 70%)
   - Val Reals (~600 images, 15%)
   - Test Reals (~600 images, 15%)
2. Synthesizes balanced Master Test Fakes from held-out Test Reals and packages the
   permanent, unalterable Master Frozen Test Set (data/benchmarks/master_frozen_test.jsonl).
3. Synthesizes Baseline D0 Fakes from Train & Val Reals, curates quality via Laplacian blur,
   and packages the initial D0_baseline benchmark dataset.
4. Trains Baseline Detector M0 on D0 and evaluates against the Master Frozen Test Set.
5. Saves checkpoints/baseline_d0_best.pt and logs baseline metrics.
"""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from src.common.config import load_config
from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_schema import ImageMetadata
from src.data.splits import assign_identity_splits, get_group_key
from src.models.detector.image_detector import DeepfakeImageDetector
from src.models.diffusion.prompt_engine import PromptGrammarEngine
from src.pipelines.curate_image_dataset import curate_dataset
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector
from src.pipelines.generate_deepfake_images import run_image_generation
from src.pipelines.train_image_detector import train_deepfake_image_detector

logger = get_logger(__name__)


def create_identity_partitions(
    raw_manifest: Path,
    partitions_dir: Path,
    ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, List[ImageMetadata]]:
    """Partition raw real faces into strictly identity-disjoint train, val, and test splits."""
    partitions_dir.mkdir(parents=True, exist_ok=True)
    if ratios is None:
        ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

    records: List[ImageMetadata] = []
    with open(raw_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(ImageMetadata.from_dict(json.loads(line)))

    assignment = assign_identity_splits(records, ratios=ratios, seed=seed)

    splits: Dict[str, List[ImageMetadata]] = {"train": [], "val": [], "test": []}
    for m in records:
        grp = get_group_key(m)
        split_name = assignment.get(grp, "train")
        splits[split_name].append(m)

    # Save partition manifests
    for name, split_records in splits.items():
        out_path = partitions_dir / f"{name}_reals.jsonl"
        with open(out_path, "w", encoding="utf-8") as pf:
            for r in split_records:
                pf.write(json.dumps(r.to_dict()) + "\n")

    logger.info(
        "Partitioned %d real images into identity-disjoint splits: %d train, %d val, %d test.",
        len(records),
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits


def load_metadata_manifest(manifest_path: Path) -> List[ImageMetadata]:
    """Load metadata records from a JSONL manifest file."""
    if not manifest_path.exists():
        return []
    records: List[ImageMetadata] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(ImageMetadata.from_dict(json.loads(line_str)))
    return records


def synthesize_and_curate_quota(
    input_manifest: Path,
    output_raw_manifest: Path,
    output_curated_manifest: Path,
    output_dir: Path,
    target_count: int,
    prompt_engine: PromptGrammarEngine,
    report_path: Path,
    device: str = "cpu",
    seed: int = 42,
    max_attempts: int = 3,
) -> List[ImageMetadata]:
    """Synthesize candidates and curate iteratively until target_count valid samples are reached."""
    curated = load_metadata_manifest(output_curated_manifest)
    if len(curated) >= target_count:
        logger.info(
            "Manifest %s already has %d valid samples (target %d), reusing.",
            output_curated_manifest.name,
            len(curated),
            target_count,
        )
        return curated[:target_count]

    attempt = 0
    while len(curated) < target_count and attempt < max_attempts:
        needed = target_count - len(curated)
        batch_size = int(needed * 1.15) + 2 if attempt > 0 else target_count
        logger.info(
            "Quota synthesis attempt %d/%d for %s: need %d valid samples, generating %d candidates...",
            attempt + 1,
            max_attempts,
            output_curated_manifest.name,
            needed,
            batch_size,
        )

        att_raw = output_raw_manifest.parent / f"{output_raw_manifest.stem}_att{attempt}.jsonl"
        att_cur = output_curated_manifest.parent / f"{output_curated_manifest.stem}_att{attempt}.jsonl"
        att_rep = report_path.parent / f"{report_path.stem}_att{attempt}.json"

        run_image_generation(
            input_manifest=input_manifest,
            output_manifest=att_raw,
            output_dir=output_dir,
            total_samples=batch_size,
            prompt_engine=prompt_engine,
            device=device,
            seed=seed + (attempt * 1000),
            resume=True,
            sample_offset=len(curated) + (attempt * 500),
        )

        curate_dataset(
            input_manifest=att_raw,
            output_manifest=att_cur,
            report_path=att_rep,
        )

        new_samples = load_metadata_manifest(att_cur)
        existing_ids = {m.image_id for m in curated}
        for s in new_samples:
            if s.image_id not in existing_ids:
                curated.append(s)
                existing_ids.add(s.image_id)
            if len(curated) >= target_count:
                break

        attempt += 1

    # Save final aggregated curated manifest
    output_curated_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_curated_manifest, "w", encoding="utf-8") as f:
        for m in curated[:target_count]:
            f.write(json.dumps(m.to_dict()) + "\n")

    logger.info(
        "Quota fulfillment for %s: achieved %d/%d valid samples.",
        output_curated_manifest.name,
        min(len(curated), target_count),
        target_count,
    )
    return curated[:target_count]


def package_benchmark_d0(
    partitions: Dict[str, List[ImageMetadata]],
    curated_fakes: Dict[str, List[ImageMetadata]],
    benchmarks_dir: Path,
) -> Dict[str, Path]:
    """Assemble Master Frozen Test Set and D0_baseline benchmark manifests."""
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    d0_dir = benchmarks_dir / "D0_baseline"
    d0_dir.mkdir(parents=True, exist_ok=True)

    master_test_records = partitions["test"] + curated_fakes.get("test", [])
    d0_train_records = partitions["train"] + curated_fakes.get("train", [])
    d0_val_records = partitions["val"] + curated_fakes.get("val", [])
    d0_test_records = list(master_test_records)

    paths = {
        "master_test": benchmarks_dir / "master_frozen_test.jsonl",
        "d0_train": d0_dir / "train.jsonl",
        "d0_val": d0_dir / "val.jsonl",
        "d0_test": d0_dir / "test.jsonl",
    }

    manifest_map = {
        paths["master_test"]: master_test_records,
        paths["d0_train"]: d0_train_records,
        paths["d0_val"]: d0_val_records,
        paths["d0_test"]: d0_test_records,
    }

    # Vault Protection: Prevent overwriting an existing sealed master test vault with smaller mock/partial sets
    if paths["master_test"].exists() and len(master_test_records) < 1178:
        try:
            with open(paths["master_test"], "r", encoding="utf-8") as f_chk:
                existing_lines = sum(1 for line in f_chk if line.strip())
            if existing_lines >= 1178:
                logger.warning(
                    "Master Frozen Test Vault (%s) is already sealed with %d samples. Preserving vault against overwrite by %d test records.",
                    paths["master_test"],
                    existing_lines,
                    len(master_test_records),
                )
                del manifest_map[paths["master_test"]]
        except Exception as exc:
            logger.warning("Could not check master test vault integrity: %s", exc)

    for path, meta_list in manifest_map.items():
        with open(path, "w", encoding="utf-8") as f:
            for m in meta_list:
                f.write(json.dumps(m.to_dict()) + "\n")

    logger.info(
        "Packaged D0 benchmark: %d train (%d Real, %d Fake), %d val (%d Real, %d Fake), %d test (%d Real, %d Fake).",
        len(d0_train_records),
        len(partitions["train"]),
        len(curated_fakes.get("train", [])),
        len(d0_val_records),
        len(partitions["val"]),
        len(curated_fakes.get("val", [])),
        len(d0_test_records),
        len(partitions["test"]),
        len(curated_fakes.get("test", [])),
    )
    return paths


def run_block1_pipeline(
    raw_manifest_path: Path = Path("data/raw/manifest.jsonl"),
    config_path: str = "configs/dataset_factory.yaml",
    test_fake_count: int = 600,
    train_fake_count: int = 1400,
    val_fake_count: int = 200,
    epochs: int = 5,
    device: str = "cpu",
    dry_run: bool = False,
    output_base_dir: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete Block 1: Partitioning, Test Vault & D0 Synthesis, and Baseline M0 Training."""
    seed_everything(seed)
    base_dir = output_base_dir or Path("data")
    ckpt_dir = checkpoints_dir or Path("experiments/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("experiments/logs") if output_base_dir is None else (output_base_dir / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    partitions_dir = base_dir / "partitions"
    benchmarks_dir = (base_dir / "benchmarks_dry_run") if (dry_run and output_base_dir is None) else (base_dir / "benchmarks")
    syn_dir = base_dir / "synthetic"

    logger.info("=== Starting Block 1: Master Frozen Test Vault & Baseline M0 Initialization ===")

    # 1. Identity-Disjoint Partitioning
    partitions = create_identity_partitions(
        raw_manifest=raw_manifest_path,
        partitions_dir=partitions_dir,
        seed=seed,
    )

    if dry_run:
        # Fast synthetic dry-run verification
        mock_fakes = {
            "test": [
                ImageMetadata(
                    image_id=f"dry_test_fake_{i}",
                    label=1,
                    source_real_id=partitions["test"][i % len(partitions["test"])].source_real_id,
                    data_source="synthetic",
                )
                for i in range(test_fake_count)
            ],
            "train": [
                ImageMetadata(
                    image_id=f"dry_train_fake_{i}",
                    label=1,
                    source_real_id=partitions["train"][i % len(partitions["train"])].source_real_id,
                    data_source="synthetic",
                )
                for i in range(train_fake_count)
            ],
            "val": [
                ImageMetadata(
                    image_id=f"dry_val_fake_{i}",
                    label=1,
                    source_real_id=partitions["val"][i % len(partitions["val"])].source_real_id,
                    data_source="synthetic",
                )
                for i in range(val_fake_count)
            ],
        }
        package_benchmark_d0(partitions, mock_fakes, benchmarks_dir)

        dry_metrics = {
            "accuracy": 0.65,
            "roc_auc": 0.70,
            "precision": 0.68,
            "recall": 0.62,
            "bypass_rate": 0.38,
            "loss": 0.58,
        }
        summary = {
            "stage": "Block1_Baseline_M0",
            "dry_run": True,
            "d0_counts": {
                "train_real": len(partitions["train"]),
                "train_fake": len(mock_fakes["train"]),
                "val_real": len(partitions["val"]),
                "val_fake": len(mock_fakes["val"]),
                "test_real": len(partitions["test"]),
                "test_fake": len(mock_fakes["test"]),
                "test_total": len(partitions["test"]) + len(mock_fakes["test"]),
            },
            "baseline_test_metrics": dry_metrics,
        }
        with open(logs_dir / "block1_baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    # Live Synthesis Execution
    prompt_eng = PromptGrammarEngine(seed=seed)

    # 2. Synthesize Master Test Fakes (Quota Ensured)
    test_fakes_raw = syn_dir / "master_test_fakes_raw.jsonl"
    test_fakes_curated = syn_dir / "master_test_fakes_curated.jsonl"
    test_fakes_dir = syn_dir / "master_test_images"
    test_reals_manifest = partitions_dir / "test_reals.jsonl"

    logger.info(">>> Synthesizing %d Master Test Fakes from held-out Test IDs...", test_fake_count)
    curated_test_fakes = synthesize_and_curate_quota(
        input_manifest=test_reals_manifest,
        output_raw_manifest=test_fakes_raw,
        output_curated_manifest=test_fakes_curated,
        output_dir=test_fakes_dir,
        target_count=test_fake_count,
        prompt_engine=prompt_eng,
        report_path=syn_dir / "master_test_curation_report.json",
        device=device,
        seed=seed + 100,
    )

    # 3. Synthesize Baseline D0 Train Fakes (Quota Ensured)
    train_fakes_raw = syn_dir / "d0_train_fakes_raw.jsonl"
    train_fakes_curated = syn_dir / "d0_train_fakes_curated.jsonl"
    train_fakes_dir = syn_dir / "d0_train_images"
    train_reals_manifest = partitions_dir / "train_reals.jsonl"

    logger.info(">>> Synthesizing %d Baseline D0 Train Fakes...", train_fake_count)
    curated_train_fakes = synthesize_and_curate_quota(
        input_manifest=train_reals_manifest,
        output_raw_manifest=train_fakes_raw,
        output_curated_manifest=train_fakes_curated,
        output_dir=train_fakes_dir,
        target_count=train_fake_count,
        prompt_engine=prompt_eng,
        report_path=syn_dir / "d0_train_curation_report.json",
        device=device,
        seed=seed + 200,
    )

    # 4. Synthesize Baseline D0 Val Fakes (Quota Ensured)
    val_fakes_raw = syn_dir / "d0_val_fakes_raw.jsonl"
    val_fakes_curated = syn_dir / "d0_val_fakes_curated.jsonl"
    val_fakes_dir = syn_dir / "d0_val_images"
    val_reals_manifest = partitions_dir / "val_reals.jsonl"

    logger.info(">>> Synthesizing %d Baseline D0 Val Fakes...", val_fake_count)
    curated_val_fakes = synthesize_and_curate_quota(
        input_manifest=val_reals_manifest,
        output_raw_manifest=val_fakes_raw,
        output_curated_manifest=val_fakes_curated,
        output_dir=val_fakes_dir,
        target_count=val_fake_count,
        prompt_engine=prompt_eng,
        report_path=syn_dir / "d0_val_curation_report.json",
        device=device,
        seed=seed + 300,
    )

    curated_fakes_dict = {
        "test": curated_test_fakes,
        "train": curated_train_fakes,
        "val": curated_val_fakes,
    }

    # 5. Package D0 Benchmarks and Frozen Test Set
    manifest_paths = package_benchmark_d0(partitions, curated_fakes_dict, benchmarks_dir)

    # 6. Train Baseline Detector M0
    logger.info(">>> Training Baseline Detector M0 on D0...")
    all_train = load_metadata_manifest(manifest_paths["d0_train"])
    all_val = load_metadata_manifest(manifest_paths["d0_val"])
    all_test = load_metadata_manifest(manifest_paths["d0_test"])

    dm = DeepfakeImageDataModule(
        train_metadata=all_train,
        val_metadata=all_val,
        test_metadata=all_test,
        batch_size=16,
        seed=seed,
    )
    dm.setup()

    m0_detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
    m0_best = ckpt_dir / "baseline_d0_best.pt"
    if m0_best.exists():
        logger.info(">>> Found pre-existing Baseline Detector M0 checkpoint at %s, skipping retraining and reusing.", m0_best)
        st = torch.load(m0_best, map_location=device)
        if "model_state_dict" in st:
            m0_detector.load_state_dict(st["model_state_dict"])
    else:
        train_deepfake_image_detector(
            model=m0_detector,
            datamodule=dm,
            num_epochs=epochs,
            learning_rate=2e-4,
            checkpoint_dir=ckpt_dir,
            checkpoint_prefix="baseline_d0",
            device=device,
            seed=seed,
        )

    # 7. Evaluate M0 on Master Frozen Test Vault
    m0_best = ckpt_dir / "baseline_d0_best.pt"
    if m0_best.exists():
        st = torch.load(m0_best, map_location=device)
        if "model_state_dict" in st:
            m0_detector.load_state_dict(st["model_state_dict"])

    test_metrics = evaluate_image_detector(m0_detector, dm.test_dataloader(), device=device)
    logger.info(">>> Baseline M0 Master Frozen Test Results: %s", test_metrics)

    summary = {
        "stage": "Block1_Baseline_M0",
        "dry_run": False,
        "d0_counts": {
            "train_real": len(partitions["train"]),
            "train_fake": len(curated_fakes_dict["train"]),
            "val_real": len(partitions["val"]),
            "val_fake": len(curated_fakes_dict["val"]),
            "test_real": len(partitions["test"]),
            "test_fake": len(curated_fakes_dict["test"]),
            "test_total": len(all_test),
        },
        "baseline_test_metrics": test_metrics,
    }

    with open(logs_dir / "block1_baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Block 1 completed successfully! Manifests & metrics saved.")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Block 1: Master Frozen Test Vault & Baseline M0 Initialization")
    parser.add_argument("--raw-manifest", type=Path, default=Path("data/raw/manifest.jsonl"))
    parser.add_argument("--test-fake-count", type=int, default=600, help="Number of test fakes to synthesize")
    parser.add_argument("--train-fake-count", type=int, default=1400, help="Number of train fakes to synthesize")
    parser.add_argument("--val-fake-count", type=int, default=200, help="Number of val fakes to synthesize")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for baseline detector training")
    parser.add_argument("--device", type=str, default="cuda", help="Target execution device (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry run verification")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dev = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    run_block1_pipeline(
        raw_manifest_path=args.raw_manifest,
        test_fake_count=args.test_fake_count,
        train_fake_count=args.train_fake_count,
        val_fake_count=args.val_fake_count,
        epochs=args.epochs,
        device=dev,
        dry_run=args.dry_run,
        seed=args.seed,
    )
