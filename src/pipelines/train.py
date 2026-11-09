"""Unified training pipeline entrypoint for dinoViT Deepfake Detection with Synthetic + Hard Samples."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import yaml

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector
from src.pipelines.train_image_detector import (
    combine_training_manifests,
    train_deepfake_image_detector,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune dinoViT Deepfake Detector on Baseline + Synthetic + Hard Samples"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute single sanity epoch for verification",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--max-real-samples",
        type=int,
        default=150,
        help="Maximum real samples to balance with synthetic fakes during training",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["D0", "D1", "D2", "all"],
        default="all",
        help="Training stage to run: D0 (baseline), D1 (+synthetic), D2 (+hard samples), or all (comparative benchmark)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config safely."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_stage_dataset(
    stage: str,
    baseline_path: Path,
    synthetic_path: Path,
    hard_path: Path,
    max_real_samples: int = 150,
) -> List[ImageMetadata]:
    """Construct dataset metadata for a specific benchmark stage (D0, D1, D2)."""
    real_records: List[ImageMetadata] = []
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                real_records.append(ImageMetadata.from_dict(d))
                if len(real_records) >= max_real_samples:
                    break

    syn_records: List[ImageMetadata] = []
    if synthetic_path.exists():
        with open(synthetic_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                syn_records.append(ImageMetadata.from_dict(json.loads(line)))

    hard_records: List[ImageMetadata] = []
    if hard_path.exists():
        with open(hard_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                hard_records.append(ImageMetadata.from_dict(json.loads(line)))

    if stage == "D0":
        # Baseline stage: real images + initial baseline fakes (first half of synthetic identities)
        # to leave held-out synthetic fakes for Phase 2 mining and testing
        n_base = max(1, len(syn_records) // 2) if len(syn_records) > 2 else len(syn_records)
        base_fakes = []
        for s in syn_records[:n_base]:
            base_fakes.append(
                ImageMetadata(
                    image_id=s.image_id,
                    label=1,
                    manipulation_type=s.manipulation_type,
                    prompt=s.prompt,
                    source_real_id=s.source_real_id,
                    file_path=s.file_path,
                    bbox=s.bbox,
                    is_hard_sample=False,
                    baseline_score=None,
                    data_source="baseline",
                    demographics=s.demographics,
                )
            )
        return real_records + base_fakes

    elif stage == "D1":
        # Stage 1: real images + all standard curated synthetic fakes (no hard-sample weighting)
        standard_fakes = [
            ImageMetadata(
                image_id=s.image_id,
                label=1,
                manipulation_type=s.manipulation_type,
                prompt=s.prompt,
                source_real_id=s.source_real_id,
                file_path=s.file_path,
                bbox=s.bbox,
                is_hard_sample=False,
                baseline_score=s.baseline_score,
                data_source="synthetic",
                demographics=s.demographics,
            )
            for s in syn_records
        ]
        return real_records + standard_fakes

    else:
        # Stage 2: real + all synthetic fakes + mined hard samples with active hard-sample flags
        combined_by_id = {m.image_id: m for m in real_records}
        for s in syn_records:
            combined_by_id[s.image_id] = s
        for h in hard_records:
            combined_by_id[h.image_id] = h
        return list(combined_by_id.values())


def train_single_stage(
    stage_name: str,
    meta_list: List[ImageMetadata],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    backbone: str,
    pretrained: bool,
    seed: int,
    hard_sample_loss_weight: float = 1.0,
    hard_sample_oversample_factor: int = 1,
    unfreeze_last_n_blocks: Optional[int] = None,
    checkpoint_dir: Path = Path("experiments/checkpoints"),
) -> Dict[str, float]:
    """Train and evaluate deepfake detector for one stage."""
    logger = get_logger("train_single_stage")
    logger.info("--- Starting Training for Stage %s (%d samples) ---", stage_name, len(meta_list))

    stage_prefix_map = {
        "D0": "baseline_d0",
        "D1": "augmented_d1",
        "D2": "hard_boosted_d2",
        "D3": "adaptive_d3",
    }
    ckpt_prefix = stage_prefix_map.get(stage_name, stage_name.lower())

    stage_folder_map = {
        "D0": "D0_baseline",
        "D1": "D1_augmented",
        "D2": "D2_hard_boosted",
    }
    folder_name = stage_folder_map.get(stage_name, stage_name)
    bench_dir = Path("data/benchmarks") / folder_name
    train_path = bench_dir / "train.jsonl"
    val_path = bench_dir / "val.jsonl"
    test_path = bench_dir / "test.jsonl"

    if meta_list:
        logger.info("Using dynamic evolutionary meta_list for stage %s (%d samples)...", stage_name, len(meta_list))
        dm = DeepfakeImageDataModule(
            metadata_list=meta_list,
            batch_size=batch_size,
            train_ratio=0.85,
            val_ratio=0.15,
            test_ratio=0.0,
            hard_sample_oversample_factor=hard_sample_oversample_factor,
            identity_disjoint=True,
            seed=seed,
        )
    elif train_path.exists() and val_path.exists() and test_path.exists():
        logger.info("Loading pre-packaged benchmark splits from %s...", bench_dir)
        with open(train_path, "r", encoding="utf-8") as f:
            train_records = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]
        with open(val_path, "r", encoding="utf-8") as f:
            val_records = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]
        with open(test_path, "r", encoding="utf-8") as f:
            test_records = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

        dm = DeepfakeImageDataModule(
            train_metadata=train_records,
            val_metadata=val_records,
            test_metadata=test_records,
            batch_size=batch_size,
            hard_sample_oversample_factor=1,
            seed=seed,
        )
    else:
        logger.info("Using meta_list partitioning for stage %s (%d samples)...", stage_name, len(meta_list))
        dm = DeepfakeImageDataModule(
            metadata_list=meta_list,
            batch_size=batch_size,
            train_ratio=0.85,
            val_ratio=0.15,
            test_ratio=0.0,
            hard_sample_oversample_factor=hard_sample_oversample_factor,
            identity_disjoint=True,
            seed=seed,
        )
    dm.setup()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for f in [checkpoint_dir / f"{ckpt_prefix}_best.pt", checkpoint_dir / f"{ckpt_prefix}_last.pt"]:
        if f.exists():
            f.unlink()

    model = DeepfakeImageDetector(backbone=backbone, num_classes=1, pretrained=pretrained)
    train_deepfake_image_detector(
        model=model,
        datamodule=dm,
        num_epochs=epochs,
        learning_rate=lr,
        hard_sample_loss_weight=hard_sample_loss_weight,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=ckpt_prefix,
        device=device,
        seed=seed,
    )

    best_file = checkpoint_dir / f"{ckpt_prefix}_best.pt"
    alias_file = checkpoint_dir / f"{ckpt_prefix}.pt"
    if best_file.exists():
        import shutil
        shutil.copy2(best_file, alias_file)
        ckpt = torch.load(best_file, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])

    test_loader = dm.test_dataloader()
    metrics = evaluate_image_detector(model, test_loader, device=device)
    logger.info("Stage %s Test Results: %s", stage_name, metrics)
    return metrics


def train_pipeline(
    config_path: str = "configs/config.yaml",
    dry_run: bool = False,
    epochs_override: Optional[int] = None,
    max_real_samples: int = 150,
    stage_choice: str = "all",
) -> dict[str, Any]:
    """Execute the dinoViT deepfake detector fine-tuning and benchmarking pipeline."""
    logger = get_logger("train_pipeline")
    cfg = load_config(config_path)

    seed = int(cfg.get("training", {}).get("seed", 42))
    seed_everything(seed=seed)

    device = "cuda" if (cfg.get("training", {}).get("device", "cpu") == "cuda" and torch.cuda.is_available()) else "cpu"
    epochs = 1 if dry_run else (epochs_override or int(cfg.get("training", {}).get("epochs", 3)))
    lr = float(cfg.get("training", {}).get("learning_rate", 2e-4))
    batch_size = int(cfg.get("training", {}).get("batch_size", 4))

    backbone = cfg.get("detection", {}).get("model", {}).get("spatial_backbone", "dinov2_small")
    pretrained = bool(cfg.get("detection", {}).get("model", {}).get("pretrained_backbone", True))

    paths_cfg = cfg.get("paths", {})
    baseline_path = Path(paths_cfg.get("baseline_manifest", "data/01_raw_faces/raw_faces_manifest.jsonl"))
    if not baseline_path.exists():
        baseline_path = Path("data/raw/manifest.jsonl")

    syn_path = Path(paths_cfg.get("synthetic_manifest", "data/synthetic/curated_manifest.jsonl"))
    if not syn_path.exists():
        syn_path = Path("data/synthetic/curated_manifest.jsonl")

    hard_path = Path(paths_cfg.get("hard_samples_manifest", "data/hard_samples/hard_samples_manifest.jsonl"))
    if not hard_path.exists():
        hard_path = Path("data/hard_samples/hard_samples_manifest.jsonl")
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if stage_choice in ["D0", "D1", "D2"]:
        hw = 2.0 if stage_choice == "D2" else 1.0
        hof = 2 if stage_choice == "D2" else 1
        meta_list = build_stage_dataset(stage_choice, baseline_path, syn_path, hard_path, max_real_samples)
        metrics = train_single_stage(
            stage_choice, meta_list, epochs, batch_size, lr, device, backbone, pretrained, seed, hw, hof
        )
        return metrics

    # Comparative benchmark across D0, D1, D2
    logger.info("Running comparative benchmark across D0 (Baseline) -> D1 (+Synthetic) -> D2 (+Hard Samples)...")
    d0_meta = build_stage_dataset("D0", baseline_path, syn_path, hard_path, max_real_samples)
    d0_metrics = train_single_stage("D0", d0_meta, epochs, batch_size, lr, device, backbone, pretrained, seed, 1.0, 1)

    d1_meta = build_stage_dataset("D1", baseline_path, syn_path, hard_path, max_real_samples)
    d1_metrics = train_single_stage("D1", d1_meta, epochs, batch_size, lr, device, backbone, pretrained, seed, 1.0, 1)

    d2_meta = build_stage_dataset("D2", baseline_path, syn_path, hard_path, max_real_samples)
    d2_metrics = train_single_stage("D2", d2_meta, epochs, batch_size, lr, device, backbone, pretrained, seed, 2.0, 2)

    comparison = {
        "D0_baseline": d0_metrics,
        "D1_synthetic": d1_metrics,
        "D2_hard_samples": d2_metrics,
        "deltas": {
            "roc_auc_gain": round(d2_metrics.get("roc_auc", 0.0) - d0_metrics.get("roc_auc", 0.0), 4),
            "hard_sample_recall_gain": round(d2_metrics.get("hard_sample_recall", 0.0) - d0_metrics.get("hard_sample_recall", 0.0), 4),
            "bypass_rate_reduction": round(d0_metrics.get("bypass_rate", 0.0) - d2_metrics.get("bypass_rate", 0.0), 4),
        },
    }

    comp_file = log_dir / "benchmark_comparison.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparative benchmark written to %s:\n%s", comp_file, json.dumps(comparison["deltas"], indent=2))
    return comparison


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(
        config_path=args.config,
        dry_run=args.dry_run,
        epochs_override=args.epochs,
        max_real_samples=args.max_real_samples,
        stage_choice=args.stage,
    )
