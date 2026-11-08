"""Unified training pipeline entrypoint for dinoViT Deepfake Detection with Synthetic + Hard Samples."""

import argparse
from pathlib import Path
from typing import Any
import torch
import yaml

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector
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
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config safely."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train_pipeline(
    config_path: str = "configs/config.yaml",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the dinoViT deepfake detector fine-tuning pipeline."""
    logger = get_logger("train_pipeline")
    cfg = load_config(config_path)

    seed = int(cfg.get("training", {}).get("seed", 42))
    seed_everything(seed=seed)

    device_str = cfg.get("training", {}).get("device", "cpu")
    device = "cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"

    meta_list = combine_training_manifests(
        baseline_manifest=Path("data/raw/manifest.jsonl"),
        synthetic_manifest=Path("data/processed/curated_manifest.jsonl"),
        hard_sample_manifest=Path("data/processed/hard_samples_manifest.jsonl"),
    )
    cache = None
    if not meta_list or dry_run:
        logger.info("Using synthetic in-memory batch for sanity verification.")
        meta_list = [
            ImageMetadata(
                image_id=f"img_{i}",
                label=i % 2,
                manipulation_type="real" if i % 2 == 0 else "sd_inpaint",
                source_real_id=f"id_{i // 2}",
                is_hard_sample=(i % 4 == 1),
                data_source="hard_sample" if (i % 4 == 1) else "baseline",
            )
            for i in range(16)
        ]
        cache = {f"img_{i}": torch.rand(3, 64, 64) for i in range(16)}

    epochs = 1 if dry_run else int(cfg.get("training", {}).get("epochs", 5))
    lr = float(cfg.get("training", {}).get("learning_rate", 2e-4))
    batch_size = int(cfg.get("training", {}).get("batch_size", 4))

    dm = DeepfakeImageDataModule(
        metadata_list=meta_list,
        cache=cache,
        batch_size=batch_size,
        hard_sample_oversample_factor=2,
        seed=seed,
    )
    model = DeepfakeImageDetector(backbone="dino_vit", num_classes=1)
    metrics = train_deepfake_image_detector(
        model=model,
        datamodule=dm,
        num_epochs=epochs,
        learning_rate=lr,
        device=device,
        seed=seed,
    )
    return metrics


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(config_path=args.config, dry_run=args.dry_run)
