"""Evaluation pipeline script for benchmarking baseline vs fine-tuned dinoViT models."""

import argparse
from pathlib import Path
from typing import Dict
import torch
import yaml

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_datamodule import DeepfakeImageDataModule
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector
from src.pipelines.train_image_detector import combine_training_manifests


def parse_args() -> argparse.Namespace:
    """Parse evaluation command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate dinoViT Deepfake Detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/best_model.pt",
        help="Path to checkpoint weights",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config safely."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def evaluate_pipeline(
    config_path: str = "configs/config.yaml",
    checkpoint_path: str | None = None,
) -> Dict[str, float]:
    """Execute dinoViT evaluation pipeline."""
    logger = get_logger("evaluate_pipeline")
    cfg = load_config(config_path)

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation running on device: %s", device)

    backbone = cfg.get("detection", {}).get("model", {}).get("spatial_backbone", "dinov2_small")
    model = DeepfakeImageDetector(backbone=backbone, num_classes=1, pretrained=True)
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info("Loading detector weights from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()

    meta_list = combine_training_manifests(
        baseline_manifest=Path("data/raw/manifest.jsonl"),
        synthetic_manifest=Path("data/processed/curated_manifest.jsonl"),
        hard_sample_manifest=Path("data/processed/hard_samples_manifest.jsonl"),
    )

    if meta_list:
        dm = DeepfakeImageDataModule(
            metadata_list=meta_list,
            batch_size=8,
            seed=42,
        )
        dm.setup()
        test_loader = dm.test_dataloader()
        return evaluate_image_detector(model, test_loader, device=device)

    mock_batch = {
        "images": torch.rand(4, 3, 64, 64),
        "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "is_hard_sample": torch.tensor([False, True, False, False], dtype=torch.bool),
    }
    return evaluate_image_detector(model, [mock_batch], device=device)


if __name__ == "__main__":
    args = parse_args()
    evaluate_pipeline(args.config, args.checkpoint)
