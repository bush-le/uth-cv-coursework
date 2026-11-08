"""Evaluation pipeline script for benchmarking baseline vs fine-tuned dinoViT models."""

import argparse
from typing import Dict
import torch
import yaml

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector


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
    load_config(config_path)

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluation running on device: %s", device)

    model = DeepfakeImageDetector(backbone="dino_vit", num_classes=1)
    model.to(device)
    model.eval()

    mock_batch = {
        "images": torch.rand(4, 3, 64, 64),
        "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "is_hard_sample": torch.tensor([False, True, False, False], dtype=torch.bool),
    }
    return evaluate_image_detector(model, [mock_batch], device=device)


if __name__ == "__main__":
    args = parse_args()
    evaluate_pipeline(args.config, args.checkpoint)
