"""Evaluation pipeline script for benchmarking trained models."""

import argparse

import torch
import yaml

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.models.builder import build_model


def parse_args() -> argparse.Namespace:
    """Parse evaluation command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Computer Vision Model")
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
) -> None:
    """Execute evaluation pipeline."""
    logger = get_logger("evaluate_pipeline")
    load_config(config_path)

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation running on device: {device}")

    model = build_model(architecture="custom_cnn", num_classes=10)
    model.to(device)
    model.eval()

    logger.info("Evaluation pipeline scaffolding ready.")


if __name__ == "__main__":
    args = parse_args()
    evaluate_pipeline(args.config, args.checkpoint)
