"""Training pipeline script adhering to script-only and full-state checkpoint rules."""

import argparse
from pathlib import Path

import torch
import yaml

from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.models.builder import build_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Computer Vision Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config safely."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train_pipeline(config_path: str = "configs/config.yaml") -> None:
    """Execute training pipeline."""
    logger = get_logger("train_pipeline")
    cfg = load_config(config_path)

    seed = cfg.get("training", {}).get("seed", 42)
    seed_everything(seed=seed)
    logger.info(f"Initialized random seed: {seed}")

    device_str = cfg.get("training", {}).get("device", "cuda")
    device = torch.device(
        device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"
    )
    logger.info(f"Target execution device: {device}")

    # Build model
    model = build_model(architecture="custom_cnn", num_classes=10)
    model.to(device)
    logger.info(f"Model instantiated successfully: {model.__class__.__name__}")

    # Checkpoint output directory
    checkpoint_dir = Path("experiments/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training pipeline scaffolding ready.")


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args.config)
