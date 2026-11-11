"""Inference pipeline for running dinoViT deepfake detection on test images."""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import torch

from src.common.logger import get_logger
from src.models.detector.image_detector import DeepfakeImageDetector


def parse_args() -> argparse.Namespace:
    """Parse inference command line arguments."""
    parser = argparse.ArgumentParser(description="Infer dinoViT Deepfake Detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/samples",
        help="Path to input image or image directory",
    )
    return parser.parse_args()


def infer_pipeline(config_path: str, input_path: str) -> None:
    """Execute dinoViT inference pipeline."""
    logger = get_logger("infer_pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepfakeImageDetector(backbone="dino_vit", num_classes=1)
    model.to(device)
    model.eval()

    logger.info("Loaded dinoViT detector for inference on target: %s", input_path)


if __name__ == "__main__":
    args = parse_args()
    infer_pipeline(args.config, args.input)
