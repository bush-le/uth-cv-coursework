"""Inference pipeline for running qualitative predictions on test images."""

import argparse

import torch

from src.common.logger import get_logger
from src.models.builder import build_model


def parse_args() -> argparse.Namespace:
    """Parse inference command line arguments."""
    parser = argparse.ArgumentParser(description="Infer Computer Vision Model")
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
    """Execute inference pipeline."""
    logger = get_logger("infer_pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(architecture="custom_cnn", num_classes=10)
    model.to(device)
    model.eval()

    logger.info(f"Loaded model for inference on target: {input_path}")


if __name__ == "__main__":
    args = parse_args()
    infer_pipeline(args.config, args.input)
