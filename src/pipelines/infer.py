"""Inference pipeline for running dinoViT deepfake detection on test images."""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from typing import Any, Dict, List
import numpy as np
from PIL import Image
import torch

from src.common.config import load_yaml_with_defaults
from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.models.detector.image_detector import DeepfakeImageDetector

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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
        required=True,
        help="Path to input image file or directory of images",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/active_learning_m3_best.pt",
        help="Path to trained detector checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold tau on P(fake) (default: 0.50)",
    )
    return parser.parse_args()


def _collect_image_paths(input_target: Path) -> List[Path]:
    """Collect single image file or sorted image files inside a directory."""
    if input_target.is_file():
        return [input_target]
    if input_target.is_dir():
        return sorted(p for p in input_target.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    raise FileNotFoundError(f"Inference input path does not exist: {input_target}")


def infer_pipeline(
    config_path: str,
    input_path: str,
    checkpoint_path: str = "experiments/checkpoints/active_learning_m3_best.pt",
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Execute dinoViT inference pipeline at native 512->504px resolution on real images."""
    logger = get_logger("infer_pipeline")
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Verified checkpoint not found at: {ckpt_file}")

    cfg = load_yaml_with_defaults(config_path) if Path(config_path).is_file() else {}
    seed = int(cfg.get("training", {}).get("seed", cfg.get("project", {}).get("seed", 42)))
    seed_everything(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg.get("detection", {}).get("model", {}).get("spatial_backbone", "dinov2_small")
    model = DeepfakeImageDetector(backbone=backbone, num_classes=1, pretrained=False)

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    image_paths = _collect_image_paths(Path(input_path))
    if not image_paths:
        raise ValueError(f"No valid images found at inference input path: {input_path}")

    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for img_path in image_paths:
            with Image.open(img_path) as im:
                rgb = im.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
                arr = np.array(rgb, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logit = model(tensor).view(-1)[0]
                prob = float(torch.sigmoid(logit).item())
            label = "fake" if prob >= threshold else "real"
            logger.info("Inference [%s] -> P(fake)=%.4f (%s @ tau=%.2f)", img_path.name, prob, label, threshold)
            results.append({"file_path": str(img_path), "p_fake": prob, "prediction": label})

    return results


if __name__ == "__main__":
    args = parse_args()
    infer_pipeline(args.config, args.input, checkpoint_path=args.checkpoint, threshold=args.threshold)
