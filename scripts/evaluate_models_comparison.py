import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#!/usr/bin/env python3
"""Unified evaluation of M0, M1, and M2 detectors on both Master Frozen Test Vault and Cumulative Challenge Test Vault."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import torch
from src.common.logger import get_logger
from src.data.image_schema import ImageMetadata
from src.data.image_datamodule import DeepfakeImageDataModule
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector

logger = get_logger(__name__)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running unified benchmark evaluation on device: %s", device)

    # 1. Load Master Frozen Test Vault
    master_vault_path = Path("data/benchmarks/master_frozen_test.jsonl")
    with open(master_vault_path, "r", encoding="utf-8") as f:
        master_records = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

    master_dm = DeepfakeImageDataModule(
        train_metadata=[],
        val_metadata=[],
        test_metadata=master_records,
        batch_size=32,
        seed=42,
    )
    master_dm.setup()
    master_loader = master_dm.test_dataloader()

    # 2. Load Cumulative Challenge Test Vault
    challenge_path = Path("data/active_learning/cumulative_challenge_test.jsonl")
    with open(challenge_path, "r", encoding="utf-8") as f:
        challenge_records = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

    challenge_dm = DeepfakeImageDataModule(
        train_metadata=[],
        val_metadata=[],
        test_metadata=challenge_records,
        batch_size=32,
        seed=42,
    )
    challenge_dm.setup()
    challenge_loader = challenge_dm.test_dataloader()

    checkpoints = {
        "M0_baseline": Path("experiments/checkpoints/baseline_d0_best.pt"),
        "M1_active_round1": Path("experiments/checkpoints/active_learning_m1_best.pt"),
        "M2_active_round2": Path("experiments/checkpoints/active_learning_m2_best.pt"),
    }

    results: Dict[str, Any] = {}

    for model_name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            logger.warning("Checkpoint %s does not exist, skipping.", ckpt_path)
            continue

        logger.info("=== Evaluating %s (%s) ===", model_name, ckpt_path.name)
        detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
        detector.unfreeze_backbone(unfreeze_last_n_blocks=1)
        state = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state:
            detector.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict):
            detector.load_state_dict(state)

        logger.info("Evaluating %s on Master Frozen Test Vault (%d samples)...", model_name, len(master_records))
        m_master = evaluate_image_detector(detector, master_loader, device=device)
        logger.info("%s Master Frozen Test: %s", model_name, m_master)

        logger.info("Evaluating %s on Cumulative Challenge Test Vault (%d samples)...", model_name, len(challenge_records))
        m_chal = evaluate_image_detector(detector, challenge_loader, device=device)
        logger.info("%s Cumulative Challenge Test: %s", model_name, m_chal)

        results[model_name] = {
            "checkpoint": str(ckpt_path),
            "master_frozen_test": m_master,
            "cumulative_challenge_test": m_chal,
        }

    out_file = Path("experiments/logs/model_comparison_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Unified comparison results saved to %s", out_file)


if __name__ == "__main__":
    main()
