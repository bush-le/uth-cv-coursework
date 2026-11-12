"""Unified Quad-Vault Benchmark Evaluation of 5 Detector Models:

Models evaluated:
1. M0_baseline: baseline_d0_best.pt
2. M1_active_r1: active_learning_m1_best.pt
3. M2_coldstart: active_learning_m2_coldstart_backup.pt
4. M2_refined: active_learning_m2_best.pt
5. M3_dual_engine: active_learning_m3_best.pt

Vaults evaluated:
1. Master Frozen Test Vault (1,178 samples - SD 1.5 in-distribution)
2. Cumulative Challenge Test Vault (213 samples - Hard samples bypassed in AL)
3. Unseen RV5.1 Test Vault (1,215 samples - Zero-leakage OOD RealisticVision v5.1)

Generates:
- experiments/benchmarks/tables/benchmark_5models_quad_vault_summary.csv
- experiments/benchmarks/tables/benchmark_5models_quad_vault_summary.json
- experiments/benchmarks/figures/benchmark_5models_vault_recall_comparison.png
"""

import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.logger import get_logger
from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector

logger = get_logger("evaluate_quad_vault_5models")


def load_metadata_manifest(path: Path) -> List[ImageMetadata]:
    """Load JSONL manifest into list of ImageMetadata."""
    with open(path, "r", encoding="utf-8") as f:
        return [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running 5-Model Quad-Vault Benchmark on device: %s", device)

    # 1. Load Vaults
    master_path = Path("data/benchmarks/master_frozen_test.jsonl")
    chal_path = Path("data/active_learning/cumulative_challenge_test.jsonl")
    unseen_rv51_path = Path("data/benchmarks/D3_clean/unseen_rv51_test.jsonl")

    master_records = load_metadata_manifest(master_path)
    chal_records = load_metadata_manifest(chal_path)
    unseen_records = load_metadata_manifest(unseen_rv51_path)

    batch_size = 32
    master_dm = DeepfakeImageDataModule(train_metadata=[], val_metadata=[], test_metadata=master_records, batch_size=batch_size, seed=42)
    master_dm.setup()
    master_loader = master_dm.test_dataloader()

    chal_dm = DeepfakeImageDataModule(train_metadata=[], val_metadata=[], test_metadata=chal_records, batch_size=batch_size, seed=42)
    chal_dm.setup()
    chal_loader = chal_dm.test_dataloader()

    unseen_dm = DeepfakeImageDataModule(train_metadata=[], val_metadata=[], test_metadata=unseen_records, batch_size=batch_size, seed=42)
    unseen_dm.setup()
    unseen_loader = unseen_dm.test_dataloader()

    vault_loaders = {
        "Master_Frozen_Test": (master_loader, len(master_records)),
        "Challenge_Hard_Test": (chal_loader, len(chal_records)),
        "Unseen_RV51_OOD_Test": (unseen_loader, len(unseen_records)),
    }

    models_config = {
        "M0 (Baseline)": Path("experiments/checkpoints/baseline_d0_best.pt"),
        "M1 (Active R1)": Path("experiments/checkpoints/active_learning_m1_best.pt"),
        "M2 (Cold-Start)": Path("experiments/checkpoints/active_learning_m2_coldstart_backup.pt"),
        "M2* (Refined)": Path("experiments/checkpoints/active_learning_m2_best.pt"),
        "M3 (Dual-Engine Champion)": Path("experiments/checkpoints/active_learning_m3_best.pt"),
    }

    full_results: Dict[str, Dict[str, Any]] = {}
    table_rows: List[Dict[str, Any]] = []

    for model_name, ckpt_path in models_config.items():
        if not ckpt_path.exists():
            logger.warning("Checkpoint %s does not exist, skipping.", ckpt_path)
            continue

        logger.info("=== Loading and Evaluating %s ===", model_name)
        detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
        state = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state:
            detector.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict):
            detector.load_state_dict(state)
        detector.unfreeze_backbone(unfreeze_last_n_blocks=1)
        detector.to(device)

        full_results[model_name] = {}

        for vault_name, (loader, count) in vault_loaders.items():
            metrics = evaluate_image_detector(detector, loader, device=device)
            full_results[model_name][vault_name] = metrics
            logger.info("  [%s on %s (%d)] -> Acc: %.4f | AUC: %.4f | Recall: %.4f | Bypass: %.4f",
                        model_name, vault_name, count,
                        metrics["accuracy"], metrics["roc_auc"], metrics["recall"], metrics["bypass_rate"])

            table_rows.append({
                "Model": model_name,
                "Vault": vault_name,
                "Samples": count,
                "Accuracy": round(metrics["accuracy"], 4),
                "ROC_AUC": round(metrics["roc_auc"], 4),
                "F1_Score": round(metrics["f1_score"], 4),
                "Precision": round(metrics["precision"], 4),
                "Fake_Recall": round(metrics["recall"], 4),
                "Bypass_Rate": round(metrics["bypass_rate"], 4),
                "Loss": round(metrics["loss"], 4),
            })

    # Save Tables
    out_dir = Path("experiments/benchmarks/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_5models_quad_vault_summary.csv"
    if table_rows:
        fieldnames = list(table_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table_rows)
    logger.info("Saved CSV summary to %s", csv_path)

    json_path = out_dir / "benchmark_5models_quad_vault_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info("Saved JSON summary to %s", json_path)

    # Generate Comparative Bar Chart
    fig_dir = Path("experiments/benchmarks/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    vault_keys = list(vault_loaders.keys())
    vault_titles = [
        "Master Frozen Test (SD 1.5)",
        "Challenge Vault (Hard Samples)",
        "Unseen RV5.1 Vault (OOD Realism)",
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172D5", "#D55E00"]

    for ax, v_key, v_title in zip(axes, vault_keys, vault_titles):
        v_rows = [r for r in table_rows if r["Vault"] == v_key]
        models = [r["Model"] for r in v_rows]
        recalls = [r["Fake_Recall"] * 100 for r in v_rows]

        bars = ax.bar(models, recalls, color=colors[:len(models)], edgecolor="black", linewidth=1.2, alpha=0.85)
        ax.set_title(f"{v_title}\nFake Recall (%)", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Fake Recall (%)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)

        for bar, val in zip(bars, recalls):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.suptitle("5-Model Evolution Across Triple Evaluation Vaults (M0 -> M1 -> M2_cold -> M2* -> M3)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plot_path = fig_dir / "benchmark_5models_vault_recall_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved comparative plot to %s", plot_path)
    logger.info("=== ĐÁNH GIÁ ĐA KHO 5 MÔ HÌNH HOÀN TẤT THÀNH CÔNG ===")


if __name__ == "__main__":
    main()
