"""Evaluate all 5 Deepfake Detector Models (M0, M1, M2_coldstart, M2_refined, M3_dual_engine) on Universal Benchmark Suite.

Evaluates 20,400 standardized square images (10,200 Real vs 10,200 Fake across 34 methods and 8 families).
Re-uses cached predictions for M0..M2* and runs fast GPU inference for M3 only.
Generates:
- experiments/benchmarks/tables/benchmark_5models_global_summary.csv
- experiments/benchmarks/tables/benchmark_5models_family_summary.csv
- experiments/benchmarks/figures/benchmark_5models_roc_curves.png
- experiments/benchmarks/figures/benchmark_5models_precision_recall_curves.png
- experiments/benchmarks/figures/benchmark_5models_family_recall.png
"""

import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.common.logger import get_logger
from src.common.metrics import (
    compute_accuracy,
    compute_bypass_rate,
    compute_precision,
    compute_recall,
    compute_roc_auc,
)
from src.models.detector.image_detector import DeepfakeImageDetector
from scripts.evaluate_universal_benchmark_4models import (
    METHOD_TAXONOMY,
    FAMILY_COLORS,
    UniversalBenchmarkDataset,
    collate_benchmark,
    compute_roc_curve_vectorized,
    compute_pr_curve_vectorized,
    compute_comprehensive_metrics,
)

logger = get_logger("evaluate_universal_benchmark_5models")

MODEL_CONFIGS = [
    {
        "key": "M0_baseline",
        "display_name": "M₀ (Baseline D₀)",
        "ckpt": Path("experiments/checkpoints/baseline_d0_best.pt"),
        "color": "#1f77b4",
        "style": "--",
    },
    {
        "key": "M1_active_r1",
        "display_name": "M₁ (Active Learning R1)",
        "ckpt": Path("experiments/checkpoints/active_learning_m1_best.pt"),
        "color": "#ff7f0e",
        "style": "-.",
    },
    {
        "key": "M2_coldstart",
        "display_name": "M₂ (Cold-Start Scratch)",
        "ckpt": Path("experiments/checkpoints/active_learning_m2_coldstart_backup.pt"),
        "color": "#d62728",
        "style": ":",
    },
    {
        "key": "M2_refined_warmstart",
        "display_name": "M₂* (Refined Warm-Start)",
        "ckpt": Path("experiments/checkpoints/active_learning_m2_best.pt"),
        "color": "#2ca02c",
        "style": "-",
    },
    {
        "key": "M3_dual_engine",
        "display_name": "M₃ (Dual-Engine Champion)",
        "ckpt": Path("experiments/checkpoints/active_learning_m3_best.pt"),
        "color": "#9467bd",
        "style": "-",
    },
]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Universal Benchmark Suite 5-Model Evaluation on device: %s", device)

    benchmark_csv = Path("data/benchmarks/universal_deepfake_benchmark/splits/test_universal_deepfake_benchmark.csv")
    base_img_dir = Path("data/benchmarks/universal_deepfake_benchmark/test_images")
    tbl_dir = Path("experiments/benchmarks/tables")
    fig_dir = Path("experiments/benchmarks/figures")
    log_dir = Path("experiments/benchmarks/logs")

    tbl_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest records
    with open(benchmark_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    logger.info("Loaded %d benchmark records", len(records))

    dataset = UniversalBenchmarkDataset(records, base_img_dir=base_img_dir, target_size=(224, 224))
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_benchmark,
    )

    npz_4models_path = log_dir / "benchmark_4models_predictions.npz"
    npz_5models_path = log_dir / "benchmark_5models_predictions.npz"

    model_predictions: Dict[str, List[float]] = {}
    y_true: np.ndarray
    methods_arr: np.ndarray
    families_arr: np.ndarray
    sources_arr: np.ndarray

    if npz_5models_path.exists():
        logger.info("Found existing 5-model prediction cache: %s", npz_5models_path)
        loaded = np.load(npz_5models_path)
        y_true = loaded["y_true"]
        methods_arr = loaded["methods"]
        families_arr = loaded["families"]
        sources_arr = loaded["sources"]
        model_predictions = {cfg["key"]: loaded[f"probs_{cfg['key']}"].tolist() for cfg in MODEL_CONFIGS}
    elif npz_4models_path.exists():
        logger.info("Found 4-model cache. Loading and computing M3 inference...")
        loaded = np.load(npz_4models_path)
        y_true = loaded["y_true"]
        methods_arr = loaded["methods"]
        families_arr = loaded["families"]
        sources_arr = loaded["sources"]
        for cfg in MODEL_CONFIGS[:4]:
            model_predictions[cfg["key"]] = loaded[f"probs_{cfg['key']}"].tolist()

        # Run inference for M3 only!
        m3_cfg = MODEL_CONFIGS[4]
        logger.info("Loading M3 model weights from %s...", m3_cfg["ckpt"])
        m3_model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
        m3_state = torch.load(m3_cfg["ckpt"], map_location="cpu")
        if "model_state_dict" in m3_state:
            m3_model.load_state_dict(m3_state["model_state_dict"])
        elif isinstance(m3_state, dict):
            m3_model.load_state_dict(m3_state)
        m3_model.unfreeze_backbone(unfreeze_last_n_blocks=1)
        m3_model.eval()
        m3_model.to(device)

        m3_preds: List[float] = []
        logger.info("Running M3 inference on 20,400 images...")
        t0 = time.time()
        with torch.no_grad():
            for batch_idx, (imgs, lbls, methods, sources, families) in enumerate(dataloader):
                imgs = imgs.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = m3_model(imgs)
                    probs = torch.sigmoid(logits).view(-1).cpu().tolist()
                m3_preds.extend(probs)
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
                    logger.info("  [Batch %d/%d (%.1f%%)] -> %d samples", batch_idx + 1, len(dataloader), (batch_idx + 1) / len(dataloader) * 100, len(m3_preds))

        logger.info("M3 inference completed in %.1f seconds!", time.time() - t0)
        model_predictions[m3_cfg["key"]] = m3_preds

        # Save 5-model cache
        npz_data = {
            "y_true": y_true,
            "methods": methods_arr,
            "families": families_arr,
            "sources": sources_arr,
        }
        for cfg in MODEL_CONFIGS:
            npz_data[f"probs_{cfg['key']}"] = np.array(model_predictions[cfg["key"]], dtype=np.float32)
        np.savez_compressed(npz_5models_path, **npz_data)
        logger.info("Saved 5-model cache to %s", npz_5models_path)
    else:
        raise FileNotFoundError(f"Missing prediction cache: {npz_4models_path}")

    # 2. Compute Global & Granular Metrics
    logger.info("Computing metrics for all 5 models...")
    global_results: Dict[str, Any] = {}
    family_results: Dict[str, Dict[str, Any]] = {}

    distinct_families = sorted(list(set(METHOD_TAXONOMY.values())))

    for cfg in MODEL_CONFIGS:
        key = cfg["key"]
        probs = np.array(model_predictions[key])
        g_mets = compute_comprehensive_metrics(y_true, probs, threshold=0.50)
        global_results[key] = {
            "display_name": cfg["display_name"],
            "checkpoint": str(cfg["ckpt"]),
            "metrics": g_mets,
        }

    for fam in distinct_families:
        fam_mask = families_arr == fam
        family_results[fam] = {"sample_count": int(np.sum(fam_mask)), "is_real": fam.startswith("Real")}
        for cfg in MODEL_CONFIGS:
            key = cfg["key"]
            fam_probs = np.array(model_predictions[key])[fam_mask]
            if fam.startswith("Real"):
                acc = float(np.mean(fam_probs < 0.50))
                family_results[fam][f"{key}_accuracy"] = acc
                family_results[fam][f"{key}_mean_conf"] = float(np.mean(fam_probs))
            else:
                rec = float(np.mean(fam_probs >= 0.50))
                family_results[fam][f"{key}_recall"] = rec
                family_results[fam][f"{key}_bypass"] = 1.0 - rec
                family_results[fam][f"{key}_mean_conf"] = float(np.mean(fam_probs))

    # 3. Export Global CSV Summary
    tbl_global_path = tbl_dir / "benchmark_5models_global_summary.csv"
    with open(tbl_global_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Display Name", "Accuracy", "Fake Recall (↑)", "Bypass Rate (↓)",
            "Real Specificity (↑)", "Real FPR (↓)", "Precision", "F1-Score", "ROC-AUC (↑)", "PR-AUC (↑)", "Brier Score (↓)"
        ])
        for cfg in MODEL_CONFIGS:
            key = cfg["key"]
            m = global_results[key]["metrics"]
            writer.writerow([
                key, cfg["display_name"],
                f"{m['accuracy']*100:.2f}%",
                f"{m['fake_recall']*100:.2f}%",
                f"{m['bypass_rate']*100:.2f}%",
                f"{m['real_accuracy']*100:.2f}%",
                f"{m['real_fpr']*100:.2f}%",
                f"{m['precision']*100:.2f}%",
                f"{m['f1_score']*100:.2f}%",
                f"{m['roc_auc']:.4f}",
                f"{m['pr_auc']:.4f}",
                f"{m['brier_score']:.4f}",
            ])
    logger.info("Saved global CSV to %s", tbl_global_path)

    # 4. Export Family CSV Summary
    tbl_family_path = tbl_dir / "benchmark_5models_family_summary.csv"
    with open(tbl_family_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["Family", "Sample Count"] + [f"{cfg['display_name']} Recall/Spec" for cfg in MODEL_CONFIGS]
        writer.writerow(header)
        for fam in distinct_families:
            row = [fam, family_results[fam]["sample_count"]]
            for cfg in MODEL_CONFIGS:
                key = cfg["key"]
                if family_results[fam]["is_real"]:
                    row.append(f"{family_results[fam][f'{key}_accuracy']*100:.2f}% (Spec)")
                else:
                    row.append(f"{family_results[fam][f'{key}_recall']*100:.2f}%")
            writer.writerow(row)
    logger.info("Saved family CSV to %s", tbl_family_path)

    # 5. Generate ROC Curves Plot
    plt.figure(figsize=(9, 7), dpi=300)
    for cfg in MODEL_CONFIGS:
        key = cfg["key"]
        probs = np.array(model_predictions[key])
        fpr, tpr = compute_roc_curve_vectorized(y_true, probs)
        auc = global_results[key]["metrics"]["roc_auc"]
        plt.plot(fpr, tpr, color=cfg["color"], linestyle=cfg["style"], linewidth=2.2, label=f"{cfg['display_name']} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    plt.title("ROC Curves Comparison Across 5 Detector Models (Universal Benchmark)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_fig_path = fig_dir / "benchmark_5models_roc_curves.png"
    plt.savefig(roc_fig_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curve figure to %s", roc_fig_path)

    # 6. Generate PR Curves Plot
    plt.figure(figsize=(9, 7), dpi=300)
    for cfg in MODEL_CONFIGS:
        key = cfg["key"]
        probs = np.array(model_predictions[key])
        recall_pts, prec_pts, _ = compute_pr_curve_vectorized(y_true, probs)
        pr_auc = global_results[key]["metrics"]["pr_auc"]
        plt.plot(recall_pts, prec_pts, color=cfg["color"], linestyle=cfg["style"], linewidth=2.2, label=f"{cfg['display_name']} (PR-AUC = {pr_auc:.4f})")

    plt.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.5, label="Baseline Prevalance (0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curves Across 5 Detector Models (Universal Benchmark)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=9, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pr_fig_path = fig_dir / "benchmark_5models_precision_recall_curves.png"
    plt.savefig(pr_fig_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved PR curve figure to %s", pr_fig_path)

    # 7. Generate Family Bar Chart
    fake_families = [fam for fam in distinct_families if not fam.startswith("Real")]
    x = np.arange(len(fake_families))
    width = 0.16

    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    for i, cfg in enumerate(MODEL_CONFIGS):
        key = cfg["key"]
        vals = [family_results[fam][f"{key}_recall"] * 100 for fam in fake_families]
        ax.bar(x + (i - 2) * width, vals, width, label=cfg["display_name"], color=cfg["color"], edgecolor="black", linewidth=0.8, alpha=0.9)

    ax.set_ylabel("Fake Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("Family Recall Breakdown Across 5 Detector Models (20,400 Images)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(" / ", "\n") for f in fake_families], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout()
    fam_fig_path = fig_dir / "benchmark_5models_family_recall.png"
    plt.savefig(fam_fig_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved family recall figure to %s", fam_fig_path)

    logger.info("=== UNIVERSAL BENCHMARK 5 MÔ HÌNH HOÀN TẤT THÀNH CÔNG ===")


if __name__ == "__main__":
    main()
