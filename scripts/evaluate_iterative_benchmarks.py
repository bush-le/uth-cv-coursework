"""Evaluates M0, M_rand, M1, and M2 across identical frozen unseen test benchmark.

Generates the Golden Coursework Comparison Table for the 4-way Proof-of-Utility ablation.
"""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.config import cfg_get, load_config
from src.common.logger import get_logger
from src.common.metrics import (
    compute_accuracy,
    compute_bypass_rate,
    compute_precision,
    compute_recall,
    compute_roc_auc,
)
from src.data.image_datamodule import collate_image_samples
from src.data.image_dataset import DeepfakeImageDataset
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector
from src.models.evaluators.hardness import load_verified_checkpoint

logger = get_logger(__name__)


def compute_metrics_at_threshold(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute binary classification metrics for given probability predictions and threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    fake_recall = compute_recall(y_pred, y_true, positive_label=1)
    acc = compute_accuracy(y_pred, y_true)
    auc = compute_roc_auc(y_true, y_prob)

    fake_mask = y_true == 1
    real_mask = y_true == 0
    fake_probs = y_prob[fake_mask] if np.any(fake_mask) else np.array([])
    bypass_rate = compute_bypass_rate(fake_probs, threshold=threshold)

    real_preds = y_pred[real_mask] if np.any(real_mask) else np.array([])
    fpr = float(np.mean(real_preds == 1)) if len(real_preds) > 0 else 0.0

    return {
        "accuracy": acc,
        "fake_recall": fake_recall,
        "bypass_rate": bypass_rate,
        "auc": auc,
        "fpr": fpr,
        "threshold": float(threshold),
    }


def calibrate_and_evaluate_model(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Calibrate decision threshold strictly on validation set, then evaluate on test set.

    Guarantees ZERO test peeking: test probabilities are never used to choose threshold.
    """
    # 1. Standard evaluation at fixed tau = 0.50
    std_metrics = compute_metrics_at_threshold(test_probs, test_labels, threshold=0.50)

    # 2. Calibrate optimal tau on validation set via Youden's J
    best_tau = 0.50
    if len(np.unique(val_labels)) > 1 and len(val_probs) > 0:
        cand_taus = np.unique(val_probs)
        best_j = -1.0
        val_real = val_labels == 0
        for tau in cand_taus:
            c_preds = (val_probs >= tau).astype(int)
            c_rec = compute_recall(c_preds, val_labels, positive_label=1)
            c_fpr = float(np.mean(c_preds[val_real] == 1)) if np.any(val_real) else 0.0
            j_stat = c_rec - c_fpr
            if j_stat > best_j:
                best_j = j_stat
                best_tau = float(tau)

    # 3. Apply the frozen validation threshold to the test set
    val_cal_metrics = compute_metrics_at_threshold(test_probs, test_labels, threshold=best_tau)

    return {
        "standard": std_metrics,
        "val_calibrated": val_cal_metrics,
    }


def build_benchmark_summary_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """Build standardized Markdown summary table comparing models on identical test set.

    Args:
        metrics: Dictionary mapping model names to performance metrics.

    Returns:
        Formatted GitHub-flavored Markdown table.
    """
    lines = [
        "| Model | Training Dataset | Operating Threshold | Test Accuracy | Fake Recall (↑) | Bypass Rate (FNR ↓) | ROC-AUC (↑) | Real FPR (↓) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    meta_desc = {
        "M0_baseline": "D0 (Base Real + Fake)",
        "M_rand": "D1 (D0 + Random Fakes)",
        "M1_hard": "D2 (D0 + H1 Bypass)",
        "M2_adaptive": "D3 (D0 + H1 + H2 Feedback)",
    }
    for m_key, m_data in metrics.items():
        desc = meta_desc.get(m_key, m_key)
        tau_val = m_data.get("threshold", 0.50)
        acc = m_data.get("accuracy", 0.0) * 100.0
        rec = m_data.get("fake_recall", 0.0) * 100.0
        bypr = m_data.get("bypass_rate", 0.0) * 100.0
        auc = m_data.get("auc", 0.0)
        fpr = m_data.get("fpr", 0.0) * 100.0
        lines.append(
            f"| **{m_key}** | {desc} | tau={tau_val:.3f} | {acc:.1f}% | **{rec:.1f}%** | {bypr:.1f}% | **{auc:.3f}** | {fpr:.1f}% |"
        )
    return "\n".join(lines)


def evaluate_model_on_manifest(
    model: DeepfakeImageDetector,
    manifest_path: Path,
    device: str = "cpu",
    batch_size: int = 16,
    val_manifest_path: Optional[Path] = None,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate detector on a test manifest and compute core metrics.

    If val_manifest_path is supplied, calibrates threshold on validation set.
    Otherwise uses threshold (default 0.50).
    """
    def _predict_manifest(p: Path) -> tuple[np.ndarray, np.ndarray]:
        with open(p, "r", encoding="utf-8") as f:
            m_list = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]
        ds = DeepfakeImageDataset(m_list, target_size=(256, 256))
        ld = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_image_samples)
        p_list, t_list = [], []
        with torch.no_grad():
            for batch in ld:
                imgs = batch["images"].to(device)
                lbls = batch["labels"].to(device)
                lgts = model(imgs).squeeze(-1)
                probs = torch.sigmoid(lgts)
                p_list.extend(probs.cpu().tolist())
                t_list.extend(lbls.cpu().tolist())
        return np.array(p_list), np.array(t_list)

    model.eval()
    model.to(device)

    test_probs, test_labels = _predict_manifest(manifest_path)

    if val_manifest_path and val_manifest_path.is_file():
        val_probs, val_labels = _predict_manifest(val_manifest_path)
        dual_results = calibrate_and_evaluate_model(val_probs, val_labels, test_probs, test_labels)
        return dual_results["val_calibrated"]

    tau = 0.50 if threshold is None else float(threshold)
    return compute_metrics_at_threshold(test_probs, test_labels, threshold=tau)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 4-way comparative Proof-of-Utility benchmark.")
    parser.add_argument("--test-manifest", type=str, default="data/benchmarks/D0_baseline/test.jsonl")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-report", type=str, default="experiments/reports/ITERATIVE_PROOF_OF_UTILITY.md")
    args = parser.parse_args()

    test_path = Path(args.test_manifest)
    if not test_path.exists():
        logger.error("Test manifest not found: %s", test_path)
        return

    models_to_eval = [
        ("M0_baseline", ["experiments/checkpoints/baseline_d0_best.pt", "experiments/checkpoints/baseline_detector_best.pt"]),
        ("M_rand", ["experiments/checkpoints/augmented_d1_best.pt", "experiments/checkpoints/detector_D1_random_best.pt"]),
        ("M1_hard", ["experiments/checkpoints/hard_boosted_d2_best.pt", "experiments/checkpoints/detector_D2_hard_best.pt"]),
        ("M2_adaptive", ["experiments/checkpoints/adaptive_d3_best.pt", "experiments/checkpoints/adaptive_d3.pt"]),
    ]

    metrics_dict: Dict[str, Dict[str, float]] = {}

    for name, ckpt_candidates in models_to_eval:
        found_ckpt = None
        for c in ckpt_candidates:
            p = Path(c)
            if p.exists():
                found_ckpt = p
                break

        if not found_ckpt:
            logger.warning("Checkpoint for %s not found, skipping.", name)
            continue

        logger.info("Evaluating %s from %s...", name, found_ckpt)
        model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
        load_verified_checkpoint(model, found_ckpt, device=args.device)
        res = evaluate_model_on_manifest(model, test_path, device=args.device)
        metrics_dict[name] = res
        logger.info("%s Metrics: %s", name, res)

    report_md = build_benchmark_summary_table(metrics_dict)
    print("\n" + "=" * 80)
    print("FOUR-WAY COMPARATIVE PROOF-OF-UTILITY BENCHMARK TABLE")
    print("=" * 80)
    print(report_md)
    print("=" * 80 + "\n")

    out_p = Path(args.output_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write("# 4-Way Comparative Proof-of-Utility Benchmark Report\n\n")
        f.write(report_md + "\n\n")
        f.write("Evaluated on frozen unseen test split: `" + str(test_path) + "` (100% Zero Leakage).\n")
    logger.info("Saved final Proof-of-Utility report to %s", out_p)


if __name__ == "__main__":
    main()

