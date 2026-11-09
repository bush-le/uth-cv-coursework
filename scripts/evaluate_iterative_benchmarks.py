"""Evaluates M0, M_rand, M1, and M2 across identical frozen unseen test benchmark.

Generates the Golden Coursework Comparison Table for the 4-way Proof-of-Utility ablation.
"""

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


def build_benchmark_summary_table(metrics: Dict[str, Dict[str, float]]) -> str:
    """Build standardized Markdown summary table comparing models on identical test set.

    Args:
        metrics: Dictionary mapping model names to performance metrics.

    Returns:
        Formatted GitHub-flavored Markdown table.
    """
    lines = [
        "| Model | Training Dataset | Test Accuracy | Fake Recall (↑) | Bypass Rate (FNR ↓) | ROC-AUC (↑) | Real FPR (↓) |",
        "|---|---|---|---|---|---|---|",
    ]
    meta_desc = {
        "M0_baseline": "D0 (Base Real + Fake)",
        "M_rand": "D1 (D0 + Random Fakes)",
        "M1_hard": "D2 (D0 + H1 Bypass)",
        "M2_adaptive": "D3 (D0 + H1 + H2 Feedback)",
    }
    for m_key, m_data in metrics.items():
        desc = meta_desc.get(m_key, m_key)
        acc = m_data.get("accuracy", 0.0) * 100.0
        rec = m_data.get("fake_recall", 0.0) * 100.0
        bypr = m_data.get("bypass_rate", 0.0) * 100.0
        auc = m_data.get("auc", 0.0)
        fpr = m_data.get("fpr", 0.0) * 100.0
        lines.append(
            f"| **{m_key}** | {desc} | {acc:.1f}% | **{rec:.1f}%** | {bypr:.1f}% | **{auc:.3f}** | {fpr:.1f}% |"
        )
    return "\n".join(lines)


def evaluate_model_on_manifest(
    model: DeepfakeImageDetector,
    manifest_path: Path,
    device: str = "cpu",
    batch_size: int = 16,
) -> Dict[str, float]:
    """Evaluate detector on a test manifest and compute core metrics."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        metas = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

    dataset = DeepfakeImageDataset(metas, image_size=(256, 256), augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_image_samples)

    all_preds = []
    all_probs = []
    all_targets = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    fake_mask = y_true == 1
    real_mask = y_true == 0

    fake_recall = compute_recall(y_pred, y_true, positive_label=1)
    acc = compute_accuracy(y_pred, y_true)
    auc = compute_roc_auc(y_true, y_prob)

    fake_probs = y_prob[fake_mask] if np.any(fake_mask) else np.array([])
    bypass_rate = compute_bypass_rate(fake_probs, threshold=0.5)

    real_preds = y_pred[real_mask] if np.any(real_mask) else np.array([])
    fpr = float(np.mean(real_preds == 1)) if len(real_preds) > 0 else 0.0

    return {
        "accuracy": acc,
        "fake_recall": fake_recall,
        "bypass_rate": bypass_rate,
        "auc": auc,
        "fpr": fpr,
    }
