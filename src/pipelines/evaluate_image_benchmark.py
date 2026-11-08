"""Benchmark evaluation pipeline for 2D deepfake image detector (dinoViT baseline vs fine-tuned)."""

from typing import Any, Dict, Iterable, List, Union
import numpy as np
import torch
import torch.nn as nn

from src.common.logger import get_logger
from src.common.metrics import (
    compute_bypass_rate,
    compute_hard_sample_recall,
    compute_roc_auc,
)

logger = get_logger(__name__)


def compute_roc_auc_vectorized(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC via Wilcoxon-Mann-Whitney rank sum test."""
    return compute_roc_auc(y_true, y_score)


def evaluate_image_detector(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: Union[str, torch.device] = "cpu",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate deepfake image detector (dinoViT) over a dataloader.

    Computes overall detection metrics (Accuracy, ROC-AUC, F1, Precision, Recall)
    as well as adversarial robustness metrics (Bypass Rate on fake images and
    Hard-Sample Recall on previously bypassed samples).

    Args:
        model: DeepfakeImageDetector instance.
        dataloader: Iterable yielding batches with 'images', 'labels', and optional 'is_hard_sample'.
        device: Device to run evaluation on.
        threshold: Classification decision threshold tau (default 0.5).

    Returns:
        Dictionary containing 'accuracy', 'roc_auc', 'f1_score', 'precision', 'recall',
        'bypass_rate', 'hard_sample_recall', and 'loss'.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds: List[int] = []
    all_probs: List[float] = []
    all_targets: List[int] = []
    all_hard_flags: List[bool] = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(dev)
            labels = batch["labels"].to(dev).float().view(-1, 1)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            preds = (probs >= threshold).long()

            total_loss += float(loss.item()) * images.size(0)
            total_samples += images.size(0)

            all_probs.extend(probs.view(-1).cpu().tolist())
            all_preds.extend(preds.view(-1).cpu().tolist())
            all_targets.extend(labels.view(-1).long().cpu().tolist())

            if "is_hard_sample" in batch:
                hard_tensor = batch["is_hard_sample"]
                if isinstance(hard_tensor, torch.Tensor):
                    all_hard_flags.extend(hard_tensor.view(-1).bool().cpu().tolist())
                else:
                    all_hard_flags.extend([bool(x) for x in hard_tensor])
            else:
                all_hard_flags.extend([False] * images.size(0))

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_hard = np.array(all_hard_flags, dtype=bool)

    acc = float(np.mean(y_pred == y_true)) if len(y_true) > 0 else 0.0
    auc = compute_roc_auc_vectorized(y_true, y_prob)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    avg_loss = total_loss / max(1, total_samples)

    fake_probs = y_prob[y_true == 1] if np.any(y_true == 1) else np.array([], dtype=np.float32)
    bypass_rate = compute_bypass_rate(fake_probs, threshold=threshold)
    hard_recall = compute_hard_sample_recall(y_pred, y_true, y_hard)

    metrics = {
        "accuracy": acc,
        "roc_auc": auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "bypass_rate": bypass_rate,
        "hard_sample_recall": hard_recall,
        "loss": avg_loss,
    }
    logger.info("Image benchmark metrics: %s", metrics)
    return metrics


def compare_baseline_vs_finetuned(
    baseline_model: nn.Module,
    finetuned_model: nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: Union[str, torch.device] = "cpu",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compare Baseline dinoViT against Fine-Tuned dinoViT (trained on Baseline + Syn + Hard Samples).

    Quantifies the exact utility of the synthetic dataset and hard-sample mining loop.
    """
    batches = list(dataloader)
    base_metrics = evaluate_image_detector(
        baseline_model, batches, device=device, threshold=threshold
    )
    ft_metrics = evaluate_image_detector(
        finetuned_model, batches, device=device, threshold=threshold
    )

    comparison = {
        "baseline": base_metrics,
        "finetuned": ft_metrics,
        "deltas": {
            "roc_auc_gain": ft_metrics["roc_auc"] - base_metrics["roc_auc"],
            "accuracy_gain": ft_metrics["accuracy"] - base_metrics["accuracy"],
            "f1_gain": ft_metrics["f1_score"] - base_metrics["f1_score"],
            "hard_sample_recall_gain": (
                ft_metrics["hard_sample_recall"] - base_metrics["hard_sample_recall"]
            ),
            "bypass_rate_reduction": base_metrics["bypass_rate"] - ft_metrics["bypass_rate"],
        },
    }
    logger.info("Baseline vs Fine-Tuned comparison deltas: %s", comparison["deltas"])
    return comparison
