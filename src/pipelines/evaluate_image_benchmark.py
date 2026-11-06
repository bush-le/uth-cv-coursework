"""Benchmark evaluation pipeline for 2D deepfake image detector."""

from typing import Any, Dict, Iterable, Union
import numpy as np
import torch
import torch.nn as nn

from src.common.logger import get_logger

logger = get_logger(__name__)


def compute_roc_auc_vectorized(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC via Wilcoxon-Mann-Whitney rank sum test."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    if n_pos == 0 or n_neg == 0:
        return 1.0

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_score)) + 1.0

    rank_sum_pos = np.sum(ranks[pos_mask])
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def evaluate_image_detector(
    model: nn.Module,
    dataloader: Iterable[Dict[str, Any]],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, float]:
    """Evaluate deepfake image detector over a dataloader.
    
    Args:
        model: DeepfakeImageDetector instance.
        dataloader: Iterable yielding batches with 'images' and 'labels'.
        device: Device to run evaluation on.
        
    Returns:
        Dictionary containing 'accuracy', 'roc_auc', 'f1_score', 'precision', 'recall', 'loss'.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds: list[int] = []
    all_probs: list[float] = []
    all_targets: list[int] = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(dev)
            labels = batch["labels"].to(dev).float().view(-1, 1)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            preds = (probs >= 0.5).long()

            total_loss += float(loss.item()) * images.size(0)
            total_samples += images.size(0)

            all_probs.extend(probs.view(-1).cpu().tolist())
            all_preds.extend(preds.view(-1).cpu().tolist())
            all_targets.extend(labels.view(-1).long().cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(np.mean(y_pred == y_true)) if len(y_true) > 0 else 0.0
    auc = compute_roc_auc_vectorized(y_true, y_prob)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_loss = total_loss / max(1, total_samples)

    metrics = {
        "accuracy": acc,
        "roc_auc": auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "loss": avg_loss,
    }
    logger.info(f"Image benchmark metrics: {metrics}")
    return metrics
