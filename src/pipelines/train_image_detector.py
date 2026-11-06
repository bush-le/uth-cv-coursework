"""Resumable training pipeline for 2D deepfake image classifier."""

import argparse
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.common.checkpoint import save_checkpoint
from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_datamodule import DeepfakeImageDataModule
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector

logger = get_logger(__name__)


def train_deepfake_image_detector(
    model: DeepfakeImageDetector,
    datamodule: DeepfakeImageDataModule,
    num_epochs: int = 5,
    learning_rate: float = 2e-4,
    checkpoint_dir: Path = Path("experiments/checkpoints"),
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, float]:
    """Train deepfake image detector with validation tracking and checkpointing."""
    seed_everything(seed)
    dev = torch.device(device)
    model.to(dev)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    best_auc = 0.0
    latest_metrics: Dict[str, float] = {}

    logger.info(f"Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()
            images = batch["images"].to(dev)
            labels = batch["labels"].to(dev).float().view(-1, 1)

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item()) * images.size(0)
            train_samples += images.size(0)

        avg_train_loss = epoch_loss / max(1, train_samples)

        # Validation step
        val_metrics = evaluate_image_detector(model, val_loader, device=device)
        logger.info(
            f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f}"
        )
        latest_metrics = val_metrics

        # Save last checkpoint
        save_checkpoint(
            checkpoint_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"val_loss": val_metrics["loss"], "val_auc": val_metrics["roc_auc"]},
        )

        # Save best checkpoint
        if val_metrics["roc_auc"] >= best_auc:
            best_auc = val_metrics["roc_auc"]
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_metrics["loss"], "val_auc": val_metrics["roc_auc"]},
            )
            logger.info(f"New best model saved with Val AUC: {best_auc:.4f}")

    return latest_metrics
