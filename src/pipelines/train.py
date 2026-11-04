"""Training pipeline script adhering to script-only, full-state checkpointing, and exact resume rules."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.common.checkpoint import load_checkpoint, save_checkpoint
from src.common.logger import get_logger
from src.common.metrics import compute_miou
from src.common.seed import seed_everything
from src.data.occlusion_dataset import FacialOcclusionDataset
from src.models.builder import build_model
from src.models.losses import CompositeOcclusionLoss


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Facial Occlusion Segmentation Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="latest",
        default=None,
        help="Path to checkpoint .pt file to resume from, or 'latest' to find last run checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute single sanity step for verification without full training",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config safely."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def train_pipeline(
    config_path: str = "configs/config.yaml",
    resume: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the fine-grained facial occlusion segmentation training pipeline.

    Args:
        config_path: Path to master configuration YAML.
        resume: Checkpoint path or 'latest' to resume from.
        dry_run: Whether to run single sanity iteration.

    Returns:
        Summary metrics dictionary.
    """
    cfg = load_config(config_path)

    run_name = cfg.get("project", {}).get("name", "facial_occlusion_baseline")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("experiments/runs") / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "logs" / f"{run_name}.log"
    logger = get_logger("train_pipeline", log_file=log_file)
    logger.info("Initializing run directory: %s", run_dir)

    # 1. Reproducibility
    seed = cfg.get("training", {}).get("seed", 42)
    seed_everything(seed=seed)
    logger.info("Initialized global random seed: %d", seed)

    # 2. Execution Device
    device_str = cfg.get("training", {}).get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info("Target execution device: %s", device)

    # 3. Model Architecture
    num_classes = cfg.get("dataset", {}).get("num_classes", 6)
    model = build_model(architecture="occlusion_segmenter", num_classes=num_classes, in_channels=3)
    model.to(device)
    logger.info("Instantiated model: %s (num_classes=%d)", model.__class__.__name__, num_classes)

    # 4. Criterion & Optimizer
    criterion = CompositeOcclusionLoss(num_classes=num_classes, ce_weight=1.0, dice_weight=1.0)
    lr = float(cfg.get("training", {}).get("learning_rate", 6e-5))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 1 if dry_run else int(cfg.get("training", {}).get("epochs", 40))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    # 5. Checkpoint Resume
    start_epoch = 1
    best_metric = 0.0
    history: dict[str, list[float]] = {"train_loss": [], "val_miou": []}

    if resume:
        ckpt_path = Path(resume)
        if resume == "latest":
            # Search runs for last pt
            candidates = sorted(Path("experiments/runs").glob(f"*/checkpoints/*_last.pt"))
            if candidates:
                ckpt_path = candidates[-1]
            else:
                logger.warning("No previous checkpoint found to resume from.")
                ckpt_path = Path("non_existent.pt")

        if ckpt_path.exists():
            loaded = load_checkpoint(ckpt_path, model, optimizer, scheduler, device=device)
            start_epoch = loaded.get("epoch", 0) + 1
            best_metric = loaded.get("best_metric", 0.0)
            history = loaded.get("history", history)
            logger.info("Resumed from epoch %d with best metric: %.4f", start_epoch - 1, best_metric)
        else:
            logger.warning("Checkpoint path does not exist: %s, starting fresh.", ckpt_path)

    # 6. Dummy / Synthetic Data Loader for Sanity Check
    train_images = list(Path("data/synthetic_diffusion/images/train").glob("*.png"))
    train_masks = list(Path("data/synthetic_diffusion/masks/train").glob("*.png"))

    if not train_images or dry_run:
        logger.info("Using lightweight synthetic batch for sanity check / dry-run.")
        # Minimal batch with resolution 128x128 for fast CPU validation
        res = 128 if dry_run else 512
        dummy_x = torch.randn(2, 3, res, res)
        dummy_y = torch.randint(0, num_classes, (2, res, res))
        train_loader = [(dummy_x, dummy_y)]
        val_loader = [(dummy_x, dummy_y)]
    else:
        train_ds = FacialOcclusionDataset(train_images, train_masks, is_train=True)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(train_ds, batch_size=4, shuffle=False)

    ckpt_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_file = metrics_dir / f"{run_name}_history.jsonl"

    # 7. Training & Validation Loop
    logger.info("Starting training loop (Epochs %d to %d)...", start_epoch, epochs)
    val_miou = best_metric

    if start_epoch > epochs:
        logger.warning(
            "Resumed epoch (%d) already exceeds target epochs (%d). Nothing to train.",
            start_epoch,
            epochs,
        )
        return {"final_val_miou": best_metric, "best_val_miou": best_metric, "run_dir": str(run_dir)}

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            if isinstance(batch, dict):
                imgs, masks = batch["image"].to(device), batch["mask"].to(device)
            else:
                imgs, masks = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss_dict = criterion(logits, masks)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if dry_run:
                break

        scheduler.step()
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        # Validation step
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    imgs, masks = batch["image"].to(device), batch["mask"].to(device)
                else:
                    imgs, masks = batch[0].to(device), batch[1].to(device)

                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
                if dry_run:
                    break

        cat_preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((1, 512, 512))
        cat_targs = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((1, 512, 512))
        val_metrics = compute_miou(cat_preds, cat_targs, num_classes=num_classes)
        val_miou = val_metrics["miou"]

        history["train_loss"].append(avg_train_loss)
        history["val_miou"].append(val_miou)

        is_best = val_miou > best_metric
        if is_best:
            best_metric = val_miou

        logger.info(
            "Epoch [%d/%d] - Train Loss: %.4f - Val mIoU: %.4f (Best: %.4f)",
            epoch,
            epochs,
            avg_train_loss,
            val_miou,
            best_metric,
        )

        # Full-state checkpointing
        state = {
            "epoch": epoch,
            "global_step": epoch * len(train_loader),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "history": history,
            "config": cfg,
        }
        save_checkpoint(state, is_best=is_best, checkpoint_dir=ckpt_dir, run_name=run_name)

        # Append epoch history to jsonl
        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_miou": val_miou,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        }
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record) + "\n")

        if dry_run:
            logger.info("Dry-run execution completed successfully.")
            break

    return {"final_val_miou": val_miou, "best_val_miou": best_metric, "run_dir": str(run_dir)}


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(config_path=args.config, resume=args.resume, dry_run=args.dry_run)
