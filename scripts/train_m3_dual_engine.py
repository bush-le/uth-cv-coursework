"""Champion Model M3: Dual-Engine Cross-Generator Adaptation.

Warm-Starts from M2* (active_learning_m2_best.pt) and fine-tunes on clean D3 partition:
- 2,813 Real images (train_reals.jsonl)
- 2,813 Fake images (629 D2 hard samples + 800 SD 1.5 baseline + 1,384 clean RealisticVision v5.1)
- Zero Data Leakage: 626 test-derived RV5.1 images strictly excluded and reserved for OOD evaluation.

Architectural Guarantees:
- Unfreezes Block 11 + Classifier Head (1.78M params, 8.06% trainable).
- Blocks 0..10 and Embeddings strictly frozen (20.01M params, 91.94%).
- Differential Learning Rate: Head lr=3e-5, Block 11 lr=3e-6.
- Max epochs = 6 with CosineAnnealingLR & Early Stopping.
- Quad-Vault Evaluation on Master Test, Challenge Test, and Unseen RV5.1 Test.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.common.checkpoint import save_checkpoint
from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector

logger = get_logger("train_m3_dual_engine")


def load_metadata_manifest(path: Path) -> List[ImageMetadata]:
    """Load JSONL manifest into list of ImageMetadata."""
    with open(path, "r", encoding="utf-8") as f:
        return [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M3 Champion Dual-Engine Model")
    parser.add_argument("--epochs", type=int, default=6, help="Maximum epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Classifier learning rate")
    parser.add_argument("--backbone-lr", type=float, default=3e-6, help="Block 11 learning rate")
    parser.add_argument("--hard-loss-weight", type=float, default=1.10, help="Weight for hard samples")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry run with 2 batches")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info("=== BẮT ĐẦU HUẤN LUYỆN CHAMPION MODEL M3 DUAL-ENGINE ===")
    logger.info("Device: %s | Max Epochs: %d | Batch Size: %d", device, args.epochs, args.batch_size)

    # 1. Paths
    d3_train_path = Path("data/benchmarks/D3_clean/train.jsonl")
    d3_val_path = Path("data/benchmarks/D3_clean/val.jsonl")
    master_test_path = Path("data/benchmarks/master_frozen_test.jsonl")
    challenge_test_path = Path("data/active_learning/cumulative_challenge_test.jsonl")
    unseen_rv51_path = Path("data/benchmarks/D3_clean/unseen_rv51_test.jsonl")
    m2_ckpt_path = Path("experiments/checkpoints/active_learning_m2_best.pt")
    ckpt_dir = Path("experiments/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Manifests
    train_recs = load_metadata_manifest(d3_train_path)
    val_recs = load_metadata_manifest(d3_val_path)
    master_recs = load_metadata_manifest(master_test_path)
    chal_recs = load_metadata_manifest(challenge_test_path)
    unseen_recs = load_metadata_manifest(unseen_rv51_path)

    if args.dry_run:
        logger.info("DRY-RUN mode enabled: truncating datasets...")
        train_recs = train_recs[:32]
        val_recs = val_recs[:32]
        master_recs = master_recs[:32]
        chal_recs = chal_recs[:32]
        unseen_recs = unseen_recs[:32]

    logger.info(
        "Manifest counts: Train=%d, Val=%d, MasterTest=%d, ChallengeTest=%d, UnseenRV51Test=%d",
        len(train_recs),
        len(val_recs),
        len(master_recs),
        len(chal_recs),
        len(unseen_recs),
    )

    # 3. Setup DataModules
    dm = DeepfakeImageDataModule(
        train_metadata=train_recs,
        val_metadata=val_recs,
        test_metadata=master_recs,
        batch_size=args.batch_size,
        hard_sample_oversample_factor=1,
        seed=args.seed,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    master_loader = dm.test_dataloader()

    chal_dm = DeepfakeImageDataModule(
        train_metadata=[], val_metadata=[], test_metadata=chal_recs, batch_size=args.batch_size, seed=args.seed
    )
    chal_dm.setup()
    chal_loader = chal_dm.test_dataloader()

    unseen_dm = DeepfakeImageDataModule(
        train_metadata=[], val_metadata=[], test_metadata=unseen_recs, batch_size=args.batch_size, seed=args.seed
    )
    unseen_dm.setup()
    unseen_loader = unseen_dm.test_dataloader()

    # 4. Initialize Model with Warm-Start from M2*
    if not m2_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint M2* not found at {m2_ckpt_path}!")

    logger.info("Initializing DeepfakeImageDetector with Warm-Start from M2*: %s", m2_ckpt_path)
    model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
    m2_state = torch.load(m2_ckpt_path, map_location="cpu")
    if "model_state_dict" in m2_state:
        model.load_state_dict(m2_state["model_state_dict"])
    else:
        model.load_state_dict(m2_state)

    # Unfreeze only the last Transformer block (Block 11) + Classifier Head
    model.unfreeze_backbone(unfreeze_last_n_blocks=1)
    model.to(dev)

    # Audit trainable parameters
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [
        p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier")
    ]
    num_trainable = sum(p.numel() for p in classifier_params + backbone_params)
    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())

    logger.info(
        "Param Audit: Total=%d | Trainable=%d (%.2f%%) | Frozen=%d (%.2f%%)",
        num_total,
        num_trainable,
        (num_trainable / num_total) * 100,
        num_frozen,
        (num_frozen / num_total) * 100,
    )
    assert num_trainable == 1777153, f"Unexpected trainable params: {num_trainable}"

    # 5. Optimizer & Scheduler
    param_groups = [
        {"params": classifier_params, "lr": args.lr},
        {"params": backbone_params, "lr": args.backbone_lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.backbone_lr * 0.2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # 6. Training Loop with Early Stopping
    best_val_auc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve_epochs = 0
    run_name = "active_learning_m3"
    best_ckpt_path = ckpt_dir / f"{run_name}_best.pt"

    start_train_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_samples = 0

        for step, batch in enumerate(train_loader):
            images = batch["images"].to(dev, non_blocking=True)
            labels = batch["labels"].to(dev, non_blocking=True).float().view(-1, 1)

            weights = torch.ones_like(labels)
            if "is_hard_sample" in batch:
                is_hard = batch["is_hard_sample"].to(dev).bool().view(-1, 1)
                weights = torch.where(is_hard, torch.full_like(weights, args.hard_loss_weight), weights)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(images)
                per_sample_loss = criterion(logits, labels)
                loss = (per_sample_loss * weights).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)

        scheduler.step()
        train_loss /= max(train_samples, 1)
        epoch_duration = time.time() - epoch_start

        # Validation evaluation
        val_metrics = evaluate_image_detector(model, val_loader, device=device)
        val_auc = val_metrics.get("roc_auc", 0.0)
        val_loss = val_metrics.get("loss", float("inf"))
        val_acc = val_metrics.get("accuracy", 0.0)

        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f | Val Loss: %.4f | Val AUC: %.4f | Val Acc: %.4f | LR: (head=%.2e, dino=%.2e)",
            epoch,
            args.epochs,
            epoch_duration,
            train_loss,
            val_loss,
            val_auc,
            val_acc,
            optimizer.param_groups[0]["lr"],
            optimizer.param_groups[1]["lr"],
        )

        # Checkpoint Saving & Early Stopping Check
        is_best = False
        if val_auc > best_val_auc or (abs(val_auc - best_val_auc) < 1e-4 and val_loss < best_val_loss):
            is_best = True
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_epochs = 0
            logger.info("  --> NEW BEST METRIC at epoch %d (Val AUC: %.4f)", epoch, val_auc)
        else:
            no_improve_epochs += 1

        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "best_metric": best_val_auc,
            "metrics": val_metrics,
            "config": vars(args),
            "warm_started_from": str(m2_ckpt_path),
        }
        save_checkpoint(
            state=checkpoint_state,
            is_best=is_best,
            checkpoint_dir=ckpt_dir,
            run_name=run_name,
        )

        if no_improve_epochs >= args.patience and epoch >= 3:
            logger.info("Early stopping triggered at epoch %d (no improvement for %d epochs).", epoch, no_improve_epochs)
            break

    total_duration = time.time() - start_train_time
    logger.info("=== HUẤN LUYỆN HOÀN TẤT TRONG %.1f GIÂY | BEST EPOCH: %d (AUC: %.4f) ===", total_duration, best_epoch, best_val_auc)

    # 7. Comprehensive Post-Training Evaluation on Best Model
    logger.info("=== ĐÁNH GIÁ ĐA KHO TRÊN BEST CHECKPOINT (%s) ===", best_ckpt_path)
    best_state = torch.load(best_ckpt_path, map_location=device)
    if "model_state_dict" in best_state:
        model.load_state_dict(best_state["model_state_dict"])

    m_master_res = evaluate_image_detector(model, master_loader, device=device)
    m_chal_res = evaluate_image_detector(model, chal_loader, device=device)
    m_unseen_res = evaluate_image_detector(model, unseen_loader, device=device)

    logger.info("=================================================================")
    logger.info("1. Master Frozen Test (SD 1.5):")
    logger.info("   -> Acc: %.4f | AUC: %.4f | Recall: %.4f | Precision: %.4f",
                m_master_res["accuracy"], m_master_res["roc_auc"], m_master_res["recall"], m_master_res["precision"])
    logger.info("2. Cumulative Challenge Test (Hard Samples):")
    logger.info("   -> Acc: %.4f | AUC: %.4f | Recall: %.4f | Precision: %.4f",
                m_chal_res["accuracy"], m_chal_res["roc_auc"], m_chal_res["recall"], m_chal_res["precision"])
    logger.info("3. Unseen RV5.1 Test Vault (OOD Generalization):")
    logger.info("   -> Acc: %.4f | AUC: %.4f | Recall: %.4f | Precision: %.4f",
                m_unseen_res["accuracy"], m_unseen_res["roc_auc"], m_unseen_res["recall"], m_unseen_res["precision"])
    logger.info("=================================================================")

    # Save summary report json
    summary_path = Path("experiments/benchmarks/tables/m3_training_evaluation_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "total_train_seconds": total_duration,
                "master_frozen_test": m_master_res,
                "cumulative_challenge_test": m_chal_res,
                "unseen_rv51_test": m_unseen_res,
            },
            f,
            indent=2,
        )
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
