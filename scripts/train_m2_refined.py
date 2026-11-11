"""Refined Round 2 Active Learning: Warm-Start from M1 with Balanced Regularization.

Resolves the Catastrophic Interference / Negative Transfer flaw where M2 was colder-started
from ImageNet and over-weighted on hard samples, degrading general test recall.

Key Architectural Adjustments:
1. Warm-Start: Initializes from active_learning_m1_best.pt instead of ImageNet.
2. Moderate Hard Weight: hard_sample_loss_weight = 1.15 (down from 2.0).
3. Low Learning Rate: lr = 5e-5 for classifier, 5e-6 for unfreezed DINOv2 block.
4. Smooth Epochs: 3 epochs with AMP.
"""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.common.checkpoint import save_checkpoint
from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector

logger = get_logger("train_m2_refined")


def load_metadata_manifest(path: Path) -> List[ImageMetadata]:
    """Load JSONL manifest into list of ImageMetadata."""
    with open(path, "r", encoding="utf-8") as f:
        return [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]


def main():
    seed = 42
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info("Starting M2 Refined Training on device: %s", device)

    # 1. Load Datasets
    train_manifest = Path("data/benchmarks/D2_hard_boosted/train.jsonl")
    val_manifest = Path("data/benchmarks/D2_hard_boosted/val.jsonl")
    master_test_manifest = Path("data/benchmarks/master_frozen_test.jsonl")
    challenge_manifest = Path("data/active_learning/cumulative_challenge_test.jsonl")

    train_recs = load_metadata_manifest(train_manifest)
    val_recs = load_metadata_manifest(val_manifest)
    master_recs = load_metadata_manifest(master_test_manifest)
    chal_recs = load_metadata_manifest(challenge_manifest)

    logger.info(
        "Loaded: Train=%d, Val=%d, MasterTest=%d, ChallengeTest=%d",
        len(train_recs),
        len(val_recs),
        len(master_recs),
        len(chal_recs),
    )

    # 2. Setup DataModule
    batch_size = 16
    dm = DeepfakeImageDataModule(
        train_metadata=train_recs,
        val_metadata=val_recs,
        test_metadata=master_recs,
        batch_size=batch_size,
        hard_sample_oversample_factor=1,
        seed=seed,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    master_loader = dm.test_dataloader()

    chal_dm = DeepfakeImageDataModule(
        train_metadata=[],
        val_metadata=[],
        test_metadata=chal_recs,
        batch_size=batch_size,
        seed=seed,
    )
    chal_dm.setup()
    chal_loader = chal_dm.test_dataloader()

    # 3. Model Initialization: WARM-START FROM M1
    m1_ckpt_path = Path("experiments/checkpoints/active_learning_m1_best.pt")
    if not m1_ckpt_path.exists():
        raise FileNotFoundError(f"M1 checkpoint not found at {m1_ckpt_path}!")

    logger.info("Initializing DeepfakeImageDetector with Warm-Start from M1: %s", m1_ckpt_path)
    model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
    m1_state = torch.load(m1_ckpt_path, map_location="cpu")
    if "model_state_dict" in m1_state:
        model.load_state_dict(m1_state["model_state_dict"])
    else:
        model.load_state_dict(m1_state)

    # Unfreeze only the last block for delicate adaptation
    unfreeze_blocks = 1
    model.unfreeze_backbone(unfreeze_last_n_blocks=unfreeze_blocks)
    model.to(dev)

    # 4. Hyperparameters: Conservative Fine-Tuning
    learning_rate = 5e-5
    hard_sample_loss_weight = 1.15
    num_epochs = 2

    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [
        p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier")
    ]
    param_groups = [
        {"params": classifier_params, "lr": learning_rate},
    ]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": min(learning_rate * 0.1, 5e-6)})

    optimizer = AdamW(param_groups, weight_decay=1e-4)
    trainable_params = classifier_params + backbone_params
    criterion_raw = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    # Initial Pre-Fine-Tuning Evaluation
    logger.info("--- Initial M1 Evaluation before Fine-Tuning ---")
    m1_val = evaluate_image_detector(model, val_loader, device=device)
    m1_mast = evaluate_image_detector(model, master_loader, device=device)
    m1_chal = evaluate_image_detector(model, chal_loader, device=device)
    logger.info("M1 Baseline Val: AUC=%.4f, Acc=%.4f, Rec=%.4f", m1_val["roc_auc"], m1_val["accuracy"], m1_val["recall"])
    logger.info("M1 Master Test : AUC=%.4f, Acc=%.4f, Rec=%.4f, Bypass=%.4f", m1_mast["roc_auc"], m1_mast["accuracy"], m1_mast["recall"], m1_mast["bypass_rate"])
    logger.info("M1 Challenge   : AUC=%.4f, Acc=%.4f, Rec=%.4f, Bypass=%.4f", m1_chal["roc_auc"], m1_chal["accuracy"], m1_chal["recall"], m1_chal["bypass_rate"])

    best_score = -1.0
    ckpt_dir = Path("experiments/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 5. Fine-Tuning Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()
            images = batch["images"].to(dev)
            labels = batch["labels"].to(dev).float().view(-1, 1)
            is_hard = batch.get("is_hard_sample")

            with torch.amp.autocast(
                device_type=dev.type,
                dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
                enabled=(dev.type == "cuda"),
            ):
                logits = model(images)
                per_sample_loss = criterion_raw(logits, labels)

                if is_hard is not None:
                    hard_mask = is_hard.to(dev).float().view(-1, 1)
                    weights = 1.0 + (hard_sample_loss_weight - 1.0) * hard_mask
                    loss = torch.sum(per_sample_loss * weights) / torch.clamp(
                        torch.sum(weights), min=1e-8
                    )
                else:
                    loss = torch.mean(per_sample_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item()) * images.size(0)
            train_samples += images.size(0)

        avg_train_loss = epoch_loss / max(1, train_samples)

        # Evaluate on validation and test vaults
        val_m = evaluate_image_detector(model, val_loader, device=device)
        mast_m = evaluate_image_detector(model, master_loader, device=device)
        chal_m = evaluate_image_detector(model, chal_loader, device=device)

        logger.info(
            "Epoch %d/%d (Train Loss: %.4f) | Val AUC: %.4f | Master Acc: %.4f, Rec: %.4f, Bypass: %.4f | Chal Acc: %.4f, Rec: %.4f",
            epoch,
            num_epochs,
            avg_train_loss,
            val_m["roc_auc"],
            mast_m["accuracy"],
            mast_m["recall"],
            mast_m["bypass_rate"],
            chal_m["accuracy"],
            chal_m["recall"],
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {
                "val": val_m,
                "master_frozen_test": mast_m,
                "cumulative_challenge_test": chal_m,
            },
        }

        # Save individual epoch checkpoint
        torch.save(state, ckpt_dir / f"active_learning_m2_epoch_{epoch}.pt")

        # Save best checkpoint strictly based on Pareto score
        score = mast_m["recall"] + chal_m["recall"]
        if score > best_score:
            best_score = score
            torch.save(state, ckpt_dir / "active_learning_m2_best.pt")
            logger.info("Saved new active_learning_m2_best.pt at epoch %d with score %.4f!", epoch, score)

    logger.info("Refined M2 Training Complete!")


if __name__ == "__main__":
    main()
