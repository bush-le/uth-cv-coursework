"""Phase 3 Pipeline: Fine-tuning dinoViT detector on Baseline Data + Synthetic DF + Mined Hard Samples."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
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

logger = get_logger(__name__)


def combine_training_manifests(
    baseline_manifest: Optional[Path] = None,
    synthetic_manifest: Optional[Path] = None,
    hard_sample_manifest: Optional[Path] = None,
) -> List[ImageMetadata]:
    """Assemble combined metadata list from Baseline Data + Synthetic DF Data + Hard Samples.

    Ensures deduplication by image_id while prioritizing Hard Sample metadata flags
    when a sample exists in both synthetic_manifest and hard_sample_manifest.
    """
    records_by_id: Dict[str, ImageMetadata] = {}

    for path, default_source in [
        (baseline_manifest, "baseline"),
        (synthetic_manifest, "synthetic"),
        (hard_sample_manifest, "hard_sample"),
    ]:
        if path is not None and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "data_source" not in data:
                        data["data_source"] = default_source
                    if default_source == "hard_sample":
                        data["is_hard_sample"] = True
                    meta = ImageMetadata.from_dict(data)
                    records_by_id[meta.image_id] = meta

    combined = list(records_by_id.values())
    logger.info("Combined training manifests: %d unique samples loaded.", len(combined))
    return combined


def train_deepfake_image_detector(
    model: DeepfakeImageDetector,
    datamodule: DeepfakeImageDataModule,
    num_epochs: int = 5,
    learning_rate: float = 2e-4,
    hard_sample_loss_weight: float = 2.0,
    unfreeze_last_n_blocks: Optional[int] = 2,
    checkpoint_dir: Path = Path("experiments/checkpoints"),
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, float]:
    """Fine-tune dinoViT deepfake image detector with hard-sample weighting and checkpointing.

    Args:
        model: DeepfakeImageDetector (e.g. backbone='dino_vit').
        datamodule: DeepfakeImageDataModule configured with baseline + synthetic + hard samples.
        num_epochs: Number of fine-tuning epochs.
        learning_rate: Learning rate for AdamW.
        hard_sample_loss_weight: Multiplicative loss weight > 1.0 applied to mined hard samples
                                 so the fine-tuned model focuses on previously bypassed fakes.
        unfreeze_last_n_blocks: Number of top DINO ViT blocks to unfreeze during fine-tuning.
        checkpoint_dir: Directory to save best_model.pt and last_model.pt.
        device: Target compute device ('cpu' or 'cuda').
        seed: Global random seed.

    Returns:
        Dictionary of validation metrics from the final epoch.
    """
    seed_everything(seed)
    dev = torch.device(device)
    model.unfreeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n_blocks)
    model.to(dev)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)
    criterion_raw = nn.BCEWithLogitsLoss(reduction="none")

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    best_auc = 0.0
    latest_metrics: Dict[str, float] = {}

    logger.info(
        "Starting dinoViT fine-tuning on %s for %d epochs (hard_sample_weight=%.2f)...",
        device,
        num_epochs,
        hard_sample_loss_weight,
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            optimizer.zero_grad()
            images = batch["images"].to(dev)
            labels = batch["labels"].to(dev).float().view(-1, 1)
            is_hard = batch.get("is_hard_sample")

            logits = model(images)
            per_sample_loss = criterion_raw(logits, labels)

            if is_hard is not None:
                hard_mask = is_hard.to(dev).float().view(-1, 1)
                weights = 1.0 + (hard_sample_loss_weight - 1.0) * hard_mask
                loss = torch.mean(per_sample_loss * weights)
            else:
                loss = torch.mean(per_sample_loss)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item()) * images.size(0)
            train_samples += images.size(0)

        avg_train_loss = epoch_loss / max(1, train_samples)

        # Validation step
        val_metrics = evaluate_image_detector(model, val_loader, device=device)
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | "
            "Val AUC: %.4f | Hard Recall: %.4f | Bypass Rate: %.4f",
            epoch,
            num_epochs,
            avg_train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["roc_auc"],
            val_metrics["hard_sample_recall"],
            val_metrics["bypass_rate"],
        )
        latest_metrics = val_metrics

        state = {
            "epoch": epoch,
            "global_step": epoch * max(1, len(train_loader)),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": max(best_auc, val_metrics["roc_auc"]),
            "metrics": val_metrics,
        }
        is_best = val_metrics["roc_auc"] >= best_auc
        if is_best:
            best_auc = val_metrics["roc_auc"]
        save_checkpoint(
            state=state,
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            run_name="dinovit_detector",
        )
        # Also maintain standard best_model.pt and last_model.pt aliases
        torch.save(state, checkpoint_dir / "last_model.pt")
        if is_best:
            torch.save(state, checkpoint_dir / "best_model.pt")
            logger.info("New best dinoViT model saved with Val AUC: %.4f", best_auc)

    return latest_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune dinoViT detector on Baseline + Synthetic DF + Hard Samples."
    )
    parser.add_argument("--baseline-manifest", type=str, default="data/raw/manifest.jsonl")
    parser.add_argument(
        "--synthetic-manifest", type=str, default="data/processed/curated_manifest.jsonl"
    )
    parser.add_argument(
        "--hard-manifest", type=str, default="data/processed/hard_samples_manifest.jsonl"
    )
    parser.add_argument("--backbone", type=str, default="dino_vit")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hard-weight", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta_list = combine_training_manifests(
        baseline_manifest=Path(args.baseline_manifest),
        synthetic_manifest=Path(args.synthetic_manifest),
        hard_sample_manifest=Path(args.hard_manifest),
    )
    if not meta_list:
        meta_list = [
            ImageMetadata(
                image_id=f"sample_{i}",
                label=i % 2,
                manipulation_type="real" if i % 2 == 0 else "sd_inpaint",
                is_hard_sample=(i % 4 == 1),
                data_source="hard_sample" if (i % 4 == 1) else "baseline",
            )
            for i in range(16)
        ]

    dm = DeepfakeImageDataModule(
        metadata_list=meta_list,
        batch_size=4,
        hard_sample_oversample_factor=2,
        seed=args.seed,
    )
    model = DeepfakeImageDetector(backbone=args.backbone, num_classes=1)
    train_deepfake_image_detector(
        model=model,
        datamodule=dm,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hard_sample_loss_weight=args.hard_weight,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
