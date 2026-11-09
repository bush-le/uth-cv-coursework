"""Phase 2 Pipeline: Baseline dinoViT Benchmark Evaluation & Bypass Hard-Sample Mining.

Passes Stable Diffusion synthetic Deepfake images (DF img) through the baseline
dinoViT detector:
- Rejected (caught): P(fake) >= threshold -> detected by baseline.
- Bypassed (fooled): P(fake) < threshold -> misclassified as Real -> mined as a Hard Sample!
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from src.common.logger import get_logger
from src.common.metrics import compute_bypass_rate
from src.common.seed import seed_everything
from src.data.image_datamodule import collate_image_samples
from src.data.image_dataset import DeepfakeImageDataset
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector, MetaDinoV2Wrapper
from torch.utils.data import DataLoader

logger = get_logger(__name__)


@dataclass
class HardSampleMiningReport:
    """Summary report from running synthetic DF images through baseline dinoViT."""

    total_synthetic: int
    rejected_count: int
    bypassed_hard_count: int
    bypass_rate: float
    mean_fake_prob: float
    threshold: float
    hard_sample_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _save_mining_results(
    updated_meta: List[ImageMetadata],
    hard_meta: List[ImageMetadata],
    mean_score: float,
    annotated_manifest_path: Optional[Path],
    hard_manifest_path: Optional[Path],
) -> None:
    """Save updated manifests, copy hard sample images, and export demographic audit report."""
    if annotated_manifest_path is not None:
        annotated_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(annotated_manifest_path, "w", encoding="utf-8") as f:
            for m in updated_meta:
                f.write(json.dumps(m.to_dict()) + "\n")

    if hard_manifest_path is not None:
        hard_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hard_manifest_path, "w", encoding="utf-8") as f:
            for m in hard_meta:
                f.write(json.dumps(m.to_dict()) + "\n")

        # Copy hard sample images and save audit
        hard_img_dir = hard_manifest_path.parent / "images"
        audit_dir = hard_manifest_path.parent / "audit_reports"
        hard_img_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        saved_names = set()
        for m in hard_meta:
            if m.file_path and Path(m.file_path).exists():
                dest_path = hard_img_dir / Path(m.file_path).name
                saved_names.add(dest_path.name)
                if Path(m.file_path).resolve() != dest_path.resolve():
                    shutil.copy2(m.file_path, dest_path)

        # Remove any legacy or obsolete images not in the current hard sample set
        for p in hard_img_dir.glob("*.png"):
            if p.name not in saved_names:
                try:
                    p.unlink()
                except OSError:
                    pass

        age_dist: Dict[str, int] = {}
        gender_dist: Dict[str, int] = {}
        for m in hard_meta:
            demo = m.demographics or {}
            a = demo.get("age_group", "unknown")
            g = demo.get("gender", "unknown")
            age_dist[a] = age_dist.get(a, 0) + 1
            gender_dist[g] = gender_dist.get(g, 0) + 1

        audit_data = {
            "total_hard_samples": len(hard_meta),
            "age_distribution": age_dist,
            "gender_distribution": gender_dist,
            "mean_score": mean_score,
        }
        with open(audit_dir / "hard_sample_demographics.json", "w", encoding="utf-8") as af:
            json.dump(audit_data, af, indent=2)


def mine_hard_samples_by_feature_similarity(
    extractor: nn.Module,
    synthetic_metadata: List[ImageMetadata],
    real_metadata: Optional[List[ImageMetadata]] = None,
    cache: Optional[Dict[str, torch.Tensor]] = None,
    similarity_threshold: float = 0.85,
    batch_size: int = 16,
    device: Union[str, torch.device] = "cpu",
    annotated_manifest_path: Optional[Path] = None,
    hard_manifest_path: Optional[Path] = None,
) -> HardSampleMiningReport:
    """Extract hard samples by measuring latent feature cosine similarity to real faces using DINOv2.

    Synthetic faces whose deep representation is >= similarity_threshold compared to their
    source real face (or closest real face) are flagged as Hard Samples.
    """
    dev = torch.device(device)
    extractor.eval()
    extractor.to(dev)

    if not synthetic_metadata:
        return HardSampleMiningReport(
            total_synthetic=0,
            rejected_count=0,
            bypassed_hard_count=0,
            bypass_rate=0.0,
            mean_fake_prob=0.0,
            threshold=similarity_threshold,
            hard_sample_ids=[],
        )

    # 1. Precompute real embeddings if real_metadata is provided
    real_embeddings: Dict[str, torch.Tensor] = {}
    if real_metadata:
        needed_real_ids = {m.source_real_id for m in synthetic_metadata if m.source_real_id}
        if needed_real_ids:
            subset_real = [m for m in real_metadata if m.image_id in needed_real_ids]
            if not subset_real:
                subset_real = real_metadata[:100]
        else:
            subset_real = real_metadata[:100]

        real_dataset = DeepfakeImageDataset(metadata_list=subset_real, cache=cache)
        real_loader = DataLoader(
            real_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_image_samples,
        )
        with torch.no_grad():
            for batch in real_loader:
                imgs = batch["images"].to(dev)
                metas: List[ImageMetadata] = batch["metadata"]
                embs = extractor(imgs)
                embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
                for meta, emb in zip(metas, embs):
                    real_embeddings[meta.image_id] = emb.detach().cpu()

    # 2. Compute synthetic embeddings and cosine similarities
    syn_dataset = DeepfakeImageDataset(metadata_list=synthetic_metadata, cache=cache)
    syn_loader = DataLoader(
        syn_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_image_samples,
    )

    all_sims: List[float] = []
    updated_meta: List[ImageMetadata] = []
    hard_meta: List[ImageMetadata] = []
    hard_ids: List[str] = []

    with torch.no_grad():
        for batch in syn_loader:
            imgs = batch["images"].to(dev)
            metas: List[ImageMetadata] = batch["metadata"]
            embs = extractor(imgs)
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)

            for meta, emb in zip(metas, embs):
                emb_cpu = emb.detach().cpu()
                sim_val = 0.0
                if meta.source_real_id and meta.source_real_id in real_embeddings:
                    target_emb = real_embeddings[meta.source_real_id]
                    sim_val = float(torch.dot(emb_cpu, target_emb).item())
                elif real_embeddings:
                    sim_val = max(
                        float(torch.dot(emb_cpu, r_emb).item())
                        for r_emb in real_embeddings.values()
                    )
                else:
                    sim_val = 0.0

                all_sims.append(sim_val)
                is_hard = sim_val >= similarity_threshold

                annotated = ImageMetadata(
                    image_id=meta.image_id,
                    label=meta.label,
                    manipulation_type=meta.manipulation_type,
                    prompt=meta.prompt,
                    source_real_id=meta.source_real_id,
                    file_path=meta.file_path,
                    bbox=meta.bbox,
                    is_hard_sample=is_hard,
                    baseline_score=sim_val,
                    data_source="hard_sample" if is_hard else "synthetic",
                    demographics=meta.demographics,
                )
                updated_meta.append(annotated)
                if is_hard:
                    hard_meta.append(annotated)
                    hard_ids.append(annotated.image_id)

    total_syn = len(updated_meta)
    hard_count = len(hard_meta)
    rejected_count = total_syn - hard_count
    bypass_rate = float(hard_count / max(1, total_syn))
    mean_sim = float(sum(all_sims) / max(1, len(all_sims)))

    _save_mining_results(
        updated_meta=updated_meta,
        hard_meta=hard_meta,
        mean_score=mean_sim,
        annotated_manifest_path=annotated_manifest_path,
        hard_manifest_path=hard_manifest_path,
    )

    report = HardSampleMiningReport(
        total_synthetic=total_syn,
        rejected_count=rejected_count,
        bypassed_hard_count=hard_count,
        bypass_rate=bypass_rate,
        mean_fake_prob=mean_sim,
        threshold=similarity_threshold,
        hard_sample_ids=hard_ids,
    )
    logger.info(
        "DINOv2 Feature similarity mining complete: total=%d | rejected=%d | hard_samples=%d (%.1f%%)",
        total_syn,
        rejected_count,
        hard_count,
        bypass_rate * 100.0,
    )
    return report


def mine_hard_samples(
    model: nn.Module,
    synthetic_metadata: List[ImageMetadata],
    cache: Optional[Dict[str, torch.Tensor]] = None,
    bypass_threshold: float = 0.5,
    batch_size: int = 16,
    device: Union[str, torch.device] = "cpu",
    annotated_manifest_path: Optional[Path] = None,
    hard_manifest_path: Optional[Path] = None,
) -> HardSampleMiningReport:
    """Evaluate synthetic DF images against baseline dinoViT and extract bypassed hard samples.

    Args:
        model: Baseline DeepfakeImageDetector (e.g., dinoViT).
        synthetic_metadata: List of ImageMetadata for synthetic fake images (label=1).
        cache: Optional in-memory image tensor cache for headless testing.
        bypass_threshold: Probability threshold tau below which a fake image is considered
                          to have bypassed the baseline detector (default 0.5).
        batch_size: Batch size for inference.
        device: Target compute device ('cpu' or 'cuda').
        annotated_manifest_path: Optional path to write all synthetic items updated with
                                 baseline_score and is_hard_sample flags.
        hard_manifest_path: Optional path to write only the bypassed hard samples.

    Returns:
        HardSampleMiningReport containing bypass analytics and mined hard sample IDs.
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    if not synthetic_metadata:
        return HardSampleMiningReport(
            total_synthetic=0,
            rejected_count=0,
            bypassed_hard_count=0,
            bypass_rate=0.0,
            mean_fake_prob=0.0,
            threshold=bypass_threshold,
            hard_sample_ids=[],
        )

    dataset = DeepfakeImageDataset(metadata_list=synthetic_metadata, cache=cache)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_image_samples,
    )

    all_probs: List[float] = []
    updated_meta: List[ImageMetadata] = []
    hard_meta: List[ImageMetadata] = []
    hard_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(dev)
            metas: List[ImageMetadata] = batch["metadata"]

            logits = model(images)
            probs = torch.sigmoid(logits).view(-1).detach().cpu().tolist()

            for meta, prob in zip(metas, probs):
                p_val = float(prob)
                all_probs.append(p_val)
                # Synthetic DF sample (label=1) that scores below threshold bypassed the baseline!
                is_bypass = (meta.label == 1) and (p_val < bypass_threshold)

                annotated = ImageMetadata(
                    image_id=meta.image_id,
                    label=meta.label,
                    manipulation_type=meta.manipulation_type,
                    prompt=meta.prompt,
                    source_real_id=meta.source_real_id,
                    file_path=meta.file_path,
                    bbox=meta.bbox,
                    is_hard_sample=is_bypass,
                    baseline_score=p_val,
                    data_source="hard_sample" if is_bypass else "synthetic",
                    demographics=meta.demographics,
                )
                updated_meta.append(annotated)
                if is_bypass:
                    hard_meta.append(annotated)
                    hard_ids.append(annotated.image_id)

    total_syn = len(updated_meta)
    hard_count = len(hard_meta)
    rejected_count = total_syn - hard_count
    bypass_rate = compute_bypass_rate(all_probs, threshold=bypass_threshold)
    mean_prob = float(sum(all_probs) / max(1, len(all_probs)))

    _save_mining_results(
        updated_meta=updated_meta,
        hard_meta=hard_meta,
        mean_score=mean_prob,
        annotated_manifest_path=annotated_manifest_path,
        hard_manifest_path=hard_manifest_path,
    )

    report = HardSampleMiningReport(
        total_synthetic=total_syn,
        rejected_count=rejected_count,
        bypassed_hard_count=hard_count,
        bypass_rate=bypass_rate,
        mean_fake_prob=mean_prob,
        threshold=bypass_threshold,
        hard_sample_ids=hard_ids,
    )
    logger.info(
        "Hard-sample mining complete: total=%d | rejected=%d | bypassed(hard)=%d (%.1f%%)",
        total_syn,
        rejected_count,
        hard_count,
        bypass_rate * 100.0,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine bypassed hard samples using DINOv2 latent feature similarity or baseline detector."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["feature_similarity", "detector"],
        default="detector",
        help="Mining method: 'feature_similarity' (DINOv2 cosine similarity against real faces) "
             "or 'detector' (P(fake) < tau from trained baseline classifier). Default: detector.",
    )
    parser.add_argument(
        "--input-manifest", type=str, default="data/synthetic/curated_manifest.jsonl"
    )
    parser.add_argument(
        "--raw-manifest",
        type=str,
        default="data/raw/manifest.jsonl",
        help="Path to real faces manifest used for feature similarity matching.",
    )
    parser.add_argument(
        "--annotated-manifest",
        type=str,
        default="data/synthetic/annotated_synthetic_manifest.jsonl",
    )
    parser.add_argument(
        "--hard-manifest", type=str, default="data/hard_samples/hard_samples_manifest.jsonl"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained baseline dinoViT checkpoint (.pt). Required when --method=detector.",
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Allow running with untrained random weights strictly for headless testing.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="facebook/dinov2-small",
        help="Backbone architecture (e.g., facebook/dinov2-small, dinov2_small, or dino_vit)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold (default 0.45 for feature_similarity, 0.5 for detector)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    # Determine default threshold based on method if not explicitly specified
    threshold = args.threshold
    if threshold is None:
        threshold = 0.45 if args.method == "feature_similarity" else 0.5

    in_path = Path(args.input_manifest)
    if not in_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {in_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        meta_list = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

    if args.method == "feature_similarity":
        logger.info(
            "Running objective DINOv2 latent feature similarity mining (threshold=%.2f)...",
            threshold,
        )
        backbone_id = (
            args.backbone if "dinov2" in args.backbone else "facebook/dinov2-small"
        )
        extractor = MetaDinoV2Wrapper(model_name=backbone_id, pretrained=True)

        real_meta_list: Optional[List[ImageMetadata]] = None
        raw_path = Path(args.raw_manifest)
        if raw_path.exists():
            with open(raw_path, "r", encoding="utf-8") as rf:
                real_meta_list = [
                    ImageMetadata.from_dict(json.loads(line))
                    for line in rf
                    if line.strip()
                ]

        mine_hard_samples_by_feature_similarity(
            extractor=extractor,
            synthetic_metadata=meta_list,
            real_metadata=real_meta_list,
            similarity_threshold=threshold,
            device=args.device,
            annotated_manifest_path=Path(args.annotated_manifest),
            hard_manifest_path=Path(args.hard_manifest),
        )
        return

    # Method == "detector": enforce checkpoint requirement
    if not args.checkpoint and not args.allow_untrained:
        raise ValueError(
            "Refusing to mine hard samples with uninitialized/untrained weights! "
            "Provide a trained baseline checkpoint via --checkpoint <path> "
            "(e.g. experiments/checkpoints/baseline_d0_best.pt). "
            "See .agents/rules/ZERO_MOCK_DATA_RULE.md"
        )

    model = DeepfakeImageDetector(
        backbone=args.backbone,
        num_classes=1,
        freeze_extractor=True,
        pretrained=True,
    )

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        logger.info("Loading baseline detector checkpoint from %s...", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        model.load_state_dict(state_dict, strict=False)
        logger.info("Successfully loaded trained weights into baseline detector.")

    mine_hard_samples(
        model=model,
        synthetic_metadata=meta_list,
        bypass_threshold=threshold,
        device=args.device,
        annotated_manifest_path=Path(args.annotated_manifest),
        hard_manifest_path=Path(args.hard_manifest),
    )


if __name__ == "__main__":
    main()
