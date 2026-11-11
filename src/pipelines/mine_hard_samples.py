"""Phase 2 Pipeline: Hard-Sample Mining via Real-Manifold Proximity or Baseline dinoViT.

Extracts hard synthetic samples:
- real_manifold: Top-k cosine similarity to real reference manifold (excluding own source face).
                 Threshold tau calibrated dynamically from leave-one-out reference quantile.
- detector: P(fake) < tau from trained and verified baseline classifier.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.config import cfg_get, load_config
from src.common.logger import get_logger
from src.common.metrics import calibrate_threshold, compute_bypass_rate
from src.common.seed import seed_everything
from src.data.image_datamodule import collate_image_samples
from src.data.image_dataset import DeepfakeImageDataset
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector, MetaDinoV2Wrapper
from src.models.evaluators.hardness import (
    HardnessReport,
    build_report,
    embed_images,
    load_verified_checkpoint,
    real_manifold_scores,
    real_reference_scores,
)

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


# Re-export from data transforms for backward compatibility
from src.data.transforms import crop_face_tensor


def filter_by_embedding_diversity(
    embeddings: torch.Tensor,
    scores: Union[torch.Tensor, List[float]],
    max_cosine_sim: float = 0.95,
) -> List[int]:
    """Prune near-duplicate hard samples using greedy diversity selection on embeddings.

    Sorts candidates in ascending order of score (hardest first, where lower fake prob is harder),
    and iteratively accepts an item only if its cosine similarity to all previously accepted
    embeddings is <= max_cosine_sim.

    Args:
        embeddings: Normalized or unnormalized embeddings of shape (N, D).
        scores: Hardness scores for each item, shape (N,). Lower is harder.
        max_cosine_sim: Maximum allowed pairwise cosine similarity.

    Returns:
        List of selected integer indices from [0, N-1].
    """
    if len(embeddings) == 0:
        return []

    norm_embs = F.normalize(embeddings.float(), dim=-1)
    if isinstance(scores, torch.Tensor):
        scores_arr = scores.detach().cpu().numpy()
    else:
        scores_arr = np.array(scores, dtype=np.float32)

    # Sort indices by score (ascending: hardest first)
    order = np.argsort(scores_arr)
    selected: List[int] = []

    for idx in order:
        if len(selected) == 0:
            selected.append(int(idx))
            continue

        cand_emb = norm_embs[idx : idx + 1]
        sel_embs = norm_embs[selected]
        sims = torch.mm(cand_emb, sel_embs.t()).view(-1)
        max_sim = float(sims.max().item())

        if max_sim <= max_cosine_sim:
            selected.append(int(idx))

    return sorted(selected)



def _save_mining_results(
    updated_meta: List[ImageMetadata],
    hard_meta: List[ImageMetadata],
    mean_score: float,
    annotated_manifest_path: Optional[Path],
    hard_manifest_path: Optional[Path],
    hardness_report: Optional[HardnessReport] = None,
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

        if hardness_report is not None:
            with open(audit_dir / "hardness_report.json", "w", encoding="utf-8") as hf:
                json.dump(hardness_report.to_dict(), hf, indent=2)


def mine_hard_samples_real_manifold(
    extractor: nn.Module,
    synthetic_metadata: List[ImageMetadata],
    real_metadata: List[ImageMetadata],
    cache: Optional[Dict[str, torch.Tensor]] = None,
    tau: Optional[float] = None,
    reference_quantile: float = 0.5,
    k_neighbors: int = 5,
    max_real_reference: Optional[int] = None,
    batch_size: int = 16,
    device: Union[str, torch.device] = "cpu",
    annotated_manifest_path: Optional[Path] = None,
    hard_manifest_path: Optional[Path] = None,
) -> HardnessReport:
    """Mine hard samples via objective real-manifold proximity."""
    if max_real_reference is not None and len(real_metadata) > max_real_reference:
        real_metadata = real_metadata[:max_real_reference]
    if not synthetic_metadata:
        rep = HardnessReport(
            method="real_manifold",
            tau=tau or 0.0,
            tau_source="empty",
            total=0,
            hard_count=0,
            hard_rate=0.0,
            hard_rate_by_mode={},
            hard_rate_by_group={},
            hard_ids=[],
        )
        return rep

    # 1. Embed real reference images and calibrate tau
    real_embs = embed_images(
        extractor=extractor,
        metas=real_metadata,
        batch_size=batch_size,
        device=device,
        cache=cache,
        crop_faces=True,
    )
    real_id_to_idx = {m.image_id: idx for idx, m in enumerate(real_metadata)}

    if tau is None:
        if len(real_metadata) > 1:
            ref_scores = real_reference_scores(real_embs, k=k_neighbors)
            tau_val = calibrate_threshold(ref_scores.tolist(), quantile=reference_quantile)
            tau_source = f"reference_quantile_{reference_quantile:.2f}"
        else:
            tau_val = 0.5
            tau_source = "default_fallback"
    else:
        tau_val = float(tau)
        tau_source = "cli_override"

    # 2. Embed synthetic images
    syn_embs = embed_images(
        extractor=extractor,
        metas=synthetic_metadata,
        batch_size=batch_size,
        device=device,
        cache=cache,
        crop_faces=True,
    )

    # 3. Exclude own source real index
    exclude_idx: List[Optional[int]] = [
        real_id_to_idx.get(m.source_real_id) if m.source_real_id else None
        for m in synthetic_metadata
    ]

    scores = real_manifold_scores(
        fake_emb=syn_embs,
        real_emb=real_embs,
        k=k_neighbors,
        exclude_idx=exclude_idx,
    )

    hardness_report = build_report(
        metas=synthetic_metadata,
        scores=scores,
        tau=tau_val,
        tau_source=tau_source,
        method="real_manifold",
        higher_is_harder=True,
    )

    updated_meta: List[ImageMetadata] = []
    hard_meta: List[ImageMetadata] = []

    for i, meta in enumerate(synthetic_metadata):
        score_val = float(scores[i].item())
        is_hard = score_val >= tau_val

        scores_dict = meta.scores.copy() if meta.scores else {}
        scores_dict["real_manifold"] = score_val

        # Diagnostic source similarity if source real face exists
        ex_idx = exclude_idx[i]
        if ex_idx is not None and ex_idx < real_embs.shape[0]:
            src_sim = float(torch.dot(syn_embs[i], real_embs[ex_idx]).item())
            scores_dict["source_similarity"] = src_sim

        annotated = ImageMetadata(
            image_id=meta.image_id,
            label=meta.label,
            manipulation_type=meta.manipulation_type,
            prompt=meta.prompt,
            source_real_id=meta.source_real_id,
            file_path=meta.file_path,
            bbox=meta.bbox,
            is_hard_sample=is_hard,
            baseline_score=score_val,
            data_source="hard_sample" if is_hard else "synthetic",
            demographics=meta.demographics,
            generation=meta.generation,
            scores=scores_dict,
        )
        updated_meta.append(annotated)
        if is_hard:
            hard_meta.append(annotated)

    _save_mining_results(
        updated_meta=updated_meta,
        hard_meta=hard_meta,
        mean_score=float(scores.mean().item()) if len(scores) > 0 else 0.0,
        annotated_manifest_path=annotated_manifest_path,
        hard_manifest_path=hard_manifest_path,
        hardness_report=hardness_report,
    )

    logger.info(
        "Real-manifold mining complete: total=%d | rejected=%d | hard_samples=%d (%.1f%%) | tau=%.3f (%s)",
        hardness_report.total,
        hardness_report.total - hardness_report.hard_count,
        hardness_report.hard_count,
        hardness_report.hard_rate * 100.0,
        hardness_report.tau,
        hardness_report.tau_source,
    )
    return hardness_report


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
    """Compatibility wrapper evaluating real-manifold proximity."""
    rep = mine_hard_samples_real_manifold(
        extractor=extractor,
        synthetic_metadata=synthetic_metadata,
        real_metadata=real_metadata or [],
        cache=cache,
        tau=similarity_threshold,
        batch_size=batch_size,
        device=device,
        annotated_manifest_path=annotated_manifest_path,
        hard_manifest_path=hard_manifest_path,
    )
    return HardSampleMiningReport(
        total_synthetic=rep.total,
        rejected_count=rep.total - rep.hard_count,
        bypassed_hard_count=rep.hard_count,
        bypass_rate=rep.hard_rate,
        mean_fake_prob=rep.tau,
        threshold=rep.tau,
        hard_sample_ids=rep.hard_ids,
    )


def mine_hard_samples(
    model: nn.Module,
    synthetic_metadata: List[ImageMetadata],
    cache: Optional[Dict[str, torch.Tensor]] = None,
    bypass_threshold: float = 0.5,
    min_hard_samples: int = 0,
    batch_size: int = 16,
    device: Union[str, torch.device] = "cpu",
    annotated_manifest_path: Optional[Path] = None,
    hard_manifest_path: Optional[Path] = None,
) -> HardSampleMiningReport:
    """Evaluate synthetic DF images against baseline dinoViT and extract bypassed hard samples."""
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
                is_bypass = (meta.label == 1) and (p_val < bypass_threshold)

                scores_dict = meta.scores.copy() if meta.scores else {}
                scores_dict["detector_fake_prob"] = p_val

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
                    generation=meta.generation,
                    scores=scores_dict,
                )
                updated_meta.append(annotated)
                if is_bypass:
                    hard_meta.append(annotated)
                    hard_ids.append(annotated.image_id)

    # Adaptive fallback: ensure at least min_hard_samples are mined from hardest candidates
    if min_hard_samples > 0 and len(hard_meta) < min_hard_samples and updated_meta:
        sorted_cands = sorted(
            updated_meta,
            key=lambda m: m.baseline_score if m.baseline_score is not None else 1.0,
        )
        needed = min(min_hard_samples, len(updated_meta))
        top_hard = sorted_cands[:needed]
        hard_meta = []
        hard_ids = []
        for m in top_hard:
            m.is_hard_sample = True
            m.data_source = "hard_sample"
            hard_meta.append(m)
            hard_ids.append(m.image_id)
        hard_id_set = set(hard_ids)
        for m in updated_meta:
            if m.image_id in hard_id_set:
                m.is_hard_sample = True
                m.data_source = "hard_sample"

    total_syn = len(updated_meta)
    hard_count = len(hard_meta)
    rejected_count = total_syn - hard_count
    bypass_rate = float(hard_count / max(1, total_syn))
    mean_prob = float(sum(all_probs) / max(1, len(all_probs)))

    scores_tensor = torch.tensor(all_probs) if all_probs else torch.empty(0)
    rep = build_report(
        metas=synthetic_metadata,
        scores=scores_tensor,
        tau=bypass_threshold,
        tau_source="detector_threshold",
        method="detector",
        higher_is_harder=False,
    )

    _save_mining_results(
        updated_meta=updated_meta,
        hard_meta=hard_meta,
        mean_score=mean_prob,
        annotated_manifest_path=annotated_manifest_path,
        hard_manifest_path=hard_manifest_path,
        hardness_report=rep,
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
        description="Mine hard samples using real-manifold proximity or baseline detector."
    )
    parser.add_argument("--config", type=str, default="configs/dataset_factory.yaml", help="Path to config YAML")
    parser.add_argument(
        "--method",
        type=str,
        choices=["real_manifold", "detector", "feature_similarity"],
        default=None,
    )
    parser.add_argument("--input-manifest", type=str, default=None)
    parser.add_argument("--raw-manifest", type=str, default=None)
    parser.add_argument("--annotated-manifest", type=str, default=None)
    parser.add_argument("--hard-manifest", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    cfg_p = Path(args.config)
    if cfg_p.is_file():
        cfg = load_config(cfg_p)

    def _res(cli_val: Any, dotted_key: str, default_val: Any) -> Any:
        if cli_val is not None:
            return cli_val
        try:
            return cfg_get(cfg, dotted_key)
        except (KeyError, TypeError):
            return default_val

    seed_val = int(_res(args.seed, "seed", 42))
    seed_everything(seed_val)

    method = str(_res(args.method, "hardness.method", "real_manifold"))
    in_manifest_path = Path(_res(args.input_manifest, "paths.curated_manifest", "data/synthetic/curated_manifest.jsonl"))
    raw_manifest_path = Path(_res(args.raw_manifest, "paths.raw_manifest", "data/raw/manifest.jsonl"))
    annotated_manifest_path = Path(_res(args.annotated_manifest, "paths.annotated_manifest", "data/synthetic/annotated_synthetic_manifest.jsonl"))
    hard_manifest_path = Path(_res(args.hard_manifest, "paths.hard_manifest", "data/hard_samples/hard_samples_manifest.jsonl"))
    backbone_id = str(_res(args.backbone, "hardness.backbone_id", "facebook/dinov2-small"))
    device = str(_res(args.device, "device", "cpu"))
    batch_size = int(_res(None, "hardness.batch_size", 16))

    # Method == "detector": enforce checkpoint requirement before file loading
    checkpoint_path = _res(args.checkpoint, "hardness.detector_checkpoint", None)
    if method == "detector" and not checkpoint_path and not args.allow_untrained:
        raise ValueError(
            "Refusing to mine hard samples with uninitialized/untrained weights! "
            "Provide a trained baseline checkpoint via --checkpoint <path> "
            "or configure hardness.detector_checkpoint in dataset_factory.yaml. "
            "See .agents/rules/ZERO_MOCK_DATA_RULE.md"
        )

    if not in_manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {in_manifest_path}")

    with open(in_manifest_path, "r", encoding="utf-8") as f:
        synthetic_meta = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

    if method in ("real_manifold", "feature_similarity"):
        if not raw_manifest_path.exists():
            raise FileNotFoundError(f"Raw reference manifest not found: {raw_manifest_path}")
        with open(raw_manifest_path, "r", encoding="utf-8") as rf:
            real_meta = [ImageMetadata.from_dict(json.loads(line)) for line in rf if line.strip()]

        k_neighbors = int(_res(None, "hardness.k_neighbors", 5))
        ref_quantile = float(_res(None, "hardness.reference_quantile", 0.5))
        max_real_ref = int(_res(None, "hardness.max_real_reference", 200))
        extractor = MetaDinoV2Wrapper(model_name=backbone_id, pretrained=True)

        mine_hard_samples_real_manifold(
            extractor=extractor,
            synthetic_metadata=synthetic_meta,
            real_metadata=real_meta,
            tau=args.threshold,
            reference_quantile=ref_quantile,
            k_neighbors=k_neighbors,
            max_real_reference=max_real_ref,
            batch_size=batch_size,
            device=device,
            annotated_manifest_path=annotated_manifest_path,
            hard_manifest_path=hard_manifest_path,
        )
        return

    # Method == "detector": initialize detector model

    model = DeepfakeImageDetector(
        backbone=backbone_id,
        num_classes=1,
        freeze_extractor=True,
        pretrained=True,
    )

    if checkpoint_path:
        load_verified_checkpoint(model, checkpoint_path, device=device)

    detector_tau = float(_res(args.threshold, "hardness.detector_tau", 0.5))

    mine_hard_samples(
        model=model,
        synthetic_metadata=synthetic_meta,
        bypass_threshold=detector_tau,
        batch_size=batch_size,
        device=device,
        annotated_manifest_path=annotated_manifest_path,
        hard_manifest_path=hard_manifest_path,
    )


if __name__ == "__main__":
    main()
