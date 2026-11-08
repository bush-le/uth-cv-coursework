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
from src.models.detector.image_detector import DeepfakeImageDetector
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
        description="Run baseline dinoViT on synthetic DF images to mine bypassed hard samples."
    )
    parser.add_argument(
        "--input-manifest", type=str, default="data/processed/curated_manifest.jsonl"
    )
    parser.add_argument(
        "--annotated-manifest",
        type=str,
        default="data/processed/annotated_synthetic_manifest.jsonl",
    )
    parser.add_argument(
        "--hard-manifest", type=str, default="data/processed/hard_samples_manifest.jsonl"
    )
    parser.add_argument("--backbone", type=str, default="dino_vit")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    in_path = Path(args.input_manifest)
    if in_path.exists():
        with open(in_path, "r", encoding="utf-8") as f:
            meta_list = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]
    else:
        meta_list = [
            ImageMetadata(image_id=f"syn_df_{i}", label=1, manipulation_type="sd_inpaint")
            for i in range(8)
        ]

    model = DeepfakeImageDetector(backbone=args.backbone, num_classes=1, freeze_extractor=True)
    mine_hard_samples(
        model=model,
        synthetic_metadata=meta_list,
        bypass_threshold=args.threshold,
        device=args.device,
        annotated_manifest_path=Path(args.annotated_manifest),
        hard_manifest_path=Path(args.hard_manifest),
    )


if __name__ == "__main__":
    main()
