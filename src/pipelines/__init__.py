"""Pipelines package for Stable Diffusion synthesis, quality curation, dinoViT hard-sample mining, fine-tuning, and benchmarking."""

from src.pipelines.curate_image_dataset import (
    ImageCurationGate,
    ImageCurationReport,
    curate_image_manifest,
)
from src.pipelines.evaluate_image_benchmark import (
    compare_baseline_vs_finetuned,
    evaluate_image_detector,
)
from src.pipelines.generate_deepfake_images import run_image_generation
from src.pipelines.mine_hard_samples import HardSampleMiningReport, mine_hard_samples
from src.pipelines.train_image_detector import (
    combine_training_manifests,
    train_deepfake_image_detector,
)

__all__ = [
    "run_image_generation",
    "ImageCurationGate",
    "ImageCurationReport",
    "curate_image_manifest",
    "HardSampleMiningReport",
    "mine_hard_samples",
    "combine_training_manifests",
    "train_deepfake_image_detector",
    "evaluate_image_detector",
    "compare_baseline_vs_finetuned",
]
