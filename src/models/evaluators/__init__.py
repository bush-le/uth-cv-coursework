"""Evaluation modules for deepfake detectors and hard sample mining."""

from src.models.evaluators.hardness import (
    HardnessReport,
    build_report,
    embed_images,
    load_verified_checkpoint,
    real_manifold_scores,
    real_reference_scores,
)

__all__ = [
    "HardnessReport",
    "build_report",
    "embed_images",
    "load_verified_checkpoint",
    "real_manifold_scores",
    "real_reference_scores",
]
