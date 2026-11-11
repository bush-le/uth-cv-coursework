"""Data module for dataset parsing, transforms, and DataLoaders."""

from src.data.datamodule import CVDataModule
from src.data.dataset import StandardCVDataset
from src.data.image_datamodule import DeepfakeImageDataModule, collate_image_samples
from src.data.image_dataset import DeepfakeImageDataset
from src.data.image_schema import ImageMetadata, ImageSample
from src.data.raw_ingestion import (
    build_baseline_manifest,
    parse_celeba_metadata,
    parse_ffhq_aging_metadata,
)

__all__ = [
    "StandardCVDataset",
    "CVDataModule",
    "ImageMetadata",
    "ImageSample",
    "DeepfakeImageDataset",
    "DeepfakeImageDataModule",
    "collate_image_samples",
    "parse_ffhq_aging_metadata",
    "parse_celeba_metadata",
    "build_baseline_manifest",
]
