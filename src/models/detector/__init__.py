"""Detector module exporting DINO ViT and CNN deepfake image detectors."""

from src.models.detector.image_detector import (
    DeepfakeImageDetector,
    DinoViTBackbone,
    SimpleImageCNN,
)

__all__ = ["DeepfakeImageDetector", "DinoViTBackbone", "SimpleImageCNN"]
