"""Model factory for instantiating CV and Deepfake Detector architectures."""

from typing import Any
import torch.nn as nn

from src.models.deep import SimpleCVCNN
from src.models.detector.image_detector import DeepfakeImageDetector


def build_model(
    architecture: str = "dino_vit",
    num_classes: int = 1,
    in_channels: int = 3,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate and configure model architecture.

    Args:
        architecture: Name of the architecture ('dino_vit', 'dinov2_vitb14', 'resnet18', 'custom_cnn').
        num_classes: Number of output target classes.
        in_channels: Number of input image channels.
        **kwargs: Additional model parameters.

    Returns:
        Configured nn.Module instance.
    """
    arch = architecture.lower()
    if arch in ("dinov2_small", "dinov2", "dinov2_vitb14"):
        return DeepfakeImageDetector(backbone="dinov2_small", num_classes=num_classes, **kwargs)

    if arch in ("dino_vit", "dinovit", "deepfake_detector"):
        return DeepfakeImageDetector(backbone="dino_vit", num_classes=num_classes, **kwargs)

    if arch in ("custom_cnn", "simple_cnn", "default"):
        return SimpleCVCNN(in_channels=in_channels, num_classes=num_classes)

    # Fallback to torchvision if available
    try:
        import torchvision.models as tv_models

        if hasattr(tv_models, arch):
            model_fn = getattr(tv_models, arch)
            model = model_fn(**kwargs)
            if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
    except ImportError:
        pass

    raise ValueError(f"Unsupported model architecture: {architecture}")
