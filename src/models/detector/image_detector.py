"""Deepfake image classifier architecture for real vs synthetic face detection."""

from typing import Optional, Union
import torch
import torch.nn as nn
from torchvision.models import resnet18

from src.common.logger import get_logger

logger = get_logger(__name__)


class SimpleImageCNN(nn.Module):
    """Lightweight 2D convolutional feature extractor for face images."""

    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class DeepfakeImageDetector(nn.Module):
    """Binary classifier detecting synthetic vs real facial images.
    
    Tensor Input Contract:
        x: (B, 3, H, W)
    Tensor Output Contract:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "resnet18":
            base = resnet18(weights=None)
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()  # type: ignore
            self.extractor = base
        else:
            feature_dim = 128
            self.extractor = SimpleImageCNN(out_features=feature_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        logger.info(f"Initialized DeepfakeImageDetector: backbone={backbone}, classes={num_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for image batch.
        
        Args:
            x: torch.Tensor of shape (B, 3, H, W).
            
        Returns:
            torch.Tensor of shape (B, num_classes) representing classification logits.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D image tensor (B, 3, H, W), got shape {tuple(x.shape)}")

        feats = self.extractor(x)
        logits = self.classifier(feats)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid probability for binary detection."""
        return torch.sigmoid(self.forward(x))
