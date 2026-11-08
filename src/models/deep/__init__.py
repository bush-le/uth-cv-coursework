"""Deep learning neural network architectures for Computer Vision."""

import torch
import torch.nn as nn


class SimpleCVCNN(nn.Module):
    """Modular lightweight Convolutional Neural Network for benchmarking and smoke testing."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        """Initialize SimpleCVCNN.

        Args:
            in_channels: Number of image channels (3 for RGB).
            num_classes: Number of prediction classes.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        feats = self.features(x)
        return self.classifier(feats)


__all__ = ["SimpleCVCNN"]
