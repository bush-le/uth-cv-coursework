"""Deep learning model architectures for facial occlusion segmentation."""

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.logger import get_logger

logger = get_logger(__name__)


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FacialOcclusionSegmenter(nn.Module):
    """U-Net styled fine-grained facial occlusion segmentation architecture.

    Processes high-resolution face portraits of shape (B, 3, H, W) and generates
    dense class logits of shape (B, num_classes, H, W) preserving high-frequency
    boundary contours via multiscale skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 32,
    ) -> None:
        """Initialize the segmentation network.

        Args:
            in_channels: Number of input image channels (default: 3).
            num_classes: Number of target segmentation classes (default: 6).
            base_channels: Initial feature channel multiplier.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder stages
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        # Decoder stages
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Final projection head
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing class logits.

        Args:
            x: Standardized image tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        # Encoder
        e1 = self.enc1(x)        # (B, C, H, W)
        e2 = self.enc2(self.pool1(e1))  # (B, 2C, H/2, W/2)
        e3 = self.enc3(self.pool2(e2))  # (B, 4C, H/4, W/4)
        e4 = self.enc4(self.pool3(e3))  # (B, 8C, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))  # (B, 16C, H/16, W/16)

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)
        return logits
