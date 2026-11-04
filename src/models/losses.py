"""Loss functions for fine-grained facial occlusion segmentation."""

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice Loss supporting multi-channel class probabilities."""

    def __init__(self, smooth: float = 1e-6, ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Soft Dice loss.

        Args:
            logits: Predicted class logits of shape (B, num_classes, H, W).
            targets: Integer ground-truth class indices of shape (B, H, W).

        Returns:
            Scalar Dice loss tensor.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Create valid mask
        valid_mask = (targets != self.ignore_index).unsqueeze(1)  # (B, 1, H, W)
        safe_targets = torch.where(targets == self.ignore_index, 0, targets)

        # One-hot encode targets to shape (B, num_classes, H, W)
        target_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        target_one_hot = target_one_hot * valid_mask.float()
        probs = probs * valid_mask.float()

        # Compute Dice per class over spatial dimensions (B, C)
        intersection = torch.sum(probs * target_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs + target_one_hot, dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)
        return dice_loss


class CompositeOcclusionLoss(nn.Module):
    """Composite Cross-Entropy + Soft Dice Loss for fine-grained segmentation.

    Balances overall class likelihood while boosting high-frequency boundary
    precision and preventing under-segmentation on small occlusion items.
    """

    def __init__(
        self,
        num_classes: int = 6,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: list[float] | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        weight_tensor: torch.Tensor | None = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        self.ce = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index)
        self.dice = SoftDiceLoss(ignore_index=ignore_index)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute composite loss.

        Args:
            logits: Tensor of shape (B, num_classes, H, W).
            targets: Tensor of shape (B, H, W).

        Returns:
            Dictionary containing 'loss', 'ce_loss', and 'dice_loss'.
        """
        ce_val = self.ce(logits, targets) if self.ce_weight > 0.0 else torch.tensor(0.0, device=logits.device)
        dice_val = self.dice(logits, targets) if self.dice_weight > 0.0 else torch.tensor(0.0, device=logits.device)

        total_loss = self.ce_weight * ce_val + self.dice_weight * dice_val

        return {
            "loss": total_loss,
            "ce_loss": ce_val,
            "dice_loss": dice_val,
        }
