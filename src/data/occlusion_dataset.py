"""Synchronized dataset implementation for paired facial images and occlusion masks."""

from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.common.logger import get_logger

logger = get_logger(__name__)


class FacialOcclusionDataset(Dataset):
    """Dataset loader enforcing synchronized image and segmentation mask pairs."""

    def __init__(
        self,
        image_paths: list[str | Path],
        mask_paths: list[str | Path],
        image_size: tuple[int, int] = (512, 512),
        is_train: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """Initialize the dataset with pairs of image and mask paths.

        Args:
            image_paths: List of file paths to RGB face portraits.
            mask_paths: List of file paths to corresponding single-channel integer masks.
            image_size: Desired spatial output size (height, width).
            is_train: Whether to apply stochastic training augmentations.
            mean: Normalization channel means.
            std: Normalization channel standard deviations.
        """
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
            )

        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.image_size = image_size
        self.is_train = is_train
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and return synchronized (image, mask) pair.

        Returns:
            Dictionary containing:
                - 'image': Float32 tensor of shape (3, H, W).
                - 'mask': Int64 tensor of shape (H, W).
                - 'image_path': String path to the image.
        """
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to load image %s: %s", img_path, e)
            image = Image.fromarray(np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8))

        try:
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            logger.error("Failed to load mask %s: %s", mask_path, e)
            mask = Image.fromarray(np.zeros(self.image_size, dtype=np.uint8))

        # Resize image using Bilinear and mask using Nearest to avoid corrupting integer labels
        image = TF.resize(image, list(self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, list(self.image_size), interpolation=TF.InterpolationMode.NEAREST)

        # Synchronized horizontal flip augmentation
        if self.is_train and torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Convert image to FloatTensor in [0, 1] and normalize
        image_tensor = TF.to_tensor(image)  # Shape: (3, H, W)
        image_tensor = TF.normalize(image_tensor, mean=list(self.mean), std=list(self.std))

        # Convert mask directly to LongTensor without scaling
        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np)  # Shape: (H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(img_path),
        }
