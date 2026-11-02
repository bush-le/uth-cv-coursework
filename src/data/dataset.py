"""Standard PyTorch Dataset implementation for Computer Vision."""

from pathlib import Path
from typing import Any, Callable
import numpy as np


class StandardCVDataset:
    """Standardized dataset wrapper supporting image-label pairs."""

    def __init__(
        self,
        samples: list[tuple[str | Path, int]],
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            samples: List of (image_path, label) tuples.
            transform: Optional transform callable applied to the loaded image.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path, label = self.samples[idx]

        # Load image via PIL or NumPy
        try:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Fallback synthetic dummy image (H, W, 3)
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": str(image_path),
        }
