"""Dataset loader for 2D real and synthetic deepfake facial images."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from src.common.logger import get_logger
from src.data.image_schema import ImageMetadata, ImageSample

logger = get_logger(__name__)


class DeepfakeImageDataset(Dataset):
    """Dataset providing 2D facial images and labels for deepfake detection.
    
    Returns ImageSample with image tensor of shape (C, H, W).
    """

    def __init__(
        self,
        metadata_list: List[ImageMetadata],
        cache: Optional[Dict[str, torch.Tensor]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_size: Optional[Tuple[int, int]] = (512, 512),
    ) -> None:
        self.metadata_list = metadata_list
        self.cache = cache or {}
        self.transform = transform
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.metadata_list)

    def _load_image_from_disk(self, file_path: Union[str, Path]) -> torch.Tensor:
        """Load image from disk and return RGB float tensor (3, H, W) in [0.0, 1.0]."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist on disk: {path}")

        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            if self.target_size is not None:
                img_rgb = img_rgb.resize(self.target_size, Image.Resampling.LANCZOS)
            np_img = np.array(img_rgb, dtype=np.float32) / 255.0
            return torch.from_numpy(np_img).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int) -> ImageSample:
        metadata = self.metadata_list[idx]

        # Use cache if available (keyed by image_id or file_path)
        if metadata.image_id in self.cache:
            img = self.cache[metadata.image_id]
        elif metadata.file_path in self.cache:
            img = self.cache[metadata.file_path]
        elif metadata.file_path:
            img = self._load_image_from_disk(metadata.file_path)
        else:
            h, w = self.target_size if self.target_size else (512, 512)
            img = torch.zeros(3, h, w, dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return ImageSample(image=img, metadata=metadata)
