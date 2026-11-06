"""Image schema contracts and metadata dataclasses for 2D deepfake analysis."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import torch

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImageMetadata:
    """Metadata specification for 2D facial images in deepfake datasets.
    
    Attributes:
        image_id: Unique string identifier for the image.
        label: Classification target (0 for Real, 1 for Fake).
        manipulation_type: Method of synthesis (e.g. 'sd_text2img', 'sd_inpaint', 'sd_img2img', 'real').
        prompt: Optional text prompt used for generative conditioning.
        source_real_id: Optional parent real image ID if synthesized via inpaint/img2img.
        file_path: Optional path to the image file on disk.
        bbox: Optional face bounding box [x1, y1, x2, y2].
    """

    image_id: str
    label: int
    manipulation_type: Optional[str] = None
    prompt: Optional[str] = None
    source_real_id: Optional[str] = None
    file_path: Optional[str] = None
    bbox: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageMetadata":
        """Deserialize metadata from a dictionary."""
        return cls(
            image_id=str(data["image_id"]),
            label=int(data["label"]),
            manipulation_type=data.get("manipulation_type"),
            prompt=data.get("prompt"),
            source_real_id=data.get("source_real_id"),
            file_path=data.get("file_path"),
            bbox=data.get("bbox"),
        )


@dataclass
class ImageSample:
    """Standardized 2D image sample holding tensor and metadata.
    
    Tensor Shape Contract:
        image: torch.Tensor of shape (C, H, W)
    """

    image: torch.Tensor
    metadata: ImageMetadata

    def __post_init__(self) -> None:
        """Validate tensor shape contracts."""
        if not isinstance(self.image, torch.Tensor):
            raise TypeError(f"image must be a torch.Tensor, got {type(self.image)}")
        if self.image.ndim != 3:
            raise ValueError(
                f"Image tensor must have exactly 3 dimensions (C, H, W), "
                f"got shape {tuple(self.image.shape)} (ndim={self.image.ndim})"
            )

    @property
    def label(self) -> int:
        """Get the integer classification label (0=Real, 1=Fake)."""
        return self.metadata.label
