"""Image schema contracts and metadata dataclasses for 2D deepfake analysis and hard-sample mining."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import torch

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImageMetadata:
    """Metadata specification for 2D facial images in deepfake and hard-sample datasets.

    Attributes:
        image_id: Unique string identifier for the image.
        label: Classification target (0 for Real, 1 for Fake).
        manipulation_type: Method of synthesis ('real', 'sd_text2img', 'sd_inpaint', 'sd_img2img').
        prompt: Optional text prompt used for generative conditioning.
        source_real_id: Optional parent real image ID if synthesized via inpaint/img2img.
        file_path: Optional path to the image file on disk.
        bbox: Optional face bounding box in Pascal VOC xyxy format [x_min, y_min, x_max, y_max].
        is_hard_sample: True if this synthetic DF sample bypassed the baseline dinoViT detector.
        baseline_score: Predicted fake probability P(fake) from baseline dinoViT in [0.0, 1.0].
        data_source: Provenance bucket ('baseline', 'synthetic', 'hard_sample').
        demographics: Optional dictionary storing demographic attributes (age_group, gender, glasses, identity_id).
    """

    image_id: str
    label: int
    manipulation_type: Optional[str] = None
    prompt: Optional[str] = None
    source_real_id: Optional[str] = None
    file_path: Optional[str] = None
    bbox: Optional[List[int]] = None
    is_hard_sample: bool = False
    baseline_score: Optional[float] = None
    data_source: str = "baseline"
    demographics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageMetadata":
        """Deserialize metadata from a dictionary."""
        baseline_score_raw = data.get("baseline_score")
        return cls(
            image_id=str(data["image_id"]),
            label=int(data["label"]),
            manipulation_type=data.get("manipulation_type"),
            prompt=data.get("prompt"),
            source_real_id=data.get("source_real_id"),
            file_path=data.get("file_path"),
            bbox=data.get("bbox"),
            is_hard_sample=bool(data.get("is_hard_sample", False)),
            baseline_score=float(baseline_score_raw) if baseline_score_raw is not None else None,
            data_source=str(data.get("data_source", "baseline")),
            demographics=data.get("demographics"),
        )


@dataclass
class ImageSample:
    """Standardized 2D image sample holding tensor and metadata.

    Tensor Shape Contract:
        image: torch.Tensor of shape (C, H, W) in RGB [0.0, 1.0] float32.
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

    @property
    def is_hard_sample(self) -> bool:
        """Return whether this sample bypassed the baseline detector."""
        return self.metadata.is_hard_sample
