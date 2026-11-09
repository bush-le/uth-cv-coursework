"""Multi-modal automated curation gate for synthetic deepfake image datasets."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import torch

from src.common.logger import get_logger
from src.common.metrics import (
    compute_clip_similarity_proxy,
    compute_laplacian_sharpness,
)

logger = get_logger(__name__)


@dataclass
class ImageCurationReport:
    """Detailed quality inspection report for a synthetic face image."""

    image_id: str
    is_valid: bool
    sharpness_score: float
    clip_score: float
    face_detected: bool
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "is_valid": self.is_valid,
            "sharpness_score": self.sharpness_score,
            "clip_score": self.clip_score,
            "face_detected": self.face_detected,
            "rejection_reasons": self.rejection_reasons,
        }


class ImageCurationGate:
    """Automated multi-modal quality filter for synthetic face images."""

    def __init__(
        self,
        min_sharpness: float = 50.0,
        min_clip_score: float = 0.28,
    ) -> None:
        self.min_sharpness = min_sharpness
        self.min_clip_score = min_clip_score

    def evaluate_image(
        self,
        image: torch.Tensor,
        prompt: str = "portrait",
        face_detected: bool = True,
        image_id: str = "unknown",
    ) -> ImageCurationReport:
        """Evaluate a synthetic face image against sharpness, CLIP alignment, and face detection.
        
        Args:
            image: Tensor of shape (3, H, W) in [0.0, 1.0].
            prompt: Text prompt used for generation.
            face_detected: Boolean indicating if a face was detected.
            image_id: String identifier.
            
        Returns:
            ImageCurationReport with validation decision.
        """
        sharpness = compute_laplacian_sharpness(image)
        clip_score = compute_clip_similarity_proxy(image, prompt)

        reasons = []
        if not face_detected:
            reasons.append("No valid face landmark detected")
        if sharpness < self.min_sharpness:
            reasons.append(f"Sharpness {sharpness:.2f} below threshold {self.min_sharpness:.2f}")
        if clip_score < self.min_clip_score:
            reasons.append(f"CLIP score {clip_score:.3f} below threshold {self.min_clip_score:.3f}")

        is_valid = len(reasons) == 0
        return ImageCurationReport(
            image_id=image_id,
            is_valid=is_valid,
            sharpness_score=sharpness,
            clip_score=clip_score,
            face_detected=face_detected,
            rejection_reasons=reasons,
        )


def curate_image_manifest(
    input_manifest: Path,
    output_manifest: Path,
    gate: ImageCurationGate,
) -> Dict[str, Any]:
    """Filter raw image manifest entries through the curation gate."""
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    kept_count = 0
    rejected_count = 0
    curated_lines = []

    if input_manifest.exists():
        with open(input_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                file_path = item.get("file_path")
                if file_path and Path(file_path).exists():
                    try:
                        with Image.open(file_path) as img:
                            np_img = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                            tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
                        report = gate.evaluate_image(
                            tensor_img,
                            prompt=item.get("prompt", ""),
                            image_id=item.get("image_id", ""),
                        )
                        if report.is_valid:
                            curated_lines.append(json.dumps(item) + "\n")
                            kept_count += 1
                        else:
                            rejected_count += 1
                    except Exception as err:
                        logger.warning("Error reading %s for curation (%s), skipping.", file_path, err)
                        rejected_count += 1
                else:
                    logger.warning("Rejecting %s: image file does not exist (%s)", item.get("image_id"), file_path)
                    rejected_count += 1

    with open(output_manifest, "w", encoding="utf-8") as f:
        f.writelines(curated_lines)

    summary = {
        "total": kept_count + rejected_count,
        "kept": kept_count,
        "rejected": rejected_count,
    }
    logger.info("Image curation complete: %s", summary)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Curate synthetic images using automated quality gate.")
    parser.add_argument("--input-manifest", type=str, default="data/synthetic/fake_manifest.jsonl")
    parser.add_argument("--output-manifest", type=str, default="data/synthetic/curated_manifest.jsonl")
    parser.add_argument("--min-sharpness", type=float, default=10.0)
    parser.add_argument("--min-clip", type=float, default=0.15)
    args = parser.parse_args()

    in_path = Path(args.input_manifest)
    if not in_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {in_path}")

    gate = ImageCurationGate(min_sharpness=args.min_sharpness, min_clip_score=args.min_clip)
    curate_image_manifest(
        input_manifest=in_path,
        output_manifest=Path(args.output_manifest),
        gate=gate,
    )


if __name__ == "__main__":
    main()
