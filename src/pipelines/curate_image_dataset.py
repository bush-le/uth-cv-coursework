"""Multi-modal automated curation gate for synthetic deepfake image datasets."""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
from PIL import Image
import torch

from src.common.config import cfg_get, load_config
from src.common.logger import get_logger
from src.common.metrics import (
    ClipScorer,
    calibrate_threshold,
    compute_laplacian_sharpness,
    detect_faces,
    verify_facial_geometry,
)
from src.data.provenance import file_sha256

logger = get_logger(__name__)


@dataclass
class ImageCurationReport:
    """Detailed quality inspection report for a synthetic face image."""

    image_id: str
    is_valid: bool
    sharpness_score: float
    clip_score: Optional[float]
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
        min_sharpness: float = 1.25,
        min_clip_score: float = 0.28,
        clip_enabled: bool = False,
        clip_model_id: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        face_backend: str = "none",
        dedup: bool = False,
    ) -> None:
        self.min_sharpness = min_sharpness
        self.min_clip_score = min_clip_score
        self.clip_enabled = clip_enabled
        self.face_backend = face_backend
        self.dedup = dedup
        self._clip_scorer: Optional[ClipScorer] = None
        if self.clip_enabled:
            self._clip_scorer = ClipScorer(model_id=clip_model_id, device=device)

    def evaluate_image(
        self,
        image: torch.Tensor,
        prompt: str = "portrait",
        face_detected: Optional[bool] = None,
        image_id: str = "unknown",
        bbox_xyxy: Optional[Sequence[float]] = None,
    ) -> ImageCurationReport:
        """Evaluate a synthetic face image against sharpness, CLIP alignment, and face detection.

        Args:
            image: Tensor of shape (3, H, W) in [0.0, 1.0].
            prompt: Text prompt used for generation.
            face_detected: Boolean indicating if a face was detected, or None to detect automatically.
            image_id: String identifier.
            bbox_xyxy: Optional bounding box [x1, y1, x2, y2] to isolate facial ROI for sharpness.

        Returns:
            ImageCurationReport with validation decision.
        """
        sharpness = compute_laplacian_sharpness(image, bbox_xyxy=bbox_xyxy)


        # Detect face if not explicitly provided
        if face_detected is None:
            if self.face_backend == "none":
                face_detected_val = True
            else:
                img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                n_faces = detect_faces(img_np, backend=self.face_backend)
                face_detected_val = n_faces > 0
        else:
            face_detected_val = face_detected

        clip_score: Optional[float] = None
        if self.clip_enabled and self._clip_scorer is not None:
            clip_score = self._clip_scorer.score(image, prompt)

        reasons = []
        if not face_detected_val:
            reasons.append("No valid face landmark detected")
        elif self.face_backend != "none":
            geom_valid, geom_reasons = verify_facial_geometry(image)
            if not geom_valid:
                reasons.extend(geom_reasons)
        if sharpness < self.min_sharpness:
            reasons.append(f"Sharpness {sharpness:.2f} below threshold {self.min_sharpness:.2f}")
        if self.clip_enabled and clip_score is not None and clip_score < self.min_clip_score:
            reasons.append(f"CLIP score {clip_score:.3f} below threshold {self.min_clip_score:.3f}")

        is_valid = len(reasons) == 0
        return ImageCurationReport(
            image_id=image_id,
            is_valid=is_valid,
            sharpness_score=sharpness,
            clip_score=clip_score,
            face_detected=face_detected_val,
            rejection_reasons=reasons,
        )


def curate_image_manifest(
    input_manifest: Path,
    output_manifest: Path,
    gate: ImageCurationGate,
    curation_report: Optional[Path] = None,
) -> Dict[str, Any]:
    """Filter raw image manifest entries through the curation gate."""
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if curation_report is not None:
        curation_report.parent.mkdir(parents=True, exist_ok=True)

    kept_count = 0
    rejected_count = 0
    curated_lines = []
    report_lines = []
    seen_hashes: Set[str] = set()

    if input_manifest.exists():
        with open(input_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                file_path = item.get("file_path")
                img_id = item.get("image_id", "unknown")

                if not file_path or not Path(file_path).exists():
                    logger.warning("Rejecting %s: image file does not exist (%s)", img_id, file_path)
                    rejected_count += 1
                    report = ImageCurationReport(
                        image_id=img_id,
                        is_valid=False,
                        sharpness_score=0.0,
                        clip_score=None,
                        face_detected=False,
                        rejection_reasons=["missing_file"],
                    )
                    report_lines.append(json.dumps(report.to_dict()) + "\n")
                    continue

                # Dedup check
                if gate.dedup:
                    try:
                        f_hash = file_sha256(file_path)
                        if f_hash in seen_hashes:
                            logger.info("Rejecting %s: duplicate content hash %s", img_id, f_hash)
                            rejected_count += 1
                            report = ImageCurationReport(
                                image_id=img_id,
                                is_valid=False,
                                sharpness_score=0.0,
                                clip_score=None,
                                face_detected=False,
                                rejection_reasons=["duplicate_content"],
                            )
                            report_lines.append(json.dumps(report.to_dict()) + "\n")
                            continue
                        seen_hashes.add(f_hash)
                    except Exception as err:
                        logger.warning("Error computing hash for %s: %s", file_path, err)

                try:
                    with Image.open(file_path) as img:
                        np_img = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()

                    report = gate.evaluate_image(
                        tensor_img,
                        prompt=item.get("prompt", ""),
                        image_id=img_id,
                    )
                    report_lines.append(json.dumps(report.to_dict()) + "\n")

                    if report.is_valid:
                        curated_lines.append(json.dumps(item) + "\n")
                        kept_count += 1
                    else:
                        rejected_count += 1
                except Exception as err:
                    logger.warning("Error reading %s for curation (%s), skipping.", file_path, err)
                    rejected_count += 1
                    report = ImageCurationReport(
                        image_id=img_id,
                        is_valid=False,
                        sharpness_score=0.0,
                        clip_score=None,
                        face_detected=False,
                        rejection_reasons=[f"read_error: {err}"],
                    )
                    report_lines.append(json.dumps(report.to_dict()) + "\n")

    with open(output_manifest, "w", encoding="utf-8") as f:
        f.writelines(curated_lines)

    if curation_report is not None:
        with open(curation_report, "w", encoding="utf-8") as rf:
            rf.writelines(report_lines)

    summary = {
        "total": kept_count + rejected_count,
        "kept": kept_count,
        "rejected": rejected_count,
    }
    logger.info("Image curation complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate synthetic images using automated quality gate.")
    parser.add_argument("--config", type=str, default="configs/dataset_factory.yaml", help="Path to config file")
    parser.add_argument("--input-manifest", type=str, default=None)
    parser.add_argument("--output-manifest", type=str, default=None)
    parser.add_argument("--curation-report", type=str, default=None)
    parser.add_argument("--raw-manifest", type=str, default=None)
    parser.add_argument("--min-sharpness", type=float, default=None)
    parser.add_argument("--min-clip", type=float, default=None)
    parser.add_argument("--enable-clip", action="store_true", default=None)
    parser.add_argument("--disable-clip", action="store_true", default=None)
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        cfg = load_config(cfg_path)

    def _res(cli_val: Any, dotted_key: str, default_val: Any) -> Any:
        if cli_val is not None:
            return cli_val
        try:
            return cfg_get(cfg, dotted_key)
        except (KeyError, TypeError):
            return default_val

    in_manifest = Path(_res(args.input_manifest, "paths.fake_manifest", "data/synthetic/fake_manifest.jsonl"))
    out_manifest = Path(_res(args.output_manifest, "paths.curated_manifest", "data/synthetic/curated_manifest.jsonl"))
    curation_report = Path(_res(args.curation_report, "paths.curation_report", "data/synthetic/curation_report.jsonl"))
    raw_manifest = Path(_res(args.raw_manifest, "paths.raw_manifest", "data/raw/manifest.jsonl"))

    if not in_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {in_manifest}")

    # Clip configuration
    clip_enabled_default = bool(_res(None, "curation.clip.enabled", False))
    if args.enable_clip:
        clip_enabled = True
    elif args.disable_clip:
        clip_enabled = False
    else:
        clip_enabled = clip_enabled_default

    clip_model_id = str(_res(None, "curation.clip.model_id", "openai/clip-vit-base-patch32"))
    min_clip_score = float(_res(args.min_clip, "curation.clip.min_score", 0.15))
    face_backend = str(_res(None, "curation.face_check.backend", "none"))
    dedup = bool(_res(None, "curation.dedup", True))

    # Sharpness calibration against reference real images if available
    min_sharpness_val = args.min_sharpness
    if min_sharpness_val is None:
        ref_quantile = None
        try:
            ref_quantile = float(cfg_get(cfg, "curation.reference_quantile_sharpness"))
        except (KeyError, TypeError):
            pass

        if ref_quantile is not None and raw_manifest.exists():
            logger.info("Calibrating sharpness threshold from %s (quantile=%.2f)...", raw_manifest, ref_quantile)
            real_sharpness = []
            with open(raw_manifest, "r", encoding="utf-8") as rf:
                for line in rf:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    fp = item.get("file_path")
                    if fp and Path(fp).exists():
                        try:
                            with Image.open(fp) as img:
                                np_img = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                                tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
                            real_sharpness.append(compute_laplacian_sharpness(tensor_img))
                            if len(real_sharpness) >= 50:
                                break
                        except Exception:
                            pass
            if real_sharpness:
                min_sharpness_val = calibrate_threshold(real_sharpness, ref_quantile)
                logger.info("Calibrated min_sharpness: %.2f", min_sharpness_val)

        if min_sharpness_val is None:
            min_sharpness_val = 10.0

    gate = ImageCurationGate(
        min_sharpness=min_sharpness_val,
        min_clip_score=min_clip_score,
        clip_enabled=clip_enabled,
        clip_model_id=clip_model_id,
        face_backend=face_backend,
        dedup=dedup,
    )
    curate_image_manifest(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        gate=gate,
        curation_report=curation_report,
    )


def curate_dataset(
    input_manifest: Path,
    output_manifest: Path,
    report_path: Optional[Path] = None,
    curation_report: Optional[Path] = None,
    gate: Optional[ImageCurationGate] = None,
    min_sharpness: float = 1.25,
    face_backend: str = "opencv_haar",
    dedup: bool = True,
) -> Dict[str, Any]:
    """High-level wrapper to curate a synthetic dataset manifest."""
    if gate is None:
        gate = ImageCurationGate(
            min_sharpness=min_sharpness,
            face_backend=face_backend,
            clip_enabled=False,
            dedup=dedup,
        )
    rep = report_path or curation_report
    return curate_image_manifest(
        input_manifest=Path(input_manifest),
        output_manifest=Path(output_manifest),
        gate=gate,
        curation_report=Path(rep) if rep else None,
    )


if __name__ == "__main__":
    main()
