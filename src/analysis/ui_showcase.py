"""Visual overlay renderers, threshold recalibration, and DINOv2 patch heatmap helpers for UI showcase."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass, field
import gc
import html
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from src.common.config import load_yaml_with_defaults
from src.common.logger import get_logger
import src.common.metrics as _metrics_mod
from src.common.metrics import (
    ClipScorer,
    audit_image_extended,
    compute_cheek_skin_fft,
    compute_fft_high_freq_ratio,
    detect_face_yunet,
    extract_face_biometrics_rgb,
    resolve_effective_min_clip,
)
from src.common.seed import seed_everything
from src.models.detector.image_detector import DeepfakeImageDetector
from src.models.diffusion.prompt_engine import PromptGrammarEngine
from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask

logger = get_logger(__name__)

_ORIG_CLIP_SCORER = ClipScorer
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
_MIN_CUDA_FREE_BYTES: int = 900 * 1024 * 1024
_CANONICAL_MODE_MAP: Dict[str, str] = {
    "text2img": "sd_text2img",
    "img2img": "sd_img2img",
    "inpaint": "sd_inpaint",
}
_FULL_MODEL_LABELS: Dict[str, str] = {
    "M0": "M0 (Baseline)",
    "M2*": "M2* (Active R2)",
    "M3": "M3 (Champion)",
}
DEFAULT_CHECKPOINT_MAP: Dict[str, str] = {
    "M0": "experiments/checkpoints/baseline_d0_best.pt",
    "M2*": "experiments/checkpoints/active_learning_m2_best.pt",
    "M3": "experiments/checkpoints/active_learning_m3_best.pt",
}


@dataclass
class CurationMetricItem:
    """Single data-curation audit metric with threshold and pass/fail state."""

    name: str
    label: str
    value: Optional[float]
    threshold: Optional[float]
    passed: bool
    unit: str = ""
    direction: str = ">="
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metric item to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class ShowcaseCatalogEntry:
    """Curated or held-out showcase sample metadata with precomputed curation and detector scores."""

    sample_id: str
    mode: str
    tab: str
    image_path: str
    source_image_path: Optional[str] = None
    mask_path: Optional[str] = None
    mask_region: Optional[str] = None
    prompt: str = ""
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    strength: Optional[float] = None
    label: int = 1
    split_provenance: str = ""
    gate_passed: bool = True
    rejection_reasons: List[str] = field(default_factory=list)
    measured: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    cheek_fft_ratio: Optional[float] = None
    is_hard_sample: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize catalog entry to a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class DetectorBundle:
    """Loaded checkpoint ensemble (M0, M2*, M3) and runtime device metadata."""

    models: Dict[str, torch.nn.Module]
    device: torch.device
    checkpoint_paths: Dict[str, str] = field(default_factory=dict)
    backbone: str = "dinov2_small"
    free_vram_mb: Optional[float] = None


@dataclass
class InspectionResult:
    """Full 4-stage inspection payload returned for live UI display and testing."""

    mode: str
    image_512_rgb: np.ndarray
    source_512_rgb: Optional[np.ndarray] = None
    mask_512: Optional[np.ndarray] = None
    gate_passed: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    rejection_reasons_vi: List[str] = field(default_factory=list)
    metrics: List[CurationMetricItem] = field(default_factory=list)
    measured: Dict[str, Any] = field(default_factory=dict)
    yunet_face: Optional[Dict[str, Any]] = None
    cheek_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    biometrics: Dict[str, Any] = field(default_factory=dict)
    detector_scores: Dict[str, float] = field(default_factory=dict)
    is_hard_sample: bool = False
    ground_truth_label: Optional[int] = None
    overlays: Dict[str, np.ndarray] = field(default_factory=dict)
    heatmaps: Dict[str, np.ndarray] = field(default_factory=dict)
    before_after_html: Optional[str] = None
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an image array to contiguous (H, W, 3) uint8 RGB."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image array, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        if arr_f.size > 0 and float(np.max(arr_f)) <= 1.0 + 1e-3 and float(np.min(arr_f)) >= 0.0:
            arr_f = arr_f * 255.0
        arr = np.clip(np.round(arr_f), 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(arr)


def _to_512_rgb_u8(image: Union[Path, str, Image.Image, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Load/normalize any Path, PIL image, Tensor, or ndarray to (512, 512, 3) uint8 RGB via LANCZOS."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as raw_im:
            pil_rgb = (
                ImageOps.exif_transpose(raw_im)
                .convert("RGB")
                .resize((512, 512), Image.Resampling.LANCZOS)
            )
            return np.ascontiguousarray(np.asarray(pil_rgb, dtype=np.uint8))

    if isinstance(image, Image.Image):
        pil_rgb = (
            ImageOps.exif_transpose(image)
            .convert("RGB")
            .resize((512, 512), Image.Resampling.LANCZOS)
        )
        return np.ascontiguousarray(np.asarray(pil_rgb, dtype=np.uint8))

    if isinstance(image, torch.Tensor):
        rgb = _metrics_mod._to_rgb_uint8(image)
    else:
        rgb = _to_rgb_uint8(image)

    if rgb.shape[:2] == (512, 512):
        return rgb
    pil_resized = Image.fromarray(rgb, mode="RGB").resize((512, 512), Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(pil_resized, dtype=np.uint8))


def _normalize_mask_2d(mask: Any, width: int, height: int) -> Optional[np.ndarray]:
    """Convert mask of shape (H, W), (1, H, W), or (H, W, C) into float32 (height, width) in [0, 1]."""
    arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if arr.size == 0:
        return None
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0] if (arr.shape[0] <= 3 and arr.shape[2] > 3) else arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D/3D mask array, got shape {arr.shape}")

    m_f32 = arr.astype(np.float32)
    if float(np.max(m_f32)) > 1.0 + 1e-3:
        m_f32 = m_f32 / 255.0
    m_f32 = np.clip(m_f32, 0.0, 1.0)
    if m_f32.shape != (height, width):
        m_f32 = cv2.resize(m_f32, (width, height), interpolation=cv2.INTER_NEAREST)
    return m_f32


def compute_cheek_roi_boxes(
    yunet_face: Optional[Dict[str, Any]],
    width: int = 512,
    height: int = 512,
) -> List[Tuple[int, int, int, int]]:
    """Compute left/right cheek ROI boxes matching `src.common.metrics.compute_cheek_skin_fft`.

    Args:
        yunet_face: Dictionary returned by `detect_face_yunet` containing `bbox_xyxy` and `landmarks`.
        width: Target image width in pixels.
        height: Target image height in pixels.

    Returns:
        List of `(x1, y1, x2, y2)` pixel boxes for valid cheek skin patches (>= 14x14).
    """
    if yunet_face is None or "landmarks" not in yunet_face or "bbox_xyxy" not in yunet_face:
        return []

    lms = yunet_face["landmarks"]
    required_keys = ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")
    if any(k not in lms for k in required_keys):
        return []

    x1, y1, x2, y2 = yunet_face["bbox_xyxy"]
    fw = max(1.0, float(x2) - float(x1))
    fh = max(1.0, float(y2) - float(y1))

    re_y = float(lms["right_eye"][1])
    le_y = float(lms["left_eye"][1])
    nose_x = float(lms["nose"][0])
    rm_y = float(lms["mouth_right"][1])
    lm_y = float(lms["mouth_left"][1])

    cy1 = max(0, min(height, int(round(max(re_y, le_y) + 0.12 * fh))))
    cy2 = max(0, min(height, int(round(max(rm_y, lm_y) - 0.04 * fh))))
    if cy2 - cy1 < 14:
        return []

    def _clip_w(val: float) -> int:
        return max(0, min(width, int(round(val))))

    x_spans = [
        (_clip_w(float(x1) + 0.08 * fw), _clip_w(nose_x - 0.06 * fw)),
        (_clip_w(nose_x + 0.06 * fw), _clip_w(float(x1) + 0.92 * fw)),
    ]

    boxes: List[Tuple[int, int, int, int]] = []
    for bx1, bx2 in x_spans:
        if bx2 - bx1 >= 14:
            boxes.append((bx1, cy1, bx2, cy2))
    return boxes


def render_face_audit_overlay(
    image_rgb: np.ndarray,
    *,
    yunet_face: Optional[Dict[str, Any]],
    passed: bool,
    cheek_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render YuNet face box, 5 landmarks, cheek ROI patches, and optional inpaint mask onto RGB copy.

    Args:
        image_rgb: Input RGB image of shape (H, W, 3).
        yunet_face: Optional YuNet detection dictionary (`bbox_xyxy`, `landmarks`).
        passed: Gate verdict controlling green (True) vs red (False) bounding box color.
        cheek_boxes: Optional precomputed `(x1, y1, x2, y2)` cheek ROI boxes.
        mask: Optional inpaint mask of shape `(H, W)` or `(1, H, W)` in `[0, 1]` or `[0, 255]`.

    Returns:
        Annotated `(H, W, 3)` `uint8` RGB image.
    """
    rgb = _to_rgb_uint8(image_rgb)
    h, w = rgb.shape[:2]

    # Explicit RGB -> BGR conversion for all OpenCV drawing calls
    canvas_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if mask is not None:
        m_2d = _normalize_mask_2d(mask, width=w, height=h)
        if m_2d is not None:
            active = m_2d > 0.5
            if np.any(active):
                # Semi-transparent violet tint: RGB (168, 85, 247) -> BGR (247, 85, 168)
                tint_bgr = np.array([247.0, 85.0, 168.0], dtype=np.float32)
                blended = canvas_bgr[active].astype(np.float32) * 0.72 + tint_bgr * 0.28
                canvas_bgr[active] = np.clip(np.round(blended), 0.0, 255.0).astype(np.uint8)
                contours, _ = cv2.findContours(
                    active.astype(np.uint8) * 255,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(canvas_bgr, contours, -1, (247, 85, 168), thickness=2)

    # Status color: Green RGB (30, 220, 80) -> BGR (80, 220, 30); Red RGB (240, 40, 40) -> BGR (40, 40, 240)
    status_bgr = (80, 220, 30) if passed else (40, 40, 240)

    if yunet_face is not None and "bbox_xyxy" in yunet_face:
        bx1, by1, bx2, by2 = [int(round(v)) for v in yunet_face["bbox_xyxy"]]
        bx1 = max(0, min(w - 1, bx1))
        by1 = max(0, min(h - 1, by1))
        bx2 = max(0, min(w - 1, bx2))
        by2 = max(0, min(h - 1, by2))
        cv2.rectangle(canvas_bgr, (bx1, by1), (bx2, by2), status_bgr, thickness=3)

        active_cheeks = (
            list(cheek_boxes)
            if cheek_boxes is not None
            else compute_cheek_roi_boxes(yunet_face, width=w, height=h)
        )
        # Cyan-sky cheek ROI box: RGB (40, 190, 250) -> BGR (250, 190, 40)
        for cx1, cy1, cx2, cy2 in active_cheeks:
            cv2.rectangle(
                canvas_bgr,
                (max(0, min(w - 1, int(cx1))), max(0, min(h - 1, int(cy1)))),
                (max(0, min(w - 1, int(cx2))), max(0, min(h - 1, int(cy2)))),
                (250, 190, 40),
                thickness=2,
            )

        lms = yunet_face.get("landmarks", {})
        # Explicit RGB -> BGR landmark palette:
        # Eyes: RGB (0, 100, 255) -> BGR (255, 100, 0)
        # Nose: RGB (255, 220, 0) -> BGR (0, 220, 255)
        # Mouth: RGB (255, 40, 40) -> BGR (40, 40, 255)
        lm_colors_bgr = {
            "right_eye": (255, 100, 0),
            "left_eye": (255, 100, 0),
            "nose": (0, 220, 255),
            "mouth_right": (40, 40, 255),
            "mouth_left": (40, 40, 255),
        }
        for key, color_bgr in lm_colors_bgr.items():
            if key in lms:
                px = max(0, min(w - 1, int(round(float(lms[key][0])))))
                py = max(0, min(h - 1, int(round(float(lms[key][1])))))
                cv2.circle(canvas_bgr, (px, py), radius=5, color=color_bgr, thickness=-1)
    else:
        cv2.rectangle(canvas_bgr, (1, 1), (max(1, w - 2), max(1, h - 2)), status_bgr, thickness=3)

    # Explicit BGR -> RGB conversion before returning
    return np.ascontiguousarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))


def render_latent_noise_preview(seed: int, size: int = 256) -> np.ndarray:
    """Render a deterministic RGB preview of the SD 1.5 Gaussian latent tensor for `seed`.

    Args:
        seed: Random seed initializing the CPU torch generator.
        size: Output square side length in pixels.

    Returns:
        (size, size, 3) uint8 RGB image representing the 64x64 latent noise map.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    latent = torch.randn((1, 3, 64, 64), generator=gen, dtype=torch.float32)
    norm = torch.clamp((latent + 2.8) / 5.6, 0.0, 1.0)
    upsampled = F.interpolate(norm, size=(size, size), mode="nearest")
    rgb_u8 = (upsampled[0].permute(1, 2, 0).numpy() * 255.0).round().clip(0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb_u8)


def _encode_rgb_jpeg_base64(image_rgb: np.ndarray, quality: int = 90) -> str:
    """Encode a (H, W, 3) RGB uint8 image as a base64 JPEG string."""
    rgb = _to_rgb_uint8(image_rgb)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_before_after_wipe_html(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    left_label: str = "INPUT FFHQ",
    right_label: str = "OUTPUT",
    height_px: int = 320,
) -> str:
    """Build a self-contained HTML/CSS/JS interactive before/after wipe slider component.

    Args:
        before_rgb: Reference/source RGB image of shape (H, W, 3).
        after_rgb: Generated/processed RGB image of shape (H, W, 3).
        left_label: Label shown over the left (before) region.
        right_label: Label shown over the right (after) region.
        height_px: Rendered height in pixels.

    Returns:
        Self-contained HTML string embedding two base64 JPEG images and `clip-path` wipe slider.
    """
    after_u8 = _to_rgb_uint8(after_rgb)
    before_u8 = _to_rgb_uint8(before_rgb)
    if before_u8.shape[:2] != after_u8.shape[:2]:
        ah, aw = after_u8.shape[:2]
        before_u8 = np.asarray(
            Image.fromarray(before_u8, mode="RGB").resize((aw, ah), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )

    b64_before = _encode_rgb_jpeg_base64(before_u8)
    b64_after = _encode_rgb_jpeg_base64(after_u8)
    safe_left = html.escape(left_label)
    safe_right = html.escape(right_label)
    uid = uuid.uuid4().hex[:12]

    return f"""<div id="wipe-root-{uid}" style="position:relative;width:100%;max-width:{height_px}px;height:{height_px}px;margin:0 auto;overflow:hidden;border-radius:10px;border:1px solid rgba(148,163,184,0.35);user-select:none;background:#0f172a;font-family:system-ui,-apple-system,sans-serif;">
  <img src="data:image/jpeg;base64,{b64_after}" alt="{safe_right}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;pointer-events:none;" />
  <img id="wipe-before-{uid}" src="data:image/jpeg;base64,{b64_before}" alt="{safe_left}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;pointer-events:none;clip-path:inset(0 50% 0 0);" />
  <div id="wipe-line-{uid}" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#ffffff;box-shadow:0 0 6px rgba(0,0,0,0.65);pointer-events:none;">
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:24px;height:24px;border-radius:999px;background:#ffffff;color:#0f172a;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 8px rgba(0,0,0,0.45);">&#8596;</div>
  </div>
  <span style="position:absolute;top:8px;left:8px;padding:2px 7px;border-radius:4px;background:rgba(15,23,42,0.75);color:#f8fafc;font-size:11px;font-weight:600;letter-spacing:0.03em;pointer-events:none;">{safe_left}</span>
  <span style="position:absolute;top:8px;right:8px;padding:2px 7px;border-radius:4px;background:rgba(15,23,42,0.75);color:#f8fafc;font-size:11px;font-weight:600;letter-spacing:0.03em;pointer-events:none;">{safe_right}</span>
  <input type="range" min="0" max="100" value="50" aria-label="Before after comparison slider"
    oninput="(function(el){{var v=el.value;document.getElementById('wipe-before-{uid}').style.clipPath='inset(0 '+(100-v)+'% 0 0)';document.getElementById('wipe-line-{uid}').style.left=v+'%';}})(this)"
    style="position:absolute;inset:0;width:100%;height:100%;margin:0;opacity:0;cursor:ew-resize;z-index:5;" />
</div>"""


def apply_interactive_thresholds(
    measured: Dict[str, Any],
    *,
    mode: str,
    min_sharpness: float,
    min_cheek_fft: float,
    min_clip: float,
    max_delta_e: float = 18.0,
    max_seam: float = 35.0,
) -> Tuple[bool, List[str]]:
    """Re-evaluate measured curation metrics against interactive UI threshold sliders.

    Args:
        measured: Dictionary of pre-measured or live-measured sample metrics.
        mode: Active generation/source mode (e.g. `sd_text2img`, `sd_img2img`, `sd_inpaint`, `rv51_ood`, `real_ffhq`).
        min_sharpness: Minimum Laplacian sharpness variance.
        min_cheek_fft: Minimum localized cheek-skin 2D FFT high-frequency ratio.
        min_clip: Minimum CLIP text-image similarity applied directly to `mode`.
        max_delta_e: Maximum CIELab delta-E drift for paired modes.
        max_seam: Maximum boundary seam gradient difference for masked inpaint mode.

    Returns:
        Tuple `(passed, rejection_reasons)`.
    """
    reasons: List[str] = []

    if measured.get("face_detected") is False:
        reasons.append("No valid face detected by YuNet")

    sharpness = measured.get("sharpness_score")
    if sharpness is not None and float(sharpness) < float(min_sharpness):
        reasons.append(f"Sharpness {float(sharpness):.2f} below {float(min_sharpness):.2f}")

    if measured.get("geometry_valid") is False:
        geom_reasons = measured.get("geometry_reasons")
        if isinstance(geom_reasons, (list, tuple)) and geom_reasons:
            reasons.extend(str(r) for r in geom_reasons)
        else:
            reasons.append("Facial geometry audit failed")

    if measured.get("eye_passed") is False:
        eye_reasons = measured.get("eye_reasons")
        if isinstance(eye_reasons, (list, tuple)) and eye_reasons:
            reasons.extend(str(r) for r in eye_reasons)
        else:
            reasons.append("Eye-pair verification failed")

    texture_flag = measured.get("texture_flag")
    whole_fft = measured.get("fft_high_freq_ratio")
    if texture_flag in ("waxy_low_freq", "crunchy_high_freq"):
        if whole_fft is not None:
            reasons.append(f"FFT texture flag {texture_flag} ({float(whole_fft):.4f})")
        else:
            reasons.append(f"FFT texture flag {texture_flag}")
    elif whole_fft is not None and (float(whole_fft) < 0.0025 or float(whole_fft) > 0.65):
        reasons.append(f"Whole-face FFT ratio {float(whole_fft):.4f} out of [0.0025, 0.6500]")

    cheek_fft = measured.get("cheek_fft_ratio")
    if cheek_fft is not None and float(cheek_fft) < float(min_cheek_fft):
        reasons.append(
            f"Cheek skin FFT {float(cheek_fft):.4f} below {float(min_cheek_fft):.4f} (waxy/plastic skin)"
        )

    clip_score = measured.get("clip_score")
    if clip_score is not None and float(clip_score) < float(min_clip):
        reasons.append(f"CLIP score {float(clip_score):.3f} below {float(min_clip):.3f}")

    # Paired color/structure/seam checks: skip cleanly when metrics are None (e.g. rv51_ood / sd_text2img)
    _ = mode  # Mode retained in signature for explicit caller context & future mode-specific extensions
    delta_e = measured.get("delta_e_lab")
    if delta_e is not None and float(delta_e) > float(max_delta_e):
        reasons.append(f"CIELab color drift {float(delta_e):.2f} exceeds {float(max_delta_e):.2f}")

    bg_delta_e = measured.get("bg_delta_e")
    if bg_delta_e is not None and float(bg_delta_e) > 12.0:
        reasons.append(f"Background CIELab drift {float(bg_delta_e):.2f} exceeds 12.00")

    structure_corr = measured.get("structure_corr")
    if structure_corr is not None and float(structure_corr) < 0.35:
        reasons.append(f"Structure correlation {float(structure_corr):.3f} below 0.35")

    seam_diff = measured.get("seam_diff")
    if seam_diff is not None and float(seam_diff) > float(max_seam):
        reasons.append(f"Boundary seam discontinuity {float(seam_diff):.2f} exceeds {float(max_seam):.2f}")

    return len(reasons) == 0, reasons


def compute_dinov2_patch_heatmap(
    detector: torch.nn.Module,
    image_512_rgb: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """Compute DINOv2 spatial patch sensitivity heatmap (`|d logit / d patch_tokens|`).

    Resizes the input RGB image to `(512, 512)` via `LANCZOS`, normalizes and interpolates to
    `(1, 3, 504, 504)` (`36x36` patches of `14x14`), runs a single gradient-enabled pass over the
    penultimate `(1, 1297, embed_dim)` token sequence feeding the final ViT block and classifier,
    and blends a `COLORMAP_JET` overlay over the original RGB image.

    Args:
        detector: `DeepfakeImageDetector` (or compatible `torch.nn.Module`).
        image_512_rgb: Input RGB image `(H, W, 3)` `uint8`.
        device: Target torch device for forward/backward computation.

    Returns:
        Dict containing `"prob"` (`float`), `"heatmap_rgb"` (`(512, 512, 3)` `uint8`),
        and `"raw_map"` (`(512, 512)` `float32` in `[0.0, 1.0]`).
    """
    rgb_512 = _to_512_rgb_u8(image_512_rgb)

    tensor_512 = (
        torch.from_numpy(rgb_512.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )  # (1, 3, 512, 512)
    tensor_504 = F.interpolate(
        tensor_512,
        size=(504, 504),
        mode="bilinear",
        align_corners=False,
    )  # (1, 3, 504, 504)

    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    detector.eval()
    try:
        with torch.enable_grad():
            extractor = getattr(detector, "extractor", None)
            backbone = getattr(extractor, "backbone", None) if extractor is not None else None
            classifier = getattr(detector, "classifier", None)

            if backbone is not None and classifier is not None:
                x_norm = ((tensor_504 - mean) / std).detach().requires_grad_(True)
                outputs = backbone(pixel_values=x_norm, output_hidden_states=True)
                last_hidden = outputs.last_hidden_state  # (1, 1297, embed_dim)
                cls_out = F.normalize(last_hidden[:, 0], p=2, dim=-1)
                logits = classifier(cls_out)  # (1, num_classes)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)
                target_logit = logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]
                prob = float(
                    torch.softmax(logits, dim=-1)[:, 1].item()
                    if logits.shape[1] > 1
                    else torch.sigmoid(logits[:, 0]).item()
                )

                patch_sal: Optional[torch.Tensor] = None
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None and len(hidden_states) >= 2:
                    # Penultimate hidden state (1, 1297, embed_dim) directly feeds the final block -> CLS logit
                    penultimate = hidden_states[-2]
                    grad_tokens = torch.autograd.grad(
                        target_logit.sum(),
                        penultimate,
                        retain_graph=False,
                        allow_unused=True,
                    )[0]
                    if grad_tokens is not None and grad_tokens.shape[1] == 1 + 36 * 36:
                        patch_tokens_grad = grad_tokens[:, 1:, :]  # (1, 1296, embed_dim)
                        patch_sal = torch.linalg.vector_norm(
                            patch_tokens_grad, ord=2, dim=-1
                        ).reshape(1, 1, 36, 36)

                if patch_sal is None:
                    grad_pix = torch.autograd.grad(
                        target_logit.sum(),
                        x_norm,
                        retain_graph=False,
                        allow_unused=True,
                    )[0]
                    if grad_pix is not None:
                        pix_norm = torch.linalg.vector_norm(grad_pix, ord=2, dim=1, keepdim=True)
                        patch_sal = F.avg_pool2d(pix_norm, kernel_size=14, stride=14)
            else:
                x_in = tensor_504.detach().requires_grad_(True)
                logits = detector(x_in)
                target_logit = logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]
                prob = float(
                    torch.softmax(logits, dim=-1)[:, 1].item()
                    if logits.shape[1] > 1
                    else torch.sigmoid(logits[:, 0]).item()
                )
                grad_pix = torch.autograd.grad(
                    target_logit.sum(),
                    x_in,
                    retain_graph=False,
                    allow_unused=True,
                )[0]
                if grad_pix is not None:
                    pix_norm = torch.linalg.vector_norm(grad_pix, ord=2, dim=1, keepdim=True)
                    patch_sal = F.avg_pool2d(pix_norm, kernel_size=14, stride=14)
                else:
                    patch_sal = None

            if patch_sal is None:
                logger.warning("Gradient patch saliency was None; returning zero sensitivity map.")
                raw_map = np.zeros((512, 512), dtype=np.float32)
            else:
                sal_512 = F.interpolate(
                    patch_sal.detach().float(),
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                sal_np = sal_512.cpu().numpy().astype(np.float32)
                s_min = float(np.min(sal_np))
                s_max = float(np.max(sal_np))
                if s_max - s_min > 1e-12:
                    raw_map = np.clip((sal_np - s_min) / (s_max - s_min), 0.0, 1.0).astype(np.float32)
                else:
                    raw_map = np.zeros((512, 512), dtype=np.float32)
    finally:
        detector.zero_grad(set_to_none=True)
        detector.eval()

    heat_u8 = np.clip(np.round(raw_map * 255.0), 0.0, 255.0).astype(np.uint8)
    jet_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.clip(
        np.round(0.55 * rgb_512.astype(np.float32) + 0.45 * jet_rgb.astype(np.float32)),
        0.0,
        255.0,
    ).astype(np.uint8)

    return {
        "prob": float(prob),
        "heatmap_rgb": np.ascontiguousarray(overlay),
        "raw_map": raw_map,
    }


def _resolve_detector_device(preferred_device: str) -> Tuple[torch.device, Optional[float]]:
    """Resolve detector device with 900 MiB CUDA free-VRAM guard to protect 4GB GPUs."""
    free_vram_mb: Optional[float] = None
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_vram_mb = float(free_bytes) / (1024.0 * 1024.0)
        except Exception as exc:
            logger.warning("Failed to query CUDA free VRAM (%s); falling back to CPU.", exc)
            return torch.device("cpu"), None

        if preferred_device == "auto":
            if free_bytes >= _MIN_CUDA_FREE_BYTES:
                return torch.device("cuda"), free_vram_mb
            logger.warning(
                "CUDA free VRAM %.1f MiB < %.1f MiB guard; loading detector bundle on CPU.",
                free_vram_mb,
                _MIN_CUDA_FREE_BYTES / (1024.0 * 1024.0),
            )
            return torch.device("cpu"), free_vram_mb

        if preferred_device.startswith("cuda"):
            if free_bytes >= _MIN_CUDA_FREE_BYTES:
                return torch.device(preferred_device), free_vram_mb
            logger.warning(
                "Requested CUDA device with only %.1f MiB free VRAM; falling back to CPU.",
                free_vram_mb,
            )
            return torch.device("cpu"), free_vram_mb

    if preferred_device.startswith("cuda"):
        logger.warning("CUDA requested (%s) but unavailable; falling back to CPU.", preferred_device)
        return torch.device("cpu"), None

    return torch.device("cpu" if preferred_device == "auto" else preferred_device), free_vram_mb


def load_detector_bundle(
    config_path: Union[str, Path] = "configs/config.yaml",
    checkpoint_map: Optional[Dict[str, str]] = None,
    preferred_device: str = "auto",
) -> DetectorBundle:
    """Load real M0, M2*, and M3 detector checkpoints with strict state verification and VRAM guard.

    Args:
        config_path: Master YAML config defining `detection.model.spatial_backbone`.
        checkpoint_map: Optional mapping `{display_name: checkpoint_path}`. Defaults to `M0`, `M2*`, `M3`.
        preferred_device: Device preference (`"auto"`, `"cuda"`, or `"cpu"`).

    Returns:
        Populated `DetectorBundle` ready for `score_image_with_bundle()`.
    """
    cfg_file = Path(config_path)
    cfg = load_yaml_with_defaults(cfg_file) if cfg_file.is_file() else {}
    det_model_cfg = cfg.get("detection", {}).get("model", {})
    backbone = str(det_model_cfg.get("spatial_backbone", "dinov2_small"))
    num_classes = int(det_model_cfg.get("num_classes", 1))

    resolved_map = dict(checkpoint_map) if checkpoint_map is not None else dict(DEFAULT_CHECKPOINT_MAP)
    if not resolved_map:
        raise ValueError("checkpoint_map cannot be empty.")

    device, free_vram_mb = _resolve_detector_device(preferred_device)
    models: Dict[str, torch.nn.Module] = {}
    ckpt_paths: Dict[str, str] = {}

    try:
        for label, ckpt_str in resolved_map.items():
            ckpt_file = Path(ckpt_str)
            if not ckpt_file.is_file():
                raise FileNotFoundError(f"Verified checkpoint for {label} not found at: {ckpt_file}")

            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
                raise ValueError(
                    f"Checkpoint {ckpt_file} for {label} is missing required 'model_state_dict'."
                )
            state_dict = ckpt["model_state_dict"]

            model = DeepfakeImageDetector(backbone=backbone, num_classes=num_classes, pretrained=False)
            head_keys = [k for k in model.state_dict().keys() if k.startswith("classifier") or k.startswith("head")]
            missing_head = [k for k in head_keys if k not in state_dict]
            if missing_head:
                raise ValueError(
                    f"Checkpoint {ckpt_file} for {label} is missing classifier head keys: {missing_head}"
                )

            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            models[label] = model
            ckpt_paths[label] = str(ckpt_file)
    except Exception:
        for loaded in list(models.values()):
            try:
                loaded.to("cpu")
            except Exception:
                pass
            del loaded
        models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    logger.info(
        "Loaded detector bundle (%s) on device=%s (free_vram_mb=%s)",
        list(models.keys()),
        device,
        f"{free_vram_mb:.1f}" if free_vram_mb is not None else "N/A",
    )
    return DetectorBundle(
        models=models,
        device=device,
        checkpoint_paths=ckpt_paths,
        backbone=backbone,
        free_vram_mb=free_vram_mb,
    )


def unload_detector_bundle(bundle: Optional[DetectorBundle]) -> None:
    """Release loaded detector models and reclaim GPU/CPU memory prior to running SD worker."""
    if bundle is None:
        return
    try:
        for model in list(bundle.models.values()):
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        bundle.models.clear()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_image_with_bundle(bundle: DetectorBundle, image_512_rgb: np.ndarray) -> Dict[str, float]:
    """Run eval-mode inference across all checkpoints in `bundle` on a `(512, 512, 3)` RGB image.

    Args:
        bundle: Loaded `DetectorBundle`.
        image_512_rgb: Input image (normalized via LANCZOS to `(512, 512, 3)` uint8 RGB).

    Returns:
        Dictionary mapping checkpoint label (e.g. `"M0"`, `"M2*"`, `"M3"`) to `P(fake)` in `[0.0, 1.0]`.
    """
    rgb_512 = _to_512_rgb_u8(image_512_rgb)
    # Input tensor shape: (B=1, C=3, H=512, W=512) float32 RGB in [0.0, 1.0]
    tensor = (
        torch.from_numpy(rgb_512.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(bundle.device)
    )
    use_amp = bundle.device.type == "cuda"
    scores: Dict[str, float] = {}
    with torch.no_grad():
        for label, model in bundle.models.items():
            model.eval()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(tensor)
                if logits.ndim == 2 and logits.shape[1] > 1:
                    prob = float(torch.softmax(logits.float(), dim=-1)[0, 1].item())
                else:
                    prob = float(torch.sigmoid(logits.float().view(-1)[0]).item())
            scores[label] = float(np.clip(prob, 0.0, 1.0))
    return scores


def _resolve_inpaint_mask_512(
    rgb_512: np.ndarray,
    source_512: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    mask_region: Optional[str],
    mode: str,
) -> Optional[np.ndarray]:
    """Normalize explicit inpaint mask or reconstruct anatomical soft mask for `sd_inpaint`."""
    if mask is not None:
        return _normalize_mask_2d(mask, width=512, height=512)

    if mode != "sd_inpaint" or source_512 is None:
        return None

    src_tensor = torch.from_numpy(source_512.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    region_alias = {"mouth": "lips", "eyes": "periocular"}
    if mask_region:
        target_reg = region_alias.get(str(mask_region), str(mask_region))
        mask_t = generate_dynamic_face_mask(src_tensor, bbox=None, blur_radius=18, target_region=target_reg)
        return mask_t[0].detach().cpu().numpy().astype(np.float32)

    diff_mask = np.mean(np.abs(rgb_512.astype(np.float32) - source_512.astype(np.float32)), axis=2) > 15.0
    if np.any(diff_mask):
        best_iou = -1.0
        best_mask: Optional[torch.Tensor] = None
        for cand_reg in ("full", "periocular", "lips"):
            cand_m = generate_dynamic_face_mask(src_tensor, bbox=None, blur_radius=18, target_region=cand_reg)
            cand_bin = cand_m[0].numpy() > 0.5
            inter = float(np.sum(cand_bin & diff_mask))
            union = float(np.sum(cand_bin | diff_mask)) + 1e-6
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_mask = cand_m
        if best_mask is not None:
            return best_mask[0].detach().cpu().numpy().astype(np.float32)

    fallback_m = generate_dynamic_face_mask(src_tensor, bbox=None, blur_radius=18, target_region="full")
    return fallback_m[0].detach().cpu().numpy().astype(np.float32)


def _translate_rejection_reasons_vi(reasons: Sequence[str]) -> List[str]:
    """Translate technical English curation rejection reasons into concise Vietnamese explanations."""
    translations: List[str] = []
    for reason in reasons:
        low = reason.lower()
        if "no valid face" in low or "no face landmark" in low:
            translations.append(f"Không phát hiện được khuôn mặt hợp lệ (YuNet): {reason}")
        elif "multiple faces" in low:
            translations.append(f"Phát hiện nhiều khuôn mặt (yêu cầu ảnh đơn chân dung): {reason}")
        elif "sharpness" in low:
            translations.append(f"Độ sắc nét Laplacian thấp hơn ngưỡng: {reason}")
        elif "cheek skin fft" in low or "waxy" in low:
            translations.append(f"Phổ FFT vùng da má quá mịn (hiệu ứng da sáp/nhựa): {reason}")
        elif "fft texture" in low or "whole-face fft" in low:
            translations.append(f"Tần số không gian FFT toàn mặt bất thường: {reason}")
        elif "clip score" in low:
            translations.append(f"Độ khớp ngữ nghĩa CLIP với prompt thấp hơn ngưỡng: {reason}")
        elif "cielab" in low:
            translations.append(f"Độ lệch màu CIELab ΔE vượt ngưỡng cho phép: {reason}")
        elif "boundary seam" in low:
            translations.append(f"Viền ghép mặt nạ (Boundary Seam) bị lộ rõ: {reason}")
        elif "structure correlation" in low:
            translations.append(f"Tương quan cấu trúc khuôn mặt so với ảnh gốc bị lệch: {reason}")
        elif "symmetry" in low:
            translations.append(f"Độ đối xứng khuôn mặt hai bên thấp: {reason}")
        elif "eyes" in low:
            translations.append(f"Không xác minh đủ 2 mắt tự nhiên: {reason}")
        elif "highlight blowout" in low:
            translations.append(f"Cháy sáng cục bộ trên vùng mặt: {reason}")
        elif "amputated" in low:
            translations.append(f"Trán hoặc cằm bị cắt sát mép khung hình: {reason}")
        elif "aspect ratio" in low or "occupies too" in low or "off-center" in low:
            translations.append(f"Bố cục/tỷ lệ khuôn mặt bất thường: {reason}")
        else:
            translations.append(reason)
    return translations


def _build_curation_metric_items(
    measured: Dict[str, Any],
    yunet_face: Optional[Dict[str, Any]],
    *,
    min_sharpness: float,
    min_cheek_fft: float,
    eff_min_clip: float,
    eff_max_delta_e: float,
    max_seam: float,
) -> List[CurationMetricItem]:
    """Construct the 6 standardized CurationMetricItem progress indicators for UI display."""
    sharpness = float(measured["sharpness_score"])
    cheek_fft = measured.get("cheek_fft_ratio")
    clip_val = measured.get("clip_score")
    delta_e = measured.get("delta_e_lab")
    seam_val = measured.get("seam_diff")
    geom_ok = bool(
        measured.get("face_detected")
        and measured.get("geometry_valid")
        and measured.get("eye_passed")
    )
    yunet_score = float(yunet_face["score"]) if yunet_face is not None and "score" in yunet_face else 0.0

    return [
        CurationMetricItem(
            name="sharpness",
            label="Laplacian Sharpness",
            value=sharpness,
            threshold=float(min_sharpness),
            passed=sharpness >= float(min_sharpness),
            direction=">=",
        ),
        CurationMetricItem(
            name="cheek_fft",
            label="Cheek Skin FFT",
            value=float(cheek_fft) if cheek_fft is not None else None,
            threshold=float(min_cheek_fft),
            passed=(cheek_fft is None or float(cheek_fft) >= float(min_cheek_fft)),
            direction=">=",
            detail=f"Whole-face FFT={float(measured.get('fft_high_freq_ratio', 0.0)):.4f}",
        ),
        CurationMetricItem(
            name="clip_score",
            label="CLIP ViT-B/32",
            value=float(clip_val) if clip_val is not None else None,
            threshold=float(eff_min_clip),
            passed=(clip_val is None or float(clip_val) >= float(eff_min_clip)),
            direction=">=",
        ),
        CurationMetricItem(
            name="geometry",
            label="YuNet Geometry & Eyes",
            value=yunet_score,
            threshold=0.60,
            passed=geom_ok,
            direction=">=",
            detail=(
                f"eyes={int(measured.get('eye_count', 0))}, "
                f"sym={float(measured.get('bilateral_symmetry', 0.0)):.2f}"
            ),
        ),
        CurationMetricItem(
            name="delta_e_lab",
            label="CIELab ΔE Drift",
            value=float(delta_e) if delta_e is not None else None,
            threshold=float(eff_max_delta_e),
            passed=(delta_e is None or float(delta_e) <= float(eff_max_delta_e)),
            direction="<=",
        ),
        CurationMetricItem(
            name="seam_diff",
            label="Boundary Seam Diff",
            value=float(seam_val) if seam_val is not None else None,
            threshold=float(max_seam),
            passed=(seam_val is None or float(seam_val) <= float(max_seam)),
            direction="<=",
            detail=(
                f"struct_corr={float(measured['structure_corr']):.3f}"
                if measured.get("structure_corr") is not None
                else ""
            ),
        ),
    ]


def inspect_pipeline_sample(
    target_image: Union[Path, Image.Image, np.ndarray],
    *,
    bundle: Optional[DetectorBundle] = None,
    source_image: Optional[Union[Path, Image.Image, np.ndarray]] = None,
    mask: Optional[np.ndarray] = None,
    mask_region: Optional[str] = None,
    mode: str = "sd_img2img",
    prompt: str = "",
    seed: Optional[int] = None,
    min_sharpness: float = 1.3,
    min_cheek_fft: float = 0.005,
    min_clip: Optional[float] = None,
    max_delta_e: float = 18.0,
    max_seam: float = 35.0,
    tau: float = 0.50,
    ground_truth_label: Optional[int] = None,
    run_detector_if_rejected: bool = False,
    clip_scorer: Optional[Any] = None,
    compute_heatmaps: bool = True,
    precomputed: Optional[Dict[str, Any]] = None,
) -> InspectionResult:
    """Run the 4-stage live inspection pipeline (Normalize -> Curation Gate -> Overlays -> Detectors & Heatmaps).

    Args:
        target_image: Path, PIL Image, or ndarray `(H, W, C)` to inspect.
        bundle: Optional loaded `DetectorBundle` (`M0`, `M2*`, `M3`).
        source_image: Optional paired source FFHQ image for `sd_img2img` / `sd_inpaint`.
        mask: Optional inpaint mask `(H, W)` or `(1, H, W)`.
        mask_region: Optional region name (`lips`, `periocular`, `full`, `nose`) if `mask` is omitted.
        mode: Pipeline mode (`sd_text2img`, `sd_img2img`, `sd_inpaint`, `rv51_ood`, `real_ffhq`, `custom_upload`).
        prompt: Generation prompt for CLIP similarity and sunglasses-intent awareness.
        seed: Optional seed used to render deterministic `Text2Img` latent noise preview.
        min_sharpness: Minimum Laplacian variance threshold.
        min_cheek_fft: Minimum localized cheek-skin FFT high-frequency ratio threshold.
        min_clip: Explicit CLIP score threshold; defaults to `resolve_effective_min_clip(0.20, mode)`.
        max_delta_e: Maximum CIELab color drift for paired modes.
        max_seam: Maximum boundary seam gradient difference for masked inpaint mode.
        tau: Detector decision threshold `tau` on `P(fake)`.
        ground_truth_label: Optional ground truth label (`0=real`, `1=fake`).
        run_detector_if_rejected: Run detector inference even when the curation gate rejects the image.
        clip_scorer: Optional pre-initialized `ClipScorer` instance.
        compute_heatmaps: Whether to compute `M0` and `M3` DINOv2 spatial patch sensitivity heatmaps.
        precomputed: Optional cached intermediate payload to skip redundant YuNet/CLIP/detector/heatmap work
            while still recomputing interactive thresholds, metric items, face audit overlay, and `is_hard_sample`.

    Returns:
        Populated `InspectionResult` dataclass.
    """
    t_total_start = time.perf_counter()
    mode = _CANONICAL_MODE_MAP.get(str(mode), str(mode))
    pre = precomputed or {}
    latencies_ms: Dict[str, float] = {}
    warnings: List[str] = list(pre["warnings"]) if pre.get("warnings") is not None else []

    t_pre = time.perf_counter()
    rgb_512 = pre["rgb_512"] if pre.get("rgb_512") is not None else _to_512_rgb_u8(target_image)
    if "source_512" in pre:
        source_512 = pre["source_512"]
    else:
        source_512 = _to_512_rgb_u8(source_image) if source_image is not None else None
    if "mask_512" in pre:
        mask_512 = pre["mask_512"]
    else:
        mask_512 = _resolve_inpaint_mask_512(rgb_512, source_512, mask, mask_region, mode)
    latencies_ms["preprocess"] = round((time.perf_counter() - t_pre) * 1000.0, 2)

    t_cur = time.perf_counter()
    eff_min_clip = float(min_clip) if min_clip is not None else float(resolve_effective_min_clip(0.20, mode))
    p_low = str(prompt or "").lower()
    has_sg_intent = mode == "sd_inpaint" and any(
        tok in p_low for tok in ("sunglasses", "dark glasses", "aviator", "tinted glasses", "shades")
    )
    eff_max_delta_e = max(float(max_delta_e), 45.0) if has_sg_intent else float(max_delta_e)

    if pre.get("measured_base") is not None:
        yunet_face = pre.get("yunet_face")
        cheek_boxes = (
            list(pre["cheek_boxes"])
            if pre.get("cheek_boxes") is not None
            else compute_cheek_roi_boxes(yunet_face, width=512, height=512)
        )
        biometrics = (
            dict(pre["biometrics"])
            if pre.get("biometrics") is not None
            else extract_face_biometrics_rgb(rgb_512, prompt=prompt)
        )
        measured = dict(pre["measured_base"])
        measured["mask_region"] = mask_region
        measured["seed"] = seed
        measured["tau"] = float(tau)
        measured["effective_min_clip"] = eff_min_clip
        measured["effective_max_delta_e"] = eff_max_delta_e
    else:
        yunet_face = detect_face_yunet(rgb_512)
        cheek_boxes = compute_cheek_roi_boxes(yunet_face, width=512, height=512)
        biometrics = extract_face_biometrics_rgb(rgb_512, prompt=prompt)
        cheek_res = compute_cheek_skin_fft(
            rgb_512,
            yunet_face=yunet_face,
            min_cheek_ratio=float(min_cheek_fft),
        )
        cheek_fft_ratio = (
            float(cheek_res["cheek_fft_ratio"])
            if cheek_res.get("cheek_fft_ratio") is not None
            else None
        )

        clip_score_val: Optional[float] = None
        has_prompt = bool(prompt and str(prompt).strip())
        if has_prompt and mode != "real_ffhq":
            active_clip = clip_scorer
            owns_clip = False
            try:
                if active_clip is None:
                    scorer_cls = ClipScorer if ClipScorer is not _ORIG_CLIP_SCORER else _metrics_mod.ClipScorer
                    active_clip = scorer_cls(device="cpu")
                    owns_clip = True
                clip_score_val = float(active_clip.score(rgb_512, prompt))
            except Exception as clip_err:
                logger.warning("Offline/unavailable CLIP scorer fallback (clip_score=None): %s", clip_err)
                warnings.append(f"CLIP scorer unavailable ({clip_err}); clip_score set to None.")
                clip_score_val = None
            finally:
                if owns_clip and active_clip is not None and hasattr(active_clip, "close"):
                    try:
                        active_clip.close()
                    except Exception:
                        pass

        mask_tensor_arg = torch.from_numpy(mask_512).unsqueeze(0) if mask_512 is not None else None
        ext_rep = audit_image_extended(
            rgb_512,
            mode=mode,
            prompt=prompt,
            source_image=source_512,
            mask=mask_tensor_arg,
            clip_scorer=None,
            min_clip_score=eff_min_clip,
            min_sharpness=float(min_sharpness),
            face_backend="yunet",
            clip_score_override=clip_score_val,
        )

        face_box = yunet_face["bbox_xyxy"] if yunet_face is not None else None
        whole_fft = compute_fft_high_freq_ratio(rgb_512, bbox_xyxy=face_box)

        eye_reasons = [
            r
            for r in ext_rep.rejection_reasons
            if any(tok in r.lower() for tok in ("bilateral facial symmetry", "expected >= 2 eyes", "eye-pair"))
        ]
        geometry_reasons = [
            r
            for r in ext_rep.rejection_reasons
            if r not in eye_reasons
            and not r.startswith("Sharpness ")
            and not r.startswith("FFT texture")
            and not r.startswith("Plastic/waxy")
            and not r.startswith("CLIP score")
            and not r.startswith("CIELab ")
            and not r.startswith("Background CIELab")
            and not r.startswith("Structure correlation")
            and not r.startswith("Boundary seam")
        ]
        face_detected_val = yunet_face is not None

        bg_delta_e: Optional[float] = None
        if source_512 is not None and mask_512 is not None and np.any(mask_512 > 0.5):
            bg_mask = mask_512 < 0.25
            if np.any(bg_mask):
                syn_lab = cv2.cvtColor(rgb_512, cv2.COLOR_RGB2LAB).astype(np.float32)
                src_lab = cv2.cvtColor(source_512, cv2.COLOR_RGB2LAB).astype(np.float32)
                lab_diff_map = np.linalg.norm(syn_lab - src_lab, axis=-1)
                bg_delta_e = float(np.mean(lab_diff_map[bg_mask]))

        measured = {
            "face_detected": face_detected_val,
            "sharpness_score": float(ext_rep.sharpness_score),
            "geometry_valid": bool(ext_rep.geometry_valid),
            "geometry_reasons": geometry_reasons,
            "eye_passed": len(eye_reasons) == 0,
            "eye_reasons": eye_reasons,
            "eye_count": int(ext_rep.eye_count),
            "sunglasses_occluded": bool(ext_rep.sunglasses_occluded),
            "bilateral_symmetry": float(ext_rep.bilateral_symmetry),
            "fft_high_freq_ratio": float(whole_fft["high_freq_ratio"]),
            "texture_flag": str(whole_fft["texture_flag"]),
            "cheek_fft_ratio": cheek_fft_ratio,
            "clip_score": float(ext_rep.clip_score) if ext_rep.clip_score is not None else None,
            "delta_e_lab": float(ext_rep.delta_e_lab) if ext_rep.delta_e_lab is not None else None,
            "bg_delta_e": bg_delta_e,
            "structure_corr": float(ext_rep.structure_corr) if ext_rep.structure_corr is not None else None,
            "seam_diff": float(ext_rep.seam_diff) if ext_rep.seam_diff is not None else None,
            "is_seamless": ext_rep.is_seamless,
            "mask_region": mask_region,
            "seed": seed,
            "tau": float(tau),
            "effective_min_clip": eff_min_clip,
            "effective_max_delta_e": eff_max_delta_e,
        }

    gate_passed, rejection_reasons = apply_interactive_thresholds(
        measured,
        mode=mode,
        min_sharpness=float(min_sharpness),
        min_cheek_fft=float(min_cheek_fft),
        min_clip=eff_min_clip,
        max_delta_e=eff_max_delta_e,
        max_seam=float(max_seam),
    )
    rejection_reasons_vi = _translate_rejection_reasons_vi(rejection_reasons)
    metrics_items = _build_curation_metric_items(
        measured,
        yunet_face,
        min_sharpness=float(min_sharpness),
        min_cheek_fft=float(min_cheek_fft),
        eff_min_clip=eff_min_clip,
        eff_max_delta_e=eff_max_delta_e,
        max_seam=float(max_seam),
    )
    latencies_ms["curation"] = round((time.perf_counter() - t_cur) * 1000.0, 2)

    t_ov = time.perf_counter()
    overlays: Dict[str, np.ndarray] = {
        "face_audit": render_face_audit_overlay(
            rgb_512,
            yunet_face=yunet_face,
            passed=gate_passed,
            cheek_boxes=cheek_boxes,
            mask=mask_512,
        )
    }
    if pre.get("source_with_mask_overlay") is not None:
        overlays["source_with_mask"] = pre["source_with_mask_overlay"]
    elif source_512 is not None and mask_512 is not None:
        overlays["source_with_mask"] = render_face_audit_overlay(
            source_512,
            yunet_face=detect_face_yunet(source_512),
            passed=True,
            cheek_boxes=[],
            mask=mask_512,
        )
    if seed is not None and mode in ("sd_text2img", "text2img"):
        overlays["latent_noise"] = render_latent_noise_preview(int(seed), size=256)

    before_after_html: Optional[str] = None
    if pre.get("before_after_html") is not None:
        before_after_html = pre["before_after_html"]
    elif source_512 is not None:
        before_after_html = build_before_after_wipe_html(
            source_512,
            rgb_512,
            left_label="INPUT FFHQ",
            right_label=f"OUTPUT ({mode.upper()})",
        )
    latencies_ms["overlays"] = round((time.perf_counter() - t_ov) * 1000.0, 2)

    detector_scores: Dict[str, float] = {}
    heatmaps: Dict[str, np.ndarray] = {}
    is_hard_sample = False

    should_run_detector = bundle is not None and (gate_passed or mode == "real_ffhq" or run_detector_if_rejected)
    cached_scores = pre.get("detector_scores")
    if cached_scores:
        detector_scores = dict(cached_scores)
    elif should_run_detector and bundle is not None:
        t_det = time.perf_counter()
        detector_scores = score_image_with_bundle(bundle, rgb_512)
        latencies_ms["detector"] = round((time.perf_counter() - t_det) * 1000.0, 2)

    if detector_scores:
        m0_prob = detector_scores.get("M0")
        is_hard_sample = bool(
            m0_prob is not None
            and m0_prob < float(tau)
            and mode != "real_ffhq"
            and (ground_truth_label is None or int(ground_truth_label) == 1)
        )

    if compute_heatmaps:
        cached_heatmaps = pre.get("heatmaps")
        if cached_heatmaps:
            heatmaps = dict(cached_heatmaps)
        elif should_run_detector and bundle is not None:
            t_hm = time.perf_counter()
            for hm_key in ("M0", "M3"):
                if hm_key in bundle.models:
                    hm_res = compute_dinov2_patch_heatmap(bundle.models[hm_key], rgb_512, bundle.device)
                    heatmaps[hm_key] = hm_res["heatmap_rgb"]
            latencies_ms["heatmaps"] = round((time.perf_counter() - t_hm) * 1000.0, 2)

    latencies_ms["total"] = round((time.perf_counter() - t_total_start) * 1000.0, 2)

    return InspectionResult(
        mode=mode,
        image_512_rgb=rgb_512,
        source_512_rgb=source_512,
        mask_512=mask_512,
        gate_passed=gate_passed,
        rejection_reasons=rejection_reasons,
        rejection_reasons_vi=rejection_reasons_vi,
        metrics=metrics_items,
        measured=measured,
        yunet_face=yunet_face,
        cheek_boxes=cheek_boxes,
        biometrics=biometrics,
        detector_scores=detector_scores,
        is_hard_sample=is_hard_sample,
        ground_truth_label=ground_truth_label,
        overlays=overlays,
        heatmaps=heatmaps,
        before_after_html=before_after_html,
        latencies_ms=latencies_ms,
        warnings=warnings,
    )


def launch_live_sd_worker(
    mode: str,
    prompt: str,
    *,
    source_image: Optional[Path] = None,
    mask_region: str = "lips",
    model_id: Optional[str] = None,
    steps: int = 20,
    guidance_scale: float = 6.2,
    strength: float = 0.30,
    seed: int = 42,
    output_root: Path = Path("experiments/demo_runs/streamlit_live"),
    timeout_s: int = 180,
) -> Dict[str, Any]:
    """Launch isolated `src.pipelines.live_sd_worker` subprocess to synthesize 1 image safely on 4GB VRAM.

    Args:
        mode: Generation mode (`text2img`/`sd_text2img`, `img2img`/`sd_img2img`, `inpaint`/`sd_inpaint`).
        prompt: Positive conditioning text prompt.
        source_image: Source image path (required for `img2img` and `inpaint`).
        mask_region: Inpaint target mask region (`lips`, `periocular`, `full`, `nose`).
        model_id: Optional SD 1.5 model override (or `'mock_sd'` in tests).
        steps: Denoising step count.
        guidance_scale: Classifier-free guidance scale.
        strength: Img2Img denoising strength.
        seed: Reproducible generation seed.
        output_root: Parent directory where an isolated run folder is created.
        timeout_s: Maximum subprocess wall-clock timeout in seconds.

    Returns:
        Metadata dictionary loaded from `metadata.json` enriched with `output_dir`, `output_path`,
        and optional `input_path` / `mask_path`.

    Raises:
        ValueError: If mode or paired source_image is invalid.
        RuntimeError: If subprocess times out, returns non-zero exit code, or omits artifacts.
    """
    mode_map = {
        "sd_text2img": "text2img",
        "sd_img2img": "img2img",
        "sd_inpaint": "inpaint",
    }
    worker_mode = mode_map.get(mode, mode)
    if worker_mode not in ("text2img", "img2img", "inpaint"):
        raise ValueError(f"Unsupported live SD mode: '{mode}'.")
    if worker_mode in ("img2img", "inpaint"):
        if source_image is None:
            raise ValueError(f"Mode '{worker_mode}' requires source_image.")
        if not Path(source_image).is_file():
            raise FileNotFoundError(f"Source image not found: {source_image}")

    run_dir = Path(output_root) / f"{worker_mode}_seed{int(seed)}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "src.pipelines.live_sd_worker",
        "--mode",
        worker_mode,
        "--prompt",
        str(prompt),
        "--output-dir",
        str(run_dir),
        "--mask-region",
        str(mask_region),
        "--steps",
        str(int(steps)),
        "--guidance-scale",
        str(float(guidance_scale)),
        "--strength",
        str(float(strength)),
        "--seed",
        str(int(seed)),
    ]
    if source_image is not None:
        cmd.extend(["--source-image", str(source_image)])
    if model_id is not None:
        cmd.extend(["--model-id", str(model_id)])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise RuntimeError(
            f"Live SD worker timed out after {timeout_s}s (mode={worker_mode})."
        ) from exc
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "unknown error").strip()[-1000:]
        shutil.rmtree(run_dir, ignore_errors=True)
        raise RuntimeError(
            f"Live SD worker failed with exit code {proc.returncode}: {err_tail}"
        )

    meta_path = run_dir / "metadata.json"
    out_img_path = run_dir / "output.png"
    if not meta_path.is_file() or not out_img_path.is_file():
        shutil.rmtree(run_dir, ignore_errors=True)
        raise RuntimeError(
            f"Live SD worker finished without producing expected artifacts in {run_dir}."
        )

    try:
        metadata: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise RuntimeError(
            f"Live SD worker produced malformed metadata.json in {run_dir}: {exc}"
        ) from exc
    result: Dict[str, Any] = dict(metadata)
    result["output_dir"] = str(run_dir)
    result["output_path"] = str(out_img_path)
    if (run_dir / "input.png").is_file():
        result["input_path"] = str(run_dir / "input.png")
    if (run_dir / "mask.png").is_file():
        result["mask_path"] = str(run_dir / "mask.png")
    return result


_LIVE_MASK_REGIONS: Tuple[str, ...] = ("lips", "periocular", "nose", "full")
_LIVE_MODES: Tuple[str, ...] = ("text2img", "img2img", "inpaint")


def _collect_valid_ffhq_paths(catalog: Dict[str, List[ShowcaseCatalogEntry]]) -> List[str]:
    """Collect deduplicated, existing FFHQ source image paths from a showcase catalog."""
    paths: List[str] = []
    seen: set[str] = set()
    for tab_entries in (catalog or {}).values():
        for entry in tab_entries:
            candidates: List[Optional[str]] = []
            if entry.mode == "real_ffhq":
                candidates.append(entry.image_path)
            if entry.source_image_path:
                candidates.append(entry.source_image_path)
            for cand in candidates:
                if not cand or cand in seen:
                    continue
                if Path(cand).is_file():
                    seen.add(cand)
                    paths.append(cand)
    return paths


def _normalize_live_mode(mode: Optional[str]) -> Optional[str]:
    """Map canonical `sd_*` mode names to live worker mode names."""
    if mode is None:
        return None
    reverse_map = {"sd_text2img": "text2img", "sd_img2img": "img2img", "sd_inpaint": "inpaint"}
    return reverse_map.get(str(mode), str(mode))


def sample_random_live_config(
    catalog: Dict[str, List[ShowcaseCatalogEntry]],
    *,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Sample a reproducible single-image live SD config using `PromptGrammarEngine`.

    Args:
        catalog: Showcase catalog mapping tab names to `ShowcaseCatalogEntry` lists.
        mode: Optional mode (`text2img`, `img2img`, `inpaint`, or `sd_*` aliases).
        seed: Optional deterministic seed; random positive integer if omitted.

    Returns:
        Dictionary with keys `mode`, `prompt`, `seed`, `source_image`, and `mask_region`.
    """
    resolved_seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % 999_999) + 1
    rng = np.random.RandomState(resolved_seed)
    ffhq_paths = _collect_valid_ffhq_paths(catalog)

    norm_mode = _normalize_live_mode(mode)
    if norm_mode is None:
        resolved_mode = str(_LIVE_MODES[int(rng.randint(0, len(_LIVE_MODES)))]) if ffhq_paths else "text2img"
    else:
        if norm_mode not in _LIVE_MODES:
            raise ValueError(f"Unsupported live SD mode: '{mode}'.")
        resolved_mode = norm_mode

    source_image: Optional[str] = None
    if resolved_mode in ("img2img", "inpaint"):
        if not ffhq_paths:
            raise ValueError(f"Mode '{resolved_mode}' requires at least one valid FFHQ source image in catalog.")
        source_image = ffhq_paths[int(rng.randint(0, len(ffhq_paths)))]

    mask_region = (
        str(_LIVE_MASK_REGIONS[int(rng.randint(0, len(_LIVE_MASK_REGIONS)))])
        if resolved_mode == "inpaint"
        else "lips"
    )
    engine = PromptGrammarEngine(seed=resolved_seed)
    prompt = engine.generate_prompt(mode=resolved_mode, seed_offset=resolved_seed % 10000)

    return {
        "mode": resolved_mode,
        "prompt": prompt,
        "seed": resolved_seed,
        "source_image": source_image,
        "mask_region": mask_region,
    }


def sample_random_batch_configs(
    catalog: Dict[str, List[ShowcaseCatalogEntry]],
    *,
    count: int,
    mode: str = "mixed",
    base_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Sample a list of reproducible live SD batch configs, falling back to `text2img` if FFHQ is absent."""
    if int(count) < 1:
        raise ValueError(f"count must be >= 1, got {count}.")

    seed_base = int(base_seed) if base_seed is not None else int(np.random.SeedSequence().entropy % 900_000) + 1
    ffhq_paths = _collect_valid_ffhq_paths(catalog)

    configs: List[Dict[str, Any]] = []
    for idx in range(int(count)):
        if mode == "mixed":
            item_mode = _LIVE_MODES[idx % len(_LIVE_MODES)] if ffhq_paths else "text2img"
        else:
            item_mode = _normalize_live_mode(mode) or "text2img"
        configs.append(
            sample_random_live_config(
                catalog,
                mode=item_mode,
                seed=seed_base + idx,
            )
        )
    return configs


def launch_live_sd_batch_worker(
    configs: Sequence[Dict[str, Any]],
    *,
    output_root: Path = Path("experiments/demo_runs/streamlit_live_batch"),
    timeout_s: int = 360,
) -> Dict[str, Any]:
    """Launch isolated `src.pipelines.live_sd_worker --batch-manifest` subprocess for a batch of configs."""
    if not configs:
        raise ValueError("Batch configs must be non-empty.")

    run_dir = Path(output_root) / f"batch_{len(configs)}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_in_path = run_dir / "batch_configs.json"

    serializable_configs: List[Dict[str, Any]] = []
    for cfg in configs:
        item = dict(cfg)
        if item.get("source_image") is not None:
            item["source_image"] = str(item["source_image"])
        serializable_configs.append(item)
    manifest_in_path.write_text(json.dumps(serializable_configs, indent=2), encoding="utf-8")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "src.pipelines.live_sd_worker",
        "--batch-manifest",
        str(manifest_in_path),
        "--output-dir",
        str(run_dir),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise RuntimeError(
            f"Live SD batch worker timed out after {timeout_s}s (items={len(configs)})."
        ) from exc
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "unknown error").strip()[-1000:]
        shutil.rmtree(run_dir, ignore_errors=True)
        raise RuntimeError(
            f"Live SD batch worker failed with exit code {proc.returncode}: {err_tail}"
        )

    batch_manifest_path = run_dir / "batch_manifest.json"
    if not batch_manifest_path.is_file():
        shutil.rmtree(run_dir, ignore_errors=True)
        raise RuntimeError(
            f"Live SD batch worker finished without producing batch_manifest.json in {run_dir}."
        )

    try:
        manifest_payload: Dict[str, Any] = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise RuntimeError(
            f"Live SD batch worker produced malformed batch_manifest.json in {run_dir}: {exc}"
        ) from exc

    manifest_payload.setdefault("output_dir", str(run_dir))
    return manifest_payload


def evaluate_batch_with_bundle(
    items: Sequence[Dict[str, Any]],
    *,
    bundle: Optional[DetectorBundle],
    min_sharpness: float,
    min_cheek_fft: float,
    min_clip: Optional[float],
    tau: float,
    clip_scorer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run 4-stage inspection across a generated batch and aggregate curation/hard-sample funnel counts."""
    evaluated_items: List[Dict[str, Any]] = []
    generated = 0
    gate_passed = 0
    gate_rejected = 0
    hard_samples = 0
    m3_rescued = 0

    for item in items:
        target_image = item.get("output_path") or item.get("image_path")
        if target_image is None:
            raise ValueError("Batch item must include 'output_path' or 'image_path'.")
        source_image = item.get("input_path") or item.get("source_image")
        res = inspect_pipeline_sample(
            Path(str(target_image)),
            bundle=bundle,
            source_image=Path(str(source_image)) if source_image is not None else None,
            mask_region=item.get("mask_region"),
            mode=str(item.get("mode", "sd_text2img")),
            prompt=str(item.get("prompt", "")),
            seed=item.get("seed"),
            min_sharpness=float(min_sharpness),
            min_cheek_fft=float(min_cheek_fft),
            min_clip=min_clip,
            tau=float(tau),
            clip_scorer=clip_scorer,
            compute_heatmaps=False,
        )
        generated += 1
        if res.gate_passed:
            gate_passed += 1
        else:
            gate_rejected += 1
        if res.is_hard_sample:
            hard_samples += 1
            m3_prob = res.detector_scores.get("M3")
            if m3_prob is not None and float(m3_prob) >= float(tau):
                m3_rescued += 1

        evaluated_items.append(
            {
                "item": dict(item),
                "inspection": res,
                "gate_passed": res.gate_passed,
                "is_hard_sample": res.is_hard_sample,
                "detector_scores": dict(res.detector_scores),
            }
        )

    return {
        "items": evaluated_items,
        "funnel": {
            "generated": generated,
            "gate_passed": gate_passed,
            "gate_rejected": gate_rejected,
            "hard_samples": hard_samples,
            "m3_rescued": m3_rescued,
        },
    }


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Read non-empty JSON lines from a manifest or report file."""
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Write a sequence of dictionaries as UTF-8 JSONL lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _score_images_batched(
    bundle: DetectorBundle,
    image_paths: Sequence[Union[Path, str]],
    batch_size: int = 16,
) -> List[Dict[str, float]]:
    """Run batched eval-mode inference across all checkpoints in `bundle` on 512x512 LANCZOS RGB tensors.

    Args:
        bundle: Loaded `DetectorBundle` (`M0`, `M2*`, `M3`).
        image_paths: Sequence of image paths to decode and score.
        batch_size: Inference batch size protecting low-VRAM devices.

    Returns:
        List of `{model_label: P(fake)}` dicts aligned 1-to-1 with `image_paths`.
    """
    if not image_paths:
        return []

    step = max(1, int(batch_size))
    use_amp = bundle.device.type == "cuda"
    per_model_probs: Dict[str, List[float]] = {label: [] for label in bundle.models}

    with torch.no_grad():
        for start in range(0, len(image_paths), step):
            batch_tensors: List[torch.Tensor] = []
            for path_item in image_paths[start : start + step]:
                rgb_512 = _to_512_rgb_u8(Path(path_item))
                # Normalized tensor shape: (C=3, H=512, W=512) float32 RGB in [0.0, 1.0]
                tensor_chw = torch.from_numpy(rgb_512.astype(np.float32) / 255.0).permute(2, 0, 1)
                batch_tensors.append(tensor_chw)

            batch = torch.stack(batch_tensors, dim=0).to(bundle.device)  # (B, 3, 512, 512)
            del batch_tensors
            for label, model in bundle.models.items():
                model.eval()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(batch)
                    if logits.ndim == 2 and logits.shape[1] > 1:
                        probs = torch.softmax(logits.float(), dim=-1)[:, 1]
                    else:
                        probs = torch.sigmoid(logits.float().view(-1))
                per_model_probs[label].extend(float(np.clip(p, 0.0, 1.0)) for p in probs.detach().cpu().tolist())
            del batch

    results: List[Dict[str, float]] = []
    for idx in range(len(image_paths)):
        results.append({label: per_model_probs[label][idx] for label in bundle.models})
    return results


def resync_demo_run_scores(
    demo_dir: Path = Path("experiments/demo_runs/sd15_pregen_200"),
    threshold: float = 0.50,
    *,
    bundle: Optional[DetectorBundle] = None,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Re-score ALL items in `fake_manifest.jsonl` (curated and rejected) with live checkpoints.

    Updates `scored_manifest.jsonl` (all 200 items with `"is_curated": bool`),
    `hard_samples_manifest.jsonl` (subset of curated items with `M0 < threshold`), and
    `e2e_summary.json` (`model_metrics`, `hard_samples_count`, `hard_sample_rate`, `threshold_tau`).
    """
    demo_root = Path(demo_dir)
    fake_path = demo_root / "fake_manifest.jsonl"
    curation_report_path = demo_root / "curation_report.jsonl"
    curated_path = demo_root / "curated_manifest.jsonl"
    scored_path = demo_root / "scored_manifest.jsonl"
    hard_path = demo_root / "hard_samples_manifest.jsonl"
    summary_path = demo_root / "e2e_summary.json"

    fake_items = _read_jsonl_records(fake_path)
    if not fake_items:
        raise FileNotFoundError(f"Demo fake_manifest.jsonl missing or empty: {fake_path}")

    report_items = _read_jsonl_records(curation_report_path)
    curated_items = _read_jsonl_records(curated_path)
    valid_by_id: Dict[str, bool] = {
        str(r["image_id"]): bool(r.get("is_valid", False))
        for r in report_items
        if "image_id" in r
    }
    curated_id_set = {str(r["image_id"]) for r in curated_items if "image_id" in r}

    image_paths: List[Path] = []
    for item in fake_items:
        fp = Path(str(item.get("file_path") or item.get("image_path") or ""))
        if not fp.is_file():
            raise FileNotFoundError(f"Missing demo image referenced by {item.get('image_id')}: {fp}")
        image_paths.append(fp)

    owns_bundle = bundle is None
    active_bundle = bundle if bundle is not None else load_detector_bundle()
    try:
        batched_scores = _score_images_batched(active_bundle, image_paths, batch_size=batch_size)
    finally:
        if owns_bundle:
            unload_detector_bundle(active_bundle)

    scored_records: List[Dict[str, Any]] = []
    curated_scored: List[Dict[str, Any]] = []
    hard_records: List[Dict[str, Any]] = []

    for item, short_scores in zip(fake_items, batched_scores):
        rec = dict(item)
        img_id = str(rec.get("image_id", ""))
        is_curated = valid_by_id[img_id] if img_id in valid_by_id else (img_id in curated_id_set)

        full_scores = {
            _FULL_MODEL_LABELS.get(k, k): float(v)
            for k, v in short_scores.items()
        }
        p0 = float(short_scores["M0"])
        is_hard = bool(is_curated and p0 < float(threshold))

        rec["is_curated"] = bool(is_curated)
        rec["detector_scores"] = full_scores
        rec["baseline_score"] = p0
        rec["is_hard_sample"] = is_hard
        scored_records.append(rec)
        if is_curated:
            curated_scored.append(rec)
            if is_hard:
                hard_records.append(rec)

    _write_jsonl_records(scored_path, scored_records)
    _write_jsonl_records(hard_path, hard_records)

    summary: Dict[str, Any] = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    hard_ids = {str(r["image_id"]) for r in hard_records}
    model_metrics: Dict[str, Dict[str, float]] = {}
    for short_key, full_label in _FULL_MODEL_LABELS.items():
        if not curated_scored:
            model_metrics[full_label] = {
                "recall": 0.0,
                "bypass_rate": 0.0,
                "mean_p_fake": 0.0,
                "hard_recall": 0.0,
            }
            continue
        probs = [float(r["detector_scores"][full_label]) for r in curated_scored]
        caught = sum(1 for p in probs if p >= float(threshold))
        hard_caught = sum(
            1
            for r in curated_scored
            if str(r["image_id"]) in hard_ids and float(r["detector_scores"][full_label]) >= float(threshold)
        )
        recall = caught / len(probs)
        model_metrics[full_label] = {
            "recall": float(recall),
            "bypass_rate": float(1.0 - recall),
            "mean_p_fake": float(np.mean(probs)),
            "hard_recall": float(hard_caught / len(hard_records)) if hard_records else 1.0,
        }

    summary["generated_images"] = len(scored_records)
    summary["curated_images"] = len(curated_scored)
    summary["rejected_images"] = len(scored_records) - len(curated_scored)
    summary["curation_pass_rate"] = (
        float(len(curated_scored) / len(scored_records)) if scored_records else 0.0
    )
    summary["hard_samples_count"] = len(hard_records)
    summary["hard_sample_rate"] = (
        float(len(hard_records) / len(curated_scored)) if curated_scored else 0.0
    )
    summary["threshold_tau"] = float(threshold)
    summary["model_metrics"] = model_metrics
    def _to_repo_relpath(p: Path) -> str:
        if not p.is_absolute():
            return p.as_posix()
        return Path(os.path.relpath(p.resolve(), Path.cwd().resolve())).as_posix()

    summary.setdefault("artifacts", {})
    summary["artifacts"]["fake_manifest"] = _to_repo_relpath(fake_path)
    summary["artifacts"]["curated_manifest"] = _to_repo_relpath(curated_path)
    summary["artifacts"]["curation_report"] = _to_repo_relpath(curation_report_path)
    summary["artifacts"]["hard_manifest"] = _to_repo_relpath(hard_path)
    summary["artifacts"]["scored_manifest"] = _to_repo_relpath(scored_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Resynced %d demo records (%d curated, %d hard samples) -> %s",
        len(scored_records),
        len(curated_scored),
        len(hard_records),
        demo_root,
    )
    return summary


def _build_real_source_lookup() -> Dict[str, str]:
    """Map FFHQ/CelebA source_real_id to existing local real image file paths."""
    lookup: Dict[str, str] = {}
    for manifest_p in (
        Path("data/raw/manifest.jsonl"),
        Path("data/benchmarks/master_frozen_test.jsonl"),
        Path("data/active_learning/cumulative_challenge_test.jsonl"),
    ):
        for rec in _read_jsonl_records(manifest_p):
            if int(rec.get("label", -1)) == 0:
                img_id = str(rec.get("image_id") or "")
                fp = str(rec.get("file_path") or "")
                if img_id and fp and Path(fp).is_file():
                    lookup[img_id] = fp
    return lookup


def _resolve_source_path(rec: Dict[str, Any], real_lookup: Dict[str, str]) -> Optional[str]:
    """Resolve paired source image path for an SD 1.5 record if available locally."""
    src_id = rec.get("source_real_id")
    if not src_id:
        return None
    src_id_str = str(src_id)
    if src_id_str in real_lookup:
        return real_lookup[src_id_str]
    if src_id_str.startswith("ffhq_"):
        cand = Path("data/raw/faces/ffhq") / f"{src_id_str.split('_', 1)[1]}.png"
        if cand.is_file():
            return str(cand)
    return None


def _collect_train_exclusions() -> Tuple[set[str], set[str]]:
    """Collect image IDs and file paths belonging to any detector training split."""
    train_ids: set[str] = set()
    train_paths: set[str] = set()
    for train_p in (
        Path("data/benchmarks/D0_baseline/train.jsonl"),
        Path("data/benchmarks/D1_augmented/train.jsonl"),
        Path("data/benchmarks/D2_hard_boosted/train.jsonl"),
        Path("data/benchmarks/D3_clean/train.jsonl"),
    ):
        for rec in _read_jsonl_records(train_p):
            if "image_id" in rec:
                train_ids.add(str(rec["image_id"]))
            if "file_path" in rec:
                train_paths.add(str(rec["file_path"]))
    return train_ids, train_paths


def _audit_and_build_catalog_entry(
    rec: Dict[str, Any],
    *,
    tab: str,
    mode: str,
    split_provenance: str,
    scores: Dict[str, float],
    source_path: Optional[str],
    clip_scorer: Optional[Any],
) -> ShowcaseCatalogEntry:
    """Run live extended audit + cheek FFT on one candidate record and return ShowcaseCatalogEntry."""
    image_path = str(rec.get("file_path") or rec.get("image_path") or "")
    gen_cfg = rec.get("generation") if isinstance(rec.get("generation"), dict) else {}
    mask_region = gen_cfg.get("inpaint_mask_target") or rec.get("mask_region")
    prompt = str(rec.get("prompt") or "")
    seed = gen_cfg.get("seed") if gen_cfg.get("seed") is not None else rec.get("seed")
    steps = gen_cfg.get("steps") if gen_cfg.get("steps") is not None else rec.get("steps")
    guidance_scale = (
        gen_cfg.get("guidance_scale")
        if gen_cfg.get("guidance_scale") is not None
        else rec.get("guidance_scale")
    )
    strength = (
        gen_cfg.get("img2img_strength")
        if gen_cfg.get("img2img_strength") is not None
        else rec.get("strength")
    )
    label = int(rec.get("label", 1))

    insp = inspect_pipeline_sample(
        Path(image_path),
        bundle=None,
        source_image=Path(source_path) if source_path else None,
        mask_region=str(mask_region) if mask_region else None,
        mode=mode,
        prompt=prompt,
        seed=int(seed) if seed is not None else None,
        ground_truth_label=label,
        clip_scorer=clip_scorer,
        compute_heatmaps=False,
    )

    m0_prob = float(scores["M0"])
    is_hard = bool(label == 1 and insp.gate_passed and m0_prob < 0.50)

    measured = dict(insp.measured)
    if rec.get("rejection_reasons") and not insp.gate_passed:
        measured["historical_rejection_reasons"] = list(rec["rejection_reasons"])

    return ShowcaseCatalogEntry(
        sample_id=str(rec.get("image_id") or Path(image_path).stem),
        mode=mode,
        tab=tab,
        image_path=image_path,
        source_image_path=source_path,
        mask_path=None,
        mask_region=str(mask_region) if mask_region else None,
        prompt=prompt,
        seed=int(seed) if seed is not None else None,
        steps=int(steps) if steps is not None else None,
        guidance_scale=float(guidance_scale) if guidance_scale is not None else None,
        strength=float(strength) if strength is not None else None,
        label=label,
        split_provenance=split_provenance,
        gate_passed=bool(insp.gate_passed),
        rejection_reasons=list(insp.rejection_reasons),
        measured=measured,
        scores={k: float(v) for k, v in scores.items()},
        cheek_fft_ratio=measured.get("cheek_fft_ratio"),
        is_hard_sample=is_hard,
    )


def _rank_passed_candidates(
    candidates: Sequence[Tuple[Dict[str, Any], str, Dict[str, float]]],
    rng: np.random.Generator,
) -> List[Tuple[Dict[str, Any], str, Dict[str, float]]]:
    """Order passed candidates prioritizing genuine M0 < 0.50 -> M3 >= 0.50 rescues first."""
    rescues: List[Tuple[float, Tuple[Dict[str, Any], str, Dict[str, float]]]] = []
    m0_misses: List[Tuple[float, Tuple[Dict[str, Any], str, Dict[str, float]]]] = []
    regular: List[Tuple[Dict[str, Any], str, Dict[str, float]]] = []

    for cand in candidates:
        rec, prov, sc = cand
        m0 = float(sc["M0"])
        m3 = float(sc["M3"])
        prov_bonus = 1.0 if prov == "master_frozen_test" else 0.0
        if m0 < 0.50 and m3 >= 0.50:
            rescues.append(((m3 - m0) + prov_bonus * 0.01, cand))
        elif m0 < 0.50:
            m0_misses.append((m3, cand))
        else:
            regular.append(cand)

    rescues.sort(key=lambda x: x[0], reverse=True)
    m0_misses.sort(key=lambda x: x[0], reverse=True)
    perm = rng.permutation(len(regular)).tolist() if regular else []
    shuffled_regular = [regular[i] for i in perm]
    return [c for _, c in rescues] + [c for _, c in m0_misses] + shuffled_regular


def build_ui_showcase_catalog(
    output_path: Path = Path("experiments/demo_runs/ui_showcase_catalog.json"),
    per_mode_pass: int = 18,
    per_mode_fail: int = 12,
    ood_count: int = 18,
    real_count: int = 12,
    *,
    bundle: Optional[DetectorBundle] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build balanced held-out showcase catalog JSON with live M0/M2*/M3 scores and cheek_fft_ratio.

    All passed SD 1.5 entries come strictly from `master_frozen_test.jsonl` or
    `cumulative_challenge_test.jsonl` (never `d0_train_images`), failed SD 1.5 entries come from
    `sd15_pregen_200/curation_report.jsonl` (topped up from held-out `master_test_fakes_raw.jsonl`
    rejects when a mode has fewer than `per_mode_fail` rejects), RV5.1 OOD entries come from
    `unseen_rv51_test.jsonl`, and real FFHQ entries (`label == 0`) come from `master_frozen_test.jsonl`.
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    train_ids, train_paths = _collect_train_exclusions()
    real_lookup = _build_real_source_lookup()

    master_test = _read_jsonl_records(Path("data/benchmarks/master_frozen_test.jsonl"))
    challenge_test = _read_jsonl_records(Path("data/active_learning/cumulative_challenge_test.jsonl"))
    unseen_rv51 = _read_jsonl_records(Path("data/benchmarks/D3_clean/unseen_rv51_test.jsonl"))

    demo_fake_by_id = {
        str(r["image_id"]): r
        for r in _read_jsonl_records(Path("experiments/demo_runs/sd15_pregen_200/fake_manifest.jsonl"))
        if "image_id" in r
    }
    demo_rejects_raw: List[Dict[str, Any]] = []
    for rep in _read_jsonl_records(Path("experiments/demo_runs/sd15_pregen_200/curation_report.jsonl")):
        if rep.get("is_valid") is False and str(rep.get("image_id", "")) in demo_fake_by_id:
            merged = dict(demo_fake_by_id[str(rep["image_id"])])
            merged["rejection_reasons"] = list(rep.get("rejection_reasons", []))
            merged["_provenance"] = "sd15_pregen_200_rejected"
            demo_rejects_raw.append(merged)

    master_curated_ids = {
        str(r["image_id"])
        for r in _read_jsonl_records(Path("data/synthetic/master_test_fakes_curated.jsonl"))
        if "image_id" in r
    }
    master_rejects_raw: List[Dict[str, Any]] = []
    for rec in _read_jsonl_records(Path("data/synthetic/master_test_fakes_raw.jsonl")):
        if str(rec.get("image_id", "")) not in master_curated_ids:
            merged = dict(rec)
            merged["_provenance"] = "master_frozen_test_rejected"
            master_rejects_raw.append(merged)

    def _is_valid_heldout(rec: Dict[str, Any]) -> bool:
        fp = str(rec.get("file_path") or "")
        rid = str(rec.get("image_id") or "")
        return bool(fp and Path(fp).is_file() and rid not in train_ids and fp not in train_paths)

    sd_modes = ("sd_text2img", "sd_img2img", "sd_inpaint")
    pass_candidates_by_mode: Dict[str, List[Tuple[Dict[str, Any], str]]] = {m: [] for m in sd_modes}
    for rec in master_test:
        if int(rec.get("label", -1)) == 1 and str(rec.get("manipulation_type")) in pass_candidates_by_mode:
            if _is_valid_heldout(rec) and "d0_train_images" not in str(rec.get("file_path", "")):
                pass_candidates_by_mode[str(rec["manipulation_type"])].append((rec, "master_frozen_test"))
    for rec in challenge_test:
        if int(rec.get("label", -1)) == 1 and str(rec.get("manipulation_type")) in pass_candidates_by_mode:
            if _is_valid_heldout(rec) and "d0_train_images" not in str(rec.get("file_path", "")):
                pass_candidates_by_mode[str(rec["manipulation_type"])].append((rec, "cumulative_challenge_test"))

    fail_candidates_by_mode: Dict[str, List[Tuple[Dict[str, Any], str]]] = {m: [] for m in sd_modes}
    seen_fail_ids: set[str] = set()
    for rec in demo_rejects_raw + master_rejects_raw:
        m = str(rec.get("manipulation_type") or "")
        rid = str(rec.get("image_id") or "")
        if m in fail_candidates_by_mode and rid not in seen_fail_ids and _is_valid_heldout(rec):
            seen_fail_ids.add(rid)
            fail_candidates_by_mode[m].append((rec, str(rec.get("_provenance", "sd15_pregen_200_rejected"))))

    rv51_candidates = [
        (rec, "unseen_rv51_test")
        for rec in unseen_rv51
        if int(rec.get("label", -1)) == 1
        and str(rec.get("manipulation_type", "")).startswith("rv51_")
        and _is_valid_heldout(rec)
    ]
    real_candidates = [
        (rec, "master_frozen_test")
        for rec in master_test
        if int(rec.get("label", -1)) == 0 and _is_valid_heldout(rec)
    ]

    all_to_score: List[Tuple[Dict[str, Any], str]] = []
    for m in sd_modes:
        all_to_score.extend(pass_candidates_by_mode[m])
        all_to_score.extend(fail_candidates_by_mode[m])
    all_to_score.extend(rv51_candidates)
    all_to_score.extend(real_candidates)

    unique_paths: List[Path] = []
    path_to_idx: Dict[str, int] = {}
    for rec, _ in all_to_score:
        fp = str(rec["file_path"])
        if fp not in path_to_idx:
            path_to_idx[fp] = len(unique_paths)
            unique_paths.append(Path(fp))

    owns_bundle = bundle is None
    active_bundle = bundle if bundle is not None else load_detector_bundle()
    try:
        batched_scores = _score_images_batched(active_bundle, unique_paths, batch_size=16)
    finally:
        if owns_bundle:
            unload_detector_bundle(active_bundle)

    def _get_score(rec: Dict[str, Any]) -> Dict[str, float]:
        return batched_scores[path_to_idx[str(rec["file_path"])]]

    clip_scorer: Optional[Any] = None
    try:
        scorer_cls = ClipScorer if ClipScorer is not _ORIG_CLIP_SCORER else _metrics_mod.ClipScorer
        clip_scorer = scorer_cls(device="cpu")
    except Exception as exc:
        logger.warning("CLIP scorer unavailable during catalog build (%s); continuing without CLIP.", exc)

    tabs: Dict[str, List[ShowcaseCatalogEntry]] = {
        "sd_text2img": [],
        "sd_img2img": [],
        "sd_inpaint": [],
        "ood_and_real": [],
    }

    try:
        for mode in sd_modes:
            scored_pass = [
                (rec, prov, _get_score(rec)) for rec, prov in pass_candidates_by_mode[mode]
            ]
            ordered_pass = _rank_passed_candidates(scored_pass, rng)
            seen_ids: set[str] = set()
            passed_entries: List[ShowcaseCatalogEntry] = []

            for rec, prov, sc in ordered_pass:
                if len(passed_entries) >= int(per_mode_pass):
                    break
                rid = str(rec.get("image_id", ""))
                if rid in seen_ids:
                    continue
                src_path = _resolve_source_path(rec, real_lookup) if mode in ("sd_img2img", "sd_inpaint") else None
                entry = _audit_and_build_catalog_entry(
                    rec,
                    tab=mode,
                    mode=mode,
                    split_provenance=prov,
                    scores=sc,
                    source_path=src_path,
                    clip_scorer=clip_scorer,
                )
                if (
                    entry.gate_passed
                    and entry.measured.get("face_detected")
                    and entry.cheek_fft_ratio is not None
                ):
                    seen_ids.add(rid)
                    passed_entries.append(entry)

            failed_entries: List[ShowcaseCatalogEntry] = []
            for rec, prov in fail_candidates_by_mode[mode]:
                if len(failed_entries) >= int(per_mode_fail):
                    break
                rid = str(rec.get("image_id", ""))
                if rid in seen_ids:
                    continue
                src_path = _resolve_source_path(rec, real_lookup) if mode in ("sd_img2img", "sd_inpaint") else None
                entry = _audit_and_build_catalog_entry(
                    rec,
                    tab=mode,
                    mode=mode,
                    split_provenance=prov,
                    scores=_get_score(rec),
                    source_path=src_path,
                    clip_scorer=clip_scorer,
                )
                has_valid_cheek = (not entry.measured.get("face_detected")) or (entry.cheek_fft_ratio is not None)
                if not entry.gate_passed and has_valid_cheek:
                    seen_ids.add(rid)
                    failed_entries.append(entry)

            tabs[mode] = passed_entries + failed_entries

        # Tab 4: RV5.1 OOD (balanced rv51_img2img + rv51_inpaint) + Held-out Real FFHQ (label == 0)
        scored_rv51 = [(rec, prov, _get_score(rec)) for rec, prov in rv51_candidates]
        rv51_by_sub: Dict[str, List[Tuple[Dict[str, Any], str, Dict[str, float]]]] = {
            "rv51_img2img": [],
            "rv51_inpaint": [],
        }
        for cand in scored_rv51:
            sub = str(cand[0].get("manipulation_type", ""))
            if sub in rv51_by_sub:
                rv51_by_sub[sub].append(cand)
        for sub in rv51_by_sub:
            rv51_by_sub[sub].sort(
                key=lambda c: (
                    1 if (c[2]["M0"] < 0.50 and c[2]["M3"] >= 0.50) else 0,
                    float(c[2]["M3"]) - float(c[2]["M0"]),
                    float(c[2]["M3"]),
                ),
                reverse=True,
            )

        interleaved_rv51: List[Tuple[Dict[str, Any], str, Dict[str, float]]] = []
        max_len = max(len(rv51_by_sub["rv51_img2img"]), len(rv51_by_sub["rv51_inpaint"]))
        for idx in range(max_len):
            for sub in ("rv51_img2img", "rv51_inpaint"):
                if idx < len(rv51_by_sub[sub]):
                    interleaved_rv51.append(rv51_by_sub[sub][idx])

        ood_entries: List[ShowcaseCatalogEntry] = []
        seen_ood_ids: set[str] = set()
        for rec, prov, sc in interleaved_rv51:
            if len(ood_entries) >= int(ood_count):
                break
            rid = str(rec.get("image_id", ""))
            if rid in seen_ood_ids:
                continue
            entry = _audit_and_build_catalog_entry(
                rec,
                tab="ood_and_real",
                mode="rv51_ood",
                split_provenance=prov,
                scores=sc,
                source_path=None,
                clip_scorer=clip_scorer,
            )
            if entry.gate_passed and entry.measured.get("face_detected") and entry.cheek_fft_ratio is not None:
                seen_ood_ids.add(rid)
                ood_entries.append(entry)

        scored_real = [(rec, prov, _get_score(rec)) for rec, prov in real_candidates]
        scored_real.sort(
            key=lambda c: (
                1 if c[2]["M3"] < 0.50 else 0,
                1 if (c[2]["M0"] >= 0.50 and c[2]["M3"] < 0.50) else 0,
                -float(c[2]["M3"]),
            ),
            reverse=True,
        )
        real_entries: List[ShowcaseCatalogEntry] = []
        seen_real_ids: set[str] = set()
        for rec, prov, sc in scored_real:
            if len(real_entries) >= int(real_count):
                break
            rid = str(rec.get("image_id", ""))
            if rid in seen_real_ids:
                continue
            entry = _audit_and_build_catalog_entry(
                rec,
                tab="ood_and_real",
                mode="real_ffhq",
                split_provenance=prov,
                scores=sc,
                source_path=None,
                clip_scorer=clip_scorer,
            )
            if entry.gate_passed and entry.measured.get("face_detected") and entry.cheek_fft_ratio is not None:
                seen_real_ids.add(rid)
                real_entries.append(entry)

        tabs["ood_and_real"] = ood_entries + real_entries
    finally:
        if clip_scorer is not None and hasattr(clip_scorer, "close"):
            try:
                clip_scorer.close()
            except Exception:
                pass

    # Sync final selected entries to exact single-image score_image_with_bundle() precision
    selected_entries = [e for entries in tabs.values() for e in entries]
    if selected_entries:
        sync_bundle = bundle if bundle is not None else load_detector_bundle()
        try:
            exact_scores = _score_images_batched(
                sync_bundle,
                [Path(e.image_path) for e in selected_entries],
                batch_size=1,
            )
            for entry, ex in zip(selected_entries, exact_scores):
                entry.scores = {k: float(v) for k, v in ex.items()}
                entry.is_hard_sample = bool(
                    entry.label == 1 and entry.gate_passed and float(ex["M0"]) < 0.50
                )
        finally:
            if bundle is None:
                unload_detector_bundle(sync_bundle)

    summary = {
        tab_name: {
            "total": len(entries),
            "passed": sum(1 for e in entries if e.gate_passed),
            "failed": sum(1 for e in entries if not e.gate_passed),
            "hard_rescues": sum(
                1
                for e in entries
                if e.gate_passed
                and e.label == 1
                and float(e.scores.get("M0", 1.0)) < 0.50
                and float(e.scores.get("M3", 0.0)) >= 0.50
            ),
        }
        for tab_name, entries in tabs.items()
    }

    payload = {
        "version": 1,
        "checkpoint_map": dict(DEFAULT_CHECKPOINT_MAP),
        "summary": summary,
        "tabs": {
            tab_name: [entry.to_dict() for entry in entries]
            for tab_name, entries in tabs.items()
        },
    }
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Built UI showcase catalog at %s -> %s", out_p, summary)
    return payload


def load_curated_showcase_catalog(
    catalog_path: Path = Path("experiments/demo_runs/ui_showcase_catalog.json"),
) -> Dict[str, List[ShowcaseCatalogEntry]]:
    """Load precomputed held-out UI showcase catalog JSON into `ShowcaseCatalogEntry` lists per tab."""
    cat_p = Path(catalog_path)
    if not cat_p.is_file():
        raise FileNotFoundError(f"Showcase catalog not found at: {cat_p}")

    raw = json.loads(cat_p.read_text(encoding="utf-8"))
    tabs_raw = raw["tabs"] if isinstance(raw, dict) and "tabs" in raw else raw
    expected_tabs = ("sd_text2img", "sd_img2img", "sd_inpaint", "ood_and_real")
    catalog: Dict[str, List[ShowcaseCatalogEntry]] = {}
    for tab in expected_tabs:
        entries_raw = tabs_raw.get(tab, [])
        catalog[tab] = [ShowcaseCatalogEntry(**item) for item in entries_raw]
    return catalog


def get_golden_presets(
    catalog: Optional[Dict[str, List[ShowcaseCatalogEntry]]] = None,
) -> List[Dict[str, Any]]:
    """Build the 4 one-click presentation Golden Presets backed by held-out catalog entries."""
    active_catalog = catalog if catalog is not None else load_curated_showcase_catalog()

    def _pick_pass(entries: Sequence[ShowcaseCatalogEntry]) -> ShowcaseCatalogEntry:
        if not entries:
            raise ValueError("Cannot select golden preset from empty catalog pool.")
        passed = [e for e in entries if e.gate_passed]
        if not passed:
            return entries[0]
        rescues = [
            e
            for e in passed
            if float(e.scores.get("M0", 1.0)) < 0.50 and float(e.scores.get("M3", 0.0)) >= 0.50
        ]
        pool = rescues if rescues else passed
        return max(
            pool,
            key=lambda e: (
                float(e.scores.get("M3", 0.0)) - float(e.scores.get("M0", 0.0)),
                float(e.scores.get("M3", 0.0)),
            ),
        )

    def _pick_fail(
        entries: Sequence[ShowcaseCatalogEntry],
        preferred_keywords: Tuple[str, ...],
    ) -> ShowcaseCatalogEntry:
        if not entries:
            raise ValueError("Cannot select golden preset from empty catalog pool.")
        failed = [e for e in entries if not e.gate_passed]
        if not failed:
            return entries[-1]
        for kw in preferred_keywords:
            for e in failed:
                joined = " ".join(e.rejection_reasons).lower()
                if kw in joined:
                    return e
        return failed[0]

    presets: List[Dict[str, Any]] = []

    t2i_entries = active_catalog["sd_text2img"]
    t2i_pass = _pick_pass(t2i_entries)
    t2i_fail = _pick_fail(t2i_entries, ("amputated", "forehead", "multiple faces", "clip"))
    presets.append(
        {
            "id": "preset_text2img",
            "label": "Ca 1: Text2Img (Đạt vs Cụt trán/Loại)",
            "tab": "sd_text2img",
            "description": "Held-out SD 1.5 Text2Img: ca đạt chuẩn cứu bởi M3 vs ca bị loại tại cổng Data.",
            "primary_sample_id": t2i_pass.sample_id,
            "secondary_sample_id": t2i_fail.sample_id,
            "primary_entry": t2i_pass.to_dict(),
            "secondary_entry": t2i_fail.to_dict(),
            "checkpoint_metrics": dict(t2i_pass.scores),
            "secondary_metrics": dict(t2i_fail.scores),
        }
    )

    i2i_entries = active_catalog["sd_img2img"]
    i2i_pass = _pick_pass(i2i_entries)
    i2i_fail = _pick_fail(i2i_entries, ("cheek", "waxy", "blowout", "clip"))
    presets.append(
        {
            "id": "preset_img2img",
            "label": "Ca 2: Img2Img (Đạt vs Da sáp FFT)",
            "tab": "sd_img2img",
            "description": "Held-out SD 1.5 Img2Img: ca qua mặt M0 nhưng M3 bắt được vs ca da sáp/lỗi gate.",
            "primary_sample_id": i2i_pass.sample_id,
            "secondary_sample_id": i2i_fail.sample_id,
            "primary_entry": i2i_pass.to_dict(),
            "secondary_entry": i2i_fail.to_dict(),
            "checkpoint_metrics": dict(i2i_pass.scores),
            "secondary_metrics": dict(i2i_fail.scores),
        }
    )

    inp_entries = active_catalog["sd_inpaint"]
    inp_pass = _pick_pass(inp_entries)
    inp_fail = _pick_fail(inp_entries, ("multiple faces", "seam", "cielab", "eyes"))
    presets.append(
        {
            "id": "preset_inpaint",
            "label": "Ca 3: Inpaint (Kính/Môi vs 2 mặt)",
            "tab": "sd_inpaint",
            "description": "Held-out SD 1.5 Inpaint: ca mặt nạ cục bộ khó đối chiếu M0 vs M3 và ca lỗi đa mặt/viền.",
            "primary_sample_id": inp_pass.sample_id,
            "secondary_sample_id": inp_fail.sample_id,
            "primary_entry": inp_pass.to_dict(),
            "secondary_entry": inp_fail.to_dict(),
            "checkpoint_metrics": dict(inp_pass.scores),
            "secondary_metrics": dict(inp_fail.scores),
        }
    )

    ood_real_entries = active_catalog.get("ood_and_real", [])
    ood_list = [e for e in ood_real_entries if e.mode == "rv51_ood"]
    real_list = [e for e in ood_real_entries if e.label == 0]
    if not ood_list or not real_list:
        raise ValueError("Cannot select OOD/Real golden preset: missing rv51_ood or real_ffhq entries.")
    ood_primary = max(
        ood_list,
        key=lambda e: (
            1 if (float(e.scores.get("M0", 1.0)) < 0.50 and float(e.scores.get("M3", 0.0)) >= 0.50) else 0,
            float(e.scores.get("M3", 0.0)) - float(e.scores.get("M0", 0.0)),
            float(e.scores.get("M3", 0.0)),
        ),
    )
    real_secondary = min(real_list, key=lambda e: float(e.scores.get("M3", 1.0)))
    presets.append(
        {
            "id": "preset_ood_real",
            "label": "Ca 4: OOD RV5.1 & Real FFHQ",
            "tab": "ood_and_real",
            "description": "Khái quát hóa ngoại miền Unseen RV5.1 (M3 bắt giả) đối chứng ảnh thật FFHQ (True Negative).",
            "primary_sample_id": ood_primary.sample_id,
            "secondary_sample_id": real_secondary.sample_id,
            "primary_entry": ood_primary.to_dict(),
            "secondary_entry": real_secondary.to_dict(),
            "checkpoint_metrics": dict(ood_primary.scores),
            "secondary_metrics": dict(real_secondary.scores),
        }
    )

    return presets


def main() -> None:
    """CLI entrypoint to resync `sd15_pregen_200` checkpoint scores and build UI showcase catalog."""
    parser = argparse.ArgumentParser(description="Resync demo run scores and build held-out UI showcase catalog.")
    parser.add_argument("--resync-demo", action="store_true", help="Re-score all 200 sd15_pregen_200 samples.")
    parser.add_argument("--build-catalog", action="store_true", help="Build held-out ui_showcase_catalog.json.")
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="experiments/demo_runs/sd15_pregen_200",
        help="Demo run directory to resync.",
    )
    parser.add_argument("--threshold", type=float, default=0.50, help="Hard-sample detector threshold tau.")
    parser.add_argument(
        "--catalog-output",
        type=str,
        default="experiments/demo_runs/ui_showcase_catalog.json",
        help="Output path for the UI showcase catalog JSON.",
    )
    args = parser.parse_args()

    if not args.resync_demo and not args.build_catalog:
        parser.print_help()
        return

    if args.resync_demo:
        resync_demo_run_scores(
            demo_dir=Path(args.demo_dir),
            threshold=float(args.threshold),
        )
    if args.build_catalog:
        build_ui_showcase_catalog(
            output_path=Path(args.catalog_output),
        )


if __name__ == "__main__":
    main()
