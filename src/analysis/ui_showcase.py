"""Visual overlay renderers, threshold recalibration, and DINOv2 patch heatmap helpers for UI showcase."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import gc
import html
import io
import json
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
from src.models.detector.image_detector import DeepfakeImageDetector
from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask

logger = get_logger(__name__)

_ORIG_CLIP_SCORER = ClipScorer
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
_MIN_CUDA_FREE_BYTES: int = 900 * 1024 * 1024
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

            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
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

    Returns:
        Populated `InspectionResult` dataclass.
    """
    t_total_start = time.perf_counter()
    latencies_ms: Dict[str, float] = {}
    warnings: List[str] = []

    t_pre = time.perf_counter()
    rgb_512 = _to_512_rgb_u8(target_image)
    source_512 = _to_512_rgb_u8(source_image) if source_image is not None else None
    mask_512 = _resolve_inpaint_mask_512(rgb_512, source_512, mask, mask_region, mode)
    latencies_ms["preprocess"] = round((time.perf_counter() - t_pre) * 1000.0, 2)

    t_cur = time.perf_counter()
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

    eff_min_clip = float(min_clip) if min_clip is not None else float(resolve_effective_min_clip(0.20, mode))
    p_low = str(prompt or "").lower()
    has_sg_intent = mode == "sd_inpaint" and any(
        tok in p_low for tok in ("sunglasses", "dark glasses", "aviator", "tinted glasses", "shades")
    )
    eff_max_delta_e = max(float(max_delta_e), 45.0) if has_sg_intent else float(max_delta_e)

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

    measured: Dict[str, Any] = {
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
    if source_512 is not None and mask_512 is not None:
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
    if source_512 is not None:
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
    if should_run_detector and bundle is not None:
        t_det = time.perf_counter()
        detector_scores = score_image_with_bundle(bundle, rgb_512)
        latencies_ms["detector"] = round((time.perf_counter() - t_det) * 1000.0, 2)

        m0_prob = detector_scores.get("M0")
        is_hard_sample = bool(
            m0_prob is not None
            and m0_prob < float(tau)
            and mode != "real_ffhq"
            and (ground_truth_label is None or int(ground_truth_label) == 1)
        )

        if compute_heatmaps:
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

    metadata: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    result: Dict[str, Any] = dict(metadata)
    result["output_dir"] = str(run_dir)
    result["output_path"] = str(out_img_path)
    if (run_dir / "input.png").is_file():
        result["input_path"] = str(run_dir / "input.png")
    if (run_dir / "mask.png").is_file():
        result["mask_path"] = str(run_dir / "mask.png")
    return result
