"""Visual overlay renderers, threshold recalibration, and DINOv2 patch heatmap helpers for UI showcase."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import html
import io
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.common.logger import get_logger

logger = get_logger(__name__)

_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


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


def _to_512_rgb_u8(image: np.ndarray) -> np.ndarray:
    """Resize any (H, W, C) RGB array to (512, 512, 3) uint8 via LANCZOS."""
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
        raise ValueError(f"Expected 2D/3D mask array, got shape {mask.shape}")

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
