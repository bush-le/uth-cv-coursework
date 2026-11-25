"""Multi-modal automated curation gate for synthetic deepfake image datasets."""

from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np
from PIL import Image
import torch

from src.analysis.eda import _compute_dhash
from src.common.config import cfg_get, load_config
from src.common.logger import get_logger
from src.common.metrics import (
    ClipScorer,
    audit_image_extended,
    calibrate_threshold,
    compute_laplacian_sharpness,
    detect_faces,
    resolve_effective_min_clip,
    verify_facial_geometry,
)
from src.data.provenance import file_sha256
from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask

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
    eye_count: Optional[int] = None
    sunglasses_occluded: Optional[bool] = None
    bilateral_symmetry: Optional[float] = None
    fft_high_freq_ratio: Optional[float] = None
    texture_flag: Optional[str] = None
    delta_e_lab: Optional[float] = None
    structure_corr: Optional[float] = None
    seam_diff: Optional[float] = None
    is_seamless: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "image_id": self.image_id,
            "is_valid": self.is_valid,
            "sharpness_score": self.sharpness_score,
            "clip_score": self.clip_score,
            "face_detected": self.face_detected,
            "rejection_reasons": self.rejection_reasons,
        }
        for key in (
            "eye_count",
            "sunglasses_occluded",
            "bilateral_symmetry",
            "fft_high_freq_ratio",
            "texture_flag",
            "delta_e_lab",
            "structure_corr",
            "seam_diff",
            "is_seamless",
        ):
            val = getattr(self, key)
            if val is not None:
                payload[key] = val
        return payload


class ImageCurationGate:
    """Automated multi-modal quality filter for synthetic face images."""

    def __init__(
        self,
        min_sharpness: float = 1.25,
        min_clip_score: float = 0.20,
        clip_enabled: bool = False,
        clip_model_id: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        face_backend: str = "none",
        dedup: bool = False,
        perceptual_dedup_threshold: Optional[int] = None,
        extended_audit: bool = False,
    ) -> None:
        self.min_sharpness = min_sharpness
        self.min_clip_score = min_clip_score
        self.clip_enabled = clip_enabled
        self.face_backend = face_backend
        self.dedup = dedup
        self.perceptual_dedup_threshold = perceptual_dedup_threshold
        self.extended_audit = extended_audit
        resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self._clip_scorer: Optional[ClipScorer] = None
        if self.clip_enabled:
            self._clip_scorer = ClipScorer(model_id=clip_model_id, device=resolved_device)

    def evaluate_image(
        self,
        image: torch.Tensor,
        prompt: str = "portrait",
        face_detected: Optional[bool] = None,
        image_id: str = "unknown",
        bbox_xyxy: Optional[Sequence[float]] = None,
        mode: str = "sd_text2img",
        source_image: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        clip_score_override: Optional[float] = None,
    ) -> ImageCurationReport:
        """Evaluate a synthetic face image against sharpness, CLIP alignment, face detection, and optional extended audit."""
        sharpness = compute_laplacian_sharpness(image, bbox_xyxy=bbox_xyxy)

        has_prompt = bool(prompt and str(prompt).strip())
        if self.extended_audit and face_detected is None and self.face_backend != "none":
            clip_score: Optional[float] = clip_score_override if has_prompt else None
            if clip_score is None and has_prompt and self.clip_enabled and self._clip_scorer is not None:
                clip_score = self._clip_scorer.score(image, prompt)
            ext_rep = audit_image_extended(
                image,
                mode=mode,
                prompt=prompt,
                source_image=source_image,
                mask=mask,
                clip_scorer=None,
                min_clip_score=self.min_clip_score,
                min_sharpness=self.min_sharpness,
                image_id=image_id,
                run_geometry=True,
                sharpness_score=sharpness,
                face_backend=self.face_backend,
                clip_score_override=clip_score,
            )
            reasons = list(ext_rep.rejection_reasons)
            face_detected_val = not any("No face" in r or "No valid face" in r for r in reasons)
            return ImageCurationReport(
                image_id=image_id,
                is_valid=len(reasons) == 0,
                sharpness_score=sharpness,
                clip_score=clip_score,
                face_detected=face_detected_val,
                rejection_reasons=reasons,
                eye_count=ext_rep.eye_count,
                sunglasses_occluded=ext_rep.sunglasses_occluded,
                bilateral_symmetry=ext_rep.bilateral_symmetry,
                fft_high_freq_ratio=ext_rep.fft_high_freq_ratio,
                texture_flag=ext_rep.texture_flag,
                delta_e_lab=ext_rep.delta_e_lab,
                structure_corr=ext_rep.structure_corr,
                seam_diff=ext_rep.seam_diff,
                is_seamless=ext_rep.is_seamless,
            )

        if face_detected is None:
            if self.face_backend == "none":
                face_detected_val = True
            else:
                img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                n_faces = detect_faces(img_np, backend=self.face_backend)
                face_detected_val = n_faces > 0
        else:
            face_detected_val = face_detected

        clip_score = clip_score_override if has_prompt else None
        if clip_score is None and has_prompt and self.clip_enabled and self._clip_scorer is not None:
            clip_score = self._clip_scorer.score(image, prompt)

        reasons = []
        if not face_detected_val:
            reasons.append("No valid face landmark detected")
        elif self.face_backend != "none":
            geom_valid, geom_reasons = verify_facial_geometry(image, face_backend=self.face_backend)
            if not geom_valid:
                reasons.extend(geom_reasons)
        if sharpness < self.min_sharpness:
            reasons.append(f"Sharpness {sharpness:.2f} below threshold {self.min_sharpness:.2f}")
        eff_min_clip = resolve_effective_min_clip(self.min_clip_score, mode)
        if has_prompt and self.clip_enabled and clip_score is not None and clip_score < eff_min_clip:
            reasons.append(f"CLIP score {clip_score:.3f} below threshold {eff_min_clip:.3f}")

        ext_kwargs: Dict[str, Any] = {}
        if self.extended_audit:
            ext_rep = audit_image_extended(
                image,
                mode=mode,
                prompt=prompt,
                source_image=source_image,
                mask=mask,
                clip_scorer=None,
                min_sharpness=self.min_sharpness,
                image_id=image_id,
                run_geometry=(self.face_backend != "none"),
                sharpness_score=sharpness,
                face_backend=self.face_backend,
            )
            ext_kwargs = {
                "eye_count": ext_rep.eye_count,
                "sunglasses_occluded": ext_rep.sunglasses_occluded,
                "bilateral_symmetry": ext_rep.bilateral_symmetry,
                "fft_high_freq_ratio": ext_rep.fft_high_freq_ratio,
                "texture_flag": ext_rep.texture_flag,
                "delta_e_lab": ext_rep.delta_e_lab,
                "structure_corr": ext_rep.structure_corr,
                "seam_diff": ext_rep.seam_diff,
                "is_seamless": ext_rep.is_seamless,
            }
            for r in ext_rep.rejection_reasons:
                if self.face_backend == "none" and any(
                    k in r for k in ("No face", "Multiple faces", "Abnormal facial", "Face ", "Forehead", "Chin", "Severe ", "Expected >= 2 eyes", "Low bilateral")
                ):
                    continue
                if r not in reasons and not r.startswith("Sharpness "):
                    reasons.append(r)

        is_valid = len(reasons) == 0
        return ImageCurationReport(
            image_id=image_id,
            is_valid=is_valid,
            sharpness_score=sharpness,
            clip_score=clip_score,
            face_detected=face_detected_val,
            rejection_reasons=reasons,
            **ext_kwargs,
        )


_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_MODE_MAP = {
    "text2img": "sd_text2img",
    "img2img": "sd_img2img",
    "inpaint": "sd_inpaint",
}
_SUBREGION_MAP = {"mouth": "lips", "eyes": "periocular"}


def calibrate_sharpness_from_manifest(
    raw_manifest: Path,
    ref_quantile: float = 0.05,
    max_samples: int = 50,
    fallback: float = 10.0,
) -> float:
    """Calibrate Laplacian sharpness floor from genuine reference images."""
    if not Path(raw_manifest).exists():
        return fallback
    real_sharpness: List[float] = []
    with open(raw_manifest, "r", encoding="utf-8") as rf:
        for line in rf:
            if not line.strip():
                continue
            item = json.loads(line)
            fp = item.get("file_path") or item.get("image_path")
            if fp and Path(fp).exists():
                try:
                    with Image.open(fp) as img:
                        np_img = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
                    real_sharpness.append(compute_laplacian_sharpness(tensor_img))
                    if len(real_sharpness) >= max_samples:
                        break
                except Exception:
                    pass
    return calibrate_threshold(real_sharpness, ref_quantile) if real_sharpness else fallback


def curate_image_manifest(
    input_manifest: Path,
    output_manifest: Path,
    gate: ImageCurationGate,
    curation_report: Optional[Path] = None,
    raw_manifest: Optional[Path] = None,
    source_dirs: Optional[Sequence[Union[str, Path]]] = None,
) -> Dict[str, Any]:
    """Filter raw image manifest entries (SD 1.5 and RV5.1 schemas) through the curation gate."""
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if curation_report is not None:
        curation_report.parent.mkdir(parents=True, exist_ok=True)

    raw_lookup: Dict[str, Path] = {}
    if gate.extended_audit and raw_manifest is not None and Path(raw_manifest).exists():
        with open(raw_manifest, "r", encoding="utf-8") as rf:
            for line in rf:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = rec.get("image_id")
                rfp = rec.get("file_path") or rec.get("image_path")
                if rid and rfp and Path(rfp).exists():
                    raw_lookup[str(rid)] = Path(rfp)

    resolved_src_dirs: List[Path] = []
    if source_dirs is not None:
        resolved_src_dirs = [Path(d) for d in source_dirs if Path(d).is_dir()]
    else:
        for default_dir in (Path("data/raw/faces/celeba_hq"), Path("data/raw/faces/ffhq")):
            if default_dir.is_dir():
                resolved_src_dirs.append(default_dir)

    kept_count = 0
    rejected_count = 0
    curated_lines = []
    report_lines = []
    seen_hashes: Set[str] = set()
    seen_dhashes_u64: List[int] = []
    seen_dhash_meta: List[Tuple[str, str, Optional[np.ndarray]]] = []
    source_cache: Dict[Path, np.ndarray] = {}
    processed_idx = 0

    pending_batch: List[Tuple[Dict[str, Any], torch.Tensor, str, str, Optional[np.ndarray], Optional[torch.Tensor], str]] = []

    def _flush_pending_batch() -> None:
        nonlocal kept_count, rejected_count
        if not pending_batch:
            return
        batch_items = list(pending_batch)
        pending_batch.clear()
        clip_scores: List[Optional[float]] = [None] * len(batch_items)
        if gate.clip_enabled and gate._clip_scorer is not None:
            prompted_indices = [idx for idx, row in enumerate(batch_items) if str(row[6] or "").strip()]
            if prompted_indices:
                imgs_b = [batch_items[idx][1] for idx in prompted_indices]
                prompts_b = [batch_items[idx][6] for idx in prompted_indices]
                try:
                    scored = gate._clip_scorer.score_batch(imgs_b, prompts_b)
                    for idx, val in zip(prompted_indices, scored):
                        clip_scores[idx] = float(val)
                except Exception as clip_err:
                    logger.warning("Batched CLIP scoring error (%s), falling back per-item.", clip_err)
                    for idx in prompted_indices:
                        try:
                            clip_scores[idx] = float(gate._clip_scorer.score(batch_items[idx][1], batch_items[idx][6]))
                        except Exception:
                            clip_scores[idx] = None
        for (b_item, b_ten, b_id, b_mode, b_src, b_mask, b_prompt), b_clip in zip(batch_items, clip_scores):
            try:
                b_rep = gate.evaluate_image(
                    b_ten,
                    prompt=b_prompt,
                    image_id=b_id,
                    mode=b_mode,
                    source_image=b_src,
                    mask=b_mask,
                    clip_score_override=b_clip,
                )
            except Exception as eval_err:
                logger.warning("Evaluation error on %s (%s), rejecting entry.", b_id, eval_err)
                b_rep = ImageCurationReport(
                    image_id=b_id,
                    is_valid=False,
                    sharpness_score=0.0,
                    clip_score=b_clip,
                    face_detected=False,
                    rejection_reasons=[f"eval_error: {eval_err}"],
                )
            report_lines.append(json.dumps(b_rep.to_dict()) + "\n")
            if b_rep.is_valid:
                curated_lines.append(json.dumps(b_item) + "\n")
                kept_count += 1
            else:
                rejected_count += 1

    if input_manifest.exists():
        with open(input_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                processed_idx += 1
                if processed_idx % 1000 == 0:
                    logger.info("Curation progress: processed %d entries from %s...", processed_idx, input_manifest.name)
                item = json.loads(line)
                file_path = item.get("file_path") or item.get("image_path")
                img_id = item.get("image_id") or (Path(str(file_path)).stem if file_path else "unknown")

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

                # Exact byte dedup check
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

                raw_mode = str(
                    item.get("manipulation_type")
                    or item.get("task")
                    or (item.get("generation") or {}).get("mode")
                    or "sd_text2img"
                )
                mode_val = _MODE_MAP.get(raw_mode, raw_mode)
                cur_src = str(item.get("source_real_id") or item.get("source_real_image") or "")

                try:
                    with Image.open(file_path) as img:
                        rgb_pil = img.convert("RGB")
                        dhash_val = int(_compute_dhash(rgb_pil), 16) if gate.perceptual_dedup_threshold is not None else None
                        thumb32_u8 = (
                            np.asarray(rgb_pil.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.uint8)
                            if (dhash_val is not None and cur_src and mode_val in ("sd_img2img", "sd_inpaint"))
                            else None
                        )
                        np_img = np.array(rgb_pil, dtype=np.float32) / 255.0
                        tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()

                    # Source-aware perceptual dHash near-duplicate check via vectorized uint64 popcount
                    if dhash_val is not None and gate.perceptual_dedup_threshold is not None:
                        is_perceptual_dup = False
                        if seen_dhashes_u64:
                            arr_seen = np.array(seen_dhashes_u64, dtype=np.uint64)
                            xor_bytes = (arr_seen ^ np.uint64(dhash_val)).view(np.uint8).reshape(-1, 8)
                            dists = _POPCOUNT_LUT[xor_bytes].sum(axis=1)
                            hit_idxs = np.flatnonzero(dists <= gate.perceptual_dedup_threshold)
                            for h_idx in hit_idxs:
                                prev_src, prev_mode, prev_thumb = seen_dhash_meta[int(h_idx)]
                                if (
                                    cur_src
                                    and prev_src == cur_src
                                    and mode_val in ("sd_img2img", "sd_inpaint")
                                    and prev_mode in ("sd_img2img", "sd_inpaint")
                                    and thumb32_u8 is not None
                                    and prev_thumb is not None
                                ):
                                    diff_p95 = float(
                                        np.percentile(
                                            np.abs(thumb32_u8.astype(np.float32) - prev_thumb.astype(np.float32)),
                                            95,
                                        )
                                    )
                                    if diff_p95 <= 2.5:
                                        is_perceptual_dup = True
                                        break
                                else:
                                    is_perceptual_dup = True
                                    break
                        if is_perceptual_dup:
                            rejected_count += 1
                            report = ImageCurationReport(
                                image_id=img_id,
                                is_valid=False,
                                sharpness_score=0.0,
                                clip_score=None,
                                face_detected=False,
                                rejection_reasons=["duplicate_perceptual_dhash"],
                            )
                            report_lines.append(json.dumps(report.to_dict()) + "\n")
                            continue
                        seen_dhashes_u64.append(dhash_val)
                        seen_dhash_meta.append((cur_src, mode_val, thumb32_u8))

                    src_u8 = None
                    mask_t = None
                    if gate.extended_audit and mode_val in ("sd_img2img", "sd_inpaint"):
                        src_path: Optional[Path] = None
                        src_id = item.get("source_real_id")
                        src_rel = item.get("source_real_image")
                        if src_id and str(src_id) in raw_lookup:
                            src_path = raw_lookup[str(src_id)]
                        elif src_rel:
                            cand_p = Path(str(src_rel))
                            if cand_p.is_file():
                                src_path = cand_p
                            else:
                                for d in resolved_src_dirs:
                                    if (d / str(src_rel)).is_file():
                                        src_path = d / str(src_rel)
                                        break
                        if src_path is not None:
                            src_u8 = source_cache.get(src_path)
                            if src_u8 is None:
                                with Image.open(src_path) as s_im:
                                    s_rgb = s_im.convert("RGB")
                                    if s_rgb.size != (tensor_img.shape[2], tensor_img.shape[1]):
                                        s_rgb = s_rgb.resize((tensor_img.shape[2], tensor_img.shape[1]), Image.Resampling.BILINEAR)
                                    src_u8 = np.array(s_rgb, dtype=np.uint8)
                                if len(source_cache) >= 512:
                                    source_cache.clear()
                                source_cache[src_path] = src_u8
                        if mode_val == "sd_inpaint" and src_u8 is not None:
                            gen_meta = item.get("generation") or {}
                            raw_box = item.get("bbox") or gen_meta.get("inpaint_mask_box")
                            bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
                            if raw_box and len(raw_box) == 4:
                                box_t = tuple(float(v) for v in raw_box)
                                if tuple(round(v, 2) for v in box_t) not in ((0.3, 0.25, 0.7, 0.75), (0.25, 0.25, 0.75, 0.75)):
                                    bbox_xyxy = box_t  # [xmin, ymin, xmax, ymax] / pixel xyxy expected by generate_dynamic_face_mask
                            raw_region = gen_meta.get("inpaint_mask_target") or item.get("sub_region")
                            src_t = torch.from_numpy(src_u8.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
                            if raw_region:
                                target_region = _SUBREGION_MAP.get(str(raw_region), str(raw_region))
                                mask_t = generate_dynamic_face_mask(
                                    src_t,
                                    bbox=bbox_xyxy,
                                    target_region=target_region,
                                )
                            else:
                                diff_mask = (
                                    np.mean(np.abs(np.rint(np_img * 255.0) - src_u8.astype(np.float32)), axis=2) > 15.0
                                )
                                if np.any(diff_mask):
                                    best_iou = -1.0
                                    best_mask: Optional[torch.Tensor] = None
                                    for cand_reg in ("full", "periocular", "lips"):
                                        cand_m = generate_dynamic_face_mask(src_t, bbox=bbox_xyxy, target_region=cand_reg)
                                        cand_bin = cand_m[0].numpy() > 0.5
                                        inter = float(np.sum(cand_bin & diff_mask))
                                        union = float(np.sum(cand_bin | diff_mask)) + 1e-6
                                        iou = inter / union
                                        if iou > best_iou:
                                            best_iou = iou
                                            best_mask = cand_m
                                    mask_t = best_mask
                                if mask_t is None:
                                    mask_t = generate_dynamic_face_mask(src_t, bbox=bbox_xyxy, target_region="full")

                    pending_batch.append((item, tensor_img, img_id, mode_val, src_u8, mask_t, item.get("prompt", "")))
                    if not (gate.clip_enabled and gate._clip_scorer is not None) or len(pending_batch) >= 32:
                        _flush_pending_batch()
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

    _flush_pending_batch()

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
    parser.add_argument("--extended-audit", action="store_true", default=None)
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
    min_clip_score = float(_res(args.min_clip, "curation.clip.min_score", 0.20))
    face_backend = str(_res(None, "curation.face_check.backend", "yunet"))
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
            min_sharpness_val = calibrate_sharpness_from_manifest(raw_manifest, ref_quantile=ref_quantile)
            logger.info("Calibrated min_sharpness: %.2f", min_sharpness_val)
        else:
            min_sharpness_val = 10.0

    extended_audit = bool(_res(args.extended_audit, "curation.extended_audit", False))
    perceptual_thresh = _res(None, "curation.perceptual_dedup_threshold", None)
    gate = ImageCurationGate(
        min_sharpness=min_sharpness_val,
        min_clip_score=min_clip_score,
        clip_enabled=clip_enabled,
        clip_model_id=clip_model_id,
        face_backend=face_backend,
        dedup=dedup,
        perceptual_dedup_threshold=int(perceptual_thresh) if perceptual_thresh is not None else None,
        extended_audit=extended_audit,
    )
    curate_image_manifest(
        input_manifest=in_manifest,
        output_manifest=out_manifest,
        gate=gate,
        curation_report=curation_report,
        raw_manifest=raw_manifest,
    )


def curate_dataset(
    input_manifest: Path,
    output_manifest: Path,
    report_path: Optional[Path] = None,
    curation_report: Optional[Path] = None,
    gate: Optional[ImageCurationGate] = None,
    min_sharpness: Optional[float] = None,
    face_backend: str = "yunet",
    clip_enabled: Optional[bool] = None,
    min_clip_score: float = 0.20,
    dedup: bool = True,
    perceptual_dedup_threshold: Optional[int] = None,
    extended_audit: bool = False,
    raw_manifest: Optional[Path] = None,
    reference_quantile_sharpness: float = 0.05,
) -> Dict[str, Any]:
    """High-level wrapper to curate a synthetic dataset manifest with optional sharpness auto-calibration."""
    if min_sharpness is None:
        if raw_manifest is not None and Path(raw_manifest).exists():
            min_sharpness_val = calibrate_sharpness_from_manifest(
                Path(raw_manifest), ref_quantile=reference_quantile_sharpness, fallback=1.25
            )
        else:
            min_sharpness_val = 1.25
    else:
        min_sharpness_val = float(min_sharpness)

    resolved_clip = (face_backend != "none") if clip_enabled is None else bool(clip_enabled)
    if gate is None:
        gate = ImageCurationGate(
            min_sharpness=min_sharpness_val,
            min_clip_score=min_clip_score,
            face_backend=face_backend,
            clip_enabled=resolved_clip,
            device="auto",
            dedup=dedup,
            perceptual_dedup_threshold=perceptual_dedup_threshold,
            extended_audit=extended_audit,
        )
    rep = report_path or curation_report
    return curate_image_manifest(
        input_manifest=Path(input_manifest),
        output_manifest=Path(output_manifest),
        gate=gate,
        curation_report=Path(rep) if rep else None,
        raw_manifest=Path(raw_manifest) if raw_manifest else None,
    )


if __name__ == "__main__":
    main()
