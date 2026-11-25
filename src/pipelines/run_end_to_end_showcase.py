"""Unified End-to-End Showcase Pipeline: Generate (N SD1.5 images) -> Curate & Extended Audit -> Multi-Model Scoring."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from PIL import Image
import torch

from src.analysis.eda import _compute_dhash, _hamming_distance
from src.common.config import cfg_get, load_config, load_yaml_with_defaults
from src.common.logger import get_logger
from src.common.metrics import calibrate_threshold, compute_laplacian_sharpness
from src.common.seed import seed_everything
from src.data.provenance import file_sha256
from src.models.detector.image_detector import DeepfakeImageDetector
from src.pipelines.curate_image_dataset import (
    ImageCurationGate,
    calibrate_sharpness_from_manifest,
    curate_image_manifest,
)
from src.pipelines.generate_deepfake_images import run_image_generation

logger = get_logger(__name__)

DEFAULT_CHECKPOINTS: Dict[str, str] = {
    "M0 (Baseline)": "experiments/checkpoints/baseline_d0_best.pt",
    "M2* (Active R2)": "experiments/checkpoints/active_learning_m2_best.pt",
    "M3 (Champion)": "experiments/checkpoints/active_learning_m3_best.pt",
}


def compute_mode_quotas(total_images: int) -> Dict[str, int]:
    """Divide total_images evenly across the three SD 1.5 generation modes."""
    if total_images < 1:
        raise ValueError(f"total_images must be >= 1, got {total_images}")
    modes = ("sd_img2img", "sd_inpaint", "sd_text2img")
    base, rem = divmod(int(total_images), len(modes))
    return {m: base + (1 if i < rem else 0) for i, m in enumerate(modes)}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _score_manifest_with_checkpoints(
    items: List[Dict[str, Any]],
    checkpoints: Dict[str, str],
    config_path: Union[str, Path] = "configs/config.yaml",
    batch_size: int = 16,
) -> Dict[str, List[float]]:
    """Run batched dinoViT inference at 512px LANCZOS across verified checkpoints."""
    if not items:
        return {name: [] for name in checkpoints}

    cfg = load_yaml_with_defaults(str(config_path)) if Path(config_path).is_file() else {}
    backbone = cfg.get("detection", {}).get("model", {}).get("spatial_backbone", "dinov2_small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-decode tensors once in (N, 3, 512, 512) RGB [0.0, 1.0] to avoid repeated disk I/O
    tensors: List[torch.Tensor] = []
    for item in items:
        fp = Path(item.get("file_path") or item.get("image_path") or "")
        with Image.open(fp) as im:
            rgb = im.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            arr = np.array(rgb, dtype=np.float32) / 255.0  # (H, W, 3) RGB
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))  # (3, H, W)
    all_images = torch.stack(tensors, dim=0)  # (N, 3, 512, 512)

    scores_by_model: Dict[str, List[float]] = {}
    for model_label, ckpt_path_str in checkpoints.items():
        ckpt_file = Path(ckpt_path_str)
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Verified checkpoint not found at: {ckpt_file}")

        model = DeepfakeImageDetector(backbone=backbone, num_classes=1, pretrained=False)
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        probs: List[float] = []
        with torch.no_grad():
            for start in range(0, len(all_images), batch_size):
                batch = all_images[start : start + batch_size].to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(batch).view(-1)
                    batch_probs = torch.sigmoid(logits).detach().cpu().tolist()
                probs.extend(float(p) for p in batch_probs)

        scores_by_model[model_label] = probs
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return scores_by_model


def _compute_diversity_metrics(items: List[Dict[str, Any]], max_dhash_samples: int = 90) -> Dict[str, Any]:
    """Compute SHA-256 uniqueness rate and pairwise 64-bit dHash Hamming distance."""
    if not items:
        return {"sha256_unique_rate": 0.0, "dhash_mean": 0.0, "dhash_min": 0}

    hashes = set()
    valid_files = 0
    for it in items:
        fp = Path(it.get("file_path") or it.get("image_path") or "")
        if fp.is_file():
            hashes.add(file_sha256(fp))
            valid_files += 1

    step = max(1, len(items) // max_dhash_samples)
    sampled = items[::step][:max_dhash_samples]
    dhashes: List[str] = []
    for it in sampled:
        fp = Path(it.get("file_path") or it.get("image_path") or "")
        if fp.is_file():
            with Image.open(fp) as im:
                dhashes.append(_compute_dhash(im.convert("RGB")))

    pair_dists = [_hamming_distance(h1, h2) for h1, h2 in combinations(dhashes, 2)]
    return {
        "sha256_unique_rate": (len(hashes) / valid_files) if valid_files else 0.0,
        "dhash_mean": float(np.mean(pair_dists)) if pair_dists else 0.0,
        "dhash_min": int(np.min(pair_dists)) if pair_dists else 0,
    }


def _aggregate_quality_stats(
    raw_items: List[Dict[str, Any]],
    report_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate curation pass rates, rejection causes, and extended audit metrics overall and by mode."""
    rep_by_id = {r["image_id"]: r for r in report_items if "image_id" in r}
    rejection_counts: Counter[str] = Counter()
    by_mode: Dict[str, List[Dict[str, Any]]] = {}

    mode_alias = {"text2img": "sd_text2img", "img2img": "sd_img2img", "inpaint": "sd_inpaint"}
    for item in raw_items:
        raw_mode = str(
            item.get("manipulation_type")
            or item.get("task")
            or (item.get("generation") or {}).get("mode")
            or "unknown"
        )
        mode = mode_alias.get(raw_mode, raw_mode)
        fp_str = item.get("file_path") or item.get("image_path") or ""
        img_id = item.get("image_id") or (Path(fp_str).stem if fp_str else "")
        rep = rep_by_id.get(img_id)
        if rep is None:
            continue
        by_mode.setdefault(mode, []).append(rep)
        for reason in rep.get("rejection_reasons", []):
            key = reason.split(":")[0].split("(")[0].strip()
            key = re.sub(r"\s+\d+(?:\.\d+)?\s+(?:exceeds|below)\s+\d+(?:\.\d+)?$", " threshold", key)
            rejection_counts[key] += 1

    def _summarize(reps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not reps:
            return {}
        n = len(reps)
        valid_n = sum(1 for r in reps if r.get("is_valid"))
        sharp = [float(r["sharpness_score"]) for r in reps if r.get("sharpness_score") is not None]
        sym = [float(r["bilateral_symmetry"]) for r in reps if r.get("bilateral_symmetry") is not None]
        fft = [float(r["fft_high_freq_ratio"]) for r in reps if r.get("fft_high_freq_ratio") is not None]
        de = [float(r["delta_e_lab"]) for r in reps if r.get("delta_e_lab") is not None]
        corr = [float(r["structure_corr"]) for r in reps if r.get("structure_corr") is not None]
        seam = [float(r["seam_diff"]) for r in reps if r.get("seam_diff") is not None]
        eye_ok = sum(
            1 for r in reps if (r.get("eye_count", 0) >= 2) or r.get("sunglasses_occluded") is True
        )
        fft_ok = sum(1 for r in reps if r.get("texture_flag") == "normal")
        return {
            "total": n,
            "passed": valid_n,
            "rejected": n - valid_n,
            "pass_rate": valid_n / n,
            "face_detected_rate": sum(1 for r in reps if r.get("face_detected")) / n,
            "sharpness_mean": float(np.mean(sharp)) if sharp else 0.0,
            "eye_pair_pass_rate": eye_ok / n if sym else None,
            "sunglasses_count": sum(1 for r in reps if r.get("sunglasses_occluded")),
            "bilateral_symmetry_mean": float(np.mean(sym)) if sym else None,
            "fft_high_freq_mean": float(np.mean(fft)) if fft else None,
            "fft_normal_rate": fft_ok / n if fft else None,
            "delta_e_lab_mean": float(np.mean(de)) if de else None,
            "structure_corr_mean": float(np.mean(corr)) if corr else None,
            "seam_diff_mean": float(np.mean(seam)) if seam else None,
        }

    return {
        "overall": _summarize(report_items),
        "by_mode": {m: _summarize(reps) for m, reps in by_mode.items()},
        "rejection_counts": dict(rejection_counts.most_common()),
    }


def _sample_pregenerated_manifest(
    pregenerated_manifest: Path,
    quotas: Dict[str, int],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample N balanced records across SD 1.5 modes from a pre-generated manifest."""
    items = [r for r in _read_jsonl(pregenerated_manifest) if Path(r.get("file_path", "")).is_file()]
    if not items:
        raise ValueError(f"No valid images found in pregenerated manifest: {pregenerated_manifest}")

    rng = np.random.default_rng(seed)
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for rec in items:
        mode = str(rec.get("manipulation_type") or (rec.get("generation") or {}).get("mode") or "unknown")
        by_mode.setdefault(mode, []).append(rec)

    selected: List[Dict[str, Any]] = []
    seen_ids = set()
    for mode, target_cnt in quotas.items():
        pool = list(by_mode.get(mode, []))
        if not pool or target_cnt <= 0:
            continue
        if len(pool) > target_cnt:
            chosen_idx = sorted(rng.choice(len(pool), size=target_cnt, replace=False).tolist())
            chosen = [pool[i] for i in chosen_idx]
        else:
            chosen = pool
        for rec in chosen:
            rid = rec.get("image_id")
            if rid not in seen_ids:
                selected.append(rec)
                seen_ids.add(rid)

    target_total = sum(quotas.values())
    if len(selected) < target_total:
        for rec in items:
            rid = rec.get("image_id")
            if rid not in seen_ids:
                selected.append(rec)
                seen_ids.add(rid)
                if len(selected) >= target_total:
                    break
    return selected


def run_end_to_end_showcase(
    total_images: int = 200,
    output_dir: Union[str, Path] = "experiments/demo_runs/sd15_200",
    raw_manifest: Union[str, Path] = "data/raw/manifest.jsonl",
    factory_config: Union[str, Path] = "configs/dataset_factory.yaml",
    detector_config: Union[str, Path] = "configs/config.yaml",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    pregenerated_manifest: Optional[Union[str, Path]] = None,
    checkpoints: Optional[Dict[str, str]] = None,
    threshold: float = 0.50,
    min_sharpness: Optional[float] = None,
    face_backend: str = "opencv_haar",
    extended_audit: bool = True,
    steps: Optional[int] = None,
    seed: int = 42,
    sample_offset: int = 0,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Execute end-to-end showcase: Generate or load N SD1.5 images -> Curate & Audit -> Score M0..M3 -> Save summary."""
    seed_everything(seed)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_manifest_p = Path(raw_manifest)
    if not raw_manifest_p.is_file():
        raise FileNotFoundError(f"Raw manifest not found: {raw_manifest_p}")

    cfg = load_config(Path(factory_config)) if Path(factory_config).is_file() else {}
    quotas = compute_mode_quotas(total_images)

    fake_manifest = out_root / "fake_manifest.jsonl"
    curated_manifest = out_root / "curated_manifest.jsonl"
    curation_report = out_root / "curation_report.jsonl"
    hard_manifest = out_root / "hard_samples_manifest.jsonl"
    scored_manifest = out_root / "scored_manifest.jsonl"
    summary_path = out_root / "e2e_summary.json"
    images_dir = out_root / "images"

    # Stage 1: Either sample N balanced pre-generated SD 1.5 images OR generate live with SD 1.5
    if pregenerated_manifest is not None:
        pre_p = Path(pregenerated_manifest)
        if not pre_p.is_file():
            raise FileNotFoundError(f"Pregenerated manifest not found: {pre_p}")
        source_mode = "pregenerated"
        sampled_records = _sample_pregenerated_manifest(pre_p, quotas=quotas, seed=seed)
        with open(fake_manifest, "w", encoding="utf-8") as f:
            for rec in sampled_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Stage 1: Sampled %d pre-generated SD 1.5 images from %s", len(sampled_records), pre_p)
    else:
        source_mode = "generated"
        logger.info("Stage 1: Generating %d SD 1.5 images (%s)...", total_images, quotas)
        run_image_generation(
            input_manifest=raw_manifest_p,
            output_manifest=fake_manifest,
            output_dir=images_dir,
            model_name=model_id,
            steps=steps if steps is not None else int(cfg.get("generation", {}).get("steps", 20)),
            guidance_scale=float(cfg.get("generation", {}).get("guidance_scale", 6.2)),
            img2img_strength=float(cfg.get("generation", {}).get("img2img_strength", 0.30)),
            mode_quotas=quotas,
            resume=True,
            seed=seed,
            config=cfg if cfg else None,
            sample_offset=sample_offset,
        )

    raw_fake_items = _read_jsonl(fake_manifest)

    # Calibrate sharpness threshold on real raw reference if min_sharpness not specified
    if min_sharpness is None:
        ref_q = float(cfg.get("curation", {}).get("reference_quantile_sharpness", 0.05))
        min_sharpness_val = calibrate_sharpness_from_manifest(
            raw_manifest_p, ref_quantile=ref_q, max_samples=40, fallback=2.5
        )
    else:
        min_sharpness_val = float(min_sharpness)

    # Stage 2: Curation Gate + Mode-Aware Extended Quality Audit
    logger.info("Stage 2: Running CurationGate (extended_audit=%s, min_sharpness=%.2f)...", extended_audit, min_sharpness_val)
    gate = ImageCurationGate(
        min_sharpness=min_sharpness_val,
        clip_enabled=False,
        face_backend=face_backend,
        dedup=True,
        extended_audit=extended_audit,
    )
    curation_summary = curate_image_manifest(
        input_manifest=fake_manifest,
        output_manifest=curated_manifest,
        gate=gate,
        curation_report=curation_report,
        raw_manifest=raw_manifest_p,
    )

    curated_items = _read_jsonl(curated_manifest)
    report_items = _read_jsonl(curation_report)
    diversity = _compute_diversity_metrics(curated_items or raw_fake_items)
    quality_stats = _aggregate_quality_stats(raw_fake_items, report_items)

    # Stage 3: Multi-Model Detector Scoring & M0 Hard-Sample Mining
    ckpts = checkpoints or DEFAULT_CHECKPOINTS
    logger.info("Stage 3: Scoring %d curated images across %d detector checkpoints...", len(curated_items), len(ckpts))
    scores_by_model = _score_manifest_with_checkpoints(
        curated_items, checkpoints=ckpts, config_path=detector_config, batch_size=batch_size
    )

    m0_key = next((k for k in ckpts if k.startswith("M0")), next(iter(ckpts)))
    m0_scores = scores_by_model.get(m0_key, [])

    hard_records: List[Dict[str, Any]] = []
    scored_records: List[Dict[str, Any]] = []
    for idx, item in enumerate(curated_items):
        rec = dict(item)
        model_scores = {m_lbl: float(scores_by_model[m_lbl][idx]) for m_lbl in ckpts}
        p0 = float(m0_scores[idx]) if idx < len(m0_scores) else 1.0
        is_hard = p0 < threshold
        rec["detector_scores"] = model_scores
        rec["is_hard_sample"] = is_hard
        rec["baseline_score"] = p0
        scored_records.append(rec)
        if is_hard:
            hard_records.append(rec)

    with open(scored_manifest, "w", encoding="utf-8") as f:
        for r in scored_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(hard_manifest, "w", encoding="utf-8") as f:
        for r in hard_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model_metrics: Dict[str, Dict[str, float]] = {}
    hard_indices = [i for i, r in enumerate(scored_records) if r["is_hard_sample"]]
    for m_lbl, probs in scores_by_model.items():
        if not probs:
            model_metrics[m_lbl] = {"recall": 0.0, "bypass_rate": 0.0, "mean_p_fake": 0.0, "hard_recall": 0.0}
            continue
        caught = sum(1 for p in probs if p >= threshold)
        hard_caught = sum(1 for i in hard_indices if probs[i] >= threshold)
        model_metrics[m_lbl] = {
            "recall": caught / len(probs),
            "bypass_rate": 1.0 - (caught / len(probs)),
            "mean_p_fake": float(np.mean(probs)),
            "hard_recall": (hard_caught / len(hard_indices)) if hard_indices else 1.0,
        }

    summary: Dict[str, Any] = {
        "source_mode": source_mode,
        "requested_images": int(total_images),
        "mode_quotas": quotas,
        "generated_images": len(raw_fake_items),
        "curated_images": curation_summary["kept"],
        "rejected_images": curation_summary["rejected"],
        "curation_pass_rate": (curation_summary["kept"] / len(raw_fake_items)) if raw_fake_items else 0.0,
        "min_sharpness_threshold": min_sharpness_val,
        "hard_samples_count": len(hard_records),
        "hard_sample_rate": (len(hard_records) / len(curated_items)) if curated_items else 0.0,
        "threshold_tau": threshold,
        "diversity": diversity,
        "quality_stats": quality_stats,
        "model_metrics": model_metrics,
        "artifacts": {
            "fake_manifest": str(fake_manifest),
            "curated_manifest": str(curated_manifest),
            "curation_report": str(curation_report),
            "hard_manifest": str(hard_manifest),
            "scored_manifest": str(scored_manifest),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("End-to-End Showcase complete -> summary saved at %s", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified N-image SD1.5 end-to-end pipeline showcase.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="experiments/demo_runs/sd15_200")
    parser.add_argument("--raw-manifest", type=str, default="data/raw/manifest.jsonl")
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--pregenerated-manifest", type=str, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_end_to_end_showcase(
        total_images=args.num_samples,
        output_dir=args.output_dir,
        raw_manifest=args.raw_manifest,
        model_id=args.model_id,
        pregenerated_manifest=args.pregenerated_manifest,
        steps=args.steps,
        threshold=args.threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
