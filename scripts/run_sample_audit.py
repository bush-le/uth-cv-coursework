"""Script to generate a 30-sample multi-modal deepfake audit batch (10 T2I, 10 I2I, 10 Inpaint),
curate them with the hardened Curation Gate, and perform exhaustive biometric scan."""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np
from PIL import Image
import torch

from src.common.logger import get_logger
from src.common.metrics import compute_laplacian_sharpness, verify_facial_geometry
from src.models.diffusion.prompt_engine import PromptGrammarEngine
from src.pipelines.curate_image_dataset import ImageCurationGate, curate_image_manifest
from src.pipelines.generate_deepfake_images import run_image_generation

logger = get_logger(__name__)


def run_audit(
    samples_per_mode: int = 10,
    device: str = "cuda",
    seed: int = 42,
    output_dir: Path = Path("data/synthetic/sample_audit_30"),
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest = output_dir / "raw_manifest.jsonl"
    curated_manifest = output_dir / "curated_manifest.jsonl"
    curation_report = output_dir / "curation_report.jsonl"
    detailed_json = output_dir / "detailed_audit.json"

    quotas = {
        "sd_img2img": samples_per_mode,
        "sd_inpaint": samples_per_mode,
        "sd_text2img": samples_per_mode,
    }

    logger.info(">>> Step 1: Synthesizing %d samples per mode (%d total) on %s...", samples_per_mode, samples_per_mode * 3, device)
    prompt_eng = PromptGrammarEngine(seed=seed)
    run_image_generation(
        input_manifest=Path("data/raw/manifest.jsonl"),
        output_manifest=raw_manifest,
        output_dir=output_dir / "images",
        total_samples=samples_per_mode * 3,
        mode_quotas=quotas,
        prompt_engine=prompt_eng,
        device=device,
        seed=seed,
        resume=False,
    )

    logger.info(">>> Step 2: Running Hardened Curation Gate (opencv_haar, min_sharpness=3.5)...")
    gate = ImageCurationGate(
        min_sharpness=3.5,
        face_backend="opencv_haar",
        clip_enabled=False,
        dedup=True,
    )
    curation_summary = curate_image_manifest(
        input_manifest=raw_manifest,
        output_manifest=curated_manifest,
        gate=gate,
        curation_report=curation_report,
    )
    logger.info("Curation Gate Result: %s", curation_summary)

    logger.info(">>> Step 3: Conducting Deep Biometric & Optical Scan on All %d Images...", samples_per_mode * 3)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # Read curation report entries
    curation_map = {}
    if curation_report.exists():
        with open(curation_report, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    curation_map[item["image_id"]] = item

    all_audits: List[Dict[str, Any]] = []
    with open(raw_manifest, "r", encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f if line.strip()]

    for rec in raw_records:
        img_id = rec["image_id"]
        file_path = rec["file_path"]
        mode = rec.get("manipulation_type", "unknown")
        prompt = rec.get("prompt", "")

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            continue
        h, w, c = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40))

        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))

        # Sharpness
        np_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t_img = torch.from_numpy(np_rgb).permute(2, 0, 1)
        sharpness = compute_laplacian_sharpness(t_img)

        # Facial geometry verification
        geom_valid, geom_reasons = verify_facial_geometry(img_bgr)

        # Biometric measurements
        face_ratio = 0.0
        face_center = (0.5, 0.5)
        aspect_ratio = 1.0
        if len(faces) >= 1:
            fx, fy, fw, fh = faces[0]
            face_ratio = float((fw * fh) / (h * w))
            face_center = (float((fx + fw / 2.0) / w), float((fy + fh / 2.0) / h))
            aspect_ratio = float(fh / max(1, fw))

        # Dynamic range & clipping
        white_clip = float(np.mean(img_bgr >= 252))
        black_clip = float(np.mean(img_bgr <= 4))

        gate_res = curation_map.get(img_id, {})
        is_curated_valid = gate_res.get("is_valid", geom_valid)
        rejection_reasons = gate_res.get("rejection_reasons", geom_reasons)

        audit_entry = {
            "image_id": img_id,
            "mode": mode,
            "file_path": file_path,
            "size": [w, h],
            "sharpness": round(sharpness, 2),
            "faces_detected": int(len(faces)),
            "face_area_ratio": round(face_ratio, 4),
            "face_center": (round(face_center[0], 3), round(face_center[1], 3)),
            "face_aspect_ratio": round(aspect_ratio, 3),
            "eyes_detected": int(len(eyes)),
            "white_clip_pct": round(white_clip * 100, 2),
            "black_clip_pct": round(black_clip * 100, 2),
            "curation_valid": is_curated_valid,
            "rejection_reasons": rejection_reasons,
            "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
        }
        all_audits.append(audit_entry)

    # Mode-level summaries
    mode_stats: Dict[str, Dict[str, Any]] = {}
    for m in ["sd_img2img", "sd_inpaint", "sd_text2img"]:
        m_entries = [e for e in all_audits if e["mode"] == m]
        if m_entries:
            mode_stats[m] = {
                "total": len(m_entries),
                "passed": sum(1 for e in m_entries if e["curation_valid"]),
                "rejected": sum(1 for e in m_entries if not e["curation_valid"]),
                "pass_rate_pct": round(sum(1 for e in m_entries if e["curation_valid"]) / len(m_entries) * 100, 1),
                "mean_sharpness": round(float(np.mean([e["sharpness"] for e in m_entries])), 2),
                "mean_face_area_pct": round(float(np.mean([e["face_area_ratio"] for e in m_entries])) * 100, 1),
                "mean_eyes": round(float(np.mean([e["eyes_detected"] for e in m_entries])), 1),
            }

    full_report = {
        "summary": curation_summary,
        "mode_stats": mode_stats,
        "audit_entries": all_audits,
    }

    with open(detailed_json, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Saved detailed audit to %s", detailed_json)
    return full_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate and scan a 30-sample multi-modal deepfake audit batch.")
    parser.add_argument("--samples-per-mode", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_audit(samples_per_mode=args.samples_per_mode, device=args.device, seed=args.seed)
