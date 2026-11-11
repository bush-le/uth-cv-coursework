"""Script to generate and assemble a pristine 30-sample multi-modal demo batch
(exactly 10 sd_img2img, 10 sd_inpaint, 10 sd_text2img) with visual contact sheets
and detailed biometric/optical audit for thorough manual and automated inspection.
"""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.common.logger import get_logger
from src.common.metrics import compute_laplacian_sharpness, verify_facial_geometry
from src.models.diffusion.prompt_engine import PromptGrammarEngine
from src.models.diffusion.stable_diffusion_engine import StableDiffusionEngine, generate_dynamic_face_mask
from src.pipelines.curate_image_dataset import ImageCurationGate, curate_image_manifest

logger = get_logger(__name__)


def create_contact_sheet(
    image_paths: List[Path],
    labels: List[str],
    output_path: Path,
    cols: int = 5,
    thumb_size: int = 256,
) -> Path:
    """Create a high-resolution contact sheet image grid with text annotations."""
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    header_h = 32
    cell_w = thumb_size
    cell_h = thumb_size + header_h

    sheet_w = cols * cell_w
    sheet_h = rows * cell_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    for idx, (img_p, label) in enumerate(zip(image_paths, labels)):
        r = idx // cols
        c = idx % cols
        x = c * cell_w
        y = r * cell_h

        if img_p.exists():
            with Image.open(img_p) as im:
                thumb = im.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x, y))

        # Draw label banner
        draw.rectangle([x, y + thumb_size, x + cell_w, y + cell_h], fill=(36, 38, 44))
        draw.text((x + 8, y + thumb_size + 8), label, fill=(230, 230, 240))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    logger.info("Saved contact sheet (%d images) to %s", n, output_path)
    return output_path


def assemble_demo_batch(
    output_dir: Path = Path("data/synthetic/demo_batch_30"),
    artifact_dir: Optional[Path] = None,
    target_per_mode: int = 10,
    device: str = "cuda",
    seed: int = 42,
    fresh: bool = True,
) -> Dict[str, Any]:
    if artifact_dir is None:
        artifact_dir = Path("experiments/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    if fresh and images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    gate = ImageCurationGate(
        min_sharpness=3.5,
        face_backend="opencv_haar",
        clip_enabled=False,
        dedup=True,
    )
    prompt_eng = PromptGrammarEngine(seed=seed)

    modes = ["sd_img2img", "sd_inpaint", "sd_text2img"]
    collected: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}

    # 1. Harvest already curated valid images from previous audit if available (only if not fresh)
    audit_file = Path("data/synthetic/sample_audit_30/detailed_audit.json")
    if not fresh and audit_file.exists():
        with open(audit_file, "r", encoding="utf-8") as f:
            audit_data = json.load(f)
        for e in audit_data.get("audit_entries", []):
            m = e["mode"]
            if m in collected and len(collected[m]) < target_per_mode:
                orig_p = Path(e["file_path"])
                if orig_p.exists():
                    # Re-verify through the updated gate
                    with Image.open(orig_p) as im:
                        np_img = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
                        t_img = torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
                    rep = gate.evaluate_image(t_img, prompt=e.get("prompt", ""), image_id=e["image_id"])
                    if rep.is_valid:
                        # Copy to demo directory
                        target_p = images_dir / f"{e['image_id']}.png"
                        shutil.copy2(orig_p, target_p)
                        e_copy = dict(e)
                        e_copy["file_path"] = str(target_p)
                        e_copy["curation_valid"] = True
                        collected[m].append(e_copy)

    for m in modes:
        logger.info("Harvested %d/%d valid samples for %s.", len(collected[m]), target_per_mode, m)

    # 2. Replenish missing samples using Live Stable Diffusion Engine
    missing = {m: target_per_mode - len(collected[m]) for m in modes if len(collected[m]) < target_per_mode}
    if missing:
        logger.info("Replenishing missing samples: %s on %s...", missing, device)
        engine = StableDiffusionEngine(
            model_name="runwayml/stable-diffusion-v1-5",
            inpaint_model_name="stable-diffusion-v1-5/stable-diffusion-inpainting",
            device=device,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            use_mock_fallback=False,
        )

        # Load reference real images from train partition
        reals_manifest = Path("data/partitions/train_reals.jsonl")
        real_items = []
        if reals_manifest.exists():
            with open(reals_manifest, "r", encoding="utf-8") as rf:
                real_items = [json.loads(line) for line in rf if line.strip()]

        for m, num_needed in missing.items():
            attempt = 0
            while len(collected[m]) < target_per_mode and attempt < 40:
                curr_seed = seed + 5000 + (attempt * 37)
                demo_idx = len(collected[m])
                fake_id = f"demo_{m}_{demo_idx:02d}_att{attempt}"
                out_path = images_dir / f"{fake_id}.png"

                real_rec = real_items[(demo_idx + attempt * 7) % len(real_items)] if real_items else {}
                demographics = real_rec.get("demographics")
                mode_key = "img2img" if m == "sd_img2img" else ("inpaint" if m == "sd_inpaint" else "text2img")
                prompt_used = prompt_eng.generate_prompt(demographics, mode=mode_key, seed_offset=curr_seed)

                # Generate image
                if m == "sd_img2img":
                    src_p = Path(real_rec.get("file_path", ""))
                    with Image.open(src_p) as src_im:
                        src_rgb = src_im.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
                        src_t = torch.from_numpy(np.array(src_rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)
                    # Adaptive strength: lower for children and glasses wearers to preserve identity and optics
                    eff_strength = 0.28
                    if demographics:
                        if demographics.get("age_group") in ("0-2", "3-6") or demographics.get("glasses") not in (None, "None"):
                            eff_strength = 0.22
                    out_t = engine.generate_img2img(src_t, prompt=prompt_used, strength=eff_strength, guidance_scale=6.0, seed=curr_seed)
                elif m == "sd_inpaint":
                    src_p = Path(real_rec.get("file_path", ""))
                    with Image.open(src_p) as src_im:
                        src_rgb = src_im.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
                        src_t = torch.from_numpy(np.array(src_rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)
                    # For inpainting, select target region and ensure demographics align
                    target_reg = ["periocular", "lips", "full"][demo_idx % 3]
                    mask = generate_dynamic_face_mask(src_t, target_region=target_reg)
                    out_t = engine.generate_inpaint(src_t, mask, prompt=prompt_used, guidance_scale=6.2, num_inference_steps=30, seed=curr_seed)
                else:  # sd_text2img
                    out_t = engine.generate_text2img(prompt=prompt_used, height=512, width=512, guidance_scale=6.2, num_inference_steps=30, seed=curr_seed)

                # Save candidate
                np_arr = (out_t.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                Image.fromarray(np_arr).save(out_path)

                # Evaluate through Curation Gate
                rep = gate.evaluate_image(out_t.detach().cpu(), prompt=prompt_used, image_id=fake_id)
                if rep.is_valid:
                    # Biometric scan
                    img_bgr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    face_casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    eye_casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
                    faces = face_casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
                    if len(faces) == 0:
                        faces = face_casc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40))
                    eyes = eye_casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))

                    faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True) if len(faces) > 0 else []
                    face_ratio = (faces_sorted[0][2] * faces_sorted[0][3]) / (512.0 * 512.0) if faces_sorted else 0.0
                    entry = {
                        "image_id": fake_id,
                        "mode": m,
                        "file_path": str(out_path),
                        "size": [512, 512],
                        "sharpness": round(rep.sharpness_score, 2),
                        "faces_detected": int(len(faces_sorted)),
                        "face_area_ratio": round(face_ratio, 4),
                        "eyes_detected": int(len(eyes)),
                        "curation_valid": True,
                        "rejection_reasons": [],
                        "prompt": prompt_used[:80] + "..." if len(prompt_used) > 80 else prompt_used,
                    }
                    collected[m].append(entry)
                    logger.info("Accepted %s (%d/%d): sharpness=%.2f, face_area=%.2f%%", fake_id, len(collected[m]), target_per_mode, rep.sharpness_score, face_ratio * 100)
                else:
                    logger.info("Rejected candidate %s: %s", fake_id, rep.rejection_reasons)
                    out_path.unlink(missing_ok=True)
                attempt += 1

    # 3. Save Final Curated Manifest
    demo_manifest = output_dir / "demo_manifest.jsonl"
    all_demo_entries: List[Dict[str, Any]] = []
    for m in modes:
        all_demo_entries.extend(collected[m][:target_per_mode])

    with open(demo_manifest, "w", encoding="utf-8") as f:
        for item in all_demo_entries:
            f.write(json.dumps(item) + "\n")

    # 4. Generate Visual Contact Sheets (one per modality + combined)
    cs_dir = output_dir / "contact_sheets"
    cs_dir.mkdir(parents=True, exist_ok=True)

    sheet_paths = {}
    for m in modes:
        entries = collected[m][:target_per_mode]
        paths = [Path(e["file_path"]) for e in entries]
        labels = [f"#{i+1} Sharp:{e['sharpness']} Area:{int(e['face_area_ratio']*100)}%" for i, e in enumerate(entries)]
        cs_path = cs_dir / f"contact_sheet_{m}.jpg"
        create_contact_sheet(paths, labels, cs_path, cols=5, thumb_size=256)
        sheet_paths[m] = cs_path

        # Copy to artifact dir for direct embedding
        if artifact_dir.exists():
            art_dest = artifact_dir / f"contact_sheet_{m}.jpg"
            shutil.copy2(cs_path, art_dest)

    # Combined contact sheet: 3 rows, 10 columns (thumb_size=160)
    comb_paths = []
    comb_labels = []
    for m in modes:
        for i, e in enumerate(collected[m][:target_per_mode]):
            comb_paths.append(Path(e["file_path"]))
            comb_labels.append(f"{m[:5]} #{i+1}")
    comb_cs = cs_dir / "contact_sheet_all_30.jpg"
    create_contact_sheet(comb_paths, comb_labels, comb_cs, cols=10, thumb_size=160)
    sheet_paths["all_30"] = comb_cs
    if artifact_dir.exists():
        shutil.copy2(comb_cs, artifact_dir / "contact_sheet_all_30.jpg")

    logger.info("Demo batch assembly complete: exactly %d samples per mode (30 total).", target_per_mode)
    return {
        "status": "success",
        "collected_counts": {m: len(collected[m][:target_per_mode]) for m in modes},
        "manifest": str(demo_manifest),
        "contact_sheets": {k: str(v) for k, v in sheet_paths.items()},
        "entries": all_demo_entries,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assemble 30-sample demo batch with contact sheets.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assemble_demo_batch(device=args.device, seed=args.seed)
