"""Phase 1: End-to-end synthetic facial dataset generation using Realistic Vision v5.1.

Generates 3 manipulation types:
1. inpaint: 50% mouth smile modification, 50% eyes sunglasses modification.
2. img2img: skin texture and subtle expression variations.
3. text2img: high-diversity demographic prompts.

Usage:
    python scripts/run_generation.py --num-samples 10
    python scripts/run_generation.py --num-samples 2000
"""

import argparse
import glob
import itertools
import json
import os
import random
import shutil
import gc
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw
import torch
# Failsafe patch for Kaggle transformers/diffusers compatibility
try:
    import transformers
    import transformers.utils
    if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
        transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    try:
        from transformers import CLIPImageProcessor  # noqa: F401
    except Exception:
        class DummyCLIPImageProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
        transformers.CLIPImageProcessor = DummyCLIPImageProcessor
except Exception:
    pass

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        DPMSolverMultistepScheduler,
    )
except ImportError:
    StableDiffusionPipeline = None  # type: ignore
    StableDiffusionImg2ImgPipeline = None  # type: ignore
    StableDiffusionInpaintPipeline = None  # type: ignore
    DPMSolverMultistepScheduler = None  # type: ignore

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"

NEGATIVE_PROMPT = (
    "deformed, bad eyes, bad anatomy, bad hands, poorly drawn face, mutation, "
    "mutated, ugly, disgusting, cartoon, 3d, anime, airbrushed, oversaturated, "
    "blurry, low quality, watermark, text, signature"
)


def generate_prompts(num_prompts: int, seed: int = 42) -> list[str]:
    """Generate structured demographic prompts with high diversity and balance."""
    random.seed(seed)
    ages = [20, 24, 28, 32, 36, 40, 45, 50, 55, 60]
    genders = ["man", "woman"]
    ethnicities = ["Asian", "Caucasian", "African", "Hispanic", "South Asian"]
    expressions = [
        "gentle natural smile",
        "neutral serious expression",
        "thoughtful calm gaze",
        "subtle warm smile",
        "confident relaxed expression",
    ]
    lightings = [
        "studio softbox lighting",
        "cinematic warm side lighting",
        "diffuse natural window lighting",
        "soft rim lighting with subtle shadows",
    ]

    combos = list(itertools.product(ages, genders, ethnicities, expressions, lightings))
    random.shuffle(combos)

    prompts = []
    for i in range(num_prompts):
        age, gender, ethnicity, expression, lighting = combos[i % len(combos)]
        p = (
            f"RAW photo, a close-up portrait of a {age}-year-old {ethnicity} {gender}, "
            f"{expression}, neutral grey studio backdrop, {lighting}, "
            f"detailed skin texture, visible pores, 8k uhd, dslr, high quality"
        )
        prompts.append(p)
    return prompts


def create_mask(size: tuple[int, int], region: str) -> Image.Image:
    """Create normalized elliptical mask for mouth or eyes in 512x512 crop."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if region == "mouth":
        box = [int(w * 0.30), int(h * 0.60), int(w * 0.70), int(h * 0.85)]
    else:  # eyes
        box = [int(w * 0.20), int(h * 0.25), int(w * 0.80), int(h * 0.52)]
    draw.ellipse(box, fill=255)
    return mask


def append_manifest(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_input_images(custom_path: Optional[str] = None) -> list[str]:
    """Auto-detect CelebA-HQ input images from custom path, Kaggle input, or local data/raw."""
    candidates: list[str] = []
    if custom_path and Path(custom_path).exists():
        p = Path(custom_path)
        if p.is_dir():
            candidates = sorted(glob.glob(str(p / "**/*.jpg"), recursive=True))
            if not candidates:
                candidates = sorted(glob.glob(str(p / "**/*.png"), recursive=True))

    if not candidates:
        search_paths = [
            "/kaggle/input/**/celeba_hq/*.jpg",
            "/kaggle/input/**/*.jpg",
            "/kaggle/input/**/*.png",
            "data/raw/celeba_hq/*.jpg",
            "data/raw/**/*.jpg",
            "data/raw/**/*.png",
        ]
        for sp in search_paths:
            found = sorted(glob.glob(sp, recursive=True))
            if found:
                candidates = found
                break

    return candidates


def run_pipeline(
    num_samples: int = 10,
    steps: int = 25,
    guidance_scale: float = 7.0,
    img2img_strength: float = 0.55,
    seed: int = 42,
    input_path: Optional[str] = None,
    output_dir: str = "/kaggle/working/data/synthetic",
    zip_output: bool = True,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    out_dir = Path(output_dir)
    dir_inpaint = out_dir / "inpaint"
    dir_img2img = out_dir / "img2img"
    dir_text2img = out_dir / "text2img"
    manifest_path = out_dir / "fake_manifest.jsonl"

    for d in [dir_inpaint, dir_img2img, dir_text2img]:
        d.mkdir(parents=True, exist_ok=True)

    candidate_files = find_input_images(input_path)
    if not candidate_files:
        raise FileNotFoundError(
            "Không tìm thấy ảnh thật nào! Hãy kiểm tra dataset celeba_hq trên Kaggle hoặc thư mục data/raw."
        )

    selected_files = candidate_files[:num_samples]
    mode_desc = f"TEST {num_samples} ẢNH" if num_samples <= 10 else f"FULL BATCH {num_samples} ẢNH"
    print("=" * 65)
    print(f"[*] Chế độ: {mode_desc}")
    print(f"[*] Tìm thấy {len(candidate_files)} ảnh gốc, chọn {len(selected_files)} ảnh.")
    print(f"[*] Device: {device} | Dtype: {dtype} | Steps: {steps}")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # TASK 1: INPAINTING
    # -------------------------------------------------------------------------
    print(f"\n[*] [1/3] BẮT ĐẦU TASK INPAINTING ({len(selected_files)} ảnh)...")
    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe_inpaint.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe_inpaint.scheduler.config, use_karras_sigmas=True
    )

    for idx, fpath in enumerate(selected_files):
        stem = Path(fpath).stem
        out_file = dir_inpaint / f"{stem}_inpaint.png"
        if out_file.exists():
            continue

        init_img = Image.open(fpath).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        region = "mouth" if (idx % 2 == 0) else "eyes"
        mask_img = create_mask(init_img.size, region)

        if region == "mouth":
            prompt = "a natural smiling mouth, realistic lips and skin texture, photorealistic, 8k uhd"
        else:
            prompt = "wearing stylish black sunglasses, natural skin reflections, photorealistic, 8k uhd"

        gen = torch.Generator(device=device).manual_seed(seed + idx)
        res = pipe_inpaint(
            prompt=prompt,
            image=init_img,
            mask_image=mask_img,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]

        res.save(out_file)
        append_manifest(
            manifest_path,
            {
                "image_path": str(out_file.relative_to(out_dir.parent)),
                "task": "inpaint",
                "sub_region": region,
                "source_real_image": Path(fpath).name,
                "prompt": prompt,
                "seed": seed + idx,
            },
        )
        if (idx + 1) % 5 == 0 or (idx + 1) == len(selected_files):
            print(f"  [Inpaint: {idx+1}/{len(selected_files)}] -> {out_file.name}")

    del pipe_inpaint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[✓] Task Inpaint hoàn tất & giải phóng VRAM.")

    # -------------------------------------------------------------------------
    # TASK 2: IMG2IMG
    # -------------------------------------------------------------------------
    print(f"\n[*] [2/3] BẮT ĐẦU TASK IMG2IMG ({len(selected_files)} ảnh)...")
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe_img2img.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
    )
    prompt_img2img = "RAW photo, a close-up portrait of an adult person, natural skin texture, studio lighting, photorealistic, 8k uhd, dslr"

    for idx, fpath in enumerate(selected_files):
        stem = Path(fpath).stem
        out_file = dir_img2img / f"{stem}_img2img.png"
        if out_file.exists():
            continue

        init_img = Image.open(fpath).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        gen = torch.Generator(device=device).manual_seed(seed + idx)
        res = pipe_img2img(
            prompt=prompt_img2img,
            image=init_img,
            strength=img2img_strength,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]

        res.save(out_file)
        append_manifest(
            manifest_path,
            {
                "image_path": str(out_file.relative_to(out_dir.parent)),
                "task": "img2img",
                "source_real_image": Path(fpath).name,
                "prompt": prompt_img2img,
                "seed": seed + idx,
                "strength": img2img_strength,
            },
        )
        if (idx + 1) % 5 == 0 or (idx + 1) == len(selected_files):
            print(f"  [Img2Img: {idx+1}/{len(selected_files)}] -> {out_file.name}")

    del pipe_img2img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[✓] Task Img2Img hoàn tất & giải phóng VRAM.")

    # -------------------------------------------------------------------------
    # TASK 3: TEXT2IMG
    # -------------------------------------------------------------------------
    print(f"\n[*] [3/3] BẮT ĐẦU TASK TEXT2IMG ({len(selected_files)} ảnh)...")
    pipe_text2img = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe_text2img.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe_text2img.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
    )

    t2i_prompts = generate_prompts(len(selected_files), seed=seed)

    for idx, (fpath, prompt) in enumerate(zip(selected_files, t2i_prompts)):
        stem = Path(fpath).stem
        out_file = dir_text2img / f"{stem}_text2img.png"
        if out_file.exists():
            continue

        gen = torch.Generator(device=device).manual_seed(seed + idx)
        res = pipe_text2img(
            prompt=prompt,
            height=512,
            width=512,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
        ).images[0]

        res.save(out_file)
        append_manifest(
            manifest_path,
            {
                "image_path": str(out_file.relative_to(out_dir.parent)),
                "task": "text2img",
                "source_real_image": None,
                "prompt": prompt,
                "seed": seed + idx,
            },
        )
        if (idx + 1) % 5 == 0 or (idx + 1) == len(selected_files):
            print(f"  [Text2Img: {idx+1}/{len(selected_files)}] -> {out_file.name}")

    del pipe_text2img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("[✓] Task Text2Img hoàn tất & giải phóng VRAM.")

    # -------------------------------------------------------------------------
    # NÉN ZIP & BÁO CÁO KẾT QUẢ
    # -------------------------------------------------------------------------
    total_inpaint = len(list(dir_inpaint.glob("*.png")))
    total_img2img = len(list(dir_img2img.glob("*.png")))
    total_text2img = len(list(dir_text2img.glob("*.png")))
    total_all = total_inpaint + total_img2img + total_text2img

    if zip_output:
        archive_name = str(out_dir.parent / "synthetic_face_dataset")
        print(f"\n[*] Đang đóng gói zip vào {archive_name}.zip...")
        shutil.make_archive(archive_name, "zip", root_dir=out_dir)
        zip_file = f"{archive_name}.zip"
        size_mb = os.path.getsize(zip_file) / (1024 * 1024)
        print(f"[✓] Đã tạo thành công file nén: {zip_file} ({size_mb:.2f} MB)")

    print("\n" + "=" * 65)
    print(f"[✓] TỔNG KẾT HOÀN TẤT:")
    print(f"  • Inpaint : {total_inpaint} ảnh")
    print(f"  • Img2Img : {total_img2img} ảnh")
    print(f"  • Text2Img: {total_text2img} ảnh")
    print(f"  • TỔNG CỘNG: {total_all} ảnh giả")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic deepfake face dataset.")
    parser.add_argument("--num-samples", type=int, default=10, help="Số lượng ảnh gốc sử dụng (10 hoặc 2000)")
    parser.add_argument("--steps", type=int, default=25, help="Số bước khuếch tán (mặc định 25)")
    parser.add_argument("--input-dir", type=str, default=None, help="Đường dẫn thư mục ảnh đầu vào")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/data/synthetic", help="Thư mục xuất ảnh")
    parser.add_argument("--no-zip", action="store_true", help="Không tạo file zip đầu ra")
    args = parser.parse_args()

    run_pipeline(
        num_samples=args.num_samples,
        steps=args.steps,
        input_path=args.input_dir,
        output_dir=args.output_dir,
        zip_output=not args.no_zip,
    )
