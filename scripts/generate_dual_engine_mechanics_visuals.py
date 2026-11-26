"""Generate visual technical figures for Section 03b (Dual-Engine Generation Mechanics).

Renders three publication-grade PNG figures directly from real codebase functions
and verified disk datasets/manifests:
1. dual_engine_mask_anatomy.png: SD 1.5 Soft-Mask (OpenCV Haar + Gaussian blur_radius=18)
   vs RV5.1 Binary Ellipse masks across target regions + alpha transition profile.
2. dual_engine_visual_3modes.png: Visual comparison across Inpainting, Img2Img, and
   Text2Img between Engine A (SD v1.5 Base) and Engine B (RealisticVision v5.1)
   with 2x skin/boundary micro-texture zoom insets.
3. dual_engine_prompt_quota_evolution.png: Mode distribution computed live from manifests
   (D0 -> AL R1 -> AL R2 vs RV5.1 6k) paired with Prompt Grammar vs 5-D Grid architecture.
"""

from collections import Counter
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from scripts.run_generation import create_mask
from src.common.logger import get_logger
from src.models.diffusion.prompt_engine import SLOT_TAXONOMY
from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask

logger = get_logger(__name__)

OUT_DIR = Path("docs/reports/assets/extracted_nb")


def _load_rgb_tensor(image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    """Load RGB image resized to 512x512 as uint8 HWC array and CHW float tensor."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        arr = np.asarray(rgb, dtype=np.uint8)
    tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
    return arr, tensor


def render_mask_anatomy(out_path: Path) -> None:
    """Compare SD 1.5 feathered dynamic masks against RV5.1 binary ellipse masks."""
    sample_face = Path("docs/reports/assets/samples/real_0.jpg")
    face_np, face_t = _load_rgb_tensor(sample_face)

    mask_specs = [
        ("SD 1.5 Soft-Mask\nVùng Mắt (periocular)", generate_dynamic_face_mask(face_t, bbox=None, blur_radius=18, target_region="periocular").squeeze(0).numpy(), "#1d4ed8"),
        ("SD 1.5 Soft-Mask\nVùng Miệng (lips)", generate_dynamic_face_mask(face_t, bbox=None, blur_radius=18, target_region="lips").squeeze(0).numpy(), "#1d4ed8"),
        ("SD 1.5 Soft-Mask\nToàn Mặt + Cằm (full, +18%)", generate_dynamic_face_mask(face_t, bbox=None, blur_radius=18, target_region="full").squeeze(0).numpy(), "#1d4ed8"),
        ("RV5.1 Binary Mask\nVùng Miệng (mouth 50%)", np.asarray(create_mask((512, 512), "mouth"), dtype=np.float32) / 255.0, "#b45309"),
        ("RV5.1 Binary Mask\nVùng Mắt / Kính (eyes 50%)", np.asarray(create_mask((512, 512), "eyes"), dtype=np.float32) / 255.0, "#b45309"),
    ]

    fig = plt.figure(figsize=(17.0, 8.8), dpi=180)
    gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.0, 0.55], hspace=0.32, wspace=0.14)

    for idx, (title, m_np, accent) in enumerate(mask_specs):
        ax_top = fig.add_subplot(gs[0, idx])
        ax_top.imshow(m_np, cmap="inferno", vmin=0.0, vmax=1.0)
        ax_top.set_title(title, fontsize=10.5, fontweight="bold", color=accent, pad=8)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for spine in ax_top.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(1.8)

        ax_mid = fig.add_subplot(gs[1, idx])
        tint = np.zeros_like(face_np, dtype=np.float32)
        tint[..., 0] = 30.0 if accent == "#1d4ed8" else 245.0
        tint[..., 1] = 144.0 if accent == "#1d4ed8" else 158.0
        tint[..., 2] = 255.0 if accent == "#1d4ed8" else 11.0
        alpha_vis = (m_np[..., None] * 0.52)
        overlay = np.clip(face_np.astype(np.float32) * (1.0 - alpha_vis) + tint * alpha_vis, 0, 255).astype(np.uint8)
        ax_mid.imshow(overlay)
        coverage_pct = float((m_np > 0.05).mean() * 100.0)
        soft_band_pct = float(((m_np > 0.02) & (m_np < 0.98)).mean() * 100.0)
        ax_mid.set_xlabel(
            f"Phủ ROI: {coverage_pct:.1f}% | Biên mờ: {soft_band_pct:.1f}%",
            fontsize=9.0,
            fontweight="bold",
            color="#334155",
            labelpad=5,
        )
        ax_mid.set_xticks([])
        ax_mid.set_yticks([])

    ax_prof = fig.add_subplot(gs[2, :])
    sd_lip = mask_specs[1][1]
    rv_mouth = mask_specs[3][1]
    row_sd = int(np.argmax(sd_lip.sum(axis=1)))
    row_rv = int(np.argmax(rv_mouth.sum(axis=1)))
    xs = np.arange(512)
    ax_prof.plot(xs, sd_lip[row_sd], color="#1d4ed8", linewidth=2.6, label="SD 1.5 Soft-Mask (Gaussian blur_radius=18 -> biên chuyển tiếp mượt [0 → 1])")
    ax_prof.plot(xs, rv_mouth[row_rv], color="#d97706", linewidth=2.2, linestyle="--", label="RV5.1 Binary Ellipse (create_mask -> nhảy bậc 0 → 1 trực tiếp ở viền ellipse)")
    ax_prof.fill_between(xs, sd_lip[row_sd], color="#3b82f6", alpha=0.14)
    ax_prof.set_xlim(90, 420)
    ax_prof.set_ylim(-0.06, 1.14)
    ax_prof.set_xlabel("Tọa độ ngang X qua tâm vùng miệng (pixel, 512×512)", fontsize=10, fontweight="bold")
    ax_prof.set_ylabel("Alpha Mask m(x)", fontsize=10, fontweight="bold")
    ax_prof.set_title("Đồ Thị Mặt Cắt Biên Alpha: Độ Dốc Chuyển Tiếp Của Soft-Mask So Với Viền Nhị Phân Sắc Cạnh", fontsize=11, fontweight="bold", pad=6)
    ax_prof.grid(True, linestyle=":", alpha=0.5)
    ax_prof.spines["top"].set_visible(False)
    ax_prof.spines["right"].set_visible(False)
    ax_prof.legend(loc="upper right", fontsize=9.5, frameon=True)

    fig.suptitle(
        "SO SÁNH HÌNH HỌC MẶT NẠ INPAINTING: SD 1.5 SOFT-MASK (OPENCV HAAR + BLUR 18px) VS RV5.1 BINARY ELLIPSE",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
        color="#0f172a",
    )
    plt.subplots_adjust(top=0.89, bottom=0.08, left=0.05, right=0.98)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def render_visual_3modes_comparison(out_path: Path) -> None:
    """Compare SD 1.5 Base vs RV5.1 outputs across all 3 generation modes with 2x skin zoom."""
    sd_samples = [
        ("Inpainting (Soft-Mask Biên Mờ)", Path("data/synthetic/d0_train_images/inpainting/fake_ffhq_00002_0.png"), (160, 290, 350, 440)),
        ("Img2Img (Biến Đổi Strength=0.30)", Path("data/synthetic/d0_train_images/img2img/fake_ffhq_00000_0.png"), (150, 210, 340, 360)),
        ("Text2Img (Sinh Từ Nhiễu Ngẫu Nhiên)", Path("data/synthetic/d0_train_images/text2image/fake_ffhq_00000_16.png"), (160, 200, 350, 350)),
    ]
    rv_samples = [
        ("Inpainting (INPAINT_MODEL + Binary Mask)", Path("data/external_benchmarks/synthetic_face_dataset_6k/inpaint/0_inpaint.png"), (170, 290, 350, 440)),
        ("Img2Img (RV5.1 Strength=0.55, RAW Photo)", Path("data/external_benchmarks/synthetic_face_dataset_6k/img2img/0_img2img.png"), (150, 210, 340, 360)),
        ("Text2Img (RV5.1 5-D Grid, Lỗ Chân Lông Thật)", Path("data/external_benchmarks/synthetic_face_dataset_6k/text2img/0_text2img.png"), (150, 200, 340, 350)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.2), dpi=180)
    plt.subplots_adjust(top=0.87, bottom=0.05, left=0.06, right=0.98, hspace=0.28, wspace=0.14)

    rows = [
        ("BỘ A: SD 1.5 BASE\nSteps=20 | CFG=6.2\nImg2Img Str=0.30", sd_samples, "#1d4ed8"),
        ("BỘ B: REALISTIC VISION 5.1\nSteps=25 | CFG=7.0\nImg2Img Str=0.55", rv_samples, "#059669"),
    ]

    for r_idx, (row_label, items, color) in enumerate(rows):
        for c_idx, (col_title, img_path, (x1, y1, x2, y2)) in enumerate(items):
            ax = axes[r_idx, c_idx]
            arr, _ = _load_rgb_tensor(img_path)
            ax.imshow(arr)
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.0, edgecolor=color, facecolor="none", linestyle="--")
            ax.add_patch(rect)

            # Bottom-right 2x zoom inset for micro-texture & boundary inspection
            ax_ins = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
            ax_ins.imshow(arr[y1:y2, x1:x2])
            ax_ins.set_xticks([])
            ax_ins.set_yticks([])
            for sp in ax_ins.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(2.2)
            ax_ins.set_title("Zoom Vi Kết Cấu", fontsize=7.5, fontweight="bold", color="#ffffff", backgroundcolor=color, pad=2)

            ax.set_title(f"{col_title}\n({img_path.name})", fontsize=9.8, fontweight="bold", color="#0f172a", pad=7)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(1.5)

        axes[r_idx, 0].set_ylabel(row_label, fontsize=9.5, fontweight="bold", color=color, labelpad=10)

    fig.suptitle(
        "ĐỐI CHIẾU TRỰC QUAN 3 CHẾ ĐỘ SINH ẢNH: SD 1.5 BASE VS REALISTIC VISION v5.1 (KÈM ZOOM VI KẾT CẤU DA & BIÊN)",
        fontsize=13.0,
        fontweight="bold",
        y=0.96,
        color="#0f172a",
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def render_prompt_and_quota_evolution(out_path: Path) -> None:
    """Plot actual manifest mode counts alongside PromptGrammarEngine 7-slot vs RV5.1 5-D structure."""
    def _counts(path: Path, key: str) -> Counter:
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        c = Counter(r.get(key) for r in recs)
        return Counter({
            "Img2Img": c.get("sd_img2img", c.get("img2img", 0)),
            "Inpaint": c.get("sd_inpaint", c.get("inpaint", 0)),
            "Text2Img": c.get("sd_text2img", c.get("text2img", 0)),
        })

    d0_counts = _counts(Path("data/synthetic/curated_manifest.jsonl"), "manipulation_type")
    r1_counts = _counts(Path("data/active_learning/round_1/candidates_raw.jsonl"), "manipulation_type")
    r2_counts = _counts(Path("data/active_learning/round_2/candidates_raw.jsonl"), "manipulation_type")
    rv_counts = _counts(Path("data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl"), "task")
    stages = [
        (f"SD 1.5 D₀\n({sum(d0_counts.values()):,} sạch)".replace(",", "."), d0_counts),
        (f"SD 1.5 AL R1\n({sum(r1_counts.values()):,} thô)".replace(",", "."), r1_counts),
        (f"SD 1.5 AL R2\n({sum(r2_counts.values()):,} thô)".replace(",", "."), r2_counts),
        (f"RV5.1 6k\n({sum(rv_counts.values()):,} mẫu)".replace(",", "."), rv_counts),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(17.4, 6.4), dpi=180, gridspec_kw={"width_ratios": [1.0, 1.05]})
    plt.subplots_adjust(top=0.82, bottom=0.14, left=0.05, right=0.98, wspace=0.56)

    labels = [s[0] for s in stages]
    x = np.arange(len(stages))
    modes = [("Img2Img", "#7e22ce"), ("Inpaint", "#1d4ed8"), ("Text2Img", "#c2410c")]
    bottom = np.zeros(len(stages))

    for mode_name, color in modes:
        pcts = []
        raw_vals = []
        for _, cnt in stages:
            total = max(1, sum(cnt.values()))
            raw_vals.append(cnt[mode_name])
            pcts.append(cnt[mode_name] * 100.0 / total)
        bars = ax_l.bar(x, pcts, bottom=bottom, width=0.56, label=mode_name, color=color, edgecolor="#ffffff", linewidth=1.2)
        for i, bar in enumerate(bars):
            h = bar.get_height()
            if h >= 10.0:
                ax_l.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bottom[i] + h / 2.0,
                    f"{h:.1f}%\n({raw_vals[i]:,})".replace(",", "."),
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    fontweight="bold",
                    color="#ffffff",
                )
        bottom += np.array(pcts)

    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=9.8, fontweight="bold")
    ax_l.set_ylim(0, 108)
    ax_l.set_ylabel("Tỷ Trọng Chế Độ Sinh (%)", fontsize=10.5, fontweight="bold")
    r2_total = max(1, sum(r2_counts.values()))
    r2_inp_pct = r2_counts["Inpaint"] * 100.0 / r2_total
    ax_l.set_title(
        f"Phân Bổ Thực Tế 3 Chế Độ Theo Từng Giai Đoạn\n(R2 Tự Dồn {r2_inp_pct:.1f}% Vào SD Inpaint Theo Error-Steering)",
        fontsize=11.0,
        fontweight="bold",
        pad=10,
    )
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)
    ax_l.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True, fontsize=9.5)

    # Right panel: Slot cardinality breakdown for PromptGrammarEngine vs RV5.1 T2I 5-D grid
    slot_names = [
        "SD 1.5 • Tuổi (subject_age)",
        "SD 1.5 • Giới tính (gender)",
        "SD 1.5 • Ống kính (optics)",
        "SD 1.5 • Ánh sáng (lighting)",
        "SD 1.5 • Biểu cảm (expression)",
        "SD 1.5 • Che khuất (occlusion)",
        "SD 1.5 • Da/lỗ chân lông",
        "RV5.1 T2I • Độ tuổi (20..60)",
        "RV5.1 T2I • Sắc tộc (ethnicities)",
        "RV5.1 T2I • Biểu cảm × Ánh sáng",
    ]
    counts = [
        len(SLOT_TAXONOMY["subject_age"]),
        len(SLOT_TAXONOMY["subject_gender"]),
        len(SLOT_TAXONOMY["optics"]),
        len(SLOT_TAXONOMY["lighting"]),
        len(SLOT_TAXONOMY["expression"]),
        len(SLOT_TAXONOMY["occlusion"]),
        len(SLOT_TAXONOMY["imperfections"]),
        10,
        5,
        20,
    ]
    bar_colors = ["#1d4ed8"] * 7 + ["#059669"] * 3
    y_pos = np.arange(len(slot_names))[::-1]
    bars_r = ax_r.barh(y_pos, counts, color=bar_colors, height=0.62, edgecolor="#ffffff", linewidth=0.8)
    for bar, val in zip(bars_r, counts):
        ax_r.text(val + 0.35, bar.get_y() + bar.get_height() / 2.0, f"{val} giá trị", va="center", ha="left", fontsize=9.0, fontweight="bold", color="#1e293b")
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(slot_names, fontsize=9.2, fontweight="bold")
    ax_r.set_xlim(0, 24.5)
    ax_r.set_xlabel("Số Biến Thể Trong Mỗi Chiều Tổ Hợp", fontsize=10.5, fontweight="bold")
    ax_r.set_title("Cấu Trúc Tổ Hợp Ngữ Pháp Prompt Thực Tế\nSD 1.5 (7 Slot, Softmax floor=0.15) vs RV5.1 T2I (Lưới 2.000 Tổ Hợp)", fontsize=11.0, fontweight="bold", pad=10)
    ax_r.grid(axis="x", linestyle=":", alpha=0.5)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

    fig.suptitle(
        "CƠ CHẾ ĐIỀU PHỐI QUOTA ACTIVE LEARNING & NGỮ PHÁP SINH PROMPT ĐA CHIỀU (SD 1.5 VS RV5.1)",
        fontsize=13.0,
        fontweight="bold",
        y=0.96,
        color="#0f172a",
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def render_resolution_degradation_comparison(out_path: Path) -> None:
    """Visualize high-frequency and DINOv2 patch density loss when downscaling 504px -> 224px."""
    import csv

    sample_187 = Path("data/external_benchmarks/synthetic_face_dataset_6k/img2img/187_img2img.png")
    sample_path = sample_187 if sample_187.exists() else Path("data/external_benchmarks/synthetic_face_dataset_6k/img2img/0_img2img.png")
    with Image.open(sample_path) as img:
        rgb_512 = img.convert("RGB")
        rgb_504 = rgb_512.resize((504, 504), Image.Resampling.LANCZOS)
        rgb_224 = rgb_512.resize((224, 224), Image.Resampling.BILINEAR)
        rgb_224_vis = rgb_224.resize((504, 504), Image.Resampling.BILINEAR)

    arr_504 = np.asarray(rgb_504, dtype=np.uint8)
    arr_224 = np.asarray(rgb_224, dtype=np.uint8)
    arr_224_vis = np.asarray(rgb_224_vis, dtype=np.uint8)

    # Measure grayscale 2D FFT high-frequency ratio & Laplacian variance (real measurement)
    def _fft_and_lap(arr_u8: np.ndarray) -> tuple[np.ndarray, float]:
        gray = arr_u8.mean(axis=2) / 255.0
        f = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log1p(np.abs(f))
        lap = (
            -4.0 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )
        return mag, float(np.var(lap) * 1000.0)

    mag_504, lap_504 = _fft_and_lap(arr_504)
    mag_224, lap_224 = _fft_and_lap(arr_224)

    legacy_512 = json.loads(Path("experiments/logs/legacy_6k_benchmark_results.json").read_text())
    legacy_224 = json.loads(Path("experiments/logs/legacy_6k_benchmark_results_224px.json").read_text())
    with open("experiments/benchmarks/tables/benchmark_5models_global_summary.csv", encoding="utf-8") as f:
        univ_504 = {r["Model"]: r for r in csv.DictReader(f)}
    with open("experiments/benchmarks/tables/benchmark_5models_global_summary_224px.csv", encoding="utf-8") as f:
        univ_224 = {r["Model"]: r for r in csv.DictReader(f)}

    fig = plt.figure(figsize=(16.8, 7.6), dpi=180)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 0.85, 1.25], hspace=0.34, wspace=0.22)

    # Col 0: 504x504 Cheek/Eye Zoom + Patch Grid 14x14
    ax_504 = fig.add_subplot(gs[0, 0])
    crop_504 = arr_504[185:335, 175:325]
    ax_504.imshow(crop_504)
    for g in range(0, 150, 14):
        ax_504.axhline(g, color="#3b82f6", alpha=0.32, linewidth=0.7)
        ax_504.axvline(g, color="#3b82f6", alpha=0.32, linewidth=0.7)
    ax_504.set_title("Chuẩn DINOv2 (504×504 = 1.296 Tokens)\nLỗ chân lông & bờ mi rõ nét", fontsize=9.8, fontweight="bold", color="#1d4ed8")
    ax_504.set_xticks([])
    ax_504.set_yticks([])

    # Col 1: 224x224 Cheek/Eye Zoom + Larger Relative 14x14 Patches (upscaled 504/224 = 2.25x)
    ax_224 = fig.add_subplot(gs[0, 1])
    crop_224 = arr_224_vis[185:335, 175:325]
    ax_224.imshow(crop_224)
    for g in np.arange(0, 150, 14 * (504.0 / 224.0)):
        ax_224.axhline(g, color="#dc2626", alpha=0.45, linewidth=0.9)
        ax_224.axvline(g, color="#dc2626", alpha=0.45, linewidth=0.9)
    ax_224.set_title("Hạ Cấp 224×224 (256 Tokens • Mất 80.2%)\nChi tiết da bị làm nhòe khi nén", fontsize=9.8, fontweight="bold", color="#dc2626")
    ax_224.set_xticks([])
    ax_224.set_yticks([])

    # Col 2: Spectral FFT Comparison (504 vs 224)
    ax_fft1 = fig.add_subplot(gs[1, 0])
    ax_fft1.imshow(mag_504, cmap="magma")
    ax_fft1.set_title("Phổ FFT 504px\n36×36 = 1.296 Patches (14×14px)", fontsize=9.3, fontweight="bold", color="#1d4ed8")
    ax_fft1.set_xticks([])
    ax_fft1.set_yticks([])

    ax_fft2 = fig.add_subplot(gs[1, 1])
    ax_fft2.imshow(mag_224, cmap="magma")
    ax_fft2.set_title("Phổ FFT 224px\n16×16 = 256 Patches (Giảm 5.06×)", fontsize=9.3, fontweight="bold", color="#dc2626")
    ax_fft2.set_xticks([])
    ax_fft2.set_yticks([])

    # Col 2-3 combined: Quantitative Resolution Impact Bar Chart (Same Vaults: 504px LANCZOS vs 224px BILINEAR)
    ax_bar = fig.add_subplot(gs[:, 2:])
    labels = [
        "M₀ Baseline\n(RV5.1 6k)",
        "M₁ Active R1\n(RV5.1 6k)",
        "M₂* Warm-Start\n(RV5.1 6k)",
        "M₃ Champion\n(Universal 20.4k)",
    ]
    hi_vals = [
        legacy_512["models"]["M0_baseline"]["overall_recall"] * 100.0,
        legacy_512["models"]["M1_active_round1"]["overall_recall"] * 100.0,
        legacy_512["models"]["M2_active_round2"]["overall_recall"] * 100.0,
        float(univ_504["M3_dual_engine"]["Fake Recall (↑)"].rstrip("%")),
    ]
    lo_vals = [
        legacy_224["models"]["M0_baseline"]["overall_recall"] * 100.0,
        legacy_224["models"]["M1_active_round1"]["overall_recall"] * 100.0,
        legacy_224["models"]["M2_active_round2"]["overall_recall"] * 100.0,
        float(univ_224["M3_dual_engine"]["Fake Recall (↑)"].rstrip("%")),
    ]
    x = np.arange(len(labels))
    w = 0.34
    b1 = ax_bar.bar(x - w / 2, hi_vals, w, label="Chuẩn 512→504px LANCZOS (1.296 patches)", color="#1d4ed8", edgecolor="#0f172a", linewidth=0.8)
    b2 = ax_bar.bar(x + w / 2, lo_vals, w, label="Lỗi nén 224×224 BILINEAR (256 patches)", color="#dc2626", edgecolor="#0f172a", linewidth=0.8)
    for b in b1:
        h = b.get_height()
        ax_bar.text(b.get_x() + b.get_width() / 2, h + 1.8, f"{h:.2f}%", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#1e40af")
    for b in b2:
        h = b.get_height()
        ax_bar.text(b.get_x() + b.get_width() / 2, h + 1.8, f"{h:.2f}%", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#991b1b")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9.2, fontweight="bold")
    ax_bar.set_ylim(0, 102)
    ax_bar.set_ylabel("Fake Recall (%)", fontsize=10.5, fontweight="bold")
    ax_bar.set_title(
        f"Đối Chứng Trực Tiếp Trước & Sau Khi Sửa Lỗi Phân Giải (Cùng Bộ Ảnh)\n"
        f"(Suy luận đúng 504px khôi phục {lo_vals[0]:.2f}% → {hi_vals[0]:.2f}% ở M₀ và {lo_vals[3]:.2f}% → {hi_vals[3]:.2f}% ở M₃)",
        fontsize=10.6,
        fontweight="bold",
        pad=10,
    )
    ax_bar.grid(axis="y", linestyle="--", alpha=0.45)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.legend(loc="upper left", fontsize=9.2, frameon=True)

    fig.suptitle(
        f"PHÂN TÍCH SUY GIẢM PHÂN GIẢI (504×504 VS 224×224) TRÊN MẪU {sample_path.name} CỦA DINOv2 ViT-S/14",
        fontsize=13.0,
        fontweight="bold",
        y=0.98,
        color="#0f172a",
    )
    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.04, right=0.98)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def render_dinov2_architecture_diagram(out_path: Path) -> None:
    """Render a clean left-to-right architectural diagram of DeepfakeImageDetector on a unified canvas."""
    fig = plt.figure(figsize=(18.5, 6.8), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_axes([0.01, 0.03, 0.98, 0.87])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    fig.suptitle(
        "KIẾN TRÚC DEEPFAKE IMAGE DETECTOR (DINOv2 ViT-S/14): TỪ LƯỚI PATCH 504×504 ĐẾN HIỆU CHUẨN XÁC SUẤT",
        fontsize=14.0,
        fontweight="bold",
        y=0.96,
        color="#0f172a",
    )

    # ---- Stage 1 & 2 (X: 1.5 -> 24.5): Real Face + 36x36 Grid + Zoomed 14x14 Patch Token ----
    s1_box = patches.FancyBboxPatch((1.5, 2.0), 23.0, 39.5, boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#2563eb", linewidth=1.8)
    ax.add_patch(s1_box)
    ax.text(13.0, 39.4, "1 & 2. TIỀN XỬ LÝ & PATCH GRID", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1e3a8a")
    ax.text(13.0, 36.8, "MetaDinoV2Wrapper (512→504×504)", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#475569")

    sample_face = Path("docs/reports/assets/samples/real_0.jpg")
    with Image.open(sample_face) as im:
        arr_504 = np.asarray(im.convert("RGB").resize((504, 504), Image.Resampling.LANCZOS), dtype=np.uint8)

    ax_face = ax.inset_axes([0.04, 0.25, 0.175, 0.53])
    ax_face.imshow(arr_504)
    for pos in range(0, 505, 14):
        lw = 1.0 if pos % (14 * 6) == 0 else 0.45
        al = 0.65 if pos % (14 * 6) == 0 else 0.30
        ax_face.axhline(pos, color="#2563eb", linewidth=lw, alpha=al)
        ax_face.axvline(pos, color="#2563eb", linewidth=lw, alpha=al)
    # Highlight one 28x28 (2x2 patch zone) around eye for clear visibility
    p_x, p_y = 14 * 12, 14 * 14
    ax_face.add_patch(patches.Rectangle((p_x, p_y), 28, 28, linewidth=2.2, edgecolor="#f59e0b", facecolor="none"))
    ax_face.set_xticks([])
    ax_face.set_yticks([])
    for sp in ax_face.spines.values():
        sp.set_edgecolor("#2563eb")
        sp.set_linewidth(1.6)

    # Mini zoomed patch preview + token sequence pill inside Stage 1
    ax_patch = ax.inset_axes([0.032, 0.08, 0.055, 0.12])
    ax_patch.imshow(arr_504[p_y : p_y + 28, p_x : p_x + 28])
    ax_patch.set_xticks([])
    ax_patch.set_yticks([])
    for sp in ax_patch.spines.values():
        sp.set_edgecolor("#f59e0b")
        sp.set_linewidth(2.0)

    ax.text(15.8, 8.7, "1 Patch 14×14px → ℝ³⁸⁴", ha="center", va="center", fontsize=8.6, fontweight="bold", color="#b45309")
    ax.text(15.8, 6.2, "[CLS] + 1.296 Tokens (36×36)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1e3a8a")
    ax.text(15.8, 3.8, "Chuẩn hóa ImageNet mean/std", ha="center", va="center", fontsize=8.0, color="#475569")

    # Connector 1 -> 2
    ax.annotate("", xy=(28.0, 21.5), xytext=(24.8, 21.5), arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.7", lw=2.4, color="#1d4ed8"))

    # ---- Stage 3 & 4 (X: 28.2 -> 59.5): DINOv2 ViT-S/14 Backbone Tower ----
    s2_box = patches.FancyBboxPatch((28.2, 2.0), 31.3, 39.5, boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#0f172a", linewidth=1.8)
    ax.add_patch(s2_box)
    ax.text(43.85, 39.4, "3 & 4. BACKBONE DINOv2 ViT-S/14", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.text(43.85, 36.8, "22.06M Tổng Params  •  Chỉ Fine-Tune 1.78M Params (8.06%)", ha="center", va="center", fontsize=8.6, fontweight="bold", color="#475569")

    # Token Input Pill at bottom of Backbone
    tok_pill = patches.FancyBboxPatch((29.6, 3.3), 28.5, 4.2, boxstyle="round,pad=0.2", facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=1.4)
    ax.add_patch(tok_pill)
    ax.text(43.85, 5.4, "Chuỗi Input Tensor:  (B,  1 + 1.296,  384)", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1e40af", family="monospace")

    ax.annotate("", xy=(43.85, 9.5), xytext=(43.85, 7.6), arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", lw=1.8, color="#475569"))

    # Frozen Tower: Blocks 0..10
    frz_box = patches.FancyBboxPatch((29.6, 9.7), 28.5, 13.6, boxstyle="round,pad=0.25", facecolor="#f1f5f9", edgecolor="#64748b", linewidth=1.8)
    ax.add_patch(frz_box)
    ax.text(43.85, 21.7, "[ĐÓNG BĂNG 91.94%]  Blocks 0 → 10 (11 Transformer Layers)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#334155")

    # Draw 6 blocks row 1 (Blk 0..5) + 5 blocks row 2 (Blk 6..10) centered symmetrically
    for c_i in range(6):
        bx = 30.8 + c_i * 4.45
        bp = patches.FancyBboxPatch((bx, 16.5), 3.85, 3.4, boxstyle="round,pad=0.12", facecolor="#cbd5e1", edgecolor="#475569", linewidth=1.0)
        ax.add_patch(bp)
        ax.text(bx + 1.92, 18.2, f"Blk {c_i}", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#1e293b")
    for c_i in range(5):
        bx = 33.0 + c_i * 4.45
        bp = patches.FancyBboxPatch((bx, 12.1), 3.85, 3.4, boxstyle="round,pad=0.12", facecolor="#cbd5e1", edgecolor="#475569", linewidth=1.0)
        ax.add_patch(bp)
        ax.text(bx + 1.92, 13.8, f"Blk {c_i + 6}", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#1e293b")
    ax.text(43.85, 10.7, "requires_grad = False  •  Giữ nguyên đặc trưng thị giác gốc", ha="center", va="center", fontsize=8.0, color="#475569", style="italic")

    ax.annotate("", xy=(43.85, 25.5), xytext=(43.85, 23.5), arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", lw=2.0, color="#d97706"))

    # Unfrozen Block 11 + LayerNorm
    unfrz_box = patches.FancyBboxPatch((29.6, 25.7), 28.5, 9.0, boxstyle="round,pad=0.25", facecolor="#fef3c7", edgecolor="#d97706", linewidth=2.2)
    ax.add_patch(unfrz_box)
    ax.text(43.85, 32.8, "[MỞ KHÓA 8.06%]  Block 11 + LayerNorm (1.78M Params)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#92400e")
    ax.text(43.85, 29.8, "Differential LR 10:1  (Block 11 = 0.1 × Head LR)", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#78350f")
    ax.text(43.85, 27.3, "PyTorch AMP bfloat16  •  Xuất Attention Heatmap", ha="center", va="center", fontsize=8.2, color="#78350f")

    # Connector Backbone -> Head
    ax.annotate("", xy=(62.8, 25.5), xytext=(59.8, 29.8), arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.7", lw=2.4, color="#1d4ed8", connectionstyle="arc3,rad=-0.12"))
    ax.text(61.1, 32.8, "[CLS]\n∈ ℝ³⁸⁴", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1d4ed8", bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec="none", alpha=0.85))

    # ---- Stage 4b (X: 63.0 -> 76.5): Classifier Head ----
    s3_box = patches.FancyBboxPatch((63.0, 8.0), 13.5, 29.5, boxstyle="round,pad=0.35", facecolor="#eff6ff", edgecolor="#2563eb", linewidth=2.0)
    ax.add_patch(s3_box)
    ax.text(69.75, 35.4, "CLASSIFIER HEAD", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1e40af")
    ax.text(69.75, 33.0, "(1.153 Params)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#475569")

    head_layers = [
        (26.2, "LayerNorm(384)", "768 params"),
        (19.0, "Dropout(p = 0.3)", "Chống đồng thích nghi"),
        (11.8, "Linear(384 → 1)", "385 params → Logit z"),
    ]
    for ly, ltitle, lsub in head_layers:
        lb = patches.FancyBboxPatch((64.0, ly), 11.5, 4.8, boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="#3b82f6", linewidth=1.3)
        ax.add_patch(lb)
        ax.text(69.75, ly + 3.1, ltitle, ha="center", va="center", fontsize=8.6, fontweight="bold", color="#0f172a", family="monospace")
        ax.text(69.75, ly + 1.2, lsub, ha="center", va="center", fontsize=7.6, color="#475569")
    ax.annotate("", xy=(69.75, 24.0), xytext=(69.75, 26.0), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#2563eb"))
    ax.annotate("", xy=(69.75, 16.8), xytext=(69.75, 18.8), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#2563eb"))

    # Connector Head -> Calibration
    ax.annotate("", xy=(79.5, 21.5), xytext=(76.7, 14.2), arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.7", lw=2.4, color="#059669", connectionstyle="arc3,rad=0.15"))
    ax.text(78.0, 10.6, "Logit\nz", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#059669", bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff", ec="none", alpha=0.85))

    # ---- Stage 5 (X: 79.7 -> 98.8): TemperatureScaler + Output Probability ----
    s4_box = patches.FancyBboxPatch((79.7, 2.0), 19.1, 39.5, boxstyle="round,pad=0.4", facecolor="#f0fdf4", edgecolor="#059669", linewidth=1.8)
    ax.add_patch(s4_box)
    ax.text(89.25, 39.4, "5. HIỆU CHUẨN L-BFGS", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#047857")
    ax.text(89.25, 36.8, "TemperatureScaler: P = σ(z / T)", ha="center", va="center", fontsize=8.6, fontweight="bold", color="#065f46")

    cl_path = Path("experiments/logs/closed_loop_metrics.json")
    cl_data = json.loads(cl_path.read_text(encoding="utf-8")) if cl_path.exists() else {}
    cl_rounds = cl_data.get("rounds", [])
    T = float(cl_rounds[-1].get("calibration_temperature", 2.4273)) if cl_rounds else 2.4273
    ax_cal = ax.inset_axes([0.834, 0.22, 0.145, 0.54])
    z = np.linspace(-5.0, 5.0, 300)
    ax_cal.axhspan(0.40, 0.60, color="#fef3c7", alpha=0.75, label="Uncertain ±0.10")
    ax_cal.plot(z, 1.0 / (1.0 + np.exp(-z)), "--", color="#94a3b8", linewidth=1.7, label="Trước T=1.0")
    ax_cal.plot(z, 1.0 / (1.0 + np.exp(-z / T)), color="#059669", linewidth=2.5, label=f"Sau T={T:.4f}")
    ax_cal.axhline(0.50, color="#dc2626", linestyle=":", linewidth=1.5, label="Ngưỡng τ=0.50")
    ax_cal.set_xlim(-5, 5)
    ax_cal.set_ylim(-0.02, 1.04)
    ax_cal.set_xlabel("Logit z", fontsize=8.2, fontweight="bold")
    ax_cal.set_ylabel("P(Fake)", fontsize=8.2, fontweight="bold")
    ax_cal.tick_params(labelsize=7.5)
    ax_cal.grid(True, linestyle=":", alpha=0.45)
    ax_cal.spines["top"].set_visible(False)
    ax_cal.spines["right"].set_visible(False)
    ax_cal.legend(loc="upper left", fontsize=7.0, frameon=True)

    ax.text(89.25, 6.4, f"T = {T:.4f} (Active Round 2)", ha="center", va="center", fontsize=8.6, fontweight="bold", color="#047857")
    ax.text(89.25, 3.9, "Real < 0.50  |  Fake ≥ 0.50", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#0f172a")

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def regenerate_soft_feathering_figure(out_path: Path) -> None:
    """Regenerate 01_diffusion_dataset_and_curation_cell8_img2.png without margin/legend collisions."""
    import cv2

    real_pil = Image.open("data/raw/faces/ffhq/00000.png").convert("RGB").resize((256, 256))
    synth_pil = Image.open("data/synthetic/d0_train_images/inpainting/fake_ffhq_00002_0.png").convert("RGB").resize((256, 256))
    real_mat = np.array(real_pil, dtype=np.float32) / 255.0
    synth_mat = np.array(synth_pil, dtype=np.float32) / 255.0

    h, w = 256, 256
    hard_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(hard_mask, (w // 2, int(h * 0.52)), (int(w * 0.28), int(h * 0.36)), 0, 0, 360, 1.0, -1)
    soft_mask = cv2.GaussianBlur(hard_mask, (25, 25), sigmaX=4.0)

    hard_blend = hard_mask[..., None] * synth_mat + (1.0 - hard_mask[..., None]) * real_mat
    soft_blend = soft_mask[..., None] * synth_mat + (1.0 - soft_mask[..., None]) * real_mat

    profile_y = int(h * 0.52)
    grad_hard = np.abs(np.gradient(hard_blend[profile_y, :, 0]))
    grad_soft = np.abs(np.gradient(soft_blend[profile_y, :, 0]))
    grad_real = np.abs(np.gradient(real_mat[profile_y, :, 0]))

    fig, axes = plt.subplots(1, 5, figsize=(16.5, 4.1), dpi=180, gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.45]})
    plt.subplots_adjust(wspace=0.45, top=0.80, bottom=0.14, left=0.02, right=0.98)

    axes[0].imshow(real_mat)
    axes[0].set_title("[1] Chân Dung Thật\n(FFHQ #00000)", fontsize=9.5, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(soft_mask, cmap="magma")
    axes[1].set_title("[2] Mặt Nạ Feathering\n(Minh họa σ = 4.0)", fontsize=9.5, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(hard_blend)
    axes[2].set_title("[3] Hard-Edge Blend\n(Biên cắt dán lộ liễu)", fontsize=9.5, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(soft_blend)
    axes[3].set_title("[4] Soft-Mask Blend\n(Đồng nhất cấu trúc da)", fontsize=9.5, fontweight="bold")
    axes[3].axis("off")

    x_coords = np.arange(w)
    axes[4].plot(x_coords, grad_hard, color="crimson", linewidth=1.6, label="Hard Mask (Spike viền)")
    axes[4].plot(x_coords, grad_soft, color="teal", linewidth=1.7, label="Soft Mask σ=4.0 (Mượt)")
    axes[4].plot(x_coords, grad_real, color="gray", linestyle=":", label="Ảnh Thật (Đối chứng)")
    axes[4].set_title("[5] Gradient Viền Biên\n(Triệt tiêu Artifact tần số cao)", fontsize=9.5, fontweight="bold")
    axes[4].set_xlabel("Tọa độ pixel X (ngang má)", fontsize=8.5, fontweight="bold")
    axes[4].set_ylabel("Biên độ gradient", fontsize=8.5, fontweight="bold", labelpad=6)
    axes[4].set_xlim(w // 2 - 80, w // 2 + 80)
    max_g = float(max(grad_hard.max(), grad_soft.max(), grad_real.max()))
    axes[4].set_ylim(-0.005, max_g * 1.28)
    axes[4].legend(fontsize=7.8, loc="upper left", frameon=True)
    axes[4].grid(True, linestyle="--", alpha=0.4)
    axes[4].spines["top"].set_visible(False)
    axes[4].spines["right"].set_visible(False)

    fig.suptitle(
        "CƠ CHẾ HÒA TRỘN SOFT-MASK GAUSSIAN FEATHERING (MINH HỌA σ = 4.0 | CODE blur_radius = 18–20)",
        fontsize=12.0,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def regenerate_universal_sweep_figure(out_path: Path) -> None:
    """Regenerate 03_champion_m3_and_universal_benchmark_cell9_img3.png with non-overlapping annotation."""
    import csv

    with open("experiments/benchmarks/tables/benchmark_5models_family_summary.csv", encoding="utf-8") as f:
        family_rows = list(csv.DictReader(f))
    with open("experiments/benchmarks/tables/benchmark_5models_global_summary.csv", encoding="utf-8") as f:
        global_rows = list(csv.DictReader(f))
    preds_data = np.load("experiments/benchmarks/logs/benchmark_5models_predictions.npz", allow_pickle=True)

    fake_families = [
        "Diffusion Baselines (DDIM)",
        "Latent Diffusion (SD 2.1)",
        "Diffusion Transformers (DiT)",
        "GAN Inversion & Editing",
        "GAN Architectures",
        "AutoEncoders / VAEs",
        "Face Swapping",
        "Reenactment & Talking Head",
    ]
    family_names_short = [
        "DDIM (Diffusion)",
        "SD 2.1 (Latent)",
        "DiT (Transformers)",
        "GAN Inversion",
        "GAN Archs",
        "AutoEncoders",
        "Face Swapping",
        "Talking Head",
    ]

    def _pct(val: str) -> float:
        return float(str(val).split("%")[0].strip())

    def _fam_stat(fam: str, col: str) -> float:
        for r in family_rows:
            if r["Family"] == fam:
                return _pct(r[col])
        return 0.0

    m0_fam = [_fam_stat(f, "M₀ (Baseline D₀) Recall/Spec") for f in fake_families]
    m2_fam = [_fam_stat(f, "M₂* (Refined Warm-Start) Recall/Spec") for f in fake_families]
    m3_fam = [_fam_stat(f, "M₃ (Dual-Engine Champion) Recall/Spec") for f in fake_families]

    y_true = preds_data["y_true"]
    probs_m3 = preds_data["probs_M3_dual_engine"]
    sources = preds_data["sources"]
    thresholds = np.linspace(0.05, 0.95, 19)
    sweep_recalls, sweep_specs, sweep_f1s, sweep_ff_fprs = [], [], [], []
    probs_ff = probs_m3[sources == "ff++_real"]

    for tau in thresholds:
        preds = (probs_m3 >= tau).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tn = np.sum((preds == 0) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        rec = tp / (tp + fn) * 100.0
        spec = tn / (tn + fp) * 100.0
        prec = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sweep_recalls.append(rec)
        sweep_specs.append(spec)
        sweep_f1s.append(f1)
        sweep_ff_fprs.append(float(np.mean(probs_ff >= tau) * 100.0))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.4), dpi=180)
    plt.subplots_adjust(hspace=0.36, wspace=0.25, top=0.90)

    y_pos = np.arange(len(family_names_short))
    bar_h = 0.26
    ax1 = axes[0, 0]
    ax1.barh(y_pos - bar_h, m0_fam, bar_h, label="M₀ Baseline", color="#95a5a6", edgecolor="black")
    ax1.barh(y_pos, m2_fam, bar_h, label="M₂* Refined", color="#2980b9", edgecolor="black")
    b_m3 = ax1.barh(y_pos + bar_h, m3_fam, bar_h, label="M₃ Champion", color="#27ae60", edgecolor="black")
    for b in b_m3:
        ax1.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", ha="left", fontsize=8.2, fontweight="bold", color="#27ae60")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(family_names_short, fontsize=8.8, fontweight="bold")
    ax1.set_xlabel("Fake Recall (%)", fontsize=9.2, fontweight="bold")
    ax1.set_title(
        f"Đột Phá Fake Recall Theo 8 Họ Công Nghệ (@ 512→504px LANCZOS)\n"
        f"(GAN {m3_fam[4]:.1f}%, SD 2.1 {m3_fam[1]:.1f}%, GAN Inv {m3_fam[3]:.1f}%, DDIM {m3_fam[0]:.1f}%)",
        fontsize=9.8,
        fontweight="bold",
    )
    ax1.set_xlim(0, max(72.0, max(m3_fam) + 12.0))
    ax1.legend(fontsize=8.2, loc="lower right")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    ax2 = axes[0, 1]
    models_g = ["M₀", "M₁", "M₂ (Cold)", "M₂*", "M₃ (Champion)"]
    rec_g = [_pct(r["Fake Recall (↑)"]) for r in global_rows]
    f1_g = [_pct(r["F1-Score"]) for r in global_rows]
    x_ind = np.arange(len(models_g))
    w_g = 0.35
    bg1 = ax2.bar(x_ind - w_g / 2, rec_g, w_g, label="Fake Recall (↑)", color="#e74c3c", edgecolor="black")
    bg2 = ax2.bar(x_ind + w_g / 2, f1_g, w_g, label="F1-Score (↑)", color="#3498db", edgecolor="black")
    for b in bg1:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.2f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#c0392b")
    for b in bg2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.2f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#2980b9")
    ax2.set_xticks(x_ind)
    ax2.set_xticklabels(models_g, fontsize=8.8, fontweight="bold")
    ax2.set_ylabel("Phần Trăm (%)", fontsize=9.2, fontweight="bold")
    ax2.set_title(
        f"Hiệu Năng Toàn Cầu Trên 20.400 Ảnh (@ 512→504px)\n"
        f"(M₃ dẫn đầu với Fake Recall {rec_g[-1]:.2f}% & F1 {f1_g[-1]:.2f}%)",
        fontsize=9.8,
        fontweight="bold",
    )
    ax2.set_ylim(0, max(42.0, max(f1_g) + 8.0))
    ax2.legend(fontsize=8.2, loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    ax3 = axes[1, 0]
    ax3.plot(thresholds, sweep_recalls, "r-o", linewidth=2, label="Fake Recall (%)", markersize=4)
    ax3.plot(thresholds, sweep_specs, "g-s", linewidth=2, label="Real Specificity (%)", markersize=4)
    ax3.plot(thresholds, sweep_f1s, "b-^", linewidth=2, label="F1-Score (%)", markersize=4)
    ax3.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Ngưỡng Mặc Định (τ = 0.5)")
    rec_05 = rec_g[-1]
    spec_05 = _pct(global_rows[-1]["Real Specificity (↑)"])
    ax3.scatter([0.5], [rec_05], color="red", s=80, zorder=5)
    ax3.annotate(
        f"τ=0.5: Recall={rec_05:.2f}%\nSpec={spec_05:.2f}%",
        xy=(0.5, rec_05),
        xytext=(0.56, min(65.0, rec_05 + 22.0)),
        bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#94a3b8", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.2),
        fontsize=8.5,
        fontweight="bold",
        color="#1e293b",
    )
    ax3.set_xlabel("Ngưỡng Quyết Định τ (Decision Threshold)", fontsize=9.2)
    ax3.set_ylabel("Tỷ Lệ Phần Trăm (%)", fontsize=9.2)
    ax3.set_title("Đường Cong Quét Ngưỡng Quyết Định (Threshold Sweep τ in [0.05, 0.95])\n(Đánh đổi giữa Fake Recall và Real Specificity)", fontsize=10.0, fontweight="bold")
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=8.2, loc="center right")
    ax3.grid(True, linestyle="--", alpha=0.4)

    ax4 = axes[1, 1]
    ff_fp = int(np.sum(probs_ff >= 0.5))
    ff_tot = len(probs_ff)
    ff_fpr_05 = (ff_fp / ff_tot * 100.0) if ff_tot else 0.0
    ax4.plot(thresholds, sweep_ff_fprs, "m-d", linewidth=2.2, markersize=4.5, label="FF++ Video FPR (%)")
    ax4.axvline(0.5, color="black", linestyle="--", linewidth=1.2)
    ax4.axhline(ff_fpr_05, color="#8e44ad", linestyle=":", linewidth=1.2)
    ax4.scatter([0.5], [ff_fpr_05], color="#8e44ad", s=80, zorder=5)
    ax4.text(0.52, ff_fpr_05 + 1.2, f"τ=0.5: FPR = {ff_fpr_05:.2f}%\n({ff_fp}/{ff_tot:,} khung báo động nhầm)".replace(",", "."), fontsize=8.5, fontweight="bold", color="#8e44ad")
    ax4.set_xlabel("Ngưỡng Quyết Định τ (Decision Threshold)", fontsize=9.2)
    ax4.set_ylabel("Tỷ Lệ Báo Động Nhầm (FPR %)", fontsize=9.2)
    ax4.set_title(f"Khả Năng Kháng Nhiễu Nén Video Trên FaceForensics++ C23\n({ff_tot:,} khung hình nén H.264 — FPR @ τ=0.5 là {ff_fpr_05:.2f}%)".replace(",", "."), fontsize=9.8, fontweight="bold")
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0, max(25.0, max(sweep_ff_fprs) + 5.0))
    ax4.legend(fontsize=8.2, loc="upper right")
    ax4.grid(True, linestyle="--", alpha=0.4)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("SÁT HẠCH TOÀN CẦU UNIVERSAL BENCHMARK (20.400 ẢNH, 34 GENERATORS, 8 HỌ CÔNG NGHỆ)", fontsize=12.5, fontweight="bold", y=0.98)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def regenerate_xai_inspector_figure(out_path: Path) -> None:
    """Regenerate 00_master_deepfake_defense_showcase_cell7_img1.png with proper Vietnamese text & non-clipped cards."""
    import torch.nn.functional as F
    from src.models.detector.image_detector import DeepfakeImageDetector

    detector = DeepfakeImageDetector(backbone="dinov2_small", pretrained=False)
    ckpt = torch.load("experiments/checkpoints/active_learning_m3_best.pt", map_location="cpu", weights_only=False)
    detector.load_state_dict(ckpt["model_state_dict"])
    detector.eval()
    detector.extractor.backbone.set_attn_implementation("eager")

    samples = [
        {
            "title": "Mẫu 1: Ảnh Chân Dung Thật (FFHQ #00301)",
            "source": "FFHQ Real Manifold (Khuôn mặt thật)",
            "path": Path("data/raw/faces/ffhq/00301.png"),
            "true_label": 0,
        },
        {
            "title": "Mẫu 2: Mẫu Khó Challenge Vault (SD 1.5 Inpainting)",
            "source": "Challenge Vault (SD 1.5 Biên Mờ Hồ)",
            "path": Path("data/active_learning/round_1/images/inpainting/fake_ffhq_00000_q1_116.png"),
            "true_label": 1,
        },
        {
            "title": "Mẫu 3: Ảnh Sinh Siêu Thực (RV5.1 #187)",
            "source": "Unseen RV5.1 Vault (Img2Img Photorealistic)",
            "path": Path("data/external_benchmarks/synthetic_face_dataset_6k/img2img/187_img2img.png"),
            "true_label": 1,
        },
    ]

    fig, axes = plt.subplots(3, 3, figsize=(17.2, 14.6), dpi=170, gridspec_kw={"width_ratios": [1.0, 1.0, 1.55]})
    plt.subplots_adjust(hspace=0.28, wspace=0.14, top=0.92, bottom=0.03, left=0.03, right=0.97)

    mean_buf = detector.extractor.mean
    std_buf = detector.extractor.std

    for idx, sample in enumerate(samples):
        raw_pil = Image.open(sample["path"]).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        img_np = np.array(raw_pil, dtype=np.float32) / 255.0
        x_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        norm_x = F.interpolate((x_t - mean_buf) / std_buf, size=(504, 504), mode="bilinear", align_corners=False)

        with torch.no_grad():
            out = detector.extractor.backbone(pixel_values=norm_x, output_attentions=True)
            p_fake = float(torch.sigmoid(detector(x_t)).item())

        att = out.attentions[-1]
        cls_attn = att[:, :, 0, 1:].mean(dim=1).view(1, 1, 36, 36)
        heatmap = F.interpolate(cls_attn, size=(512, 512), mode="bilinear", align_corners=False)[0, 0].numpy()
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        max_y, max_x = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
        y_rel, x_rel = max_y / 512.0, max_x / 512.0
        if y_rel < 0.45 and 0.25 < x_rel < 0.75:
            region = "Vùng Mắt / Hốc Mắt (Periorbital)"
        elif 0.45 <= y_rel < 0.65 and 0.35 < x_rel < 0.65:
            region = "Vùng Mũi / Sống Mũi (Nasal Ridge)"
        elif y_rel >= 0.65 and 0.30 < x_rel < 0.70:
            region = "Vùng Miệng / Răng / Môi (Oral & Dental)"
        elif x_rel <= 0.25 or x_rel >= 0.75:
            region = "Vành Tai / Đường Biên Mép (Facial Margin)"
        else:
            region = "Kết Cấu Da Mặt (Micro-Skin Texture)"

        pred_label = "DEEPFAKE (GIẢ)" if p_fake >= 0.5 else "REAL FACE (THẬT)"
        pred_color = "#c0392b" if p_fake >= 0.5 else "#27ae60"
        conf = abs(p_fake - 0.5) * 200.0
        is_correct = (p_fake >= 0.5) == (sample["true_label"] == 1)
        status_str = "CHÍNH XÁC (CORRECT)" if is_correct else "SAI LỆCH (INCORRECT)"

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f"{sample['title']}\n[{sample['source']}]", fontsize=9.5, fontweight="bold")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(img_np)
        axes[idx, 1].imshow(heatmap_norm, cmap="jet", alpha=0.55)
        axes[idx, 1].set_title(f"DINOv2 Spatial Attention Heatmap\n(Tiêu điểm chú ý: Y={max_y}, X={max_x})", fontsize=9.5, fontweight="bold")
        axes[idx, 1].axis("off")

        ax_card = axes[idx, 2]
        ax_card.set_xlim(0, 1)
        ax_card.set_ylim(0, 1)
        ax_card.axis("off")
        card_text = (
            f"[BÁO CÁO GIÁM ĐỊNH PHÁP Y THỊ GIÁC]\n"
            f"─────────────────────────────────────────\n"
            f"• Mẫu kiểm định: {sample['path'].name}\n"
            f"• Nguồn dữ liệu: {sample['source']}\n"
            f"• Nhãn gốc (GT): {'FAKE (1)' if sample['true_label'] == 1 else 'REAL (0)'}\n"
            f"─────────────────────────────────────────\n"
            f"• Xác suất Giả P(Fake):   {p_fake * 100:.2f}%\n"
            f"• Xác suất Thật P(Real): {(1.0 - p_fake) * 100:.2f}%\n"
            f"• Phân loại (τ = 0.50):  {pred_label}\n"
            f"• Độ tin cậy (Margin):   {conf:.2f}%\n"
            f"• Vùng chú ý pháp y:\n  → {region}\n"
            f"─────────────────────────────────────────\n"
            f"• Mô hình: Champion M₃ Dual-Engine\n"
            f"• Kết quả: {status_str}"
        )
        ax_card.text(
            0.03,
            0.5,
            card_text,
            fontsize=8.8,
            fontfamily="monospace",
            verticalalignment="center",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.65", facecolor="#f8f9fa", edgecolor=pred_color, linewidth=2.2),
        )

    fig.suptitle("TRẠM GIÁM ĐỊNH PHÁP Y THỊ GIÁC TRỰC TIẾP (LIVE FORENSIC INSPECTOR — CHAMPION M₃)", fontsize=13.0, fontweight="bold", y=0.97)
    fig.savefig(out_path, bbox_inches="tight", dpi=170)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("dual_engine_mask_anatomy.png", render_mask_anatomy),
        ("dual_engine_visual_3modes.png", render_visual_3modes_comparison),
        ("dual_engine_prompt_quota_evolution.png", render_prompt_and_quota_evolution),
        ("resolution_504_vs_224_degradation.png", render_resolution_degradation_comparison),
        ("dinov2_detector_architecture.png", render_dinov2_architecture_diagram),
        ("01_diffusion_dataset_and_curation_cell8_img2.png", regenerate_soft_feathering_figure),
        ("03_champion_m3_and_universal_benchmark_cell9_img3.png", regenerate_universal_sweep_figure),
        ("00_master_deepfake_defense_showcase_cell7_img1.png", regenerate_xai_inspector_figure),
    ]
    for fname, fn in targets:
        out_path = OUT_DIR / fname
        fn(out_path)
        size_kb = out_path.stat().st_size / 1024.0
        logger.info("Generated %s (%.1f KB)", out_path, size_kb)


if __name__ == "__main__":
    main()
