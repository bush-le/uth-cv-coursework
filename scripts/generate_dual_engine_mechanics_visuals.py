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

    stages = [
        ("SD 1.5 D₀\n(2.200 sạch)", _counts(Path("data/synthetic/curated_manifest.jsonl"), "manipulation_type")),
        ("SD 1.5 AL R1\n(1.900 thô)", _counts(Path("data/active_learning/round_1/candidates_raw.jsonl"), "manipulation_type")),
        ("SD 1.5 AL R2\n(1.900 thô)", _counts(Path("data/active_learning/round_2/candidates_raw.jsonl"), "manipulation_type")),
        ("RV5.1 6k\n(6.000 mẫu)", _counts(Path("data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl"), "task")),
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
    ax_l.set_title("Phân Bổ Thực Tế 3 Chế Độ Theo Từng Giai Đoạn\n(R2 Tự Dồn 47.2% Vào SD Inpaint Theo Error-Steering)", fontsize=11.0, fontweight="bold", pad=10)
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("dual_engine_mask_anatomy.png", render_mask_anatomy),
        ("dual_engine_visual_3modes.png", render_visual_3modes_comparison),
        ("dual_engine_prompt_quota_evolution.png", render_prompt_and_quota_evolution),
    ]
    for fname, fn in targets:
        out_path = OUT_DIR / fname
        fn(out_path)
        size_kb = out_path.stat().st_size / 1024.0
        logger.info("Generated %s (%.1f KB)", out_path, size_kb)


if __name__ == "__main__":
    main()
