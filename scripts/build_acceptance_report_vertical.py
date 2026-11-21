"""Build script for Topic 11 Final Acceptance Vertical Scientific Report.

Generates docs/reports/presentation_final_acceptance.html with:
- Exactly 1 artifact per row (NO multi-column squished grids)
- Full-width, high-resolution, unconstrained visual artifact containers
- Interactive Lightbox Zoom Modal (click any image to inspect in full-screen)
- Compact, slim, collapsible sidebar navigation (takes ~15% width instead of ~50%)
- Clean native typography without raw LaTeX syntax ($ replaced with clean Unicode)
- 30+ visual artifacts and high-res face samples
"""

import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUTPUT_PATH = Path("docs/reports/presentation_final_acceptance.html")


def generate_m0_collapse_chart() -> None:
    """Generate M0 Baseline failure comparison chart from real JSON evaluation logs."""
    quad_path = Path("experiments/benchmarks/tables/benchmark_5models_quad_vault_summary.json")
    legacy_path = Path("experiments/logs/legacy_6k_benchmark_results.json")
    out_path = Path("docs/reports/assets/benchmarks/m0_baseline_collapse_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quad = json.loads(quad_path.read_text(encoding="utf-8"))
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))

    m0_quad = quad["M0 (Baseline)"]
    m0_legacy = legacy["models"]["M0_baseline"]

    labels = [
        "Master Frozen Test\n(SD 1.5, 512px, 1.178 ảnh)",
        "Challenge Vault\n(Mẫu Khó, 512px, 213 ảnh)",
        "Unseen RV5.1 Vault\n(Test-Derived, 512px, 1.215 ảnh)",
        "Legacy 6k RV5.1\n(Stress-Test 224px, 6.000 ảnh)",
    ]
    recalls = [
        m0_quad["Master_Frozen_Test"]["recall"] * 100.0,
        m0_quad["Challenge_Hard_Test"]["recall"] * 100.0,
        m0_quad["Unseen_RV51_OOD_Test"]["recall"] * 100.0,
        m0_legacy["overall_recall"] * 100.0,
    ]
    bypasses = [
        m0_quad["Master_Frozen_Test"]["bypass_rate"] * 100.0,
        m0_quad["Challenge_Hard_Test"]["bypass_rate"] * 100.0,
        m0_quad["Unseen_RV51_OOD_Test"]["bypass_rate"] * 100.0,
        m0_legacy["overall_bypass_rate"] * 100.0,
    ]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(12.5, 5.4), dpi=220)
    b1 = ax.bar(x - width / 2, recalls, width, label="Fake Recall (% ↑)", color="#2563eb", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width / 2, bypasses, width, label="Bypass Rate / Bỏ Lọt (% ↓)", color="#dc2626", edgecolor="black", linewidth=0.8)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1e40af")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#991b1b")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Tỷ Lệ Phần Trăm (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 114)
    ax.set_title("Thực Trạng Mô Hình Cơ Sở M₀ (Baseline D₀): Tỷ Lệ Bắt Giả vs Bỏ Lọt Trên 4 Kịch Bản Kiểm Thử", fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def regenerate_clean_defect_taxonomy() -> None:
    """Regenerate biological defect taxonomy figure with non-overlapping wrapped titles."""
    out_path = Path("docs/reports/assets/extracted_nb/02_hard_samples_and_active_learning_cell6_img2.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    defect_samples = [
        {
            "name": "[1] Mắt & Phản Xạ Giác Mạc",
            "file": "data/active_learning/round_1/images/inpainting/fake_ffhq_00000_q1_116.png",
            "roi_box": (150, 180, 360, 270),
            "desc": "Lệch trục đồng tử, phản quang giác mạc không trùng hướng chiếu sáng",
            "color": "#dc2626",
        },
        {
            "name": "[2] Vành Tai & Sụn Giải Phẫu",
            "file": "data/active_learning/round_1/images/inpainting/fake_ffhq_00010_q1_424.png",
            "roi_box": (350, 210, 470, 350),
            "desc": "Sụn vành tai ngoài biến dạng, mờ nhòe, nối bất thường với cơ má",
            "color": "#d97706",
        },
        {
            "name": "[3] Hàm Răng & Biên Kẽ Răng",
            "file": "data/active_learning/round_1/images/inpainting/fake_ffhq_00014_q1_126.png",
            "roi_box": (180, 320, 330, 420),
            "desc": "Dị tật 'unitooth' (răng liền khối), mất rãnh kẽ răng sinh học",
            "color": "#7c3aed",
        },
        {
            "name": "[4] Da & Vi Cấu Trúc Chân Lông",
            "file": "data/active_learning/round_1/images/inpainting/fake_ffhq_00018_q1_429.png",
            "roi_box": (190, 190, 330, 310),
            "desc": "Vùng da láng mịn giả tạo (plastic skin), mất vi lỗ chân lông",
            "color": "#2563eb",
        },
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), dpi=160)
    plt.subplots_adjust(hspace=0.34, wspace=0.18, top=0.86, bottom=0.04, left=0.02, right=0.98)

    for idx, item in enumerate(defect_samples):
        img_np = np.array(Image.open(item["file"]).convert("RGB"))
        x1, y1, x2, y2 = item["roi_box"]

        ax_top = axes[0, idx]
        ax_top.imshow(img_np)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5, edgecolor=item["color"], facecolor="none", linestyle="--")
        ax_top.add_patch(rect)
        ax_top.set_title(f"{item['name']}\n(Toàn mặt định vị)", fontsize=10, fontweight="bold", pad=8)
        ax_top.axis("off")

        ax_bot = axes[1, idx]
        ax_bot.imshow(img_np[y1:y2, x1:x2])
        for spine in ax_bot.spines.values():
            spine.set_edgecolor(item["color"])
            spine.set_linewidth(3)
        wrapped = textwrap.fill(item["desc"], width=34)
        ax_bot.set_title(f"Vi Vết Giải Phẫu Phóng Đại\n{wrapped}", fontsize=8.5, fontweight="bold", color=item["color"], pad=8)
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])

    fig.suptitle("KHÁM NGHIỆM THỊ GIÁC: 4 DỊ TẬT SINH HỌC CỦA MÔ HÌNH KHUẾCH TÁN (BIOLOGICAL DEFECT TAXONOMY)", fontsize=12.5, fontweight="bold", y=0.96)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def regenerate_clean_d3_audit() -> None:
    """Regenerate D3 clean partition audit figure from actual JSONL manifests without overlapping titles."""
    out_path = Path("docs/reports/assets/extracted_nb/03_champion_m3_and_universal_benchmark_cell3_img0.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _counts(path_str: str):
        reals, fakes = 0, 0
        with open(path_str, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lbl = json.loads(line)["label"]
                    if lbl == 0:
                        reals += 1
                    else:
                        fakes += 1
        return reals, fakes

    tr_r, tr_f = _counts("data/benchmarks/D3_clean/train.jsonl")
    va_r, va_f = _counts("data/benchmarks/D3_clean/val.jsonl")
    te_r, te_f = _counts("data/benchmarks/D3_clean/unseen_rv51_test.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    splits = ["D3 Train", "Validation", "Unseen Test (RV5.1)"]
    reals_counts = [tr_r, va_r, te_r]
    fakes_counts = [tr_f, va_f, te_f]
    x = np.arange(len(splits))
    w = 0.35

    b1 = axes[0].bar(x - w / 2, reals_counts, w, label="Ảnh Thật (Real, y=0)", color="#27ae60", edgecolor="black", linewidth=0.8)
    b2 = axes[0].bar(x + w / 2, fakes_counts, w, label="Ảnh Giả (Fake, y=1)", color="#e74c3c", edgecolor="black", linewidth=0.8)
    for b in list(b1) + list(b2):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 55, f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits, fontsize=9.5, fontweight="bold")
    axes[0].set_ylabel("Số Lượng Ảnh (Images)", fontsize=9.5, fontweight="bold")
    axes[0].set_title("Cơ Cấu Phân Bổ Real vs Fake Qua 3 Tập Dữ Liệu D₃\n(Cân bằng 1:1 và cách ly hoàn toàn 626 ảnh test)", fontsize=10.5, fontweight="bold", pad=10)
    axes[0].set_ylim(0, 3350)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    labels_donut = [
        "RV5.1 Sạch ID\n(1,384 ảnh - 49.2%)",
        "SD 1.5 Baseline\n(800 ảnh - 28.4%)",
        "AL Hard Samples\n(629 ảnh - 22.4%)",
    ]
    sizes_donut = [1384, 800, 629]
    colors_donut = ["#e67e22", "#3498db", "#9b59b6"]
    _, _, autotexts = axes[1].pie(
        sizes_donut,
        labels=labels_donut,
        autopct="%1.1f%%",
        startangle=120,
        colors=colors_donut,
        textprops={"fontsize": 9},
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(9.5)
        at.set_fontweight("bold")
    axes[1].set_title("Cơ Cấu 2.813 Ảnh Giả Trong D₃ Train\n(Đa Động Cơ: RV5.1 + SD 1.5 + AL Hard Samples)", fontsize=10.5, fontweight="bold", pad=10)

    fig.suptitle("KIỂM TOÁN DỮ LIỆU D₃: ZERO IDENTITY LEAKAGE & CƠ CẤU ĐA ĐỘNG CƠ KHUẾCH TÁN", fontsize=12.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_m2star_rv51_profiling_chart() -> None:
    """Generate M2* OOD hard-sample profiling chart on 6,000 RV5.1 images from legacy_6k_predictions.npz."""
    npz_path = Path("experiments/logs/legacy_6k_predictions.npz")
    out_path = Path("docs/reports/assets/benchmarks/m2star_rv51_hard_profiling.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    npz = np.load(npz_path, allow_pickle=True)
    m2_probs = npz["m2_probs"]
    tasks = npz["tasks"]
    regions = npz["sub_regions"]

    categories = [
        "img2img\n(n=2.000)",
        "inpaint: mouth\n(n=1.000)",
        "inpaint: eyes\n(n=1.000)",
        "inpaint (tổng)\n(n=2.000)",
        "text2img\n(n=2.000)",
        "TOÀN BỘ RV5.1\n(n=6.000)",
    ]
    masks = [
        tasks == "img2img",
        regions == "mouth",
        regions == "eyes",
        tasks == "inpaint",
        tasks == "text2img",
        np.ones(len(m2_probs), dtype=bool),
    ]
    bypass_rates = [float(np.mean(m2_probs[m] < 0.5)) * 100.0 for m in masks]
    bypass_counts = [int(np.sum(m2_probs[m] < 0.5)) for m in masks]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), dpi=200)

    # Left: Bypass rate breakdown
    colors = ["#dc2626", "#ea580c", "#f59e0b", "#d97706", "#2563eb", "#0f172a"]
    x = np.arange(len(categories))
    bars = axes[0].bar(x, bypass_rates, color=colors, edgecolor="black", width=0.58, alpha=0.9)
    for bar, cnt in zip(bars, bypass_counts):
        h = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.2,
            f"{h:.2f}%\n({cnt:,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=8.5, fontweight="bold")
    axes[0].set_ylabel("Tỷ Lệ Bỏ Lọt M₂* (Bypass Rate % ↓)", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 52)
    axes[0].set_title("Phân Rã Điểm Mù M₂* Theo Chế Độ Sinh RV5.1\n(224×224 Stress-Test: 1.792 / 6.000 ảnh bị bỏ lọt)", fontsize=11, fontweight="bold", pad=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: P(Fake) probability distribution
    axes[1].hist(m2_probs, bins=40, range=(0, 1), color="#3b82f6", edgecolor="white", alpha=0.85)
    axes[1].axvspan(0.0, 0.5, color="#fee2e2", alpha=0.45, label="Vùng Bỏ Lọt (P < 0.50): 1.792 ảnh (29.87%)")
    axes[1].axvspan(0.35, 0.65, color="#fef3c7", alpha=0.45, label="Vùng Phân Vân (|P - 0.50| ≤ 0.15): 2.611 ảnh (43.52%)")
    axes[1].axvline(0.5, color="#dc2626", linestyle="--", linewidth=2, label="Ngưỡng quyết định τ = 0.50")
    axes[1].set_xlabel("Xác Suất Dự Đoán Giả M₂* — P(Fake)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Số Lượng Ảnh", fontsize=10, fontweight="bold")
    axes[1].set_title("Phân Phối Điểm Tin Cậy M₂* Trên 6.000 Ảnh RV5.1\n(Tổng hợp vùng khó P < 0.50 hoặc |P - 0.50| ≤ 0.15: 3.633 ảnh)", fontsize=11, fontweight="bold", pad=10)
    axes[1].legend(loc="upper left", fontsize=8.5, frameon=True)
    axes[1].grid(axis="y", linestyle="--", alpha=0.45)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("OOD HARD-SAMPLE PROFILING BẰNG M₂* (active_learning_m2_best.pt) TRÊN 6.000 ẢNH RV5.1", fontsize=12.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_m3_convergence_chart() -> None:
    """Generate Champion M3 6-epoch training convergence chart from verified training logs."""
    out_path = Path("docs/reports/assets/benchmarks/m3_training_convergence.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array([1, 2, 3, 4, 5, 6])
    train_loss = np.array([0.1833, 0.1318, 0.1131, 0.0969, 0.0900, 0.0831])
    val_loss = np.array([0.2552, 0.2757, 0.2394, 0.2317, 0.2213, 0.2168])
    val_auc = np.array([0.9745, 0.9778, 0.9820, 0.9822, 0.9841, 0.9846])
    val_acc = np.array([92.90, 92.49, 93.57, 93.82, 93.91, 94.07])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), dpi=200)

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2.4, color="#2563eb", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="s", linewidth=2.4, color="#dc2626", label="Validation Loss (D3 Val)")
    for e, tl, vl in zip(epochs, train_loss, val_loss):
        axes[0].text(e, tl - 0.013, f"{tl:.4f}", ha="center", va="top", fontsize=8, color="#1e40af", fontweight="bold")
        axes[0].text(e, vl + 0.008, f"{vl:.4f}", ha="center", va="bottom", fontsize=8, color="#991b1b", fontweight="bold")
    axes[0].set_xticks(epochs)
    axes[0].set_xlabel("Epoch (CosineAnnealingLR, T_max=6)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("BCEWithLogits Loss", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0.04, 0.31)
    axes[0].set_title("Động Học Giảm Hàm Mất Mát (Loss Convergence)", fontsize=11, fontweight="bold", pad=10)
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(loc="upper right", fontsize=9, frameon=True)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    ax2 = axes[1].twinx()
    l1 = axes[1].plot(epochs, val_auc, marker="o", linewidth=2.5, color="#059669", label="Validation ROC-AUC (↑)")
    l2 = ax2.plot(epochs, val_acc, marker="^", linestyle="--", linewidth=2.0, color="#7c3aed", label="Validation Accuracy % (↑)")
    for e, auc_v in zip(epochs, val_auc):
        axes[1].text(e, auc_v + 0.0008, f"{auc_v:.4f}", ha="center", va="bottom", fontsize=8, color="#065f46", fontweight="bold")
    axes[1].set_xticks(epochs)
    axes[1].set_xlabel("Epoch (Warm-Start từ M₂*)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Validation ROC-AUC", fontsize=10, fontweight="bold", color="#059669")
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=10, fontweight="bold", color="#7c3aed")
    axes[1].set_ylim(0.970, 0.989)
    ax2.set_ylim(91.5, 95.0)
    axes[1].set_title("Tăng Trưởng Validation AUC & Accuracy Qua 6 Epochs\n(Best Epoch 6: Val AUC = 0.9846, Val Acc = 94.07%)", fontsize=11, fontweight="bold", pad=10)
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    axes[1].legend(lines, labels, loc="lower right", fontsize=8.5, frameon=True)

    fig.suptitle("HUẤN LUYỆN CHAMPION M₃ DUAL-ENGINE (Block 11 + Head, 1.78M Params, AMP)", fontsize=12.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_curation_funnel_chart() -> None:
    """Generate SD 1.5 CurationGate funnel & rejection breakdown chart from actual manifests and curation reports."""
    from collections import Counter

    out_path = Path("docs/reports/assets/benchmarks/curation_funnel_sd15.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _count_lines(path_str: str) -> int:
        with open(path_str, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    b1_raw = _count_lines("data/synthetic/fake_manifest.jsonl")
    b1_cur = _count_lines("data/synthetic/curated_manifest.jsonl")
    r1_raw = _count_lines("data/active_learning/round_1/candidates_raw.jsonl")
    r1_cur = _count_lines("data/active_learning/round_1/candidates_curated.jsonl")
    r2_raw = _count_lines("data/active_learning/round_2/candidates_raw.jsonl")
    r2_cur = _count_lines("data/active_learning/round_2/candidates_curated.jsonl")

    stages = [
        f"Block 1 (D₀)\nPass {b1_cur/b1_raw*100:.1f}%",
        f"AL Round 1\nPass {r1_cur/r1_raw*100:.1f}%",
        f"AL Round 2\nPass {r2_cur/r2_raw*100:.1f}%",
        f"TỔNG SD 1.5\nPass {(b1_cur+r1_cur+r2_cur)/(b1_raw+r1_raw+r2_raw)*100:.1f}%",
    ]
    raws = [b1_raw, r1_raw, r2_raw, b1_raw + r1_raw + r2_raw]
    curs = [b1_cur, r1_cur, r2_cur, b1_cur + r1_cur + r2_cur]
    rejs = [r - c for r, c in zip(raws, curs)]

    # Rejection reasons across AL Round 1 + Round 2 (exact JSON logs)
    rej_counter: Counter = Counter()
    label_map = {
        "Face occupies too small area of frame": "Mặt quá nhỏ (<20% khung)",
        "No valid face landmark detected": "Không phát hiện được mặt",
        "No face landmark detected": "Không phát hiện được mặt",
        "Forehead / scalp amputated by top canvas border": "Cắt cụt vùng trán (mép trên)",
        "Multiple faces detected": "Nhiều hơn 1 khuôn mặt",
        "Face occupies too large area of frame": "Mặt quá lớn (>65% khung)",
        "Severe highlight blowout on face": "Cháy sáng cục bộ (>8%)",
        "Face is severely off-center": "Khuôn mặt lệch tâm nặng",
        "Chin / jaw amputated by bottom canvas border": "Cắt cụt vùng cằm (mép dưới)",
        "Severe horizontal boundary discontinuity across face": "Đứt gãy biên ngang vùng mặt",
    }
    for rp in ["data/active_learning/round_1/curation_report.json", "data/active_learning/round_2/curation_report.json"]:
        with open(rp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if not rec.get("is_valid"):
                        for reason in rec.get("rejection_reasons", []):
                            key = reason.split("(")[0].split(":")[0].strip()
                            rej_counter[label_map.get(key, key)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), dpi=200)

    # Left: Stacked bar (Kept vs Rejected)
    x = np.arange(len(stages))
    b_cur = axes[0].bar(x, curs, width=0.52, color="#10b981", edgecolor="black", label="Đạt chuẩn (Kept)")
    b_rej = axes[0].bar(x, rejs, bottom=curs, width=0.52, color="#ef4444", edgecolor="black", alpha=0.88, label="Bị loại (Rejected)")
    for i, (c, r, tot) in enumerate(zip(curs, rejs, raws)):
        axes[0].text(i, c / 2, f"{c:,}", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
        axes[0].text(i, c + r / 2, f"-{r:,}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
        axes[0].text(i, tot + 130, f"Thô: {tot:,}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0f172a")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages, fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Số Lượng Ảnh (SD 1.5)", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 7400)
    axes[0].set_title("Phễu Lọc CurationGate Qua 3 Giai Đoạn\n(6.545 Thô → 5.251 Đạt Chuẩn | Loại 1.294 Ảnh Lỗi)", fontsize=11, fontweight="bold", pad=10)
    axes[0].legend(loc="upper left", fontsize=9, frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: Horizontal bar of rejection reasons (R1 + R2 = 749 rejections / 794 reason hits)
    items = rej_counter.most_common()
    r_labels = [it[0] for it in reversed(items)]
    r_vals = [it[1] for it in reversed(items)]
    y_pos = np.arange(len(r_labels))
    axes[1].barh(y_pos, r_vals, color="#f97316", edgecolor="black", height=0.62, alpha=0.9)
    for y_i, val in zip(y_pos, r_vals):
        axes[1].text(val + 8, y_i, f"{val:,}", va="center", ha="left", fontsize=9, fontweight="bold")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(r_labels, fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Số Lượt Vi Phạm Tiêu Chí Lọc (AL R1 + R2)", fontsize=10, fontweight="bold")
    axes[1].set_xlim(0, max(r_vals) * 1.16)
    axes[1].set_title("Phân Rã Lý Do Loại Bỏ Trong Active Learning\n(Tổng hợp từ curation_report.json Vòng 1 & Vòng 2)", fontsize=11, fontweight="bold", pad=10)
    axes[1].grid(axis="x", linestyle="--", alpha=0.45)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("MINH BẠCH HÓA PHỄU LỌC CURATIONGATE: TỪ 6.545 ẢNH THÔ → 5.251 ẢNH SẠCH", fontsize=12.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_closed_loop_attribution_chart() -> None:
    """Generate Closed-Loop Error Attribution & Adaptive Budget shift chart from closed_loop_metrics.json."""
    out_path = Path("docs/reports/assets/benchmarks/closed_loop_attribution.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cl = json.loads(Path("experiments/logs/closed_loop_metrics.json").read_text(encoding="utf-8"))
    r1 = cl["rounds"][0]
    r2 = cl["rounds"][1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=200)

    # Left: Mode error drop R1 -> R2 + R2 adaptive quota
    modes = ["sd_inpaint", "sd_img2img", "sd_text2img"]
    mode_labels = ["sd_inpaint\n(R2 Quota: 897)", "sd_img2img\n(R2 Quota: 540)", "sd_text2img\n(R2 Quota: 463)"]
    r1_err = [r1["attribution"]["mode_errors"][m] * 100.0 for m in modes]
    r2_err = [r2["attribution"]["mode_errors"][m] * 100.0 for m in modes]

    x = np.arange(len(modes))
    w = 0.34
    b1 = axes[0].bar(x - w / 2, r1_err, w, label="Vòng 1 Error Rate (%)", color="#ef4444", edgecolor="black")
    b2 = axes[0].bar(x + w / 2, r2_err, w, label="Vòng 2 Error Rate (%)", color="#10b981", edgecolor="black")
    for b in list(b1) + list(b2):
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2, h + 1.4, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(mode_labels, fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Tỷ Lệ Đánh Lừa Detector (% ↓)", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 84)
    axes[0].set_title("Đóng Vòng Phản Hồi: AdaptiveBudgetController\n(Vòng 1 phát hiện Inpaint lỗi 71.70% → dồn 897 quota cho Vòng 2)", fontsize=10.5, fontweight="bold", pad=10)
    axes[0].legend(loc="upper right", fontsize=9, frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.45)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: R2 Region & Keyword Error Attribution
    reg_err = r2["attribution"].get("region_errors", {})
    kw_err = r2["attribution"].get("keyword_errors", {})
    combined = [
        ("Vùng: periocular (mắt)", reg_err.get("periocular", 0) * 100.0, "#7c3aed"),
        ("Ánh sáng: soft studio key light", kw_err.get("lighting:soft studio key light", 0) * 100.0, "#2563eb"),
        ("Nhóm tuổi: elderly", kw_err.get("demographics:elderly", 0) * 100.0, "#ea580c"),
        ("Ánh sáng: harsh camera flash", kw_err.get("lighting:direct harsh camera flash", 0) * 100.0, "#2563eb"),
        ("Ánh sáng: warm golden sunlight", kw_err.get("lighting:warm golden hour sunlight", 0) * 100.0, "#2563eb"),
        ("Vùng: lips (môi/miệng)", reg_err.get("lips", 0) * 100.0, "#7c3aed"),
        ("Ánh sáng: dramatic side lighting", kw_err.get("lighting:dramatic cinematic side lighting", 0) * 100.0, "#2563eb"),
    ]
    combined_rev = list(reversed(combined))
    y_p = np.arange(len(combined_rev))
    axes[1].barh(y_p, [c[1] for c in combined_rev], color=[c[2] for c in combined_rev], edgecolor="black", height=0.6, alpha=0.88)
    for y_i, (_, val, _) in enumerate(combined_rev):
        axes[1].text(val + 0.7, y_i, f"{val:.2f}%", va="center", ha="left", fontsize=9, fontweight="bold")
    axes[1].set_yticks(y_p)
    axes[1].set_yticklabels([c[0] for c in combined_rev], fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Error Rate Tại Vòng 2 (%)", fontsize=10, fontweight="bold")
    axes[1].set_xlim(0, 47)
    axes[1].set_title("Truy Vết Điểm Yếu Theo Vùng & Prompt (Vòng 2)\n(PromptGrammarEngine ưu tiên sinh vào các ca khó này)", fontsize=10.5, fontweight="bold", pad=10)
    axes[1].grid(axis="x", linestyle="--", alpha=0.45)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("CƠ CHẾ ĐÓNG VÒNG THÔNG MINH: ERROR ATTRIBUTION & ADAPTIVE BUDGETING", fontsize=12.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def build_vertical_report_html() -> str:
    html_content = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Báo Cáo Nghiệm Thu Khoa Học Đề Tài 11 | UTH Computer Vision (Nhóm 4)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #1e2bfa;
      --primary-hover: #1420d4;
      --primary-soft: rgba(30, 43, 250, 0.08);
      --primary-border: rgba(30, 43, 250, 0.25);
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-soft: #f1f5f9;
      --text: #0f172a;
      --text-muted: #475569;
      --text-light: #94a3b8;
      --border: #e2e8f0;
      --border-strong: #cbd5e1;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --shadow-sm: 0 2px 6px rgba(15, 23, 42, 0.04);
      --shadow-md: 0 10px 25px -5px rgba(15, 23, 42, 0.06), 0 8px 10px -6px rgba(15, 23, 42, 0.04);
      --shadow-lg: 0 20px 35px -10px rgba(15, 23, 42, 0.08);
      --font-display: 'Plus Jakarta Sans', -apple-system, sans-serif;
      --font-body: 'Inter', -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-body);
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
    }

    /* Fixed Top Navigation Bar */
    .top-header {
      position: sticky;
      top: 0;
      z-index: 1000;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 10px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: var(--shadow-sm);
    }
    .header-branding {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .toggle-sidebar-btn {
      background: var(--surface-soft);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 13px;
      padding: 6px 12px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .toggle-sidebar-btn:hover {
      background: #e2e8f0;
      color: var(--primary);
    }
    .brand-logo-badge {
      background: var(--primary);
      color: #ffffff;
      font-family: var(--font-mono);
      font-weight: 800;
      font-size: 12px;
      padding: 5px 8px;
      border-radius: 6px;
    }
    .brand-title {
      font-family: var(--font-display);
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }
    .brand-sub {
      font-size: 11px;
      color: var(--text-muted);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    /* Layout Container: Slim Sidebar ~200px vs Main Area */
    .layout-wrapper {
      width: 100%;
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px 24px 80px;
      display: grid;
      grid-template-columns: 200px minmax(0, 1fr);
      gap: 28px;
      transition: grid-template-columns 0.25s ease;
    }
    .layout-wrapper.sidebar-collapsed {
      grid-template-columns: 0px minmax(0, 1fr);
      gap: 0px;
    }
    .layout-wrapper.sidebar-collapsed .sidebar-nav {
      opacity: 0;
      pointer-events: none;
      visibility: hidden;
      width: 0;
      padding: 0;
    }

    /* Slim, Elegant Sticky Sidebar */
    .sidebar-nav {
      position: sticky;
      top: 70px;
      height: calc(100vh - 90px);
      overflow-y: auto;
      padding-right: 8px;
      transition: opacity 0.2s ease, width 0.25s ease;
    }
    .sidebar-nav::-webkit-scrollbar {
      width: 4px;
    }
    .sidebar-nav::-webkit-scrollbar-thumb {
      background: var(--border-strong);
      border-radius: 4px;
    }
    .nav-group-title {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 700;
      color: var(--text-light);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 14px 0 6px;
    }
    .nav-group-title:first-child {
      margin-top: 0;
    }
    .nav-link {
      display: block;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-decoration: none;
      transition: all 0.15s ease;
      margin-bottom: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .nav-link:hover {
      background: var(--surface-soft);
      color: var(--primary);
    }
    .nav-link.active {
      background: var(--primary-soft);
      color: var(--primary);
      font-weight: 700;
      border-left: 3px solid var(--primary);
    }

    /* Content Area */
    .main-content {
      display: flex;
      flex-direction: column;
      gap: 40px;
      min-width: 0;
    }

    /* Section Containers */
    .report-section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 36px 40px;
      box-shadow: var(--shadow-sm);
    }
    .report-section.hero-section {
      background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
      border: 1px solid var(--primary-border);
    }

    .section-eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 700;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }
    .section-title {
      font-family: var(--font-display);
      font-size: 26px;
      font-weight: 800;
      color: var(--text);
      line-height: 1.3;
      margin-bottom: 12px;
    }
    .section-title span.accent {
      color: var(--primary);
    }
    .section-desc {
      font-size: 15px;
      color: var(--text-muted);
      max-width: 1200px;
      line-height: 1.7;
      margin-bottom: 20px;
    }

    /* KPI Highlights */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin: 20px 0 28px;
    }
    @media (max-width: 1200px) {
      .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    .kpi-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 18px 20px;
      box-shadow: var(--shadow-sm);
      position: relative;
      overflow: hidden;
    }
    .kpi-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: var(--primary);
    }
    .kpi-card.champion::before {
      background: #10b981;
    }
    .kpi-card.alert::before {
      background: #ef4444;
    }
    .kpi-label {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      color: var(--text-light);
      text-transform: uppercase;
    }
    .kpi-value {
      font-family: var(--font-display);
      font-size: 32px;
      font-weight: 800;
      color: var(--text);
      margin: 6px 0 4px;
    }
    .kpi-card.champion .kpi-value {
      color: #059669;
    }
    .kpi-card.alert .kpi-value {
      color: #dc2626;
    }
    .kpi-note {
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 500;
    }

    /* 1 ARTIFACT TRÊN 1 HÀNG (FULL-WIDTH, HIGH-RESOLUTION) */
    .artifact-row-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      overflow: hidden;
      margin: 26px 0;
      box-shadow: var(--shadow-md);
      width: 100%;
    }
    .artifact-row-header {
      background: var(--surface-soft);
      padding: 14px 22px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .artifact-row-title {
      font-family: var(--font-display);
      font-size: 16px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .artifact-badge {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      padding: 3px 9px;
      border-radius: 6px;
      background: #ffffff;
      border: 1px solid var(--border);
      color: var(--primary);
    }
    .artifact-badge.champion {
      background: #ecfdf5;
      color: #059669;
      border-color: #a7f3d0;
    }
    .artifact-badge.alert {
      background: #fef2f2;
      color: #b91c1c;
      border-color: #fecaca;
    }
    .artifact-row-body {
      padding: 24px;
      background: #ffffff;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .artifact-img-full {
      width: 100%;
      height: auto;
      max-height: 850px;
      object-fit: contain;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: #ffffff;
      cursor: zoom-in;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .artifact-img-full:hover {
      box-shadow: 0 10px 30px rgba(30, 43, 250, 0.15);
      border-color: var(--primary);
    }
    .zoom-hint-banner {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--primary);
      background: var(--primary-soft);
      padding: 5px 12px;
      border-radius: 20px;
      margin-top: 10px;
      cursor: pointer;
    }
    .artifact-row-footer {
      background: var(--surface-soft);
      padding: 14px 22px;
      border-top: 1px solid var(--border);
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
    }

    /* Face Comparison Subject Box (1 Khung Lớn / 1 Hàng) */
    .face-subject-container {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 20px;
      margin: 20px 0;
      box-shadow: var(--shadow-sm);
    }
    .face-subject-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }
    .face-subject-title {
      font-family: var(--font-display);
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }
    .face-strip-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }
    .face-strip-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      overflow: hidden;
      text-align: center;
    }
    .face-strip-card img {
      width: 100%;
      height: 260px;
      object-fit: cover;
      display: block;
      cursor: zoom-in;
      transition: transform 0.15s ease;
    }
    .face-strip-card img:hover {
      transform: scale(1.02);
    }
    .face-strip-info {
      padding: 10px 8px;
    }
    .face-tag {
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 4px;
      margin-bottom: 4px;
    }
    .face-tag.real { background: #ecfdf5; color: #047857; }
    .face-tag.inpaint { background: #eff6ff; color: #1d4ed8; }
    .face-tag.img2img { background: #faf5ff; color: #7e22ce; }
    .face-tag.text2img { background: #fff7ed; color: #c2410c; }
    .face-name {
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
    }

    /* Tables */
    .data-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
      border-radius: var(--radius-md);
      overflow: hidden;
      border: 1px solid var(--border);
      margin: 20px 0;
    }
    .data-table th {
      background: var(--surface-soft);
      padding: 12px 18px;
      text-align: left;
      font-weight: 700;
      color: var(--text);
      border-bottom: 2px solid var(--border-strong);
      font-family: var(--font-display);
    }
    .data-table td {
      padding: 11px 18px;
      border-bottom: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
    }
    .data-table tr:last-child td {
      border-bottom: none;
    }
    .data-table tr.highlight-champion td {
      background: rgba(30, 43, 250, 0.05);
      font-weight: 600;
    }
    .data-table tr.highlight-champion td:first-child {
      border-left: 3px solid var(--primary);
    }

    .badge-pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px 9px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 11px;
      font-family: var(--font-mono);
    }
    .badge-pill.green { background: #dcfce7; color: #15803d; }
    .badge-pill.red { background: #fee2e2; color: #b91c1c; }
    .badge-pill.amber { background: #fef3c7; color: #b45309; }

    /* Callouts */
    .callout-box {
      border-radius: var(--radius-md);
      padding: 18px 24px;
      margin: 20px 0;
      border: 1px solid var(--border);
      background: var(--surface-soft);
    }
    .callout-box.info { background: #f0f7ff; border-color: #bfdbfe; }
    .callout-box.success { background: #f0fdf4; border-color: #bbf7d0; }
    .callout-box.danger { background: #fef2f2; border-color: #fecaca; }
    .callout-title {
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 14px;
      margin-bottom: 6px;
    }
    .callout-box.info .callout-title { color: #1e40af; }
    .callout-box.success .callout-title { color: #166534; }
    .callout-box.danger .callout-title { color: #991b1b; }
    .callout-desc {
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
    }

    /* Buttons */
    .btn-action {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 16px;
      border-radius: var(--radius-sm);
      font-weight: 700;
      font-size: 13px;
      text-decoration: none;
      transition: all 0.15s ease;
      cursor: pointer;
    }
    .btn-action.primary {
      background: var(--primary);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(30, 43, 250, 0.25);
    }
    .btn-action.primary:hover {
      background: var(--primary-hover);
      transform: translateY(-1px);
    }
    .btn-outline {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--surface-soft);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 12px;
      padding: 6px 11px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .btn-outline:hover {
      background: #e2e8f0;
      color: var(--primary);
    }
    .table-responsive {
      width: 100%;
      overflow-x: auto;
      margin: 20px 0 28px;
    }

    /* Interactive End-to-End Pipeline Flow Map */
    .pipeline-flow-box {
      background: #ffffff;
      border: 1px solid var(--primary-border);
      border-radius: var(--radius-md);
      padding: 20px 22px;
      margin: 22px 0 26px;
      box-shadow: var(--shadow-sm);
    }
    .pipeline-flow-heading {
      font-family: var(--font-display);
      font-size: 14px;
      font-weight: 800;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pipeline-flow-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
    }
    .flow-step-card {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      background: var(--surface-soft);
      border: 1px solid var(--border);
      border-top: 4px solid var(--primary);
      border-radius: var(--radius-sm);
      padding: 12px 14px;
      text-decoration: none;
      color: var(--text);
      transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .flow-step-card:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
      border-color: var(--primary);
      background: #ffffff;
    }
    .flow-step-card.alert { border-top-color: #ef4444; }
    .flow-step-card.champion { border-top-color: #10b981; background: #f0fdf4; }
    .flow-step-num {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 4px;
    }
    .flow-step-card.alert .flow-step-num { color: #dc2626; }
    .flow-step-card.champion .flow-step-num { color: #059669; }
    .flow-step-title {
      font-family: var(--font-display);
      font-size: 13.5px;
      font-weight: 800;
      margin-bottom: 6px;
      line-height: 1.3;
    }
    .flow-step-desc {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.45;
      margin-bottom: 8px;
    }
    .flow-step-kpi {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      color: var(--text);
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 3px 7px;
      display: inline-block;
      align-self: flex-start;
    }
    @media (max-width: 1100px) {
      .pipeline-flow-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    /* =======================================================================
       LIGHTBOX MODAL (BẤM ẢNH ĐỂ PHÓNG TO TOÀN MÀN HÌNH)
       ======================================================================= */
    .lightbox-modal {
      display: none;
      position: fixed;
      z-index: 9999;
      inset: 0;
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(10px);
      align-items: center;
      justify-content: center;
      flex-direction: column;
      padding: 24px;
      cursor: zoom-out;
    }
    .lightbox-modal.active {
      display: flex;
    }
    .lightbox-content-box {
      max-width: 96vw;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      position: relative;
    }
    .lightbox-img {
      max-width: 95vw;
      max-height: 85vh;
      object-fit: contain;
      border-radius: 8px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      background: #ffffff;
      animation: modalZoom 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes modalZoom {
      from { transform: scale(0.92); opacity: 0; }
      to { transform: scale(1); opacity: 1; }
    }
    .lightbox-caption {
      margin-top: 12px;
      color: #ffffff;
      font-family: var(--font-display);
      font-size: 14px;
      font-weight: 600;
      text-align: center;
      background: rgba(0, 0, 0, 0.6);
      padding: 6px 16px;
      border-radius: 20px;
    }
    .lightbox-close {
      position: absolute;
      top: 20px;
      right: 32px;
      color: #ffffff;
      font-size: 38px;
      font-weight: 700;
      cursor: pointer;
      line-height: 1;
      transition: color 0.15s ease;
      z-index: 10001;
    }
    .lightbox-close:hover {
      color: #ef4444;
    }

    @media (max-width: 900px) {
      .layout-wrapper {
        grid-template-columns: 1fr;
      }
      .sidebar-nav {
        display: none;
      }
      .kpi-grid {
        grid-template-columns: 1fr 1fr;
      }
      .face-strip-row {
        grid-template-columns: 1fr 1fr;
      }
    }

    @media print {
      .top-header, .sidebar-nav, .zoom-hint-banner, .lightbox-modal {
        display: none !important;
      }
      .layout-wrapper {
        display: block;
        padding: 0;
      }
      .report-section {
        box-shadow: none;
        border: 1px solid #cbd5e1;
        margin-bottom: 20px;
        break-inside: avoid-page;
      }
      .artifact-row-card, .face-subject-container {
        box-shadow: none;
        break-inside: avoid;
      }
    }
  </style>
</head>
<body>

  <!-- Top Sticky Header -->
  <header class="top-header">
    <div class="header-branding">
      <button class="toggle-sidebar-btn" id="toggle-sidebar-btn" title="Thu gọn / Mở rộng Mục Lục bên trái">
        ✕ Ẩn Mục Lục
      </button>
      <span class="brand-logo-badge">UTH CV</span>
      <div>
        <div class="brand-title">Đề Tài 11 • Báo Cáo Nghiệm Thu Khoa Học Chung Cuộc</div>
        <div class="brand-sub">Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection • Nhóm 4</div>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn-outline" id="prev-sec-btn" title="Mục trước (Alt + ↑)">▲ Trước</button>
      <button class="btn-outline" id="next-sec-btn" title="Mục tiếp theo (Alt + ↓)">▼ Tiếp</button>
      <button class="btn-outline" id="print-report-btn" title="In hoặc Xuất PDF báo cáo">🖨️ In / PDF</button>
    </div>
  </header>

  <div class="layout-wrapper" id="layout-wrapper">
    <!-- Slim, Compact Sidebar Navigation (~200px) -->
    <nav class="sidebar-nav" id="sidebar-nav">
      <div class="nav-group-title">Mục Lục Báo Cáo</div>
      <a href="#sec-overview" class="nav-link active">1. Tổng Quan Nghiệm Thu</a>
      <a href="#sec-problem" class="nav-link">2. Thách Thức & Sự Cố M0</a>
      <a href="#sec-dataset" class="nav-link">3. Bộ Dữ Liệu 12.545 Ảnh</a>
      <a href="#sec-curation" class="nav-link">4. Quy Trình Lọc CurationGate</a>
      <a href="#sec-hard-samples" class="nav-link">5. 740 Mẫu Khó & 213 Vault</a>
      <a href="#sec-active-learning" class="nav-link">6. Đột Phá M2* (+80.18%)</a>
      <a href="#sec-zero-leakage" class="nav-link">7. Profiling 6k &amp; Zero Leakage</a>
      <a href="#sec-champion-m3" class="nav-link">8. Champion M3 (93.13%)</a>
      <a href="#sec-universal" class="nav-link">9. Universal 20.4k Benchmark</a>
      <a href="#sec-xai" class="nav-link">10. Trực Quan Hóa Attention Map</a>
      <a href="#sec-conclusion" class="nav-link">11. Bảng KPI & Kết Luận</a>
    </nav>

    <!-- Main Content Area -->
    <main class="main-content">

      <!-- ===================================================================
           PHẦN 1: TỔNG QUAN NGHIỆM THU & HERO ARTIFACT
           =================================================================== -->
      <section id="sec-overview" class="report-section hero-section">
        <span class="section-eyebrow">01 / TỔNG QUAN NGHIỆM THU • TOPIC 11</span>
        <h1 class="section-title">
          Đề Tài 11: <span class="accent">Tạo Bộ Dữ Liệu Khuôn Mặt Giả Bằng Stable Diffusion Phục Vụ Phát Hiện Deepfake</span>
          <div style="font-size: 15px; font-weight: 600; color: var(--text-muted); margin-top: 6px; font-family: var(--font-mono);">Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection</div>
        </h1>
        <p class="section-desc">
          Đồ án thực hiện trọn vẹn vòng đời dữ liệu-lấy-làm-trung-tâm (<em>Dataset-First</em>) theo đúng 3 bước cốt lõi của giảng viên và mở rộng lên thích nghi đa động cơ: từ <strong>4.000 ảnh thật</strong> (FFHQ + CelebA-HQ) → sinh <strong>12.545 ảnh giả</strong> từ 2 mô hình (SD 1.5: 6.545 thô / <strong>5.251 đạt chuẩn CurationGate</strong>; RealisticVision v5.1: 6.000 ảnh) → khai thác <strong>740 mẫu khó SD 1.5</strong> → nâng Fake Recall trên 213 Challenge Vault từ <strong>11.71% → 91.89%</strong> (<code>M₂*</code>) → dùng <code>M₂*</code> profiling 6.000 RV5.1 (phát hiện <strong>1.792 ca bỏ lọt</strong> chỉ để quan sát) và cách ly danh tính sạch để huấn luyện <strong>Champion M₃</strong> đạt <strong>93.13%</strong> trên Unseen RV5.1.
        </p>

        <!-- Interactive 5-Stage End-to-End Visual Flow -->
        <div class="pipeline-flow-box">
          <div class="pipeline-flow-heading">
            <span>🧭 Sơ Đồ Luồng Thực Nghiệm Toàn Trình (End-to-End Pipeline Flow)</span>
            <span style="font-size: 11px; color: var(--text-muted); font-weight: 600;">4.000 Real + 12.545 Fake → D₀ → D₁ → D₂ → D₃ → M₃</span>
          </div>
          <div class="pipeline-flow-grid">
            <a href="#sec-dataset" class="flow-step-card">
              <div>
                <div class="flow-step-num">BƯỚC 1 • SINH &amp; LỌC</div>
                <div class="flow-step-title">Dual-Engine &amp; CurationGate</div>
                <div class="flow-step-desc">4.000 ảnh thật → 6.545 SD 1.5 thô (lọc còn 5.251 sạch) + 6.000 RV5.1</div>
              </div>
              <span class="flow-step-kpi">12.545 Fake</span>
            </a>
            <a href="#sec-problem" class="flow-step-card alert">
              <div>
                <div class="flow-step-num">BƯỚC 2 • BASELINE M₀</div>
                <div class="flow-step-title">Điểm Mù Làm Mờ Biên</div>
                <div class="flow-step-desc">Huấn luyện M₀ trên D₀ (4.224 ảnh) → sụp đổ trước ảnh Soft-Mask</div>
              </div>
              <span class="flow-step-kpi">11.71% Recall</span>
            </a>
            <a href="#sec-active-learning" class="flow-step-card">
              <div>
                <div class="flow-step-num">BƯỚC 3 • ACTIVE LEARNING</div>
                <div class="flow-step-title">740 Mẫu Khó &amp; M₂*</div>
                <div class="flow-step-desc">2 vòng AL (553+187) → 629 train D₂ + 111 vào 213 Challenge Vault</div>
              </div>
              <span class="flow-step-kpi">91.89% (+80.18%)</span>
            </a>
            <a href="#sec-zero-leakage" class="flow-step-card">
              <div>
                <div class="flow-step-num">BƯỚC 4 • PROFILING &amp; D₃</div>
                <div class="flow-step-title">M₂* Mining 6k &amp; Zero Leakage</div>
                <div class="flow-step-desc">M₂* chạy 6k RV5.1: 1.792 bypass (chỉ quan sát) + cách ly 626 test &amp; 600 val</div>
              </div>
              <span class="flow-step-kpi">1.792 OOD | D₃: 5.626</span>
            </a>
            <a href="#sec-champion-m3" class="flow-step-card champion">
              <div>
                <div class="flow-step-num">BƯỚC 5 • CHAMPION M₃</div>
                <div class="flow-step-title">M₃ + Universal + XAI</div>
                <div class="flow-step-desc">Warm-Start M₂*→M₃, kiểm định Quad-Vault, 20.4k Zero-Shot &amp; Heatmap</div>
              </div>
              <span class="flow-step-kpi">93.13% RV5.1</span>
            </a>
          </div>
        </div>

        <!-- KPI Scorecard Highlights -->
        <div class="kpi-grid">
          <div class="kpi-card">
            <span class="kpi-label">Tổng Dữ Liệu Sinh Mới</span>
            <div class="kpi-value">12.545</div>
            <span class="kpi-note">SD 1.5 (6.545 thô / 5.251 lọc) + RV5.1 (6.000)</span>
          </div>
          <div class="kpi-card alert">
            <span class="kpi-label">Mẫu Khó Nội Miền (SD 1.5)</span>
            <div class="kpi-value">740</div>
            <span class="kpi-note">629 Train (D₂/D₃) + 111 trong 213 Challenge Vault</span>
          </div>
          <div class="kpi-card alert">
            <span class="kpi-label">Điểm Mù OOD M₂* (RV5.1 6k)</span>
            <div class="kpi-value">1.792</div>
            <span class="kpi-note">29.87% bị M₂* bỏ lọt (+ 2.611 phân vân, chỉ quan sát)</span>
          </div>
          <div class="kpi-card champion">
            <span class="kpi-label">Tăng Fake Recall M2*</span>
            <div class="kpi-value">+80.18%</div>
            <span class="kpi-note">Nhảy vọt từ 11.71% → 91.89% (Challenge Vault)</span>
          </div>
          <div class="kpi-card champion">
            <span class="kpi-label">Fake Recall Mô Hình M3</span>
            <div class="kpi-value">93.13%</div>
            <span class="kpi-note">Trên Unseen RV5.1 Vault (1.215 ảnh, AUC 0.9828)</span>
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG KHỔ LỚN -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">🏆 Bảng Tổng Hợp Kết Quả: Tiến Trình Nâng Cấp 4 Phiên Bản Mô Hình (M0 → M1 → M2* → M3)</span>
            <span class="artifact-badge champion">GOLDEN SCORECARD</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/00_master_deepfake_defense_showcase_cell4_img0.png" class="artifact-img-full" alt="Golden Scorecard M0 to M3 Evolution" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Phân tích kết quả:</strong> Mô hình cơ sở M0 bỏ sót tới 88.29% ảnh giả trên Challenge Vault (Recall chỉ 11.71%); sau Vòng 1 (M1) tăng lên 70.27%; Vòng 2 (M2*) nâng lên 91.89%; và mô hình Champion M3 đạt đỉnh cao 93.13% trên tập kiểm thử Unseen RV5.1 mà vẫn giữ vững 91.89% trên Challenge Vault và 69.44% trên Master Frozen Test (SD 1.5).
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 2: THỰC TRẠNG & SỰ CỐ SỤP ĐỔ CỦA M0 BASELINE
           =================================================================== -->
      <section id="sec-problem" class="report-section">
        <span class="section-eyebrow">02 / THÁCH THỨC BÀI TOÁN</span>
        <h2 class="section-title">
          Thực Trạng: <span class="accent">Vì Sao Ảnh Giả Làm Mờ Biên Có Thể Đánh Lừa Các Bộ Dò Truyền Thống?</span>
        </h2>
        <p class="section-desc">
          Các mô hình deepfake ghép mặt truyền thống (Face-Swap) thường để lại đường cắt ghép sắc nhọn (Hard Mask Boundary), khiến các mạng CNN dễ dàng dựa vào vết cắt này để phân loại. Trong đề tài này, pipeline sinh ảnh áp dụng kỹ thuật <strong>Làm Mịn Biên Gaussian (Soft-Mask Feathering)</strong> trên mặt nạ elip giải phẫu (<code>blur_radius = 18–20</code>, tương đương <code>σ = 9.0–10.0</code> trong <code>stable_diffusion_engine.py</code>; hình mô phỏng minh họa dùng <code>σ = 4.0</code>), hòa trộn mượt vùng sinh mới vào khuôn mặt gốc và triệt tiêu đường biên nhân tạo.
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: SOFT FEATHERING -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Hình Ảnh Thực Nghiệm: Cơ Chế Làm Mịn Biên Gaussian (Soft-Mask Feathering)</span>
            <span class="artifact-badge">NOTEBOOK 01 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/01_diffusion_dataset_and_curation_cell8_img2.png" class="artifact-img-full" alt="Soft-Mask Feathering Mechanism" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Nguyên lý kỹ thuật:</strong> Mặt nạ được làm mờ Gaussian tạo dải chuyển tiếp êm dịu giữa vùng khuôn mặt sinh mới và ảnh gốc, triệt tiêu đỉnh gradient sắc nhọn ở viền mặt nạ và buộc bộ dò DINOv2 phải học các vi vết sinh học bên trong khuôn mặt.
          </div>
        </div>

        <!-- Empirical Failure Callout -->
        <div class="callout-box danger">
          <div class="callout-title">🚨 Thử Thách Lớn: Mô Hình Cơ Sở (M0) Bị Đánh Lừa Trên Tập Mẫu Khó Challenge Vault &amp; Legacy 6k</div>
          <div class="callout-desc">
            Khi thử nghiệm mô hình cơ sở M0 trên 213 mẫu của Challenge Vault (111 ảnh giả khó + 102 ảnh thật), <strong>Fake Recall chỉ đạt 11.71%</strong> (Bypass Rate lên tới <strong>88.29%</strong>; Overall Accuracy chỉ đạt <strong>53.99%</strong> — tương đương đoán ngẫu nhiên; F1-Score sụp đổ còn <strong>0.2097</strong>). Ngoài ra, khi kiểm thử trên 6.000 ảnh RV5.1 ở độ phân giải chuẩn hóa 224×224 (stress-test), M0 chỉ bắt được <strong>2.00%</strong> (bỏ lọt <strong>98.00%</strong>), còn ở độ phân giải 512×512 trên Unseen RV5.1 cũng vẫn bỏ lọt <strong>33.55%</strong>.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: M0 FAILURE COMPARISON -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Biểu Đồ Thực Nghiệm: Tỷ Lệ Bắt Giả vs Bỏ Lọt Của M₀ Baseline Trên 4 Kịch Bản Kiểm Thử</span>
            <span class="artifact-badge alert">M0 COLLAPSE ANALYSIS</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/m0_baseline_collapse_comparison.png" class="artifact-img-full" alt="M0 Baseline Collapse Comparison Chart" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Nguồn số liệu thực nghiệm:</strong> Tổng hợp trực tiếp từ <code>benchmark_5models_quad_vault_summary.json</code> (512×512) và <code>legacy_6k_benchmark_results.json</code> (stress-test 224×224). Khoảng trống bỏ lọt lên đến 88.29% (Challenge Vault) và 33.55%–98.00% (RV5.1) là động lực trực tiếp cho Active Learning và Champion M3.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 3: BỘ DỮ LIỆU KHUẾCH TÁN KÉP 12.545 ẢNH & MẪU THỰC TẾ
           =================================================================== -->
      <section id="sec-dataset" class="report-section">
        <span class="section-eyebrow">03 / XÂY DỰNG DỮ LIỆU</span>
        <h2 class="section-title">
          Kiến Trúc Dữ Liệu: <span class="accent">4.000 Ảnh Thật &amp; 12.545 Ảnh Sinh Từ Dual-Engine Diffusion</span>
        </h2>
        <p class="section-desc">
          Từ kho nền tảng <strong>4.000 ảnh thật</strong> (2.000 FFHQ + 2.000 CelebA-HQ, chia độc lập danh tính thành 2.813 train / 598 val / 589 test), nhóm đã sinh tổng cộng <strong>12.545 ảnh giả</strong> qua Dual-Engine Diffusion (<strong>SD 1.5: 6.545 thô → 5.251 đạt chuẩn CurationGate</strong>; <strong>RealisticVision v5.1: 6.000 ảnh</strong>) trên cả 3 chế độ Inpainting, Img2Img và Text-to-Image.
        </p>

        <!-- BẢNG TỔNG HỢP PHÂN BỔ 2 ĐỘNG CƠ SINH DỮ LIỆU (DUAL-ENGINE) -->
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th style="width: 24%;">Nguồn Dữ Liệu</th>
                <th style="width: 28%;">Checkpoint / Kho Gốc</th>
                <th style="width: 20%;">Quy Mô Thực Tế</th>
                <th style="width: 28%;">Vai Trò Trong Pipeline</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Ảnh Thật Gốc (Real)</strong></td>
                <td>FFHQ (2.000) + CelebA-HQ (2.000)</td>
                <td><strong>4.000 ảnh</strong></td>
                <td>Chia identity-disjoint: 2.813 train / 598 val / 589 test</td>
              </tr>
              <tr>
                <td><strong>Bộ A: Stable Diffusion v1.5</strong></td>
                <td><code>runwayml/stable-diffusion-v1-5</code></td>
                <td><strong>6.545 thô</strong> (<a href="#sec-curation" style="color: var(--primary); font-weight: 700;">5.251 sạch</a>)</td>
                <td>Khởi tạo D₀ (2.200 sạch) + AL 2 vòng (1.484 R1 + 1.567 R2 sạch) — <a href="#sec-curation" style="color: var(--primary); font-weight: 600;">xem phễu lọc ở Mục 4</a></td>
              </tr>
              <tr>
                <td><strong>Bộ B: RealisticVision v5.1</strong></td>
                <td><code>SG161222/Realistic_Vision_V5.1_noVAE</code></td>
                <td><strong>6.000 ảnh</strong> (2k×3 chế độ)</td>
                <td>OOD Profiling (M₀–M₂*) &amp; trích 1.384 sạch vào D₃ (cách ly 626 test + 600 val)</td>
              </tr>
              <tr class="highlight-champion">
                <td><strong>Tổng cộng ảnh giả sinh mới</strong></td>
                <td>Dual-Engine Diffusion</td>
                <td><strong style="color: var(--primary); font-size: 15px;">12.545 ảnh giả</strong></td>
                <td>Tổng kho toàn dự án: <strong>16.545 ảnh</strong> (4.000 Real + 12.545 Fake thô / 11.251 Fake sau lọc; riêng nhánh SD 1.5 đạt chuẩn <strong>10.000 ảnh</strong> trong <code>clean_10k_production_report.json</code>)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- BẢNG TIẾN HÓA CÁC PHÂN VÙNG D0 -> D1 -> D2 -> D3 -->
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>Phiên Bản Tập Dữ Liệu</th>
                <th>Train (Ảnh Thật + Giả)</th>
                <th>Validation</th>
                <th>Tập Kiểm Thử Độc Lập</th>
                <th>Mô Hình Huấn Luyện</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>D₀ (Baseline)</strong></td>
                <td><strong>4.224</strong> (2.813 Real + 1.411 SD 1.5)</td>
                <td>798 (598 Real + 200 SD 1.5)</td>
                <td>Master Frozen Test: <strong>1.178</strong> (589+589)</td>
                <td><code>M₀</code> (baseline_d0_best.pt)</td>
              </tr>
              <tr>
                <td><strong>D₁ (Active R1)</strong></td>
                <td><strong>4.694</strong> (D₀ + 470 Hard R1)</td>
                <td>798 (như D₀)</td>
                <td>Master (1.178) + Challenge R1 (83 Fake)</td>
                <td><code>M₁</code> (active_learning_m1_best.pt)</td>
              </tr>
              <tr>
                <td><strong>D₂ (Active R2)</strong></td>
                <td><strong>4.853</strong> (D₁ + 159 Hard R2)</td>
                <td>798 (như D₀)</td>
                <td>Master (1.178) + Challenge Vault (<strong>213</strong>)</td>
                <td><code>M₂</code> (Cold) &amp; <code>M₂*</code> (Warm-Start)</td>
              </tr>
              <tr class="highlight-champion">
                <td><strong>D₃ (Dual-Engine Sạch)</strong></td>
                <td><strong>5.626</strong> (2.813 Real + 629 Hard + 800 SD + 1.384 RV5.1)</td>
                <td>1.198 (598 Real + 600 RV5.1 Val)</td>
                <td>Unseen RV5.1 Vault: <strong>1.215</strong> (589 Real + 626 RV5.1 Test)</td>
                <td><code>M₃</code> (active_learning_m3_best.pt)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: DUAL ENGINE PORTFOLIO GRID -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">So Sánh Trực Quan: Ảnh Thật vs 3 Chế Độ Sinh Ảnh (Inpaint, Img2Img, Text2Img)</span>
            <span class="artifact-badge champion">NOTEBOOK 01 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/01_diffusion_dataset_and_curation_cell10_img3.png" class="artifact-img-full" alt="Dual Engine Portfolio Grid" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Tiêu chuẩn kỹ thuật:</strong> Ảnh được sinh và chuẩn hóa ở độ phân giải <code>512×512</code> (pipeline huấn luyện/đánh giá chính của DINOv2 tự động nội suy về <code>504×504</code> bội số patch 14; riêng các bài stress-test legacy 6k và universal benchmark chạy ở <code>224×224</code>), bảo toàn ánh sáng tự nhiên và tỷ lệ nhân khẩu học hài hòa.
          </div>
        </div>

        <h3 style="font-family: var(--font-display); font-size: 18px; margin: 32px 0 16px;">Một Số Mẫu Khuôn Mặt Điển Hình Trong Bộ Dữ Liệu:</h3>

        <!-- Mẫu 1: Nhân vật Nữ #0 -->
        <div class="face-subject-container">
          <div class="face-subject-header">
            <span class="face-subject-title">Mẫu 1: Đối Chứng Khuôn Mặt Nữ (CelebA-HQ #0) Qua 4 Chế Độ Sinh</span>
            <span class="artifact-badge">CHỦ THỂ NỮ</span>
          </div>
          <div class="face-strip-row">
            <div class="face-strip-card">
              <img src="assets/samples/real_0.jpg" alt="Real Face 0" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag real">ẢNH THẬT (GỐC)</span>
                <div class="face-name">CelebA-HQ ID #0</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/inpaint_0.png" alt="Inpaint 0" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag inpaint">SD INPAINT</span>
                <div class="face-name">0_inpaint.png (Soft Biên)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/img2img_0.png" alt="Img2Img 0" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag img2img">SD IMG2IMG</span>
                <div class="face-name">0_img2img.png (Biến Đổi)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/text2img_0.png" alt="Text2Img 0" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag text2img">SD TEXT2IMG</span>
                <div class="face-name">0_text2img.png (Sinh Hoàn Toàn)</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Mẫu 2: Nhân vật Nam #9 -->
        <div class="face-subject-container">
          <div class="face-subject-header">
            <span class="face-subject-title">Mẫu 2: Đối Chứng Khuôn Mặt Nam (CelebA-HQ #9) Khớp Râu & Cằm Tự Nhiên</span>
            <span class="artifact-badge">CHỦ THỂ NAM</span>
          </div>
          <div class="face-strip-row">
            <div class="face-strip-card">
              <img src="assets/samples/real_9.jpg" alt="Real Face 9" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag real">ẢNH THẬT (GỐC)</span>
                <div class="face-name">CelebA-HQ ID #9</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/inpaint_9.png" alt="Inpaint 9" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag inpaint">SD INPAINT</span>
                <div class="face-name">9_inpaint.png (Khớp Râu/Cằm)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/img2img_9.png" alt="Img2Img 9" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag img2img">SD IMG2IMG</span>
                <div class="face-name">9_img2img.png (Biến Đổi Da)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/text2img_9.png" alt="Text2Img 9" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag text2img">SD TEXT2IMG</span>
                <div class="face-name">9_text2img.png (Sinh Hoàn Toàn)</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Mẫu 3: Nhân vật Thiếu Sáng / Ngược Sáng #107 -->
        <div class="face-subject-container">
          <div class="face-subject-header">
            <span class="face-subject-title">Mẫu 3: Đối Chứng Điều Kiện Ánh Sáng Yếu & Tương Phản Cao (CelebA-HQ #107)</span>
            <span class="artifact-badge alert">ÁNH SÁNG PHỨC TẠP</span>
          </div>
          <div class="face-strip-row">
            <div class="face-strip-card">
              <img src="assets/samples/real_107.jpg" alt="Real Face 107" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag real">ẢNH THẬT (GỐC)</span>
                <div class="face-name">CelebA-HQ #107 (Ngược Sáng)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/inpaint_107.png" alt="Inpaint 107" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag inpaint">SD INPAINT</span>
                <div class="face-name">107_inpaint.png (Ánh Sáng Yếu)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/img2img_107.png" alt="Img2Img 107" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag img2img">SD IMG2IMG</span>
                <div class="face-name">107_img2img.png (Tương Phản)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/text2img_107.png" alt="Text2Img 107" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag text2img">SD TEXT2IMG</span>
                <div class="face-name">107_text2img.png (Studio Lighting)</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: DEMOGRAPHICS PANEL -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Thống Kê Độ Đa Dạng Dữ Liệu: Độ Tuổi, Giới Tính Và Đặc Điểm Khuôn Mặt</span>
            <span class="artifact-badge">EDA FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/eda/02_demographics_panel.png" class="artifact-img-full" alt="Demographics Panel" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Đảm bảo tính đa dạng:</strong> Dữ liệu phân bổ đều các nhóm tuổi từ thanh niên đến người lớn tuổi, tỷ lệ nam/nữ cân bằng, và góc nghiêng khuôn mặt nằm trong khoảng tự nhiên (±30°).
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 4: QUY TRÌNH LỌC DỮ LIỆU TỰ ĐỘNG (CURATIONGATE)
           =================================================================== -->
      <section id="sec-curation" class="report-section">
        <span class="section-eyebrow">04 / KIỂM SOÁT CHẤT LƯỢNG</span>
        <h2 class="section-title">
          CurationGate: <span class="accent">Phễu Lọc Tự Động 3 Tầng (6.545 Thô → 5.251 Ảnh SD 1.5 Đạt Chuẩn)</span>
        </h2>
        <p class="section-desc">
          Triển khai thực tế trong <code>ImageCurationGate</code> (<code>src/pipelines/curate_image_dataset.py</code> &amp; <code>configs/dataset_factory.yaml</code>), mỗi ảnh sinh mới phải vượt qua 3 tầng kiểm duyệt tự động trước khi vào tập huấn luyện:
        </p>

        <div class="callout-box info">
          <div class="callout-title">⚙️ Chi Tiết Kỹ Thuật Của 3 Tầng Lọc CurationGate Trong Mã Nguồn</div>
          <div class="callout-desc">
            1. <strong>Tầng 1 — Độ nét quang học (Laplacian Variance Calibrated):</strong> Tính biến sai toán tử Laplacian trên tensor <code>[0, 1]</code> nhân hệ số 1000 (<code>compute_laplacian_sharpness</code>); ngưỡng loại bỏ ảnh mờ được hiệu chỉnh động tại <strong>phân vị 5% (<code>q = 0.05</code>)</strong> của ảnh thật gốc (mặc định ≥ 1.25).<br>
            2. <strong>Tầng 2 — Hình học &amp; Giải phẫu khuôn mặt (<code>opencv_haar</code> + <code>verify_facial_geometry</code>):</strong> Phát hiện khuôn mặt bằng <strong>OpenCV Haar Cascade</strong> (loại bỏ ảnh không thấy mặt hoặc có ≥ 2 khuôn mặt); kiểm tra tỷ lệ khung mặt <code>0.65 ≤ h/w ≤ 1.6</code>, diện tích mặt chiếm <code>20%–65%</code> khung hình, không bị cắt cụt trán/cằm (<code>skin ≤ 35%</code> tại mép ảnh), không cháy sáng ROI mặt (<code>&gt; 8.0%</code>), không lệch tâm (<code>0.15–0.85</code>), và không có đường cắt ngang bất thường ở miệng/cằm (<code>seam_spike &gt; 8.0 &amp; peak &gt; 20.0</code>).<br>
            3. <strong>Tầng 3 — Chống trùng lặp &amp; Kết quả lọc thực tế:</strong> Khử trùng lặp tệp bằng băm nội dung <strong>SHA-256</strong> (<code>file_sha256</code>) trong CurationGate (và khử trùng lặp ngữ nghĩa bằng khoảng cách Cosine DINOv2 <code>≥ 0.05</code> ở bước mining). Tổng cộng loại bỏ 1.294 ảnh lỗi trên SD 1.5: Block 1 giữ <strong>2.200/2.745</strong> (80.15%, kết hợp sinh bù tối đa 3 lượt <code>att0→att1→att2</code>); AL Vòng 1 giữ <strong>1.484/1.900</strong> (78.11%); AL Vòng 2 giữ <strong>1.567/1.900</strong> (82.47%) — tổng <strong>5.251/6.545 (80.23%)</strong> đạt chuẩn.
          </div>
        </div>

        <!-- Bảng Phễu Lọc 6.545 -> 5.251 -->
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>Giai Đoạn Sinh SD 1.5</th>
                <th>Ảnh Sinh Thô (Raw)</th>
                <th>Bị Loại (Rejected)</th>
                <th>Đạt Chuẩn Sạch (Curated)</th>
                <th>Tỷ Lệ Đạt (Pass Rate)</th>
                <th>Phân Bổ Tiếp Theo</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Khởi tạo Block 1 (D₀)</strong></td>
                <td>2.745</td>
                <td>545</td>
                <td><strong>2.200</strong></td>
                <td>80.15%</td>
                <td>1.411 Train + 200 Val + 589 Master Test</td>
              </tr>
              <tr>
                <td><strong>Active Learning Vòng 1 (R1)</strong></td>
                <td>1.900</td>
                <td>416</td>
                <td><strong>1.484</strong></td>
                <td>78.11%</td>
                <td>Chấm bởi M₀ → lọc <strong>553</strong> mẫu khó (470 Train + 83 Test)</td>
              </tr>
              <tr>
                <td><strong>Active Learning Vòng 2 (R2)</strong></td>
                <td>1.900</td>
                <td>333</td>
                <td><strong>1.567</strong></td>
                <td>82.47%</td>
                <td>Chấm bởi M₁ → lọc <strong>187</strong> mẫu khó (159 Train + 28 Test)</td>
              </tr>
              <tr class="highlight-champion">
                <td><strong>Tổng cộng SD 1.5</strong></td>
                <td><strong>6.545</strong></td>
                <td><strong>1.294</strong></td>
                <td><span class="badge-pill green">5.251 sạch</span></td>
                <td><strong>80.23%</strong></td>
                <td>Đóng gói hoàn chỉnh D₀ → D₁ → D₂ (740 Hard Samples)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: CURATION FUNNEL & REJECTION BREAKDOWN -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Phễu Lọc CurationGate &amp; Thống Kê Lý Do Loại Bỏ Ảnh Lỗi (6.545 Thô → 5.251 Sạch)</span>
            <span class="artifact-badge champion">CURATION AUDIT FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/curation_funnel_sd15.png" class="artifact-img-full" alt="SD 1.5 CurationGate Funnel and Rejection Reasons" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Nguồn kiểm toán trực tiếp:</strong> Tổng hợp từ <code>fake_manifest.jsonl</code>, <code>curated_manifest.jsonl</code> và <code>round_{1,2}/curation_report.json</code>. Đa số ảnh bị loại là do khung mặt quá nhỏ (&lt;20%), không bắt được mốc mặt (Haar), cắt cụt trán/cằm hoặc xuất hiện ≥ 2 mặt.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: SKIN GAMUT EDA -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Phân Tích EDA Đa Tạp Ảnh Thật: Phổ Màu Da YCbCr &amp; Độ Sáng Nền Tảng (4.000 Real Faces)</span>
            <span class="artifact-badge">NOTEBOOK 01 EDA ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/01_diffusion_dataset_and_curation_cell6_img1.png" class="artifact-img-full" alt="Real Manifold Skin Color Gamut YCbCr" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Phân tích đa tạp ảnh thật gốc:</strong> Khảo sát 4.000 ảnh thật FFHQ/CelebA-HQ trong không gian YCbCr (vùng da hội tụ tự nhiên quanh <code>Cb ∈ [77, 127], Cr ∈ [133, 173]</code>, Mean Y = 129.0) để làm chuẩn đối chiếu trắc quang cho dữ liệu sinh mới.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: IMAGE QUALITY PANEL -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Thống Kê Độ Nét Quang Học, Độ Sáng &amp; Tương Phản Nền Tảng (CelebA-HQ vs FFHQ)</span>
            <span class="artifact-badge">EDA FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/eda/04_image_quality_panel.png" class="artifact-img-full" alt="Image Quality Panel" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            Phân phối độ nét Laplacian, độ sáng trung bình (112.7 vs 116.4) và độ tương phản RMS giữa CelebA-HQ và FFHQ giúp hiệu chỉnh ngưỡng lọc CurationGate khách quan theo phân phối thực.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 5: KHAI THÁC 740 MẪU KHÓ & TẬP THỬ THÁCH 213 CHALLENGE VAULT
           =================================================================== -->
      <section id="sec-hard-samples" class="report-section">
        <span class="section-eyebrow">05 / 740 MẪU KHÓ &amp; CHALLENGE VAULT</span>
        <h2 class="section-title">
          Challenge Vault: <span class="accent">Khai Thác 740 Mẫu Khó SD 1.5 &amp; Phân Tích 4 Lỗi Giải Phẫu (Anatomical Artifacts)</span>
        </h2>
        <p class="section-desc">
          Qua 2 vòng Active Learning đóng trên SD 1.5 (<code>CalibratedHardMiner</code>, <code>configs/active_learning.yaml</code>), mô hình chấm điểm các ảnh đã qua CurationGate sau bước <em>Temperature Scaling</em> (<code>T = 1.9831</code> ở Vòng 2) dựa trên 2 tiêu chí: <strong>False Negatives (<code>P &lt; 0.50</code>)</strong> và <strong>Boundary Uncertain (<code>|P - 0.50| ≤ 0.10</code>)</strong>, kết hợp lọc đa dạng embedding Cosine <code>≥ 0.05</code>. Theo nhật ký <code>closed_loop_metrics.json</code>, thu được <strong>570 FN + 170 Uncertain = 740 mẫu khó</strong> (Vòng 1 bởi <code>M₀</code>: 429 + 124 = 553; Vòng 2 bởi <code>M₁</code>: 141 + 46 = 187); khi đối chiếu lại theo <code>baseline_score</code> trong <code>hard_samples_manifest.jsonl</code> có <strong>616 mẫu &lt; 0.50</strong> và <strong>124 mẫu ≥ 0.50</strong>. Từ 740 mẫu này, nhóm tách <strong>629 mẫu</strong> vào train (D₂, D₃) và <strong>111 mẫu</strong> ghép cùng <strong>102 ảnh thật</strong> thành <strong>213 Challenge Vault</strong>. <em>(Riêng kết quả dùng <code>M₂*</code> profiling trên 6.000 ảnh RV5.1 — <strong>1.792 ca bỏ lọt</strong> &amp; <strong>2.611 ca phân vân</strong> chỉ để quan sát — xem tại <a href="#sec-zero-leakage" style="color: var(--primary); font-weight: 700;">Mục 7</a>).</em>
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: BIOLOGICAL DEFECTS GALLERY -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">4 Dạng Lỗi Giải Phẫu Điển Hình Trên Khuôn Mặt Giả (Challenge Vault)</span>
            <span class="artifact-badge alert">CHALLENGE VAULT ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/02_hard_samples_and_active_learning_cell6_img2.png" class="artifact-img-full" alt="Biological Defect Taxonomy" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>4 Dạng lỗi giải phẫu điển hình:</strong> (1) Lệch trục đồng tử &amp; phản quang giác mạc (catchlight) không khớp hướng sáng; (2) Sụn vành tai ngoài (helix) biến dạng, mờ nhòe nối bất thường với cơ má; (3) Lỗi răng liền khối (unitooth), mất rãnh kẽ răng; (4) Vùng da láng mịn như nhựa (plastic skin), mất vi cấu trúc lỗ chân lông.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: FULL CONTACT SHEET (SCROLLABLE & ZOOMABLE) -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Toàn Bộ 213 Mẫu Thuộc Tập Thử Thách Challenge Vault (111 Hard Fake + 102 Real)</span>
            <span class="artifact-badge">ASSET 10.3 MB</span>
          </div>
          <div class="artifact-row-body" style="padding: 12px; max-height: 850px; overflow-y: auto;">
            <img src="assets/hard_samples_contact_sheet.png" class="artifact-img-full" alt="Hard Samples Contact Sheet" title="Nhấp để phóng to toàn màn hình" style="max-height: none;">
          </div>
          <div class="artifact-row-footer">
            Toàn bộ 213 mẫu trong <code>cumulative_challenge_test.jsonl</code> (111 mẫu giả khó + 102 ảnh thật) được cô lập danh tính tuyệt đối, không bao giờ lọt vào tập huấn luyện của bất kỳ mô hình nào.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 6: ĐỘT PHÁ CỦA ACTIVE LEARNING VÒNG 2 (M2*)
           =================================================================== -->
      <section id="sec-active-learning" class="report-section">
        <span class="section-eyebrow">06 / BƯỚC ĐỘT PHÁ CỦA ACTIVE LEARNING</span>
        <h2 class="section-title">
          Active Learning Vòng 2: <span class="accent">Mô Hình M2* Tăng Tỷ Lệ Phát Hiện Ảnh Giả Từ 11.71% Lên 91.89%</span>
        </h2>
        <p class="section-desc">
          Bằng cách chỉ mở khóa tinh chỉnh tầng Block 11 của DINOv2 (chiếm 8.06% tham số) kết hợp Warm-Start từ M1 và Differential Learning Rates, mô hình M2* đã dịch chuyển ranh giới quyết định chính xác, phát hiện đúng <strong>102/111 mẫu giả cực khó</strong> và nâng ROC-AUC trên Challenge Vault từ <strong>0.8713</strong> lên <strong>0.9875</strong>!
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: RECALL JUMP + ROC CURVE + CONFUSION MATRICES -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Đột Phá Active Learning Trên Challenge Vault: Ma Trận Nhầm Lẫn, Fake Recall &amp; Đường Cong ROC (0.8713 → 0.9875)</span>
            <span class="artifact-badge champion">NOTEBOOK 02 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/02_hard_samples_and_active_learning_cell9_img3.png" class="artifact-img-full" alt="Challenge Vault Confusion Matrices, Recall Jump, and ROC Curves" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Bằng chứng trực quan trên 213 Challenge Vault:</strong> M0 chỉ bắt được 13/111 mẫu giả (11.71%, AUC 87.13%) → M1 tăng lên 78/111 (70.27%) → <strong>M2* đạt 102/111 (91.89%, tăng +80.18%, ROC-AUC đạt 98.75%)</strong> đồng thời giữ vững 98/102 ảnh thật chính xác.
          </div>
        </div>

        <div class="callout-box info">
          <div class="callout-title">🔄 Cơ Chế Đóng Vòng Thông Minh: Error Attribution + AdaptiveBudgetController + PromptGrammarEngine</div>
          <div class="callout-desc">
            Active Learning của đồ án không sinh ảnh ngẫu nhiên mà tự động điều hướng theo điểm yếu của detector:<br>
            • <strong>Phân rã lỗi Vòng 1 (Error Attribution):</strong> Phát hiện <code>sd_inpaint</code> đánh lừa <code>M₀</code> tới <strong>71.70%</strong> (so với <code>sd_img2img</code> 20.82% và <code>sd_text2img</code> 5.59%) → <code>AdaptiveBudgetController</code> tự động dồn ngân sách sinh Vòng 2 vào <strong>Inpaint (897)</strong>, <strong>Img2Img (540)</strong>, <strong>Text2Img (463)</strong>.<br>
            • <strong>Kết quả Vòng 2:</strong> Error rate của Inpaint giảm sâu từ <strong>71.70% → 19.88%</strong>; đồng thời <code>PromptGrammarEngine</code> truy vết và tập trung vào vùng khó nhất như quanh mắt (<code>periocular: 39.53%</code>), ánh sáng studio (<code>13.87%</code>), người cao tuổi (<code>elderly: 13.61%</code>) và flash gắt (<code>13.30%</code>).
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: CLOSED LOOP ATTRIBUTION -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Phản Hồi Đóng Vòng: Error Attribution &amp; Phân Bổ Ngân Sách Sinh Thích Ứng (Vòng 1 → Vòng 2)</span>
            <span class="artifact-badge">CLOSED-LOOP FEEDBACK</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/closed_loop_attribution.png" class="artifact-img-full" alt="Closed-Loop Error Attribution and Adaptive Budgeting" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Nguồn số liệu:</strong> Trích xuất trực tiếp từ <code>experiments/logs/closed_loop_metrics.json</code> (thuật toán trong <code>AdaptiveBudgetController</code> và <code>PromptGrammarEngine</code>).
          </div>
        </div>

        <!-- Empirical Table -->
        <table class="data-table">
          <thead>
            <tr>
              <th>Phiên Bản Mô Hình</th>
              <th>Fake Recall (↑)</th>
              <th>Bypass Rate (↓)</th>
              <th>Accuracy (↑)</th>
              <th>ROC-AUC (↑)</th>
              <th>Số Mẫu Bắt Được (/111)</th>
              <th>Đánh Giá Kỹ Thuật</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>M0 (D0 Baseline)</strong></td>
              <td>11.71%</td>
              <td>88.29%</td>
              <td>53.99%</td>
              <td>0.8713</td>
              <td>13 / 111 mẫu</td>
              <td><span class="badge-pill red">BỎ SÓT NHIỀU</span></td>
            </tr>
            <tr>
              <td><strong>M1 (Active Round 1)</strong></td>
              <td>70.27%</td>
              <td>29.73%</td>
              <td>83.10%</td>
              <td>0.9655</td>
              <td>78 / 111 mẫu</td>
              <td><span class="badge-pill amber">CẢI THIỆN RÕ RỆT</span></td>
            </tr>
            <tr>
              <td><strong>M2 (Cold-Start Retrain)</strong></td>
              <td>86.49%</td>
              <td>13.51%</td>
              <td>92.49%</td>
              <td>0.9937</td>
              <td>96 / 111 mẫu</td>
              <td><span class="badge-pill green">KẾT QUẢ TỐT</span></td>
            </tr>
            <tr class="highlight-champion">
              <td><strong>🏆 M2* (Refined Champion)</strong></td>
              <td><strong>91.89% (+80.18%)</strong></td>
              <td><strong>8.11%</strong></td>
              <td><strong>93.90%</strong></td>
              <td><strong>0.9875</strong></td>
              <td><strong>102 / 111 mẫu</strong></td>
              <td><span class="badge-pill green">HIỆU QUẢ CAO NHẤT</span></td>
            </tr>
          </tbody>
        </table>
      </section>


      <!-- ===================================================================
           PHẦN 7: BẢO TOÀN TÍNH KHÁCH QUAN: ZERO IDENTITY LEAKAGE
           =================================================================== -->
      <section id="sec-zero-leakage" class="report-section">
        <span class="section-eyebrow">07 / BẢO TOÀN KHÁCH QUAN &amp; GIẢI TRÌNH RV5.1</span>
        <h2 class="section-title">
          Zero Identity Leakage: <span class="accent">Phân Vai 740 Mẫu Khó SD 1.5 vs 6.000 Ảnh RV5.1 &amp; Cách Ly Danh Tính Tập Test</span>
        </h2>
        <p class="section-desc">
          Khi kiểm toán 6.000 ảnh từ động cơ siêu thực <strong>RealisticVision v5.1</strong>, nhóm phát hiện có <strong>626 ảnh</strong> (<code>313 inpaint + 313 img2img</code>) được sinh từ các khuôn mặt gốc thuộc tập kiểm thử <code>test_reals.jsonl</code>, và <strong>600 ảnh</strong> thuộc tập <code>val_reals.jsonl</code>. Nhóm cách ly tuyệt đối 626 ảnh test sang <code>unseen_rv51_test.jsonl</code> (cùng 589 ảnh thật thành <strong>1.215 ảnh</strong>) và 600 ảnh val sang <code>D3_clean/val.jsonl</code>; từ <strong>4.774 ảnh RV5.1 sạch danh tính</strong> còn lại, nhóm trích mẫu phân tầng <strong>1.384 ảnh</strong> kết hợp với <strong>629 mẫu khó SD 1.5</strong> và <strong>800 mẫu SD 1.5 cơ bản</strong> thành tập huấn luyện <code>D3_clean/train.jsonl</code> cân bằng 1:1 (<strong>2.813 Thật : 2.813 Giả = 5.626 ảnh</strong>).
        </p>

        <div class="callout-box info">
          <div class="callout-title">🛡️ Giải Trình Khoa Học: Vì Sao Con Số "740 Mẫu Khó" Không Gộp 6.000 Ảnh RealisticVision v5.1?</div>
          <div class="callout-desc">
            1) <strong>Đúng phân vai của Đề tài (docs/PURPOSE.md):</strong> <code>740 mẫu khó</code> là sản phẩm Active Learning nội miền của <strong>SD 1.5</strong> (553 vòng 1 + 187 vòng 2), chứng minh Proof of Utility cấp độ 1 trên <code>D0 → D1 → D2</code>.<br>
            2) <strong>RV5.1 đóng vai trò Thử Nghiệm Đối Kháng Ngoại Miền (OOD cho M₀–M₂*):</strong> Ở độ phân giải chuẩn <code>512×512</code>, <code>M₀</code> bỏ lọt <strong>33.55%</strong> và <code>M₂*</code> vẫn bỏ lọt <strong>28.12%</strong> trên RV5.1; còn ở kịch bản giảm phân giải <code>224×224</code> trên toàn bộ 6.000 ảnh (<code>benchmark_legacy_6k.py</code>), <code>M₀</code> bỏ lọt tới <strong>98.00%</strong> (5.880/6.000 ảnh) và <code>M₂</code> bỏ lọt <strong>29.87%</strong> (1.792/6.000 ảnh).<br>
            3) <strong>Chống rò rỉ danh tính tuyệt đối:</strong> Nếu gộp bừa ảnh RV5.1 bị bỏ lọt vào tập train, 626 ảnh có khuôn mặt trùng <code>test_reals.jsonl</code> sẽ rò rỉ vào tập học. Việc cách ly 626 ảnh này theo nguồn gốc danh tính giúp kiểm chứng <code>M₃</code> thích nghi đa động cơ trên <strong>danh tính hoàn toàn độc lập</strong> (93.13% Recall).
          </div>
        </div>

        <!-- Bảng đối chiếu 2 cấp độ Proof of Utility -->
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th style="width: 22%;">Cấp Độ Thực Nghiệm</th>
                <th style="width: 24%;">Nguồn Dữ Liệu Sinh</th>
                <th style="width: 26%;">Cơ Chế Khai Thác / Kiểm Định</th>
                <th style="width: 28%;">Kết Quả Chứng Minh (Proof of Utility)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Cấp 1: Nội Miền SD 1.5</strong></td>
                <td>Stable Diffusion v1.5 (~6.545 ảnh)</td>
                <td>Active Learning 2 vòng → <strong>740 mẫu khó</strong> (629 train D₂, 111 Challenge Vault)</td>
                <td><code>M₂*</code> tăng Recall Challenge Vault <strong>11.71% → 91.89%</strong> (+80.18%, AUC 0.9875)</td>
              </tr>
              <tr class="highlight-champion">
                <td><strong>Cấp 2: Đa Động Cơ (D₃ → M₃)</strong></td>
                <td>RealisticVision v5.1 (6.000 ảnh) + SD 1.5</td>
                <td>Cách ly <strong>626 ảnh trùng test</strong> + 600 ảnh val; trích phân tầng 1.384 RV5.1 sạch + 629 hard SD 1.5</td>
                <td><code>M₃</code> tăng Recall trên <code>unseen_rv51_test</code> <strong>71.88% → 93.13%</strong> (bypass 6.87%), giữ nguyên 91.89% SD 1.5</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: CLEAN D3 PARTITION -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Sơ Đồ Phân Vùng Dữ Liệu Sạch D3: Cách Ly Hoàn Toàn Danh Tính Tập Test</span>
            <span class="artifact-badge champion">NOTEBOOK 03 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/03_champion_m3_and_universal_benchmark_cell3_img0.png" class="artifact-img-full" alt="Zero Identity Leakage Clean Partition Schema" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Cấu trúc tập D3 sạch (5.626 ảnh):</strong> Cân bằng 1:1 chuẩn xác (2.813 Thật : 2.813 Giả = 1.384 RV5.1 sạch + 629 AL Hard Samples + 800 SD 1.5 Baseline). Không một ảnh nào trùng danh tính tập test lọt vào tập huấn luyện!
          </div>
        </div>

        <!-- M2* OOD Hard-Sample Profiling Callout & Chart -->
        <div class="callout-box success">
          <div class="callout-title">🔍 Error Analysis / OOD Hard-Sample Profiling Bằng M₂* Trên 6.000 Ảnh RV5.1 (Không Dùng Để Train → Không Leakage)</div>
          <div class="callout-desc">
            Chạy suy luận 1 chiều (inference-only) để quan sát và phân tích điểm mù của mô hình trên 6.000 ảnh RV5.1 hoàn toàn hợp lệ về phương pháp luận (khớp Bước 2 trong <code>PURPOSE.md</code>) và <strong>không gây rò rỉ dữ liệu</strong> vì các mẫu này không đưa vào train. Nhóm chọn <strong>M₂* (Refined Warm-Start, <code>active_learning_m2_best.pt</code>)</strong> thay vì M₂ (Cold-Start) vì M₂ bị <em>Catastrophic Interference</em> (tụt Recall RV5.1 512px xuống <strong>56.55%</strong>, kém cả M₀ 66.45%), nên nếu dùng M₂ sẽ lọc nhầm nhiều mẫu do quên kiến thức; trong khi M₂* giữ Recall <strong>71.88%</strong> (512px) và <strong>91.89%</strong> (Challenge Vault), nên những ca đánh lừa được M₂* mới là điểm mù siêu thực đích thực:<br>
            • <strong>Bị M₂* bỏ lọt (<code>P(Fake) &lt; 0.50</code>): 1.792 / 6.000 ảnh (29.87%)</strong> — gồm <code>img2img</code> <strong>834/2.000 (41.70%)</strong>, <code>inpaint</code> <strong>691/2.000 (34.55%</strong>; vùng miệng 35.30%, vùng mắt 33.80%), và <code>text2img</code> <strong>267/2.000 (13.35%)</strong>.<br>
            • <strong>Vùng phân vân (<code>|P - 0.50| ≤ 0.15</code>): 2.611 / 6.000 ảnh (43.52%)</strong> (hợp cả 2 tiêu chí: <strong>3.633 / 6.000 ảnh</strong>).
          </div>
        </div>

        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Hồ Sơ Điểm Mù Ngoại Miền (OOD Hard-Sample Profiling) Của M₂* Trên 6.000 Ảnh RV5.1</span>
            <span class="artifact-badge">ERROR ANALYSIS FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/m2star_rv51_hard_profiling.png" class="artifact-img-full" alt="M2* OOD Hard-Sample Profiling on 6,000 RV5.1 Images" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Kết luận phân tích điểm mù:</strong> Chế độ <code>img2img</code> (41.70% bypass) và <code>inpaint</code> vùng miệng (35.30%) của RV5.1 là nguồn đánh lừa mạnh nhất đối với bộ dò chỉ học SD 1.5 (<code>M₂*</code>), trực tiếp lý giải vì sao cần thích nghi đa động cơ để tạo ra <code>M₃</code>.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 8: CHAMPION M3 VÀ ĐÁNH GIÁ TRÊN 3 TẬP KIỂM THỬ ĐỘC LẬP
           =================================================================== -->
      <section id="sec-champion-m3" class="report-section">
        <span class="section-eyebrow">08 / MÔ HÌNH TOÀN DIỆN CHAMPION M3</span>
        <h2 class="section-title">
          Mô Hình Champion M3: <span class="accent">Đạt 93.13% Trên Unseen RV5.1 Mà Vẫn Giữ Vững Hiệu Năng SD 1.5</span>
        </h2>
        <p class="section-desc">
          Kế thừa trọng số từ M2* (Warm-Start) và tinh chỉnh Block 11 + Head qua 6 epochs với Cosine Annealing trên tập <code>D3_clean/train.jsonl</code> (5.626 ảnh), mô hình Champion M3 đạt <code>best_val_auc = 0.9846</code> ở Epoch 6 và nâng Fake Recall trên <code>unseen_rv51_test.jsonl</code> (độc lập danh tính) từ 71.88% lên <strong>93.13%</strong> (ROC-AUC <strong>0.9828</strong>) mà không bị quên kiến thức SD 1.5!
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: QUAD-VAULT RECALL CHART -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">So Sánh Tỷ Lệ Phát Hiện (Fake Recall) Của 5 Mô Hình Trên 3 Tập Kiểm Thử Độc Lập</span>
            <span class="artifact-badge champion">BENCHMARK FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_5models_vault_recall_comparison.png" class="artifact-img-full" alt="Quad-Vault Recall Comparison" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            Mô hình M3 đạt Fake Recall <strong>93.13%</strong> trên tập kiểm thử <code>unseen_rv51_test</code> (Bypass Rate giảm từ 28.12% còn <strong>6.87%</strong>), đồng thời duy trì trọn vẹn <strong>91.89%</strong> trên Challenge Vault và tăng lên <strong>69.44%</strong> trên Master Frozen Test (SD 1.5).
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: M3 6-EPOCH TRAINING CONVERGENCE -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Động Học Huấn Luyện Champion M3 Qua 6 Epochs (Train/Val Loss &amp; Validation AUC 0.9846)</span>
            <span class="artifact-badge champion">TRAINING DYNAMICS</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/m3_training_convergence.png" class="artifact-img-full" alt="Champion M3 6-Epoch Training Convergence" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Hội tụ mượt mà không Overfitting:</strong> Train Loss giảm đều <code>0.1833 → 0.0831</code>, Val Loss giảm <code>0.2552 → 0.2168</code>, và Val ROC-AUC trên <code>D3_clean/val.jsonl</code> tăng liên tục từ <code>0.9745</code> lên <strong>0.9846</strong> (Val Acc <strong>94.07%</strong>) tại Epoch 6.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: M3 LIVE INFERENCE EXAMPLES -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Suy Luận Trực Tiếp (Live Inference) Của Champion M3 Trên Unseen RV5.1 (Phân Tích Ca Đúng &amp; Ca Sai)</span>
            <span class="artifact-badge">NOTEBOOK 03 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/03_champion_m3_and_universal_benchmark_cell5_img1.png" class="artifact-img-full" alt="M3 Live Inference Examples on Unseen RV5.1" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Minh họa trực quan từ tập Unseen RV5.1:</strong> Hàng trên hiển thị ảnh thật (2 ca nhận đúng với P(fake) 15.1%–45.2% và 2 ca báo động nhầm 65.5%–72.6%); hàng dưới hiển thị ảnh giả RV5.1 (3 ca bắt đúng với P(fake) 72.0%–86.5% và 1 ca bỏ lọt 14.4% — phản ánh trực tiếp Bypass Rate 6.87% còn lại).
          </div>
        </div>

        <!-- Full Quad-Vault Scorecard Table with Accuracy, ROC-AUC, F1, Recall, Bypass -->
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>Phiên Bản Mô Hình</th>
                <th>Master Frozen Test (SD 1.5 — 1.178 ảnh)<br>Recall | Bypass | Acc | ROC-AUC | F1</th>
                <th>Challenge Vault (Mẫu Khó — 213 ảnh)<br>Recall | Bypass | Acc | ROC-AUC | F1</th>
                <th>Unseen RV5.1 Vault (1.215 ảnh — Độc Lập Danh Tính)<br>Recall | Bypass | Acc | ROC-AUC | F1</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>M0 (D0 Baseline)</strong></td>
                <td>49.07% | 50.93% | 74.11% | 0.8850 | 0.6546</td>
                <td>11.71% | 88.29% | 53.99% | 0.8713 | 0.2097</td>
                <td>66.45% | 33.55% | 82.30% | 0.9162 | 0.7947 (OOD)</td>
              </tr>
              <tr>
                <td><strong>M1 (Active Round 1)</strong></td>
                <td>62.14% | 37.86% | 79.20% | 0.9161 | 0.7492</td>
                <td>70.27% | 29.73% | 83.10% | 0.9655 | 0.8125</td>
                <td>72.04% | 27.96% | 83.79% | 0.9170 | 0.8207 (OOD)</td>
              </tr>
              <tr>
                <td><strong>M2 (Cold-Start Retrain)</strong></td>
                <td>52.63% | 47.37% | 75.04% | 0.8910 | 0.6783</td>
                <td>86.49% | 13.51% | 92.49% | 0.9937 | 0.9231</td>
                <td>56.55% | 43.45% | 76.38% | 0.8804 | 0.7116 (OOD)</td>
              </tr>
              <tr>
                <td><strong>M2* (Refined Warm-Start)</strong></td>
                <td>66.89% | 33.11% | 80.81% | 0.9177 | 0.7771</td>
                <td><strong>91.89%</strong> | <strong>8.11%</strong> | <strong>93.90%</strong> | 0.9875 | <strong>0.9401</strong></td>
                <td>71.88% | 28.12% | 82.96% | 0.9158 | 0.8130 (OOD)</td>
              </tr>
              <tr class="highlight-champion">
                <td><strong>🏆 M3 (Dual-Engine Champion)</strong></td>
                <td><span class="badge-pill green">69.44% | 30.56% | 82.26% | 0.9206 | 0.7965</span></td>
                <td><span class="badge-pill green">91.89% | 8.11% | 93.43% | 0.9875 | 0.9358</span></td>
                <td><span class="badge-pill green">93.13% | 6.87% | 94.07% | 0.9828 | 0.9418</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Hyperparameter Progression Table -->
        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th>Mô Hình</th>
                <th>Khởi Tạo (Init)</th>
                <th>Tầng Mở Khóa (Unfreeze)</th>
                <th>Learning Rate &amp; Scheduler</th>
                <th>Hard Loss Weight &amp; Epochs</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>M₀ / M₁ / M₂ (Cold)</strong></td>
                <td>DINOv2 ImageNet Pretrained</td>
                <td>Block 11 + Head (1.78M = 8.06%)</td>
                <td>Head <code>2e-4</code> | Block 11 <code>2e-5</code></td>
                <td><code>hard_weight = 2.0</code>, 5 epochs (M₀ ep 5, M₁ ep 4, M₂ ep 5)</td>
              </tr>
              <tr>
                <td><strong>M₂* (Refined)</strong></td>
                <td>Warm-Start từ <code>M₁</code></td>
                <td>Block 11 + Head (1.78M = 8.06%)</td>
                <td>Head <code>5e-5</code> | Block 11 <code>5e-6</code></td>
                <td><code>hard_weight = 1.15</code>, 2 epochs (AMP, best ep 2)</td>
              </tr>
              <tr class="highlight-champion">
                <td><strong>M₃ (Champion)</strong></td>
                <td>Warm-Start từ <code>M₂*</code></td>
                <td>Block 11 + Head (1.78M = 8.06%)</td>
                <td>Head <code>3e-5</code> | Block 11 <code>3e-6</code> (CosineAnnealingLR)</td>
                <td><code>hard_weight = 1.10</code>, 6 epochs (AMP, best ep 6)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 9: ĐÁNH GIÁ MỞ RỘNG TOÀN DIỆN (UNIVERSAL BENCHMARK 20.400 ẢNH)
           =================================================================== -->
      <section id="sec-universal" class="report-section">
        <span class="section-eyebrow">09 / ĐÁNH GIÁ MỞ RỘNG ZERO-SHOT CROSS-FAMILY</span>
        <h2 class="section-title">
          Universal Benchmark: <span class="accent">Stress-Test Zero-Shot Trên 20.400 Ảnh Từ 34 Thuật Toán (8 Họ Công Nghệ)</span>
        </h2>
        <p class="section-desc">
          Kiểm định giới hạn Zero-Shot (ở độ phân giải <code>224×224</code>) của mô hình chỉ huấn luyện trên ảnh tĩnh Diffusion khi đối mặt với 10.200 ảnh Thật và 10.200 ảnh Giả từ <strong>34 phương pháp</strong> thuộc 8 nhóm: <em>Reenactment &amp; Talking Head, Face Swapping, GAN Architectures, Diffusion Transformers (DiT), GAN Inversion &amp; Editing, Diffusion Baselines (DDIM), Latent Diffusion (SD 2.1), và AutoEncoders/VAEs</em>.
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: 5-MODEL 34-METHOD HEATMAP -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Heatmap Fake Recall (%) Của Đủ 5 Mô Hình (M0 → M3) Trên 34 Phương Pháp Sinh Ảnh</span>
            <span class="artifact-badge champion">5-MODEL HEATMAP</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_5models_method_heatmap.png" class="artifact-img-full" alt="Universal Benchmark 5-Model 34-Method Heatmap" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Phân tích chi tiết trên 34 phương pháp:</strong> So với M₀–M₂*, mô hình <strong>Champion M₃</strong> cải thiện rõ rệt trên <code>e4e</code> (42.3%), <code>styleclip</code> (35.7%), <code>StyleGAN2</code> (26.0%), <code>stargan</code> (19.7%), <code>ddim</code> (17.3%), <code>pixart</code> (13.0%) và <code>sd2.1</code> (8.3%), đồng thời giữ Real Specificity ở mức <strong>90.42%</strong>.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: 5-MODEL GLOBAL COMPARISON -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">So Sánh Toàn Cục 5 Mô Hình Trên Universal Benchmark (20.400 Ảnh)</span>
            <span class="artifact-badge">5-MODEL GLOBAL CHART</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_5models_global_comparison.png" class="artifact-img-full" alt="Universal Benchmark 5-Model Global Comparison" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            Fake Recall tổng thể tăng hơn gấp đôi từ <strong>2.48%</strong> (M₂*) lên <strong>5.49%</strong> (M₃), giảm Bypass Rate xuống 94.51% trong điều kiện hoàn toàn Zero-Shot đa họ công nghệ.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: 8-FAMILY RECALL -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Tỷ Lệ Nhận Diện (Fake Recall) Phân Theo 8 Nhóm Công Nghệ Sinh Ảnh</span>
            <span class="artifact-badge">BENCHMARK FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_5models_family_recall.png" class="artifact-img-full" alt="8-Family Recall Chart" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            Tỷ lệ phát hiện trên nhóm <strong>Diffusion Baselines (DDIM)</strong> tăng gần gấp 3 lần (từ 6.00% lên <strong>17.33%</strong>), <strong>Latent Diffusion (SD 2.1)</strong> tăng gần gấp đôi (từ 4.67% lên <strong>8.33%</strong>), và <strong>GAN Inversion</strong> đạt đỉnh <strong>21.50%</strong>.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: THRESHOLD SWEEP & FF++ C23 VIDEO FPR -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Quét Ngưỡng Quyết Định (τ) &amp; Kiểm Định Báo Động Nhầm Trên Video Nén FaceForensics++ C23</span>
            <span class="artifact-badge champion">NOTEBOOK 03 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/03_champion_m3_and_universal_benchmark_cell9_img3.png" class="artifact-img-full" alt="Threshold Sweep and FF++ C23 Video FPR" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Bằng chứng trực quan:</strong> Ở ngưỡng mặc định <code>τ = 0.50</code>, M₃ đạt Real Specificity <strong>90.42%</strong> và kiểm soát tỷ lệ báo động sai (FPR) ở mức <strong>2.60%</strong> (chỉ 43/1.655 khung hình video nén H.264 của FaceForensics++ C23).
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: ROC CURVES (TRUNG THỰC VỀ DOMAIN GAP) -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Đường Cong ROC (20.400 Ảnh) &amp; Phân Tích Trung Thực Khoảng Trống Miền (Domain Gap)</span>
            <span class="artifact-badge alert">ZERO-SHOT DOMAIN GAP</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_5models_roc_curves.png" class="artifact-img-full" alt="ROC Curves 5 Models on Universal Benchmark" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Đánh giá khoa học trung thực:</strong> Vì tập huấn luyện chỉ gồm ảnh khuôn mặt tĩnh từ Diffusion (SD 1.5 + RV5.1) và bài test này chạy ở <code>224×224</code> trên đa số phương pháp ngoài Diffusion (FaceSwap, Talking Head, GAN, VAE), chỉ số ROC-AUC tổng hợp hiện chỉ đạt <strong>0.3820</strong> (PR-AUC 0.4233). Điều này chứng minh rõ ràng giới hạn chuyên biệt theo họ sinh ảnh và mở ra hướng nâng cấp bổ sung dữ liệu GAN/FaceSwap/Video ở giai đoạn tiếp theo.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 10: TRỰC QUAN HÓA GIẢI THÍCH MÔ HÌNH (XAI)
           =================================================================== -->
      <section id="sec-xai" class="report-section">
        <span class="section-eyebrow">10 / MINH BẠCH VÀ GIẢI THÍCH DỰ ĐOÁN (XAI)</span>
        <h2 class="section-title">
          Giải Thích Mô Hình (XAI): <span class="accent">Attention Heatmap Của DINOv2 Định Vị Vùng Chỉnh Sửa</span>
        </h2>
        <p class="section-desc">
          Mô hình không phải là "hộp đen" khó hiểu — trên nền kiến trúc gọn nhẹ <strong>DINOv2 ViT-S/14 (22.06M tham số; chỉ mở khóa tinh chỉnh 1.78M tham số ở Block 11 + Head = 8.06%)</strong>, cơ chế trích xuất bản đồ Self-Attention từ Block 11 cho phép định vị trực quan các vùng bị chỉnh sửa trên khuôn mặt.
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: DINOV2 ATTENTION HEATMAP -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">DINOv2 Spatial Attention Heatmap (Block 11)</span>
            <span class="artifact-badge champion">NOTEBOOK 00 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/00_master_deepfake_defense_showcase_cell7_img1.png" class="artifact-img-full" alt="DINOv2 Spatial Attention Heatmap" title="Nhấp để phóng to toàn màn hình">
          </div>
          <div class="artifact-row-footer">
            <strong>Bằng chứng trực quan trên 3 mẫu đối chứng:</strong> (1) Ảnh thật FFHQ <code>00301.png</code> được nhận diện chuẩn xác Real (<code>P(Fake)=1.99%</code>); (2) Mẫu khó Challenge Vault <code>fake_ffhq_00000_q1_116.png</code> bị bắt ở <code>P(Fake)=99.79%</code> tập trung vào vùng miệng/mắt; (3) Ảnh siêu thực RV5.1 <code>187_img2img.png</code> bị bắt ở <code>P(Fake)=99.99%</code>.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 11: TỔNG KẾT VÀ ĐỐI CHIẾU KPI
           =================================================================== -->
      <section id="sec-conclusion" class="report-section">
        <span class="section-eyebrow">11 / TỔNG KẾT VÀ ĐỐI CHIẾU KPI</span>
        <h2 class="section-title">
          Tổng Kết Nghiệm Thu: <span class="accent">Hoàn Thành Trọn Vẹn Các Mục Tiêu Cam Kết Của Đồ Án</span>
        </h2>
        <p class="section-desc">
          Bảng đối chiếu tổng hợp minh chứng Đề tài 11 đã hoàn thành xuất sắc tất cả các mục tiêu kỹ thuật ban đầu, mã nguồn chuẩn hóa, vượt qua toàn bộ các bài kiểm thử tự động và đảm bảo tính khách quan trong nghiên cứu.
        </p>

        <div class="table-responsive">
          <table class="data-table">
            <thead>
              <tr>
                <th style="width: 24%;">Hạng Mục Nghiệm Thu</th>
                <th style="width: 28%;">Mục Tiêu Ban Đầu (Define &amp; Design)</th>
                <th style="width: 33%;">Kết Quả Thực Nghiệm Đạt Được</th>
                <th style="width: 15%;">Đánh Giá</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>1. Dữ Liệu Dual-Engine Diffusion</strong></td>
                <td>Sinh ảnh Stable Diffusion với 3 chế độ Inpaint / Img2Img / Text2Img</td>
                <td>4.000 ảnh thật + 12.545 ảnh giả (SD 1.5: 6.545 thô / 5.251 sạch + RV5.1: 6.000); làm mờ biên mềm (Soft-Mask Feathering, <code>blur_radius = 18–20</code>)</td>
                <td><span class="badge-pill green">ĐẠT 100%</span></td>
              </tr>
              <tr>
                <td><strong>2. Quy Trình CurationGate</strong></td>
                <td>Lọc chất lượng quang học, hình học khuôn mặt và chống trùng lặp</td>
                <td>Laplacian hiệu chỉnh theo phân vị 5% ảnh thật, OpenCV Haar + kiểm tra hình học/cháy sáng/seam, khử trùng lặp SHA-256 (giữ 5.251/6.545 SD 1.5)</td>
                <td><span class="badge-pill green">ĐẠT 100%</span></td>
              </tr>
              <tr class="highlight-champion">
                <td><strong>3. Khai Thác Mẫu Khó</strong></td>
                <td>Áp dụng Active Learning cải thiện ranh giới phân loại</td>
                <td>740 mẫu khó SD 1.5 (616 FN + 124 Uncertain |P−0.5|≤0.10; 629 train + 111 challenge); 213 Challenge Vault; nâng Recall 11.71% → 91.89%</td>
                <td><span class="badge-pill green">VƯỢT (+80.18%)</span></td>
              </tr>
              <tr>
                <td><strong>4. Zero Identity Leakage</strong></td>
                <td>Đảm bảo không rò rỉ danh tính người giữa tập học và tập test</td>
                <td>M₂* profiling 6k RV5.1 (1.792 bypass); cách ly 626 ảnh test + 600 val; tập D3 sạch 5.626 ảnh cân bằng 1:1 (2.813 Real : 2.813 Fake)</td>
                <td><span class="badge-pill green">KHÁCH QUAN</span></td>
              </tr>
              <tr class="highlight-champion">
                <td><strong>5. Champion M3 &amp; Benchmark</strong></td>
                <td>Thử nghiệm Proof of Utility đa động cơ &amp; đánh giá mở rộng</td>
                <td>Champion M3 đạt 93.13% trên Unseen RV5.1 (AUC 0.9828); kiểm chứng Zero-Shot trên 20.400 ảnh (34 phương pháp) &amp; FF++ C23 FPR 2.60%</td>
                <td><span class="badge-pill green">VƯỢT CHỈ TIÊU</span></td>
              </tr>
              <tr>
                <td><strong>6. XAI Trực Quan Hóa</strong></td>
                <td>Trực quan hóa vùng đặc trưng giúp giải thích quyết định của mô hình</td>
                <td>Bản đồ nhiệt Attention Heatmap từ Block 11 của DINOv2 ViT-S/14 (22.06M params) định vị rõ vùng mắt, miệng và viền mặt</td>
                <td><span class="badge-pill green">HOÀN THÀNH</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="callout-box success">
          <div class="callout-title">🧪 Engineering Rigor &amp; Tính Tái Lập (Reproducibility)</div>
          <div class="callout-desc">
            • <strong>Thời gian thực thi thực nghiệm:</strong> Pipeline 10.000 mẫu SD 1.5 (sinh ảnh + CurationGate + Active Learning 2 vòng) hoàn tất trong <strong>183.71 phút</strong> (<code>clean_10k_production_report.json</code>); huấn luyện <code>Champion M₃</code> 6 epochs với PyTorch AMP hết <strong>24.7 phút</strong> (<code>1.483,3s</code>, <code>m3_training_evaluation_summary.json</code>).<br>
            • <strong>Kiểm soát tái lập &amp; kiểm thử tự động:</strong> Khóa cố định <code>seed = 42</code> qua <code>seed_everything(42)</code>; toàn bộ codebase được bảo vệ bởi <strong>198 unit tests</strong> trong <code>tests/</code> (trong đó <strong>41 smoke tests</strong> chạy xác minh dưới 5 giây).
          </div>
        </div>

        <div class="callout-box info">
          <div class="callout-title">🔭 Giới Hạn Nghiên Cứu &amp; Hướng Phát Triển Tiếp Theo</div>
          <div class="callout-desc">
            • <strong>Khoảng trống ngoài Diffusion:</strong> Kết quả Universal Benchmark 20.400 ảnh cho thấy mô hình chuyên sâu Diffusion chưa bao phủ tốt các kỹ thuật GAN/Face-Swap/Video Talking Head (AUC tổng 0.3820). Giai đoạn tiếp theo sẽ mở rộng vòng Active Learning sang các họ mô hình không phải Diffusion và bổ sung đặc trưng tần số (FFT/DCT) cũng như tính nhất quán thời gian (temporal consistency) cho video.<br>
            • <strong>Chuẩn hóa độ phân giải đầu vào:</strong> Suy luận ở <code>224×224</code> làm mờ vi vết tần số cao so với <code>512×512</code>; các hệ thống triển khai thực tế nên giữ nguyên độ phân giải bản địa ≥ 504×504 của DINOv2.
          </div>
        </div>

        <div style="background: #fdfae7; border: 2px solid var(--primary); border-radius: var(--radius-md); padding: 24px; text-align: center; margin-top: 28px;">
          <h3 style="font-family: var(--font-display); font-size: 18px; color: var(--primary); margin-bottom: 8px;">
            KÍNH CẢM ƠN HỘI ĐỒNG GIẢNG VIÊN BỘ MÔN THỊ GIÁC MÁY TÍNH — TRƯỜNG ĐH GIAO THÔNG VẬN TẢI TP.HCM!
          </h3>
          <p style="font-size: 13px; color: var(--text-muted); max-width: 800px; margin: 0 auto;">
            Nhóm 4 kính mời Thầy Cô trong Hội đồng đánh giá đặt câu hỏi phản biện chuyên sâu và đóng góp ý kiến cho Đề tài 11.
          </p>
        </div>
      </section>

    </main>
  </div>

  <!-- LIGHTBOX MODAL (POPUP PHÓNG TO ẢNH FULL-SCREEN) -->
  <div id="image-lightbox" class="lightbox-modal">
    <span class="lightbox-close" id="lightbox-close" title="Đóng phóng to (hoặc nhấn Esc)">&times;</span>
    <div class="lightbox-content-box">
      <img class="lightbox-img" id="lightbox-img" src="" alt="Full size artifact">
      <div class="lightbox-caption" id="lightbox-caption"></div>
    </div>
  </div>

  <script>
    document.addEventListener('DOMContentLoaded', () => {
      // 1. Sidebar Toggle Functionality
      const toggleBtn = document.getElementById('toggle-sidebar-btn');
      const layoutWrapper = document.getElementById('layout-wrapper');

      toggleBtn.addEventListener('click', () => {
        layoutWrapper.classList.toggle('sidebar-collapsed');
        if (layoutWrapper.classList.contains('sidebar-collapsed')) {
          toggleBtn.innerHTML = '☰ Hiện Mục Lục';
        } else {
          toggleBtn.innerHTML = '✕ Ẩn Mục Lục';
        }
      });

      // 2. Active Section Highlighting on Scroll (with bottom-of-page fallback)
      const sections = Array.from(document.querySelectorAll('.report-section'));
      const navLinks = document.querySelectorAll('.sidebar-nav .nav-link');

      function getCurrentSectionIndex() {
        if ((window.innerHeight + window.scrollY) >= (document.documentElement.scrollHeight - 24)) {
          return sections.length - 1;
        }
        let activeIdx = 0;
        sections.forEach((section, idx) => {
          if (window.pageYOffset >= section.offsetTop - 120) {
            activeIdx = idx;
          }
        });
        return activeIdx;
      }

      function updateActiveNav() {
        const activeIdx = getCurrentSectionIndex();
        const currentId = sections[activeIdx] ? sections[activeIdx].getAttribute('id') : '';
        navLinks.forEach(link => {
          link.classList.remove('active');
          if (link.getAttribute('href') === `#${currentId}`) {
            link.classList.add('active');
          }
        });
      }

      window.addEventListener('scroll', updateActiveNav, { passive: true });
      updateActiveNav();

      // 3. Prev / Next Section Navigation & Print PDF
      const prevBtn = document.getElementById('prev-sec-btn');
      const nextBtn = document.getElementById('next-sec-btn');
      const printBtn = document.getElementById('print-report-btn');

      function jumpSection(delta) {
        const idx = Math.max(0, Math.min(sections.length - 1, getCurrentSectionIndex() + delta));
        sections[idx].scrollIntoView({ behavior: 'smooth', block: 'start' });
      }

      prevBtn.addEventListener('click', () => jumpSection(-1));
      nextBtn.addEventListener('click', () => jumpSection(1));
      printBtn.addEventListener('click', () => window.print());

      // 4. Interactive Lightbox Modal (Click to View Full Resolution)
      const lightbox = document.getElementById('image-lightbox');
      const lightboxImg = document.getElementById('lightbox-img');
      const lightboxCaption = document.getElementById('lightbox-caption');
      const lightboxClose = document.getElementById('lightbox-close');

      const clickableImages = document.querySelectorAll('.artifact-img-full, .face-strip-card img, .zoom-hint-banner');
      clickableImages.forEach(el => {
        el.addEventListener('click', () => {
          let targetImg = el;
          if (el.classList.contains('zoom-hint-banner')) {
            const body = el.closest('.artifact-row-body');
            targetImg = body ? body.querySelector('img') : el.previousElementSibling;
          }
          if (targetImg && targetImg.tagName === 'IMG') {
            lightboxImg.src = targetImg.src;
            lightboxCaption.textContent = targetImg.alt || targetImg.title || 'Artifact Khảo Sát Chi Tiết';
            lightbox.classList.add('active');
            document.body.style.overflow = 'hidden';
          }
        });
      });

      function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = '';
      }

      lightboxClose.addEventListener('click', closeLightbox);
      lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox || e.target === lightboxClose) {
          closeLightbox();
        }
      });

      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && lightbox.classList.contains('active')) {
          closeLightbox();
        } else if (e.altKey && e.key === 'ArrowDown') {
          e.preventDefault();
          jumpSection(1);
        } else if (e.altKey && e.key === 'ArrowUp') {
          e.preventDefault();
          jumpSection(-1);
        }
      });
    });
  </script>
</body>
</html>
"""
    return html_content

def main():
    generate_m0_collapse_chart()
    regenerate_clean_defect_taxonomy()
    regenerate_clean_d3_audit()
    generate_m2star_rv51_profiling_chart()
    generate_m3_convergence_chart()
    generate_curation_funnel_chart()
    generate_closed_loop_attribution_chart()
    html_content = build_vertical_report_html()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Successfully generated {OUTPUT_PATH} ({len(html_content)} bytes)")

if __name__ == "__main__":
    main()
