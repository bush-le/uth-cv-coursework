"""Comprehensive Exploratory Data Analysis (EDA) on Universal Multi-Method Deepfake Benchmark.

Analyzes the curated Golden Balanced Suite (21,000 samples, 35 methods, 8 technology families),
computes optical metrics, and produces publication-ready visualizations and summary tables.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 1. 8-Tier Deepfake Method Taxonomy for Curated 35 Methods
# -----------------------------------------------------------------------------

METHOD_TAXONOMY: Dict[str, str] = {
    # Authentic / Real
    "real": "Real Faces (FFHQ / FF++)",
    # Diffusion Transformers (DiT)
    "DiT": "Diffusion Transformers (DiT)",
    "SiT": "Diffusion Transformers (DiT)",
    "pixart": "Diffusion Transformers (DiT)",
    "RDDM": "Diffusion Transformers (DiT)",
    # Latent Diffusion
    "sd2.1": "Latent Diffusion (SD 2.1)",
    # GAN Architectures
    "StyleGAN2": "GAN Architectures",
    "StyleGAN3": "GAN Architectures",
    "StyleGANXL": "GAN Architectures",
    "stargan": "GAN Architectures",
    "styleclip": "GAN Architectures",
    # AutoEncoders / VAEs
    "VQGAN": "AutoEncoders / VAEs",
    # Diffusion Baselines
    "ddim": "Diffusion Baselines (DDIM)",
    # Face Swapping
    "mobileswap": "Face Swapping",
    "deepfake_faceswap": "Face Swapping",
    "faceswap": "Face Swapping",
    "simswap": "Face Swapping",
    "facedancer": "Face Swapping",
    "inswap": "Face Swapping",
    "blendface": "Face Swapping",
    "danet": "Face Swapping",
    "uniface": "Face Swapping",
    "fsgan": "Face Swapping",
    # Reenactment & Talking Head
    "sadtalker": "Reenactment & Talking Head",
    "wav2lip": "Reenactment & Talking Head",
    "fomm": "Reenactment & Talking Head",
    "MRAA": "Reenactment & Talking Head",
    "pirender": "Reenactment & Talking Head",
    "tpsm": "Reenactment & Talking Head",
    "one_shot_free": "Reenactment & Talking Head",
    "hyperreenact": "Reenactment & Talking Head",
    "facevid2vid": "Reenactment & Talking Head",
    "lia": "Reenactment & Talking Head",
    "mcnet": "Reenactment & Talking Head",
    # GAN Inversion & Editing
    "e4e": "GAN Inversion & Editing",
    "e4s": "GAN Inversion & Editing",
}

FAMILY_COLORS = {
    "Real Faces (FFHQ / FF++)": "#2b5c8f",
    "Face Swapping": "#e66101",
    "Reenactment & Talking Head": "#5e3c99",
    "GAN Architectures": "#0571b0",
    "Diffusion Transformers (DiT)": "#ca0020",
    "Latent Diffusion (SD 2.1)": "#f4a582",
    "GAN Inversion & Editing": "#92c5de",
    "Diffusion Baselines (DDIM)": "#8073ac",
    "AutoEncoders / VAEs": "#b2abd2",
}


def run_universal_benchmark_eda(
    golden_csv: Path = Path("data/benchmarks/universal_deepfake_benchmark/splits/test_coursework_35methods_golden_balanced.csv"),
    base_img_dir: Path = Path("data/benchmarks/universal_deepfake_benchmark/test_images"),
    output_dir: Path = Path("experiments/eda"),
) -> Dict[str, Any]:
    print("=" * 75)
    print("🚀 BẮT ĐẦU PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) TRÊN BỘ BENCHMARK CHUẨN HÓA 21K")
    print("=" * 75)

    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    report_dir = output_dir / "reports"
    for d in [fig_dir, table_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(golden_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_samples = len(rows)
    print(f"[*] Nạp thành công {total_samples:,} bản ghi chuẩn hóa từ {golden_csv.name}")

    method_counts = Counter()
    label_counts = Counter()
    source_counts = Counter()
    family_counts = Counter()
    method_to_sources = defaultdict(set)

    for r in rows:
        m = r["method"]
        l = r["label"]
        s = r["source"]
        method_counts[m] += 1
        label_counts[l] += 1
        source_counts[s] += 1
        fam = METHOD_TAXONOMY.get(m, "Other Deepfakes")
        family_counts[fam] += 1
        method_to_sources[m].add(s)

    # -------------------------------------------------------------------------
    # 2. Image Optical Quality Sampling (25 samples per method)
    # -------------------------------------------------------------------------
    print("\n[*] Đang kiểm toán chất lượng quang học & phân giải trên 35 phương pháp...")
    method_samples = defaultdict(list)
    for r in rows:
        if len(method_samples[r["method"]]) < 25:
            method_samples[r["method"]].append(r)

    resolutions = Counter()
    aspect_ratios = Counter()
    method_metrics = defaultdict(lambda: {"sharpness": [], "brightness": [], "contrast": []})
    family_metrics = defaultdict(lambda: {"sharpness": [], "brightness": [], "contrast": []})
    gallery_candidates = defaultdict(list)

    for m, r_list in method_samples.items():
        fam = METHOD_TAXONOMY.get(m, "Other Deepfakes")
        for r in r_list:
            rel = r["path"].lstrip("/")
            p = base_img_dir / rel
            if not p.exists():
                continue
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    resolutions[(w, h)] += 1
                    aspect_ratios[round(w / h, 2)] += 1
                    if len(gallery_candidates[m]) < 2:
                        gallery_candidates[m].append((p, r))

                    # Compute optical metrics
                    arr = np.array(img.convert("RGB"))
                    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    bright = float(gray.mean())
                    contr = float(gray.std())

                    method_metrics[m]["sharpness"].append(lap_var)
                    method_metrics[m]["brightness"].append(bright)
                    method_metrics[m]["contrast"].append(contr)

                    family_metrics[fam]["sharpness"].append(lap_var)
                    family_metrics[fam]["brightness"].append(bright)
                    family_metrics[fam]["contrast"].append(contr)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 3. Export Summary CSV Tables
    # -------------------------------------------------------------------------
    print("\n[*] Đang xuất các bảng thống kê CSV chuẩn hóa...")

    # Table 1: Methods Summary (35 Methods + Real)
    tbl_methods_path = table_dir / "universal_methods_summary.csv"
    with open(tbl_methods_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Family", "Count", "Percentage", "Label", "Sources", "Avg_Sharpness_Lap", "Avg_Brightness", "Avg_Contrast"])
        for m, count in method_counts.most_common():
            pct = count / total_samples * 100
            fam = METHOD_TAXONOMY.get(m, "Other")
            mets = method_metrics[m]
            avg_sharp = np.mean(mets["sharpness"]) if mets["sharpness"] else float("nan")
            avg_br = np.mean(mets["brightness"]) if mets["brightness"] else float("nan")
            avg_ct = np.mean(mets["contrast"]) if mets["contrast"] else float("nan")
            label_desc = "Real (0)" if m == "real" else "Fake (1)"
            src_str = "; ".join(sorted(method_to_sources[m]))
            writer.writerow([m, fam, count, f"{pct:.2f}%", label_desc, src_str, f"{avg_sharp:.1f}", f"{avg_br:.1f}", f"{avg_ct:.1f}"])
    print(f"  [✓] {tbl_methods_path.name}")

    # Table 2: Families Summary
    tbl_families_path = table_dir / "universal_families_summary.csv"
    with open(tbl_families_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Taxonomy Family", "Total Samples", "Percentage", "Avg_Sharpness_Lap", "Avg_Brightness", "Avg_Contrast"])
        for fam, count in family_counts.most_common():
            pct = count / total_samples * 100
            mets = family_metrics[fam]
            avg_sharp = np.mean(mets["sharpness"]) if mets["sharpness"] else float("nan")
            avg_br = np.mean(mets["brightness"]) if mets["brightness"] else float("nan")
            avg_ct = np.mean(mets["contrast"]) if mets["contrast"] else float("nan")
            writer.writerow([fam, count, f"{pct:.2f}%", f"{avg_sharp:.1f}", f"{avg_br:.1f}", f"{avg_ct:.1f}"])
    print(f"  [✓] {tbl_families_path.name}")

    # Table 3: Resolutions Summary
    tbl_res_path = table_dir / "universal_resolution_summary.csv"
    with open(tbl_res_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Resolution (W x H)", "Sampled Count", "Sampled Percentage", "Aspect Ratio"])
        total_sampled = sum(resolutions.values())
        for (w, h), cnt in resolutions.most_common():
            writer.writerow([f"{w}x{h}", cnt, f"{cnt/total_sampled*100:.2f}%", f"{w/h:.2f}"])
    print(f"  [✓] {tbl_res_path.name}")

    # -------------------------------------------------------------------------
    # 4. Generate Clean Visualizations (Matplotlib)
    # -------------------------------------------------------------------------
    print("\n[*] Đang khởi tạo các đồ thị trực quan hóa chuẩn mực (300 DPI)...")

    # Figure 1: Methods Distribution Bar Chart (Grouped by Technology Family)
    fake_methods_sorted = sorted([m for m in method_counts if m != "real"], key=lambda x: (METHOD_TAXONOMY.get(x, ""), x))
    counts_sorted = [method_counts[m] for m in fake_methods_sorted]
    colors = [FAMILY_COLORS.get(METHOD_TAXONOMY.get(m, ""), "#777777") for m in fake_methods_sorted]

    plt.figure(figsize=(14, 10))
    bars = plt.barh(fake_methods_sorted, counts_sorted, color=colors, edgecolor="black", alpha=0.85)
    plt.title("Universal Multi-Method Benchmark: Phân Bố Chuẩn Hóa 35 Phương Pháp Deepfake (N=300)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Số lượng mẫu kiểm thử (Mỗi phương pháp chuẩn đúng 300 mẫu)", fontsize=11, labelpad=10)
    plt.ylabel("Phương pháp tạo ảnh giả (Phân nhóm theo họ công nghệ)", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.xlim(0, 360)

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 5, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig1_path = fig_dir / "universal_method_distribution.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"  [✓] Đã lưu biểu đồ phân bố: {fig1_path.name}")

    # Figure 2: Taxonomy Families Donut Chart (21,000 Samples)
    plt.figure(figsize=(10, 8))
    fam_names = list(family_counts.keys())
    fam_vals = list(family_counts.values())
    donut_colors = [FAMILY_COLORS.get(f, "#999999") for f in fam_names]

    wedges, texts, autotexts = plt.pie(
        fam_vals,
        labels=fam_names,
        autopct="%1.1f%%",
        pctdistance=0.78,
        colors=donut_colors,
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    plt.setp(autotexts, size=10, weight="bold")
    plt.title("Universal Multi-Method Benchmark: Tỷ Lệ 8 Họ Công Nghệ & Đối Chứng Real (21.000 Ảnh)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig2_path = fig_dir / "universal_method_families.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"  [✓] Đã lưu biểu đồ donut: {fig2_path.name}")

    # Figure 3: Optical Quality Comparison (Sharpness & Contrast across Families)
    plt.figure(figsize=(12, 6))
    fams = [f for f in fam_names if family_metrics[f]["sharpness"]]
    avg_sharps = [np.mean(family_metrics[f]["sharpness"]) for f in fams]
    avg_contrasts = [np.mean(family_metrics[f]["contrast"]) for f in fams]

    x = np.arange(len(fams))
    width = 0.38
    plt.bar(x - width / 2, avg_sharps, width, label="Độ Sắc Nét (Laplacian Variance)", color="#2b5c8f", edgecolor="black")
    plt.bar(x + width / 2, avg_contrasts, width, label="Độ Tương Phản (RMS Contrast)", color="#d95f02", edgecolor="black")

    plt.xticks(x, fams, rotation=35, ha="right", fontsize=10)
    plt.ylabel("Giá trị đo lường quang học trung bình", fontsize=11)
    plt.title("So Sánh Chất Lượng Quang Học Giữa Các Họ Công Nghệ Trong Bộ Benchmark 21k", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig3_path = fig_dir / "universal_quality_by_family.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"  [✓] Đã lưu biểu đồ quang học: {fig3_path.name}")

    # Figure 4: Representative Visual Gallery (12 Major Methods)
    plt.figure(figsize=(16, 12))
    key_methods = [
        "real", "DiT", "SiT", "pixart",
        "StyleGAN3", "StyleGANXL", "VQGAN", "mobileswap",
        "facedancer", "simswap", "sadtalker", "wav2lip"
    ]
    for idx, km in enumerate(key_methods):
        ax = plt.subplot(3, 4, idx + 1)
        if km in gallery_candidates and gallery_candidates[km]:
            img_p, r_meta = gallery_candidates[km][0]
            try:
                pil_im = Image.open(img_p).convert("RGB")
                ax.imshow(pil_im)
                ax.set_title(f"{km}\n[{'REAL' if km=='real' else 'FAKE'}]", fontsize=11, fontweight="bold", color="darkblue" if km=="real" else "darkred")
            except Exception:
                ax.text(0.5, 0.5, "Image Error", ha="center", va="center")
        ax.axis("off")

    plt.suptitle("Universal Multi-Method Benchmark: Bộ Sưu Tập Ảnh Mẫu Đại Diện 12 Phương Pháp SOTA", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig4_path = fig_dir / "universal_representative_gallery.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"  [✓] Đã lưu biểu đồ gallery: {fig4_path.name}")

    print("\n[✓] HOÀN TẤT PHÂN TÍCH VÀ XUẤT CÁC TÀI LIỆU QUANG HỌC!")
    return {
        "total_samples": total_samples,
        "methods_count": len(method_counts),
        "labels": dict(label_counts),
        "families": dict(family_counts),
    }


if __name__ == "__main__":
    run_universal_benchmark_eda()
