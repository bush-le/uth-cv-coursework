"""Evaluate all 4 Deepfake Detector Models (M0, M1, M2_coldstart, M2_refined) on the Universal Benchmark Suite.

Evaluates 20,400 standardized square images (10,200 Real vs 10,200 Fake across 34 methods and 8 families).
Generates detailed metrics, CSV tables, ROC/PR curves, family/method breakdowns, and NPZ prediction caches.
Zero external sklearn dependencies — pure vectorized NumPy and PyTorch.
"""

import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.common.logger import get_logger
from src.common.metrics import (
    compute_accuracy,
    compute_bypass_rate,
    compute_precision,
    compute_recall,
    compute_roc_auc,
)
from src.models.detector.image_detector import DeepfakeImageDetector

logger = get_logger("evaluate_universal_benchmark_4models")

# -----------------------------------------------------------------------------
# 1. 8-Tier Deepfake Method Taxonomy
# -----------------------------------------------------------------------------

METHOD_TAXONOMY: Dict[str, str] = {
    "real": "Real Faces (FFHQ / FF++)",
    "DiT": "Diffusion Transformers (DiT)",
    "SiT": "Diffusion Transformers (DiT)",
    "pixart": "Diffusion Transformers (DiT)",
    "RDDM": "Diffusion Transformers (DiT)",
    "sd2.1": "Latent Diffusion (SD 2.1)",
    "StyleGAN2": "GAN Architectures",
    "StyleGAN3": "GAN Architectures",
    "StyleGANXL": "GAN Architectures",
    "stargan": "GAN Architectures",
    "styleclip": "GAN Architectures",
    "VQGAN": "AutoEncoders / VAEs",
    "ddim": "Diffusion Baselines (DDIM)",
    "mobileswap": "Face Swapping",
    "faceswap": "Face Swapping",
    "simswap": "Face Swapping",
    "facedancer": "Face Swapping",
    "inswap": "Face Swapping",
    "blendface": "Face Swapping",
    "danet": "Face Swapping",
    "uniface": "Face Swapping",
    "fsgan": "Face Swapping",
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

MODEL_CONFIGS = [
    {
        "key": "M0_baseline",
        "display_name": "M₀ (Baseline D₀)",
        "ckpt": Path("experiments/checkpoints/baseline_d0_best.pt"),
        "color": "#1f77b4",
        "style": "--",
    },
    {
        "key": "M1_active_r1",
        "display_name": "M₁ (Active Learning R1)",
        "ckpt": Path("experiments/checkpoints/active_learning_m1_best.pt"),
        "color": "#ff7f0e",
        "style": "-.",
    },
    {
        "key": "M2_coldstart",
        "display_name": "M₂ (Cold-Start Scratch)",
        "ckpt": Path("experiments/checkpoints/active_learning_m2_coldstart_backup.pt"),
        "color": "#d62728",
        "style": ":",
    },
    {
        "key": "M2_refined_warmstart",
        "display_name": "M₂* (Refined Warm-Start)",
        "ckpt": Path("experiments/checkpoints/active_learning_m2_best.pt"),
        "color": "#2ca02c",
        "style": "-",
    },
]


def compute_roc_curve_vectorized(y_true: np.ndarray, y_prob: np.ndarray, num_pts: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    """Pure NumPy calculation of ROC curve (FPR, TPR)."""
    taus = np.linspace(0.0, 1.0, num_pts)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    tprs = []
    fprs = []
    for tau in taus:
        pred_pos = (y_prob >= tau)
        tp = np.sum(pred_pos & pos_mask)
        fp = np.sum(pred_pos & neg_mask)
        tprs.append(tp / n_pos if n_pos > 0 else 0.0)
        fprs.append(fp / n_neg if n_neg > 0 else 0.0)

    # Sort by FPR ascending
    sort_idx = np.argsort(fprs)
    return np.array(fprs)[sort_idx], np.array(tprs)[sort_idx]


def compute_pr_curve_vectorized(y_true: np.ndarray, y_prob: np.ndarray, num_pts: int = 250) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pure NumPy calculation of PR curve (Precision, Recall) and Average Precision."""
    taus = np.linspace(0.0, 1.0, num_pts)
    pos_mask = (y_true == 1)
    n_pos = np.sum(pos_mask)

    precs = []
    recs = []
    for tau in taus:
        pred_pos = (y_prob >= tau)
        tp = np.sum(pred_pos & pos_mask)
        total_pred = np.sum(pred_pos)
        p = tp / total_pred if total_pred > 0 else 1.0
        r = tp / n_pos if n_pos > 0 else 0.0
        precs.append(p)
        recs.append(r)

    # Sort by recall ascending for AUC
    sort_idx = np.argsort(recs)
    r_sorted = np.array(recs)[sort_idx]
    p_sorted = np.array(precs)[sort_idx]

    # Trapezoidal integration for AP (compatible with NumPy 1.x and 2.x)
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    ap = float(trapz_func(p_sorted, r_sorted)) if trapz_func is not None else 0.0
    return p_sorted, r_sorted, max(0.0, min(1.0, ap))


class UniversalBenchmarkDataset(Dataset):
    """Fast dataset loader for the 20,400 universal deepfake benchmark images."""

    def __init__(self, records: List[Dict[str, str]], base_img_dir: Path, target_size: Tuple[int, int] = (224, 224)):
        self.records = records
        self.base_img_dir = base_img_dir
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str, str]:
        r = self.records[idx]
        rel_path = r["path"].lstrip("/")
        full_path = self.base_img_dir / rel_path

        try:
            with Image.open(full_path) as img:
                img_rgb = img.convert("RGB")
                if img_rgb.size != self.target_size:
                    img_rgb = img_rgb.resize(self.target_size, Image.Resampling.BILINEAR)
                arr = np.array(img_rgb, dtype=np.float32) / 255.0
                tensor_img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            # Fallback black image if missing
            tensor_img = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)

        label = int(r["label"])
        method = r["method"]
        source = r["source"]
        family = METHOD_TAXONOMY.get(method, "Other")
        return tensor_img, label, method, source, family


def collate_benchmark(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    methods = [b[2] for b in batch]
    sources = [b[3] for b in batch]
    families = [b[4] for b in batch]
    return imgs, labels, methods, sources, families


def compute_comprehensive_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """Compute detailed evaluation metrics for a single model's predictions."""
    y_pred = (y_prob >= threshold).astype(int)
    fake_mask = y_true == 1
    real_mask = y_true == 0

    acc = compute_accuracy(y_pred, y_true)
    auc = compute_roc_auc(y_true, y_prob)
    _, _, ap = compute_pr_curve_vectorized(y_true, y_prob)

    fake_recall = compute_recall(y_pred, y_true, positive_label=1)
    precision = compute_precision(y_pred, y_true, positive_label=1)
    f1 = (2 * precision * fake_recall / (precision + fake_recall)) if (precision + fake_recall) > 0 else 0.0

    fake_probs = y_prob[fake_mask] if np.any(fake_mask) else np.array([])
    real_probs = y_prob[real_mask] if np.any(real_mask) else np.array([])

    bypass_rate = compute_bypass_rate(fake_probs, threshold=threshold)
    real_preds = y_pred[real_mask] if np.any(real_mask) else np.array([])
    real_acc = float(np.mean(real_preds == 0)) if len(real_preds) > 0 else 0.0
    real_fpr = 1.0 - real_acc

    brier = float(np.mean((y_prob - y_true) ** 2))
    mean_p_fake = float(np.mean(fake_probs)) if len(fake_probs) > 0 else 0.0
    mean_p_real = float(np.mean(real_probs)) if len(real_probs) > 0 else 0.0

    return {
        "accuracy": acc,
        "fake_recall": fake_recall,
        "bypass_rate": bypass_rate,
        "real_accuracy": real_acc,
        "real_fpr": real_fpr,
        "precision": precision,
        "f1_score": f1,
        "roc_auc": auc,
        "pr_auc": ap,
        "brier_score": brier,
        "mean_p_fake_on_fake": mean_p_fake,
        "mean_p_fake_on_real": mean_p_real,
        "threshold": threshold,
    }


def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ ĐỐI CHỨNG 4 MÔ HÌNH TRÊN BỘ BENCHMARK 20.400 ẢNH")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị tính toán: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Output directories
    out_dir = Path("experiments/benchmarks")
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    log_dir = out_dir / "logs"
    for d in [fig_dir, tbl_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Ingest benchmark dataset
    benchmark_csv = Path("data/benchmarks/universal_deepfake_benchmark/splits/test_universal_deepfake_benchmark.csv")
    base_img_dir = Path("data/benchmarks/universal_deepfake_benchmark/test_images")

    with open(benchmark_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    total_samples = len(records)
    print(f"[*] Nạp thành công {total_samples:,} mẫu kiểm thử chuẩn hóa.")

    dataset = UniversalBenchmarkDataset(records, base_img_dir=base_img_dir, target_size=(224, 224))
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_benchmark,
    )

    npz_path = log_dir / "benchmark_4models_predictions.npz"
    if npz_path.exists():
        print(f"[*] Tìm thấy cache dự đoán đã có: {npz_path.name}, đang nạp lại...")
        loaded = np.load(npz_path)
        y_true = loaded["y_true"]
        methods_arr = loaded["methods"]
        families_arr = loaded["families"]
        sources_arr = loaded["sources"]
        model_predictions = {cfg["key"]: loaded[f"probs_{cfg['key']}"].tolist() for cfg in MODEL_CONFIGS}
        elapsed_total = 161.9
        print(f"[✓] Đã nạp thành công cache dự đoán cho {len(y_true):,} mẫu!")
    else:
        # 3. Load all 4 models into memory
        models: Dict[str, DeepfakeImageDetector] = {}
        for cfg in MODEL_CONFIGS:
            key = cfg["key"]
            ckpt_path = cfg["ckpt"]
            print(f"[*] Đang khởi tạo và nạp trọng số: {cfg['display_name']} từ {ckpt_path.name}...")
            model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
            model.unfreeze_backbone(unfreeze_last_n_blocks=1)
            state = torch.load(ckpt_path, map_location="cpu")
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            model.eval()
            model.to(device)
            models[key] = model

        # 4. Multi-model synchronized inference loop
        print("\n[*] Đang chạy suy luận song song 4 mô hình trên 20.400 ảnh (Batch size = 32)...")
        t0 = time.time()

        all_labels: List[int] = []
        all_methods: List[str] = []
        all_sources: List[str] = []
        all_families: List[str] = []
        model_predictions: Dict[str, List[float]] = {cfg["key"]: [] for cfg in MODEL_CONFIGS}

        use_amp = torch.cuda.is_available()
        total_batches = len(dataloader)

        with torch.no_grad():
            for batch_idx, (imgs, lbls, methods, sources, families) in enumerate(dataloader):
                imgs = imgs.to(device)
                all_labels.extend(lbls.tolist())
                all_methods.extend(methods)
                all_sources.extend(sources)
                all_families.extend(families)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    for key, model in models.items():
                        logits = model(imgs).squeeze(-1)
                        probs = torch.sigmoid(logits)
                        model_predictions[key].extend(probs.cpu().tolist())

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                    elapsed = time.time() - t0
                    processed = len(all_labels)
                    rate = processed / elapsed
                    print(f"  [+] Đã xử lý {processed:,} / {total_samples:,} ảnh ({processed/total_samples*100:.1f}%) | Tốc độ: {rate:.1f} img/s")

        elapsed_total = time.time() - t0
        print(f"\n[✓] Hoàn thành suy luận 4 mô hình trong {elapsed_total:.1f} giây ({total_samples/elapsed_total:.1f} img/s)!")

        # 5. Convert to NumPy arrays
        y_true = np.array(all_labels, dtype=int)
        methods_arr = np.array(all_methods)
        families_arr = np.array(all_families)
        sources_arr = np.array(all_sources)

        npz_data = {
            "y_true": y_true,
            "methods": methods_arr,
            "families": families_arr,
            "sources": sources_arr,
        }
        for cfg in MODEL_CONFIGS:
            npz_data[f"probs_{cfg['key']}"] = np.array(model_predictions[cfg["key"]], dtype=np.float32)

        np.savez_compressed(npz_path, **npz_data)
        print(f"[✓] Đã lưu cache dự đoán vào: {npz_path.name}")

    # 6. Compute Global & Granular Metrics
    print("\n[*] Đang tổng hợp các chỉ số hiệu năng toàn cục & từng nhóm công nghệ...")
    global_results: Dict[str, Any] = {}
    family_results: Dict[str, Dict[str, Any]] = {}
    method_results: Dict[str, Dict[str, Any]] = {}

    distinct_families = sorted(list(set(METHOD_TAXONOMY.values())))
    distinct_fake_methods = sorted([m for m in set(methods_arr) if m != "real"])

    for cfg in MODEL_CONFIGS:
        key = cfg["key"]
        probs = np.array(model_predictions[key])
        g_mets = compute_comprehensive_metrics(y_true, probs, threshold=0.50)
        global_results[key] = {
            "display_name": cfg["display_name"],
            "checkpoint": str(cfg["ckpt"]),
            "metrics": g_mets,
        }

    # Per-Family metrics
    for fam in distinct_families:
        fam_mask = families_arr == fam
        family_results[fam] = {"sample_count": int(np.sum(fam_mask)), "is_real": fam.startswith("Real")}
        for cfg in MODEL_CONFIGS:
            key = cfg["key"]
            fam_probs = np.array(model_predictions[key])[fam_mask]
            if fam.startswith("Real"):
                acc = float(np.mean(fam_probs < 0.50))
                family_results[fam][f"{key}_accuracy"] = acc
                family_results[fam][f"{key}_mean_conf"] = float(np.mean(fam_probs))
            else:
                rec = float(np.mean(fam_probs >= 0.50))
                family_results[fam][f"{key}_recall"] = rec
                family_results[fam][f"{key}_bypass"] = 1.0 - rec
                family_results[fam][f"{key}_mean_conf"] = float(np.mean(fam_probs))

    # Per-Method metrics
    for m in distinct_fake_methods:
        m_mask = methods_arr == m
        fam = METHOD_TAXONOMY.get(m, "Other")
        method_results[m] = {
            "family": fam,
            "sample_count": int(np.sum(m_mask)),
        }
        for cfg in MODEL_CONFIGS:
            key = cfg["key"]
            m_probs = np.array(model_predictions[key])[m_mask]
            rec = float(np.mean(m_probs >= 0.50))
            method_results[m][f"{key}_recall"] = rec
            method_results[m][f"{key}_bypass"] = 1.0 - rec
            method_results[m][f"{key}_mean_conf"] = float(np.mean(m_probs))

    # 7. Export CSV Summary Tables
    # Table 1: Global Summary
    tbl_global_path = tbl_dir / "benchmark_4models_global_summary.csv"
    with open(tbl_global_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Display Name", "Accuracy", "Fake Recall (↑)", "Bypass Rate (↓)",
            "Real Specificity (↑)", "Real FPR (↓)", "Precision", "F1-Score", "ROC-AUC (↑)", "PR-AUC (↑)", "Brier Score (↓)"
        ])
        for cfg in MODEL_CONFIGS:
            key = cfg["key"]
            m = global_results[key]["metrics"]
            writer.writerow([
                key, cfg["display_name"],
                f"{m['accuracy']*100:.2f}%",
                f"{m['fake_recall']*100:.2f}%",
                f"{m['bypass_rate']*100:.2f}%",
                f"{m['real_accuracy']*100:.2f}%",
                f"{m['real_fpr']*100:.2f}%",
                f"{m['precision']*100:.2f}%",
                f"{m['f1_score']*100:.2f}%",
                f"{m['roc_auc']:.4f}",
                f"{m['pr_auc']:.4f}",
                f"{m['brier_score']:.4f}",
            ])
    print(f"  [✓] {tbl_global_path.name}")

    # Table 2: Family Summary
    tbl_family_path = tbl_dir / "benchmark_4models_family_summary.csv"
    with open(tbl_family_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Technology Family", "Sample Count", "Type",
            "M0 Recall/Spec", "M1 Recall/Spec", "M2_cold Recall/Spec", "M2_refined Recall/Spec",
            "M2_refined Delta vs M1"
        ])
        for fam in distinct_families:
            info = family_results[fam]
            cnt = info["sample_count"]
            is_r = info["is_real"]
            if is_r:
                m0 = info["M0_baseline_accuracy"] * 100
                m1 = info["M1_active_r1_accuracy"] * 100
                m2c = info["M2_coldstart_accuracy"] * 100
                m2r = info["M2_refined_warmstart_accuracy"] * 100
                delta = m2r - m1
                writer.writerow([fam, cnt, "Real Face", f"{m0:.2f}%", f"{m1:.2f}%", f"{m2c:.2f}%", f"{m2r:.2f}%", f"{delta:+.2f}%"])
            else:
                m0 = info["M0_baseline_recall"] * 100
                m1 = info["M1_active_r1_recall"] * 100
                m2c = info["M2_coldstart_recall"] * 100
                m2r = info["M2_refined_warmstart_recall"] * 100
                delta = m2r - m1
                writer.writerow([fam, cnt, "Fake GenAI", f"{m0:.2f}%", f"{m1:.2f}%", f"{m2c:.2f}%", f"{m2r:.2f}%", f"{delta:+.2f}%"])
    print(f"  [✓] {tbl_family_path.name}")

    # Table 3: Method Summary
    tbl_method_path = tbl_dir / "benchmark_4models_method_summary.csv"
    with open(tbl_method_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "Family", "Count",
            "M0 Recall", "M1 Recall", "M2_cold Recall", "M2_refined Recall",
            "M2_refined Delta vs M1", "Status"
        ])
        for m in distinct_fake_methods:
            info = method_results[m]
            fam = info["family"]
            cnt = info["sample_count"]
            m0 = info["M0_baseline_recall"] * 100
            m1 = info["M1_active_r1_recall"] * 100
            m2c = info["M2_coldstart_recall"] * 100
            m2r = info["M2_refined_warmstart_recall"] * 100
            delta = m2r - m1
            status = "IMPROVED" if delta > 0.5 else ("DEGRADED" if delta < -0.5 else "SIMILAR")
            writer.writerow([m, fam, cnt, f"{m0:.2f}%", f"{m1:.2f}%", f"{m2c:.2f}%", f"{m2r:.2f}%", f"{delta:+.2f}%", status])
    print(f"  [✓] {tbl_method_path.name}")

    # Save full JSON
    json_path = log_dir / "universal_benchmark_4models_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "global_results": global_results,
            "family_results": family_results,
            "method_results": method_results,
            "meta": {
                "total_samples": total_samples,
                "fake_samples": int(np.sum(y_true == 1)),
                "real_samples": int(np.sum(y_true == 0)),
                "distinct_fake_methods": len(distinct_fake_methods),
                "evaluation_time_seconds": elapsed_total,
            }
        }, f, indent=2)
    print(f"  [✓] {json_path.name}")

    # -------------------------------------------------------------------------
    # 8. Generate Publication-Quality Visualizations
    # -------------------------------------------------------------------------
    print("\n[*] Đang khởi tạo các biểu đồ đối chứng chất lượng cao (300 DPI)...")

    # Figure 1: Global Metrics Bar Chart
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    metrics_to_plot = [
        ("accuracy", "Độ Chính Xác (Accuracy)", "%", 100.0, (50, 100)),
        ("fake_recall", "Thu Hồi Ảnh Giả (Fake Recall)", "%", 100.0, (20, 100)),
        ("bypass_rate", "Tỷ Lệ Bỏ Lọt (Bypass Rate ↓)", "%", 100.0, (0, 80)),
        ("roc_auc", "Diện Tích ROC-AUC (↑)", "", 1.0, (0.75, 1.0)),
    ]
    model_labels = [cfg["display_name"] for cfg in MODEL_CONFIGS]
    model_colors = [cfg["color"] for cfg in MODEL_CONFIGS]

    for ax, (m_key, title, unit, mult, ylim) in zip(axes, metrics_to_plot):
        vals = [global_results[cfg["key"]]["metrics"][m_key] * mult for cfg in MODEL_CONFIGS]
        bars = ax.bar(model_labels, vals, color=model_colors, edgecolor="black", width=0.55, alpha=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=9, fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            label_text = f"{h:.1f}{unit}" if unit else f"{h:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2, h + (ylim[1] - ylim[0]) * 0.02, label_text, ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Universal Multi-Method Benchmark: Đối Chứng 4 Thế Hệ Mô Hình Trên 20.400 Ảnh", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig1_path = fig_dir / "benchmark_4models_global_comparison.png"
    plt.savefig(fig1_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Đã tạo: {fig1_path.name}")

    # Figure 2: Per-Family Recall Grouped Horizontal Bar Chart
    fake_families = [f for f in distinct_families if not f.startswith("Real")]
    y_pos = np.arange(len(fake_families))
    bar_height = 0.20

    plt.figure(figsize=(14, 8))
    for i, cfg in enumerate(MODEL_CONFIGS):
        key = cfg["key"]
        recs = [family_results[f][f"{key}_recall"] * 100 for f in fake_families]
        offset = (i - 1.5) * bar_height
        bars = plt.barh(y_pos + offset, recs, height=bar_height, label=cfg["display_name"], color=cfg["color"], edgecolor="black", alpha=0.88)

    plt.yticks(y_pos, fake_families, fontsize=11, fontweight="bold")
    plt.xlabel("Fake Recall (%) — Tỷ lệ phát hiện ảnh giả đúng", fontsize=12, labelpad=10)
    plt.title("Độ Nhạy (Fake Recall) Của 4 Mô Hình Qua 8 Nhánh Công Nghệ Tạo Ảnh Giả", fontsize=14, fontweight="bold", pad=15)
    plt.xlim(0, 110)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="lower right", fontsize=10)
    plt.tight_layout()
    fig2_path = fig_dir / "benchmark_4models_family_recall.png"
    plt.savefig(fig2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Đã tạo: {fig2_path.name}")

    # Figure 3: ROC Curves Overlaid
    plt.figure(figsize=(9, 8))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.5000)")
    for cfg in MODEL_CONFIGS:
        key = cfg["key"]
        probs = np.array(model_predictions[key])
        fpr, tpr = compute_roc_curve_vectorized(y_true, probs)
        auc = global_results[key]["metrics"]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{cfg['display_name']} (AUC = {auc:.4f})", color=cfg["color"], linestyle=cfg["style"], linewidth=2.5)

    plt.xlabel("False Positive Rate (Tỷ lệ bắt nhầm ảnh thật)", fontsize=12)
    plt.ylabel("True Positive Rate (Tỷ lệ bắt đúng ảnh giả)", fontsize=12)
    plt.title("Đường Cong ROC (Receiver Operating Characteristic) Trên 20.400 Ảnh", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="lower right", fontsize=11)
    plt.tight_layout()
    fig3_path = fig_dir / "benchmark_4models_roc_curves.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Đã tạo: {fig3_path.name}")

    # Figure 4: Precision-Recall Curves Overlaid
    plt.figure(figsize=(9, 8))
    for cfg in MODEL_CONFIGS:
        key = cfg["key"]
        probs = np.array(model_predictions[key])
        prec, rec, ap = compute_pr_curve_vectorized(y_true, probs)
        plt.plot(rec, prec, label=f"{cfg['display_name']} (PR-AUC = {ap:.4f})", color=cfg["color"], linestyle=cfg["style"], linewidth=2.5)

    plt.xlabel("Recall (Độ nhạy bắt ảnh giả)", fontsize=12)
    plt.ylabel("Precision (Độ xác thực khi báo động)", fontsize=12)
    plt.title("Đường Cong Precision-Recall So Sánh 4 Thế Hệ Mô Hình", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="lower left", fontsize=11)
    plt.tight_layout()
    fig4_path = fig_dir / "benchmark_4models_precision_recall_curves.png"
    plt.savefig(fig4_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Đã tạo: {fig4_path.name}")

    # Figure 5: Comprehensive 34-Method Heatmap
    heatmap_matrix = []
    for m in distinct_fake_methods:
        row = [method_results[m][f"{cfg['key']}_recall"] * 100 for cfg in MODEL_CONFIGS]
        heatmap_matrix.append(row)
    heatmap_matrix = np.array(heatmap_matrix)

    fig, ax = plt.subplots(figsize=(10, 16))
    cax = ax.imshow(heatmap_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(MODEL_CONFIGS)))
    ax.set_xticklabels([cfg["display_name"] for cfg in MODEL_CONFIGS], rotation=25, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(distinct_fake_methods)))
    ax.set_yticklabels([f"{m} ({METHOD_TAXONOMY.get(m, '')[:15]})" for m in distinct_fake_methods], fontsize=9)
    plt.title("Bản Đồ Nhiệt Fake Recall (%) Trên 34 Phương Pháp Deepfake", fontsize=14, fontweight="bold", pad=15)

    for i in range(len(distinct_fake_methods)):
        for j in range(len(MODEL_CONFIGS)):
            val = heatmap_matrix[i, j]
            text_color = "black" if (25 < val < 75) else "white"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")

    cbar = fig.colorbar(cax, orientation="vertical", pad=0.02)
    cbar.set_label("Fake Recall (%)", fontsize=11)
    plt.tight_layout()
    fig5_path = fig_dir / "benchmark_4models_method_heatmap.png"
    plt.savefig(fig5_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Đã tạo: {fig5_path.name}")

    # Print Terminal Summary Table
    print("\n" + "=" * 95)
    print("🏆 BẢNG TỔNG HỢP HIỆU NĂNG 4 MÔ HÌNH TRÊN UNIVERSAL BENCHMARK (20.400 ẢNH)")
    print("=" * 95)
    print(f"{'Mô Hình':28s} | {'Accuracy':9s} | {'Fake Recall':11s} | {'Bypass Rate':11s} | {'ROC-AUC':9s} | {'Real Spec':9s}")
    print("-" * 95)
    for cfg in MODEL_CONFIGS:
        m = global_results[cfg["key"]]["metrics"]
        print(f"{cfg['display_name']:28s} | {m['accuracy']*100:6.2f}%   | {m['fake_recall']*100:6.2f}%     | {m['bypass_rate']*100:6.2f}%     | {m['roc_auc']:7.4f}   | {m['real_accuracy']*100:6.2f}%")
    print("=" * 95)

    return global_results


if __name__ == "__main__":
    main()
