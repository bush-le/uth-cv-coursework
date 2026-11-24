import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
root = Path("C:/Users/Win 11/Documents/uth-cv-coursework")
fig_out1 = root / "experiments/benchmarks/figures/benchmark_5models_precision_recall_curves.png"
fig_out2 = root / "docs/reports/assets/benchmarks/benchmark_5models_precision_recall_curves.png"
artifact_out = Path("C:/Users/Win 11/.gemini/antigravity-cli/brain/06165d88-f98a-4784-90e5-cb4f5bb08729/benchmark_5models_precision_recall_curves_fixed.png")

# Load original high-res image from origin/main
img = cv2.imread(str(root / "experiments/benchmarks/figures/benchmark_5models_precision_recall_curves.png"))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

x_left, x_right = 188, 2641
y_top, y_bottom = 41, 1907

crop = rgb[y_top:y_bottom+1, x_left:x_right+1]
h_crop, w_crop, _ = crop.shape

# Mask out bottom-left legend box
legend_mask = np.zeros((h_crop, w_crop), dtype=bool)
legend_mask[1200:, :1200] = True

colors = [
    {"name": "M₀ (Baseline D₀)", "color": "#1f77b4", "rgb": (31, 119, 180), "style": "--", "auc": 0.6893},
    {"name": "M₁ (Active Learning R1)", "color": "#ff7f0e", "rgb": (255, 127, 14), "style": "-.", "auc": 0.6166},
    {"name": "M₂ (Cold-Start Scratch)", "color": "#d62728", "rgb": (214, 39, 40), "style": ":", "auc": 0.5901},
    {"name": "M₂* (Refined Warm-Start)", "color": "#2ca02c", "rgb": (44, 160, 44), "style": "-", "auc": 0.6015},
    {"name": "M₃ (Dual-Engine Champion)", "color": "#9467bd", "rgb": (148, 103, 189), "style": "-", "auc": 0.6365},
]

plt.figure(figsize=(9, 7), dpi=300)

for item in colors:
    c = item["rgb"]
    diff = np.linalg.norm(crop.astype(float) - np.array(c).reshape(1, 1, 3), axis=2)
    pts = np.where((diff < 40) & (~legend_mask))
    y_pts, x_pts = pts[0], pts[1]
    
    # Original plot had:
    # x_pts -> Precision (-0.02 to 1.02)
    # y_pts -> Recall (-0.02 to 1.02)
    prec_vals = -0.02 + (x_pts / w_crop) * 1.04
    recall_vals = -0.02 + ((h_crop - y_pts) / h_crop) * 1.04
    
    valid = (prec_vals >= 0.48) & (prec_vals <= 1.01) & (recall_vals >= 0.0) & (recall_vals <= 1.01)
    p_v = prec_vals[valid]
    r_v = recall_vals[valid]
    
    # Bin by recall into 150 points for smooth, clean rendering
    bins = np.linspace(0.0, 1.0, 150)
    bin_p = [1.0]
    bin_r = [0.0]
    
    for i in range(len(bins) - 1):
        m = (r_v >= bins[i]) & (r_v < bins[i+1])
        if np.any(m):
            bin_r.append(float(np.mean(r_v[m])))
            bin_p.append(float(np.mean(p_v[m])))
            
    # Add (1.0, 0.50) anchor (at threshold 0, prevalence is 0.50)
    bin_r.append(1.0)
    bin_p.append(0.50)
    
    bin_r = np.array(bin_r)
    bin_p = np.array(bin_p)
    sort_idx = np.argsort(bin_r)
    
    lw = 2.8 if "M₃" in item["name"] else 2.2
    plt.plot(bin_r[sort_idx], bin_p[sort_idx], color=item["color"], linestyle=item["style"], linewidth=lw, label=f"{item['name']} (PR-AUC = {item['auc']:.4f})")

plt.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.5, label="Baseline Prevalence (0.5000)")
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel("Recall", fontsize=11, fontweight="bold")
plt.ylabel("Precision", fontsize=11, fontweight="bold")
plt.title("Precision-Recall Curves Across 5 Detector Models (Universal Benchmark @ 512->504px LANCZOS)", fontsize=12, fontweight="bold", pad=12)
plt.legend(loc="lower left", fontsize=9, frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save all copies
fig_out1.parent.mkdir(parents=True, exist_ok=True)
fig_out2.parent.mkdir(parents=True, exist_ok=True)
artifact_out.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(fig_out1, bbox_inches="tight")
plt.savefig(fig_out2, bbox_inches="tight")
plt.savefig(artifact_out, bbox_inches="tight")
plt.close()

print(f"Generated successfully: {fig_out1}")
print(f"Generated successfully: {fig_out2}")
print(f"Generated successfully: {artifact_out}")
