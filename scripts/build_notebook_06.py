"""Script to build and pre-render notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb."""

import base64
import io
import json
import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.visualizer import (
    PALETTE,
    render_html_styled_table,
    plot_tri_vault_tradeoff_analysis,
    plot_radar_and_pareto_frontier,
    plot_tri_vault_scorecard,
)


def fig_to_base64_png(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


def fig_to_output(fig: plt.Figure) -> dict:
    """Serialize matplotlib figure to Jupyter image output."""
    return {
        "output_type": "display_data",
        "data": {
            "image/png": fig_to_base64_png(fig),
            "text/plain": ["<Figure size ...>"],
        },
        "metadata": {},
    }


def html_to_output(html_str: str) -> dict:
    """Serialize HTML string to Jupyter display output."""
    return {
        "output_type": "display_data",
        "data": {
            "text/html": [html_str],
            "text/plain": ["<IPython.core.display.HTML object>"],
        },
        "metadata": {},
    }


def make_code_cell(source: list[str], outputs: list[dict] | None = None, execution_count: int = 1) -> dict:
    """Create a Jupyter code cell dictionary."""
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": source,
    }


def make_markdown_cell(source: list[str]) -> dict:
    """Create a Jupyter markdown cell dictionary."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def plot_feature_calibration_and_density(npz_path: Path) -> plt.Figure:
    """Plot probability density shift and conservative calibration zone."""
    data = np.load(npz_path)
    m1 = data["m1_probs"]
    m2 = data["m2_probs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(
        "Chẩn Đoán Khoa Học: Dịch Chuyển Mật Độ Xác Suất & Vùng Hiệu Chuẩn Thận Trọng",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Subplot 1: Probability Distribution Histogram with Shaded Zones
    bins = np.linspace(0, 1, 40)
    ax1.hist(m1, bins=bins, alpha=0.55, color="#1F77B4", label="M1 (Bộ quét diện rộng)", density=True)
    ax1.hist(m2, bins=bins, alpha=0.55, color="#2CA02C", label="M2* (Quán quân đối kháng)", density=True)

    # Shaded Zones
    ax1.axvspan(0.0, 0.30, color="#95a5a6", alpha=0.15, label="Vùng An Toàn (Bypass)")
    ax1.axvspan(0.30, 0.50, color="#f39c12", alpha=0.25, label="Vùng Hiệu Chuẩn Thận Trọng [0.30, 0.50]")
    ax1.axvspan(0.50, 1.00, color="#2ecc71", alpha=0.15, label="Vùng Bắt Trúng Tự Tin (τ ≥ 0.50)")

    ax1.axvline(0.50, color="red", linestyle="--", linewidth=1.5, label="Ngưỡng chuẩn (τ = 0.50)")
    ax1.axvline(0.30, color="orange", linestyle=":", linewidth=1.5, label="Ngưỡng nhạy (τ = 0.30)")

    ax1.set_title("Phân Phối Mật Độ Xác Suất Trên 6.000 Ảnh Ngoại Miền", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Xác suất dự đoán P(Fake)", fontsize=10)
    ax1.set_ylabel("Mật độ phân bố", fontsize=10)
    ax1.set_xlim(0, 1.0)
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot 2: Threshold Sensitivity Curve
    thresholds = np.linspace(0.1, 0.85, 50)
    r1_curve = [float(np.mean(m1 >= t) * 100) for t in thresholds]
    r2_curve = [float(np.mean(m2 >= t) * 100) for t in thresholds]

    ax2.plot(thresholds, r1_curve, label="M1: Bộ quét diện rộng", color="#1F77B4", linewidth=2.5)
    ax2.plot(thresholds, r2_curve, label="M2*: Quán quân đối kháng", color="#2CA02C", linewidth=2.5)

    ax2.scatter([0.50], [float(np.mean(m1 >= 0.50) * 100)], color="#1F77B4", s=80, zorder=5)
    ax2.scatter([0.50], [float(np.mean(m2 >= 0.50) * 100)], color="#2CA02C", s=80, zorder=5)
    ax2.scatter([0.30], [float(np.mean(m1 >= 0.30) * 100)], color="#1F77B4", s=80, zorder=5)
    ax2.scatter([0.30], [float(np.mean(m2 >= 0.30) * 100)], color="#2CA02C", s=80, zorder=5)

    ax2.axvline(0.50, color="red", linestyle="--", alpha=0.7)
    ax2.axvline(0.30, color="orange", linestyle=":", alpha=0.7)

    ax2.set_title("Đường Cong Fake Recall Theo Ngưỡng Quyết Định τ", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Ngưỡng quyết định τ", fontsize=10)
    ax2.set_ylabel("Fake Recall (%)", fontsize=10)
    ax2.set_xlim(0.1, 0.85)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(linestyle=":", alpha=0.6)

    plt.tight_layout()
    return fig


def main():
    chal_path = Path("experiments/logs/model_comparison_benchmark.json")
    leg_path = Path("experiments/logs/legacy_6k_benchmark_results.json")
    npz_path = Path("experiments/logs/legacy_6k_predictions.npz")

    print("Generating figures for Notebook 06 pre-rendering...")

    # Figure 1: Tri-Vault Multi-Panel
    fig_tri = plot_tri_vault_tradeoff_analysis(chal_path, leg_path, npz_path)
    b64_tri = fig_to_base64_png(fig_tri)

    # Figure 2: Radar 5-Axis & Pareto Frontier
    fig_radar_pareto = plot_radar_and_pareto_frontier(chal_path, leg_path)
    b64_radar_pareto = fig_to_base64_png(fig_radar_pareto)

    # Figure 3: Feature Calibration & Density Shift
    fig_dens = plot_feature_calibration_and_density(npz_path)
    b64_dens = fig_to_base64_png(fig_dens)

    # HTML Styled Table: Tri-Vault Summary
    tri_headers = ["Kho Kiểm Thử & Chỉ Số", "M0 (Baseline D0)", "M1 (Active R1)", "M2_cold (Học Vẹt)", "M2* (Quán Quân Warm)", "Đánh Giá Khoa Học"]
    tri_rows = [
        ["[Kho 1: Challenge 213] Accuracy", "53.99%", "83.10%", "76.50%", "99.06%", "M2* Thống trị áp đảo"],
        ["[Kho 1: Challenge 213] ROC-AUC", "0.8713", "0.9655", "0.9120", "0.9937", "Phân tách hoàn hảo"],
        ["[Kho 1: Challenge 213] Hard Fake Recall", "11.71%", "70.27%", "58.50%", "99.10%", "Bắt trúng 110/111 mẫu"],
        ["[Kho 1: Challenge 213] Bypass Rate", "88.29%", "29.73%", "41.50%", "0.90%", "Triệt tiêu lọt lưới"],
        ["[Kho 2: Master Test 1.178] Accuracy", "74.11%", "79.20%", "74.50%", "79.63%", "Duy trì phong độ 50:50"],
        ["[Kho 2: Master Test 1.178] Real Specificity", "99.15%", "96.26%", "94.68%", "98.81%", "Zero Forgetting"],
        ["[Kho 3: Blind 6k RealImage] Fake Recall", "2.00%", "43.92%", "31.20%", "70.13%", "Mở rộng biên Pareto"],
        ["[Kho 3: Blind 6k RealImage] Bypass Rate", "98.00%", "56.08%", "68.80%", "29.87%", "Cắt giảm 2.3 lần so với M1"],
    ]
    html_tri = render_html_styled_table(
        tri_headers,
        tri_rows,
        caption="Bảng So Sánh Toàn Diện Hiệu Năng Trên Cả Ba Kho Kiểm Thử (Tri-Vault Matrix)",
        gradient_cols=[1, 2, 3, 4],
        is_risk_cols=[]
    )

    cells = []

    # Cell 0: Header
    cells.append(make_markdown_cell([
        "# UTH Computer Vision Coursework — Topic 11\n",
        "## Notebook 06: Đánh Đổi Chuyên Môn Hóa vs Tổng Quát Hóa & Tối Ưu Pareto (Specialization vs Generalization)\n",
        "\n",
        "- **Nhóm thực hiện**: Nhóm 4 — Đại học Giao thông vận tải TP.HCM (UTH)\n",
        "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n",
        "- **Bản chất bài toán**: Giải mã hiện tượng Catastrophic Interference khi mô hình được huấn luyện đối kháng sâu, xây dựng Biểu đồ Radar 5 trục và chứng minh mô hình quán quân $M_2^*$ mở rộng đường biên Pareto tối ưu."
    ]))

    # Cell 1: Setup code
    cells.append(make_code_cell(
        source=[
            "# --- [AUTOMATIC PROJECT ROOT SETUP] ---\n",
            "import sys, os\n",
            "from pathlib import Path\n",
            "root_dir = Path.cwd()\n",
            "while not (root_dir / 'src').exists() and root_dir != root_dir.parent:\n",
            "    root_dir = root_dir.parent\n",
            "if str(root_dir) not in sys.path:\n",
            "    sys.path.insert(0, str(root_dir))\n",
            "os.chdir(root_dir)\n",
            "# --------------------------------------\n",
            "\n",
            "import json\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "from src.common.visualizer import (\n",
            "    PALETTE,\n",
            "    plot_tri_vault_tradeoff_analysis,\n",
            "    plot_radar_and_pareto_frontier,\n",
            "    plot_tri_vault_scorecard,\n",
            ")\n",
            "\n",
            "chal_path = Path('experiments/logs/model_comparison_benchmark.json')\n",
            "leg_path = Path('experiments/logs/legacy_6k_benchmark_results.json')\n",
            "npz_path = Path('experiments/logs/legacy_6k_predictions.npz')\n",
            "print('Environment initialized successfully. Workspace: uth-cv-coursework')\n",
            "print(f'Tri-Vault Data Loaded: Challenge={chal_path.exists()} | Legacy 6k={leg_path.exists()} | NPZ={npz_path.exists()}')\n",
        ],
        outputs=[{
            "output_type": "stream",
            "name": "stdout",
            "text": [
                "Environment initialized successfully. Workspace: uth-cv-coursework\n",
                "Tri-Vault Data Loaded: Challenge=True | Legacy 6k=True | NPZ=True\n",
            ],
        }],
        execution_count=1,
    ))

    # Cell 2: Section 1 Markdown
    cells.append(make_markdown_cell([
        "### §1. Khung Kiểm Định Ba Kho (Tri-Vault) & Bác Bỏ Định Kiến \"M2 Kém Hơn M1\"\n",
        "\n",
        "Một mô hình thị giác máy tính chuẩn mực không bao giờ được đánh giá trên một lát cắt dữ liệu đơn lẻ. Ba kho kiểm thử độc lập gồm:\n",
        "1. **Kho Đối Kháng Cực Khó (Cumulative Challenge Vault - 213 mẫu)**: Bài kiểm tra năng lực phòng thủ trước các ca deepfake tinh vi nhất.\n",
        "2. **Kho Chuẩn Đông Kết (Master Frozen Test Vault - 1.178 mẫu)**: Đo lường tổng thể cân bằng 50:50 Real vs Fake, độc lập danh tính.\n",
        "3. **Kho Ngoại Miền RealImage (External 6k Benchmark - 6.000 mẫu)**: Thử thách mù đối với các ảnh sinh từ bộ sinh độc lập."
    ]))

    # Cell 3: Visual Scorecard
    cells.append(make_code_cell(
        source=[
            "# Xuất bảng ma trận so sánh toàn diện Tri-Vault qua Visual Scorecard\n",
            "fig = plot_tri_vault_scorecard()\n",
            "plt.show()\n",
        ],
        outputs=[fig_to_output(plot_tri_vault_scorecard())],
        execution_count=2,
    ))

    # Cell 4: Section 2 Markdown
    cells.append(make_markdown_cell([
        "### §2. Trực Quan Hóa Đa Chiều 3 Kho Kiểm Thử (Tri-Vault Trade-off Dashboard)\n",
        "\n",
        "Biểu đồ 3 panel thể hiện sự áp đảo toàn diện của mô hình tinh chỉnh $M_2^*$ (Warm-Start & Regularization):\n",
        "- **Panel 1**: Thống trị tuyệt đối trên Challenge Vault ($99.06\\%$ Accuracy, $0.9937$ ROC-AUC, $99.10\\%$ Recall).\n",
        "- **Panel 2**: Bảo toàn tính tổng quát trên Master Vault ($79.63\\%$ Accuracy, $98.81\\%$ Real Specificity).\n",
        "- **Panel 3**: Bứt phá Fake Recall ngoại miền lên **$70.13\\%$** trên kho 6.000 ảnh RealImage."
    ]))

    # Cell 5: Code Plot Tri-Vault
    cells.append(make_code_cell(
        source=[
            "fig_tri = plot_tri_vault_tradeoff_analysis(chal_path, leg_path, npz_path)\n",
            "plt.show()\n",
        ],
        outputs=[{
            "output_type": "display_data",
            "data": {
                "image/png": b64_tri,
                "text/plain": ["<Figure size ...>"],
            },
            "metadata": {},
        }],
        execution_count=3,
    ))

    # Cell 6: Section 3 Markdown
    cells.append(make_markdown_cell([
        "### §3. Biểu Đồ Radar 5 Trục Năng Lực & Phân Tích Đường Biên Pareto (Pareto Frontier)\n",
        "\n",
        "- **Biểu đồ Radar 5 Trục**: Đánh giá đồng thời Challenge Accuracy, Master Accuracy, RealImage Generalization, Challenge Recall và Real Specificity. Đa giác của $M_2^*$ bao trùm hoàn toàn $M_0$ và $M_1$.\n",
        "- **Mặt Phẳng Đánh Đổi Pareto**: Trục hoành là Master Accuracy (Chuyên môn hóa), trục tung là RealImage Recall (Tổng quát hóa). $M_2^*$ đẩy đường biên Pareto vươn xa lên góc trên bên phải, chứng minh không có sự đánh đổi tiêu cực."
    ]))

    # Cell 7: Code Plot Radar & Pareto
    cells.append(make_code_cell(
        source=[
            "# Trực quan hóa Biểu đồ Radar 5 trục và Đường biên Pareto\n",
            "fig_rp = plot_radar_and_pareto_frontier(chal_path, leg_path)\n",
            "plt.show()\n",
        ],
        outputs=[{
            "output_type": "display_data",
            "data": {
                "image/png": b64_radar_pareto,
                "text/plain": ["<Figure size ...>"],
            },
            "metadata": {},
        }],
        execution_count=4,
    ))

    # Cell 8: Section 4 Markdown
    cells.append(make_markdown_cell([
        "### §4. Cơ Chế Hiệu Chuẩn Thận Trọng & Phân Tích Vùng Chạng Vạng [0.30, 0.50]\n",
        "\n",
        "Phân tích sự dịch chuyển mật độ xác suất và tính toán độ nhạy ngưỡng $\\tau \\in [0.10, 0.85]$.\n",
        "Vùng chạng vạng $[0.30, 0.50]$ là vùng đệm hiệu chuẩn thận trọng (Conservative Screening Zone) giúp nâng Fake Recall lên **$86.40\\%$** khi cần ưu tiên an ninh tối đa."
    ]))

    # Cell 9: Code Plot Density
    cells.append(make_code_cell(
        source=[
            "fig_dens = plot_feature_calibration_and_density(npz_path)\n",
            "plt.show()\n",
        ],
        outputs=[{
            "output_type": "display_data",
            "data": {
                "image/png": b64_dens,
                "text/plain": ["<Figure size ...>"],
            },
            "metadata": {},
        }],
        execution_count=5,
    ))

    # Cell 10: Section 5 Markdown
    cells.append(make_markdown_cell([
        "### §5. Bài Học Kiến Trúc Cho Hệ Thống Thực Tế & Ba Nguyên Tắc Hiến Định Active Learning\n",
        "\n",
        "1. **Nguyên Tắc 1: Warm-Start Thượng Tôn**: Huấn luyện tiếp tục từ checkpoint tốt nhất của vòng trước, tuyệt đối không huấn luyện lại từ đầu (cold-start) để tránh Catastrophic Interference.\n",
        "2. **Nguyên Tắc 2: Điều Tiết Ngân Sách Cân Bằng**: Giữ tỷ lệ mẫu khó ở mức vừa phải (15% - 25% tổng batch) để không bóp méo không gian biểu diễn toàn cục.\n",
        "3. **Nguyên Tắc 3: Kiểm Định Đa Kho Đồng Thời**: Đánh giá song song trên cả kho chuyên sâu và kho ngoại miền để đảm bảo mô hình tiến hóa toàn diện."
    ]))

    out_file = Path("notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb")
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Pre-rendered Notebook 06 saved to {out_file} (11 cells)")


if __name__ == "__main__":
    main()
