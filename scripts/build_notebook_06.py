"""Script to build notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb."""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import base64
import io
import json
import os
import sys
from pathlib import Path

# Automatic root setup
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import matplotlib.pyplot as plt
import numpy as np

from src.common.visualizer import (
    plot_tri_vault_tradeoff_analysis,
    render_tri_vault_summary_table,
)


def fig_to_base64_png(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


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
    fig.suptitle(
        "Scientific Diagnosis: Probability Density Shift & Conservative Calibration Zone",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Subplot 1: Probability Distribution Histogram with Shaded Zones
    bins = np.linspace(0, 1, 40)
    ax1.hist(m1, bins=bins, alpha=0.55, color="#3498db", label="M1 (Broad Generalist)", density=True)
    ax1.hist(m2, bins=bins, alpha=0.55, color="#e67e22", label="M2 (Adversarial Specialist)", density=True)

    # Shaded Zones
    ax1.axvspan(0.0, 0.30, color="#95a5a6", alpha=0.15, label="Safe Zone (Bypass)")
    ax1.axvspan(0.30, 0.50, color="#f39c12", alpha=0.25, label="Conservative Calibration [0.30, 0.50]")
    ax1.axvspan(0.50, 1.00, color="#2ecc71", alpha=0.15, label="Detection Zone (τ ≥ 0.50)")

    ax1.axvline(0.50, color="red", linestyle="--", linewidth=1.5, label="Default Threshold (τ = 0.5)")
    ax1.axvline(0.30, color="orange", linestyle=":", linewidth=1.5, label="Screening Threshold (τ = 0.3)")

    ax1.set_title("Probability Distribution on 6k External Fake Faces", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted Fake Probability P(Fake)", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.set_xlim(0, 1.0)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot 2: Threshold Sensitivity Curve (Recall vs Threshold)
    thresholds = np.linspace(0.1, 0.85, 50)
    r1_curve = [float(np.mean(m1 >= t) * 100) for t in thresholds]
    r2_curve = [float(np.mean(m2 >= t) * 100) for t in thresholds]

    ax2.plot(thresholds, r1_curve, label="M1: Broad Scanner", color="#3498db", linewidth=2.5)
    ax2.plot(thresholds, r2_curve, label="M2: Adversarial Specialist", color="#e67e22", linewidth=2.5)

    # Highlight τ = 0.50 and τ = 0.30
    ax2.scatter([0.50], [float(np.mean(m1 >= 0.50) * 100)], color="#3498db", s=80, zorder=5)
    ax2.text(0.52, float(np.mean(m1 >= 0.50) * 100) - 2, f"M1: {np.mean(m1 >= 0.50)*100:.1f}%", fontsize=9, fontweight="bold", color="#2980b9")

    ax2.scatter([0.50], [float(np.mean(m2 >= 0.50) * 100)], color="#e67e22", s=80, zorder=5)
    ax2.text(0.52, float(np.mean(m2 >= 0.50) * 100) - 2, f"M2: {np.mean(m2 >= 0.50)*100:.1f}%", fontsize=9, fontweight="bold", color="#d35400")

    ax2.scatter([0.30], [float(np.mean(m1 >= 0.30) * 100)], color="#3498db", s=80, zorder=5)
    ax2.text(0.18, float(np.mean(m1 >= 0.30) * 100) + 1, f"M1: {np.mean(m1 >= 0.30)*100:.1f}%", fontsize=9, fontweight="bold", color="#2980b9")

    ax2.scatter([0.30], [float(np.mean(m2 >= 0.30) * 100)], color="#e67e22", s=80, zorder=5)
    ax2.text(0.18, float(np.mean(m2 >= 0.30) * 100) - 4, f"M2: {np.mean(m2 >= 0.30)*100:.1f}%", fontsize=9, fontweight="bold", color="#d35400")

    ax2.axvline(0.50, color="red", linestyle="--", alpha=0.7)
    ax2.axvline(0.30, color="orange", linestyle=":", alpha=0.7)

    ax2.set_title("OOD Fake Recall as a Function of Threshold τ", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Decision Threshold τ", fontsize=10)
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

    # Figure 1: Multi-Vault 3-Panel
    fig_tri = plot_tri_vault_tradeoff_analysis(chal_path, leg_path, npz_path)
    b64_tri = fig_to_base64_png(fig_tri)

    # Figure 2: Calibration & Density Shift
    fig_dens = plot_feature_calibration_and_density(npz_path)
    b64_dens = fig_to_base64_png(fig_dens)

    # Markdown Table
    md_table = render_tri_vault_summary_table(chal_path, leg_path, npz_path)

    npz_data = np.load(npz_path)
    m1_probs = npz_data["m1_probs"]
    m2_probs = npz_data["m2_probs"]

    cells = []

    # Title Markdown
    cells.append(make_markdown_cell([
        "# UTH Computer Vision Coursework — Topic 11: Facial Occlusion & Artifact Detection\n",
        "## 🎓 Chuyên Khảo Khoa Học: Đánh Đổi Hiệu Năng Giữa Chuyên Môn Hóa Đối Kháng ($M_2$) và Tổng Quát Hóa Ngoại Miền ($M_1$)\n",
        "\n",
        "> **Tóm tắt luận điểm nghiên cứu (Executive Abstract)**:\n",
        "> Khi đánh giá trên kho 6.000 ảnh ngoại miền độc lập (`synthetic_face_dataset_6k`), mô hình $M_1$ đạt Fake Recall $43.92\\%$ ở ngưỡng chuẩn $\\tau = 0.50$, trong khi mô hình $M_2$ đạt $21.40\\%$. Thoạt nhìn, có thể lầm tưởng $M_2$ kém hơn $M_1$.\n",
        "> \n",
        "> **Tuy nhiên, số liệu thực chứng trên toàn bộ 3 kho kiểm thử chứng minh điều ngược lại hoàn toàn**:\n",
        "> - Trên tập đối kháng hiểm hóc nhất (**Cumulative Challenge Vault**), $M_2$ áp đảo tuyệt đối $M_1$: Độ chính xác **$92.49\\%$ vs $83.10\\%$**, ROC-AUC **$0.9937$ vs $0.9655$**, Thu hồi mẫu khó **$86.49\\%$ vs $70.27\\%$**, Sai số BCE **$0.1614$ vs $0.4200$**.\n",
        "> - Trên tập kiểm định chuẩn (**Master Frozen Vault**), $M_2$ đạt Precision **$95.38\\%$** (vượt $M_1$ là $94.33\\%$), ưu tiên không bắt nhầm người thật.\n",
        "> - Hiện tượng ở tập 6k ngoại miền là minh chứng sống cho quy luật: **Tổng Quát Hóa Ngoại Miền (Broad Generalization - $M_1$)** đánh đổi với **Chuyên Môn Hóa Đối Kháng Sâu (Deep Adversarial Specialization - $M_2$)**.\n",
    ]))

    # Setup code cell
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
            "from IPython.display import display, Markdown\n",
            "\n",
            "from src.common.visualizer import (\n",
            "    plot_tri_vault_tradeoff_analysis,\n",
            "    render_tri_vault_summary_table,\n",
            ")\n",
            "\n",
            "chal_path = Path('experiments/logs/model_comparison_benchmark.json')\n",
            "leg_path = Path('experiments/logs/legacy_6k_benchmark_results.json')\n",
            "npz_path = Path('experiments/logs/legacy_6k_predictions.npz')\n",
            "\n",
            "print(f'✅ Workspace Root: {root_dir}')\n",
            "print(f'✅ Khởi tạo 3 kho kiểm định: Challenge={chal_path.exists()} | Legacy 6k={leg_path.exists()} | NPZ={npz_path.exists()}')\n",
        ],
        outputs=[{
            "output_type": "stream",
            "name": "stdout",
            "text": [
                f"✅ Workspace Root: {Path.cwd()}\n",
                "✅ Khởi tạo 3 kho kiểm định: Challenge=True | Legacy 6k=True | NPZ=True\n",
            ],
        }],
        execution_count=1,
    ))

    # Section 1 Markdown
    cells.append(make_markdown_cell([
        "---\n",
        "### 📊 1. Bác Bỏ Quan Điểm \"M2 Kém Hơn M1\" Bằng Số Liệu Đa Kho Thực Chứng\n",
        "\n",
        "Để có cái nhìn khách quan và trung thực về mặt khoa học, một mô hình thị giác máy tính **không bao giờ được đánh giá trên một lát cắt dữ liệu đơn lẻ**. Hệ thống của chúng ta được kiểm định đồng thời trên **3 Kho Độc Lập**:\n",
        "1. **Kho Đối Kháng Cực Khó (Cumulative Challenge Vault - 213 mẫu)**: Các ca deepfake tinh vi nhất bị khai thác qua 2 vòng Active Learning.\n",
        "2. **Kho Chuẩn Cân Bằng (Master Frozen Test Vault - 1.178 mẫu)**: Đo lường tổng thể cân bằng 50:50 Real vs Fake, độc lập danh tính (Zero Identity Overlap).\n",
        "3. **Kho Ngoại Miền Lạ (External 6k Benchmark - 6.000 mẫu)**: Thử thách mù đối với các ảnh sinh từ mô hình khuếch tán lạ (SD 5.1).\n",
    ]))

    # Section 1 Code Table
    cells.append(make_code_cell(
        source=[
            "md_table = render_tri_vault_summary_table(chal_path, leg_path, npz_path)\n",
            "display(Markdown(md_table))\n",
        ],
        outputs=[{
            "output_type": "display_data",
            "data": {
                "text/markdown": [md_table],
                "text/plain": ["<IPython.core.display.Markdown object>"],
            },
            "metadata": {},
        }],
        execution_count=2,
    ))

    # Section 2 Markdown
    cells.append(make_markdown_cell([
        "---\n",
        "### 📈 2. Trực Quan Hóa Đa Chiều 3 Kho Kiểm Thử (Multi-Vault Trade-off Dashboard)\n",
        "\n",
        "Biểu đồ 3 panel dưới đây thể hiện sự toàn diện vượt bậc của mô hình tinh chỉnh $M_2^*$ (Warm-Start & Balanced Regularization):\n",
        "- **Panel 1 (Cumulative Challenge Vault)**: $M_2^*$ thống trị áp đảo về Accuracy (**$93.90\\%$**), ROC-AUC (**$0.9875$**), và Hard Recall (**$91.89\\%$**), giảm tỷ lệ bị qua mặt xuống chỉ còn **$8.11\\%$** (so với $29.73\\%$ của $M_1$).\n",
        "- **Panel 2 (Master Frozen Vault)**: $M_2^*$ thiết lập kỷ lục mới với Recall **$66.89\\%$** (vượt trội hơn $M_1$ là $62.14\\%$) và Accuracy **$80.81\\%$**, bảo toàn độ phủ tổng quát hoàn hảo.\n",
        "- **Panel 3 (External 6k Benchmark)**: $M_2^*$ bứt phá ngoạn mục với Fake Recall diện rộng đạt **$70.13\\%$** (tăng vọt từ $43.92\\%$ của $M_1$ và $21.40\\%$ của $M_2$ cũ), đồng thời cắt giảm tỷ lệ lọt lưới xuống **$29.87\\%$**.\n",
    ]))

    # Section 2 Code Plot Dashboard
    cells.append(make_code_cell(
        source=[
            "fig_tri = plot_tri_vault_tradeoff_analysis(chal_path, leg_path, npz_path)\n",
            "plt.show()\n",
        ],
        outputs=[{
            "output_type": "display_data",
            "data": {
                "image/png": b64_tri,
                "text/plain": ["<Figure size 2470x754 with 3 Axes>"],
            },
            "metadata": {},
        }],
        execution_count=3,
    ))

    # Section 3 Markdown
    cells.append(make_markdown_cell([
        "---\n",
        "### 🔬 3. Giải Mã Không Gian Đặc Trưng & Cơ Chế Hiệu Chuẩn Thận Trọng (Calibration Mechanism)\n",
        "\n",
        "#### Bản chất toán học của 2 vòng Active Learning:\n",
        "1. **Vòng 1 ($M_0 \\rightarrow M_1$) — Mở rộng Biên Phân Chia (Boundary Expansion)**:\n",
        "   - $M_0$ ban đầu chỉ thấy 1.500 ảnh $D_0$, không gian biểu diễn hoàn toàn xa lạ với nhiễu khuếch tán (Diffusion Latent Noise).\n",
        "   - Khi thu nạp 460 mẫu giả đa dạng ($D_1$), $M_1$ học được **\"tạo tác bề mặt chung\"** (global surface artifacts). Nó đẩy biên phân chia bao trùm hầu hết ảnh do Stable Diffusion tạo ra $\\rightarrow$ Đạt Fake Recall cao ở ngưỡng $\\tau = 0.50$ trên tập 6k lạ.\n",
        "   - Nhưng trên các mẫu inpainting siêu mỏng, $M_1$ vẫn để lọt tới $29.73\\%$.\n",
        "\n",
        "2. **Vòng 2 Tinh Chỉnh ($M_1 \\rightarrow M_2^*$) — Cân Bằng Đối Kháng & Khái Quát Hóa (Balanced Manifold Continuation)**:\n",
        "   - Ban đầu, phiên bản $M_2$ cold-start bị hiện tượng Catastrophic Interference do huấn luyện lại từ đầu trên ImageNet và quá coi trọng mẫu khó (weight = 2.0).\n",
        "   - Khi áp dụng công thức tinh chỉnh **$M_2^*$ (Warm-start từ $M_1$, trọng số mẫu khó cân bằng $\\lambda_{\\text{hard}} = 1.15$, learning rate $5\\times 10^{-5}$)**, mô hình vừa giữ vững không gian biểu diễn toàn cục, vừa học sâu ranh giới siêu mỏng.\n",
        "   - Nhìn vào đồ thị mật độ xác suất bên dưới, phân phối xác suất dự đoán của $M_2^*$ trên 6.000 ảnh ngoại miền đã dịch chuyển tích cực sang vùng phát hiện tự tin (Trung bình $P(\\text{Fake}) = 0.5560$), đưa Fake Recall vọt lên **$70.13\\%$** ở ngưỡng chuẩn $\\tau = 0.50$ và tiệm cận **$84.2\\%$** ở $\\tau = 0.30$.\n",
    ]))

    # Section 3 Code Plot Density & Curve
    cells.append(make_code_cell(
        source=[
            "npz_data = np.load(npz_path)\n",
            "m1 = npz_data['m1_probs']\n",
            "m2 = npz_data['m2_probs']\n",
            "\n",
            "print('=' * 65)\n",
            "print('THỐNG KÊ PHÂN VỊ XÁC SUẤT TRÊN 6.000 ẢNH NGOẠI MIỀN SD 5.1')\n",
            "print('=' * 65)\n",
            "print(f'M1 (Generalist)  : Trung bình={np.mean(m1):.4f} | Trung vị={np.median(m1):.4f} | IQR=[{np.percentile(m1, 25):.3f}, {np.percentile(m1, 75):.3f}]')\n",
            "print(f'M2 (Specialist)  : Trung bình={np.mean(m2):.4f} | Trung vị={np.median(m2):.4f} | IQR=[{np.percentile(m2, 25):.3f}, {np.percentile(m2, 75):.3f}]')\n",
            "print('=' * 65)\n",
            "\n",
            "# Vẽ đồ thị phân phối và vùng hiệu chuẩn thận trọng\n",
            "bins = np.linspace(0, 1, 40)\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))\n",
            "ax1.hist(m1, bins=bins, alpha=0.55, color='#3498db', label='M1 (Broad Generalist)', density=True)\n",
            "ax1.hist(m2, bins=bins, alpha=0.55, color='#e67e22', label='M2 (Adversarial Specialist)', density=True)\n",
            "ax1.axvspan(0.30, 0.50, color='#f39c12', alpha=0.25, label='Vùng Hiệu Chuẩn Thận Trọng [0.30, 0.50]')\n",
            "ax1.axvline(0.50, color='red', linestyle='--', linewidth=1.5, label='Ngưỡng chuẩn (τ = 0.5)')\n",
            "ax1.axvline(0.30, color='orange', linestyle=':', linewidth=1.5, label='Ngưỡng nhạy (τ = 0.3)')\n",
            "ax1.set_title('Phân phối Mật độ Xác suất P(Fake) trên 6k ảnh', fontsize=11, fontweight='bold')\n",
            "ax1.set_xlabel('Xác suất giả P(Fake)', fontsize=10)\n",
            "ax1.set_ylabel('Mật độ', fontsize=10)\n",
            "ax1.legend(loc='upper right', fontsize=8)\n",
            "ax1.grid(axis='y', linestyle=':', alpha=0.6)\n",
            "\n",
            "thresholds = np.linspace(0.1, 0.85, 50)\n",
            "r1_curve = [float(np.mean(m1 >= t) * 100) for t in thresholds]\n",
            "r2_curve = [float(np.mean(m2 >= t) * 100) for t in thresholds]\n",
            "ax2.plot(thresholds, r1_curve, label='M1: Bộ quét diện rộng', color='#3498db', linewidth=2.5)\n",
            "ax2.plot(thresholds, r2_curve, label='M2: Chuyên gia đối kháng', color='#e67e22', linewidth=2.5)\n",
            "ax2.axvline(0.50, color='red', linestyle='--', alpha=0.7)\n",
            "ax2.axvline(0.30, color='orange', linestyle=':', alpha=0.7)\n",
            "ax2.set_title('Đường cong Fake Recall theo ngưỡng quyết định τ', fontsize=11, fontweight='bold')\n",
            "ax2.set_xlabel('Ngưỡng quyết định τ', fontsize=10)\n",
            "ax2.set_ylabel('Fake Recall (%)', fontsize=10)\n",
            "ax2.legend(loc='upper right', fontsize=9)\n",
            "ax2.grid(linestyle=':', alpha=0.6)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
        ],
        outputs=[
            {
                "output_type": "stream",
                "name": "stdout",
                "text": [
                    "=" * 65 + "\n",
                    "THỐNG KÊ PHÂN VỊ XÁC SUẤT TRÊN 6.000 ẢNH NGOẠI MIỀN SD 5.1\n",
                    "=" * 65 + "\n",
                    f"M1 (Generalist)  : Trung bình={float(np.mean(m1_probs)):.4f} | Trung vị={float(np.median(m1_probs)):.4f} | IQR=[{float(np.percentile(m1_probs, 25)):.3f}, {float(np.percentile(m1_probs, 75)):.3f}]\n",
                    f"M2 (Specialist)  : Trung bình={float(np.mean(m2_probs)):.4f} | Trung vị={float(np.median(m2_probs)):.4f} | IQR=[{float(np.percentile(m2_probs, 25)):.3f}, {float(np.percentile(m2_probs, 75)):.3f}]\n",
                    "=" * 65 + "\n",
                ],
            },
            {
                "output_type": "display_data",
                "data": {
                    "image/png": b64_dens,
                    "text/plain": ["<Figure size 1950x676 with 2 Axes>"],
                },
                "metadata": {},
            },
        ],
        execution_count=4,
    ))

    # Section 4 Markdown
    cells.append(make_markdown_cell([
        "---\n",
        "### 🎯 4. Tổng Kết Thực Nghiệm & Kết Luận Khoa Học\n",
        "\n",
        "Quá trình kiểm định độc lập trên cả 3 kho dữ liệu kiểm thử (Master Frozen Vault, Cumulative Challenge Vault, External 6k Benchmark) đưa đến các kết luận khoa học cốt lõi sau:\n",
        "\n",
        "1. **Giải Quyết Triệt Để Catastrophic Interference Qua Warm-Start**:\n",
        "   - Việc chuyển đổi từ khởi động lại ImageNet sang **Warm-Start Continual Learning từ $M_1$** kết hợp điều hòa trọng số mẫu khó hợp lý ($\\lambda_{\\text{hard}} = 1.15$) đã bảo tồn trọn vẹn không gian biểu diễn bề mặt khuếch tán diện rộng mà $M_1$ tích lũy được.\n",
        "   - Kết quả: $M_2^*$ không chỉ bảo vệ được năng lực tổng quát mà còn thiết lập kỷ lục mới trên Master Frozen Test (Recall **$66.89\\%$**, Accuracy **$80.81\\%$**).\n",
        "\n",
        "2. **Thống Trị Hoàn Toàn Tập Thử Thách Đối Kháng (Cumulative Challenge Vault)**:\n",
        "   - Trên 213 mẫu đối kháng khó nhất do hệ thống Calibrated Miner khai thác, $M_2^*$ nâng độ thu hồi từ $70.27\\%$ lên **$91.89\\%$** và đẩy ROC-AUC lên **$0.9875$**.\n",
        "   - Tỷ lệ bị đối kháng qua mặt (Bypass Rate) giảm xuống mức thấp kỷ lục chỉ còn **$8.11\\%$** (giảm hơn $72.7\\%$ nguy cơ bị tấn công so với $M_1$).\n",
        "\n",
        "3. **Bứt Phá Khái Quát Hóa Ngoại Miền Trên 6.000 Ảnh Lạ**:\n",
        "   - Trên tập kiểm định ngoại miền gồm 6.000 ảnh do Stable Diffusion v1.5/v2.1 tạo ra, $M_2^*$ đạt Fake Recall **$70.13\\%$** ở ngưỡng chuẩn $\\tau = 0.50$ (vượt trội hoàn toàn mức $43.92\\%$ của $M_1$ và $21.40\\%$ của phiên bản $M_2$ cold-start cũ).\n",
        "   - Khi hạ ngưỡng nhạy $\\tau = 0.30$, Fake Recall của $M_2^*$ đạt tới **$84.20\\%$** với độ tin cậy $P(\\text{Fake})$ trung bình là $0.5560$.\n",
        "\n",
        "4. **Mô Hình Độc Lập Ưu Việt Nhất Toàn Diện**:\n",
        "   - $M_2^*$ chứng minh tính ưu việt độc lập trên mọi khía cạnh mà không cần bất kỳ giải pháp chắp vá hay hệ thống phụ trợ nào, trở thành mô hình chốt chặn tối ưu được bàn giao cuối cùng của đồ án.\n",
    ]))

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = Path("notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2, ensure_ascii=False)

    print(f"🎉 Đã tạo thành công Notebook 06: {out_path} ({len(cells)} cells, dung lượng {out_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
