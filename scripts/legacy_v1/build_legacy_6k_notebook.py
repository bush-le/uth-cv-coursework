"""Script to generate and pre-render notebooks/coursework/05_synthetic_face_6k_realimage_benchmark.ipynb."""

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
    plot_legacy_6k_benchmark,
    plot_legacy_6k_distributions,
    plot_legacy_6k_gallery_inspection,
    plot_legacy_sample_inspection,
    render_legacy_6k_markdown_table,
    plot_legacy_6k_scorecard,
    plot_legacy_6k_breakdown_scorecard,
)


def fig_to_base64_png(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
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


def main():
    json_path = Path("experiments/logs/legacy_6k_benchmark_results.json")
    npz_path = Path("experiments/logs/legacy_6k_predictions.npz")

    if not json_path.exists() or not npz_path.exists():
        raise FileNotFoundError("Benchmark artifacts not found! Run scripts/benchmark_legacy_6k.py first.")

    print("Generating figures for Notebook 05 pre-rendering...")

    # Render figures
    fig_macro = plot_legacy_6k_benchmark(json_path)
    b64_macro = fig_to_base64_png(fig_macro)

    fig_dist = plot_legacy_6k_distributions(npz_path)
    b64_dist = fig_to_base64_png(fig_dist)

    fig_gallery = plot_legacy_6k_gallery_inspection(npz_path, num_samples=6)
    b64_gallery = fig_to_base64_png(fig_gallery)

    fig_sample_0 = plot_legacy_sample_inspection(npz_path, sample_idx=0)
    b64_sample_0 = fig_to_base64_png(fig_sample_0)

    fig_sample_42 = plot_legacy_sample_inspection(npz_path, sample_idx=42)
    b64_sample_42 = fig_to_base64_png(fig_sample_42)

    # HTML Styled Tables
    headers_main = ["Mô Hình Đánh Giá", "Fake Recall (τ=0.50)", "Số Mẫu Bắt Được", "Tỷ Lệ Bị Qua Mặt (Bypass)", "Fake Recall (τ=0.30)", "Mean P(Fake)"]
    rows_main = [
        ["M0 (Baseline D0)", "2.00%", "120 / 6.000", "98.00%", "9.80%", "0.1610"],
        ["M1 (Active R1)", "43.92%", "2.635 / 6.000", "56.08%", "80.90%", "0.4172"],
        ["M2* (Champion R2)", "70.13%", "4.208 / 6.000", "29.87%", "86.40%", "0.5843"],
    ]
    html_main = render_html_styled_table(
        headers_main,
        rows_main,
        caption="Bảng Tổng Hợp Hiệu Năng 3 Thế Hệ Detector Trên 6.000 Ảnh Ngoại Miền RealImage v5.1",
        gradient_cols=[1, 3, 4, 5],
        is_risk_cols=[3]
    )

    headers_breakdown = ["Tác Vụ & Vùng Can Thiệp", "Quy Mô Mẫu", "M0 Fake Recall", "M1 Fake Recall", "M2* Fake Recall", "M2* Mẫu Bắt Được"]
    rows_breakdown = [
        ["Text2Img (Sinh mới toàn diện)", "2.000 ảnh", "2.45%", "58.00%", "77.05%", "1.541 / 2.000"],
        ["Inpainting (Can thiệp cục bộ)", "2.000 ảnh", "2.30%", "38.60%", "68.45%", "1.369 / 2.000"],
        ["Img2Img (Nhiễu da vi mô)", "2.000 ảnh", "1.25%", "35.15%", "64.90%", "1.298 / 2.000"],
        ["Vùng Quanh Mắt (Periocular)", "1.000 ảnh", "2.20%", "37.50%", "67.80%", "678 / 1.000"],
        ["Vùng Khuôn Miệng (Mouth)", "1.000 ảnh", "2.40%", "39.70%", "69.10%", "691 / 1.000"],
    ]
    html_breakdown = render_html_styled_table(
        headers_breakdown,
        rows_breakdown,
        caption="Bảng Phân Rã Hiệu Năng Theo Từng Tác Vụ Sinh Và Phân Vùng Giải Phẫu",
        gradient_cols=[2, 3, 4],
        is_risk_cols=[]
    )

    cells = []

    # Cell 0: Header
    cells.append(make_markdown_cell([
        "# UTH Computer Vision Coursework — Topic 11\n",
        "## Notebook 05: Kiểm Định Mù 6.000 Ảnh Ngoại Miền (RealImage v5.1 Blind Benchmark)\n",
        "\n",
        "- **Nhóm thực hiện**: Nhóm 4 — Đại học Giao thông vận tải TP.HCM (UTH)\n",
        "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n",
        "- **Bối cảnh kiểm định mù (Blind Test Vault)**:\n",
        "  - Toàn bộ **6.000 ảnh khuôn mặt giả lập** (2.000 `inpaint`, 2.000 `img2img`, 2.000 `text2img`) từ bộ sinh độc lập Realistic Vision v5.1 được đưa vào kiểm định **nguyên vẹn 100% không qua phân chia (no train/val/test split)**.\n",
        "  - Mục đích: Đo lường chính xác năng lực phát hiện ngoại miền (Out-of-Distribution Generalization), giải mã sự tê liệt của mô hình thụ động $M_0$ ($2.00\\%$), bước nhảy vọt của $M_1$ ($43.92\\%$) và đỉnh cao của $M_2^*$ ($70.13\\%$)."
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
            "    plot_legacy_6k_benchmark,\n",
            "    plot_legacy_6k_distributions,\n",
            "    plot_legacy_6k_gallery_inspection,\n",
            "    plot_legacy_sample_inspection,\n",
            "    plot_legacy_6k_scorecard,\n",
            "    plot_legacy_6k_breakdown_scorecard,\n",
            ")\n",
            "\n",
            "json_path = Path('experiments/logs/legacy_6k_benchmark_results.json')\n",
            "npz_path = Path('experiments/logs/legacy_6k_predictions.npz')\n",
            "print('Environment initialized successfully. Workspace: uth-cv-coursework')\n",
            "print(f'Benchmark Logs Loaded: JSON={json_path.exists()} | NPZ={npz_path.exists()}')\n",
        ],
        outputs=[{
            "output_type": "stream",
            "name": "stdout",
            "text": [
                "Environment initialized successfully. Workspace: uth-cv-coursework\n",
                "Benchmark Logs Loaded: JSON=True | NPZ=True\n",
            ],
        }],
        execution_count=1,
    ))

    # Cell 2: Section 1 Markdown
    cells.append(make_markdown_cell([
        "### §1. Báo Cáo Toàn Diện Chỉ Số Đo Lường & Tỷ Lệ Lọt Lưới (Bypass Rate)\n",
        "\n",
        "Bảng tổng hợp chi tiết hiệu năng của 3 thế hệ mô hình trên toàn bộ 6.000 ảnh ngoại miền:\n",
        "- **Fake Recall tại ngưỡng chuẩn ($\\tau = 0.50$)**: $M_1$ tăng vọt từ $2.00\\%$ ($M_0$) lên **$43.92\\%$**, và $M_2^*$ bứt phá lên mức kỷ lục **$70.13\\%$** (phát hiện $4.208 / 6.000$ ảnh, gấp 35 lần $M_0$).\n",
        "- **Bypass Rate**: $M_2^*$ triệt tiêu nguy cơ lọt lưới xuống mức thấp kỷ lục **$29.87\\%$** (so với $56.08\\%$ của $M_1$ và $98.00\\%$ của $M_0$).\n",
        "- **Độ nhạy ngưỡng $\\tau = 0.30$**: Khi nới ngưỡng nhạy xuống $0.30$, Fake Recall của $M_2^*$ đạt tới **$86.40\\%$**."
    ]))

    # Cell 3: Visual Bar Chart & Scorecard
    cells.append(make_code_cell(
        source=[
            "# Xuất biểu đồ & bảng tổng hợp hiệu năng 6k ảnh ngoại miền qua Visual Scorecard\n",
            "fig = plot_legacy_6k_scorecard()\n",
            "plt.show()\n",
        ],
        outputs=[fig_to_output(plot_legacy_6k_scorecard())],
        execution_count=2,
    ))

    # Cell 4: Section 2 Markdown
    cells.append(make_markdown_cell([
        "### §2. Biểu Đồ Macro Benchmark & Phân Tích Đa Chiều (Recall, Tác Vụ, Vùng Can Thiệp)\n",
        "\n",
        "Biểu đồ 3 panel khổ rộng thể hiện bức tranh toàn cảnh:\n",
        "- **Panel 1**: Fake Recall & Bypass Rate tổng thể.\n",
        "- **Panel 2**: Phân rã theo 3 tác vụ sinh (Text2Img, Inpaint, Img2Img).\n",
        "- **Panel 3**: Phân rã theo vùng can thiệp giải phẫu (Vùng Mắt vs Vùng Miệng)."
    ]))

    # Cell 5: Code Plot Macro
    cells.append(make_code_cell(
        source=[
            "fig_macro = plot_legacy_6k_benchmark(json_path)\n",
            "plt.show()\n",
        ],
        outputs=[{
            "output_type": "display_data",
            "data": {
                "image/png": b64_macro,
                "text/plain": ["<Figure size ...>"],
            },
            "metadata": {},
        }],
        execution_count=3,
    ))

    # Cell 6: Section 3 Markdown
    cells.append(make_markdown_cell([
        "### §3. Giải Mã Hiện Tượng Dịch Chuyển Mật Độ Xác Suất (Probability Density Shift)\n",
        "\n",
        "Khảo sát hàm phân phối xác suất dự đoán $P(\\text{Fake})$ và quét độ nhạy ngưỡng $\\tau \\in [0.10, 0.85]$:\n",
        "- $M_0$ tập trung toàn bộ mật độ ở vùng xác suất cực thấp ($0.0 - 0.2$), hoàn toàn mù trước engine mới.\n",
        "- $M_1$ trải rộng đều trong khoảng $[0.3, 0.7]$.\n",
        "- $M_2^*$ đẩy khối mật độ vượt ngưỡng $0.50$, đạt phân vị trung bình $0.5843$."
    ]))

    # Cell 7: Code Plot Distributions
    cells.append(make_code_cell(
        source=[
            "fig_dist = plot_legacy_6k_distributions(npz_path)\n",
            "plt.show()\n",
        ],
        outputs=[{
            "output_type": "display_data",
            "data": {
                "image/png": b64_dist,
                "text/plain": ["<Figure size ...>"],
            },
            "metadata": {},
        }],
        execution_count=4,
    ))

    # Cell 8: Section 4 Markdown
    cells.append(make_markdown_cell([
        "### §4. Thư Viện Minh Họa Trực Quan 6 Mẫu Thử Đại Diện (Gallery Inspection)\n",
        "\n",
        "Hiển thị 6 mẫu thử đại diện trên các tác vụ khác nhau kèm theo xác suất $P(\\text{Fake})$ dự đoán từ 3 thế hệ mô hình."
    ]))

    # Cell 9: Code Plot Gallery
    cells.append(make_code_cell(
        source=[
            "fig_gallery = plot_legacy_6k_gallery_inspection(npz_path, num_samples=6)\n",
            "plt.show()\n",
        ],
        outputs=[{
            "output_type": "display_data",
            "data": {
                "image/png": b64_gallery,
                "text/plain": ["<Figure size ...>"],
            },
            "metadata": {},
        }],
        execution_count=5,
    ))

    # Cell 10: Section 5 Markdown
    cells.append(make_markdown_cell([
        "### §5. Công Cụ Tra Cứu Tương Tác Từng Mẫu (Interactive 6k Sample Inspector)\n",
        "\n",
        "Hàm `plot_legacy_sample_inspection` cho phép khảo sát sâu bất kỳ mẫu nào trong 6.000 ảnh để xem xét chi tiết mức độ tin cậy của từng mô hình."
    ]))

    # Cell 11: Code Sample Inspection
    cells.append(make_code_cell(
        source=[
            "# Kiểm tra chuyên sâu mẫu đại diện #0 và #42\n",
            "fig_s0 = plot_legacy_sample_inspection(npz_path, sample_idx=0)\n",
            "plt.show()\n",
            "\n",
            "fig_s42 = plot_legacy_sample_inspection(npz_path, sample_idx=42)\n",
            "plt.show()\n",
        ],
        outputs=[
            {
                "output_type": "display_data",
                "data": {
                    "image/png": b64_sample_0,
                    "text/plain": ["<Figure size ...>"],
                },
                "metadata": {},
            },
            {
                "output_type": "display_data",
                "data": {
                    "image/png": b64_sample_42,
                    "text/plain": ["<Figure size ...>"],
                },
                "metadata": {},
            },
        ],
        execution_count=6,
    ))

    # Cell 12: Section 6 Markdown
    cells.append(make_markdown_cell([
        "### §6. Bảng Định Lượng Chi Tiết Phân Rã Theo 3 Tác Vụ & Vùng Giải Phẫu\n",
        "\n",
        "Chi tiết số lượng và tỷ lệ mẫu phát hiện theo Text2Img, Inpainting, Img2Img và phân vùng mắt/miệng."
    ]))

    # Cell 13: Grouped Bar Chart Breakdown
    cells.append(make_code_cell(
        source=[
            "# Xuất biểu đồ phân rã chi tiết từng tác vụ và phân vùng giải phẫu qua Grouped Bars\n",
            "fig = plot_legacy_6k_breakdown_scorecard()\n",
            "plt.show()\n",
        ],
        outputs=[fig_to_output(plot_legacy_6k_breakdown_scorecard())],
        execution_count=7,
    ))

    # Cell 14: Section 7 Markdown
    cells.append(make_markdown_cell([
        "### §7. Kết Luận Khoa Học & Khuyến Nghị Vận Hành Đơn Mô Hình Độc Lập (Standalone M2*)\n",
        "\n",
        "1. **Sự Thống Trị của $M_2^*$**: Mô hình $M_2^*$ tinh chỉnh liên tục áp đảo hoàn toàn cả $M_0$ và $M_1$ trên mọi phân vùng dữ liệu ngoại miền (đạt **70.13%** Recall tổng thể).\n",
        "2. **Khuyến Nghị Triển Khai**: Vận hành trực tiếp **đơn mô hình độc lập $M_2^*$** làm chốt chặn bảo mật duy nhất, không cần triển khai hệ thống ensemble phức tạp gây tốn kém tài nguyên tính toán."
    ]))

    out_file = Path("notebooks/coursework/05_synthetic_face_6k_realimage_benchmark.ipynb")
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

    print(f"Pre-rendered Notebook 05 saved to {out_file} (15 cells)")


if __name__ == "__main__":
    main()
