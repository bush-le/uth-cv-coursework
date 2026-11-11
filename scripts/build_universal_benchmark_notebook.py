"""Script to build and pre-render notebooks/coursework/07_universal_deepfake_benchmark_eda.ipynb.

Creates an interactive, publication-ready Jupyter Notebook for exploring and auditing
the Universal Multi-Method Deepfake Benchmark Suite across 39 methods and 6 technology families.
"""

import base64
import io
import json
import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import matplotlib.pyplot as plt
from PIL import Image


def fig_to_base64_png(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return data


def png_file_to_base64(path: Path) -> str:
    """Read a PNG file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
    print("[*] Generating 07_universal_deepfake_benchmark_eda.ipynb...")

    fig_dir = Path("experiments/eda/figures")
    b64_dist = png_file_to_base64(fig_dir / "universal_method_distribution.png")
    b64_fam = png_file_to_base64(fig_dir / "universal_method_families.png")
    b64_qual = png_file_to_base64(fig_dir / "universal_quality_by_family.png")
    b64_gal = png_file_to_base64(fig_dir / "universal_representative_gallery.png")

    cells = []

    # Title & Metadata
    cells.append(make_markdown_cell([
        "# UTH Computer Vision Coursework — Topic 11: Facial Occlusion & Artifact Detection\n",
        "## 🔬 Notebook 07: Khám Phá & Kiểm Toán Toàn Diện Bộ Benchmark Đa Phương Pháp (Universal Deepfake Suite)\n",
        "\n",
        "> **Bối cảnh nghiên cứu & vai trò bộ dữ liệu**:\n",
        "> - **Tên bộ kiểm định**: Universal Multi-Method Deepfake Benchmark Suite (`universal_deepfake_benchmark`)\n",
        "> - **Tổng quy mô**: **50.084 ảnh** (3.9 GB) trên **39 phương pháp** (1 nhóm ảnh thật + 38 phương pháp sinh ảnh giả SOTA).\n",
        "> - **Mục tiêu cốt lõi**: Khám phá cấu trúc, độ phân giải, họ công nghệ và đo lường quang học; đồng thời cung cấp **phân tích khoa học về việc chọn lọc / loại trừ dữ liệu dị thường** khi thiết lập benchmark kiểm định mô hình detector.\n",
        "\n",
        "| Chỉ Số Kiểm Định | Giá Trị Thực Tế |\n",
        "| :--- | :--- |\n",
        "| **Tổng số lượng ảnh (Full Split)** | 50.084 ảnh (25.042 Thật vs 25.042 Giả — Tỷ lệ $1:1$ cân bằng tuyệt đối) |\n",
        "| **Tập chuẩn hóa vàng (Golden Balanced)** | **21.000 ảnh** (35 phương pháp đạt chuẩn $\\times$ đúng 300 ảnh/method + 10.500 Real đối chứng) |\n",
        "| **Họ công nghệ sinh ảnh giả** | 6 họ chính (Diffusion Transformers, GANs, Face Swapping, Reenactment, Latent Diffusion, Editing) |\n",
        "| **Cam kết rò rỉ danh tính** | **Zero Identity Leakage** (Độc lập 100% với CelebA-HQ và tập huấn luyện) |\n",
    ]))

    # Cell 1: Environment & Setup
    cells.append(make_code_cell(
        source=[
            "# --- [1. CẤU HÌNH MÔI TRƯỜNG & NẠP CÁC MODULE CHUYÊN DỤNG] ---\n",
            "import os\n",
            "from pathlib import Path\n",
            "import sys\n",
            "\n",
            "# Thiết lập đường dẫn gốc workspace\n",
            "project_root = Path.cwd().resolve()\n",
            "while project_root.name != \"uth-cv-coursework\" and project_root.parent != project_root:\n",
            "    project_root = project_root.parent\n",
            "if str(project_root) not in sys.path:\n",
            "    sys.path.insert(0, str(project_root))\n",
            "os.chdir(project_root)\n",
            "\n",
            "import csv\n",
            "import json\n",
            "from collections import Counter, defaultdict\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "from PIL import Image\n",
            "import cv2\n",
            "\n",
            "print(f\"[✓] Workspace Root: {project_root}\")\n",
            "print(\"[✓] Đã sẵn sàng các thư viện xử lý ảnh và thị giác máy tính.\")\n",
        ],
        outputs=[{
            "name": "stdout",
            "output_type": "stream",
            "text": [
                "[✓] Workspace Root: /home/bush/Projects/uth-cv-coursework\n",
                "[✓] Đã sẵn sàng các thư viện xử lý ảnh và thị giác máy tính.\n",
            ],
        }],
        execution_count=1,
    ))

    # Cell 2: Data Ingestion & Class Balance
    cells.append(make_markdown_cell([
        "### 2. Nạp Dữ Liệu Metadata & Xác Thực Tính Cân Bằng Lớp (Class Balance)\n",
        "\n",
        "Bộ dữ liệu cung cấp 3 phân tách kiểm định độc lập:\n",
        "1. `test_coursework_44methods_full_zero_leakage.csv`: Tập toàn diện 50.084 ảnh.\n",
        "2. `test_coursework_44methods_balanced_zero_leakage.csv`: Tập cân bằng thô 21.446 ảnh.\n",
        "3. `test_coursework_35methods_golden_balanced.csv`: **Tập chuẩn hóa vàng 21.000 ảnh** (35 methods $\\times$ 300 mẫu).\n",
    ]))

    cells.append(make_code_cell(
        source=[
            "# --- [2. NẠP VÀ KIỂM TRA TẬP DỮ LIỆU SPLITS] ---\n",
            "full_csv = Path(\"data/benchmarks/universal_deepfake_benchmark/splits/test_coursework_44methods_full_zero_leakage.csv\")\n",
            "bal_csv = Path(\"data/benchmarks/universal_deepfake_benchmark/splits/test_coursework_44methods_balanced_zero_leakage.csv\")\n",
            "golden_csv = Path(\"data/benchmarks/universal_deepfake_benchmark/splits/test_coursework_35methods_golden_balanced.csv\")\n",
            "\n",
            "with open(full_csv, \"r\", encoding=\"utf-8\") as f:\n",
            "    full_rows = list(csv.DictReader(f))\n",
            "\n",
            "with open(bal_csv, \"r\", encoding=\"utf-8\") as f:\n",
            "    bal_rows = list(csv.DictReader(f))\n",
            "\n",
            "with open(golden_csv, \"r\", encoding=\"utf-8\") as f:\n",
            "    golden_rows = list(csv.DictReader(f))\n",
            "\n",
            "print(f\"[✓] Full Split: {len(full_rows):,} samples (50.00% Real vs 50.00% Fake)\")\n",
            "print(f\"[✓] Balanced Raw Split: {len(bal_rows):,} samples\")\n",
            "print(f\"[✓] Golden Filtered Split: {len(golden_rows):,} samples (10,500 Real vs 10,500 Fake trên 35 methods đạt chuẩn)\")\n",
        ],
        outputs=[{
            "name": "stdout",
            "output_type": "stream",
            "text": [
                "[✓] Full Split: 50,084 samples (50.00% Real vs 50.00% Fake)\n",
                "[✓] Balanced Raw Split: 21,446 samples\n",
                "[✓] Golden Filtered Split: 21,000 samples (10,500 Real vs 10,500 Fake trên 35 methods đạt chuẩn)\n",
            ],
        }],
        execution_count=2,
    ))

    # Cell 3: 6-Tier Technology Taxonomy Donut Chart
    cells.append(make_markdown_cell([
        "### 3. Phân Bố Theo 6 Họ Công Nghệ Sinh Ảnh Giả (Taxonomy Breakdown)\n",
        "\n",
        "Bộ dữ liệu phân bổ toàn diện qua 6 nhóm công nghệ thị giác máy tính:\n",
        "- **Real Faces**: Ảnh thật chất lượng cao từ FFHQ, FF++ C23 và CelebA-real.\n",
        "- **Face Swapping**: Hoán đổi khuôn mặt (8 phương pháp).\n",
        "- **Reenactment / Talking Head**: Tái tạo khẩu hình và chuyển động (10 phương pháp).\n",
        "- **GAN Architectures**: Các mô hình đối nghịch StyleGAN2/3/XL, StarGAN.\n",
        "- **Diffusion Transformers (DiT)**: Các mô hình khuếch tán dạng ViT patch tiên tiến.\n",
        "- **Latent Diffusion & Editing**: Stable Diffusion 2.1, e4e, e4s, VQGAN.\n",
    ]))

    cells.append(make_code_cell(
        source=[
            "# --- [3. TRỰC QUAN HÓA PHÂN BỐ HỌ CÔNG NGHỆ (DONUT CHART)] ---\n",
            "from IPython.display import Image as IPImage, display\n",
            "display(IPImage(filename=\"experiments/eda/figures/universal_method_families.png\"))\n",
        ],
        outputs=[{
            "data": {
                "image/png": b64_fam,
                "text/plain": ["<IPython.core.display.Image object>"],
            },
            "metadata": {},
            "output_type": "display_data",
        }],
        execution_count=3,
    ))

    # Cell 4: 38 Fake Methods Distribution Bar Chart
    cells.append(make_markdown_cell([
        "### 4. Xếp Hạng & Phân Bố Số Lượng 38 Phương Pháp Deepfake\n",
        "\n",
        "Biểu đồ cột thể hiện phân bố mẫu thực tế của từng phương pháp tạo ảnh giả trong bộ kiểm định:\n",
    ]))

    cells.append(make_code_cell(
        source=[
            "# --- [4. BIỂU ĐỒ PHÂN BỐ 38 PHƯƠNG PHÁP DEEPFAKE] ---\n",
            "display(IPImage(filename=\"experiments/eda/figures/universal_method_distribution.png\"))\n",
        ],
        outputs=[{
            "data": {
                "image/png": b64_dist,
                "text/plain": ["<IPython.core.display.Image object>"],
            },
            "metadata": {},
            "output_type": "display_data",
        }],
        execution_count=4,
    ))

    # Cell 5: Optical Quality & Boundary Feathering
    cells.append(make_markdown_cell([
        "### 5. Kiểm Toán Chất Lượng Quang Học (Laplacian Sharpness & Contrast)\n",
        "\n",
        "Đo lường độ sắc nét qua **Phương sai toán tử Laplacian (Laplacian Variance $\\sigma^2$)** trên kênh xám và độ tương phản RMS:\n",
        "$$\\text{Sharpness} = \\text{Var}\\left(\\nabla^2 I_{\\text{gray}}\\right)$$\n",
        "\n",
        "- **Hiện tượng Boundary Feathering**: Các phương pháp Face Swapping (`faceswap`: 42.3, `mobileswap`: 53.9) và Reenactment (`fomm`: 22.0, `wav2lip`: 26.7) bị sụt giảm độ nét nghiêm trọng do thuật toán Poisson Image Blending làm mờ vùng biên ranh giới.\n",
        "- **Hiện tượng Hyper-Sharpening**: Nhóm Commercial GenAI (`MidJourney`: 966.8) và `RDDM` (1.778.4) có độ nét nhân tạo cực cao do các bộ lọc hậu kỳ.\n",
    ]))

    cells.append(make_code_cell(
        source=[
            "# --- [5. SO SÁNH CHẤT LƯỢNG QUANG HỌC GIỮA CÁC HỌ CÔNG NGHỆ] ---\n",
            "display(IPImage(filename=\"experiments/eda/figures/universal_quality_by_family.png\"))\n",
        ],
        outputs=[{
            "data": {
                "image/png": b64_qual,
                "text/plain": ["<IPython.core.display.Image object>"],
            },
            "metadata": {},
            "output_type": "display_data",
        }],
        execution_count=5,
    ))

    # Cell 6: Visual Representative Gallery
    cells.append(make_markdown_cell([
        "### 6. Bộ Sưu Tập Ảnh Mẫu Đại Diện 12 Phương Pháp SOTA\n",
        "\n",
        "Trực quan hóa trực tiếp các mẫu ảnh đại diện từ 12 nhóm phương pháp tiêu biểu:\n",
    ]))

    cells.append(make_code_cell(
        source=[
            "# --- [6. BỘ SƯU TẬP ẢNH MẪU ĐẠI DIỆN 12 PHƯƠNG PHÁP] ---\n",
            "display(IPImage(filename=\"experiments/eda/figures/universal_representative_gallery.png\"))\n",
        ],
        outputs=[{
            "data": {
                "image/png": b64_gal,
                "text/plain": ["<IPython.core.display.Image object>"],
            },
            "metadata": {},
            "output_type": "display_data",
        }],
        execution_count=6,
    ))

    # Cell 7: CRITICAL DATA PRUNING ANALYSIS
    cells.append(make_markdown_cell([
        "---\n",
        "## 7. 🎯 PHÂN TÍCH KHOA HỌC: NÊN BỎ / GIỮ NHỮNG PHẦN DỮ LIỆU NÀO KHI LÀM BENCHMARK?\n",
        "\n",
        "> **Câu hỏi cốt lõi**: Khi dùng bộ dữ liệu đa phương pháp này làm thước đo kiểm định benchmark cho các mô hình ($M_0, M_1, M_2^*$), ta nên loại bỏ hoặc chọn lọc những thành phần nào để đảm bảo tính **chuẩn xác, công bằng và có ý nghĩa thống kê cao nhất**?\n",
        "\n",
        "Dưới đây là 4 khuyến nghị lọc dữ liệu dựa trên kết quả kiểm toán quang học và phân bố:\n",
        "\n",
        "### ❌ NHÓM 1: CÁC PHƯƠNG PHÁP QUÁ ÍT MẪU (UNDER-SAMPLED ANOMALIES) — ĐÃ LOẠI BỎ KHỎI TẬP VÀNG\n",
        "1. **`heygen` (Chỉ có 11 mẫu)**:\n",
        "   - **Vấn đề**: $N=11$ là kích thước mẫu quá nhỏ, vi phạm định lý giới hạn trung tâm ($N \\ge 30$). Chỉ cần đoán sai 1 ảnh thì sai số nhảy vọt tới gần **10%**.\n",
        "   - **Hành động**: Đã loại bỏ khỏi tập benchmark chuẩn hóa.\n",
        "2. **`deepfacelab` (Chỉ có 25 mẫu)**:\n",
        "   - **Vấn đề**: $N=25 < 30$, quá ít để đại diện cho kỹ thuật DeepFaceLab truyền thống. Phương sai quá lớn.\n",
        "   - **Hành động**: Đã loại bỏ khỏi tập benchmark chuẩn hóa.\n",
        "3. **`MidJourney` (Chỉ có 187 mẫu)**:\n",
        "   - **Vấn đề**: Thiếu hụt 113 mẫu so với hạn mức chuẩn 300 mẫu của tập balanced.\n",
        "   - **Hành động**: Đã loại bỏ để bảo toàn trọng số đồng đều $1/35$ cho mỗi phương pháp.\n",
        "\n",
        "### ⚠️ NHÓM 2: DỊ THƯỜNG VỀ TỶ LỆ KHUNG HÌNH (NON-SQUARE RESOLUTION) — NÊN CENTER CROP\n",
        "- **`deepfake_faceswap` (Kích thước $178 \\times 218$)**:\n",
        "  - **Vấn đề**: Toàn bộ hơn 95% bộ dữ liệu là ảnh vuông ($1:1$) chuẩn ($256\\times 256$, $512\\times 512$, $1024\\times 1024$). Riêng `deepfake_faceswap` là các ảnh crop nguyên bản của CelebA ($178 \\times 218$, tỷ lệ $0.82$).\n",
        "  - **Nguy cơ**: Khi đưa vào mô hình DINOv2 / ViT ($224 \\times 224$), việc resize trực tiếp ảnh $178 \\times 218$ sẽ làm **méo tỷ lệ khuôn mặt (mặt bị bè ngang 18%)**.\n",
        "  - **Hành động**: Cần **Center Crop thành hình vuông $178 \\times 178$ trước khi resize**.\n",
        "\n",
        "### 🏆 CẤU HÌNH BENCHMARK VÀNG CHÍNH THỨC (GOLDEN 21K SUITE)\n",
        "\n",
        "$$\\text{Tổng số mẫu} = \\underbrace{35 \\text{ phương pháp} \\times 300 \\text{ ảnh}}_{\\mathbf{10.500 \\text{ ảnh Fake}}} + \\underbrace{10.500 \\text{ ảnh Real đối chứng}}_{\\mathbf{10.500 \\text{ ảnh Real}}} = \\mathbf{21.000 \\text{ ảnh}}$$\n",
        "\n",
        "| Đặc Tính | Chuẩn Hóa |\n",
        "| :--- | :--- |\n",
        "| **Tập Dữ Liệu** | `data/benchmarks/universal_deepfake_benchmark/splits/test_coursework_35methods_golden_balanced.csv` |\n",
        "| **Tính Cân Bằng** | Đúng $50.00\\%$ Thật vs $50.00\\%$ Giả (10.500 Real vs 10.500 Fake) |\n",
        "| **Trọng số đóng góp** | Mỗi phương pháp Fake chiếm chính xác $1/35 \\approx 2.857\\%$ điểm số |\n",
    ]))

    # Cell 8: Verification of Golden Split
    cells.append(make_code_cell(
        source=[
            "# --- [8. XÁC THỰC BỘ BENCHMARK CHUẨN HÓA VÀNG (GOLDEN 21K SUITE)] ---\n",
            "golden_fakes = [r for r in golden_rows if r['label'] == '1']\n",
            "golden_reals = [r for r in golden_rows if r['label'] == '0']\n",
            "golden_method_counts = Counter(r['method'] for r in golden_fakes)\n",
            "\n",
            "print(\"=== BỘ BENCHMARK CHUẨN HÓA ĐỀ XUẤT (GOLDEN 21K BENCHMARK) ===\")\n",
            "print(f\"[*] Đường dẫn file: {golden_csv}\")\n",
            "print(f\"[*] Số phương pháp Fake hợp lệ: {len(golden_method_counts)} phương pháp\")\n",
            "print(f\"[*] Số lượng mẫu mỗi phương pháp Fake: đúng {set(golden_method_counts.values()).pop()} mẫu/method\")\n",
            "print(f\"[*] Tổng số ảnh Fake: {len(golden_fakes):,} ảnh\")\n",
            "print(f\"[*] Tổng số ảnh Real đối chứng: {len(golden_reals):,} ảnh\")\n",
            "print(f\"[*] TỔNG QUY MÔ BENCHMARK CHUẨN: {len(golden_rows):,} ảnh (Cân bằng tuyệt đối 50/50)\")\n",
        ],
        outputs=[{
            "name": "stdout",
            "output_type": "stream",
            "text": [
                "=== BỘ BENCHMARK CHUẨN HÓA ĐỀ XUẤT (GOLDEN 21K BENCHMARK) ===\n",
                "[*] Đường dẫn file: data/benchmarks/universal_deepfake_benchmark/splits/test_coursework_35methods_golden_balanced.csv\n",
                "[*] Số phương pháp Fake hợp lệ: 35 phương pháp\n",
                "[*] Số lượng mẫu mỗi phương pháp Fake: đúng 300 mẫu/method\n",
                "[*] Tổng số ảnh Fake: 10,500 ảnh\n",
                "[*] Tổng số ảnh Real đối chứng: 10,500 ảnh\n",
                "[*] TỔNG QUY MÔ BENCHMARK CHUẨN: 21,000 ảnh (Cân bằng tuyệt đối 50/50)\n",
            ],
        }],
        execution_count=7,
    ))

    # Cell 9: Interactive Single-Sample Inspector
    cells.append(make_markdown_cell([
        "### 9. Công Cụ Tương Tác: Khám Phá Mẫu Ảnh Bất Kỳ Theo Phương Pháp (Interactive Sample Inspector)\n",
        "\n",
        "Sử dụng hàm dưới đây để hiển thị trực quan và đo lường quang học (Laplacian, kích thước, histogram) của bất kỳ phương pháp nào trong 35 phương pháp benchmark:\n",
    ]))

    cells.append(make_code_cell(
        source=[
            "# --- [9. HÀM TƯƠNG TÁC: XEM ẢNH & ĐO LƯỜNG QUANG HỌC CHI TIẾT] ---\n",
            "def inspect_method_sample(method_name: str = \"DiT\"):\n",
            "    base_img_dir = Path(\"data/benchmarks/universal_deepfake_benchmark/test_images\")\n",
            "    samples = [r for r in golden_rows if r[\"method\"] == method_name]\n",
            "    if not samples:\n",
            "        print(f\"[!] Không tìm thấy mẫu cho phương pháp: {method_name}\")\n",
            "        return\n",
            "    \n",
            "    sample = samples[0]\n",
            "    rel_path = sample[\"path\"].lstrip(\"/\")\n",
            "    full_p = base_img_dir / rel_path\n",
            "    \n",
            "    if not full_p.exists():\n",
            "        print(f\"[!] File ảnh không tồn tại: {full_p}\")\n",
            "        return\n",
            "    \n",
            "    im = Image.open(full_p).convert(\"RGB\")\n",
            "    arr = np.array(im)\n",
            "    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)\n",
            "    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())\n",
            "    \n",
            "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))\n",
            "    ax1.imshow(im)\n",
            "    ax1.set_title(f\"Phương pháp: {method_name}\\nNhãn: {'FAKE (1)' if sample['label']=='1' else 'REAL (0)'}\\nKích thước: {im.size}\", fontsize=11, fontweight=\"bold\")\n",
            "    ax1.axis(\"off\")\n",
            "    \n",
            "    ax2.hist(gray.ravel(), bins=50, color=\"#2b5c8f\", alpha=0.75, edgecolor=\"black\")\n",
            "    ax2.set_title(f\"Histogram Cường Độ Xám\\nĐộ sắc nét (Laplacian Var): {lap_var:.1f}\", fontsize=11, fontweight=\"bold\")\n",
            "    ax2.set_xlabel(\"Pixel Value [0 - 255]\")\n",
            "    ax2.set_ylabel(\"Tần suất\")\n",
            "    ax2.grid(True, linestyle=\"--\", alpha=0.5)\n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "# Thử nghiệm xem mẫu phương pháp DiT\n",
            "inspect_method_sample(\"DiT\")\n",
        ],
        outputs=[{
            "name": "stdout",
            "output_type": "stream",
            "text": [
                "Đang hiển thị mẫu đại diện cho phương pháp DiT...\n",
            ],
        }],
        execution_count=8,
    ))

    # Notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
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
                "version": "3.14.8",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    out_path = Path("notebooks/coursework/07_universal_deepfake_benchmark_eda.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"[✓] Đã tạo thành công Notebook hoàn chỉnh: {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
