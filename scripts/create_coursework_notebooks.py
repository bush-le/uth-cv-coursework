"""Script to generate clean coursework Jupyter Notebooks with proper markdown formatting."""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def create_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.14.0"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def md(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def code(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def main(output_dir: Optional[Union[str, Path]] = None) -> None:
    out_dir = Path(output_dir) if output_dir is not None else Path("notebooks/coursework")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Notebook 01: EDA Faces & Demographics ---
    nb1_cells = [
        md(
            "# UTH Computer Vision Coursework — Topic 11\n"
            "## Notebook 01: Exploratory Data Analysis (EDA) on Real Faces (FFHQ & CelebA-HQ)\n"
            "- **Nhóm thực hiện**: Nhóm 4\n"
            "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n"
            "- **Mục tiêu**: Khảo sát phân bố ảnh khuôn mặt thật từ hai nguồn chuẩn hóa FFHQ và CelebA-HQ, bao gồm độ tuổi, giới tính và phụ kiện che mặt (Glasses/Occlusion)."
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "\n"
            "# Import modular classes and config from src\n"
            "from src.common.config import load_config\n"
            "from src.data.image_schema import ImageMetadata\n"
            "from src.data.splits import normalize_demographics"
        ),
        md("### 1. Nạp và kiểm tra dữ liệu thô chuẩn hóa (`paths.raw_manifest`)"),
        code(
            "cfg = load_config(Path('configs/dataset_factory.yaml'))\n"
            "manifest_path = Path(cfg['paths']['raw_manifest'])\n"
            "records = []\n"
            "with open(manifest_path, 'r', encoding='utf-8') as f:\n"
            "    for line in f:\n"
            "        if line.strip():\n"
            "            records.append(ImageMetadata.from_dict(json.loads(line)))\n"
            "\n"
            "print(f'Tổng số ảnh khuôn mặt thật đã nạp: {len(records)}')\n"
            "if records:\n"
            "    print('Ví dụ bản ghi đầu tiên:', json.dumps(records[0].to_dict(), indent=2))"
        ),
        md("### 2. Thống kê phân bố nhân khẩu học (Tuổi, Giới tính, Kính)"),
        code(
            "datasets = {}\n"
            "age_groups = {}\n"
            "genders = {}\n"
            "glasses = {}\n"
            "\n"
            "for r in records:\n"
            "    d = r.demographics or {}\n"
            "    src = d.get('dataset', 'unknown')\n"
            "    datasets[src] = datasets.get(src, 0) + 1\n"
            "    \n"
            "    norm_d = normalize_demographics(d)\n"
            "    age = norm_d['age_group']\n"
            "    age_groups[age] = age_groups.get(age, 0) + 1\n"
            "    \n"
            "    g = norm_d['gender']\n"
            "    genders[g] = genders.get(g, 0) + 1\n"
            "    \n"
            "    gl = norm_d['glasses']\n"
            "    glasses[gl] = glasses.get(gl, 0) + 1\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "axes[0].bar(list(datasets.keys()), list(datasets.values()), color='#3498db')\n"
            "axes[0].set_title('Nguồn Dữ Liệu Gốc')\n"
            "axes[0].set_ylabel('Số lượng ảnh')\n"
            "\n"
            "axes[1].bar(list(genders.keys()), list(genders.values()), color='#e74c3c')\n"
            "axes[1].set_title('Phân Bố Giới Tính')\n"
            "\n"
            "axes[2].bar(list(glasses.keys()), list(glasses.values()), color='#2ecc71')\n"
            "axes[2].set_title('Phụ Kiện Che Mặt (Glasses/Occlusion)')\n"
            "axes[2].tick_params(axis='x', rotation=15)\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("### 3. Trực quan hóa mẫu ảnh khuôn mặt thật từ FFHQ và CelebA-HQ"),
        code(
            "fig, axes = plt.subplots(2, 4, figsize=(12, 6))\n"
            "ffhq_samples = [r for r in records if (r.demographics or {}).get('dataset') == 'ffhq'][:4]\n"
            "celeba_samples = [r for r in records if (r.demographics or {}).get('dataset') == 'celeba_hq'][:4]\n"
            "\n"
            "for i, r in enumerate(ffhq_samples):\n"
            "    ax = axes[0, i]\n"
            "    if r.file_path and Path(r.file_path).exists():\n"
            "        im = Image.open(r.file_path).convert('RGB')\n"
            "        ax.imshow(im)\n"
            "    demo = r.demographics or {}\n"
            "    ax.set_title(f'FFHQ: {demo.get(\"age_group\")}, {demo.get(\"gender\")}', fontsize=9)\n"
            "    ax.axis('off')\n"
            "\n"
            "for i, r in enumerate(celeba_samples):\n"
            "    ax = axes[1, i]\n"
            "    if r.file_path and Path(r.file_path).exists():\n"
            "        im = Image.open(r.file_path).convert('RGB')\n"
            "        ax.imshow(im)\n"
            "    demo = r.demographics or {}\n"
            "    ax.set_title(f'CelebA: {demo.get(\"gender\")}', fontsize=9)\n"
            "    ax.axis('off')\n"
            "\n"
            "plt.suptitle('Mẫu Khuôn Mặt Thật: Hàng 1 (FFHQ) vs Hàng 2 (CelebA-HQ)', fontsize=12, fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ]

    # --- Notebook 02: SD Generation & Curation ---
    nb2_cells = [
        md(
            "# UTH Computer Vision Coursework — Topic 11\n"
            "## Notebook 02: Stable Diffusion Generation & Automated Curation Gate\n"
            "- **Nhóm thực hiện**: Nhóm 4\n"
            "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n"
            "- **Mục tiêu**: Trực quan hóa các chế độ sinh ảnh giả bằng Stable Diffusion (`sd_text2img`, `sd_inpaint`, `sd_img2img`) và kiểm chứng bộ lọc chất lượng tự động `ImageCurationGate`."
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "import torch\n"
            "\n"
            "from src.common.config import load_config\n"
            "from src.common.metrics import compute_laplacian_sharpness\n"
            "from src.pipelines.curate_image_dataset import ImageCurationGate"
        ),
        md("### 1. Nạp danh mục ảnh giả đã sinh (`paths.curated_manifest`)"),
        code(
            "cfg = load_config(Path('configs/dataset_factory.yaml'))\n"
            "curated_manifest = Path(cfg['paths']['curated_manifest'])\n"
            "curated_items = []\n"
            "if curated_manifest.exists():\n"
            "    with open(curated_manifest, 'r', encoding='utf-8') as f:\n"
            "        for line in f:\n"
            "            if line.strip():\n"
            "                curated_items.append(json.loads(line))\n"
            "\n"
            "print(f'Số lượng ảnh giả đạt chuẩn Curation Gate: {len(curated_items)}')"
        ),
        md("### 2. Kiểm định độ sắc nét Laplacian qua Curation Gate"),
        code(
            "sharpness_scores = []\n"
            "for item in curated_items:\n"
            "    fp = item.get('file_path')\n"
            "    if fp and Path(fp).exists():\n"
            "        with Image.open(fp) as im:\n"
            "            np_arr = np.array(im.convert('RGB'), dtype=np.float32) / 255.0\n"
            "            t = torch.from_numpy(np_arr).permute(2, 0, 1)\n"
            "            score = compute_laplacian_sharpness(t)\n"
            "            sharpness_scores.append(score)\n"
            "\n"
            "if sharpness_scores:\n"
            "    print(f'Độ sắc nét Laplacian trung bình: {np.mean(sharpness_scores):.2f}')\n"
            "    print(f'Giá trị nhỏ nhất / lớn nhất: {np.min(sharpness_scores):.2f} / {np.max(sharpness_scores):.2f}')\n"
            "    plt.figure(figsize=(7, 4))\n"
            "    plt.hist(sharpness_scores, bins=8, color='#9b59b6', edgecolor='black')\n"
            "    plt.title('Phân Bố Độ Sắc Nét Laplacian Trên Tập Ảnh Giả')\n"
            "    plt.xlabel('Sharpness Score')\n"
            "    plt.ylabel('Số lượng ảnh')\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        ),
        md("### 3. Trực quan hóa ảnh khuôn mặt giả sinh bởi Stable Diffusion"),
        code(
            "n = min(4, len(curated_items))\n"
            "if n > 0:\n"
            "    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))\n"
            "    if n == 1:\n"
            "        axes = [axes]\n"
            "    for idx in range(n):\n"
            "        item = curated_items[idx]\n"
            "        fp = item.get('file_path')\n"
            "        if fp and Path(fp).exists():\n"
            "            im = Image.open(fp)\n"
            "            axes[idx].imshow(im)\n"
            "        axes[idx].set_title(item.get('manipulation_type', 'fake'), fontsize=10)\n"
            "        axes[idx].axis('off')\n"
            "    plt.suptitle('Mẫu Khuôn Mặt Giả Sinh Bằng Stable Diffusion Đạt Chuẩn Curation Gate', fontsize=12, fontweight='bold')\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        ),
    ]

    # --- Notebook 03: Hard Samples & Benchmark ---
    nb3_cells = [
        md(
            "# UTH Computer Vision Coursework — Topic 11\n"
            "## Notebook 03: Real-Manifold Hard-Sample Mining & Benchmark Analytics\n"
            "- **Nhóm thực hiện**: Nhóm 4\n"
            "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n"
            "- **Mục tiêu**: Phân tích các Mẫu Khó (`Hard Samples`) nằm sát đa tạp ảnh thật và kiểm tra các bộ benchmark đối chứng."
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "from PIL import Image\n"
            "\n"
            "from src.common.config import load_config"
        ),
        md("### 1. Báo cáo thống kê toàn diện bộ dữ liệu (`paths.stats_file`)"),
        code(
            "cfg = load_config(Path('configs/dataset_factory.yaml'))\n"
            "stats_path = Path(cfg['paths']['stats_file'])\n"
            "\n"
            "if stats_path.exists():\n"
            "    with open(stats_path, 'r', encoding='utf-8') as f:\n"
            "        stats = json.load(f)\n"
            "    print('Báo cáo thống kê tổng thể Dataset:')\n"
            "    print(json.dumps(stats, indent=2))\n"
            "else:\n"
            "    print('File thống kê chưa được tạo. Chạy: python3 -m scripts.organize_topic_dataset')"
        ),
        md("### 2. Phân bố Mẫu Khó theo từng chế độ sinh"),
        code(
            "if stats_path.exists():\n"
            "    hard_rates = stats.get('hard_rate_by_mode', {})\n"
            "    if hard_rates:\n"
            "        modes = list(hard_rates.keys())\n"
            "        rates = [hard_rates[m] * 100.0 for m in modes]\n"
            "        plt.figure(figsize=(6, 3.5))\n"
            "        plt.bar(modes, rates, color=['#3498db', '#e67e22', '#2ecc71'])\n"
            "        plt.title('Tỷ Lệ Mẫu Khó Theo Chế Độ Sinh (Real-Manifold)')\n"
            "        plt.ylabel('Tỷ lệ (%)')\n"
            "        plt.ylim([0, 100])\n"
            "        plt.tight_layout()\n"
            "        plt.show()"
        ),
        md("### 3. Lưới ảnh đối chứng trực quan toàn diện (Visual Gallery)"),
        code(
            "gallery_path = Path(cfg['paths']['gallery_image'])\n"
            "if gallery_path.exists():\n"
            "    im = Image.open(gallery_path)\n"
            "    plt.figure(figsize=(14, 10))\n"
            "    plt.imshow(im)\n"
            "    plt.axis('off')\n"
            "    plt.title('Lưới Ảnh Đối Chứng: Real Faces vs Synthetic SD vs Mined Hard Samples', fontsize=14, fontweight='bold')\n"
            "    plt.show()\n"
            "else:\n"
            "    print('Chưa có file gallery, chạy: python3 -m scripts.build_visual_gallery')"
        ),
    ]

    with open(out_dir / "01_eda_faces_demographics.ipynb", "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb1_cells), f, indent=2)

    with open(out_dir / "02_sd_generation_and_curation.ipynb", "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb2_cells), f, indent=2)

    with open(out_dir / "03_hard_samples_and_benchmark_comparison.ipynb", "w", encoding="utf-8") as f:
        json.dump(create_notebook(nb3_cells), f, indent=2)

    print(f"All 3 notebooks cleanly generated at {out_dir}!")


if __name__ == "__main__":
    main()
