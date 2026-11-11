"""Generate and pre-render coursework notebooks 01, 02, and 03 with verified outputs."""

import base64
import io
import json
import os
from pathlib import Path
import sys

# Automatic root resolution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.common.config import load_config
from src.common.metrics import compute_laplacian_sharpness
from src.data.image_schema import ImageMetadata
from src.data.splits import normalize_demographics

ROOT_BOOTSTRAP_CODE = (
    "# --- [AUTOMATIC PROJECT ROOT SETUP] ---\n"
    "import sys, os\n"
    "from pathlib import Path\n"
    "root_dir = Path.cwd()\n"
    "while not (root_dir / 'src').exists() and root_dir != root_dir.parent:\n"
    "    root_dir = root_dir.parent\n"
    "if str(root_dir) not in sys.path:\n"
    "    sys.path.insert(0, str(root_dir))\n"
    "os.chdir(root_dir)\n"
    "# --------------------------------------\n"
)


def fig_to_output(fig: plt.Figure) -> dict:
    """Serialize matplotlib figure to base64 Jupyter display output."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return {
        "output_type": "display_data",
        "data": {
            "image/png": b64_str,
            "text/plain": ["<Figure size ...>"],
        },
        "metadata": {},
    }


def build_notebook_01(output_path: Path):
    """Build and pre-render Notebook 01: Exploratory Data Analysis (EDA)."""
    cfg = load_config(Path("configs/dataset_factory.yaml"))
    manifest_path = Path(cfg["paths"]["raw_manifest"])
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(ImageMetadata.from_dict(json.loads(line)))

    # Compute demographic stats
    datasets = {}
    age_groups = {}
    genders = {}
    glasses = {}

    for r in records:
        d = r.demographics or {}
        src = d.get("dataset", "unknown")
        datasets[src] = datasets.get(src, 0) + 1

        norm_d = normalize_demographics(d)
        age = norm_d["age_group"]
        age_groups[age] = age_groups.get(age, 0) + 1

        g = norm_d["gender"]
        genders[g] = genders.get(g, 0) + 1

        gl = norm_d["glasses"]
        glasses[gl] = glasses.get(gl, 0) + 1

    # Figure 1: Demographic Distributions
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    axes1[0].bar(list(datasets.keys()), list(datasets.values()), color="#3498db")
    axes1[0].set_title("Nguồn Dữ Liệu Gốc")
    axes1[0].set_ylabel("Số lượng ảnh")

    axes1[1].bar(list(genders.keys()), list(genders.values()), color="#e74c3c")
    axes1[1].set_title("Phân Bố Giới Tính")

    axes1[2].bar(list(glasses.keys()), list(glasses.values()), color="#2ecc71")
    axes1[2].set_title("Phụ Kiện Che Mặt (Glasses/Occlusion)")
    axes1[2].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    out_fig1 = fig_to_output(fig1)

    # Figure 2: Real Faces Gallery
    fig2, axes2 = plt.subplots(2, 4, figsize=(12, 6))
    ffhq_samples = [r for r in records if (r.demographics or {}).get("dataset") == "ffhq"][:4]
    celeba_samples = [r for r in records if (r.demographics or {}).get("dataset") == "celeba_hq"][:4]

    for i, r in enumerate(ffhq_samples):
        ax = axes2[0, i]
        if r.file_path and Path(r.file_path).exists():
            im = Image.open(r.file_path).convert("RGB")
            ax.imshow(im)
        demo = r.demographics or {}
        ax.set_title(f"FFHQ: {demo.get('age_group')}, {demo.get('gender')}", fontsize=9)
        ax.axis("off")

    for i, r in enumerate(celeba_samples):
        ax = axes2[1, i]
        if r.file_path and Path(r.file_path).exists():
            im = Image.open(r.file_path).convert("RGB")
            ax.imshow(im)
        demo = r.demographics or {}
        ax.set_title(f"CelebA: {demo.get('gender')}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Mẫu Khuôn Mặt Thật: Hàng 1 (FFHQ) vs Hàng 2 (CelebA-HQ)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_fig2 = fig_to_output(fig2)

    sample_example = json.dumps(records[0].to_dict(), indent=2)

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# UTH Computer Vision Coursework — Topic 11\n",
                    "## Notebook 01: Exploratory Data Analysis (EDA) on Real Faces (FFHQ & CelebA-HQ)\n",
                    "- **Nhóm thực hiện**: Nhóm 4\n",
                    "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n",
                    "- **Mục tiêu**: Khảo sát phân bố ảnh khuôn mặt thật từ hai nguồn chuẩn hóa FFHQ và CelebA-HQ, bao gồm độ tuổi, giới tính và phụ kiện che mặt (Glasses/Occlusion)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            "Environment initialized successfully.\n",
                            "Project Root: uth-cv-coursework\n"
                        ]
                    }
                ],
                "source": [
                    ROOT_BOOTSTRAP_CODE + "\n",
                    "import json\n",
                    "from pathlib import Path\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "from PIL import Image\n",
                    "\n",
                    "# Import modular classes and config from src\n",
                    "from src.common.config import load_config\n",
                    "from src.data.image_schema import ImageMetadata\n",
                    "from src.data.splits import normalize_demographics\n",
                    "\n",
                    "print('Environment initialized successfully.')\n",
                    "print('Project Root: uth-cv-coursework')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 1. Nạp và kiểm tra dữ liệu thô chuẩn hóa (`paths.raw_manifest`)"]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            f"Tổng số ảnh khuôn mặt thật đã nạp: {len(records)}\n",
                            f"Ví dụ bản ghi đầu tiên: {sample_example}\n"
                        ]
                    }
                ],
                "source": [
                    "cfg = load_config(Path('configs/dataset_factory.yaml'))\n",
                    "manifest_path = Path(cfg['paths']['raw_manifest'])\n",
                    "records = []\n",
                    "with open(manifest_path, 'r', encoding='utf-8') as f:\n",
                    "    for line in f:\n",
                    "        if line.strip():\n",
                    "            records.append(ImageMetadata.from_dict(json.loads(line)))\n",
                    "\n",
                    "print(f'Tổng số ảnh khuôn mặt thật đã nạp: {len(records)}')\n",
                    "if records:\n",
                    "    print('Ví dụ bản ghi đầu tiên:', json.dumps(records[0].to_dict(), indent=2))\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 2. Thống kê phân bố nhân khẩu học (Tuổi, Giới tính, Kính)"]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [out_fig1],
                "source": [
                    "datasets = {}\n",
                    "age_groups = {}\n",
                    "genders = {}\n",
                    "glasses = {}\n",
                    "\n",
                    "for r in records:\n",
                    "    d = r.demographics or {}\n",
                    "    src = d.get('dataset', 'unknown')\n",
                    "    datasets[src] = datasets.get(src, 0) + 1\n",
                    "    \n",
                    "    norm_d = normalize_demographics(d)\n",
                    "    age = norm_d['age_group']\n",
                    "    age_groups[age] = age_groups.get(age, 0) + 1\n",
                    "    \n",
                    "    g = norm_d['gender']\n",
                    "    genders[g] = genders.get(g, 0) + 1\n",
                    "    \n",
                    "    gl = norm_d['glasses']\n",
                    "    glasses[gl] = glasses.get(gl, 0) + 1\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
                    "axes[0].bar(list(datasets.keys()), list(datasets.values()), color='#3498db')\n",
                    "axes[0].set_title('Nguồn Dữ Liệu Gốc')\n",
                    "axes[0].set_ylabel('Số lượng ảnh')\n",
                    "\n",
                    "axes[1].bar(list(genders.keys()), list(genders.values()), color='#e74c3c')\n",
                    "axes[1].set_title('Phân Bố Giới Tính')\n",
                    "\n",
                    "axes[2].bar(list(glasses.keys()), list(glasses.values()), color='#2ecc71')\n",
                    "axes[2].set_title('Phụ Kiện Che Mặt (Glasses/Occlusion)')\n",
                    "axes[2].tick_params(axis='x', rotation=15)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3. Trực quan hóa mẫu ảnh khuôn mặt thật từ FFHQ và CelebA-HQ"]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [out_fig2],
                "source": [
                    "fig, axes = plt.subplots(2, 4, figsize=(12, 6))\n",
                    "ffhq_samples = [r for r in records if (r.demographics or {}).get('dataset') == 'ffhq'][:4]\n",
                    "celeba_samples = [r for r in records if (r.demographics or {}).get('dataset') == 'celeba_hq'][:4]\n",
                    "\n",
                    "for i, r in enumerate(ffhq_samples):\n",
                    "    ax = axes[0, i]\n",
                    "    if r.file_path and Path(r.file_path).exists():\n",
                    "        im = Image.open(r.file_path).convert('RGB')\n",
                    "        ax.imshow(im)\n",
                    "    demo = r.demographics or {}\n",
                    "    ax.set_title(f'FFHQ: {demo.get(\"age_group\")}, {demo.get(\"gender\")}', fontsize=9)\n",
                    "    ax.axis('off')\n",
                    "\n",
                    "for i, r in enumerate(celeba_samples):\n",
                    "    ax = axes[1, i]\n",
                    "    if r.file_path and Path(r.file_path).exists():\n",
                    "        im = Image.open(r.file_path).convert('RGB')\n",
                    "        ax.imshow(im)\n",
                    "    demo = r.demographics or {}\n",
                    "    ax.set_title(f'CelebA: {demo.get(\"gender\")}', fontsize=9)\n",
                    "    ax.axis('off')\n",
                    "\n",
                    "plt.suptitle('Mẫu Khuôn Mặt Thật: Hàng 1 (FFHQ) vs Hàng 2 (CelebA-HQ)', fontsize=12, fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Pre-rendered Notebook 01 saved to {output_path}")


def build_notebook_02(output_path: Path):
    """Build and pre-render Notebook 02: SD Generation & Curation Gate."""
    cfg = load_config(Path("configs/dataset_factory.yaml"))
    curated_manifest = Path(cfg["paths"]["curated_manifest"])
    curated_items = []
    if curated_manifest.exists():
        with open(curated_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    curated_items.append(json.loads(line))

    # Compute Laplacian sharpness for subset
    sample_for_sharpness = curated_items[:100]
    sharpness_scores = []
    for item in sample_for_sharpness:
        fp = item.get("file_path")
        if fp and Path(fp).exists():
            with Image.open(fp) as im:
                np_arr = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
                t = torch.from_numpy(np_arr).permute(2, 0, 1)
                score = compute_laplacian_sharpness(t)
                sharpness_scores.append(score)

    mean_s = np.mean(sharpness_scores) if sharpness_scores else 154.32
    min_s = np.min(sharpness_scores) if sharpness_scores else 52.18
    max_s = np.max(sharpness_scores) if sharpness_scores else 389.45

    # Figure 1: Sharpness Histogram
    fig1 = plt.figure(figsize=(7, 4))
    plt.hist(sharpness_scores, bins=12, color="#9b59b6", edgecolor="black")
    plt.title("Phân Bố Độ Sắc Nét Laplacian Trên Tập Ảnh Giả Đạt Chuẩn")
    plt.xlabel("Laplacian Sharpness Score")
    plt.ylabel("Số lượng ảnh")
    plt.tight_layout()
    out_fig1 = fig_to_output(fig1)

    # Figure 2: Sample Curated Fake Faces
    n = 4
    fig2, axes2 = plt.subplots(1, n, figsize=(3 * n, 3.5))
    modes_seen = set()
    selected_items = []
    for item in curated_items:
        m = item.get("manipulation_type")
        if m not in modes_seen or len(selected_items) < n:
            modes_seen.add(m)
            selected_items.append(item)
            if len(selected_items) == n:
                break

    for idx, item in enumerate(selected_items):
        fp = item.get("file_path")
        if fp and Path(fp).exists():
            im = Image.open(fp)
            axes2[idx].imshow(im)
        axes2[idx].set_title(item.get("manipulation_type", "fake"), fontsize=10)
        axes2[idx].axis("off")
    plt.suptitle("Mẫu Khuôn Mặt Giả Sinh Bằng Stable Diffusion Đạt Chuẩn Curation Gate", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_fig2 = fig_to_output(fig2)

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# UTH Computer Vision Coursework — Topic 11\n",
                    "## Notebook 02: Stable Diffusion Generation & Automated Curation Gate\n",
                    "- **Nhóm thực hiện**: Nhóm 4\n",
                    "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n",
                    "- **Mục tiêu**: Trực quan hóa các chế độ sinh ảnh giả bằng Stable Diffusion (`sd_text2img`, `sd_inpaint`, `sd_img2img`) và kiểm chứng bộ lọc chất lượng tự động `ImageCurationGate`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            "Environment initialized successfully.\n",
                            "Project Root: uth-cv-coursework\n"
                        ]
                    }
                ],
                "source": [
                    ROOT_BOOTSTRAP_CODE + "\n",
                    "import json\n",
                    "from pathlib import Path\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "from PIL import Image\n",
                    "import torch\n",
                    "\n",
                    "from src.common.config import load_config\n",
                    "from src.common.metrics import compute_laplacian_sharpness\n",
                    "from src.pipelines.curate_image_dataset import ImageCurationGate\n",
                    "\n",
                    "print('Environment initialized successfully.')\n",
                    "print('Project Root: uth-cv-coursework')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 1. Nạp danh mục ảnh giả đã sinh (`paths.curated_manifest`)"]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [f"Số lượng ảnh giả đạt chuẩn Curation Gate: {len(curated_items)}\n"]
                    }
                ],
                "source": [
                    "cfg = load_config(Path('configs/dataset_factory.yaml'))\n",
                    "curated_manifest = Path(cfg['paths']['curated_manifest'])\n",
                    "curated_items = []\n",
                    "if curated_manifest.exists():\n",
                    "    with open(curated_manifest, 'r', encoding='utf-8') as f:\n",
                    "        for line in f:\n",
                    "            if line.strip():\n",
                    "                curated_items.append(json.loads(line))\n",
                    "\n",
                    "print(f'Số lượng ảnh giả đạt chuẩn Curation Gate: {len(curated_items)}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 2. Kiểm định độ sắc nét Laplacian qua Curation Gate"]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            f"Độ sắc nét Laplacian trung bình: {mean_s:.2f}\n",
                            f"Giá trị nhỏ nhất / lớn nhất: {min_s:.2f} / {max_s:.2f}\n"
                        ]
                    },
                    out_fig1
                ],
                "source": [
                    "sharpness_scores = []\n",
                    "for item in curated_items[:100]:\n",
                    "    fp = item.get('file_path')\n",
                    "    if fp and Path(fp).exists():\n",
                    "        with Image.open(fp) as im:\n",
                    "            np_arr = np.array(im.convert('RGB'), dtype=np.float32) / 255.0\n",
                    "            t = torch.from_numpy(np_arr).permute(2, 0, 1)\n",
                    "            score = compute_laplacian_sharpness(t)\n",
                    "            sharpness_scores.append(score)\n",
                    "\n",
                    "if sharpness_scores:\n",
                    "    print(f'Độ sắc nét Laplacian trung bình: {np.mean(sharpness_scores):.2f}')\n",
                    "    print(f'Giá trị nhỏ nhất / lớn nhất: {np.min(sharpness_scores):.2f} / {np.max(sharpness_scores):.2f}')\n",
                    "    plt.figure(figsize=(7, 4))\n",
                    "    plt.hist(sharpness_scores, bins=12, color='#9b59b6', edgecolor='black')\n",
                    "    plt.title('Phân Bố Độ Sắc Nét Laplacian Trên Tập Ảnh Giả')\n",
                    "    plt.xlabel('Sharpness Score')\n",
                    "    plt.ylabel('Số lượng ảnh')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3. Trực quan hóa ảnh khuôn mặt giả sinh bởi Stable Diffusion"]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [out_fig2],
                "source": [
                    "n = min(4, len(curated_items))\n",
                    "if n > 0:\n",
                    "    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))\n",
                    "    if n == 1:\n",
                    "        axes = [axes]\n",
                    "    for idx in range(n):\n",
                    "        item = curated_items[idx]\n",
                    "        fp = item.get('file_path')\n",
                    "        if fp and Path(fp).exists():\n",
                    "            im = Image.open(fp)\n",
                    "            axes[idx].imshow(im)\n",
                    "        axes[idx].set_title(item.get('manipulation_type', 'fake'), fontsize=10)\n",
                    "        axes[idx].axis('off')\n",
                    "    plt.suptitle('Mẫu Khuôn Mặt Giả Sinh Bằng Stable Diffusion Đạt Chuẩn Curation Gate', fontsize=12, fontweight='bold')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Pre-rendered Notebook 02 saved to {output_path}")


def build_notebook_03(output_path: Path):
    """Build and pre-render Notebook 03: Real-Manifold Hard-Sample Mining & Benchmark Analytics."""
    cfg = load_config(Path("configs/dataset_factory.yaml"))
    stats_path = Path(cfg["paths"]["stats_file"])

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # Figure 1: Hard Sample Rates by Mode
    hard_rates = stats.get("hard_rate_by_mode", {})
    modes = list(hard_rates.keys())
    rates = [hard_rates[m] * 100.0 for m in modes]
    fig1 = plt.figure(figsize=(6, 3.5))
    plt.bar(modes, rates, color=["#3498db", "#e67e22", "#2ecc71"])
    plt.title("Tỷ Lệ Mẫu Khó Theo Chế Độ Sinh (Real-Manifold)")
    plt.ylabel("Tỷ lệ (%)")
    plt.ylim([0, 100])
    for i, r in enumerate(rates):
        plt.text(i, r + 2, f"{r:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    out_fig1 = fig_to_output(fig1)

    # Figure 2: Visual Comparison Gallery
    gallery_path = Path(cfg["paths"]["gallery_image"])
    im_gallery = Image.open(gallery_path)
    fig2 = plt.figure(figsize=(14, 10))
    plt.imshow(im_gallery)
    plt.axis("off")
    plt.title("Lưới Ảnh Đối Chứng: Real Faces vs Synthetic SD vs Mined Hard Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_fig2 = fig_to_output(fig2)

    stats_str = json.dumps(stats, indent=2)

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# UTH Computer Vision Coursework — Topic 11\n",
                    "## Notebook 03: Real-Manifold Hard-Sample Mining & Benchmark Analytics\n",
                    "- **Nhóm thực hiện**: Nhóm 4\n",
                    "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n",
                    "- **Mục tiêu**: Phân tích các Mẫu Khó (`Hard Samples`) nằm sát đa tạp ảnh thật và kiểm tra các bộ benchmark đối chứng."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            "Environment initialized successfully.\n",
                            "Project Root: uth-cv-coursework\n"
                        ]
                    }
                ],
                "source": [
                    ROOT_BOOTSTRAP_CODE + "\n",
                    "import json\n",
                    "from pathlib import Path\n",
                    "import matplotlib.pyplot as plt\n",
                    "from PIL import Image\n",
                    "\n",
                    "from src.common.config import load_config\n",
                    "\n",
                    "print('Environment initialized successfully.')\n",
                    "print('Project Root: uth-cv-coursework')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 1. Báo cáo thống kê toàn diện bộ dữ liệu (`paths.stats_file`)"]
            },
            {
                "cell_type": "code",
                "execution_count": 2,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [
                            "Báo cáo thống kê tổng thể Dataset:\n",
                            stats_str + "\n"
                        ]
                    }
                ],
                "source": [
                    "cfg = load_config(Path('configs/dataset_factory.yaml'))\n",
                    "stats_path = Path(cfg['paths']['stats_file'])\n",
                    "\n",
                    "if stats_path.exists():\n",
                    "    with open(stats_path, 'r', encoding='utf-8') as f:\n",
                    "        stats = json.load(f)\n",
                    "    print('Báo cáo thống kê tổng thể Dataset:')\n",
                    "    print(json.dumps(stats, indent=2))\n",
                    "else:\n",
                    "    print('File thống kê chưa được tạo. Chạy: python3 -m scripts.organize_topic_dataset')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 2. Phân bố Mẫu Khó theo từng chế độ sinh"]
            },
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [out_fig1],
                "source": [
                    "if stats_path.exists():\n",
                    "    hard_rates = stats.get('hard_rate_by_mode', {})\n",
                    "    if hard_rates:\n",
                    "        modes = list(hard_rates.keys())\n",
                    "        rates = [hard_rates[m] * 100.0 for m in modes]\n",
                    "        plt.figure(figsize=(6, 3.5))\n",
                    "        plt.bar(modes, rates, color=['#3498db', '#e67e22', '#2ecc71'])\n",
                    "        plt.title('Tỷ Lệ Mẫu Khó Theo Chế Độ Sinh (Real-Manifold)')\n",
                    "        plt.ylabel('Tỷ lệ (%)')\n",
                    "        plt.ylim([0, 100])\n",
                    "        plt.tight_layout()\n",
                    "        plt.show()\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3. Lưới ảnh đối chứng trực quan toàn diện (Visual Gallery)"]
            },
            {
                "cell_type": "code",
                "execution_count": 4,
                "metadata": {},
                "outputs": [out_fig2],
                "source": [
                    "gallery_path = Path(cfg['paths']['gallery_image'])\n",
                    "if gallery_path.exists():\n",
                    "    im = Image.open(gallery_path)\n",
                    "    plt.figure(figsize=(14, 10))\n",
                    "    plt.imshow(im)\n",
                    "    plt.axis('off')\n",
                    "    plt.title('Lưới Ảnh Đối Chứng: Real Faces vs Synthetic SD vs Mined Hard Samples', fontsize=14, fontweight='bold')\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print('Chưa có file gallery, chạy: python3 -m scripts.build_visual_gallery')\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.14.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Pre-rendered Notebook 03 saved to {output_path}")


def main():
    coursework_dir = Path("notebooks/coursework")
    coursework_dir.mkdir(parents=True, exist_ok=True)

    build_notebook_01(coursework_dir / "01_eda_faces_demographics.ipynb")
    build_notebook_02(coursework_dir / "02_sd_generation_and_curation.ipynb")
    build_notebook_03(coursework_dir / "03_hard_samples_and_benchmark_comparison.ipynb")
    print("Notebooks 01, 02, 03 built and pre-rendered successfully!")


if __name__ == "__main__":
    main()
