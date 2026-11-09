"""Script to generate clean coursework Jupyter Notebooks with proper markdown formatting."""

import json
from pathlib import Path


def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.14.0"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def main():
    out_dir = Path("notebooks/coursework")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Notebook 01: EDA Faces & Demographics ---
    nb1_cells = [
        md(
            "# UTH Computer Vision Coursework — Topic 11\n"
            "## Notebook 01: Exploratory Data Analysis (EDA) on Real Faces (FFHQ & CelebA-HQ)\n"
            "- **Nhóm thực hiện**: Nhóm 4\n"
            "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n"
            "- **Mục tiêu**: Khảo sát phân bố 4.000 ảnh khuôn mặt thật từ hai nguồn chuẩn hóa FFHQ và CelebA-HQ, bao gồm độ tuổi, giới tính và phụ kiện che mặt (Glasses/Occlusion)."
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "\n"
            "# Import modular classes from src\n"
            "from src.data.image_schema import ImageMetadata"
        ),
        md("### 1. Nạp và kiểm tra dữ liệu thô chuẩn hóa (`data/raw/manifest.jsonl`)"),
        code(
            "manifest_path = Path('data/raw/manifest.jsonl')\n"
            "records = []\n"
            "with open(manifest_path, 'r', encoding='utf-8') as f:\n"
            "    for line in f:\n"
            "        if line.strip():\n"
            "            records.append(ImageMetadata.from_dict(json.loads(line)))\n"
            "\n"
            "print(f'Tổng số ảnh khuôn mặt thật đã nạp: {len(records)}')\n"
            "print('Ví dụ bản ghi đầu tiên:', json.dumps(records[0].to_dict(), indent=2))"
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
            "    age = d.get('age_group', 'unknown')\n"
            "    age_groups[age] = age_groups.get(age, 0) + 1\n"
            "    \n"
            "    g = d.get('gender', 'unknown')\n"
            "    genders[g] = genders.get(g, 0) + 1\n"
            "    \n"
            "    gl = d.get('glasses', 'unknown')\n"
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
            "- **Mục tiêu**: Trực quan hóa 3 chế độ sinh ảnh giả bằng Stable Diffusion (`text2image`, `inpainting`, `img2img`) và kiểm chứng bộ lọc chất lượng tự động `ImageCurationGate`."
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "import torch\n"
            "\n"
            "from src.common.metrics import compute_laplacian_sharpness\n"
            "from src.pipelines.curate_image_dataset import ImageCurationGate"
        ),
        md("### 1. Nạp danh mục ảnh giả đã sinh (`data/synthetic/curated_manifest.jsonl`)"),
        code(
            "curated_manifest = Path('data/synthetic/curated_manifest.jsonl')\n"
            "if not curated_manifest.exists():\n"
            "    curated_manifest = Path('data/processed/curated_manifest.jsonl')\n"
            "\n"
            "curated_items = []\n"
            "with open(curated_manifest, 'r', encoding='utf-8') as f:\n"
            "    for line in f:\n"
            "        if line.strip():\n"
            "            curated_items.append(json.loads(line))\n"
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
            "    plt.axvline(x=10.0, color='r', linestyle='--', label='Ngưỡng lọc Curation Gate (min=10.0)')\n"
            "    plt.title('Phân Bố Độ Sắc Nét Laplacian Trên Tập Ảnh Giả')\n"
            "    plt.xlabel('Sharpness Score')\n"
            "    plt.ylabel('Số lượng ảnh')\n"
            "    plt.legend()\n"
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
            "## Notebook 03: Meta DINOv2 Hard-Sample Mining & Closed-Loop Benchmark\n"
            "- **Nhóm thực hiện**: Nhóm 4\n"
            "- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*\n"
            "- **Mục tiêu**: Phân tích các Mẫu Khó (`Hard Samples`) vượt qua được mạng DINOv2 baseline và kiểm chứng hiệu quả fine-tuning đối chứng $D_0 \\to D_1 \\to D_2$."
        ),
        code(
            "import json\n"
            "from pathlib import Path\n"
            "import matplotlib.pyplot as plt\n"
            "from PIL import Image"
        ),
        md("### 1. Phân tích lỗ hổng mô hình gốc qua Mẫu Khó (`data/hard_samples/`)"),
        code(
            "audit_file = Path('data/hard_samples/audit_reports/hard_sample_demographics.json')\n"
            "if audit_file.exists():\n"
            "    with open(audit_file, 'r', encoding='utf-8') as f:\n"
            "        audit = json.load(f)\n"
            "    print('Báo cáo phân tích Mẫu Khó (Hard Samples Audit):')\n"
            "    print(json.dumps(audit, indent=2))\n"
            "else:\n"
            "    print('File audit chưa được tạo. Chạy: python3 -m src.pipelines.mine_hard_samples')"
        ),
        md("### 2. So sánh kết quả thực nghiệm 3 giai đoạn đối chứng ($D_0 \\to D_1 \\to D_2$)"),
        code(
            "benchmark_log = Path('experiments/logs/benchmark_comparison.json')\n"
            "if benchmark_log.exists():\n"
            "    with open(benchmark_log, 'r', encoding='utf-8') as f:\n"
            "        bench = json.load(f)\n"
            "    \n"
            "    stages = ['D0 (Baseline)', 'D1 (Synthetic)', 'D2 (Hard Boosted)']\n"
            "    losses = [bench['D0_baseline']['loss'], bench['D1_synthetic']['loss'], bench['D2_hard_samples']['loss']]\n"
            "    aucs = [bench['D0_baseline']['roc_auc'], bench['D1_synthetic']['roc_auc'], bench['D2_hard_samples']['roc_auc']]\n"
            "    \n"
            "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
            "    axes[0].bar(stages, losses, color=['#95a5a6', '#f39c12', '#27ae60'])\n"
            "    axes[0].set_title('Test Loss Qua Các Giai Đoạn (Càng thấp càng tốt)')\n"
            "    axes[0].set_ylabel('Binary Cross Entropy Loss')\n"
            "    \n"
            "    axes[1].bar(stages, aucs, color=['#95a5a6', '#f39c12', '#2980b9'])\n"
            "    axes[1].set_title('ROC-AUC Score')\n"
            "    axes[1].set_ylim([0.0, 1.1])\n"
            "    axes[1].set_ylabel('ROC-AUC')\n"
            "    \n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    \n"
            "    print('Mức cải thiện delta:')\n"
            "    print(json.dumps(bench.get('deltas', {}), indent=2))"
        ),
        md("### 3. Lưới ảnh đối chứng trực quan toàn diện (Visual Gallery)"),
        code(
            "gallery_path = Path('experiments/predictions/real_vs_synthetic_gallery.png')\n"
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

    print("All 3 notebooks cleanly generated with proper formatting!")


if __name__ == "__main__":
    main()
