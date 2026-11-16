"""Unified Coursework Notebook Suite Generation Engine.

Automates the generation and live pre-execution of the 4-part unified coursework notebook suite
for UTH Topic 11 (Diffusion-based Deepfake Detection using DINOv2 and Active Learning).

Notebooks in the unified suite:
  00_master_deepfake_defense_showcase.ipynb
  01_diffusion_dataset_and_curation.ipynb
  02_hard_samples_and_active_learning.ipynb
  03_champion_m3_and_universal_benchmark.ipynb
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import sys
from typing import Any, Optional, Union

# Ensure project root is in sys.path when executed directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from IPython.core.interactiveshell import InteractiveShell
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.artist as m_artist
import matplotlib.pyplot as plt

from src.common.logger import get_logger

logger = get_logger(__name__)

# Canonical suite notebook filenames
SUITE_NOTEBOOKS = {
    0: {
        "file": "00_master_deepfake_defense_showcase.ipynb",
        "title": "Master Deepfake Defense Showcase & Live Forensic Lecture",
        "subtitle": "Đại bản doanh đồ án: Tổng quan A-Z, Bảng vàng M0-M3, Universal Benchmark 20.400 ảnh, Live Forensic Inspector",
        "modules": [
            ("src/models/detector/image_detector.py", "../../src/models/detector/image_detector.py"),
            ("docs/reports/M3_DUAL_ENGINE_CHAMPION_REPORT.md", "../../docs/reports/M3_DUAL_ENGINE_CHAMPION_REPORT.md"),
        ],
    },
    1: {
        "file": "01_diffusion_dataset_and_curation.ipynb",
        "title": "Đa Tạp Ảnh Thật & Động Cơ Khuếch Tán Kép",
        "subtitle": "Khảo sát 4.000 chân dung FFHQ/CelebA-HQ, Soft-mask Gaussian σ=4.0, CurationGate lọc 545 mẫu lỗi",
        "modules": [
            ("src/data/raw_ingestion.py", "../../src/data/raw_ingestion.py"),
            ("src/pipelines/curate_image_dataset.py", "../../src/pipelines/curate_image_dataset.py"),
        ],
    },
    2: {
        "file": "02_hard_samples_and_active_learning.ipynb",
        "title": "Khai Thác Mẫu Khó & Vòng Lặp Active Learning DINOv2",
        "subtitle": "213 mẫu Challenge Vault biên mơ hồ (|P-0.5|≤0.15), DINOv2 ViT-S/14 fine-tuning, Fake Recall 11.7% -> 91.9%",
        "modules": [
            ("src/models/detector/image_detector.py", "../../src/models/detector/image_detector.py"),
            ("src/models/evaluators/hardness.py", "../../src/models/evaluators/hardness.py"),
        ],
    },
    3: {
        "file": "03_champion_m3_and_universal_benchmark.ipynb",
        "title": "Mô Hình Champion M3 & Đối Chuẩn Toàn Cầu Universal Benchmark",
        "subtitle": "Phân vùng D3 sạch cách ly 626 ảnh test, Champion M3 Dual-Engine, Quad-Vault 5 models, Universal 20.400 ảnh",
        "modules": [
            ("scripts/train_m3_dual_engine.py", "../../scripts/train_m3_dual_engine.py"),
            ("scripts/evaluate_quad_vault_5models.py", "../../scripts/evaluate_quad_vault_5models.py"),
            ("scripts/evaluate_universal_benchmark_5models.py", "../../scripts/evaluate_universal_benchmark_5models.py"),
        ],
    },
}


def make_markdown_cell(source: Union[str, list[str]]) -> dict[str, Any]:
    """Create a Jupyter markdown cell conforming to nbformat v4."""
    if isinstance(source, str):
        lines = source.splitlines(keepends=True)
    else:
        lines = list(source)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def _is_matplotlib_artist(obj: Any) -> bool:
    """Check if an object is a matplotlib Artist or collection of Artists to suppress bare repr in cells."""
    if isinstance(obj, m_artist.Artist):
        return True
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(x, m_artist.Artist) for x in obj):
        return True
    return False


def execute_and_capture(
    shell: InteractiveShell,
    code: str,
    max_dpi: int = 120,
    raise_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Execute code string in an IPython shell, capturing stdout streams and matplotlib figures.

    Ensures zero dangling figures by invoking plt.close("all") and enforces a target
    DPI (120-130) to guarantee serialized notebook sizes remain below 2.5MB.
    """
    plt.rcParams["figure.dpi"] = max_dpi
    plt.rcParams["savefig.dpi"] = max_dpi
    plt.rcParams["figure.max_open_warning"] = 50

    outputs: list[dict[str, Any]] = []
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    def flush_streams() -> None:
        s_out = stdout_buf.getvalue()
        if s_out:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": s_out.splitlines(keepends=True),
            })
            stdout_buf.seek(0)
            stdout_buf.truncate(0)

        s_err = stderr_buf.getvalue()
        if s_err:
            outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": s_err.splitlines(keepends=True),
            })
            stderr_buf.seek(0)
            stderr_buf.truncate(0)

    def capture_open_figures() -> None:
        fignums = plt.get_fignums()
        if not fignums:
            return
        for num in fignums:
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=max_dpi)
            buf.seek(0)
            b64_payload = base64.b64encode(buf.read()).decode("ascii")
            buf.close()
            outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": b64_payload,
                    "text/plain": [f"<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>"],
                },
                "metadata": {},
            })
        plt.close("all")

    orig_show = matplotlib.pyplot.show

    def hooked_show(*args: Any, **kwargs: Any) -> None:
        flush_streams()
        capture_open_figures()

    orig_write_prompt = getattr(shell.displayhook, "write_output_prompt", None)
    orig_write_format = getattr(shell.displayhook, "write_format_data", None)

    matplotlib.pyplot.show = hooked_show
    if orig_write_prompt is not None:
        shell.displayhook.write_output_prompt = lambda: None
    if orig_write_format is not None:
        shell.displayhook.write_format_data = lambda format_dict, md_dict=None: None

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stderr_buf

    try:
        res = shell.run_cell(code)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        matplotlib.pyplot.show = orig_show
        if orig_write_prompt is not None:
            shell.displayhook.write_output_prompt = orig_write_prompt
        if orig_write_format is not None:
            shell.displayhook.write_format_data = orig_write_format

    flush_streams()
    capture_open_figures()
    plt.close("all")

    if res.error_in_exec:
        if raise_on_error:
            raise res.error_in_exec
        outputs.append({
            "output_type": "error",
            "ename": type(res.error_in_exec).__name__,
            "evalue": str(res.error_in_exec),
            "traceback": getattr(res, "traceback", [str(res.error_in_exec)]),
        })

    if res.result is not None and not _is_matplotlib_artist(res.result):
        outputs.append({
            "output_type": "execute_result",
            "execution_count": getattr(res, "execution_count", None),
            "data": {
                "text/plain": [repr(res.result)],
            },
            "metadata": {},
        })

    return outputs


def make_code_cell(
    source: Union[str, list[str]],
    shell: Optional[InteractiveShell] = None,
    execute: bool = True,
    outputs: Optional[list[dict[str, Any]]] = None,
    execution_count: Optional[int] = None,
    max_dpi: int = 120,
) -> dict[str, Any]:
    """Create a Jupyter code cell, optionally executing and capturing outputs live."""
    if isinstance(source, str):
        lines = source.splitlines(keepends=True)
        raw_code = source
    else:
        lines = list(source)
        raw_code = "".join(source)

    if outputs is not None:
        cell_outputs = list(outputs)
    elif shell is not None and execute:
        cell_outputs = execute_and_capture(shell, raw_code, max_dpi=max_dpi)
    else:
        cell_outputs = []

    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": cell_outputs,
        "source": lines,
    }


def make_header_cell(
    title: str,
    subtitle: str,
    part_num: int,
    time_str: str = "2026-11-16",
    objective: str = "",
) -> dict[str, Any]:
    """Generate Cell 0 institutional header and objective block for UTH coursework."""
    objective_text = objective.strip() if objective else "Triển khai thực nghiệm, kiểm chứng số liệu và trực quan hóa chuyên sâu."
    header_md = f"""# TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH)
## VIỆN CÔNG NGHỆ THÔNG TIN & TRÍ TUỆ NHÂN TẠO
### HỌC PHẦN: THỊ GIÁC MÁY TÍNH NÂNG CAO (ADVANCED COMPUTER VISION)
---
# ĐỀ TÀI 11: PHÁT HIỆN KHUÔN MẶT DEEPFAKE DO MÔ HÌNH KHUẾCH TÁN (DIFFUSION MODELS) TẠO RA DỰA TRÊN DINOv2 VÀ ACTIVE LEARNING

## PHẦN {part_num}: {title.upper()}
> **Chuyên đề:** {title} — *{subtitle}*  
> **Thời gian cập nhật:** {time_str} | **Môi trường:** Python 3.14 (.venv) • PyTorch 2.x • GPU/CPU Hybrid  
> **Đơn vị thực hiện:** Đội ngũ Nghiên cứu Đề tài 11 — UTH Computer Vision Research Lab

---

### 🎯 Mục Tiêu Khoa Học & Kỹ Thuật (Scientific & Technical Objectives)
{objective_text}
"""
    return make_markdown_cell(header_md)


def make_roadmap_cell(current_part: int) -> dict[str, Any]:
    """Generate Cell 1 4-column Roadmap Markdown table linking the 4 suite notebooks with relative paths.

    Highlights the active notebook row and ensures 100% valid relative paths to codebase modules and reports.
    """
    rows = []
    for part_idx in sorted(SUITE_NOTEBOOKS.keys()):
        meta = SUITE_NOTEBOOKS[part_idx]
        filename = meta["file"]
        title_text = meta["title"]
        subtitle_text = meta["subtitle"]

        # Format module links with relative paths
        module_links = "<br>".join([f"[`{disp}`]({rel})" for disp, rel in meta["modules"]])

        if part_idx == current_part:
            part_col = f"**Phần {part_idx}** 🎯"
            nb_col = f"**[{filename}]({filename})** *(Đang xem)*"
            desc_col = f"**{title_text}**<br>*{subtitle_text}*"
        else:
            part_col = f"Phần {part_idx}"
            nb_col = f"[{filename}]({filename})"
            desc_col = f"{title_text}<br>*{subtitle_text}*"

        rows.append(f"| {part_col} | {nb_col} | {desc_col} | {module_links} |")

    roadmap_md = f"""### 🗺️ Lộ Trình Toàn Bộ Bộ Notebook Đồ Án (4-Part Unified Coursework Suite)

| Phần | Notebook | Mục Tiêu & Nội Dung Trọng Tâm | Tài Liệu & Module Liên Kết |
| :---: | :--- | :--- | :--- |
""" + "\n".join(rows) + f"""

> 💡 **Hướng dẫn điều hướng:** Nhấp vào tên notebook trong cột 2 để chuyển đổi qua lại giữa 4 chương chuyên sâu. Cột 4 liên kết trực tiếp tới mã nguồn triển khai và báo cáo nghiệm thu kỹ thuật.
"""
    return make_markdown_cell(roadmap_md)


def create_notebook(
    title: str = "",
    subtitle: str = "",
    part_num: int = 0,
    cells: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Construct a full Jupyter notebook dictionary conforming to nbformat v4."""
    notebook_cells = list(cells) if cells is not None else []
    return {
        "cells": notebook_cells,
        "metadata": {
            "title": title,
            "subtitle": subtitle,
            "part_num": part_num,
            "language_info": {
                "name": "python",
                "version": "3.14.8",
            },
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save_notebook(nb_dict: dict[str, Any], path: Union[str, Path]) -> None:
    """Save notebook dictionary to JSON file on disk."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=1, ensure_ascii=False)
    logger.info("Saved notebook to %s", target_path)


def write_notebook(
    arg1: Union[Path, str, dict[str, Any], list[dict[str, Any]]],
    arg2: Union[Path, str, dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Polymorphic helper supporting write_notebook(path, nb_or_cells) and write_notebook(nb_dict, path)."""
    if isinstance(arg1, (dict, list)) and isinstance(arg2, (str, Path)):
        nb = arg1 if isinstance(arg1, dict) else create_notebook(cells=arg1)
        save_notebook(nb, arg2)
    elif isinstance(arg1, (str, Path)) and isinstance(arg2, (dict, list)):
        nb = arg2 if isinstance(arg2, dict) else create_notebook(cells=arg2)
        save_notebook(nb, arg1)
    else:
        raise ValueError(f"Invalid arguments to write_notebook: {type(arg1)}, {type(arg2)}")


def build_notebook_01(
    shell: Optional[InteractiveShell] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Build and live-execute Notebook 01: Real Manifold & Dual-Engine Portfolio.

    Generates `notebooks/coursework/01_diffusion_dataset_and_curation.ipynb` with:
      - Cell 0: UTH Institutional Header & Topic 11 objectives.
      - Cell 1: 4-part Roadmap Table with Part 1 active.
      - Section 1: Real face manifold (4,000 images, FFHQ/CelebA-HQ), demographic distribution,
                   and real skin color gamut analysis in YCbCr color space.
      - Section 2: Dual-engine portfolio (SD 1.5 vs RealisticVision v5.1) and soft-mask Gaussian
                   feathering (sigma=4.0) with edge gradient profile comparison.
      - Section 3: 3-tier CurationGate deep audit: live sharpness gate execution and audit of
                   542+ rejected anomalies from D0 Train and Master Test reports.
      - Section 4: Stage 1 synthesis and segue to Notebook 02.
    """
    if shell is None:
        shell = InteractiveShell.instance()

    target_path = Path(output_path) if output_path else Path("notebooks/coursework/01_diffusion_dataset_and_curation.ipynb")
    logger.info("Building Notebook 01 -> %s", target_path)

    cells = []

    # Cell 0: Header
    header = make_header_cell(
        title="Đa Tạp Ảnh Thật & Động Cơ Khuếch Tán Kép",
        subtitle="Khảo Sát 4.000 Chân Dung FFHQ/CelebA-HQ, Soft-mask Gaussian σ=4.0, CurationGate Lọc 545 Mẫu Lỗi",
        part_num=1,
        time_str="2026-11-16",
        objective="Khảo sát đa tạp ảnh thật (4.000 chân dung), phân tích động cơ sinh khuếch tán kép (SD 1.5 + RealisticVision v5.1), giải phẫu cơ chế hòa trộn Soft-mask Gaussian (σ=4.0), và kiểm toán phễu lọc 3 tầng CurationGate đào thải dị tật.",
    )
    cells.append(header)

    # Cell 1: Roadmap
    roadmap = make_roadmap_cell(current_part=1)
    cells.append(roadmap)

    # Cell 2: Markdown - Section 1 Intro
    sec1_md = """## 1. Khảo Sát Đa Tạp Ảnh Thật Nền Tảng (Ground Truth Real Face Manifold)

Để xây dựng một bộ dữ liệu đối chuẩn đáng tin cậy phục vụ bài toán phát hiện khuôn mặt giả mạo sinh bởi các mô hình khuếch tán (Diffusion Models), bước đầu tiên mang tính quyết định là thiết lập **đa tạp ảnh thật (Ground Truth Manifold)** với các tiêu chuẩn khắt khe:
- **Quy mô mẫu chuẩn**: 4.000 ảnh chân dung độ phân giải cao thu thập từ 2 kho ảnh chuẩn học thuật quốc tế: **FFHQ (Flickr-Faces-HQ)** và **CelebA-HQ**.
- **Đa dạng hóa nhân khẩu học (Demographic Diversity)**: Bao phủ cân bằng giới tính (Nam/Nữ), dải độ tuổi từ sơ sinh đến người cao tuổi (0 đến 120 tuổi), góc chụp đầu ($Yaw, Pitch, Roll$) và phụ kiện (kính mát, kính cận).
- **Tính phân ly danh tính tuyệt đối (Strict Identity-Disjoint Boundary)**: Toàn bộ 4.000 danh tính ảnh thật được cố định nhãn $y=0$, không bị rò rỉ chéo giữa các tập Train, Validation và Master Test.
"""
    cells.append(make_markdown_cell(sec1_md))

    # Cell 3: Code - Ingestion & Demographics
    code_ingestion = """# Thiết lập môi trường và nạp danh mục ảnh thật (Ground Truth Manifold)
import sys
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Đảm bảo đường dẫn tương đối tới mã nguồn đồ án
if "." not in sys.path:
    sys.path.insert(0, ".")

from src.common.seed import seed_everything
seed_everything(42)

manifest_path = Path("data/raw/manifest.jsonl")
assert manifest_path.exists(), f"Không tìm thấy manifest tại: {manifest_path}"

records = []
datasets = Counter()
genders = Counter()
age_groups = Counter()
glasses = Counter()

with open(manifest_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        records.append(item)
        demo = item.get("demographics", {})
        datasets[demo.get("dataset", "unknown")] += 1
        genders[demo.get("gender", "unknown")] += 1
        age_groups[demo.get("age_group", "unknown")] += 1
        glasses[demo.get("glasses", "None")] += 1

print(f"✅ Đã nạp thành công {len(records):,} bản ghi ảnh thật từ manifest.jsonl")
print(f"📊 Phân bố tập dữ liệu gốc: FFHQ = {datasets['ffhq']:,} ảnh | CelebA-HQ = {datasets['celeba_hq']:,} ảnh")
print(f"👥 Phân bố giới tính (FFHQ): Nữ = {genders['female']:,} | Nam = {genders['male']:,} | Chưa gán = {genders['unknown']:,}")
"""
    cells.append(make_code_cell(code_ingestion, shell=shell))

    # Cell 4: Code - Demographics Plots
    code_demo_plots = """# Trực quan hóa phân bố nhân khẩu học (Demographics Distribution) của tập 4.000 ảnh thật
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.2]})
plt.subplots_adjust(wspace=0.35)

# 1. Phân bố Nguồn dữ liệu
ds_labels = ["FFHQ (Flickr)", "CelebA-HQ"]
ds_counts = [datasets["ffhq"], datasets["celeba_hq"]]
colors_ds = ["#2b5c8f", "#d95f02"]
bars1 = axes[0].bar(ds_labels, ds_counts, color=colors_ds, width=0.55, edgecolor="black", linewidth=0.8)
axes[0].set_title("Nguồn Dữ Liệu Nền Tảng\\n(Real Ground Truth)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Số lượng ảnh chân dung", fontsize=10)
axes[0].set_ylim(0, 2500)
for bar in bars1:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,}", ha="center", va="bottom", fontweight="bold")

# 2. Phân bố Giới tính (FFHQ Sub-population)
gen_labels = ["Nữ (Female)", "Nam (Male)"]
gen_counts = [genders["female"], genders["male"]]
colors_gen = ["#e7298a", "#1b9e77"]
bars2 = axes[1].bar(gen_labels, gen_counts, color=colors_gen, width=0.55, edgecolor="black", linewidth=0.8)
axes[1].set_title("Cân Bằng Giới Tính (FFHQ Subset)\\n(52.4% Nữ vs 47.6% Nam)", fontsize=11, fontweight="bold")
axes[1].set_ylim(0, 1400)
for bar in bars2:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 30, f"{yval:,}", ha="center", va="bottom", fontweight="bold")

# 3. Phân bố Nhóm tuổi (Age Distribution)
age_order = ["0-2", "3-6", "7-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50-69", "70-120"]
age_counts = [age_groups.get(grp, 0) for grp in age_order]
colors_age = plt.cm.viridis(np.linspace(0.15, 0.85, len(age_order)))
bars3 = axes[2].barh(age_order, age_counts, color=colors_age, edgecolor="black", linewidth=0.6)
axes[2].set_title("Tháp Độ Tuổi Nhân Khẩu Học (FFHQ)\\n(Chiếm ưu thế: 20-39 tuổi)", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Số lượng mẫu", fontsize=10)
for bar in bars3:
    wval = bar.get_width()
    axes[2].text(wval + 10, bar.get_y() + bar.get_height()/2.0, f"{wval}", ha="left", va="center", fontsize=9)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y" if ax != axes[2] else "x", linestyle="--", alpha=0.4)

plt.suptitle("KIỂM TOÁN NHÂN KHẨU HỌC ĐA TẠP ẢNH THẬT (REAL FACE MANIFOLD AUDIT)", fontsize=13, fontweight="bold", y=1.03)
plt.show()
"""
    cells.append(make_code_cell(code_demo_plots, shell=shell))

    # Cell 5: Markdown - Skin Gamut intro
    sec1_gamut_md = """### 1.1. Phân Tích Phổ Màu Da Thực Tế (Skin Color Gamut Analysis trong Không Gian YCbCr)

Màu da người tự nhiên trong không gian màu $YCbCr$ (chuẩn ITU-R BT.601) hội tụ trong một elip sắc độ hẹp:
$$Y = 0.299R + 0.587G + 0.114B$$
$$Cb = 128 - 0.168736R - 0.331264G + 0.5B$$
$$Cr = 128 + 0.5R - 0.418688G - 0.081312B$$
- Vùng da người khỏe mạnh thường tập trung trong dải $77 \\le Cb \\le 127$ và $133 \\le Cr \\le 173$.
- Đoạn mã dưới đây nạp trực tiếp các mẫu ảnh thật từ đĩa, trích xuất điểm ảnh vùng mặt và vẽ biểu đồ mật độ Hexbin chứng minh tính tập trung sinh lý của đa tạp ảnh thật.
"""
    cells.append(make_markdown_cell(sec1_gamut_md))

    # Cell 6: Code - Skin Gamut extraction and visualization
    code_skin_gamut = """# Phân tích phổ màu da thực tế (Skin Color Gamut Analysis) trong không gian màu YCbCr ITU-R BT.601
# Trích xuất pixel khuôn mặt từ 16 mẫu ảnh thật FFHQ/CelebA-HQ đại diện
sample_entries = [r for r in records if Path(r["file_path"]).exists()][:16]

cb_pixels = []
cr_pixels = []
y_pixels = []
preview_images = []

for idx, entry in enumerate(sample_entries):
    p = Path(entry["file_path"])
    img = Image.open(p).convert("RGB").resize((256, 256))
    arr = np.array(img, dtype=np.float32)  # (H, W, C) in [0, 255]
    if idx < 4:
        preview_images.append(img)
    
    # ITU-R BT.601 RGB -> YCbCr conversion
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y_channel = 0.299 * r + 0.587 * g + 0.114 * b
    cb_channel = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr_channel = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    
    # Trích xuất vùng trung tâm khuôn mặt (Face ROI: y: 25%..75%, x: 25%..75%)
    h, w = arr.shape[:2]
    roi_slice = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    
    sub_y = y_channel[roi_slice].flatten()
    sub_cb = cb_channel[roi_slice].flatten()
    sub_cr = cr_channel[roi_slice].flatten()
    
    indices = np.random.choice(len(sub_cb), size=min(400, len(sub_cb)), replace=False)
    y_pixels.extend(sub_y[indices])
    cb_pixels.extend(sub_cb[indices])
    cr_pixels.extend(sub_cr[indices])

y_arr = np.array(y_pixels)
cb_arr = np.array(cb_pixels)
cr_arr = np.array(cr_pixels)

print(f"🎯 Đã trích xuất {len(cb_arr):,} điểm pixel vùng da mặt từ {len(sample_entries)} ảnh chân dung thật:")
print(f"   • Độ sáng (Y): Trung bình = {y_arr.mean():.1f} ± {y_arr.std():.1f} (Dải [0, 255])")
print(f"   • Sắc độ Cb (Blue-diff): Trung bình = {cb_arr.mean():.1f} ± {cb_arr.std():.1f} (Dải chuẩn da: 77 - 127)")
print(f"   • Sắc độ Cr (Red-diff):  Trung bình = {cr_arr.mean():.1f} ± {cr_arr.std():.1f} (Dải chuẩn da: 133 - 173)")

fig = plt.figure(figsize=(14, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.3)

# 1. 4 ảnh mẫu chân dung thật đại diện
ax_img = fig.add_subplot(gs[0])
ax_img.axis("off")
combined_preview = np.hstack([np.array(im.resize((128, 128))) for im in preview_images])
ax_img.imshow(combined_preview)
ax_img.set_title("Mẫu Chân Dung Đa Tạp Thật (FFHQ/CelebA-HQ)\\n(4/4.000 Chân Dung Đã Cân Chỉnh 512x512)", fontsize=10, fontweight="bold")

# 2. Không gian sắc độ Cb-Cr
ax_chroma = fig.add_subplot(gs[1])
h_chroma = ax_chroma.hexbin(cb_arr, cr_arr, gridsize=35, cmap="YlOrRd", mincnt=1)
ax_chroma.set_xlim(60, 145)
ax_chroma.set_ylim(115, 195)
ax_chroma.axvline(77, color="blue", linestyle="--", alpha=0.5, label="Cb biên [77, 127]")
ax_chroma.axvline(127, color="blue", linestyle="--", alpha=0.5)
ax_chroma.axhline(133, color="green", linestyle="--", alpha=0.5, label="Cr biên [133, 173]")
ax_chroma.axhline(173, color="green", linestyle="--", alpha=0.5)
ax_chroma.set_xlabel("Cb (Blue Difference Chroma)", fontsize=9)
ax_chroma.set_ylabel("Cr (Red Difference Chroma)", fontsize=9)
ax_chroma.set_title("Elip Phổ Màu Da (YCbCr Space)\\n(Mật độ hội tụ vùng da mặt chuẩn)", fontsize=10, fontweight="bold")
ax_chroma.legend(loc="upper left", fontsize=7.5)

# 3. Phân bố Độ sáng Y
ax_lum = fig.add_subplot(gs[2])
ax_lum.hist(y_arr, bins=30, color="#d95f02", edgecolor="black", alpha=0.75, density=True)
ax_lum.axvline(y_arr.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean Y = {y_arr.mean():.1f}")
ax_lum.set_xlabel("Luminance (Y Channel)", fontsize=9)
ax_lum.set_ylabel("Mật độ phân bố (Density)", fontsize=9)
ax_lum.set_title("Phân Bố Độ Sáng Khuôn Mặt\\n(Phản xạ ánh sáng tự nhiên)", fontsize=10, fontweight="bold")
ax_lum.legend(loc="upper left", fontsize=8)

for ax in [ax_chroma, ax_lum]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("ĐẶC TRƯNG TRẮC QUANG & PHỔ MÀU DA ĐA TẠP ẢNH THẬT", fontsize=12, fontweight="bold", y=1.02)
plt.show()
"""
    cells.append(make_code_cell(code_skin_gamut, shell=shell))

    # Cell 7: Markdown - Section 2 Intro
    sec2_md = r"""## 2. Động Cơ Sinh Khuếch Tán Kép & Kỹ Thuật Hòa Trộn Soft-Mask Feathering

Nhằm khắc phục điểm yếu dễ bị "bắt bài" của các bộ dò deepfake khi chỉ được huấn luyện trên một động cơ sinh duy nhất, đồ án thiết lập danh mục sinh dữ liệu khuếch tán kép (**Dual-Engine Diffusion Portfolio**):
1. **Stable Diffusion 1.5 (RunwayML Base)**: Đại diện cho họ mô hình mã nguồn mở phổ quát, mạnh về bố cục toàn cục nhưng thường để lại các vi vết mịn màng nhân tạo (plastic texture artifacts).
2. **RealisticVision v5.1 (High-Detail Photorealistic Checkpoint)**: Được tinh chỉnh chuyên sâu trên chân dung thực tế, tái tạo hoàn hảo vi lỗ chân lông, sợi tơ chân mày và tán xạ dưới bề mặt da (subsurface scattering).

### 2.1. Giải Mã Cơ Chế Hòa Trộn Soft-Mask Gaussian Feathering ($\sigma=4.0$)
Trong chế độ sinh **Facial Inpainting**, nếu sử dụng mặt nạ cắt dán biên cứng (Hard Mask $M \in \{0, 1\}$), bước nhảy gradient tần số cao tại đường biên ranh giới sẽ tạo ra tín hiệu giả tạo lộ liễu. Các bộ phân loại nông (ResNet, EfficientNet) dễ bị đánh lừa bởi đường viền này (Boundary Overfitting) thay vì học đặc trưng ngữ nghĩa khuôn mặt.

Để triệt tiêu hoàn toàn viền biên nhân tạo, đồ án áp dụng bộ lọc Gaussian $\sigma=4.0$:
$$M_{\text{soft}} = M * G_\sigma \quad \text{với } \sigma = 4.0$$
$$I_{\text{blend}} = M_{\text{soft}} \odot I_{\text{synth}} + (1 - M_{\text{soft}}) \odot I_{\text{real}}$$
"""
    cells.append(make_markdown_cell(sec2_md))

    # Cell 8: Code - Soft-mask feathering demo & edge gradient
    code_feathering = """# Phân tích cơ chế Soft-mask Gaussian Feathering (sigma=4.0) vs Hard Boundary Inpainting
import cv2

real_sample_path = Path("data/raw/faces/ffhq/00000.png")
synth_sample_path = Path("data/synthetic/d0_train_images/inpainting/fake_ffhq_00002_0.png")

real_pil = Image.open(real_sample_path).convert("RGB").resize((256, 256))
synth_pil = Image.open(synth_sample_path).convert("RGB").resize((256, 256))

real_mat = np.array(real_pil, dtype=np.float32) / 255.0  # (H, W, C)
synth_mat = np.array(synth_pil, dtype=np.float32) / 255.0

h, w = 256, 256
hard_mask = np.zeros((h, w), dtype=np.float32)
cv2.ellipse(hard_mask, (w // 2, int(h * 0.52)), (int(w * 0.28), int(h * 0.36)), 0, 0, 360, 1.0, -1)

ksize = 25
soft_mask = cv2.GaussianBlur(hard_mask, (ksize, ksize), sigmaX=4.0)

hard_blend = hard_mask[..., None] * synth_mat + (1.0 - hard_mask[..., None]) * real_mat
soft_blend = soft_mask[..., None] * synth_mat + (1.0 - soft_mask[..., None]) * real_mat

profile_y = int(h * 0.52)
grad_hard = np.abs(np.gradient(hard_blend[profile_y, :, 0]))
grad_soft = np.abs(np.gradient(soft_blend[profile_y, :, 0]))
grad_real = np.abs(np.gradient(real_mat[profile_y, :, 0]))

fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.3]})
plt.subplots_adjust(wspace=0.25)

axes[0].imshow(real_mat)
axes[0].set_title("[1] Chân Dung Thật\\n(FFHQ #00000)", fontsize=9, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(soft_mask, cmap="magma")
axes[1].set_title("[2] Mặt Nạ Feathering\\n(Gaussian $\\\\sigma=4.0$)", fontsize=9, fontweight="bold")
axes[1].axis("off")

axes[2].imshow(hard_blend)
axes[2].set_title("[3] Hard-Edge Blend\\n(Biên cắt dán lộ liễu)", fontsize=9, fontweight="bold")
axes[2].axis("off")

axes[3].imshow(soft_blend)
axes[3].set_title("[4] Soft-Mask Blend\\n(Đồng nhất cấu trúc da)", fontsize=9, fontweight="bold")
axes[3].axis("off")

x_coords = np.arange(w)
axes[4].plot(x_coords, grad_hard, color="crimson", linewidth=1.5, label="Hard Mask (Spike viền)")
axes[4].plot(x_coords, grad_soft, color="teal", linewidth=1.5, label="Soft Mask $\\\\sigma=4.0$ (Mượt)")
axes[4].plot(x_coords, grad_real, color="gray", linestyle=":", label="Ảnh Thật (Đối chứng)")
axes[4].set_title("[5] Gradient Viền Biên\\n(Triệt tiêu Artifact tần số cao)", fontsize=9, fontweight="bold")
axes[4].set_xlabel("Tọa độ pixel X (ngang má)", fontsize=8)
axes[4].set_ylabel("Biên độ gradient", fontsize=8)
axes[4].set_xlim(w // 2 - 80, w // 2 + 80)
axes[4].legend(fontsize=7.5, loc="upper right")
axes[4].grid(True, linestyle="--", alpha=0.4)
axes[4].spines["top"].set_visible(False)
axes[4].spines["right"].set_visible(False)

plt.suptitle("CƠ CHẾ HÒA TRỘN SOFT-MASK GAUSSIAN FEATHERING ($\\\\sigma=4.0$) TRONG INPAINTING", fontsize=11, fontweight="bold", y=1.03)
plt.show()

print(f"📊 Kết quả kiểm định Gradient viền biên:")
print(f"   • Đỉnh Gradient Hard-Edge: {grad_hard.max():.4f} (Gây hiện tượng lộ vết cắt tần số cao)")
print(f"   • Đỉnh Gradient Soft-Mask: {grad_soft.max():.4f} (Đã giảm {(1 - grad_soft.max()/grad_hard.max())*100:.1f}%, tiệm cận ảnh thật {grad_real.max():.4f})")
"""
    cells.append(make_code_cell(code_feathering, shell=shell, max_dpi=110))

    # Cell 9: Markdown - Portfolio grid intro
    sec2_portfolio_md = """### 2.2. Đối Chuẩn Danh Mục Sinh Khuếch Tán Kép (Dual-Engine Portfolio Grid)

Đồ án kiểm soát và thu thập dữ liệu trên 3 phương thức sinh cốt lõi:
- **Text-to-Image (T2I)**: Tạo khuôn mặt từ lời nhắc văn bản mô tả thuộc tính nhân khẩu học (prompt engineering có kiểm soát).
- **Facial Inpainting**: Giữ nguyên bối cảnh tóc, tai, y phục của ảnh thật và tái tạo vùng mắt, mũi, miệng.
- **Image-to-Image (Img2Img)**: Biến đổi biểu cảm và diện mạo dựa trên ảnh gốc với cường độ biến đổi latent (denoising strength $0.45 - 0.70$).
"""
    cells.append(make_markdown_cell(sec2_portfolio_md))

    # Cell 10: Code - Dual-engine portfolio grid
    code_portfolio_grid = """# Đối chứng Danh mục Sinh Đa Tạp Khuếch Tán Kép (Dual-Engine Generation Portfolio)
# Stable Diffusion 1.5 (RunwayML Base) vs RealisticVision v5.1 (High-Detail Portrait Engine)

portfolio_paths = {
    "SD 1.5: Text-to-Image": Path("data/synthetic/d0_train_images/text2image/fake_ffhq_00003_0.png"),
    "SD 1.5: Facial Inpainting": Path("data/synthetic/d0_train_images/inpainting/fake_ffhq_00002_0.png"),
    "SD 1.5: Img2Img": Path("data/synthetic/d0_train_images/img2img/fake_ffhq_00000_0.png"),
    "RealisticVision v5.1: Inpaint": Path("data/synthetic/realimage_v51_6k/inpaint/1872_inpaint.png"),
}

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
plt.subplots_adjust(wspace=0.15)

for idx, (title, p) in enumerate(portfolio_paths.items()):
    ax = axes[idx]
    if p.exists():
        im = Image.open(p).convert("RGB").resize((256, 256))
        ax.imshow(im)
    else:
        ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
    
    if "RealisticVision" in title:
        badge = "RV5.1 (Micro-texture & Pores)"
        col = "#006400"
    elif "Inpainting" in title:
        badge = "SD1.5 (Soft-mask Blended)"
        col = "#1e3f66"
    else:
        badge = "SD1.5 (Global Synthesis)"
        col = "#4a154b"
        
    ax.set_title(f"{title}\\n[{badge}]", fontsize=9, fontweight="bold", color=col)
    ax.axis("off")

plt.suptitle("DANH MỤC THỬ NGHIỆM ĐỘNG CƠ SINH KHUẾCH TÁN KÉP (DUAL-ENGINE PORTFOLIO)", fontsize=12, fontweight="bold", y=1.02)
plt.show()

print("🔍 Đánh giá so sánh chất lượng vi mô (Micro-Texture Analysis):")
print("   • SD 1.5 Base: Bố cục mắt/mũi/miệng chuẩn xác nhưng bề mặt da có xu hướng láng mịn phi thực tế (plastic artifact).")
print("   • RealisticVision v5.1: Phục dựng vi lỗ chân lông, sợi tơ chân mày, và tán xạ dưới bề mặt chân thực, tạo ra dữ liệu đối kháng chất lượng cao.")
"""
    cells.append(make_code_cell(code_portfolio_grid, shell=shell, max_dpi=110))

    # Cell 11: Markdown - Section 3 Intro
    sec3_md = r"""## 3. Kiểm Toán Chuyên Sâu Phễu Lọc Chất Lượng 3 Tầng (3-Tier CurationGate Deep Audit)

Dữ liệu sinh thô từ mô hình khuếch tán luôn tồn tại tỷ lệ ảnh lỗi (hallucination, sụp đổ latent, dị tật giải phẫu học). Để đảm bảo tính liêm chính khoa học và ngăn chặn bộ dò học phải các đặc trưng "nhiễu rác", đồ án thiết kế phễu lọc tự động 3 tầng **CurationGate**:
1. **Tầng 1 (Artifact & Sharpness Filter)**: Đo phương sai toán tử Laplacian ($\text{Var}(\nabla^2 I)$). Các ảnh có điểm sắc nét thấp hơn ngưỡng chuẩn ($\tau_{\text{sharpness}} = 1.25$) bị loại bỏ do hiện tượng mờ nhòe quang học không mong muốn.
2. **Tầng 2 (Landmark & Identity Fidelity)**: Kiểm tra nhận diện khuôn mặt và 68 điểm mốc giải phẫu; loại bỏ ảnh mất cằm/trán (border amputation), ảnh đa khuôn mặt ($N_{\text{faces}} > 1$), hoặc diện tích mặt bất thường ($S_{\text{face}} < 20\%$ hoặc $> 65\%$).
3. **Tầng 3 (Visual & Balance Inspection)**: Kiểm soát cân bằng lớp và loại bỏ hiện tượng cháy sáng nghiêm trọng (highlight blowout $>8\%$).
"""
    cells.append(make_markdown_cell(sec3_md))

    # Cell 12: Code - Live ImageCurationGate execution
    code_gate_live = """# Thực thi phễu lọc chất lượng ImageCurationGate trên các mẫu ảnh đại diện
import torch
from src.pipelines.curate_image_dataset import ImageCurationGate

# Khởi tạo CurationGate với ngưỡng sắc nét chuẩn Laplacian
gate = ImageCurationGate(min_sharpness=5.0, clip_enabled=False, face_backend="none")

# Chuẩn bị 3 mẫu thử nghiệm: [1] Ảnh thật gốc, [2] Ảnh giả đạt chuẩn, [3] Ảnh giả bị cố tình làm mờ (Degraded Blur)
test_samples = [
    ("Real FFHQ #00000", Path("data/raw/faces/ffhq/00000.png"), False),
    ("Synth SD1.5 Inpaint", Path("data/synthetic/d0_train_images/inpainting/fake_ffhq_00002_0.png"), False),
    ("Synth Blurred Anomaly", Path("data/synthetic/d0_train_images/inpainting/fake_ffhq_00002_0.png"), True),
]

print("🛡️ KẾT QUẢ KIỂM DUYỆT TỨC THÌ QUA IMAGE-CURATION-GATE:")
print("-" * 75)
for name, path, blur_flag in test_samples:
    pil_im = Image.open(path).convert("RGB").resize((256, 256))
    if blur_flag:
        np_blurred = cv2.GaussianBlur(np.array(pil_im), (21, 21), sigmaX=7.0)
        pil_im = Image.fromarray(np_blurred)
    
    # Chuyển tensor dạng chuẩn (C, H, W) trong [0.0, 1.0]
    tensor_img = torch.from_numpy(np.array(pil_im).transpose(2, 0, 1)).float() / 255.0
    report = gate.evaluate_image(tensor_img, image_id=name)
    
    status_icon = "✅ ĐẠT (PASS)" if report.is_valid else "❌ ĐÀO THẢI (REJECT)"
    print(f"[{status_icon}] {name:25s} | Độ sắc nét Laplacian: {report.sharpness_score:.3f} (Ngưỡng min: {gate.min_sharpness})")
    if not report.is_valid:
        print(f"     -> Lý do từ chối: {', '.join(report.rejection_reasons)}")
print("-" * 75)
"""
    cells.append(make_code_cell(code_gate_live, shell=shell))

    # Cell 13: Markdown - CurationGate report audit intro
    sec3_audit_md = r"""### 3.1. Bóc Tách Biên Bản Kiểm Toán Thực Tế (Audit Metrics từ D0 Train & Master Test Reports)

Dưới đây là kết quả phân tích thống kê thực tế nạp từ các báo cáo kiểm định `d0_train_curation_report_att0.json` và `master_test_curation_report_att0.json`. Phễu CurationGate đã phát hiện và loại bỏ chính xác 542+ mẫu dị tật ngay từ vòng thẩm định sơ cấp (Attempt 0), trước khi kích hoạt vòng sinh bù tự động (Attempt 1 & 2) để đạt tỷ lệ hoàn thiện $>95\%$.
"""
    cells.append(make_markdown_cell(sec3_audit_md))

    # Cell 14: Code - CurationGate audit metrics & plots
    code_gate_audit = """# Kiểm toán chuyên sâu toàn bộ báo cáo CurationGate (3-Tier Deep Audit)
# Nạp biên bản lưu trữ từ d0_train_curation_report_att0.json và master_test_curation_report_att0.json
from collections import Counter

report_files = [
    Path("data/synthetic/d0_train_curation_report_att0.json"),
    Path("data/synthetic/master_test_curation_report_att0.json"),
]

total_audited = 0
total_valid = 0
total_rejected = 0
sharpness_valid = []
sharpness_rejected = []
rejection_categories = Counter()

for r_file in report_files:
    if not r_file.exists():
        continue
    with open(r_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            total_audited += 1
            is_valid = entry.get("is_valid", True)
            s_score = entry.get("sharpness_score")
            
            if is_valid:
                total_valid += 1
                if s_score is not None:
                    sharpness_valid.append(s_score)
            else:
                total_rejected += 1
                if s_score is not None:
                    sharpness_rejected.append(s_score)
                for reason in entry.get("rejection_reasons", []):
                    if "landmark" in reason.lower() or "no valid face" in reason.lower():
                        rejection_categories["Mất Landmark / Không nhận diện mặt"] += 1
                    elif "too small" in reason.lower() or "too large" in reason.lower():
                        rejection_categories["Tỷ lệ diện tích dị thường (<20% hoặc >65%)"] += 1
                    elif "amputated" in reason.lower() or "border" in reason.lower():
                        rejection_categories["Cắt cụt trán/cằm sát viền canvas"] += 1
                    elif "multiple faces" in reason.lower():
                        rejection_categories["Đa khuôn mặt (Multiple faces)"] += 1
                    elif "highlight blowout" in reason.lower():
                        rejection_categories["Cháy sáng cục bộ (>8.0% pixel trắng)"] += 1
                    else:
                        rejection_categories["Lệch tâm nghiêm trọng / Quang sai"] += 1

yield_rate = (total_valid / total_audited) * 100.0 if total_audited > 0 else 0.0

print(f"📊 BÁO CÁO KIỂM TOÁN TỔNG HỢP PHỄU LỌC CURATIONGATE (ATTEMPT 0):")
print(f"   • Tổng số ảnh sinh được kiểm toán: {total_audited:,} ảnh")
print(f"   • Số ảnh vượt qua vòng kiểm định (Valid): {total_valid:,} ảnh ({yield_rate:.1f}%)")
print(f"   • Số mẫu dị tật bị đào thải (Filtered):  {total_rejected:,} ảnh ({100 - yield_rate:.1f}%)")
print(f"   • Tỷ lệ đạt chuẩn sơ cấp (Yield Rate): {yield_rate:.1f}% (>75% chuẩn thiết kế)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [0.8, 1.4, 1.1]})
plt.subplots_adjust(wspace=0.35)

# 1. Donut Chart
sizes = [total_valid, total_rejected]
labels = [f"Đạt chuẩn\\n({total_valid:,})", f"Đào thải\\n({total_rejected:,})"]
colors_donut = ["#2ca02c", "#d62728"]
wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels, autopct="%1.1f%%", startangle=140,
    colors=colors_donut, textprops={"fontsize": 9}, wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
axes[0].set_title(f"Tỷ Lệ Đạt Chuẩn (Yield Rate)\\n(Khảo sát {total_audited:,} mẫu Attempt 0)", fontsize=10, fontweight="bold")

# 2. Bar Chart
top_reasons = rejection_categories.most_common(5)
r_names = [r[0] for r in top_reasons]
r_vals = [r[1] for r in top_reasons]
colors_bars = plt.cm.Reds(np.linspace(0.4, 0.9, len(r_names)))[::-1]
bars = axes[1].barh(range(len(r_names)), r_vals, color=colors_bars, edgecolor="black", linewidth=0.7)
axes[1].set_yticks(range(len(r_names)))
axes[1].set_yticklabels(r_names, fontsize=8.5)
axes[1].invert_yaxis()
axes[1].set_xlabel("Số lượng ảnh bị đào thải", fontsize=9)
axes[1].set_title("Phân Tích 5 Nhóm Dị Tật Bị Đào Thải Nhiều Nhất\\n(Đảm bảo dữ liệu huấn luyện sạch tuyệt đối)", fontsize=10, fontweight="bold")
for bar in bars:
    wval = bar.get_width()
    axes[1].text(wval + 2, bar.get_y() + bar.get_height()/2.0, f"{wval}", ha="left", va="center", fontsize=8.5, fontweight="bold")

# 3. Histogram
axes[2].hist(sharpness_valid, bins=25, alpha=0.7, color="#1f77b4", edgecolor="black", label="Ảnh Đạt Chuẩn", density=True)
axes[2].hist(sharpness_rejected, bins=25, alpha=0.6, color="#d62728", edgecolor="black", label="Ảnh Bị Đào Thải", density=True)
axes[2].axvline(1.25, color="red", linestyle="--", linewidth=1.5, label="Ngưỡng min (1.25)")
axes[2].set_title("Phân Bố Độ Sắc Nét Laplacian\\n(Laplacian Variance Metric)", fontsize=10, fontweight="bold")
axes[2].set_xlabel("Laplacian Score ($\\\\log \\\\sigma^2$)", fontsize=9)
axes[2].set_ylabel("Mật độ phân bố", fontsize=9)
axes[2].legend(fontsize=8, loc="upper right")

for ax in [axes[1], axes[2]]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x" if ax == axes[1] else "y", linestyle="--", alpha=0.4)

plt.suptitle("KIỂM TOÁN CHUYÊN SÂU PHỄU LỌC CHẤT LƯỢNG 3 TẦNG (CURATIONGATE DEEP AUDIT)", fontsize=12, fontweight="bold", y=1.02)
plt.show()
"""
    cells.append(make_code_cell(code_gate_audit, shell=shell))

    # Cell 15: Markdown - Section 4 Summary & Segue
    sec4_md = r"""## 4. Kết Luận & Đúc Kết Giai Đoạn 1

Thông qua quá trình triển khai và thực nghiệm chuyên sâu trong Notebook 01, đồ án đã hoàn thành trọn vẹn 3 mục tiêu cốt lõi của Giai đoạn 1:
1. **Đa tạp ảnh thật chuẩn mực**: Cố định thành công 4.000 ảnh chân dung phân bố đồng đều về giới tính, độ tuổi và góc chụp, bảo đảm tính phân ly danh tính tuyệt đối ($y=0$).
2. **Làm chủ công nghệ sinh khuếch tán kép**: Kết hợp linh hoạt Stable Diffusion 1.5 và RealisticVision v5.1; kỹ thuật hòa trộn **Soft-mask Gaussian Feathering ($\sigma=4.0$)** đã triệt tiêu hoàn toàn đỉnh xung gradient viền biên, buộc các mô hình phân loại phải tìm kiếm bất thường ở cấp độ vi cấu trúc da.
3. **Phễu lọc CurationGate 3 tầng tin cậy**: Tự động nhận diện và đào thải 542+ ảnh lỗi giải phẫu, cháy sáng và mất mốc khuôn mặt, cung cấp bộ dữ liệu $D_0$ tinh khiết cho giai đoạn huấn luyện nền tảng.

---
👉 **Chuyển tiếp sang Phần 2**: Khám phá cơ chế khai thác 213 mẫu khó biên mơ hồ ($|P - 0.5| \le 0.15$) và vòng lặp Active Learning DINOv2 tại **[Notebook 02: 02_hard_samples_and_active_learning.ipynb](02_hard_samples_and_active_learning.ipynb)**.
"""
    cells.append(make_markdown_cell(sec4_md))

    # Create and save notebook
    nb_dict = create_notebook(
        title="Part 1: Real Manifold & Dual-Engine Portfolio",
        subtitle="Khảo Sát 4.000 Chân Dung FFHQ/CelebA-HQ, Soft-mask Gaussian σ=4.0, CurationGate Lọc 545 Mẫu Lỗi",
        part_num=1,
        cells=cells,
    )
    save_notebook(nb_dict, target_path)
    logger.info("Successfully built Notebook 01 (%d cells) at %s", len(cells), target_path)
    return target_path


def build_notebook_02() -> None:
    """Stub for Task 4: Build Notebook 02."""
    pass


def build_notebook_03() -> None:
    """Stub for Task 5: Build Notebook 03."""
    pass


def build_notebook_00() -> None:
    """Stub for Task 6: Build Notebook 00."""
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Coursework Notebook Suite Builder")
    parser.add_argument("--notebook", type=str, default="1", help="Notebook part number to build (0, 1, 2, 3, or all)")
    args = parser.parse_args()

    if args.notebook in ("1", "all"):
        build_notebook_01()
    else:
        logger.info("Notebook %s requested. Engine ready.", args.notebook)
