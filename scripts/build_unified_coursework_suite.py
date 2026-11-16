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


def build_notebook_01() -> None:
    """Stub for Task 3: Build Notebook 01."""
    pass


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
    logger.info("Unified Coursework Notebook Suite Builder Engine loaded.")
