"""Tests for generated coursework notebooks outputs, structure, and integrity.

Ensures generated notebooks strictly conform to nbformat v4, size constraints (<2.5MB),
clean execution without errors, and zero leaks of absolute paths or usernames.
"""

import json
from pathlib import Path
import re
import pytest

NOTEBOOK_01_PATH = Path("notebooks/coursework/01_diffusion_dataset_and_curation.ipynb")


@pytest.mark.smoke
def test_notebook_01_exists_and_size():
    """Verify Notebook 01 exists on disk and is strictly below the 2.5MB ceiling."""
    assert NOTEBOOK_01_PATH.is_file(), f"Notebook 01 file does not exist at {NOTEBOOK_01_PATH}"
    file_size_bytes = NOTEBOOK_01_PATH.stat().st_size
    size_mb = file_size_bytes / (1024 * 1024)
    assert size_mb < 2.5, f"Notebook 01 size {size_mb:.2f}MB exceeds 2.5MB threshold"
    assert file_size_bytes > 10000, f"Notebook 01 suspiciously small: {file_size_bytes} bytes"


@pytest.mark.smoke
def test_notebook_01_schema_and_navigation():
    """Verify nbformat v4 validity, institutional Cell 0 header, and active Cell 1 roadmap."""
    assert NOTEBOOK_01_PATH.is_file(), f"Notebook 01 file missing at {NOTEBOOK_01_PATH}"

    with open(NOTEBOOK_01_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4, f"Expected nbformat 4, got {nb.get('nbformat')}"
    cells = nb.get("cells", [])
    assert len(cells) >= 6, f"Expected at least 6 cells in Notebook 01, found {len(cells)}"

    # Cell 0: Header
    cell0 = cells[0]
    assert cell0["cell_type"] == "markdown"
    cell0_text = "".join(cell0["source"])
    assert "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM" in cell0_text
    assert "ĐỀ TÀI 11" in cell0_text
    assert "PHẦN 1" in cell0_text

    # Cell 1: Roadmap
    cell1 = cells[1]
    assert cell1["cell_type"] == "markdown"
    cell1_text = "".join(cell1["source"])
    assert "Lộ Trình" in cell1_text or "Roadmap" in cell1_text
    assert "Phần 1" in cell1_text
    assert "🎯" in cell1_text or "(Đang xem)" in cell1_text


@pytest.mark.smoke
def test_notebook_01_execution_outputs():
    """Verify all code cells executed with zero errors, producing stream and display_data outputs."""
    assert NOTEBOOK_01_PATH.is_file(), f"Notebook 01 file missing at {NOTEBOOK_01_PATH}"

    with open(NOTEBOOK_01_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    raw_nb_content = NOTEBOOK_01_PATH.read_text(encoding="utf-8")
    assert "/home/bush" not in raw_nb_content, "Found leaked user home directory in notebook JSON"
    assert "file:///" not in raw_nb_content, "Found leaked file:/// scheme in notebook JSON"

    code_cells = [c for c in nb.get("cells", []) if c["cell_type"] == "code"]
    assert len(code_cells) >= 3, f"Expected at least 3 executable code cells, found {len(code_cells)}"

    has_display_data = False
    has_stream_data = False

    for idx, cell in enumerate(code_cells):
        src = "".join(cell["source"]).strip()
        assert src != "plt.show()", f"Code cell {idx} is a bare plt.show() cell"
        outputs = cell.get("outputs", [])
        assert len(outputs) > 0, f"Code cell {idx} has no outputs: {src[:50]}..."

        for out in outputs:
            assert out.get("output_type") != "error", (
                f"Code cell {idx} failed with execution error: {out.get('ename')}: {out.get('evalue')}"
            )
            if out.get("output_type") == "display_data":
                data = out.get("data", {})
                if "image/png" in data:
                    has_display_data = True
                    assert len(data["image/png"]) > 200, f"Malformed image data in cell {idx}"
            if out.get("output_type") == "stream":
                has_stream_data = True

    assert has_display_data, "Notebook 01 should contain at least one matplotlib display_data visualization"
    assert has_stream_data, "Notebook 01 should contain stream stdout log outputs"
