"""Tests for generated coursework notebooks outputs, structure, and integrity.

Ensures generated notebooks strictly conform to nbformat v4, size constraints (<2.5MB),
clean execution without errors, and zero leaks of absolute paths or usernames.
"""

import json
from pathlib import Path
import re
import pytest

NOTEBOOK_00_PATH = Path("notebooks/coursework/00_master_deepfake_defense_showcase.ipynb")
NOTEBOOK_01_PATH = Path("notebooks/coursework/01_diffusion_dataset_and_curation.ipynb")
NOTEBOOK_02_PATH = Path("notebooks/coursework/02_hard_samples_and_active_learning.ipynb")
NOTEBOOK_03_PATH = Path("notebooks/coursework/03_champion_m3_and_universal_benchmark.ipynb")


@pytest.mark.smoke
def test_notebook_00_exists_and_size():
    """Verify Notebook 00 exists on disk and is strictly below the 2.5MB ceiling."""
    assert NOTEBOOK_00_PATH.is_file(), f"Notebook 00 file does not exist at {NOTEBOOK_00_PATH}"
    file_size_bytes = NOTEBOOK_00_PATH.stat().st_size
    size_mb = file_size_bytes / (1024 * 1024)
    assert size_mb < 2.5, f"Notebook 00 size {size_mb:.2f}MB exceeds 2.5MB threshold"
    assert file_size_bytes > 10000, f"Notebook 00 suspiciously small: {file_size_bytes} bytes"


@pytest.mark.smoke
def test_notebook_00_schema_and_navigation():
    """Verify nbformat v4 validity, institutional Cell 0 header, and active Cell 1 roadmap for Part 0."""
    assert NOTEBOOK_00_PATH.is_file(), f"Notebook 00 file missing at {NOTEBOOK_00_PATH}"

    with open(NOTEBOOK_00_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4, f"Expected nbformat 4, got {nb.get('nbformat')}"
    cells = nb.get("cells", [])
    assert len(cells) >= 6, f"Expected at least 6 cells in Notebook 00, found {len(cells)}"

    # Cell 0: Header
    cell0 = cells[0]
    assert cell0["cell_type"] == "markdown"
    cell0_text = "".join(cell0["source"])
    assert "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM" in cell0_text
    assert "ĐỀ TÀI 11" in cell0_text
    assert "PHẦN 0" in cell0_text or "MASTER" in cell0_text or "SHOWCASE" in cell0_text

    # Cell 1: Roadmap
    cell1 = cells[1]
    assert cell1["cell_type"] == "markdown"
    cell1_text = "".join(cell1["source"])
    assert "Lộ Trình" in cell1_text or "Roadmap" in cell1_text
    assert "Phần 0" in cell1_text
    assert "🎯" in cell1_text or "(Đang xem)" in cell1_text


@pytest.mark.smoke
def test_notebook_00_execution_outputs():
    """Verify all code cells executed with zero errors, zero SyntaxWarning, and no local path leaks."""
    assert NOTEBOOK_00_PATH.is_file(), f"Notebook 00 file missing at {NOTEBOOK_00_PATH}"

    with open(NOTEBOOK_00_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    raw_nb_content = NOTEBOOK_00_PATH.read_text(encoding="utf-8")
    assert "/home/bush" not in raw_nb_content, "Found leaked user home directory in notebook JSON"
    assert "file:///" not in raw_nb_content, "Found leaked file:/// scheme in notebook JSON"

    code_cells = [c for c in nb.get("cells", []) if c["cell_type"] == "code"]
    assert len(code_cells) >= 3, f"Expected at least 3 executable code cells, found {len(code_cells)}"

    code_text = " ".join("".join(c["source"]) for c in code_cells)
    assert "active_learning_m3_best.pt" in code_text, "Missing M3 checkpoint reference in code cells"

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
                stream_text = "".join(out.get("text", []))
                assert "SyntaxWarning" not in stream_text, f"Cell {idx} contains SyntaxWarning in output stream: {stream_text}"

    assert has_display_data, "Notebook 00 should contain at least one matplotlib display_data visualization"
    assert has_stream_data, "Notebook 00 should contain stream stdout log outputs"



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
                stream_text = "".join(out.get("text", []))
                assert "SyntaxWarning" not in stream_text, f"Cell {idx} contains SyntaxWarning in output stream: {stream_text}"

    assert has_display_data, "Notebook 01 should contain at least one matplotlib display_data visualization"
    assert has_stream_data, "Notebook 01 should contain stream stdout log outputs"


@pytest.mark.smoke
def test_notebook_02_exists_and_size():
    """Verify Notebook 02 exists on disk and is strictly below the 2.5MB ceiling."""
    assert NOTEBOOK_02_PATH.is_file(), f"Notebook 02 file does not exist at {NOTEBOOK_02_PATH}"
    file_size_bytes = NOTEBOOK_02_PATH.stat().st_size
    size_mb = file_size_bytes / (1024 * 1024)
    assert size_mb < 2.5, f"Notebook 02 size {size_mb:.2f}MB exceeds 2.5MB threshold"
    assert file_size_bytes > 10000, f"Notebook 02 suspiciously small: {file_size_bytes} bytes"


@pytest.mark.smoke
def test_notebook_02_schema_and_navigation():
    """Verify nbformat v4 validity, institutional Cell 0 header, and active Cell 1 roadmap for Part 2."""
    assert NOTEBOOK_02_PATH.is_file(), f"Notebook 02 file missing at {NOTEBOOK_02_PATH}"

    with open(NOTEBOOK_02_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4, f"Expected nbformat 4, got {nb.get('nbformat')}"
    cells = nb.get("cells", [])
    assert len(cells) >= 6, f"Expected at least 6 cells in Notebook 02, found {len(cells)}"

    # Cell 0: Header
    cell0 = cells[0]
    assert cell0["cell_type"] == "markdown"
    cell0_text = "".join(cell0["source"])
    assert "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM" in cell0_text
    assert "ĐỀ TÀI 11" in cell0_text
    assert "PHẦN 2" in cell0_text

    # Cell 1: Roadmap
    cell1 = cells[1]
    assert cell1["cell_type"] == "markdown"
    cell1_text = "".join(cell1["source"])
    assert "Lộ Trình" in cell1_text or "Roadmap" in cell1_text
    assert "Phần 2" in cell1_text
    assert "🎯" in cell1_text or "(Đang xem)" in cell1_text


@pytest.mark.smoke
def test_notebook_02_execution_outputs():
    """Verify all code cells executed with zero errors, zero SyntaxWarning, and no local path leaks."""
    assert NOTEBOOK_02_PATH.is_file(), f"Notebook 02 file missing at {NOTEBOOK_02_PATH}"

    with open(NOTEBOOK_02_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    raw_nb_content = NOTEBOOK_02_PATH.read_text(encoding="utf-8")
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
                stream_text = "".join(out.get("text", []))
                assert "SyntaxWarning" not in stream_text, f"Cell {idx} contains SyntaxWarning in output stream: {stream_text}"

    assert has_display_data, "Notebook 02 should contain at least one matplotlib display_data visualization"
    assert has_stream_data, "Notebook 02 should contain stream stdout log outputs"


@pytest.mark.smoke
def test_notebook_03_exists_and_size():
    """Verify Notebook 03 exists on disk and is strictly below the 2.5MB ceiling."""
    assert NOTEBOOK_03_PATH.is_file(), f"Notebook 03 file does not exist at {NOTEBOOK_03_PATH}"
    file_size_bytes = NOTEBOOK_03_PATH.stat().st_size
    size_mb = file_size_bytes / (1024 * 1024)
    assert size_mb < 2.5, f"Notebook 03 size {size_mb:.2f}MB exceeds 2.5MB threshold"
    assert file_size_bytes > 10000, f"Notebook 03 suspiciously small: {file_size_bytes} bytes"


@pytest.mark.smoke
def test_notebook_03_schema_and_navigation():
    """Verify nbformat v4 validity, institutional Cell 0 header, and active Cell 1 roadmap for Part 3."""
    assert NOTEBOOK_03_PATH.is_file(), f"Notebook 03 file missing at {NOTEBOOK_03_PATH}"

    with open(NOTEBOOK_03_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4, f"Expected nbformat 4, got {nb.get('nbformat')}"
    cells = nb.get("cells", [])
    assert len(cells) >= 6, f"Expected at least 6 cells in Notebook 03, found {len(cells)}"

    # Cell 0: Header
    cell0 = cells[0]
    assert cell0["cell_type"] == "markdown"
    cell0_text = "".join(cell0["source"])
    assert "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM" in cell0_text
    assert "ĐỀ TÀI 11" in cell0_text
    assert "PHẦN 3" in cell0_text or "CHAMPION" in cell0_text or "M3" in cell0_text

    # Cell 1: Roadmap
    cell1 = cells[1]
    assert cell1["cell_type"] == "markdown"
    cell1_text = "".join(cell1["source"])
    assert "Lộ Trình" in cell1_text or "Roadmap" in cell1_text
    assert "Phần 3" in cell1_text
    assert "🎯" in cell1_text or "(Đang xem)" in cell1_text


@pytest.mark.smoke
def test_notebook_03_execution_outputs():
    """Verify all code cells executed with zero errors, zero SyntaxWarning, and no local path leaks."""
    assert NOTEBOOK_03_PATH.is_file(), f"Notebook 03 file missing at {NOTEBOOK_03_PATH}"

    with open(NOTEBOOK_03_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    raw_nb_content = NOTEBOOK_03_PATH.read_text(encoding="utf-8")
    assert "/home/bush" not in raw_nb_content, "Found leaked user home directory in notebook JSON"
    assert "file:///" not in raw_nb_content, "Found leaked file:/// scheme in notebook JSON"

    code_cells = [c for c in nb.get("cells", []) if c["cell_type"] == "code"]
    assert len(code_cells) >= 3, f"Expected at least 3 executable code cells, found {len(code_cells)}"

    code_text = " ".join("".join(c["source"]) for c in code_cells)
    assert "active_learning_m3_best.pt" in code_text, "Missing M3 checkpoint reference in code cells"
    assert "unseen_rv51_test.jsonl" in code_text, "Missing unseen_rv51_test.jsonl reference in code cells"

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
                stream_text = "".join(out.get("text", []))
                assert "SyntaxWarning" not in stream_text, f"Cell {idx} contains SyntaxWarning in output stream: {stream_text}"

    assert has_display_data, "Notebook 03 should contain at least one matplotlib display_data visualization"
    assert has_stream_data, "Notebook 03 should contain stream stdout log outputs"

