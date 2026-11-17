"""End-to-End Suite Integrity and Governance Verification for Coursework Notebooks.

Verifies that the entire 4-part unified coursework notebook suite conforms to:
- Institutional standards (Cell 0 UTH header, Cell 1 roadmap navigation).
- Strict size boundaries (< 2.5 MB per notebook, < 10 MB total suite).
- Valid Jupyter nbformat v4 schema.
- Clean execution state with zero errors and zero SyntaxWarnings.
- All relative markdown links (../../src/..., ../../docs/..., sister notebooks) resolve on disk.
- Zero leaks of personal usernames or absolute filesystem paths.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

COURSEWORK_DIR = Path("notebooks/coursework")

EXPECTED_NOTEBOOKS = {
    0: "00_master_deepfake_defense_showcase.ipynb",
    1: "01_diffusion_dataset_and_curation.ipynb",
    2: "02_hard_samples_and_active_learning.ipynb",
    3: "03_champion_m3_and_universal_benchmark.ipynb",
}


@pytest.mark.smoke
def test_all_coursework_notebooks_exist_and_size_bounds():
    """Verify all 4 notebooks exist, each is < 2.5 MB, and total suite is < 10 MB."""
    total_size_bytes = 0

    for part_num, filename in EXPECTED_NOTEBOOKS.items():
        nb_path = COURSEWORK_DIR / filename
        assert nb_path.is_file(), f"Expected notebook missing at {nb_path}"

        size_bytes = nb_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        assert size_mb < 2.5, f"Notebook {filename} ({size_mb:.2f} MB) exceeds 2.5 MB limit"
        assert size_bytes > 50_000, f"Notebook {filename} suspiciously small: {size_bytes} bytes"
        total_size_bytes += size_bytes

    total_size_mb = total_size_bytes / (1024 * 1024)
    assert total_size_mb < 10.0, f"Total suite size ({total_size_mb:.2f} MB) exceeds 10 MB ceiling"


@pytest.mark.smoke
def test_coursework_suite_schema_and_institutional_headers():
    """Verify nbformat v4 schema, Cell 0 UTH header, and Cell 1 roadmap navigation."""
    for part_num, filename in EXPECTED_NOTEBOOKS.items():
        nb_path = COURSEWORK_DIR / filename
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert nb.get("nbformat") == 4, f"{filename}: expected nbformat 4, got {nb.get('nbformat')}"
        cells = nb.get("cells", [])
        assert len(cells) >= 8, f"{filename}: expected >= 8 cells, found {len(cells)}"

        # Cell 0: Institutional Header
        cell0 = cells[0]
        assert cell0["cell_type"] == "markdown", f"{filename}: Cell 0 must be markdown"
        c0_text = "".join(cell0["source"])
        assert "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM" in c0_text, f"{filename}: Missing UTH header"
        assert "ĐỀ TÀI 11" in c0_text, f"{filename}: Missing Topic 11 indicator"
        assert f"PHẦN {part_num}" in c0_text or f"Phần {part_num}" in c0_text, (
            f"{filename}: Missing Part {part_num} in header"
        )

        # Cell 1: Roadmap Navigation Table
        cell1 = cells[1]
        assert cell1["cell_type"] == "markdown", f"{filename}: Cell 1 must be markdown"
        c1_text = "".join(cell1["source"])
        assert "Lộ Trình" in c1_text or "Roadmap" in c1_text, f"{filename}: Missing roadmap title"
        assert "🎯" in c1_text or "(Đang xem)" in c1_text, f"{filename}: Missing active badge"


@pytest.mark.smoke
def test_coursework_suite_relative_links_resolve_on_disk():
    """Verify that all markdown relative links across all notebook cells resolve to existing files on disk."""
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    total_relative_links = 0
    src_links_found = 0
    docs_links_found = 0
    sister_nb_links_found = 0

    for part_num, filename in EXPECTED_NOTEBOOKS.items():
        nb_path = COURSEWORK_DIR / filename
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for cell_idx, cell in enumerate(nb["cells"]):
            cell_text = "".join(cell.get("source", []))
            matches = link_pattern.findall(cell_text)

            for label, url in matches:
                if (
                    url.startswith("http://")
                    or url.startswith("https://")
                    or url.startswith("#")
                    or url.startswith("mailto:")
                ):
                    continue

                clean_rel_path = url.split("#")[0]
                if not clean_rel_path:
                    continue

                target_path = (COURSEWORK_DIR / clean_rel_path).resolve()
                assert target_path.exists(), (
                    f"{filename} cell {cell_idx}: Relative link [{label}]({url}) "
                    f"resolves to non-existent path {target_path}"
                )

                total_relative_links += 1
                if "../../src" in clean_rel_path:
                    src_links_found += 1
                if "../../docs" in clean_rel_path:
                    docs_links_found += 1
                if clean_rel_path.endswith(".ipynb"):
                    sister_nb_links_found += 1

    assert total_relative_links >= 15, f"Expected relative links >= 15, found {total_relative_links}"
    assert src_links_found >= 4, f"Expected src/ links >= 4, found {src_links_found}"
    assert docs_links_found >= 4, f"Expected docs/ links >= 4, found {docs_links_found}"
    assert sister_nb_links_found >= 8, f"Expected sister notebook links >= 8, found {sister_nb_links_found}"


@pytest.mark.smoke
def test_coursework_suite_execution_integrity_and_privacy():
    """Verify zero execution errors, zero SyntaxWarnings, and zero username / path leaks."""
    for part_num, filename in EXPECTED_NOTEBOOKS.items():
        nb_path = COURSEWORK_DIR / filename
        raw_content = nb_path.read_text(encoding="utf-8")

        # Privacy and portability invariants
        assert "/home/bush" not in raw_content, f"{filename}: Leaked /home/bush in notebook JSON"
        assert "file:///" not in raw_content, f"{filename}: Leaked file:/// scheme in notebook JSON"

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 3, f"{filename}: Expected at least 3 code cells"

        has_display = False
        has_stream = False

        for idx, cell in enumerate(code_cells):
            outputs = cell.get("outputs", [])
            assert len(outputs) > 0, f"{filename} cell {idx}: missing execution outputs"

            for out in outputs:
                assert out.get("output_type") != "error", (
                    f"{filename} cell {idx} failed: {out.get('ename')}: {out.get('evalue')}"
                )
                if out.get("output_type") == "display_data":
                    data = out.get("data", {})
                    if "image/png" in data:
                        has_display = True
                if out.get("output_type") == "stream":
                    has_stream = True
                    text = "".join(out.get("text", []))
                    assert "SyntaxWarning" not in text, f"{filename} cell {idx} leaked SyntaxWarning"

        assert has_display, f"{filename}: Missing matplotlib visual outputs"
        assert has_stream, f"{filename}: Missing text stream outputs"
