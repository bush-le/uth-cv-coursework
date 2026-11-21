"""Unit tests for the unified coursework notebook builder engine.

Tests InteractiveShell cell execution, stdout/display_data capture, matplotlib cleanup,
and schema compliance for the 4-part unified coursework notebook suite.
"""

from pathlib import Path
import json
import re
import pytest
from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt

from scripts.build_unified_coursework_suite import (
    make_markdown_cell,
    make_code_cell,
    make_header_cell,
    make_roadmap_cell,
    create_notebook,
    execute_and_capture,
    save_notebook,
    write_notebook,
)


@pytest.mark.smoke
def test_cell_creators():
    """Verify markdown and code cell construction dictionaries."""
    md = make_markdown_cell("# Title")
    assert md["cell_type"] == "markdown"
    assert md["source"] == ["# Title"]

    code = make_code_cell("a = 1")
    assert code["cell_type"] == "code"
    assert code["source"] == ["a = 1"]
    assert code["outputs"] == []


@pytest.mark.smoke
def test_execute_and_capture_live_shell():
    """Verify live IPython execution captures stdout streams and matplotlib display_data."""
    shell = InteractiveShell.instance()
    code = (
        "import matplotlib.pyplot as plt\n"
        "print('Execution start')\n"
        "fig, ax = plt.subplots(figsize=(6, 3))\n"
        "ax.plot([1, 2], [3, 4])\n"
        "plt.title('Test Plot')\n"
        "plt.show()\n"
        "print('Execution end')\n"
    )
    outputs = execute_and_capture(shell, code, max_dpi=100)
    assert len(outputs) >= 2, f"Expected at least 2 outputs, got {len(outputs)}"

    # Check for stdout streams
    stream_outputs = [o for o in outputs if o.get("output_type") == "stream"]
    assert len(stream_outputs) >= 1
    stdout_text = "".join("".join(s.get("text", [])) for s in stream_outputs)
    assert "Execution start" in stdout_text
    assert "Execution end" in stdout_text

    # Check for matplotlib PNG base64 display_data
    display_outputs = [o for o in outputs if o.get("output_type") == "display_data"]
    assert len(display_outputs) == 1
    img_b64 = display_outputs[0]["data"]["image/png"]
    assert isinstance(img_b64, str)
    assert len(img_b64) > 100


@pytest.mark.smoke
def test_plt_close_cleanup():
    """Ensure execute_and_capture closes all figures and leaves zero dangling figures."""
    shell = InteractiveShell.instance()
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig1, ax1 = plt.subplots()\n"
        "ax1.plot([0, 1])\n"
        "fig2, ax2 = plt.subplots()\n"
        "ax2.plot([2, 3])\n"
        "# Note: no plt.show() called intentionally\n"
    )
    outputs = execute_and_capture(shell, code, max_dpi=100)
    assert len(outputs) == 2, f"Expected 2 captured figures, got {len(outputs)}"
    assert all(o.get("output_type") == "display_data" for o in outputs)
    assert len(plt.get_fignums()) == 0, f"Dangling figures remain: {plt.get_fignums()}"


@pytest.mark.smoke
def test_create_notebook_and_schema_compliance(tmp_path: Path):
    """Verify notebook dictionary schema and serialization to disk."""
    header = make_header_cell(
        title="Đa Tạp Ảnh Thật & Động Cơ Khuếch Tán Kép",
        subtitle="Khảo Sát 4.000 Chân Dung FFHQ/CelebA-HQ",
        part_num=1,
        time_str="2026-11-16",
        objective="Kiểm định tính phân ly danh tính và phân tích phổ màu da.",
    )
    roadmap = make_roadmap_cell(current_part=1)
    code = make_code_cell("print('Test notebook')")

    nb = create_notebook(
        title="Part 1: Real Manifold",
        subtitle="Diffusion Dataset",
        part_num=1,
        cells=[header, roadmap, code],
    )

    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] >= 2
    assert "metadata" in nb
    assert len(nb["cells"]) == 3
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert nb["cells"][1]["cell_type"] == "markdown"
    assert nb["cells"][2]["cell_type"] == "code"

    out_file = tmp_path / "test_nb.ipynb"
    save_notebook(nb, out_file)
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["nbformat"] == 4
    assert len(loaded["cells"]) == 3

    # Test polymorphic write_notebook alias
    out_file_2 = tmp_path / "test_nb_2.ipynb"
    write_notebook(out_file_2, nb)
    assert out_file_2.is_file()


@pytest.mark.smoke
def test_roadmap_cell_relative_paths():
    """Verify roadmap cell contains a 4-column Markdown table and valid relative links."""
    for current_part in [0, 1, 2, 3]:
        cell = make_roadmap_cell(current_part=current_part)
        assert cell["cell_type"] == "markdown"
        content = "".join(cell["source"])

        # Check for 4-column table header
        assert "| Phần" in content or "| Part" in content
        assert "Notebook" in content

        # Check highlighting of the active part
        part_indicator = f"Phần {current_part}"
        assert part_indicator in content

        # Extract all markdown links [text](target)
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        assert len(links) >= 4, f"Roadmap table must contain at least 4 links, found {len(links)}"

        # Verify relative path targets
        for text, target in links:
            assert not target.startswith("/"), f"Absolute path forbidden: {target}"
            if target.startswith("../../"):
                # Path relative to notebooks/coursework/
                resolved = (Path("notebooks/coursework") / target).resolve()
                assert resolved.exists(), f"Broken roadmap link target: {target} (resolved to {resolved})"
            elif target.endswith(".ipynb"):
                # Sibling notebook in notebooks/coursework/
                assert target in [
                    "00_master_deepfake_defense_showcase.ipynb",
                    "01_diffusion_dataset_and_curation.ipynb",
                    "02_hard_samples_and_active_learning.ipynb",
                    "03_champion_m3_and_universal_benchmark.ipynb",
                ], f"Unexpected notebook link: {target}"


@pytest.mark.smoke
def test_header_cell_metadata():
    """Verify header cell includes institutional info, topic, and objectives."""
    header = make_header_cell(
        title="Master Deepfake Defense Showcase",
        subtitle="Live Forensic Lecture",
        part_num=0,
        time_str="2026-11-16",
        objective="Tổng hợp toàn diện kết quả nghiên cứu Đề tài 11.",
    )
    text = "".join(header["source"])
    assert "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM" in text
    assert "ĐỀ TÀI 11" in text
    assert "PHẦN 0" in text or "Part 0" in text
    assert "Master Deepfake Defense Showcase" in text
    assert "Mục Tiêu" in text or "Objective" in text


@pytest.mark.smoke
def test_challenge_vault_roc_auc_execution():
    """Verify Challenge Vault cumulative test set has balanced classes and avoids single-class AUC crash."""
    from src.common.metrics import compute_roc_auc

    challenge_path = Path("data/active_learning/cumulative_challenge_test.jsonl")
    assert challenge_path.is_file(), f"Missing challenge set: {challenge_path}"

    labels = []
    with open(challenge_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            labels.append(item["label"])

    assert len(labels) == 213, f"Expected 213 samples, got {len(labels)}"
    assert 0 in labels and 1 in labels, "Challenge set must contain both Real (0) and Fake (1) labels"

    # Verify compute_roc_auc evaluates cleanly without ValueError or NaN on two-class data
    dummy_probs = [0.1 if y == 0 else 0.9 for y in labels]
    auc = compute_roc_auc(labels, dummy_probs)
    assert isinstance(auc, float)
    assert not (auc != auc), "ROC-AUC must not be NaN"
    assert auc == 1.0


@pytest.mark.smoke
def test_checkpoint_cpu_fallback():
    """Verify model checkpoints can be loaded safely on CPU with map_location."""
    import torch
    from src.models.detector.image_detector import DeepfakeImageDetector

    ckpt_path = Path("experiments/checkpoints/active_learning_m2_best.pt")
    if not ckpt_path.is_file():
        pytest.skip(f"Checkpoint file not present in workspace (gitignored): {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt

    model = DeepfakeImageDetector(backbone="dino_vit", pretrained=False)
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
    assert out.shape == (2, 1)
