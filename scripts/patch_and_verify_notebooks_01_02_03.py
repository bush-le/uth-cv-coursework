#!/usr/bin/env python3
"""Patch bare plt.show() cells and re-execute notebooks 01, 02, and 03 with verified outputs."""

import base64
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import matplotlib
matplotlib.use("Agg")
from IPython.core.interactiveshell import InteractiveShell

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


def run_cell_and_capture(shell: InteractiveShell, code: str) -> list[dict]:
    captured_displays = []

    def custom_display_pub(data, metadata=None):
        captured_displays.append({
            "data": data,
            "metadata": metadata or {},
            "output_type": "display_data",
        })

    shell.display_pub.publish = custom_display_pub

    stdout_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
        res = shell.run_cell(code)

    if not res.success:
        raise RuntimeError(f"Cell execution failed:\nCode:\n{code}\nError: {res.error_in_exec}")

    outputs = []
    stdout_text = stdout_buf.getvalue()
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": stdout_text.splitlines(keepends=True),
        })

    for disp in captured_displays:
        outputs.append(disp)

    plt_mod = shell.user_ns.get("plt")
    if plt_mod:
        fignums = plt_mod.get_fignums()
        for fignum in fignums:
            fig = plt_mod.figure(fignum)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            buf.seek(0)
            b64_str = base64.b64encode(buf.read()).decode("utf-8")
            outputs.append({
                "data": {
                    "image/png": b64_str,
                    "text/plain": [f"<Figure size {fig.get_size_inches()[0]*fig.dpi}x{fig.get_size_inches()[1]*fig.dpi} with {len(fig.axes)} Axes>"]
                },
                "metadata": {},
                "output_type": "display_data",
            })
        plt_mod.close("all")

    return outputs


def patch_notebook_01():
    print("=== Patching Notebook 01 ===")
    p = Path("notebooks/coursework/01_eda_faces_demographics.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    shell = InteractiveShell.instance()
    shell.run_cell(ROOT_BOOTSTRAP_CODE)
    shell.run_cell("import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot as plt")

    patches = {
        5: (
            "# Trực quan hóa lưới 4x4 chân dung ảnh thật chuẩn hóa với viền nhận diện thương hiệu\n"
            "import matplotlib.pyplot as plt\n"
            "from src.common.visualizer import plot_real_faces_gallery\n"
            "fig = plot_real_faces_gallery()\n"
            "plt.show()"
        ),
        9: (
            "# Hiển thị ma trận tương quan thuộc tính khuôn mặt\n"
            "import matplotlib.pyplot as plt\n"
            "from src.common.visualizer import plot_attribute_correlation_matrix\n"
            "fig = plot_attribute_correlation_matrix()\n"
            "plt.show()"
        ),
        11: (
            "# Hiển thị phân bố góc quay đầu 3D\n"
            "import matplotlib.pyplot as plt\n"
            "from src.common.visualizer import plot_3d_head_pose_distribution\n"
            "fig = plot_3d_head_pose_distribution()\n"
            "plt.show()"
        ),
        15: (
            "# Hiển thị biểu đồ phổ màu và độ bão hòa sắc da\n"
            "import matplotlib.pyplot as plt\n"
            "from src.common.visualizer import plot_skin_color_gamut\n"
            "fig = plot_skin_color_gamut()\n"
            "plt.show()"
        ),
    }

    for idx, new_code in patches.items():
        print(f"  Executing and updating Cell {idx:02d}...")
        outputs = run_cell_and_capture(shell, new_code)
        nb["cells"][idx]["source"] = [l + "\n" for l in new_code.split("\n")]
        nb["cells"][idx]["outputs"] = outputs
        print(f"    -> Updated with {len(outputs)} output items.")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("  Notebook 01 patched and saved successfully!")


def patch_notebook_02():
    print("\n=== Patching Notebook 02 ===")
    p = Path("notebooks/coursework/02_sd_generation_and_curation.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    shell = InteractiveShell.instance()
    shell.run_cell(ROOT_BOOTSTRAP_CODE)
    shell.run_cell("import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot as plt")

    patches = {
        5: (
            "# Mô phỏng quy trình làm mờ biên mặt nạ mềm (Soft-Mask Feathering)\n"
            "import matplotlib.pyplot as plt\n"
            "from src.common.visualizer import plot_soft_mask_feathering_simulation\n"
            "fig = plot_soft_mask_feathering_simulation()\n"
            "plt.show()"
        ),
    }

    for idx, new_code in patches.items():
        print(f"  Executing and updating Cell {idx:02d}...")
        outputs = run_cell_and_capture(shell, new_code)
        nb["cells"][idx]["source"] = [l + "\n" for l in new_code.split("\n")]
        nb["cells"][idx]["outputs"] = outputs
        print(f"    -> Updated with {len(outputs)} output items.")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("  Notebook 02 patched and saved successfully!")


def patch_notebook_03():
    print("\n=== Patching Notebook 03 ===")
    p = Path("notebooks/coursework/03_hard_samples_and_benchmark_comparison.ipynb")
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    shell = InteractiveShell.instance()
    shell.run_cell(ROOT_BOOTSTRAP_CODE)
    shell.run_cell("import matplotlib; matplotlib.use('Agg')\nimport matplotlib.pyplot as plt")

    patches = {
        11: (
            "# Thực thi đồ thị phân rã vùng giải phẫu sinh học và điều kiện chiếu sáng\n"
            "import matplotlib.pyplot as plt\n"
            "from src.common.visualizer import plot_biological_error_breakdown\n"
            "fig = plot_biological_error_breakdown()\n"
            "plt.show()"
        ),
        13: (
            "# Hiển thị thư viện ảnh đối chứng 3 hàng trực quan (Top Hard Samples)\n"
            "import matplotlib.pyplot as plt\n"
            "from src.common.visualizer import plot_hard_samples_comparison_gallery\n"
            "fig = plot_hard_samples_comparison_gallery()\n"
            "plt.show()"
        ),
    }

    for idx, new_code in patches.items():
        print(f"  Executing and updating Cell {idx:02d}...")
        outputs = run_cell_and_capture(shell, new_code)
        nb["cells"][idx]["source"] = [l + "\n" for l in new_code.split("\n")]
        nb["cells"][idx]["outputs"] = outputs
        print(f"    -> Updated with {len(outputs)} output items.")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("  Notebook 03 patched and saved successfully!")


if __name__ == "__main__":
    patch_notebook_01()
    patch_notebook_02()
    patch_notebook_03()
    print("\nAll notebooks 01, 02, and 03 successfully patched and verified!")
