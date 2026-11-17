# Coursework Notebooks Redesign into 4-Part Unified Narrative Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tái cấu trúc và nâng cấp toàn bộ hệ thống 9 notebook rời rạc (~50MB) thành bộ 4 Notebook chuyên sâu tích hợp (<2.5MB/file), có mã tương tác thực tế nạp tensor/mô hình thật, không còn cell rỗng/hàm vẽ hộp đen, và kể trọn vẹn câu chuyện khoa học từ A-Z của Đề tài 11.

**Architecture:** Sử dụng kiến trúc kịch bản xây dựng tự động `scripts/build_unified_coursework_suite.py` chạy qua IPython `InteractiveShell` để sinh và tiền thực thi 4 notebook (`00_master_deepfake_defense_showcase.ipynb`, `01_diffusion_dataset_and_curation.ipynb`, `02_hard_samples_and_active_learning.ipynb`, `03_champion_m3_and_universal_benchmark.ipynb`). Mỗi notebook tuân thủ nghiêm ngặt Cell 0 (Header & Mục tiêu), Cell 1 (Bảng Roadmap 4 cột và liên kết tài liệu), sau đó là các cell code tương tác thực tế với `DeepfakeImageDataModule`, `DeepfakeImageDetector` và checkpoint thực tế.

**Tech Stack:** Python 3.14.8, PyTorch 2.x, Torchvision, HuggingFace Transformers (DINOv2 ViT-S/14), Matplotlib, OpenCV, scikit-learn (`roc_auc_score`, `confusion_matrix`), IPython InteractiveShell, nbformat v4.

**Spec:** [docs/plans/2026_11_16_COURSEWORK_NOTEBOOKS_REDESIGN_DESIGN.md](2026_11_16_COURSEWORK_NOTEBOOKS_REDESIGN_DESIGN.md)

## Global Constraints

- Python 3.14.8 tại môi trường ảo `.venv`.
- 5 CV Domain Invariants: Tensor shape `(B, C, H, W)` cho PyTorch và `(H, W, C)` cho OpenCV; BGR <-> RGB tường minh; Seeding tất định bằng `seed_everything(42)`; GPU hygiene (`device = torch.device(...)`).
- Quy chuẩn Notebook Blueprint: Cell 0 (Markdown Header & Objective), Cell 1 (Bảng Roadmap 4 cột chuẩn có link bấm trực tiếp tới `src/` và quick links tới reports), không đưa sơ đồ workflow pipeline vào notebook.
- Giới hạn dung lượng: Mỗi notebook sau khi render output phải < 2.5MB (tổng bộ 4 notebook < 10MB).
- Quy chuẩn Zero Mock Data & Zero Opaque Wrappers: Code cells phải chuẩn bị dữ liệu, trích xuất tensor, nạp checkpoint thực và tính toán kết quả trực tiếp, không gọi hàm vẽ hộp đen một dòng.
- Bảo tồn lịch sử: Toàn bộ 9 notebook cũ được lưu trữ vào `notebooks/coursework/legacy_v1/`.

## Review Focus

1. **Vượt ngưỡng dung lượng đồ thị (File Size Runaway)**: Biểu đồ vẽ ở DPI quá cao (>150) hoặc không gọi `plt.close("all")` khiến base64 nhúng làm notebook phình to > 2.5MB. (Test: `tests/test_notebook_suite_integrity.py::test_notebook_file_sizes`).
2. **Lỗi tính ROC-AUC trên nhãn đơn (Single-Class Batch Crash)**: Khi tính ROC-AUC live trên mini-batch ngẫu nhiên, nếu batch chỉ có nhãn 0 hoặc 1 sẽ gây crash `ValueError`. Giải quyết bằng cách tính ROC-AUC live trên tập 213 mẫu Challenge Vault đa lớp cân bằng. (Test: `tests/test_unified_suite_builder.py::test_challenge_vault_roc_auc_execution`).
3. **Lỗi tương thích Attention của DINOv2 (SDPA vs Eager Attention)**: HuggingFace Transformers mặc định SDPA không hỗ trợ `output_attentions=True`. Cần khởi tạo hoặc cấu hình `attn_implementation="eager"` để trích xuất spatial attention map. (Test: `tests/test_unified_suite_builder.py::test_dinov2_eager_attention_extraction`).
4. **Liên kết hỏng trong Bảng Roadmap Cell 1 (Broken Import Paths)**: Các đường dẫn tương đối trong cột 4 của Bảng Roadmap (`[src/...](../../src/...)`) phải trỏ chính xác đến các file tồn tại trên đĩa. (Test: `tests/test_notebook_suite_integrity.py::test_roadmap_links_validity`).
5. **Độc lập nền tảng tính toán (GPU / CPU Graceful Fallback)**: Checkpoints PyTorch phải nạp được trên cả máy có GPU và máy chỉ có CPU (`map_location=device`). (Test: `tests/test_unified_suite_builder.py::test_checkpoint_cpu_fallback`).

---

### Task 1: Archive Legacy Notebooks & Prepare Architecture Blueprint

**Files:**
- Create: `notebooks/coursework/legacy_v1/README.md`
- Move: `notebooks/coursework/00_master_deepfake_defense_coursework.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/01_eda_faces_demographics.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/02_sd_generation_and_curation.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/03_hard_samples_and_benchmark_comparison.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/04_active_learning_production_visualizer.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/05_synthetic_face_6k_realimage_benchmark.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/07_universal_deepfake_benchmark.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/08_champion_m3_dual_engine_quad_vault_benchmark.ipynb` -> `notebooks/coursework/legacy_v1/`
- Move: `notebooks/coursework/generator_realistic_vision_kaggle.ipynb` -> `notebooks/coursework/legacy_v1/`
- Test: `tests/test_notebook_migration.py`

**Interfaces:**
- Consumes: Existing files in `notebooks/coursework/`
- Produces: Safe legacy storage `notebooks/coursework/legacy_v1/` with 10 archived notebook files and clear documentation.

- [ ] **Step 1: Write the failing test for legacy migration**

```python
# tests/test_notebook_migration.py
from pathlib import Path

def test_legacy_v1_directory_structure():
    legacy_dir = Path("notebooks/coursework/legacy_v1")
    assert legacy_dir.is_dir(), "legacy_v1 directory must exist"
    readme_path = legacy_dir / "README.md"
    assert readme_path.is_file(), "legacy_v1/README.md must exist"
    
    expected_files = [
        "00_master_deepfake_defense_coursework.ipynb",
        "01_eda_faces_demographics.ipynb",
        "02_sd_generation_and_curation.ipynb",
        "03_hard_samples_and_benchmark_comparison.ipynb",
        "04_active_learning_production_visualizer.ipynb",
        "05_synthetic_face_6k_realimage_benchmark.ipynb",
        "06_specialization_vs_generalization_synthesis.ipynb",
        "07_universal_deepfake_benchmark.ipynb",
        "08_champion_m3_dual_engine_quad_vault_benchmark.ipynb",
        "generator_realistic_vision_kaggle.ipynb",
    ]
    for fname in expected_files:
        assert (legacy_dir / fname).is_file(), f"Missing archived file: {fname}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_notebook_migration.py -v`
Expected: FAIL with "legacy_v1 directory must exist"

- [ ] **Step 3: Move notebooks and create legacy README.md**

Move all 10 legacy `.ipynb` files to `notebooks/coursework/legacy_v1/` and write `notebooks/coursework/legacy_v1/README.md` explaining the legacy version history and the transition to the unified 4-notebook suite.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_notebook_migration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/coursework/ tests/test_notebook_migration.py
git commit -m "chore(notebooks): archive legacy coursework notebooks to legacy_v1"
```

---

### Task 2: Build Interactive Automation Engine for Notebook Generation

**Files:**
- Create: `scripts/build_unified_coursework_suite.py`
- Test: `tests/test_unified_suite_builder.py`

**Interfaces:**
- Consumes: `IPython.core.interactiveshell.InteractiveShell`, `nbformat`, `matplotlib`
- Produces: Functions `make_markdown_cell(content: str) -> dict`, `make_code_cell(source: str, outputs: list) -> dict`, `execute_and_capture(shell: InteractiveShell, code: str, max_dpi: int = 120) -> list[dict]`, `write_notebook(path: Path, cells: list[dict]) -> None`.

- [ ] **Step 1: Write the failing test for suite builder engine**

```python
# tests/test_unified_suite_builder.py
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
import pytest

from scripts.build_unified_coursework_suite import (
    make_markdown_cell,
    make_code_cell,
    execute_and_capture,
    write_notebook,
)

def test_cell_creators():
    md = make_markdown_cell("# Title")
    assert md["cell_type"] == "markdown"
    assert md["source"] == ["# Title"]
    
    code = make_code_cell("a = 1")
    assert code["cell_type"] == "code"
    assert code["source"] == ["a = 1"]

def test_execute_and_capture_live_shell():
    shell = InteractiveShell.instance()
    code = "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1, 2], [3, 4])\nplt.show()"
    outputs = execute_and_capture(shell, code, max_dpi=100)
    assert len(outputs) > 0
    assert any(o.get("output_type") == "display_data" for o in outputs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_unified_suite_builder.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'scripts.build_unified_coursework_suite'"

- [ ] **Step 3: Implement core engine in `scripts/build_unified_coursework_suite.py`**

Implement helper utilities with robust stdout/display/figure interception, standard DPI scaling (120-130 DPI) to guarantee small JSON sizes, and clean nbformat v4 serialization.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_unified_suite_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/build_unified_coursework_suite.py tests/test_unified_suite_builder.py
git commit -m "feat(scripts): implement unified coursework notebook builder engine"
```

---

### Task 3: Build & Live-Execute Notebook 01 — Real Manifold & Dual-Engine Portfolio

**Files:**
- Generate: `notebooks/coursework/01_diffusion_dataset_and_curation.ipynb`
- Modify: `scripts/build_unified_coursework_suite.py` (add `build_notebook_01()`)
- Test: `tests/test_notebook_outputs.py`

**Interfaces:**
- Consumes: `src.data.image_datamodule.DeepfakeImageDataModule`, `data/raw/manifest.jsonl`, `src.pipelines.curate_image_dataset.ImageCurationGate`, `data/synthetic/d0_train_curation_report_att0.json`
- Produces: Notebook 01 with live code executing:
  1. DataModule setup and demographic balance checks (age, gender, glasses).
  2. Live calculation of real skin color gamut from actual face samples.
  3. Soft-mask Gaussian feathering simulation ($\sigma=4.0$).
  4. CurationGate execution and audit of 545 rejected anomalies.

- [ ] **Step 1: Write test for Notebook 01**

```python
# tests/test_notebook_outputs.py (part 1)
from pathlib import Path
import json

def test_notebook_01_structure_and_size():
    nb_path = Path("notebooks/coursework/01_diffusion_dataset_and_curation.ipynb")
    assert nb_path.is_file(), "Notebook 01 must exist"
    size_mb = nb_path.stat().st_size / (1024 * 1024)
    assert size_mb < 2.5, f"Notebook 01 size {size_mb:.2f}MB exceeds 2.5MB ceiling"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]
    assert cells[0]["cell_type"] == "markdown"
    assert "Real Manifold" in "".join(cells[0]["source"])
    assert cells[1]["cell_type"] == "markdown"
    assert "Roadmap" in "".join(cells[1]["source"]) or "Lộ Trình" in "".join(cells[1]["source"])
    
    # Verify no empty code outputs or bare plt.show()
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    for idx, c in enumerate(code_cells):
        src = "".join(c["source"]).strip()
        assert src != "plt.show()", f"Code cell {idx} is bare plt.show()"
        assert len(c["outputs"]) > 0, f"Code cell {idx} has empty outputs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_01 -v`
Expected: FAIL with "Notebook 01 must exist"

- [ ] **Step 3: Implement `build_notebook_01()` in `scripts/build_unified_coursework_suite.py` and run it**

Implement the full section structure for Notebook 01:
- Cell 0: Header UTH & Mục tiêu.
- Cell 1: Roadmap Table trỏ trực tiếp `src/data/raw_ingestion.py`, `src/models/diffusion/`, `src/pipelines/curate_image_dataset.py`.
- Interactive cells: Nạp `data/raw/manifest.jsonl`, tính toán độ bão hòa HSV và RGB trên ảnh thật, chạy thử nghiệm `ImageCurationGate` đo Laplacian, phân tích dữ liệu 545 ảnh bị loại từ `d0_train_curation_report_att0.json`.
- Execute: `.venv/bin/python3 -c "from scripts.build_unified_coursework_suite import build_notebook_01; build_notebook_01()"`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_01 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/coursework/01_diffusion_dataset_and_curation.ipynb scripts/build_unified_coursework_suite.py tests/test_notebook_outputs.py
git commit -m "feat(notebooks): build and live-execute Notebook 01 on real manifold and curation"
```

---

### Task 4: Build & Live-Execute Notebook 02 — Hard-Sample Mining & Active Learning

**Files:**
- Generate: `notebooks/coursework/02_hard_samples_and_active_learning.ipynb`
- Modify: `scripts/build_unified_coursework_suite.py` (add `build_notebook_02()`)
- Test: `tests/test_notebook_outputs.py`

**Interfaces:**
- Consumes: `src.models.detector.image_detector.DeepfakeImageDetector`, `data/active_learning/cumulative_challenge_test.jsonl`, `experiments/checkpoints/active_learning_m2_best.pt`, `experiments/checkpoints/baseline_d0_best.pt`
- Produces: Notebook 02 with live code executing:
  1. DINOv2 model architecture inspection and layer freeze verification.
  2. Real-manifold distance and boundary ambiguity filtering ($|P-0.5| \le 0.15$).
  3. Visual diagnosis of micro-biological defects (eyes, teeth, ears, skin texture).
  4. Live batch inference of $M_0$ vs $M_2^*$ on 213 Challenge samples, showing Challenge Fake Recall jump (11.71% -> 91.89%) and bypass rate reduction (88.29% -> 8.11%).

- [ ] **Step 1: Write test for Notebook 02**

```python
# tests/test_notebook_outputs.py (part 2)
def test_notebook_02_structure_and_size():
    nb_path = Path("notebooks/coursework/02_hard_samples_and_active_learning.ipynb")
    assert nb_path.is_file(), "Notebook 02 must exist"
    size_mb = nb_path.stat().st_size / (1024 * 1024)
    assert size_mb < 2.5, f"Notebook 02 size {size_mb:.2f}MB exceeds 2.5MB ceiling"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]
    assert "Active Learning" in "".join(cells[0]["source"])
    
    code_text = " ".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "cumulative_challenge_test.jsonl" in code_text
    assert "active_learning_m2_best.pt" in code_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_02 -v`
Expected: FAIL with "Notebook 02 must exist"

- [ ] **Step 3: Implement `build_notebook_02()` in `scripts/build_unified_coursework_suite.py` and run it**

Implement the full section structure for Notebook 02:
- Cell 0 & 1: Header and Roadmap linking `src/models/detector/image_detector.py`, `src/models/evaluators/hardness.py`.
- Interactive cells: Nạp DINOv2 detector, hiển thị số lượng tham số trainable (8.06%) vs frozen (91.94%), nạp 213 mẫu Challenge Vault, chạy suy luận live bằng checkpoint $M_0$ và $M_2^*$, vẽ confusion matrix và đường cong ROC trực tiếp.
- Execute: `.venv/bin/python3 -c "from scripts.build_unified_coursework_suite import build_notebook_02; build_notebook_02()"`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_02 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/coursework/02_hard_samples_and_active_learning.ipynb scripts/build_unified_coursework_suite.py tests/test_notebook_outputs.py
git commit -m "feat(notebooks): build and live-execute Notebook 02 on hard-sample mining and DINOv2 active learning"
```

---

### Task 5: Build & Live-Execute Notebook 03 — Champion $M_3$ & Universal Benchmark

**Files:**
- Generate: `notebooks/coursework/03_champion_m3_and_universal_benchmark.ipynb`
- Modify: `scripts/build_unified_coursework_suite.py` (add `build_notebook_03()`)
- Test: `tests/test_notebook_outputs.py`

**Interfaces:**
- Consumes: `experiments/checkpoints/active_learning_m3_best.pt`, `data/benchmarks/D3_clean/unseen_rv51_test.jsonl`, `experiments/benchmarks/tables/benchmark_5models_*.csv`
- Produces: Notebook 03 with live code executing:
  1. Data integrity audit demonstrating zero identity leakage (626 test RV5.1 images isolated).
  2. Champion $M_3$ checkpoint loading and live evaluation on Unseen RV5.1 Vault (Fake Recall = 93.13%, Bypass = 6.87%).
  3. Quad-Vault 5-model comparative performance table.
  4. Universal Benchmark 20.400 images breakdown across 34 generation algorithms and 8 families.
  5. Decision threshold sweep ($\tau \in [0.005, 0.800]$) and video compression robustness check (FF++ C23 FPR = 1.45%).

- [ ] **Step 1: Write test for Notebook 03**

```python
# tests/test_notebook_outputs.py (part 3)
def test_notebook_03_structure_and_size():
    nb_path = Path("notebooks/coursework/03_champion_m3_and_universal_benchmark.ipynb")
    assert nb_path.is_file(), "Notebook 03 must exist"
    size_mb = nb_path.stat().st_size / (1024 * 1024)
    assert size_mb < 2.5, f"Notebook 03 size {size_mb:.2f}MB exceeds 2.5MB ceiling"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]
    assert "Champion" in "".join(cells[0]["source"]) or "M3" in "".join(cells[0]["source"])
    
    code_text = " ".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "active_learning_m3_best.pt" in code_text
    assert "unseen_rv51_test.jsonl" in code_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_03 -v`
Expected: FAIL with "Notebook 03 must exist"

- [ ] **Step 3: Implement `build_notebook_03()` in `scripts/build_unified_coursework_suite.py` and run it**

Implement the full section structure for Notebook 03:
- Cell 0 & 1: Header and Roadmap linking `scripts/train_m3_dual_engine.py`, `scripts/evaluate_quad_vault_5models.py`, `scripts/evaluate_universal_benchmark_5models.py`.
- Interactive cells: Nạp tập `unseen_rv51_test.jsonl`, nạp checkpoint $M_3$, chạy suy luận live trên tập kiểm thử chưa từng thấy, vẽ heatmap Quad-Vault và radar chart 8 họ công nghệ, quét ngưỡng $\tau$.
- Execute: `.venv/bin/python3 -c "from scripts.build_unified_coursework_suite import build_notebook_03; build_notebook_03()"`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_03 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/coursework/03_champion_m3_and_universal_benchmark.ipynb scripts/build_unified_coursework_suite.py tests/test_notebook_outputs.py
git commit -m "feat(notebooks): build and live-execute Notebook 03 on Champion M3 and Universal Benchmark"
```

---

### Task 6: Build & Live-Execute Notebook 00 — Master Showcase & Live Forensic Lecture

**Files:**
- Generate: `notebooks/coursework/00_master_deepfake_defense_showcase.ipynb`
- Modify: `scripts/build_unified_coursework_suite.py` (add `build_notebook_00()`)
- Test: `tests/test_notebook_outputs.py`

**Interfaces:**
- Consumes: All project modules and checkpoints (`active_learning_m3_best.pt`)
- Produces: Master Notebook 00 containing:
  1. Complete executive presentation and ASCII architecture map.
  2. 6 core thematic chapters synthesizing findings from Notebooks 01, 02, and 03.
  3. Master Golden Scorecard comparing $M_0 \to M_1 \to M_2 \to M_2^* \to M_3$.
  4. Live Forensic Inspector: Nạp ảnh mẫu, chạy qua $M_3$, trích xuất spatial attention weights từ DINOv2 self-attention layer, xuất heatmap vùng mắt/miệng và điểm $P(\text{Fake})$.

- [ ] **Step 1: Write test for Notebook 00**

```python
# tests/test_notebook_outputs.py (part 4)
def test_notebook_00_structure_and_size():
    nb_path = Path("notebooks/coursework/00_master_deepfake_defense_showcase.ipynb")
    assert nb_path.is_file(), "Notebook 00 must exist"
    size_mb = nb_path.stat().st_size / (1024 * 1024)
    assert size_mb < 2.5, f"Notebook 00 size {size_mb:.2f}MB exceeds 2.5MB ceiling"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]
    assert "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM" in "".join(cells[0]["source"])
    
    code_text = " ".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "Live Forensic Inspector" in code_text or "sample_idx" in code_text or "Inspector" in code_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_00 -v`
Expected: FAIL with "Notebook 00 must exist"

- [ ] **Step 3: Implement `build_notebook_00()` in `scripts/build_unified_coursework_suite.py` and run it**

Implement Master Showcase with 6 overarching chapters:
- Cell 0 & 1: Grand header and complete Roadmap Table referencing all 3 phase notebooks.
- Interactive live inspector cell: Nạp ảnh chân dung từ Challenge Vault, nạp model $M_3$, trích xuất attention weights hoặc patch activations, vẽ ảnh kèm heatmap phủ màu đỏ-xanh và bảng xác suất của cả 4 thế hệ mô hình.
- Execute: `.venv/bin/python3 -c "from scripts.build_unified_coursework_suite import build_notebook_00; build_notebook_00()"`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_00 -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add notebooks/coursework/00_master_deepfake_defense_showcase.ipynb scripts/build_unified_coursework_suite.py tests/test_notebook_outputs.py
git commit -m "feat(notebooks): build and live-execute Notebook 00 Master Showcase and Live Forensic Inspector"
```

---

### Task 7: Update Governance Docs & End-to-End Suite Smoke Verification

**Files:**
- Modify: `notebooks/README.md`
- Create: `tests/test_notebook_suite_integrity.py`
- Test: Full test suite verification via `make test-smoke` and `pytest tests/test_notebook_suite_integrity.py`

**Interfaces:**
- Consumes: All 4 newly built notebooks in `notebooks/coursework/`
- Produces: Updated governance documentation and passing test gate.

- [ ] **Step 1: Write comprehensive suite integrity test**

```python
# tests/test_notebook_suite_integrity.py
from pathlib import Path
import json

def test_full_coursework_suite_integrity():
    suite_files = [
        "00_master_deepfake_defense_showcase.ipynb",
        "01_diffusion_dataset_and_curation.ipynb",
        "02_hard_samples_and_active_learning.ipynb",
        "03_champion_m3_and_universal_benchmark.ipynb",
    ]
    base_dir = Path("notebooks/coursework")
    total_size = 0
    for name in suite_files:
        p = base_dir / name
        assert p.is_file(), f"Missing notebook: {name}"
        size_mb = p.stat().st_size / (1024 * 1024)
        assert size_mb < 2.5, f"Notebook {name} exceeds 2.5MB: {size_mb:.2f}MB"
        total_size += size_mb
        
        with open(p, "r", encoding="utf-8") as f:
            nb = json.load(f)
        assert nb.get("nbformat") == 4
        
        # Verify Cell 0 & Cell 1 invariants
        assert nb["cells"][0]["cell_type"] == "markdown"
        assert nb["cells"][1]["cell_type"] == "markdown"
        assert "Step" in "".join(nb["cells"][1]["source"])
        assert "Import path" in "".join(nb["cells"][1]["source"])
        
    assert total_size < 10.0, f"Total suite size {total_size:.2f}MB exceeds 10MB"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_notebook_suite_integrity.py -v`
Expected: PASS

- [ ] **Step 3: Update `notebooks/README.md`**

Update `notebooks/README.md` to document the 4-part Unified Suite, its pedagogical structure, and link to the archived `legacy_v1` directory.

- [ ] **Step 4: Run smoke tests and linter**

Run: `make test-smoke`
Expected: All 12+ smoke tests pass in < 6 seconds.

- [ ] **Step 5: Commit**

```bash
git add notebooks/README.md tests/test_notebook_suite_integrity.py
git commit -m "docs(notebooks): update notebooks governance README and add suite integrity test"
```
