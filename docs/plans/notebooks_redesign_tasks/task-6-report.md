# Task 6 Completion Report: Build & Live-Execute Notebook 00 — Master Deepfake Defense Showcase & Live Forensic Lecture

## Executive Summary
Task 6 of the Coursework Notebooks Redesign has been completed following strict Test-Driven Development (TDD). `notebooks/coursework/00_master_deepfake_defense_showcase.ipynb` was generated via `build_notebook_00()` inside `scripts/build_unified_coursework_suite.py`, live-executed through IPython InteractiveShell with real PyTorch model weights and checkpoints, and verified by `tests/test_notebook_outputs.py` and `make test-smoke`.

- **Commit**: `1fee36a` (`feat(notebooks): generate and verify notebook 00 master showcase and live forensic inspector`)
- **Generated Notebook**: `notebooks/coursework/00_master_deepfake_defense_showcase.ipynb` (1.7 MB, strictly below the 2.5 MB ceiling)
- **Test File**: `tests/test_notebook_outputs.py` (12 smoke tests across Notebooks 00, 01, 02, and 03, 100% passing)
- **Suite Smoke Gate**: `make test-smoke` (33 passed in 5.19s)

---

## TDD Implementation Flow

### 1. RED Phase
- Added 3 `@pytest.mark.smoke` test cases to `tests/test_notebook_outputs.py`:
  1. `test_notebook_00_exists_and_size`: Checks file existence and validates that file size is strictly `< 2.5 MB` and `> 10 KB`.
  2. `test_notebook_00_schema_and_navigation`: Verifies valid nbformat v4 schema, institutional Cell 0 header (Topic 11, UTH, Part 0), and Cell 1 roadmap table highlighting Part 0 (`🎯 / (Đang xem)`).
  3. `test_notebook_00_execution_outputs`: Verifies zero execution errors, zero `SyntaxWarning` occurrences, presence of both stream logs and `display_data` figures, presence of `active_learning_m3_best.pt` reference, and zero leakage of local home directory paths or `file:///` schemes.
- Ran `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_00 -v`:
  - Result: **3 failed** as expected (`Notebook 00 file does not exist`).

### 2. GREEN Phase
- Implemented `build_notebook_00(shell=None, output_path=None)` in `scripts/build_unified_coursework_suite.py`:
  - **Cell 0**: UTH Institutional Header & Topic 11 executive objectives for Part 0 (Master Defense Showcase).
  - **Cell 1**: 4-column Roadmap Table highlighting Part 0 (`🎯 / (Đang xem)`) with valid relative links to project modules (`src/models/detector/image_detector.py`, `docs/reports/M3_DUAL_ENGINE_CHAMPION_REPORT.md`).
  - **Section 1 (Diễn Ngôn Khoa Học & Bảng Vàng Thành Tích Golden Scorecard)**:
    - Detailed the coursework narrative across all 4 experimental milestones:
      - $M_0$ (Baseline D0): Vulnerability to ambiguous biological hard samples (Bypass Rate 88.29% on Challenge Vault).
      - $M_1$ (Active Learning Round 1): Adding 350 mined hard boundary cases ($|P - 0.5| \le 0.15$), surging Challenge Recall from 11.71% to 70.27%.
      - $M_2^*$ (Refined Warm-Start): Knowledge-preserving fine-tuning on 740 hard samples, achieving 91.89% Challenge Recall (Bypass drops to 8.11%).
      - $M_3$ (Dual-Engine Champion): Expanding to RealisticVision v5.1 with Zero Identity Leakage, surging Unseen RV5.1 Recall from 66.45% to 93.13% and more than doubling global benchmark recall (2.48% $\to$ 5.49%).
    - Ingested real experimental results from `experiments/benchmarks/tables/benchmark_5models_quad_vault_summary.csv` and `benchmark_5models_global_summary.csv`.
    - Generated a formatted Golden Scorecard interactive table comparing $M_0, M_1, M_2^*, M_3$ across:
      - Master Frozen Test (Recall / Bypass)
      - Challenge Vault (Recall / Bypass)
      - Unseen RV5.1 Vault (Recall / Bypass)
      - Universal Benchmark 20.4k (Recall / AUC)
    - Rendered a multi-metric visual dashboard (3 subplots at 110 DPI):
      - Top-Left: Grouped bar chart comparing Fake Recall across all 4 evaluation sets for $M_0, M_1, M_2^*, M_3$.
      - Top-Right: Curves showing the steep reduction in Bypass Rate (Challenge bypass 88.29% $\to$ 8.11%, RV5.1 bypass 33.55% $\to$ 6.87%).
      - Bottom: Cross-generator generalization across 8 technology families from `benchmark_5models_family_summary.csv` highlighting $M_3$'s breakthrough.
  - **Section 2 (Trạm Giám Định Pháp Y Trực Tiếp — Live Forensic Inspector)**:
    - Formalized the mathematical mechanism of DINOv2 Vision Transformer spatial attention extraction:
      - CLS token attention to $H_{\text{grid}} \times W_{\text{grid}}$ spatial patch tokens ($36 \times 36 = 1296$ patches for $504\times 504$, or $16 \times 16 = 256$ for $224 \times 224$).
      - Multi-head averaging ($H=6$ heads) and bilinear upsampling to $(512, 512)$ or $(224, 224)$.
    - Loaded Champion model checkpoint `experiments/checkpoints/active_learning_m3_best.pt` into `DeepfakeImageDetector`.
    - Reconfigured backbone with `_attn_implementation = 'eager'` to extract attention matrices live.
    - Audited model parameters: 21.9M total parameters, 20.1M frozen (Blocks 0-10, 91.94%), 1.78M trainable (Block 11 + Head, 8.06%).
    - Executed live inspection on 3 representative test cases:
      1. Real Portrait: `data/raw/faces/ffhq/00301.png` (Label: 0 Real) -> P(Fake) = 1.99%, Predicted: Real, Confidence: 96.0%.
      2. Challenge Vault Hard Fake: `data/active_learning/round_1/images/inpainting/fake_ffhq_00000_q1_116.png` (Label: 1 Fake) -> P(Fake) = 99.79%, Predicted: Deepfake, Confidence: 99.6%.
      3. Photorealistic Unseen Fake: `data/external_benchmarks/synthetic_face_dataset_6k/img2img/187_img2img.png` (Label: 1 Fake) -> P(Fake) = 99.99%, Predicted: Deepfake, Confidence: 100.0%.
    - Rendered a 3-part diagnostic panel for each sample (9 subplots total at 95 DPI):
      - Column 1: Original Face (RGB) with metadata.
      - Column 2: DINOv2 Spatial Attention Heatmap overlay (jet colormap) displaying peak attention focus coordinates.
      - Column 3: Forensic Diagnostic Summary Card (P(Fake), Decision Class at $\tau=0.5$, Confidence %, Anatomical Focus Region: Periorbital, Nasal, Oral/Dental, Facial Margin, or Micro-Skin Texture).
  - **Section 3 (Ba Trụ Cột Khoa Học Vững Chắc Phục Vụ Bảo Vệ Đồ Án Trước Hội Đồng UTH)**:
    - Pillar 1: Empirical Validation Over Heuristic Guessing (Challenge Recall jumps from 11.71% to 91.89%, RV5.1 Recall from 66.45% to 93.13%).
    - Pillar 2: Zero Identity Leakage Invariant ($D_3$ clean partition, 626 RV5.1 test images strictly quarantined).
    - Pillar 3: Cross-Generator Generalization on 20,400 Images (DDIM x2.89, SD 2.1 x1.78, DiT x2.45, GAN Inversion 21.50%, FF++ C23 video compression FPR $\le 2.60\%$).
  - **Section 4 (Hướng Dẫn Điều Hướng Sang Các Phần Chuyên Sâu)**:
    - Direct relative links to Notebook 01, Notebook 02, and Notebook 03.
- Updated CLI builder argument parser in `scripts/build_unified_coursework_suite.py` to support `--notebook 0` and `--notebook all`.
- Executed `build_notebook_00()`: Generated in 8.1 seconds with 10 cells and 2 rich composite figures.
- Ran `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_00 -v`:
  - Result: **3 passed in 0.04s**.
- Ran full test suite:
  - `pytest tests/test_notebook_outputs.py`: **12 passed in 0.11s**.
  - `make test-smoke`: **33 passed in 5.19s**.
  - `make lint`: Passed with zero syntax/compilation errors.

---

## Verification & Boundary Checklist
- [x] Zero mock data: Live PyTorch forward inference executed on real checkpoint `active_learning_m3_best.pt` with eager attention extraction.
- [x] Exact metric alignment: Live Golden Scorecard matches official reports ($49.07\% \to 69.44\%$ Master Test, $11.71\% \to 91.89\%$ Challenge Vault, $66.45\% \to 93.13\%$ Unseen RV5.1 Vault, DDIM $17.33\%$, SD 2.1 $8.33\%$, DiT $3.67\%$, GAN Inv $21.50\%$).
- [x] Zero username / path leaks: Absolute paths and local user identifiers (`bush`) strictly absent and verified by automated regex assertion.
- [x] All figures cleanly closed: `plt.close("all")` called on every figure; zero dangling figures.
- [x] File size: 1.7 MB (strictly `< 2.5 MB`).
- [x] Zero `SyntaxWarning` occurrences across stdout/stderr streams.
- [x] Code executed with 0 errors across all 4 code cells.
