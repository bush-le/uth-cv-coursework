# Task 3 Completion Report: Build & Live-Execute Notebook 01 — Real Manifold & Dual-Engine Portfolio

## Executive Summary
Task 3 of the Coursework Notebooks Redesign has been completed following strict Test-Driven Development (TDD). `notebooks/coursework/01_diffusion_dataset_and_curation.ipynb` was generated via `build_notebook_01()` inside `scripts/build_unified_coursework_suite.py`, live-executed through IPython, and verified by `tests/test_notebook_outputs.py` and `make test-smoke`.

- **Commits**:
  - `f572829` (`feat(notebooks): generate and verify notebook 01 real manifold and curation pipeline`)
  - `2c3fdc7` (`fix(notebooks): address review feedback on notebook 01 and CurationGate logic`)
- **Generated Notebook**: `notebooks/coursework/01_diffusion_dataset_and_curation.ipynb` (1.7 MB, well below 2.5 MB ceiling)
- **Test File**: `tests/test_notebook_outputs.py` (3 smoke tests, 100% passing)
- **Suite Smoke Gate**: `make test-smoke` (24 passed in 5.16s)

---

## TDD Implementation Flow

### 1. RED Phase
- Created `tests/test_notebook_outputs.py` with 3 `@pytest.mark.smoke` test cases:
  1. `test_notebook_01_exists_and_size`: Checks file existence and ensures file size is strictly under 2.5 MB.
  2. `test_notebook_01_schema_and_navigation`: Verifies nbformat v4 schema, institutional Cell 0 header (Topic 11, UTH), and Cell 1 roadmap table highlighting Part 1.
  3. `test_notebook_01_execution_outputs`: Verifies zero execution errors across all code cells, presence of both stream outputs and matplotlib `display_data` figures, and zero leakage of local usernames or absolute paths.
- Ran `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_01 -v`:
  - Result: **3 failed** as expected (`Notebook 01 file does not exist`).

### 2. GREEN Phase
- Implemented `build_notebook_01(shell=None, output_path=None)` in `scripts/build_unified_coursework_suite.py`:
  - **Cell 0**: UTH Institutional Header & Topic 11 objectives.
  - **Cell 1**: 4-column Roadmap Table highlighting Part 1 (`🎯 / (Đang xem)`) with valid relative links to project modules (`src/data/raw_ingestion.py`, `src/pipelines/curate_image_dataset.py`).
  - **Section 1 (Real Manifold & Demographics)**:
    - Ingests real metadata from `data/raw/manifest.jsonl` (4,000 real images: 2,000 FFHQ + 2,000 CelebA-HQ).
    - Computes demographic distributions (gender, 10 age brackets, eyewear accessories) and renders horizontal bar charts.
    - Computes genuine human skin color gamut in ITU-R BT.601 YCbCr space ($Y, Cb, Cr$) from real face pixels on disk, displaying sample faces alongside a $Cb-Cr$ chrominance density hexbin plot and luminance histogram.
  - **Section 2 (Dual-Engine Portfolio & Feathering Mechanics)**:
    - Demonstrates soft-mask Gaussian feathering ($\sigma=4.0$) vs hard-edge boundary inpainting.
    - Visualizes 5-panel comparative grid: Real Face, Soft Mask, Hard-edge Blending, Soft-mask Blending, and horizontal edge gradient profile proving dramatic boundary spike suppression.
    - Provides a 4-panel visual comparison grid across Stable Diffusion 1.5 (T2I, Inpaint, Img2Img) and RealisticVision v5.1.
  - **Section 3 (3-Tier CurationGate Deep Audit)**:
    - Executes live `ImageCurationGate.evaluate_image()` measuring Laplacian sharpness variance ($\tau = 1.25$) on real and degraded test images.
    - Parses actual audit logs from `data/synthetic/d0_train_curation_report_att0.json` and `data/synthetic/master_test_curation_report_att0.json`.
    - Renders an audit dashboard: Donut chart for yield rate (>75% primary yield), horizontal bar chart breakdown of top 5 anomaly rejection categories (landmark loss, extreme face scale, border amputation, multiple faces, blowout), and Laplacian score distribution histogram.
  - **Section 4 (Conclusion & Roadmap Segue)**:
    - Summarizes Stage 1 findings and provides relative link to Part 2 (`02_hard_samples_and_active_learning.ipynb`).
- Executed `build_notebook_01()`: Generated in ~5 seconds with 16 cells and 5 high-resolution figures (110 DPI).
- Ran `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_01 -v`:
  - Result: **3 passed in 0.06s**.
- Ran `make test-smoke`:
  - Result: **24 passed in 5.19s**.

### 3. Review Feedback & Post-Review Hardening (Commit `2c3fdc7`)
Following peer code review, 4 specific issues were resolved and verified:
1. **Self-Contained Module Imports**: Added root path insertion to `sys.path` in `scripts/build_unified_coursework_suite.py` (`_ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_ROOT))`) so the generator script runs standalone without external `PYTHONPATH` exports.
2. **LaTeX Escape Sequences**: Fixed double-escaped backslashes in matplotlib LaTeX strings (`\\\\sigma=4.0`) across section titles, subtitles, and labels in both `scripts/build_unified_coursework_suite.py` and `notebooks/coursework/01_diffusion_dataset_and_curation.ipynb`, removing Python 3.12+ `SyntaxWarning: invalid escape sequence '\s'` from cell stderr streams.
3. **CurationGate Sharpness Calibration**: Adjusted `min_sharpness` from 1.25 to 5.0 in the live evaluation cell of Notebook 01 (`ImageCurationGate(min_sharpness=5.0, clip_enabled=False, face_backend="none")`), ensuring degraded blurred anomalies are properly flagged and rejected (`[❌ ĐÀO THẢI (REJECT)] Synth Blurred Anomaly | Lý do từ chối: Sharpness 4.08 below threshold 5.00`) instead of incorrectly passing.
4. **Automated Warning Regression Testing**: Added a test assertion in `tests/test_notebook_outputs.py` (`assert "SyntaxWarning" not in stream_text`) to permanently protect against `SyntaxWarning` regressions in executed cell outputs.

- Re-ran `.venv/bin/pytest tests/test_notebook_outputs.py -k test_notebook_01 -v`:
  - Result: **3 passed in 0.06s**.
- Re-ran `make test-smoke`:
  - Result: **24 passed in 5.16s**.

---

## Verification & Boundary Checklist
- [x] Ground Truth real manifold: 4,000 actual face images indexed from `data/raw/manifest.jsonl`.
- [x] Zero mock data: All demographic stats, skin pixel chrominance values, and curation metrics calculated from real disk files.
- [x] Zero username / path leaks: Absolute paths and local user identifiers (`bush`) strictly forbidden and verified absent.
- [x] All figures cleanly closed: Zero lingering matplotlib figures in session memory.
- [x] File size: 1.7 MB (strictly `< 2.5 MB`).
- [x] Code executed with 0 errors.
