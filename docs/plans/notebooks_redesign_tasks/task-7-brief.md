# Task 7 Brief: Update Governance Docs & End-to-End Suite Integrity Verification

## Target
Finalize `notebooks/README.md`, implement `tests/test_notebook_suite_integrity.py`, and run the complete smoke verification across the entire project.

## Requirements
1. **Governance Documentation (`notebooks/README.md`)**:
   - Provide a clean, comprehensive guide for instructors, grading committees, and students.
   - Describe the 4-Part Unified Narrative Suite:
     - `00_master_deepfake_defense_showcase.ipynb`: Master showcase & live forensic inspector.
     - `01_diffusion_dataset_and_curation.ipynb`: Real manifold & dual-engine curation pipeline.
     - `02_hard_samples_and_active_learning.ipynb`: DINOv2 architecture & active learning progression.
     - `03_champion_m3_and_universal_benchmark.ipynb`: Zero leakage $D_3$, Champion $M_3$ & Universal benchmark (20.4k images).
   - Document the archival of the 10 legacy notebooks in `notebooks/coursework/legacy_v1/`.
   - Explain how to view or re-render notebooks using `scripts/build_unified_coursework_suite.py`.
   - Zero absolute paths, zero local usernames.
2. **End-to-End Integrity Suite (`tests/test_notebook_suite_integrity.py`)**:
   - Marked `@pytest.mark.smoke`.
   - Verifies all 4 notebooks exist in `notebooks/coursework/`.
   - Verifies individual size < 2.5 MB and aggregate suite size < 10 MB.
   - Verifies valid Jupyter nbformat 4 JSON schema.
   - Verifies that Cell 0 is the UTH institutional header and Cell 1 is the 4-part Roadmap table for all 4 notebooks.
   - Verifies all relative links inside the notebooks resolve to valid files.
   - Verifies zero error outputs / tracebacks in any code cell.
3. **Full Project Verification**:
   - Run `pytest tests/test_notebook_suite_integrity.py -v`.
   - Run `make test-smoke`.
   - Run `make lint`.
   - Confirm git status is clean.
