# Task 3 Brief: Build & Live-Execute Notebook 01 — Real Manifold & Dual-Engine Portfolio

## Target
Implement `build_notebook_01()` inside `scripts/build_unified_coursework_suite.py` to generate:
`notebooks/coursework/01_diffusion_dataset_and_curation.ipynb`
and implement output verification test in `tests/test_notebook_outputs.py` (specifically `test_notebook_01`).

## Detailed Content Requirements for Notebook 01
1. **Institutional Header (Cell 0)**:
   - Topic: TOPIC 11 - Stable Diffusion Synthetic Facial Dataset Generation for Deepfake Detection
   - Group: Group 4 (UTH Computer Vision Coursework)
   - Notebook: Notebook 01 / 03 (Part 1: Real Manifold, Dual-Engine Portfolio & Curation Pipeline)
   - Objective: Establish ground truth real manifold (4k faces), explore dual-engine generation (SD 1.5 + RealisticVision v5.1) with soft-mask feathering, and audit 3-tier CurationGate filtering.
2. **Roadmap Navigation (Cell 1)**:
   - Standard 4-part table linking all 4 notebooks with relative paths. Part 1 marked as `🎯 / (Đang xem)`.
3. **Section 1: Real Face Manifold (4,000 Images)**:
   - Code loads `data/raw/manifest.jsonl`.
   - Displays real dataset distribution (FFHQ vs CelebA-HQ, resolution 512x512, label 0).
   - Computes genuine color space metrics (e.g. YCbCr / HSV skin gamut or RGB channel distributions) from actual sample images on disk (using `PIL.Image` or `cv2` and `numpy`).
   - Visualizes real sample faces and color gamut distribution plot.
4. **Section 2: Dual-Engine Synthetic Generation Portfolio**:
   - Compares Stable Diffusion 1.5 (RunwayML base) and RealisticVision v5.1.
   - Demonstrates the inpainting blending mechanics:
     - Real face vs Inpainting mask vs Soft-mask Gaussian blurred feathering ($\sigma=4$) vs Blended result.
     - Explains why hard-edge inpainting creates high-frequency boundary artifacts that detectors easily overfit, while soft-mask feathering forces the detector to learn facial texture anomalies.
5. **Section 3: 3-Tier CurationGate Deep Audit**:
   - Loads genuine curation reports from `data/synthetic/d0_train_curation_report_att0.json` (and `data/synthetic/master_test_curation_report_att0.json`).
   - Breaks down the 3 tiers:
     - Tier 1: Artifact & Blur rejection (Laplacian variance threshold).
     - Tier 2: Landmark & Identity fidelity (Cosine similarity).
     - Tier 3: Visual inspection & class balance.
   - Plots pass/fail yield rate pie chart / bar chart (>85% yield rate) and metric distribution histograms.
6. **Execution & Size Constraints**:
   - Notebook must be fully executed and saved to `notebooks/coursework/01_diffusion_dataset_and_curation.ipynb`.
   - File size must be strictly < 2.5 MB.
   - Zero markdown or code cells with errors or blank/missing outputs.
   - All matplotlib figures cleanly closed (`plt.close("all")`).
   - Zero hardcoded local usernames or absolute paths (use relative paths).
7. **Verification**:
   - Create/update `tests/test_notebook_outputs.py` with `test_notebook_01` marked `@pytest.mark.smoke`.
   - Run `pytest tests/test_notebook_outputs.py -k test_notebook_01 -v` and `make test-smoke`.
