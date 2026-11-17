# Task 6 Brief: Build & Live-Execute Notebook 00 — Master Showcase & Live Forensic Lecture

## Target
Implement `build_notebook_00()` inside `scripts/build_unified_coursework_suite.py` to generate:
`notebooks/coursework/00_master_deepfake_defense_showcase.ipynb`
and implement output verification test `test_notebook_00` in `tests/test_notebook_outputs.py`.

## Detailed Content Requirements for Notebook 00
1. **Institutional Header (Cell 0)**:
   - Topic: TOPIC 11 - Stable Diffusion Synthetic Facial Dataset Generation for Deepfake Detection
   - Group: Group 4 (UTH Computer Vision Coursework)
   - Notebook: Notebook 00 / 03 (Part 0: Master Defense Showcase, Golden Scorecard & Live Forensic Inspector)
   - Objective: Provide the master executive summary of the entire coursework, present the Golden Scorecard across all 4 experimental milestones ($M_0 \to M_3$), and demonstrate an interactive Live Forensic Inspector with DINOv2 spatial attention heatmaps.
2. **Roadmap Navigation (Cell 1)**:
   - 4-part table linking all 4 notebooks with relative paths. Part 0 marked as `🎯 / (Đang xem)`.
3. **Section 1: Executive Summary & Golden Scorecard**:
   - Coursework narrative: From initial baseline vulnerabilities ($M_0$), through hard-sample mining and active learning ($M_1, M_2^*$), to multi-generator champion generalization ($M_3$).
   - Golden Scorecard interactive table comparing:
     - Model name & training dataset
     - Master Frozen Test Recall & Bypass
     - Challenge Vault Recall & Bypass
     - Unseen RV5.1 Vault Recall & Bypass
     - Universal Benchmark (20.4k) Recall & AUC
   - Multi-metric visual radar or grouped bar chart comparing all 4 generations.
4. **Section 2: Live Forensic Inspector with DINOv2 Spatial Attention Heatmaps**:
   - Load Champion model $M_3$ checkpoint (`experiments/checkpoints/active_learning_m3_best.pt`).
   - Spatial Attention Extraction:
     - Extract attention weights from DINOv2 small backbone (`torch.no_grad()`, `eval()` mode).
     - Compute the attention distribution from `[CLS]` token to the $16 \times 16 = 256$ spatial patch tokens.
     - Reshape to $(16, 16)$, normalize, and upsample to $(224, 224)$ or $(512, 512)$ using bilinear interpolation or Gaussian smoothing.
   - Side-by-side diagnostic display:
     - Left: Original facial image (Real face vs Synthetic face).
     - Center: DINOv2 Spatial Attention Heatmap overlay (highlighting high-attention regions in `jet` or `inferno` colormap).
     - Right: Prediction summary card (Probability, Predicted Class: Real / Fake, Confidence, Forensic Focus Region).
5. **Section 3: Defense Arguments for UTH Committee**:
   - Three irrefutable scientific pillars:
     1. Empirical Validation Over Heuristic Guessing (real metric jumps from 11.71% to 91.89% and 66.45% to 93.13%).
     2. Zero Identity Leakage Invariant (clean $D_3$ partition, zero train/test overlap).
     3. Cross-Generator Generalization on 20,400 Images (DDIM recall 3x, Latent Diffusion recall 2x).
6. **Execution & Size Constraints**:
   - Notebook size < 2.5MB.
   - Zero empty/blank outputs.
   - Figures cleanly closed with `plt.close("all")`.
   - Zero hardcoded usernames or absolute paths.
7. **Verification**:
   - Implement `test_notebook_00` in `tests/test_notebook_outputs.py`.
   - Run `pytest tests/test_notebook_outputs.py -k test_notebook_00 -v` and `make test-smoke`.
