# Task 4 Brief: Build & Live-Execute Notebook 02 — Hard-Sample Mining & Active Learning

## Target
Implement `build_notebook_02()` inside `scripts/build_unified_coursework_suite.py` to generate:
`notebooks/coursework/02_hard_samples_and_active_learning.ipynb`
and implement output verification test `test_notebook_02` in `tests/test_notebook_outputs.py`.

## Detailed Content Requirements for Notebook 02
1. **Institutional Header (Cell 0)**:
   - Topic: TOPIC 11 - Stable Diffusion Synthetic Facial Dataset Generation for Deepfake Detection
   - Group: Group 4 (UTH Computer Vision Coursework)
   - Notebook: Notebook 02 / 03 (Part 2: Detection Architecture, Hard-Sample Mining & Active Learning Feedback Loop)
   - Objective: Demystify the DINOv2 ViT-S/14 detector architecture, classify biological failure modes of diffusion generation (eyes, teeth, ears, hair), and validate active learning progression ($M_0 \to M_1 \to M_2^*$) via live inference on the 213 Challenge Vault samples.
2. **Roadmap Navigation (Cell 1)**:
   - 4-part table linking all 4 notebooks with relative paths. Part 2 marked as `🎯 / (Đang xem)`.
3. **Section 1: Detector Architecture & Feature Extraction**:
   - DINOv2 ViT-S/14 (`DeepfakeImageDetector` from `src.models.detector`).
   - Tokenization: 224x224 input $\to$ $16 \times 16 = 256$ spatial patch tokens + 1 `[CLS]` token = 257 tokens, embedding dimension $D = 384$.
   - Classification head: LayerNorm + Linear + Dropout + Linear $\to$ binary logit.
   - Code snippet showing layer summary and parameter breakdown (unfrozen Block 11 vs frozen backbone).
4. **Section 2: Hard-Sample Mining & Biological Defect Taxonomy**:
   - Total 740 hard samples mined across rounds: 553 from `data/active_learning/round_1/hard_samples.jsonl` + 187 from `data/active_learning/round_2/hard_samples.jsonl`.
   - Challenge Vault: 213 samples (`data/active_learning/cumulative_challenge_test.jsonl`).
   - Biological Defect Taxonomy:
     1. Pupil asymmetry & irregular corneal reflection
     2. Ear cartilage / helix malformation
     3. Teeth unnatural alignment & border smearing
     4. Hair fine texture loss & skin pore blur
   - Visual display of representative challenging faces from `cumulative_challenge_test.jsonl`.
5. **Section 3: Active Learning Progression & Live Challenge Vault Inference**:
   - Load checkpoint $M_0$ (`experiments/checkpoints/baseline_d0_best.pt`) and $M_2^*$ (`experiments/checkpoints/active_learning_m2_best.pt`).
   - Run live forward inference on the 213 Challenge samples (`torch.no_grad()`, `eval()` mode).
   - Compute metrics:
     - $M_0$: Fake Recall = 11.71%, Bypass Rate = 88.29%
     - $M_2^*$: Fake Recall = 91.89%, Bypass Rate = 8.11%
     - Massive +80.18% recall jump!
   - Plot confusion matrices side-by-side ($M_0$ vs $M_2^*$), ROC curve, and active learning recall progression bar chart ($M_0: 11.71\% \to M_1: 70.27\% \to M_2^*: 91.89\%$).
6. **Execution & Size Constraints**:
   - Notebook file size < 2.5 MB.
   - Zero empty/blank outputs.
   - All plots cleanly closed with `plt.close("all")`.
   - Zero hardcoded usernames or absolute paths.
7. **Verification**:
   - Implement `test_notebook_02` in `tests/test_notebook_outputs.py`.
   - Run `pytest tests/test_notebook_outputs.py -k test_notebook_02 -v` and `make test-smoke`.
