# Task 5 Brief: Build & Live-Execute Notebook 03 — Champion M3 & Universal Benchmark

## Target
Implement `build_notebook_03()` inside `scripts/build_unified_coursework_suite.py` to generate:
`notebooks/coursework/03_champion_m3_and_universal_benchmark.ipynb`
and implement output verification test `test_notebook_03` in `tests/test_notebook_outputs.py`.

## Detailed Content Requirements for Notebook 03
1. **Institutional Header (Cell 0)**:
   - Topic: TOPIC 11 - Stable Diffusion Synthetic Facial Dataset Generation for Deepfake Detection
   - Group: Group 4 (UTH Computer Vision Coursework)
   - Notebook: Notebook 03 / 03 (Part 3: Champion M3 Model, Cross-Generator Generalization & Universal Deepfake Benchmark)
   - Objective: Audit the Zero Identity Leakage partition ($D_3$), inspect Champion $M_3$ Warm-Start fine-tuning dynamics, benchmark Quad-Vault performance, and analyze cross-generator generalization on the 20,400-image Universal Benchmark across 8 generator families.
2. **Roadmap Navigation (Cell 1)**:
   - 4-part table linking all 4 notebooks with relative paths. Part 3 marked as `🎯 / (Đang xem)`.
3. **Section 1: Zero Identity Leakage Audit & $D_3$ Clean Partitioning**:
   - Audit 6,000 RealisticVision v5.1 images:
     - 626 images (313 inpaint + 313 img2img) were derived from 313 real identities in `test_reals.jsonl`.
     - Strict isolation: quarantined to `data/benchmarks/D3_clean/unseen_rv51_test.jsonl` (1,215 images total: 626 RV5.1 fakes + 589 test reals).
   - Training partition $D_3$: `data/benchmarks/D3_clean/train.jsonl` (5,626 images: 2,813 real vs 2,813 fake, 1:1 balance).
   - Verify intersection of identities between train and test is strictly 0 (Zero Identity Leakage).
4. **Section 2: Champion $M_3$ Training Dynamics & Quad-Vault Benchmark**:
   - Champion architecture: Warm-Start from $M_2^*$ (`experiments/checkpoints/active_learning_m2_best.pt`), unfreeze Block 11 + Head (1.78M params, 8.06% trainable).
   - Differential LR: $\eta_{head} = 3 \times 10^{-5}$, $\eta_{block11} = 3 \times 10^{-6}$ with Cosine Annealing (6 epochs, val AUC 0.9745 $\to$ 0.9846).
   - Load checkpoint $M_3$ (`experiments/checkpoints/active_learning_m3_best.pt`).
   - Run live forward inference on a subset of `unseen_rv51_test.jsonl` to confirm high detection fidelity.
   - Load `benchmark_5models_quad_vault_summary.csv` and visualize the 3 key breakthroughs:
     1. Master Frozen Test (SD 1.5): $M_0$ 49.07% $\to$ $M_2^*$ 66.89% $\to$ $M_3$ 69.44%!
     2. Challenge Vault: $M_0$ 11.71% $\to$ $M_2^*$ 91.89% $\to$ $M_3$ 91.89%!
     3. Unseen RV5.1 Vault: $M_0$ 66.45% $\to$ $M_2^*$ 71.88% $\to$ $M_3$ 93.13% (bypass drops to 6.87%)!
5. **Section 3: Universal Benchmark (20,400 Images, 34 Generators, 8 Families)**:
   - Ingest `experiments/benchmarks/tables/benchmark_5models_global_summary.csv` and `benchmark_5models_family_summary.csv`.
   - Global metrics comparison across all 5 models ($M_0, M_1, M_2^{cold}, M_2^*, M_3$).
   - Family Breakdown visual radar / bar chart:
     - DDIM: 6.00% $\to$ 17.33% (almost 3x gain)
     - SD 2.1: 4.67% $\to$ 8.33% (almost 2x gain)
     - DiT: 1.50% $\to$ 3.67% (2.45x gain)
     - GAN Inversion: 12.50% $\to$ 21.50%
     - GAN Architectures: 7.33% $\to$ 17.33%
   - Interactive Decision Threshold Sweep: sweep $\tau \in [0.1, 0.9]$ showing the operational trade-off between recall and false alarms.
6. **Execution & Size Constraints**:
   - Notebook size < 2.5MB.
   - Zero empty/blank outputs.
   - All plots cleanly closed with `plt.close("all")`.
   - Zero hardcoded usernames or absolute paths.
7. **Verification**:
   - Implement `test_notebook_03` in `tests/test_notebook_outputs.py`.
   - Run `pytest tests/test_notebook_outputs.py -k test_notebook_03 -v` and `make test-smoke`.
