# SDD ledger — plan: docs/plans/2026_11_16_COURSEWORK_NOTEBOOKS_REDESIGN_PLAN.md

## Pre-flight Conflict Scan

| Task Pair / Task | Produced vs Consumed / Self-Consistency | Finding | Ruling |
|---|---|---|---|
| Task 1 vs Task 2-7 | Legacy notebooks moved to `legacy_v1/`, leaves `notebooks/coursework/` ready for new suite | Clean. No name collisions between old `00_master_deepfake_defense_coursework.ipynb` and new `00_master_deepfake_defense_showcase.ipynb`. | Proceed. Legacy files safely preserved. |
| Task 2 vs Tasks 3-6 | Task 2 produces builder helpers; Tasks 3-6 consume them to build notebooks | Clean interface. Helper signatures `make_markdown_cell`, `make_code_cell`, `execute_and_capture` match consumption. | Proceed. |
| Task 3 vs Dataset | Notebook 01 consumes `data/raw/manifest.jsonl`, `d0_train_curation_report_att0.json` | Clean. Manifest and report files exist and verified on disk. | Ruling: compute real skin gamut from actual raw images instead of synthetic Gaussian curve. |
| Task 4 vs Checkpoints | Notebook 02 consumes `active_learning_m2_best.pt`, `baseline_d0_best.pt`, `cumulative_challenge_test.jsonl` | Clean. Checkpoints exist in `experiments/checkpoints/` and Challenge Vault has 213 samples. | Ruling: run live inference on all 213 Challenge samples (~2s) to guarantee exact match with 11.71% -> 91.89% report. |
| Task 5 vs Checkpoints | Notebook 03 consumes `active_learning_m3_best.pt`, `unseen_rv51_test.jsonl`, `benchmark_5models_*.csv` | Clean. All 626 test-derived RV5.1 samples isolated in `unseen_rv51_test.jsonl`, summary tables exist. | Ruling: live inference on sample of unseen RV5.1, macro table ingestion for 20.4k universal benchmark. |
| Task 6 vs Inspector | Notebook 00 consumes DINOv2 self-attention layer for spatial heatmap | Clean. `AutoModel.from_pretrained('facebook/dinov2-small', attn_implementation='eager')` outputs 12 attention layers. | Ruling: use eager attention or patch activations for spatial attention heatmap in live inspector. |
| Task 7 vs Suite | Validates file size (<2.5MB each, <10MB total), nbformat v4, Cell 0/1 blueprint | Clean. Strict gate for final branch acceptance. | Proceed. |

## Tasks Progress

- [x] Task 1: Archive Legacy Notebooks & Prepare Architecture Blueprint (Commit: `820369c`, Approved by Reviewer)
- [x] Task 2: Build Interactive Automation Engine for Notebook Generation (Commit: `8f90fc7`, Approved by Reviewer)
- [x] Task 3: Build & Live-Execute Notebook 01 — Real Manifold & Dual-Engine Portfolio (Commit: `2c3fdc7`, Approved by Reviewer)
- [x] Task 4: Build & Live-Execute Notebook 02 — Hard-Sample Mining & Active Learning (Commit: `99bbad4`, Approved by Reviewer)
- [x] Task 5: Build & Live-Execute Notebook 03 — Champion M3 & Universal Benchmark (Commit: `44272d3`, Approved by Reviewer)
- [x] Task 6: Build & Live-Execute Notebook 00 — Master Showcase & Live Forensic Lecture (Commit: `1fee36a`, Approved by Reviewer)
- [x] Task 7: Update Governance Docs & End-to-End Suite Smoke Verification (Commit: `3633f4e`, Approved by Reviewer)
