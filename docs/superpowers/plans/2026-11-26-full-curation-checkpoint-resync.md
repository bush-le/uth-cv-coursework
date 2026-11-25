# Full Extended-Curation Dataset, Checkpoint & Benchmark Resync Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize 100% of synthetic datasets (`D0` -> `D3_clean`, `data/hard_samples/`), detector checkpoints (`M0` -> `M1` -> `M2_coldstart` -> `M2*` -> `M3`), benchmark tables/figures, HTML reports, and coursework notebooks with the upgraded `extended_audit` curation pipeline (`YuNet`, eye-pair, cheek FFT, CIELab delta-E, boundary seam, perceptual dHash dedup, mode-aware CLIP) without regenerating the raw Stable Diffusion pools unnecessarily.

**Architecture:** Archive the v1 Nov 10–12 artifacts cleanly for rollback/A-B traceability, re-curate all existing raw SD 1.5 and RV5.1 pools on disk under the extended curation gate, retrain all five `DeepfakeImageDetector` checkpoints in exact causal order on the RTX 3050 4GB GPU, recompute all benchmark suites without stale prediction caches, and update report/notebook builders and smoke assertions to read dynamic counts/metrics from disk rather than Nov 10–12 constants.

**Tech Stack:** Python 3.14 (`.venv/bin/python`), PyTorch + CUDA (`NVIDIA GeForce RTX 3050 Laptop GPU`, 4GB VRAM), OpenCV YuNet/Haar, DINOv2-small (`facebook/dinov2-small`), CLIP, Pytest (`make test-smoke`, `make test`).

## Global Constraints
- Follow `AGENTS.md`: never place training loops inside `notebooks/`, never commit absolute local filesystem paths in Markdown docs, and run `make test-smoke` after code/test changes.
- Preserve identity-disjoint partitions (`train_reals.jsonl`, `val_reals.jsonl`, `test_reals.jsonl`) and verify 0 identity overlap across all rebuilt manifests.
- Before overwriting `experiments/checkpoints/active_learning_m2_best.pt` with refined `M2*`, copy the Round 2 cold-start checkpoint to `experiments/checkpoints/active_learning_m2_coldstart_backup.pt`.
- When running `scripts/evaluate_universal_benchmark_5models.py`, always pass `--force-recompute` so cached M0–M2* prediction NPZ files from v1 are not reused silently.

## Review Focus
- **Sealed Master Vault vs Clean Resync:** `src/pipelines/run_block1_baseline.py` preserves `data/benchmarks/master_frozen_test.jsonl` when `>= 1178` rows exist unless `--overwrite-master-test` is passed; choose consistently whether `master_frozen_test.jsonl` stays frozen for v1-vs-v2 comparison or gets rebuilt with curated test fakes (`--overwrite-master-test`).
- **Stale Cached Universal Predictions:** `scripts/evaluate_universal_benchmark_5models.py` reuses cached predictions for `M0..M2*` by default; omitting `--force-recompute` produces hybrid v1/v2 metrics.
- **Cold-Start M2 Backup Overwrite:** Running `scripts/train_m2_refined.py` overwrites `experiments/checkpoints/active_learning_m2_best.pt`; failing to copy Cold-Start `M2` first breaks 5-model comparison.
- **Hardcoded V1 Counts in Report Builders & Smoke Tests:** `tests/test_unified_suite_builder.py:200` (`assert len(labels) == 213`) and `tests/test_acceptance_presentation.py:74-77` (`"213"`, `"93.13%"`, `"91.89%"`, `"11.71%"`), plus narrative blocks in `scripts/build_acceptance_report_vertical.py`, `scripts/build_acceptance_presentation.py`, and `scripts/build_unified_coursework_suite.py`, will fail or drift out of sync unless bound to live JSON/JSONL outputs.
- **Perceptual/Quality Yield Variance in D0:** Re-curating Block 1 `d0_train` (1,770 raw), `d0_val` (239 raw), `master_test` (736 raw) under stricter `extended_audit` may leave one split slightly below target (`589 / 1411 / 200`) under `--curate-only`; inspect per-split yields immediately and decide whether to top up shortfall via `--no-resume` or accept calibrated count.

---

### Task 1: Archive Pre-Resync V1 State for Rollback & Comparison

**Files:**
- Create directory: `experiments/archive_v1/` (local gitignored backup, not tracked in git)

- [ ] **Step 1: Snapshot existing manifests, checkpoints, and benchmark tables**

```bash
mkdir -p experiments/archive_v1/data experiments/archive_v1/experiments
cp -a data/synthetic experiments/archive_v1/data/
cp -a data/active_learning experiments/archive_v1/data/
cp -a data/benchmarks experiments/archive_v1/data/
cp -a data/hard_samples experiments/archive_v1/data/
cp -a experiments/checkpoints experiments/archive_v1/experiments/
cp -a experiments/benchmarks experiments/archive_v1/experiments/
cp -a experiments/logs experiments/archive_v1/experiments/
```

Expected: All Nov 10–12 baseline data/checkpoints/metrics preserved intact inside `experiments/archive_v1/` (~333G free disk available).

- [ ] **Step 2: Commit pending code bug fixes cleanly before overwriting training artifacts**

Review `git status -s` and commit verified `src/`, `scripts/`, and `tests/` changes so code history and the upcoming training run are cleanly separated.

---

### Task 2: Re-Curate Block 1 (`D0_baseline`) & Retrain Baseline `M0`

**Files:**
- Modify on disk: `data/synthetic/d0_*_curated*.jsonl`, `data/synthetic/master_test_fakes_curated.jsonl`, `data/benchmarks/D0_baseline/*.jsonl`, `experiments/checkpoints/baseline_d0_*.pt`, `experiments/logs/block1_baseline_metrics.json`

- [ ] **Step 1: Re-curate existing D0 raw attempts without regenerating (including `master_frozen_test.jsonl` for 100% full-sync), and inspect yields**

Run:
```bash
.venv/bin/python -m src.pipelines.run_block1_baseline \
  --config configs/dataset_factory.yaml \
  --curate-only \
  --overwrite-master-test \
  --epochs 5 \
  --device cuda \
  --seed 42
```

*(Decision locked: `--overwrite-master-test` re-curates `master_frozen_test.jsonl` under `extended_audit` while preserving v1 inside `experiments/archive_v1/` for offline comparison.)*

- [ ] **Step 2: Check curated D0 split counts and decide on shortfall handling**

Run:
```bash
wc -l data/synthetic/d0_train_fakes_curated.jsonl data/synthetic/d0_val_fakes_curated.jsonl data/synthetic/master_test_fakes_curated.jsonl data/benchmarks/D0_baseline/*.jsonl
```

Expected: Target is `1411` train fakes / `200` val fakes / `589` test fakes. If `d0_val` or another split drops slightly below target due to `extended_audit` rejections and exact target quotas are desired, re-run with `--no-resume` instead of `--curate-only` so SD 1.5 generates only the small shortfall.

---

### Task 3: Re-Curate Active Learning Rounds 1–2 & Retrain `M1`, `M2` (Cold-Start)

**Files:**
- Modify on disk: `data/active_learning/round_1/*`, `data/active_learning/round_2/*`, `data/active_learning/cumulative_challenge_test.jsonl`, `data/benchmarks/D1_augmented/*`, `data/benchmarks/D2_hard_boosted/*`, `experiments/checkpoints/active_learning_m1_best.pt`, `experiments/checkpoints/active_learning_m2_best.pt`, `experiments/logs/closed_loop_metrics.json`

- [ ] **Step 1: Re-curate Round 1 and Round 2 raw candidate pools (1,900 each) and retrain M1 -> M2**

Run:
```bash
.venv/bin/python -m src.pipelines.run_closed_loop_active_learning \
  --config configs/dataset_factory.yaml \
  --curate-only \
  --rounds 2 \
  --candidates-per-round 1900 \
  --epochs 5 \
  --device cuda \
  --seed 42
```

- [ ] **Step 2: Archive Cold-Start M2 checkpoint immediately before refined warm-start**

Run:
```bash
cp experiments/checkpoints/active_learning_m2_best.pt experiments/checkpoints/active_learning_m2_coldstart_backup.pt
ls -lh experiments/checkpoints/active_learning_m1_best.pt experiments/checkpoints/active_learning_m2_coldstart_backup.pt
```

Expected: Both `active_learning_m1_best.pt` and `active_learning_m2_coldstart_backup.pt` exist on disk alongside `D1_augmented`, `D2_hard_boosted`, and rebuilt `cumulative_challenge_test.jsonl`.

---

### Task 4: Retrain Refined Warm-Start `M2*`

**Files:**
- Modify on disk: `experiments/checkpoints/active_learning_m2_best.pt`, `experiments/checkpoints/active_learning_m2_epoch_*.pt`
- Consumes: `experiments/checkpoints/active_learning_m1_best.pt`, `data/benchmarks/D2_hard_boosted/{train,val}.jsonl`, `data/benchmarks/master_frozen_test.jsonl`, `data/active_learning/cumulative_challenge_test.jsonl`

- [ ] **Step 1: Train M2* warm-started from new M1**

Run:
```bash
.venv/bin/python scripts/train_m2_refined.py
```

Expected: Logs validation/Master/Challenge AUC & Recall for Epoch 1 and Epoch 2 and writes updated `experiments/checkpoints/active_learning_m2_best.pt` via Pareto checkpoint selection.

---

### Task 5: Curate RV5.1 6K Pool, Rebuild `D3_clean` & Sync `data/hard_samples/` Vault

**Files:**
- Create/Modify on disk: `data/external_benchmarks/synthetic_face_dataset_6k/curated_manifest.jsonl`, `data/external_benchmarks/synthetic_face_dataset_6k/curation_report.jsonl`, `data/benchmarks/D3_clean/{train,val,unseen_rv51_test}.jsonl`, `data/hard_samples/*`

- [ ] **Step 1: Run extended curation on RV5.1 6k external synthetic pool**

Run:
```bash
.venv/bin/python -m src.pipelines.curate_image_dataset \
  --config configs/dataset_factory.yaml \
  --input-manifest data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl \
  --output-manifest data/external_benchmarks/synthetic_face_dataset_6k/curated_manifest.jsonl \
  --raw-manifest data/raw/manifest.jsonl \
  --curation-report data/external_benchmarks/synthetic_face_dataset_6k/curation_report.jsonl \
  --extended-audit
```

- [ ] **Step 2: Build clean identity-disjoint `D3_clean` from curated RV5.1 + updated `D2_hard_boosted`**

Run:
```bash
.venv/bin/python scripts/build_d3_clean_dataset.py --sd-nonhard-quota 800 --seed 42
```

Expected: Output logs `[✓] ZERO IDENTITY LEAKAGE CONFIRMED: 0 leaked IDs found!` and writes balanced 1:1 `D3_clean/train.jsonl`, `val.jsonl`, and `unseen_rv51_test.jsonl`.

- [ ] **Step 3: Synchronize `data/hard_samples/` vault symlinks, audit JSON, and vault README**

Run:
```bash
.venv/bin/python scripts/populate_hard_samples_vault.py
```

Expected: Rebuilds `data/hard_samples/hard_samples_manifest.jsonl`, symlinks, `audit_reports/hardness_report.json`, and `data/hard_samples/README.md` with exact v2 counts.

---

### Task 6: Retrain Champion `M3` Dual-Engine Detector

**Files:**
- Modify on disk: `experiments/checkpoints/active_learning_m3_best.pt`, `experiments/checkpoints/active_learning_m3_last.pt`, `experiments/benchmarks/tables/m3_training_evaluation_summary.json`

- [ ] **Step 1: Train M3 warm-started from M2* on rebuilt `D3_clean`**

Run:
```bash
.venv/bin/python scripts/train_m3_dual_engine.py --epochs 6 --batch-size 16 --seed 42
```

Expected: Saves updated `experiments/checkpoints/active_learning_m3_best.pt` and `experiments/benchmarks/tables/m3_training_evaluation_summary.json`.

---

### Task 7: Re-Evaluate All 5 Models Across Quad-Vault, Legacy 6K, Universal 20.4K & Full Portfolio Audit

**Files:**
- Modify on disk: `experiments/benchmarks/tables/benchmark_5models_*.{csv,json}`, `experiments/benchmarks/figures/*.png`, `experiments/logs/legacy_6k_*`, `experiments/logs/full_portfolio_extended_audit.json`

- [ ] **Step 1: Run 5-Model Quad-Vault Evaluation**

Run:
```bash
.venv/bin/python scripts/evaluate_quad_vault_5models.py
```

- [ ] **Step 2: Recompute Legacy 6K RV5.1 benchmark (512px and 224px comparisons) and Universal 20.4K 5-model benchmark without stale caches**

Run:
```bash
.venv/bin/python scripts/benchmark_legacy_6k.py --target-size 224 && \
cp experiments/logs/legacy_6k_benchmark_results.json experiments/logs/legacy_6k_benchmark_results_224px.json && \
cp experiments/logs/legacy_6k_predictions.npz experiments/logs/legacy_6k_predictions_224px.npz && \
.venv/bin/python scripts/benchmark_legacy_6k.py --target-size 512 && \
.venv/bin/python scripts/evaluate_universal_benchmark_5models.py --target-size 512 --force-recompute && \
.venv/bin/python scripts/scan_full_portfolio_audit.py --summary-out experiments/logs/full_portfolio_extended_audit.json
```

Expected: All CSV/JSON/NPZ benchmark artifacts and figures under `experiments/benchmarks/` and `experiments/logs/` reflect the freshly trained v2 checkpoints.

---

### Task 8: Synchronize Dynamic Report Builders, Notebooks & Hardcoded Test Assertions

**Files:**
- Modify: `scripts/build_acceptance_report_vertical.py`, `scripts/build_acceptance_presentation.py`, `scripts/build_unified_coursework_suite.py`
- Modify: `tests/test_acceptance_presentation.py`, `tests/test_unified_suite_builder.py`
- Regenerate: `docs/reports/presentation_final_acceptance.html`, `docs/reports/presentation_5min_define_and_design.html`, `notebooks/coursework/*.ipynb`

- [ ] **Step 1: Audit and parameterize static v1 metrics/counts in report & notebook builders**

Search `scripts/build_acceptance_report_vertical.py`, `scripts/build_acceptance_presentation.py`, and `scripts/build_unified_coursework_suite.py` for Nov 10–12 constants (`740`, `553`, `187`, `629`, `111`, `213`, `5.626`, `626`, `1.9831`, `93.13%`, `91.89%`, `11.71%`, `94.07%`, `94.37%`, `21.15%`, `1,484`, `1,567`) and replace static prose numbers with formatted values loaded from:
- `experiments/logs/closed_loop_metrics.json`
- `experiments/benchmarks/tables/benchmark_5models_quad_vault_summary.json`
- `experiments/benchmarks/tables/benchmark_5models_global_summary.csv`
- `data/benchmarks/D3_clean/*.jsonl`, `data/active_learning/cumulative_challenge_test.jsonl`, `data/hard_samples/hard_samples_manifest.jsonl`

- [ ] **Step 2: Rebuild all HTML presentations and coursework notebooks**

Run:
```bash
.venv/bin/python scripts/build_acceptance_presentation.py && \
.venv/bin/python scripts/build_acceptance_report_vertical.py && \
.venv/bin/python scripts/build_unified_coursework_suite.py && \
.venv/bin/python scripts/create_coursework_notebooks.py
```

Execute coursework notebooks via Jupyter/nbconvert if refreshed cell outputs are required on disk, ensuring zero error outputs.

- [ ] **Step 3: Update smoke tests in `tests/test_acceptance_presentation.py` and `tests/test_unified_suite_builder.py` to validate against live artifacts**

- In `tests/test_acceptance_presentation.py::test_acceptance_presentation_scientific_metrics`: read expected challenge count and `M0`/`M2*`/`M3` recall percentages directly from `data/active_learning/cumulative_challenge_test.jsonl` and `experiments/benchmarks/tables/benchmark_5models_quad_vault_summary.json` instead of hardcoded `"213"`, `"93.13%"`, `"91.89%"`, `"11.71%"`.
- In `tests/test_unified_suite_builder.py::test_challenge_vault_roc_auc_execution`: assert balanced binary classes, non-empty size (`>= 50` / balanced hard fakes + test reals), and non-NaN AUC instead of `assert len(labels) == 213`.

---

### Task 9: Final Full Verification Gate

- [ ] **Step 1: Verify partition disjointness, symlink integrity, and zero missing files across all rebuilt manifests**

Run custom manifest sanity verification across `data/partitions/*.jsonl`, `data/benchmarks/{D0_baseline,D1_augmented,D2_hard_boosted,D3_clean}/*.jsonl`, and `data/active_learning/*.jsonl`.

- [ ] **Step 2: Run full project quality & test suite**

Run:
```bash
bash -n scripts/run_pipeline.sh && make lint && make type-check && make test-smoke && make test
```

Expected: All checks and 100% of unit/integration tests pass with zero failures or skipped regression assertions.

- [ ] **Step 3: Remove or archive temporary workspace artifacts only after confirming v2 pass**

Verify `git status -s` contains only intended clean code, reports, and notebook diffs (never large `.pt`/`.png` binary data if gitignored), and make clean atomic commits.
