# External 6k Legacy Synthetic Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the 6,000 external synthetic face images from `/home/bush/Downloads/synthetic_face_dataset_6k.zip`, benchmark detector models ($M_0, M_1, M_2$) for out-of-distribution Fake Recall and Bypass Rate, log the structured metrics, and integrate rich visual comparisons into Notebook 04.

**Architecture:** Standalone benchmark architecture under `data/external_benchmarks/`, keeping the production 10,000 dataset isolated. Dedicated evaluation script `scripts/benchmark_legacy_6k.py` running GPU batch inference across models, persisting results to `experiments/logs/legacy_6k_benchmark_results.json`, visualized via `src.common.visualizer` and embedded into `04_active_learning_production_visualizer.ipynb`.

**Tech Stack:** PyTorch, torchvision, PIL, Matplotlib, JSONL, pytest.

**Spec:** [`docs/phases/PHASE_EXTERNAL_6K_BENCHMARK.md`](file:///home/bush/Projects/uth-cv-coursework/docs/phases/PHASE_EXTERNAL_6K_BENCHMARK.md)

## Global Constraints
- Strictly isolate `data/external_benchmarks/` from the production 10,000 dataset in `data/partitions/` and `data/benchmarks/`.
- No absolute system paths in code or documentation (`GEMINI.md`).
- Pure fake detection evaluation contract: compute Fake Recall ($P \ge 0.5$) and Bypass Rate ($P < 0.5$).
- Full type annotations (PEP 484) on all new Python signatures.
- Keep test suite passing 100% green at all checkpoints.

## Review Focus
- Invalid or missing image paths in extracted zip: verify all 6,000 paths resolve correctly before inference.
- GPU VRAM consumption: use batch size 32 with `torch.no_grad()` to maintain stable memory footprint.
- Task attribution breakdown: handle all tasks (`inpaint`, `img2img`, `text2img`) gracefully even if counts vary.
- Region attribution: gracefully handle samples where `sub_region` is null or missing.
- Notebook self-contained execution: ensure project root bootstrap is preserved in all updated notebook cells.

---

### Task 1: Dataset Extraction & Manifest Rebasing

**Files:**
- Create: `data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl`
- Create: `scripts/prepare_legacy_6k_dataset.py`

**Steps:**
- [x] Write `scripts/prepare_legacy_6k_dataset.py` to extract `/home/bush/Downloads/synthetic_face_dataset_6k.zip` to `data/external_benchmarks/synthetic_face_dataset_6k/`.
- [x] Rebase paths in `fake_manifest.jsonl` from `data/synthetic/...` to `data/external_benchmarks/synthetic_face_dataset_6k/...`.
- [x] Run script and verify 6,000 image files and manifest records exist on disk.

---

### Task 2: Dedicated GPU Benchmark Execution Pipeline

**Files:**
- Create: `scripts/benchmark_legacy_6k.py`
- Output: `experiments/logs/legacy_6k_benchmark_results.json`
- Output: `experiments/logs/legacy_6k_predictions.npz`

**Steps:**
- [x] Write `scripts/benchmark_legacy_6k.py` loading `fake_manifest.jsonl` and creating a PyTorch `DataLoader` with eval transforms.
- [x] Implement inference loop evaluating $M_0$ (`baseline_d0_best.pt`), $M_1$ (`active_learning_m1_best.pt`), and $M_2$ (`active_learning_m2_best.pt`).
- [x] Calculate Overall Recall, Bypass Rate, Mean Fake Probability, breakdown by `task` (`inpaint`, `img2img`, `text2img`), and breakdown by `sub_region` (`eyes`, `mouth`, `full`).
- [x] Persist per-sample raw predictions to `experiments/logs/legacy_6k_predictions.npz`.
- [x] Execute `scripts/benchmark_legacy_6k.py` on CUDA and verify JSON & NPZ output structure.

---

### Task 3: Visualizer Module & Unit Test Certification

**Files:**
- Modify: `src/common/visualizer.py`
- Modify: `tests/test_visualizer.py`

**Steps:**
- [x] Implement `plot_legacy_6k_benchmark(benchmark_path, output_path)` in `src/common/visualizer.py`.
- [x] Implement `plot_legacy_6k_distributions(npz_path, output_path)` in `src/common/visualizer.py`.
- [x] Implement `plot_legacy_6k_gallery_inspection(npz_path, output_path)` in `src/common/visualizer.py`.
- [x] Implement `plot_legacy_sample_inspection(npz_path, sample_idx, output_path)` in `src/common/visualizer.py`.
- [x] Implement `render_legacy_6k_markdown_table(benchmark_path, npz_path)` in `src/common/visualizer.py`.
- [x] Add unit tests in `tests/test_visualizer.py`.
- [x] Run `.venv/bin/pytest tests/test_visualizer.py` and verify all tests pass.

---

### Task 4: Coursework Notebook Integration & End-to-End Verification

**Files:**
- Create: `scripts/build_legacy_6k_notebook.py`
- Create: `notebooks/coursework/05_legacy_6k_out_of_distribution_benchmark.ipynb`
- Modify: `notebooks/coursework/04_active_learning_production_visualizer.ipynb`

**Steps:**
- [x] Integrate legacy 6k benchmark summary into Notebook 04.
- [x] Create dedicated, self-contained notebook `notebooks/coursework/05_legacy_6k_out_of_distribution_benchmark.ipynb` with 15 rich sections, pre-rendered charts, and interactive single-sample inspector.
- [x] Run full test suite `.venv/bin/pytest tests/` to confirm 146/146 green.
