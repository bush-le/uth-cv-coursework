# PLAN_NOTEBOOK_06_SPECIALIZATION_VS_GENERALIZATION.md — Implementation Plan for Notebook 06 & Synthesis

- **Motivation/Background**: Fulfill the user's explicit request for a writing-plan and Coursework Notebook 06 to demonstrate and analyze the trade-off between Deep Adversarial Specialization ($M_2$) and Broad Out-of-Distribution Generalization ($M_1$).
- **Purpose**: Provide actionable, granular, test-driven steps to implement the required visualizer functions, cascaded defense simulator, and generate pre-rendered Notebook 06.
- **Overview Pipeline**: Task 1: Cascaded defense simulator in `src/common/metrics.py` -> Task 2: Multi-vault trade-off visualizer in `src/common/visualizer.py` -> Task 3: Unit tests in `tests/test_metrics.py` and `tests/test_visualizer.py` -> Task 4: Script `scripts/build_notebook_06.py` and generate `notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb` -> Task 5: Full verification and documentation sync.
- **Detailed Plan**: Task 1 through Task 5 with checkable steps.
- **References**: `docs/phases/PHASE_NOTEBOOK_06_SPECIALIZATION_VS_GENERALIZATION.md`, `GEMINI.md`, `experiments/logs/model_comparison_benchmark.json`.
- **Created**: 2026-11-11T08:38:00+07:00
- **Last Updated**: 2026-11-11T17:40:00+07:00

---

> **Status:** COMPLETED. With the refined continual learning model $M_2^*$ achieving 70.13% Fake Recall standalone on the 6,000 SD benchmark, the project operates with a single optimal model ($M_2^*$) eliminating multi-stage ensemble overhead. Notebook 06 is fully built and pre-rendered.

**Goal:** Implement the multi-vault trade-off visualizer and create a comprehensive, pre-rendered Coursework Notebook 06 explaining the empirical progression across $M_0 \rightarrow M_1 \rightarrow M_2^*$.

**Architecture:** Add modular simulation logic to `src/common/metrics.py`, multi-vault visualizer to `src/common/visualizer.py`, test both via `pytest`, and build self-contained Notebook 06 via `scripts/build_notebook_06.py`.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Matplotlib, JSON, Jupyter Notebook, pytest.

**Spec:** [`docs/phases/PHASE_NOTEBOOK_06_SPECIALIZATION_VS_GENERALIZATION.md`](docs/phases/PHASE_NOTEBOOK_06_SPECIALIZATION_VS_GENERALIZATION.md)

## Global Constraints
- Clean modular code in `src/`: no raw monolithic logic inside notebooks (`GEMINI.md`).
- No absolute system paths in code or markdown files.
- Full type annotations (PEP 484) on all new signatures.
- Keep test suite passing 100% green at all times.

## Review Focus
- Check handling of zero division when no samples pass Stage 1 in `simulate_cascaded_defense`.
- Verify multi-vault data alignment across challenge benchmark and external 6k benchmark.
- Ensure all notebook cells contain automatic root bootstrap to avoid `ModuleNotFoundError: No module named 'src'`.

---

### Task 1: Cascaded Defense Simulation Function in `src/common/metrics.py`

**Files:**
- Modify: `src/common/metrics.py`
- Test: `tests/test_metrics.py`

**Steps:**
- [x] Add `simulate_cascaded_defense` in `src/common/metrics.py` accepting `m1_probs`, `m2_probs`, `labels`, `tau1=0.30`, `tau2=0.50`, returning dictionary of metrics: `stage1_intercept_rate`, `overall_accuracy`, `overall_precision`, `overall_recall`, `overall_bypass_rate`, `false_positive_rate`.
- [x] Add unit test in `tests/test_metrics.py` covering standard and edge-case inputs.
- [x] Run `.venv/bin/pytest tests/test_metrics.py` and verify all tests pass.

---

### Task 2: Multi-Vault Trade-Off Visualizer in `src/common/visualizer.py`

**Files:**
- Modify: `src/common/visualizer.py`
- Test: `tests/test_visualizer.py`

**Steps:**
- [x] Implement `plot_tri_vault_tradeoff_analysis(challenge_benchmark_path, legacy_6k_benchmark_path, output_path)` in `src/common/visualizer.py`.
- [x] Layout 3 panels:
  - Panel 1: Cumulative Challenge Vault ($M_0$ vs $M_1$ vs $M_2$: Accuracy, ROC-AUC, Hard Recall).
  - Panel 2: Master Frozen Test Vault ($M_0$ vs $M_1$ vs $M_2$: Precision, Recall, Accuracy).
  - Panel 3: External 6k Unseen Benchmark ($M_0$ vs $M_1$ vs $M_2$: Recall $\tau=0.5$ vs Recall $\tau=0.3$).
- [x] Add unit test in `tests/test_visualizer.py`.
- [x] Run `.venv/bin/pytest tests/test_visualizer.py` and verify all tests pass.

---

### Task 3: Notebook 06 Builder Script & Generation

**Files:**
- Create: `scripts/build_notebook_06.py`
- Output: `notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb`

**Steps:**
- [x] Write `scripts/build_notebook_06.py` constructing the notebook with 11 rich cells (Header, Imports & Root setup, Section 1: Debunking the Myth, Section 2: Multi-Vault Tri-Model Visual Dashboard, Section 3: Feature Space & Calibration Mechanism, Section 4: Cascaded Two-Stage Defense Simulation, Section 5: Thesis Defense & Examiner Q&A Guide).
- [x] Pre-render all figures into base64 PNGs and outputs.
- [x] Run script to generate `notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb`.
- [x] Verify generated notebook structure and JSON validity.

---

### Task 4: End-to-End Verification & Documentation Update

**Files:**
- Modify: `docs/progress/PLAN_NOTEBOOK_06_SPECIALIZATION_VS_GENERALIZATION.md`
- Modify: `docs/phases/PHASE_NOTEBOOK_06_SPECIALIZATION_VS_GENERALIZATION.md`

**Steps:**
- [x] Run full test suite: `.venv/bin/pytest tests/` (147/147 passed 100% green).
- [x] Mark completed tasks in plan.
- [x] Present Notebook 06 and key scientific findings to user with clickable links.
