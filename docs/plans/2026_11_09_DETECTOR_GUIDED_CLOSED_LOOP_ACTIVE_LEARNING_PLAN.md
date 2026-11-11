# DETECTOR_GUIDED_CLOSED_LOOP_ACTIVE_LEARNING_PLAN.md — Detector-Guided Closed-Loop Active Learning Architecture & Execution Plan

- **Motivation/Background**: Conventional deepfake generation pipelines suffer from "blind generation": producing thousands of synthetic images without detector vulnerability feedback, resulting in high redundancy, low hard-sample yield, and rapid detector plateauing. A detector-guided active learning loop uses calibrated model misclassifications (False Negatives & High-Uncertainty Boundary samples) to adaptively steer combinatorial prompt grammar, generation modalities (T2I, I2I, Inpaint), and facial mask regions, systematically breaking detector blindspots over successive rounds ($M_0 \to M_1 \to M_2 \dots$).
- **Purpose**: Serve as the comprehensive, authoritative implementation plan and benchmark blueprint for Topic 11 (Facial Deepfake Detection under Active Adversarial Feedback) with Dual-Benchmark evaluation, stateful round resumption, and 9-channel inpainting UNet integration.
- **Overview Pipeline**: Baseline Detector $M_0$ $\to$ Modality Budget Allocation (Tier 1) & Regional Mask Targeting (Tier 2) $\to$ Combinatorial Prompt Generation (7 slots, CLIP $\le 60$ words) $\to$ Synthesis Pool Generation (T2I / I2I / 9-ch Inpaint) $\to$ Quality Curation Gate $\to$ Calibrated Inference (Validation Temperature Scaling) $\to$ Hard-Sample Mining ($H_r$) $\to$ Hard Split ($85\%$ Train, $15\%$ Challenge Test) $\to$ Error Attribution Feedback $\to$ Dataset Expansion ($D_r = D_{r-1} \cup H_r^{\text{train}}$) $\to$ Retraining $M_r$ $\to$ Dual-Benchmark Evaluation (Master Frozen Test + Cumulative Challenge Test) $\to$ Trajectory Logging.
- **Detailed Plan**: §1 Executive Architecture & Core Innovations; §2 Global Constraints & Guardrails; §3 Completed Milestones (Tasks 1–4); §4 Task 5: Dual-Benchmark Splitting & Evaluation Integration; §5 Task 6: Multi-Round Production Scaling & Proof-of-Utility Benchmark; §6 Task 7: Active Learning Progression Visualizer; §7 Task 8: Coursework Deliverables & Documentation Synchronization.
- **References**: [GEMINI.md](GEMINI.md), [SESSION_HANDOVER.md](SESSION_HANDOVER.md), [docs/progress/SESSION_HANDOVER_2026_11_09.md](docs/progress/SESSION_HANDOVER_2026_11_09.md), [configs/dataset_factory.yaml](configs/dataset_factory.yaml), [src/pipelines/run_closed_loop_active_learning.py](src/pipelines/run_closed_loop_active_learning.py).
- **Created**: 2026-11-09T19:26:00+07:00
- **Last Updated**: 2026-11-09T20:05:00+07:00

---

# Detector-Guided Closed-Loop Active Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` (or `superpowers:subagent-driven-development`) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an end-to-end detector-guided closed-loop active learning system where detector feedback adaptively steers prompt synthesis, modality quotas, and regional facial masks, demonstrating measurable reduction in detector bypass rate under a rigorous Dual-Benchmark evaluation protocol.

**Architecture:** A 2-Tier Adaptive Active Learning Loop with a Dual-Benchmark evaluation vault:
- **Tier 1 (Modality Quotas):** Dynamically distributes synthesis quotas across Text-to-Image (T2I), Image-to-Image (I2I), and Inpainting based on mode-level False Negative rates with a strict $15\%$ exploration floor and Hare-Niemeyer largest-remainder allocation.
- **Tier 2 (Attribute & Regional Steering):** Modulates Inpainting masks toward vulnerable facial regions (`periocular`, `lips`, `nose`, `full`) and samples 7-slot combinatorial photographic prompts via softmax error weighting with a CLIP token limit ($\le 60$ words) and a Demographic Shield.
- **Dual-Benchmark Evaluation:**
  1. *Master Frozen Test Set (Anchor Vault):* Identity-disjoint test set held completely frozen across all rounds ($M_0 \to M_1 \to M_2$) to detect catastrophic forgetting and provide unpolluted apples-to-apples ROC-AUC tracking.
  2. *Cumulative Challenge Test Set (Dynamic Vault):* Slices $15\%$ of each round's hard samples ($H_1^{\text{test}} \cup H_2^{\text{test}} \dots$) into an expanding adversarial challenge suite to measure hard-sample recall and bypass rate elimination without train-test leakage.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, Torchvision, Hugging Face Diffusers (Stable Diffusion v1.5 base & 9-channel UNet `stable-diffusion-v1-5/stable-diffusion-inpainting`), DINOv2 ViT (Linear Probe / LayerNorm), OpenCV, NumPy, Pytest.

**Spec:** [docs/OVERVIEW.md](docs/OVERVIEW.md), [SESSION_HANDOVER.md](SESSION_HANDOVER.md), [GEMINI.md](GEMINI.md).

---

## Global Constraints & Invariants

1. **Strict No-Mock Invariant (`GEMINI.md §2`)**: Never use mock data, synthetic noise, or uninitialized random-weight heads for evaluation or mining. All mining and inference must execute on real checkpoints (`best_model.pt`).
2. **Dual-Benchmark Identity & Split Disjointness**:
   - Master Frozen Test Set remains 100% untouched during candidate generation, prompt weighting, and detector retraining.
   - For every mined hard sample set $H_r$, samples are split with a deterministic hash: $85\%$ to $H_r^{\text{train}}$ and $15\%$ to $H_r^{\text{test}}$. No sample in $H_r^{\text{test}}$ can ever enter the training pool $D_r$.
3. **GPU Hygiene & VRAM Ceiling (3.73 GB VRAM)**:
   - RTX 3050 Laptop GPU constraint: Base diffusion and inpainting pipelines use `torch.float16`, attention slicing, and explicit cleanup (`del engine; gc.collect(); torch.cuda.empty_cache()`) between pipeline steps.
   - Prompt grammar engine and budget controllers execute strictly on CPU in pure NumPy/Python (< 1ms, 0 MB VRAM).
4. **Reproducibility**: All dynamic slot sampling, seed generation, and dataset splits derive deterministically via CRC32 hashing from user-configured seeds (`src.common.seed.seed_everything`).
5. **Clean Code & Strict Typing**: 100% vectorized NumPy/PyTorch math (zero pixel loops), zero `print()` in `src/` (use `src.common.logger.get_logger`), full PEP 484 type annotations.

---

## Review Focus & Failure Mode Mitigations

1. **Catastrophic Forgetting during Hard-Sample Fine-Tuning**: Over-indexing on hard adversarial samples ($H_r$) causes the detector to forget general deepfake artifacts or inflate false positive rates on real faces.
   - *Mitigation:* Master Frozen Test Set benchmark run after every round; replay buffer retains base dataset $D_0$ with hard-sample oversampling factor $\le 2.0$.
2. **Demographic Distribution Collapse**: Generator focuses exclusively on a single demographic slice that fooled the detector, collapsing diversity.
   - *Mitigation:* Demographic Shield enforces uniform demographic balancing across candidate generation slots.
3. **Uncalibrated Confidence Misleading Mining**: An uncalibrated detector outputting extreme probabilities ($P=0.01$ or $P=0.99$) corrupts hard sample boundaries.
   - *Mitigation:* Validation temperature scaling optimization ($T > 0$) fitted via PyTorch before mining.
4. **CLIP Token Window Truncation**: Overly verbose combinatorial prompts getting clipped by the 77-token CLIP text encoder.
   - *Mitigation:* Hard 60-word length ceiling in `PromptGrammarEngine.generate_prompt()`.
5. **Stateful Round Desynchronization**: Process interruption causing lost progress or double-training of earlier rounds.
   - *Mitigation:* Automated stateful resume scanning `data/active_learning/round_*` and `experiments/checkpoints/active_learning_m*_best.pt` to seamlessly resume from round $r+1$.

---

## File Structure

```
src/
├── models/
│   ├── diffusion/
│   │   ├── prompt_engine.py             # DONE: Dynamic 7-Slot Combinatorial Grammar Engine with Softmax Sampling
│   │   └── stable_diffusion_engine.py   # DONE: SD v1.5 base & 9-channel UNet Inpainting engine
│   └── evaluators/
│       ├── calibrated_mining.py         # DONE: Temperature scaling & dual-criterion hard-sample mining
│       └── hardness.py                  # DONE: Real-manifold proximity & LOO tau
├── pipelines/
│   ├── budget_controller.py             # DONE: Hierarchical 2-Tier Budget & Regional Mask Controller
│   ├── run_closed_loop_active_learning.py # IN-PROGRESS: Master Active Learning Orchestrator with Dual-Benchmark
│   ├── generate_deepfake_images.py      # DONE: SD generation backend with dynamic quotas
│   ├── curate_image_dataset.py          # DONE: Laplacian blur & SHA-256 deduplication gate
│   └── train.py                         # DONE: Multi-stage benchmark training
scripts/
└── plot_active_learning_progression.py   # NEW (Task 7): Active learning visualization suite
tests/
├── test_prompt_engine.py                # DONE: 4/4 passed
├── test_budget_controller.py            # DONE: 4/4 passed
├── test_calibrated_mining.py            # DONE: 3/3 passed
└── test_closed_loop_active_learning.py  # IN-PROGRESS: Integration & Dual-Benchmark tests
```

---

## Completed Tasks (Milestones 1–4)

### Task 1: Dynamic Combinatorial Grammar Engine (`prompt_engine.py`)
- [x] **Step 1: Write unit tests** (`tests/test_prompt_engine.py`)
- [x] **Step 2: Implement `PromptGrammarEngine`** in `src/models/diffusion/prompt_engine.py`:
  - 7 orthogonal slots: `subject_age`, `subject_gender`, `optics`, `lighting`, `expression`, `occlusion`, `imperfections`.
  - Softmax error weighting: $P(\text{tag}_k) \propto \exp(\text{ErrorRate}_k / \tau)$ with a 15–20% exploration floor.
  - Hard 60-word length cap to strictly prevent CLIP 77-token truncation.
- [x] **Step 3: Verification**: `pytest tests/test_prompt_engine.py` (4 passed).
- [x] **Step 4: Commit**: `71e0398` (`feat(diffusion): implement dynamic combinatorial prompt grammar engine with error weighting`).

### Task 2: Hierarchical 2-Tier Budget & Mask Controller (`budget_controller.py`)
- [x] **Step 1: Write unit tests** (`tests/test_budget_controller.py`)
- [x] **Step 2: Implement `AdaptiveBudgetController`** in `src/pipelines/budget_controller.py`:
  - Tier 1: Softmax modality quotas ($T2I, I2I, Inpaint$) with $\ge 15\%$ exploration floor and Hare-Niemeyer exact sum distribution.
  - Tier 2: Regional inpainting mask target allocator (`periocular`, `lips`, `nose`, `full`).
  - Tier 2: Demographic Shield enforcing balanced demographic round-robin distribution to prevent shortcut learning.
- [x] **Step 3: Verification**: `pytest tests/test_budget_controller.py` (4 passed).
- [x] **Step 4: Commit**: `89a96f7` (`feat(pipeline): implement hierarchical 2-tier adaptive budget and mask controller`).

### Task 3: Calibrated Hard-Sample Mining (`calibrated_mining.py`)
- [x] **Step 1: Write unit tests** (`tests/test_calibrated_mining.py`)
- [x] **Step 2: Implement `CalibratedHardMiner`** in `src/models/evaluators/calibrated_mining.py`:
  - Temperature scaling optimization: $\min_T \text{NLL}(\text{logits}/T, \text{labels})$.
  - Dual-criterion selection: False Negatives ($\hat{p} < 0.50$) + High Uncertainty ($|\hat{p} - 0.50| \le 0.10$).
  - Perceptual diversity deduplication (cosine distance threshold $\ge 0.05$).
- [x] **Step 3: Verification**: `pytest tests/test_calibrated_mining.py` (3 passed).
- [x] **Step 4: Commit**: `522ea7e` (`feat(evaluators): implement temperature-calibrated dual-criterion hard sample miner`).

### Task 4: Master Closed-Loop Orchestrator Foundation & Stateful Auto-Resume
- [x] **Step 1: Write integration tests** (`tests/test_closed_loop_active_learning.py`)
- [x] **Step 2: Implement orchestration loop** in `src/pipelines/run_closed_loop_active_learning.py`:
  - Chains Modality Quotas $\to$ Generation $\to$ Curation Gate $\to$ Calibrated Inference $\to$ Hard Mining $\to$ Error Attribution $\to$ Retraining $\to$ Evaluation.
  - Stateful round resume: Auto-detects completed rounds and picks up from $r+1$.
  - Curation gate wrapper with SHA-256 deduplication and blur filtering.
- [x] **Step 3: Live Verification Run on CUDA GPU**:
  - Generated 15 candidates, curated 7, fitted $T=0.7705$, mined 7 hard samples ($H_1$), updated grammar weights across 7 slots, expanded dataset to $D_1$ (112 samples, 119 oversampled), retrained DINOv2 detector $M_1$ (Val AUC: $0.6121 \to 0.6848$), evaluated achieving **0.7543 Test ROC-AUC (100% precision)**.
- [x] **Step 4: Commit**: `dc395bf`, `d2c6052`, `79e03fc`.

---

## Remaining Implementation Tasks

### Task 5: Dual-Benchmark Splitting & Evaluation Integration

**Motivation:** An active learning detector should not just be evaluated on a static test set or a contaminated pool. We must cleanly split mined hard samples: $85\%$ goes into the training buffer $D_r$, while $15\%$ enters the expanding Cumulative Challenge Test Set ($T^{\text{challenge}}$). Both the Master Frozen Test Set ($T^{\text{master}}$) and $T^{\text{challenge}}$ are evaluated at each round.

**Files:**
- Modify: `src/pipelines/run_closed_loop_active_learning.py`
- Modify: `tests/test_closed_loop_active_learning.py`

**Interfaces:**
- In `run_closed_loop_active_learning.py`:
  - Add hard-sample split: `split_hard_samples(mined_hard: List[ImageMetadata], test_ratio: float = 0.15, seed: int = 42) -> Tuple[List[ImageMetadata], List[ImageMetadata]]`
  - Maintain `challenge_test_records: List[ImageMetadata]` across rounds, saved to `data/active_learning/cumulative_challenge_test.jsonl`.
  - Evaluate detector on both:
    - `master_test_metrics = evaluate_image_detector(round_detector, master_test_loader)`
    - `challenge_test_metrics = evaluate_image_detector(round_detector, challenge_test_loader)`
  - Log both metrics distinctly in `closed_loop_metrics.json`.

- [x] **Step 1: Write unit/integration test for Dual-Benchmark splitting in `test_closed_loop_active_learning.py`**
  - Verified no sample overlap between train and challenge test records.
  - Verified dry run records `master_test_metrics` and `challenge_test_metrics`.

- [x] **Step 2: Run test to verify failure/gap**
  - Verified before implementation.

- [x] **Step 3: Implement `split_hard_samples` and Dual-Benchmark logging in `run_closed_loop_active_learning.py`**
  - Deterministic random/hash splitting.
  - Challenge test vault maintained at `data/active_learning/cumulative_challenge_test.jsonl`.
  - Evaluations executed on both master and challenge loaders.

- [x] **Step 4: Run test suite to verify all pass**
  - All 122 repository unit and integration tests passing (`taskset -c 0-3 pytest`).

- [x] **Step 5: Commit changes**
  - Committed as `7c416c9` (`feat(pipeline): implement task 5 dual-benchmark splitting with cumulative challenge test`).

---

### Task 6: Multi-Round Production Scaling & Proof-of-Utility Execution

**Motivation:** Execute the active learning pipeline over multiple rounds ($R \ge 2$) with production sample volume on the RTX 3050 GPU, demonstrating empirical convergence:
1. $M_0 \to M_1 \to M_2$ ROC-AUC trajectory on Master Frozen Test.
2. Progressive elimination of Bypass Rate ($FNR$) on the Cumulative Challenge Test.
3. Dynamic shift of generation quotas toward the hardest modalities and mask regions.

**Execution:**
- [ ] **Step 1: Verify environment and checkpoints**
  - Check GPU availability and existing checkpoint states:
    ```bash
    python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"
    ```
- [ ] **Step 2: Run Multi-Round Active Learning Pipeline**
  - Execute 2 rounds with 60 candidates per round:
    ```bash
    python3 -m src.pipelines.run_closed_loop_active_learning \
        --rounds 2 \
        --candidates-per-round 60 \
        --epochs 5 \
        --device cuda
    ```
- [ ] **Step 3: Verify outputs and state logs**
  - Verify `data/active_learning/round_1/` and `data/active_learning/round_2/` manifests.
  - Verify `experiments/logs/closed_loop_metrics.json` contains full trajectory records for both rounds.
- [ ] **Step 4: Commit generated metrics and logs**
  - `git add experiments/logs/closed_loop_metrics.json`
  - `git commit -m "chore(experiments): record multi-round active learning benchmark metrics"`

---

### Task 7: Active Learning Progression Visualizer (`plot_active_learning_progression.py`)

**Motivation:** Visual evidence is essential for Coursework Topic 11. Create a publication-ready multi-panel visualization script that reads `experiments/logs/closed_loop_metrics.json` and plots:
1. **Modality & Mask Budget Evolution**: Stacked bar chart showing quota shifts ($T2I \to I2I \to Inpaint$, periocular vs mouth vs nose).
2. **ROC-AUC & Precision Trajectory**: Dual-curve on Master Frozen Test vs Cumulative Challenge Test.
3. **Bypass Rate Elimination**: Sharp decline of false negative bypass rate across rounds ($M_0 \to M_1 \to M_2$).
4. **Hard-Sample Yield & Temperature Convergence**: Temperature parameter $T$ and mining yield % per round.

**Files:**
- Create: `scripts/plot_active_learning_progression.py`
- Output: `experiments/predictions/active_learning_progression.png`

- [ ] **Step 1: Implement `scripts/plot_active_learning_progression.py`**
  - Clean Matplotlib 4-panel figure with scientific styling (`seaborn-v0_8-whitegrid` or clean grid).
  - Handles dry-run or live metrics JSON gracefully.
  - Vectorized calculations, zero hardcoded paths, CLI `--metrics-path` and `--output-path`.
- [ ] **Step 2: Execute visualizer on live metrics**
  - Run: `python3 scripts/plot_active_learning_progression.py --metrics experiments/logs/closed_loop_metrics.json --output experiments/predictions/active_learning_progression.png`
  - Expected: PNG generated without errors.
- [ ] **Step 3: Commit visualizer and generated plot**
  - `git add scripts/plot_active_learning_progression.py experiments/predictions/active_learning_progression.png`
  - `git commit -m "feat(vis): implement publication-grade active learning progression visualizer"`

---

### Task 8: Coursework Deliverables & Documentation Synchronization

**Motivation:** Ensure all coursework progress, architectural findings, and instructions are fully aligned with the authoritative handover documents.

**Files:**
- Modify: `docs/progress/SESSION_HANDOVER_2026_11_09.md`
- Modify: `docs/OVERVIEW.md`
- Create: `docs/reports/DETECTOR_GUIDED_ACTIVE_LEARNING_REPORT.md`

- [ ] **Step 1: Generate comprehensive report `docs/reports/DETECTOR_GUIDED_ACTIVE_LEARNING_REPORT.md`**
  - Follow standard 7-field header (`GEMINI.md`).
  - Summarize theoretical formulation ($M_t \to H_t \to D_{t+1} \to M_{t+1}$), grammar matrix, Dual-Benchmark results, and ablation comparison.
- [ ] **Step 2: Update `docs/progress/SESSION_HANDOVER_2026_11_09.md` & `SESSION_HANDOVER.md`**
  - Synchronize completed milestones, current branch commit hashes, and reproduction commands.
- [ ] **Step 3: Run full verification suite (`make test-smoke` / `pytest`)**
  - Verify 100% test pass rate across all 119+ unit and integration tests.
- [ ] **Step 4: Final Git Push**
  - Commit and push to branch `baseline/initial-baseline`.

---

## Definition of Done (DoD)

- [ ] All 8 tasks implemented, tested, and marked complete (`[x]`).
- [ ] Unit & integration test suite achieves 100% pass rate (`pytest` >= 120 tests).
- [ ] Closed-loop active learning successfully runs on GPU without VRAM exhaustion.
- [ ] Dual-Benchmark tables generated and documented without train-test leakage.
- [ ] Publication plot `experiments/predictions/active_learning_progression.png` generated.
- [ ] All markdown files follow 7-field header and UPPERCASE naming invariants.
