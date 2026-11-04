# 2026-11-04-checkpoint-losses-and-resumable-training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement full-state checkpointing with crash recovery (`--resume`), composite segmentation loss (Cross-Entropy + Soft Dice), and the complete resumable training pipeline in `src/pipelines/train.py`.

**Architecture:** Build `src/common/checkpoint.py` with RNG-preserving state saving/loading adhering to `.agents/rules/LOGGING_CHECKPOINT_RULES.md`, implement `CompositeOcclusionLoss` in `src/models/losses.py`, and wire validation evaluation + resumable execution loop into `src/pipelines/train.py`.

**Tech Stack:** PyTorch 2.15, Torchvision, NumPy 2.5, PyYAML, pytest.

**Spec:** [.agents/rules/LOGGING_CHECKPOINT_RULES.md](.agents/rules/LOGGING_CHECKPOINT_RULES.md), [docs/phases/PHASE_02_SEGMENTATION_BASELINE_SPEC.md](docs/phases/PHASE_02_SEGMENTATION_BASELINE_SPEC.md).

## Global Constraints

- 100% Vectorized processing; zero raw pixel loops.
- Checkpoints must preserve: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_metric`, `history`, `rng_state`.
- Save `weights_only=False` or structured dict when restoring optimizer/RNG.
- Dual logging: StreamHandler to stdout + FileHandler to `experiments/runs/<run_name>/logs/<run_name>.log`.
- Zero raw `print()` in source files; use `src.common.logger`.

## Review Focus

- RNG state desynchronization on CPU vs GPU: Ensure both `torch.get_rng_state()` and `torch.cuda.get_rng_state_all()` (if CUDA available) are saved.
- Dice loss zero denominator: Add epsilon $\epsilon = 10^{-6}$ smoothing to prevent NaN gradients on absent classes.
- Resuming past total epochs: If requested resume checkpoint has `epoch >= total_epochs`, cleanly log warning and finish without crashing.
- Single-channel mask targets: Softmax/One-hot encoding inside Dice loss must match target shape `(B, C, H, W)`.
- Directory auto-creation: Ensure checkpoint and log parent directories are created automatically before writing.

---

### Task 1: Full-State Checkpoint & Resume Utility (`src/common/checkpoint.py`)

**Files:**
- Create: `src/common/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Produces:
  - `save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str | Path, run_name: str) -> tuple[Path, Path | None]`
  - `load_checkpoint(checkpoint_path: str | Path, model: nn.Module, optimizer: Optimizer | None, scheduler: Any | None, device: torch.device) -> dict`

- [ ] **Step 1: Write unit tests in `tests/test_checkpoint.py`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `save_checkpoint` and `load_checkpoint` in `src/common/checkpoint.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit changes**

---

### Task 2: Composite Occlusion Loss (`src/models/losses.py`)

**Files:**
- Create: `src/models/losses.py`
- Modify: `src/models/__init__.py`
- Test: `tests/test_losses.py`

**Interfaces:**
- Produces:
  - `CompositeOcclusionLoss(num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0, class_weights: list[float] | None = None, ignore_index: int = 255)`

- [ ] **Step 1: Write unit tests in `tests/test_losses.py`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `CompositeOcclusionLoss` in `src/models/losses.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit changes**

---

### Task 3: Resumable Training Pipeline (`src/pipelines/train.py`)

**Files:**
- Modify: `src/pipelines/train.py`
- Test: `tests/test_pipelines.py`

**Interfaces:**
- Consumes: `FacialOcclusionSegmenter`, `CompositeOcclusionLoss`, `save_checkpoint`, `load_checkpoint`, `compute_miou`.
- Produces: Executable CLI supporting `--config`, `--resume`, and `--dry-run`.

- [ ] **Step 1: Update unit tests in `tests/test_pipelines.py` to test `--dry-run` and checkpoint persistence**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Refactor `src/pipelines/train.py` with resume logic, evaluation loop, and dual logging**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit changes**
