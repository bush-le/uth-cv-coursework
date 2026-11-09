# DETECTOR_GUIDED_CLOSED_LOOP_ACTIVE_LEARNING_PLAN.md — Detector-Guided Closed-Loop Active Learning for Deepfake Dataset Generation

- **Motivation/Background**: Conventional deepfake dataset generation relies on brute-force, one-way generation across arbitrary checkpoints without awareness of detector vulnerabilities. This creates redundant data with low downstream training utility. A closed-loop active learning system leverages detector feedback (False Negatives & Uncertain samples) to adaptively drive generation budgets, regional inpainting masks, and prompt attributes.
- **Purpose**: Formalize, implement, and verify the 2-Tier Detector-Guided Active Learning Pipeline ($M_t \to \text{Calibrated Inference} \to \text{Mine } H_t \to \text{Adaptive Budget/Prompt Allocation} \to D_{t+1} \to M_{t+1}$) with strict demographic fairness safeguards, zero test leakage, and calibrated uncertainty.
- **Overview Pipeline**: Baseline Detector $M_0$ $\to$ Candidate Synthesis Pool (T2I, I2I, Inpaint) $\to$ Calibrated Inference (Temperature Scaling) $\to$ Dual-Criterion Hard Mining (FN + Uncertain) $\to$ Error Attribution & Budget Allocation $\to$ Dynamic Combinatorial Prompt Generation $\to$ Dataset Expansion & Retraining $\to$ Frozen Test Evaluation.
- **Detailed Plan**: §1 Dynamic Combinatorial Grammar Engine; §2 Hierarchical 2-Tier Budget Controller; §3 Calibrated Hard-Sample Mining; §4 Closed-Loop Active Learning Orchestrator; §5 End-to-End Verification & Benchmark Curves.
- **References**: `configs/dataset_factory.yaml`, `src/data/image_schema.py`, `src/pipelines/generate_deepfake_images.py`, `src/pipelines/train.py`, `GEMINI.md`.
- **Created**: 2026-11-09T19:26:00+07:00
- **Last Updated**: 2026-11-09T19:26:00+07:00

---

# Detector-Guided Closed-Loop Active Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` (or `superpowers:subagent-driven-development`) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete closed-loop active learning pipeline where detector misclassification feedback directly guides prompt generation, modality budgets, and regional masks across iterative generation rounds, provably boosting detector robustness.

**Architecture:** A 2-Tier Adaptive Architecture: Tier 1 dynamically allocates generation quotas among T2I, I2I, and Inpainting based on mode-level False Negative rates with a 20% exploration floor. Tier 2 performs granular error attribution, directing Inpainting masks toward vulnerable facial regions (periocular, lips, nose) and modulating T2I prompt tokens using a combinatorial grammar matrix with error-weighted softmax sampling.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, Torchvision, Hugging Face Diffusers (Stable Diffusion v1.5), DINOv2 ViT, OpenCV, NumPy, Pytest.

**Spec:** `docs/OVERVIEW.md`, `configs/dataset_factory.yaml`, `GEMINI.md`.

## Global Constraints

- **Strict No-Mock Invariant**: No synthetic noise, uninitialized weights, or mock data for production evaluation or hard sample mining (`GEMINI.md §2`).
- **Identity-Disjoint Test Protection**: The 43-sample test set in `data/benchmarks/D0_baseline/test.jsonl` remains 100% frozen, held-out, and never touched during candidate generation, mining, or prompt tuning.
- **Zero VRAM Waste**: Prompt engine and budget controllers must execute on CPU in pure Python/NumPy (< 1ms, 0 MB VRAM) to leave all 3.73 GB VRAM for Stable Diffusion FP16 and DINOv2 fine-tuning.
- **Reproducibility**: All dynamic slot sampling and seed generation must derive deterministically via CRC32 hashing from base seeds.
- **Strict Typing**: Full PEP 484 type annotations on all function signatures, zero `print()` in `src/`.

## Review Focus

1. **Demographic Shortcut Learning**: Detector fails exclusively on one demographic slice causing the generator to produce only that group, collapsing diversity. *Mitigation: Demographic Shield enforcing uniform demographic distribution across candidate generation.*
2. **Uncalibrated Confidence Misleading Mining**: An overconfident detector outputs extreme probabilities ($P=0.01$ or $P=0.99$) on out-of-distribution fakes, corrupting hard sample selection. *Mitigation: Temperature scaling calibration fitted on independent validation set before mining.*
3. **Exploration Starvation**: One modality achieves high early yield, starving other modalities and narrowing vulnerability coverage. *Mitigation: Hard minimum 15% exploration quota per generation modality.*
4. **Stale Checkpoint Leakage**: Checkpoints from previous runs or earlier stages contaminating model evaluation. *Mitigation: Explicit prefix namespacing and pre-training cache invalidation.*
5. **Memory Leakage across Iterative Rounds**: PyTorch CUDA memory accumulation during repeated train-inference cycles. *Mitigation: Explicit `torch.cuda.empty_cache()` and context scoping between rounds.*

---

## File Structure

```
src/
├── models/
│   ├── diffusion/
│   │   ├── prompt_engine.py             # NEW: Dynamic Combinatorial Grammar Engine with Softmax Sampling
│   │   └── stable_diffusion_engine.py   # EXISTING: SD generation backend
│   └── evaluators/
│       ├── calibrated_mining.py         # NEW: Temperature scaling & dual-criterion hard-sample mining
│       └── hardness.py                  # EXISTING: Real-manifold proximity & LOO tau
├── pipelines/
│   ├── budget_controller.py             # NEW: 2-Tier Modality Budget & Regional Mask Allocator
│   ├── run_closed_loop_active_learning.py # NEW: Master Active Learning Orchestrator (Rounds M0 -> M1 -> M2)
│   ├── generate_deepfake_images.py      # MODIFY: Integrate prompt_engine & dynamic budget
│   └── train.py                         # EXISTING: Multi-stage benchmark training
tests/
├── test_prompt_engine.py                # NEW: Unit tests for grammar matrix & sampling distribution
├── test_budget_controller.py            # NEW: Unit tests for modality quotas & demographic shield
├── test_calibrated_mining.py            # NEW: Unit tests for temperature scaling & FN/uncertain selection
└── test_closed_loop_active_learning.py  # NEW: Integration test for full loop cycle
```

---

### Task 1: Dynamic Combinatorial Grammar Engine (`prompt_engine.py`)

**Files:**
- Create: `src/models/diffusion/prompt_engine.py`
- Test: `tests/test_prompt_engine.py`

**Interfaces:**
- Produces:
  - `class PromptGrammarEngine`:
    - `__init__(seed: int = 42, exploration_ratio: float = 0.20)`
    - `update_error_weights(error_stats: Dict[str, float], temperature: float = 1.0) -> None`
    - `generate_prompt(demographics: Optional[Dict[str, Any]] = None, mode: str = "text2img", seed_offset: int = 0) -> str`
    - `get_slot_probabilities() -> Dict[str, Dict[str, float]]`

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_prompt_engine.py
import pytest
from src.models.diffusion.prompt_engine import PromptGrammarEngine

def test_prompt_grammar_deterministic_generation():
    engine = PromptGrammarEngine(seed=42)
    p1 = engine.generate_prompt(mode="text2img", seed_offset=100)
    p2 = engine.generate_prompt(mode="text2img", seed_offset=100)
    assert p1 == p2
    assert "RAW photo" in p1 or "portrait" in p1

def test_prompt_grammar_error_weight_adaptation():
    engine = PromptGrammarEngine(seed=42)
    # Simulate high error on direct flash and elderly
    error_stats = {
        "lighting:direct harsh camera flash": 0.85,
        "lighting:soft studio lighting": 0.10,
        "demographics:elderly": 0.90,
    }
    engine.update_error_weights(error_stats)
    probs = engine.get_slot_probabilities()
    assert probs["lighting"]["direct harsh camera flash"] > probs["lighting"]["soft studio lighting"]
    
    # Generate batch and verify increased prevalence of hard token
    prompts = [engine.generate_prompt(mode="text2img", seed_offset=i) for i in range(50)]
    flash_count = sum(1 for p in prompts if "camera flash" in p)
    assert flash_count >= 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.diffusion.prompt_engine'`

- [ ] **Step 3: Implement `PromptGrammarEngine` in `src/models/diffusion/prompt_engine.py`**

Define 4 orthogonal slot dictionaries:
- `subject`: age groups, skin texture descriptors, micro-facial features.
- `optics`: lenses (`35mm`, `85mm f/1.8`, `50mm`), focus types, bokeh depth.
- `lighting`: studio soft, direct flash, side window, natural golden hour, outdoor overcast.
- `occlusion`: rimless glasses, wireframe eyeglasses, sunglasses, hair strands.
Implement softmax sampling with a configurable exploration floor ($\ge 20\%$) to avoid distribution collapse.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_engine.py -v`
Expected: PASS (all tests pass)

- [ ] **Step 5: Commit**

```bash
git add src/models/diffusion/prompt_engine.py tests/test_prompt_engine.py
git commit -m "feat(diffusion): implement dynamic combinatorial prompt grammar engine with error weighting"
```

---

### Task 2: Hierarchical 2-Tier Budget & Mask Controller (`budget_controller.py`)

**Files:**
- Create: `src/pipelines/budget_controller.py`
- Test: `tests/test_budget_controller.py`

**Interfaces:**
- Produces:
  - `class AdaptiveBudgetController`:
    - `__init__(total_candidate_budget: int = 150, min_mode_ratio: float = 0.15, exploration_floor: float = 0.20)`
    - `compute_mode_quotas(mode_errors: Dict[str, float]) -> Dict[str, int]`
    - `compute_inpainting_mask_targets(region_errors: Dict[str, float]) -> List[str]`
    - `enforce_demographic_shield(candidate_records: List[Dict[str, Any]], target_distribution: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]`

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_budget_controller.py
from src.pipelines/budget_controller import AdaptiveBudgetController

def test_budget_allocation_respects_exploration_floor():
    controller = AdaptiveBudgetController(total_candidate_budget=100, min_mode_ratio=0.15)
    # T2I has 0% error, Inpaint has 100% error
    mode_errors = {"sd_text2img": 0.0, "sd_img2img": 0.2, "sd_inpaint": 0.9}
    quotas = controller.compute_mode_quotas(mode_errors)
    
    assert sum(quotas.values()) == 100
    assert quotas["sd_text2img"] >= 15  # Protected by min_mode_ratio
    assert quotas["sd_inpaint"] > quotas["sd_img2img"]

def test_inpainting_regional_targeting():
    controller = AdaptiveBudgetController()
    region_errors = {"periocular": 0.80, "lips": 0.10, "full": 0.10}
    targets = controller.compute_inpainting_mask_targets(region_errors, count=10)
    assert targets.count("periocular") >= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_budget_controller.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `AdaptiveBudgetController` in `src/pipelines/budget_controller.py`**

Implement:
- Tier 1: Softmax-based modality budget allocation with constrained lower bounds ($\ge \text{min\_mode\_ratio}$).
- Tier 2 (Inpainting): Weighted regional target selection (`periocular`, `lips`, `nose`, `full`).
- Tier 2 (Demographic Shield): Stratified quota balancer preventing demographic overconcentration.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_budget_controller.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/budget_controller.py tests/test_budget_controller.py
git commit -m "feat(pipeline): implement hierarchical 2-tier adaptive budget and mask controller"
```

---

### Task 3: Calibrated Hard-Sample Mining (`calibrated_mining.py`)

**Files:**
- Create: `src/models/evaluators/calibrated_mining.py`
- Test: `tests/test_calibrated_mining.py`

**Interfaces:**
- Produces:
  - `class CalibratedHardMiner`:
    - `__init__(temperature: float = 1.0, decision_threshold: float = 0.5, uncertainty_window: float = 0.10)`
    - `calibrate_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor) -> float`
    - `mine_hard_samples(candidate_records: List[ImageMetadata], probabilities: List[float], embeddings: Optional[np.ndarray] = None) -> Tuple[List[ImageMetadata], Dict[str, Any]]`

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_calibrated_mining.py
import torch
import numpy as np
from src.data.image_schema import ImageMetadata
from src.models.evaluators.calibrated_mining import CalibratedHardMiner

def test_temperature_calibration():
    miner = CalibratedHardMiner()
    # Uncalibrated logits with large scale
    logits = torch.tensor([-5.0, -3.0, 3.0, 5.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    temp = miner.calibrate_temperature(logits, labels)
    assert temp > 0.1 and temp < 10.0

def test_dual_criterion_mining():
    miner = CalibratedHardMiner(decision_threshold=0.5, uncertainty_window=0.1)
    candidates = [
        ImageMetadata(image_id="f1", label=1, file_path="f1.png"),  # FN: P=0.2 (bypassed)
        ImageMetadata(image_id="f2", label=1, file_path="f2.png"),  # Uncertain: P=0.45
        ImageMetadata(image_id="f3", label=1, file_path="f3.png"),  # Easy fake: P=0.95
    ]
    probs = [0.20, 0.45, 0.95]
    hard_samples, stats = miner.mine_hard_samples(candidates, probs)
    
    assert len(hard_samples) == 2
    assert {h.image_id for h in hard_samples} == {"f1", "f2"}
    assert all(h.is_hard_sample for h in hard_samples)
    assert stats["false_negative_count"] == 1
    assert stats["uncertain_count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibrated_mining.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `CalibratedHardMiner` in `src/models/evaluators/calibrated_mining.py`**

- Fit temperature $T$ using NLL optimization on validation logits via PyTorch Adam/L-BFGS.
- Calibrated probability: $\hat{p} = \sigma(\text{logit} / T)$.
- Mining rule:
  - False Negative: $\hat{p} < \tau$ (model predicts Real on a Fake image).
  - Uncertain Sample: $|\hat{p} - \tau| \le \delta$ (e.g. $\delta = 0.10$, model is ambiguous).
- Perceptual diversity filter: If embeddings provided, remove near-duplicate candidates (cosine distance $< 0.05$).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibrated_mining.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/evaluators/calibrated_mining.py tests/test_calibrated_mining.py
git commit -m "feat(evaluators): implement temperature-calibrated dual-criterion hard sample miner"
```

---

### Task 4: Closed-Loop Active Learning Orchestrator (`run_closed_loop_active_learning.py`)

**Files:**
- Create: `src/pipelines/run_closed_loop_active_learning.py`
- Test: `tests/test_closed_loop_active_learning.py`

**Interfaces:**
- Produces:
  - `def execute_closed_loop_round(`
    `    round_idx: int,`
    `    detector_checkpoint: Path,`
    `    budget_controller: AdaptiveBudgetController,`
    `    prompt_engine: PromptGrammarEngine,`
    `    calibrated_miner: CalibratedHardMiner,`
    `    config_path: str,`
    `    output_dir: Path,`
    `) -> Dict[str, Any]`
  - `def run_active_learning_loop(`
    `    num_rounds: int = 2,`
    `    candidates_per_round: int = 150,`
    `    config_path: str = "configs/dataset_factory.yaml",`
    `) -> Dict[str, Any]`

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_closed_loop_active_learning.py
from pathlib import Path
from src.pipelines.run_closed_loop_active_learning import run_active_learning_loop

def test_closed_loop_single_round_dry_run(tmp_path):
    metrics = run_active_learning_loop(
        num_rounds=1,
        candidates_per_round=10,
        dry_run=True,
        output_base_dir=tmp_path / "active_learning",
    )
    assert "round_0" in metrics
    assert "test_metrics" in metrics["round_0"]
    assert "feedback_attribution" in metrics["round_0"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_closed_loop_active_learning.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `run_closed_loop_active_learning.py`**

Connect all components:
1. Load or train baseline detector $M_0$.
2. For round $r = 1 \dots R$:
   - Compute modality budget ($T2I, I2I, Inpaint$) via `budget_controller`.
   - Sample adaptive prompts via `prompt_engine`.
   - Generate candidate pool using `generate_deepfake_images.py`.
   - Run curation gate (Laplacian sharpness + SHA-256 deduplication).
   - Inference curated candidates using detector $M_{r-1}$.
   - Calibrate uncertainty and extract hard samples $H_r$ using `calibrated_miner`.
   - Extract error attribution (which prompt tags and mask regions fooled $M_{r-1}$).
   - Feed error attribution back into `prompt_engine.update_error_weights()` and `budget_controller`.
   - Expand dataset: $D_r = D_{r-1} \cup H_r$.
   - Retrain detector $M_r$ on $D_r$ using `train_deepfake_image_detector`.
   - Evaluate $M_r$ on the frozen test set `data/benchmarks/D0_baseline/test.jsonl`.
   - Save round checkpoint: `experiments/checkpoints/active_learning_m{r}_best.pt`.
3. Export comprehensive trajectory: `experiments/logs/closed_loop_metrics.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_closed_loop_active_learning.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/run_closed_loop_active_learning.py tests/test_closed_loop_active_learning.py
git commit -m "feat(pipeline): implement master closed-loop active learning orchestrator"
```

---

### Task 5: End-to-End Verification Run & Proof of Utility Report

**Files:**
- Output: `experiments/logs/closed_loop_metrics.json`
- Output: `experiments/predictions/active_learning_progression.png`
- Documentation: `docs/reports/DETECTOR_GUIDED_ACTIVE_LEARNING_REPORT.md`

- [ ] **Step 1: Execute full test suite**

Run: `pytest -q`
Expected: All tests pass (>= 115 tests).

- [ ] **Step 2: Run 2-round Active Learning Demonstration on CUDA GPU**

Run:
```bash
python3 -m src.pipelines.run_closed_loop_active_learning \
    --config configs/dataset_factory.yaml \
    --rounds 2 \
    --candidates-per-round 60 \
    --epochs 5 \
    --device cuda
```
Expected:
- Round 0 ($M_0$): Baseline on raw data.
- Round 1 ($M_1$): Retrained on $D_0 \cup H_1$ $\to$ Hard-Sample Recall increases, test AUC improves.
- Round 2 ($M_2$): Retrained on $D_1 \cup H_2$ $\to$ Lowest bypass rate, highest adversarial robustness.
- Verified test set: exactly 43 frozen samples.

- [ ] **Step 3: Generate visual progression report and plots**

Export `experiments/predictions/active_learning_progression.png` illustrating:
- Modality budget shift across rounds ($T2I \to Inpainting$ focus).
- AUC and Hard-Sample Recall growth curves.
- Sample gallery of hardest False Negatives caught by $M_2$.

- [ ] **Step 4: Commit and finalize**

```bash
git add experiments/logs/ docs/reports/DETECTOR_GUIDED_ACTIVE_LEARNING_REPORT.md
git commit -m "docs(report): complete closed-loop active learning proof of utility verification"
```
