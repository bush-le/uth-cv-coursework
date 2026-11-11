# PLAN_REMEDIATE_ALL_BUGS_AND_BIAS.md — Master Remediation Plan for All Systemic Bugs & Evaluation Bias

- **Motivation/Background**: An exhaustive code and empirical audit revealed evaluation leakage (Youden's J tuned on test set), hard negative mining failure (empty $H_1$ due to static threshold on uncalibrated logits), demographic schema misalignment (CelebA-HQ boolean attributes misclassified as unknown), and PyTorch CPU pin_memory deprecation warnings.
- **Purpose**: Provide a step-by-step, test-driven engineering plan to eliminate all bugs, eliminate evaluation bias, enforce strict validation-based calibration, and produce scientifically sound coursework deliverables.
- **Overview Pipeline**: Fix Demographic Schema $\to$ Fix Dynamic Quantile Mining ($H_1, H_2 \neq \emptyset$) $\to$ Fix Validation Calibration ($\tau^*_{\text{val}}$ peeking prevention) $\to$ Fix DINOv2 Unfreezing & Logit Scaling $\to$ Fix CPU Pin Memory & Split Floor $\to$ Execute End-to-End Verification Pipeline.
- **Detailed Plan**: §1 Root Cause & Target Architecture; §2 Global Invariants & Review Focus; §3 Six Executable TDD Tasks; §4 Definition of Done & Success Criteria.
- **References**: [GEMINI.md](GEMINI.md), [ITERATIVE_PROOF_OF_UTILITY.md](experiments/reports/ITERATIVE_PROOF_OF_UTILITY.md), [evaluate_iterative_benchmarks.py](scripts/evaluate_iterative_benchmarks.py), [mine_hard_samples.py](src/pipelines/mine_hard_samples.py).
- **Created**: 2026-11-09T17:56:00+07:00
- **Last Updated**: 2026-11-09T17:56:00+07:00

---

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all 6 identified systemic bugs in data schemas, mining logic, logit scaling, and evaluation calibration to achieve 100% rigorous, leak-free, reproducible scientific benchmarks.

**Architecture:** 
- Ingestion & Splits layer normalizes demographic dictionaries across heterogeneous datasets (CelebA-HQ & FFHQ).
- Mining layer shifts from static thresholding (`p < 0.50`) to adaptive quantile and top-$k$ hardest sample extraction to guarantee non-empty hard manifests.
- Evaluation layer strictly decouples calibration (fit on $D_{\text{val}}$) from test inference (freeze $\tau^*$, apply blindly to $D_{\text{test}}$), reporting both uncalibrated ($\tau=0.50$) and val-calibrated metrics.
- DataLoader layer checks `torch.cuda.is_available()` before pinning memory.

**Tech Stack:** Python 3.14, PyTorch 2.x, DINOv2 (`facebook/dinov2-small`), Diffusers (Stable Diffusion v1-5), NumPy, pytest.

---

## 1. Global Constraints

- **Zero Test Leakage**: Never calculate, optimize, or select decision thresholds or temperature parameters using $D_{\text{test}}$. All calibration MUST happen on $D_{\text{val}}$.
- **Zero Mock / Neural Integrity**: Checkpoints on disk must be real trained weights; mining must use real model inference.
- **Strict Typing**: All added functions must include PEP 484 type annotations and docstrings with shape contracts.
- **No Print**: Never use `print()` in `src/`; always use `src.common.logger.get_logger(__name__)`.
- **CPU & GPU Safety**: DataLoader `pin_memory` must only be True when CUDA is available. Pinned to CPU cores `0-5`.

---

## 2. Review Focus (Failure Modes & Edge Cases)

1. **Empty Hard Manifest ($H = \emptyset$)**: If all candidates in a pool yield $P(\text{Fake}) \ge \tau$, fallback to top-$k$ hardest (lowest probability) candidates so evolutionary stages never train on empty sets.
2. **Degenerate Validation Set ($N_{\text{val}} < 2$)**: Enforce minimum validation sample floor ($N_{\text{val}} \ge 2$) so AUC and Youden's J are never computed on a single point.
3. **CelebA-HQ Schema Mismatch**: When metadata dictionary contains `{"male": bool, "young": bool, "eyeglasses": bool}`, map deterministically to `gender` ("male"/"female"), `age_group` ("young"/"mature"), and `glasses` ("Glasses"/"None").
4. **Logit Saturation / Uniform Scores**: When logit margin is $< 0.05$, calibration must not divide by zero (clamp temperature $T \ge 10^{-3}$).
5. **Double Evaluation Table**: The final benchmark table must explicitly display both **Standard Threshold ($\tau = 0.50$)** and **Val-Calibrated Threshold ($\tau^*_{\text{val}}$)** so reviewers see the true baseline versus calibrated gains transparently.

---

## 3. Tasks

### Task 1: Demographic Schema Unification (CelebA-HQ vs FFHQ)

**Files:**
- Modify: `src/data/splits.py:205-245`
- Modify: `scripts/create_coursework_notebooks.py:78-98`
- Test: `tests/test_splits.py`

**Interfaces:**
- Produces: `normalize_demographics(demo: Optional[Dict[str, Any]]) -> Dict[str, Any]`

- [ ] **Step 1: Write failing test in `tests/test_splits.py`**
```python
def test_normalize_demographics_celeba_and_ffhq():
    from src.data.splits import normalize_demographics
    
    celeba_demo = {"dataset": "celeba_hq", "male": False, "young": True, "eyeglasses": False}
    ffhq_demo = {"dataset": "ffhq", "gender": "male", "age_group": "30-39", "glasses": "None"}
    
    norm_celeba = normalize_demographics(celeba_demo)
    assert norm_celeba["gender"] == "female"
    assert norm_celeba["age_group"] == "young"
    assert norm_celeba["glasses"] == "None"
    
    norm_ffhq = normalize_demographics(ffhq_demo)
    assert norm_ffhq["gender"] == "male"
    assert norm_ffhq["age_group"] == "30-39"
    assert norm_ffhq["glasses"] == "None"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_splits.py::test_normalize_demographics_celeba_and_ffhq -v`
Expected: FAIL (ImportError / AttributeError)

- [ ] **Step 3: Implement `normalize_demographics` in `src/data/splits.py` and integrate into `compute_dataset_stats`**
Map boolean attributes from CelebA-HQ to standard unified string values and use in aggregation loop.

- [ ] **Step 4: Update `scripts/create_coursework_notebooks.py` to use unified demographic mapping**

- [ ] **Step 5: Run tests and re-generate `data/dataset_statistics.json`**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_splits.py -v`
Run: `taskset -c 0-5 .venv/bin/python3 -c "from src.data.splits import compute_dataset_stats; from src.common.config import load_config; from pathlib import Path; compute_dataset_stats(load_config(Path('configs/dataset_factory.yaml')))"`
Expected: PASS, and `unknown` in `dataset_statistics.json` drops from 2025 to 0.

- [ ] **Step 6: Commit**
```bash
git add src/data/splits.py scripts/create_coursework_notebooks.py tests/test_splits.py data/dataset_statistics.json
git commit -m "fix(data): normalize demographics across CelebA-HQ and FFHQ schemas"
```

---

### Task 2: Adaptive Quantile & Top-k Hard Negative Mining ($H_1 \neq \emptyset$)

**Files:**
- Modify: `src/pipelines/mine_hard_samples.py:440-505`
- Modify: `scripts/run_iterative_auto_pipeline.py:195-215`
- Test: `tests/test_hardness.py`

**Interfaces:**
- Consumes: Model predictions on synthetic candidate pool.
- Produces: Guaranteed non-empty hard manifest ordered by hardness with fallback to top-$k$ lowest fake probabilities.

- [ ] **Step 1: Write failing test in `tests/test_hardness.py`**
```python
def test_mine_hard_samples_guarantees_non_empty_when_candidates_exist():
    from src.pipelines.mine_hard_samples import mine_hard_samples
    from src.data.image_schema import ImageMetadata
    import torch
    import torch.nn as nn
    
    # Model that outputs high fake probability ~0.55 for all fakes
    class StubDetector(nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1), 0.2)  # sigmoid(0.2) = 0.55 > 0.50
            
    metas = [
        ImageMetadata(image_id=f"syn_{i}", label=1, file_path="dummy.png")
        for i in range(4)
    ]
    # At static tau=0.50, normally 0 pass. With fallback top_k=2, exactly 2 hardest must be mined.
    report = mine_hard_samples(
        model=StubDetector(),
        synthetic_metadata=metas,
        cache={"dummy.png": torch.zeros(3, 32, 32)},
        bypass_threshold=0.50,
        min_hard_samples=2,
    )
    assert report.bypassed_hard_count == 2
    assert len(report.hard_sample_ids) == 2
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_hardness.py::test_mine_hard_samples_guarantees_non_empty_when_candidates_exist -v`
Expected: FAIL (`unexpected keyword argument 'min_hard_samples'` or assertion failure).

- [ ] **Step 3: Implement `min_hard_samples` and relative quantile mining in `src/pipelines/mine_hard_samples.py`**
If standard threshold filtering produces fewer than `min_hard_samples`, rank candidates by $P(\text{Fake})$ ascending and select the top `min_hard_samples` hardest candidates.

- [ ] **Step 4: Update `scripts/run_iterative_auto_pipeline.py` to pass `min_hard_samples=max(1, len(pool1_fakes) // 2)`**

- [ ] **Step 5: Run tests**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_hardness.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add src/pipelines/mine_hard_samples.py scripts/run_iterative_auto_pipeline.py tests/test_hardness.py
git commit -m "fix(mining): implement adaptive quantile fallback to guarantee non-empty hard negative pools"
```

---

### Task 3: Decouple Calibration from Test Set (Eliminate Test Peeking)

**Files:**
- Modify: `scripts/evaluate_iterative_benchmarks.py:64-150`
- Modify: `src/pipelines/evaluate_image_benchmark.py:20-60`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `val_manifest_path` (optional) and `test_manifest_path`.
- Produces: Dual benchmark evaluation: Uncalibrated ($\tau = 0.50$) and Val-Calibrated ($\tau^*_{\text{val}}$ applied to Test).

- [ ] **Step 1: Write failing test in `tests/test_calibration.py`**
```python
def test_calibrate_threshold_on_val_and_apply_to_test_without_peeking():
    from scripts.evaluate_iterative_benchmarks import calibrate_and_evaluate_model
    import numpy as np
    
    val_probs = np.array([0.48, 0.49, 0.52, 0.53])
    val_labels = np.array([0, 0, 1, 1])
    test_probs = np.array([0.47, 0.49, 0.51, 0.54])
    test_labels = np.array([0, 0, 1, 1])
    
    results = calibrate_and_evaluate_model(val_probs, val_labels, test_probs, test_labels)
    assert "standard_tau_0_5" in results
    assert "val_calibrated" in results
    # Calibrated tau must come strictly from val_probs, never test_probs
    assert 0.50 <= results["val_calibrated"]["operating_threshold"] <= 0.52
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_calibration.py::test_calibrate_threshold_on_val_and_apply_to_test_without_peeking -v`
Expected: FAIL.

- [ ] **Step 3: Implement `calibrate_and_evaluate_model` and update `evaluate_model_on_manifest`**
In `scripts/evaluate_iterative_benchmarks.py`:
- Remove test-peeking Youden's J loop from `evaluate_model_on_manifest`.
- Always compute raw metrics at standard $\tau = 0.50$.
- When a validation set is provided, compute $\tau^* = \arg\max_{\tau \in \text{val}} (\text{TPR}_{\text{val}} - \text{FPR}_{\text{val}})$, then evaluate test set using that fixed $\tau^*$.
- Report both standard and calibrated results in output markdown tables.

- [ ] **Step 4: Run tests**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_calibration.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/evaluate_iterative_benchmarks.py tests/test_calibration.py
git commit -m "fix(eval): strictly decouple threshold calibration to validation set and eliminate test peeking"
```

---

### Task 4: Fix DINOv2 Logit Scaling & Top Transformer Block Unfreezing

**Files:**
- Modify: `src/pipelines/train_image_detector.py:60-95`
- Modify: `src/models/detector/image_detector.py:45-75`
- Test: `tests/test_image_detector.py`

**Interfaces:**
- Consumes: Configurable `unfreeze_last_n_blocks` parameter.
- Produces: Gradient flow through top ViT transformer block to allow confident logit scaling.

- [ ] **Step 1: Write test in `tests/test_image_detector.py` verifying top block parameters have gradients**
```python
def test_unfreeze_last_n_blocks_enables_gradients():
    from src.models.detector.image_detector import DeepfakeImageDetector
    model = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)
    model.unfreeze_backbone(unfreeze_last_n_blocks=1)
    
    trainable_named = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any("blocks.11" in name or "norm" in name for name in trainable_named)
    assert any("head" in name for name in trainable_named)
```

- [ ] **Step 2: Run test to verify behavior**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_image_detector.py::test_unfreeze_last_n_blocks_enables_gradients -v`

- [ ] **Step 3: Ensure `train_image_detector.py` unfreezes the top block when requested and scales learning rates appropriately**
Set `unfreeze_last_n_blocks=1` with backbone lr $10^{-5}$ and head lr $10^{-4}$.

- [ ] **Step 4: Run tests**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_image_detector.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/pipelines/train_image_detector.py src/models/detector/image_detector.py tests/test_image_detector.py
git commit -m "feat(model): support fine-tuning top DINOv2 block for robust logit margin learning"
```

---

### Task 5: CPU Pin Memory Deprecation Fix & Minimum Validation Split Floor

**Files:**
- Modify: `src/data/datamodule.py:32`
- Modify: `src/data/image_datamodule.py:110-130`
- Test: `tests/test_image_data.py`

**Interfaces:**
- Consumes: PyTorch DataLoader config.
- Produces: Deprecation-free DataLoader and non-degenerate validation splits ($N_{\text{val}} \ge 2$).

- [ ] **Step 1: Write test in `tests/test_image_data.py` verifying split floor**
```python
def test_image_datamodule_enforces_minimum_val_samples():
    from src.data.image_datamodule import DeepfakeImageDataModule
    from src.data.image_schema import ImageMetadata
    
    metas = [ImageMetadata(image_id=f"id_{i}", label=i % 2, file_path="dummy.png") for i in range(10)]
    dm = DeepfakeImageDataModule(metadata_list=metas, train_val_test_split=(0.8, 0.1, 0.1), batch_size=4)
    dm.setup()
    
    assert len(dm.val_dataset) >= 2
    assert len(dm.test_dataset) >= 2
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_image_data.py::test_image_datamodule_enforces_minimum_val_samples -v`

- [ ] **Step 3: Implement `pin_memory = pin_memory and torch.cuda.is_available()` in `src/data/datamodule.py` and enforce sample floor in `src/data/image_datamodule.py`**

- [ ] **Step 4: Run tests**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_image_data.py tests/test_data.py -v`
Expected: PASS with 0 deprecation warnings.

- [ ] **Step 5: Commit**
```bash
git add src/data/datamodule.py src/data/image_datamodule.py tests/test_image_data.py
git commit -m "fix(data): guard cpu pin_memory against deprecation warnings and enforce minimum val split floor"
```

---

### Task 6: Execute Full Re-run of Evolutionary Loop & Generate Real Benchmark Report

**Files:**
- Modify: `scripts/run_iterative_auto_pipeline.py`
- Modify: `experiments/reports/ITERATIVE_PROOF_OF_UTILITY.md`
- Test: Full pipeline execution on GPU

- [ ] **Step 1: Run full unit test suite**
Run: `taskset -c 0-5 .venv/bin/pytest tests/ -v`
Expected: 100% tests pass (104+ tests, 0 warnings).

- [ ] **Step 2: Execute updated `scripts/run_iterative_auto_pipeline.py` on GPU**
Run: `PYTHONPATH=. taskset -c 0-5 .venv/bin/python3 scripts/run_iterative_auto_pipeline.py --samples-per-round 8 --steps 15 --epochs 4`
Verify:
- `h1_manifest.jsonl` contains $\ge 2$ valid hard negative samples ($|H_1| > 0$).
- `h2_manifest.jsonl` contains $\ge 2$ valid hard negative samples ($|H_2| > 0$).
- $M_1$ and $M_2$ actually receive and train on distinct hard pools.
- Benchmark reports both standard $\tau=0.50$ and val-calibrated performance.

- [ ] **Step 3: Verify and update `experiments/reports/ITERATIVE_PROOF_OF_UTILITY.md`**
Verify the report presents honest, scientifically defensible numbers (realistic accuracy in $70-90\%$ range, clear contrast between $M_0$, $M_{\text{rand}}$, $M_1$, $M_2$, without artificial 100% inflation).

- [ ] **Step 4: Commit & Push to Git**
```bash
git add scripts/ data/iterative_loop/ experiments/reports/
git commit -m "feat(benchmark): complete leak-free multi-generation loop with validation-calibrated metrics"
git push origin baseline/initial-baseline
```

---

## 4. Definition of Done & Success Criteria

1. **Zero Inflation**: Test Accuracy and AUC are not artificially forced to 100% via test peeking; standard $\tau=0.50$ and val-calibrated $\tau^*$ are reported side-by-side.
2. **Guaranteed Hard Mining**: $H_1$ and $H_2$ manifests are guaranteed non-empty ($|H_1| > 0, |H_2| > 0$).
3. **Unified Demographics**: `data/dataset_statistics.json` contains 0 `"unknown"` entries for CelebA-HQ faces.
4. **Clean Execution**: 0 PyTorch deprecation warnings during testing and inference.
5. **Full Test Suite**: All unit tests in `tests/` pass cleanly in $< 15$ seconds.
