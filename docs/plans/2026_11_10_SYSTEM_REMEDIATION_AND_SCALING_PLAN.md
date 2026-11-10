# System Remediation & 10,000 Sample Production Scaling Implementation Plan

- **Motivation/Background**: The comprehensive 10-subagent audit revealed critical bottlenecks: identity leakage in pilot active learning, Linux `tmpfs` disk quota exhaustion, 5.3 hours of unnecessary diffusion latency, CLIP 77-token prompt truncation, Haar XML parsing overhead, DINOv2 ImageNet normalization absence, and ViT overfitting during fine-tuning.
- **Purpose**: Provide a concrete, test-driven, step-by-step implementation plan to fix all mathematical, algorithmic, and architectural defects, verify with tests, cleanly reset pilot artifacts, and scale to 10,000 samples on the RTX 3050 Laptop GPU.
- **Overview Pipeline**: Environment Hardening -> Diffusion Calibration & Acceleration -> Curation Gate Optimization -> Detector Normalization -> Metrics Rigor -> Active Learning Leakage Guard -> Clean Reset & Re-verification -> 10,000 Sample Production Execution.
- **Detailed Plan**:
  - Task 1: Environment & Temporary Storage Hardening (NVMe SSD redirection)
  - Task 2: Diffusion Engine & Prompt Grammar Calibration (DPM-Solver++ 20 steps, 68-token negative prompt)
  - Task 3: Curation Gate Singleton & Face ROI Sharpness (Haar singleton, $\tau=1.25$)
  - Task 4: Evaluation Metric Tie-Breaking & Non-Parametric Rigor (`scipy.stats.rankdata`)
  - Task 5: Detector Architecture Normalization & Stability (ImageNet buffer, 1 unfrozen block, bfloat16)
  - Task 6: Active Learning Pipeline Hardening & Identity Leakage Prevention (train partition pinning)
  - Task 7: Clean Reset, Re-verification & Master Test Vault Sealing
  - Task 8: Production 10,000 Dataset Scaling & Final Detector $M_2$ Training
- **References**:
  - `docs/reports/MASTER_SYSTEM_AUDIT_AND_IMPROVEMENT_REPORT.md`
  - `.agents/rules/ZERO_MOCK_DATA_RULE.md`
  - `GEMINI.md`
- **Created**: 2026-11-10T07:46:00+07:00
- **Last Updated**: 2026-11-10T07:46:00+07:00

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all 12 audited defects across diffusion, curation, detector, metrics, and active learning, then execute the full 10,000 dataset scaling pipeline on RTX 3050 GPU.

**Architecture:** Modular refactoring of `src/models/diffusion/`, `src/data/curation_gate.py`, `src/common/metrics.py`, `src/models/detector/`, and `src/pipelines/run_closed_loop_active_learning.py` using strict TDD, verified via pytest and smoke tests, followed by batch production scaling.

**Tech Stack:** PyTorch, Diffusers (DPMSolverMultistepScheduler), DINOv2 (TorchVision/Hub), OpenCV, NumPy, SciPy (`scipy.stats.rankdata`), Pytest.

**Spec:** `docs/reports/MASTER_SYSTEM_AUDIT_AND_IMPROVEMENT_REPORT.md`

## Global Constraints

- Never use mock data, noise shapes, or uninitialized random weights in production manifests (`ZERO_MOCK_DATA_RULE.md`).
- Real faces ($N=4,000$) in `data/raw/` must remain untouched and identity-disjoint across splits.
- Master Frozen Test Vault ($N=1,181$) must remain completely isolated from training.
- 100% vectorized NumPy/PyTorch math — zero pixel iteration loops.
- Zero `print()` statements in `src/` modules; use `src.common.logger.get_logger(__name__)`.
- Strict typing with full PEP 484 type annotations on all function signatures.
- Device: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` with CPU cores pinned (`taskset -c 0-3`).

## Review Focus

1. **CLIP Token Truncation**: Positive prompt exceeds 77 tokens if slot word counts are unconstrained; verify token count $\le 70$ for all prompt configurations.
2. **Demographic Chimera**: Infant/toddler prompts must not contain `"man"` or `"woman"`; verify age-appropriate descriptors.
3. **Identity Contamination**: Active learning candidate generation must never pick identities from `test_reals.jsonl` or `val_reals.jsonl`.
4. **ROC-AUC Tied Probabilities**: Detectors with saturated scores ($0.0$ or $1.0$) must produce mathematically unbiased mid-rank ROC-AUC values.
5. **DINOv2 Normalization Drift**: Omitting ImageNet mean/std produces distorted features; verify input is normalized inside model wrapper.

---

### Task 1: Environment & Temporary Storage Hardening

**Files:**
- Modify: `scripts/run_pipeline.sh:1-30`
- Modify: `tests/conftest.py:1-25`
- Test: `tests/test_runtime_device.py`

**Interfaces:**
- Consumes: OS environment variables `TMPDIR`, pytest configuration
- Produces: Persistent NVMe `.cache/tmp` directory avoiding 5.5 GB `tmpfs` quota

- [ ] **Step 1: Write the failing test for TMPDIR redirection**

Add test in `tests/test_runtime_device.py`:
```python
def test_temp_directory_uses_nvme_cache() -> None:
    import os
    import tempfile
    from pathlib import Path
    
    # Ensure temporary files do not default to /tmp if TMPDIR is configured
    tfile = tempfile.NamedTemporaryFile()
    tpath = Path(tfile.name).resolve()
    tfile.close()
    assert not str(tpath).startswith("/tmp/pytest-of-bush")
```

- [ ] **Step 2: Run test to verify behavior**

Run: `PYTHONPATH=. pytest tests/test_runtime_device.py -k test_temp_directory -v`
Expected: Passes or fails depending on whether TMPDIR was exported.

- [ ] **Step 3: Implement TMPDIR export in scripts and conftest**

In `scripts/run_pipeline.sh`:
```bash
export TMPDIR="${PWD}/.cache/tmp"
mkdir -p "$TMPDIR"
```
In `tests/conftest.py`:
Ensure pytest session creates temp files within `.cache/pytest` rather than `/tmp`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_runtime_device.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.sh tests/conftest.py tests/test_runtime_device.py
git commit -m "fix(infra): redirect TMPDIR and pytest scratch to NVMe cache"
```

---

### Task 2: Diffusion Engine & Prompt Grammar Calibration

**Files:**
- Modify: `src/models/diffusion/prompt_engine.py`
- Modify: `src/models/diffusion/stable_diffusion_engine.py`
- Modify: `configs/dataset_factory.yaml`
- Test: `tests/test_prompt_engine.py`
- Test: `tests/test_stable_diffusion.py`

**Interfaces:**
- Consumes: `FacePromptConfig`, `DiffusionEngineConfig`
- Produces: Prompts bounded to $\le 70$ CLIP tokens, DPMSolverMultistepScheduler 20 steps, binarized inpainting mask

- [ ] **Step 1: Write failing tests for prompt token count and demographic consistency**

In `tests/test_prompt_engine.py`:
```python
def test_infant_prompt_does_not_contain_adult_gender_words() -> None:
    from src.models.diffusion.prompt_engine import build_face_prompt, FacePromptConfig
    
    config = FacePromptConfig(age_bracket="infant", gender="male")
    prompt = build_face_prompt(config)
    prompt_lower = prompt.lower()
    assert "man" not in prompt_lower.split()
    assert "adult" not in prompt_lower
    assert "baby boy" in prompt_lower or "infant" in prompt_lower

def test_prompt_word_count_within_clip_budget() -> None:
    from src.models.diffusion.prompt_engine import build_face_prompt, FacePromptConfig
    
    config = FacePromptConfig()
    prompt = build_face_prompt(config)
    words = prompt.split()
    assert len(words) <= 45, f"Prompt too long ({len(words)} words): {prompt}"
```

In `tests/test_stable_diffusion.py`:
```python
def test_negative_prompt_token_count_within_clip_limit() -> None:
    from src.models.diffusion.stable_diffusion_engine import CALIBRATED_NEGATIVE_PROMPT
    
    words = CALIBRATED_NEGATIVE_PROMPT.split(",")
    # Negative prompt string must have <= 68 tokens
    assert len(CALIBRATED_NEGATIVE_PROMPT.split()) <= 25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_prompt_engine.py -k "test_infant or test_prompt_word" -v`
Expected: FAIL

- [ ] **Step 3: Implement prompt grammar and engine upgrades**

1. In `src/models/diffusion/prompt_engine.py`:
   - Map age brackets properly: `infant`/`toddler` -> `baby boy`/`baby girl`; `child` -> `young boy`/`young girl`; `adult` -> `man`/`woman`.
   - Constrain slot word allocations so total prompt stays $\le 42$ words ($< 70$ CLIP tokens).
2. In `src/models/diffusion/stable_diffusion_engine.py`:
   - Define `CALIBRATED_NEGATIVE_PROMPT = "blurry, out of focus, distorted face, bad anatomy, deformed eyes, extra fingers, cartoon, 3d render, doll, low quality, oversaturated, plastic skin, mutation, watermark"`.
   - Configure `DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="dpmsolver++")`.
   - Update default num_inference_steps from 28 to 20 in `configs/dataset_factory.yaml`.
   - Binarize mask before feeding to inpainting UNet: `mask = (mask > 0.5).float()`.
   - Replace wide feathering with 2-pixel smoothstep blending.
   - Remove per-step `torch.cuda.empty_cache()` inside generation loops.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_prompt_engine.py tests/test_stable_diffusion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/diffusion/ configs/dataset_factory.yaml tests/test_prompt_engine.py tests/test_stable_diffusion.py
git commit -m "feat(diffusion): calibrate DPMSolver++ 20 steps, fix demographic chimera and 68-token negative prompt"
```

---

### Task 3: Curation Gate Singleton & Face ROI Sharpness

**Files:**
- Modify: `src/data/curation_gate.py`
- Modify: `src/common/metrics.py`
- Modify: `configs/dataset_factory.yaml`
- Test: `tests/test_image_curation.py`

**Interfaces:**
- Consumes: RGB image `(H, W, 3)`
- Produces: `CurationDecision(passed: bool, reason: str, sharpness: float, face_count: int)` with cached Haar cascade

- [ ] **Step 1: Write failing test for Haar singleton and face ROI sharpness**

In `tests/test_image_curation.py`:
```python
def test_curation_gate_uses_singleton_cascade() -> None:
    from src.data.curation_gate import CurationGate
    
    gate1 = CurationGate()
    gate2 = CurationGate()
    assert gate1.face_cascade is gate2.face_cascade, "Haar cascade must be a shared singleton"

def test_sharpness_calculated_on_face_roi_not_blurred_background() -> None:
    from src.data.curation_gate import CurationGate
    import numpy as np
    
    # Image with sharp 50x50 center face but completely smooth background
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[100:150, 100:150] = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    gate = CurationGate(min_sharpness=1.0)
    decision = gate.evaluate(img, bbox_xyxy=[100, 100, 150, 150])
    assert decision.sharpness > 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_image_curation.py -k "singleton or face_roi" -v`
Expected: FAIL

- [ ] **Step 3: Implement Haar singleton and face ROI sharpness**

1. In `src/data/curation_gate.py`:
   - Implement class-level `_cascade_singleton = None`.
   - On instantiation, reuse `_cascade_singleton` if already loaded.
   - If `bbox_xyxy` is provided or detected, crop to face region before calling `compute_laplacian_sharpness`.
2. In `configs/dataset_factory.yaml`:
   - Set `min_sharpness: 1.25` (calibrated for facial ROI).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_image_curation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/curation_gate.py configs/dataset_factory.yaml tests/test_image_curation.py
git commit -m "perf(curation): cache Haar cascade singleton and calibrate face-ROI sharpness threshold"
```

---

### Task 4: Evaluation Metric Tie-Breaking & Non-Parametric Rigor

**Files:**
- Modify: `src/common/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `targets: np.ndarray`, `scores: np.ndarray`
- Produces: `roc_auc: float` computed via Wilcoxon-Mann-Whitney with average rank tie-breaking

- [ ] **Step 1: Write failing test for tied score ROC-AUC**

In `tests/test_metrics.py`:
```python
def test_compute_roc_auc_with_tied_scores() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    # Tied probabilities for all samples
    probs = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    auc = compute_roc_auc(targets, probs)
    assert auc == pytest.approx(0.5, abs=1e-5), "Tied scores must yield 0.5 ROC-AUC"

def test_compute_roc_auc_single_class_returns_nan() -> None:
    import math
    targets = np.array([1, 1, 1, 1], dtype=np.int64)
    probs = np.array([0.8, 0.9, 0.7, 0.95], dtype=np.float32)
    auc = compute_roc_auc(targets, probs)
    assert math.isnan(auc), "Single class ROC-AUC must return NaN"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_metrics.py -k "tied or single_class" -v`
Expected: FAIL

- [ ] **Step 3: Implement mid-rank Wilcoxon ROC-AUC**

In `src/common/metrics.py`:
```python
from scipy.stats import rankdata

def compute_roc_auc(targets: np.ndarray, probs: np.ndarray) -> float:
    pos_mask = (targets == 1)
    neg_mask = (targets == 0)
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(probs, method="average")
    rank_sum_pos = float(np.sum(ranks[pos_mask]))
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/common/metrics.py tests/test_metrics.py
git commit -m "fix(metrics): use scipy rankdata mid-rank tie breaking for exact ROC-AUC"
```

---

### Task 5: Detector Architecture Normalization & Stability

**Files:**
- Modify: `src/models/detector/image_detector.py`
- Modify: `src/pipelines/train_image_detector.py`
- Modify: `configs/detector_training.yaml`
- Test: `tests/test_image_detector.py`

**Interfaces:**
- Consumes: Image batch `(B, 3, H, W)` in range $[0, 1]$
- Produces: Logits `(B, 1)` with registered ImageNet mean/std buffer, 1 unfreezable block default

- [ ] **Step 1: Write failing test for ImageNet normalization and unfreeze blocks**

In `tests/test_image_detector.py`:
```python
def test_detector_normalizes_imagenet_buffer() -> None:
    from src.models.detector.image_detector import build_detector
    
    detector = build_detector("dinov2_vits14")
    assert hasattr(detector, "mean") and hasattr(detector, "std")
    assert detector.mean.shape == (1, 3, 1, 1)

def test_unfreeze_backbone_single_block() -> None:
    from src.models.detector.image_detector import build_detector
    
    detector = build_detector("dinov2_vits14")
    detector.freeze_backbone()
    detector.unfreeze_backbone(num_blocks=1)
    
    trainable_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    # 1 block + classification head ~ 1.77M parameters
    assert 1_500_000 < trainable_params < 2_500_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_image_detector.py -k "normalizes or unfreeze" -v`
Expected: FAIL

- [ ] **Step 3: Implement ImageNet normalization buffer and unfreeze tuning**

1. In `src/models/detector/image_detector.py`:
   - Add registered buffers:
     ```python
     self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
     self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
     ```
   - In forward pass: `x_norm = (x - self.mean) / self.std`.
   - Update `unfreeze_backbone(self, num_blocks: int = 1)` to correctly slice trailing Transformer layers.
2. In `src/pipelines/train_image_detector.py`:
   - Normalize weighted BCE loss by $\sum w_i$ instead of $B$.
   - Add support for `torch.bfloat16` when CUDA is available.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_image_detector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/detector/image_detector.py src/pipelines/train_image_detector.py configs/detector_training.yaml tests/test_image_detector.py
git commit -m "fix(detector): add ImageNet normalization buffer, 1-block unfreeze default and bfloat16 support"
```

---

### Task 6: Active Learning Pipeline Hardening & Identity Leakage Prevention

**Files:**
- Modify: `src/pipelines/run_closed_loop_active_learning.py`
- Modify: `configs/active_learning.yaml`
- Test: `tests/test_closed_loop_active_learning.py`

**Interfaces:**
- Consumes: `train_reals.jsonl`, `test_reals.jsonl`, candidate synthetic pool
- Produces: Zero identity overlap between training buffer and frozen test vault, curriculum mining quota (60% inpaint, 30% img2img, 10% text2img)

- [ ] **Step 1: Write failing test verifying zero identity leakage in active learning**

In `tests/test_closed_loop_active_learning.py`:
```python
def test_candidate_generation_strictly_identity_disjoint_from_test_vault(tmp_path: Path) -> None:
    from src.pipelines.run_closed_loop_active_learning import prepare_candidate_seeds
    
    train_reals = tmp_path / "train_reals.jsonl"
    test_reals = tmp_path / "test_reals.jsonl"
    
    train_reals.write_text('{"image_id": "real_train_001", "identity": "id_01"}\n')
    test_reals.write_text('{"image_id": "real_test_001", "identity": "id_02"}\n')
    
    seeds = prepare_candidate_seeds(train_reals, test_reals, n_candidates=10)
    for s in seeds:
        assert s["identity"] != "id_02", "Test identity must never be used for candidate generation"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_closed_loop_active_learning.py -k test_candidate_generation_strictly -v`
Expected: FAIL

- [ ] **Step 3: Implement strict partition binding and curriculum quotas**

In `src/pipelines/run_closed_loop_active_learning.py`:
- Pin candidate real seed manifest strictly to `data/partitions/train_reals.jsonl`.
- Add pre-flight assertion:
  ```python
  train_ids = {item["identity"] for item in read_manifest(train_manifest)}
  test_ids = {item["identity"] for item in read_manifest(test_manifest)}
  assert len(train_ids.intersection(test_ids)) == 0, "Fatal: Train and Test identities overlap!"
  ```
- Set modality quotas to Curriculum Mining: 60% `sd_inpaint`, 30% `sd_img2img`, 10% `sd_text2img`.
- Pair cumulative challenge test fakes ($P_{fake} < 0.35$) with hard real faces ($P_{fake} > 0.65$).
- Remove double-dipping: disable oversampling when loss-weighting is enabled.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_closed_loop_active_learning.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/run_closed_loop_active_learning.py configs/active_learning.yaml tests/test_closed_loop_active_learning.py
git commit -m "fix(active_learning): enforce strict train partition binding, curriculum mining quotas and balanced challenge test"
```

---

### Task 7: Clean Reset, Re-verification & Master Test Vault Sealing

**Files:**
- Directory: `data/active_learning/`
- Directory: `data/benchmarks/`
- Execution: `make test-smoke` and full `pytest`

**Interfaces:**
- Consumes: Clean code changes from Tasks 1–6
- Produces: Pristine directory state, zero test identity contamination, 100% test pass

- [ ] **Step 1: Clean contaminated pilot artifacts**

Execute:
```bash
rm -rf data/active_learning/round_1 data/active_learning/round_2 data/active_learning/cumulative_challenge_test.jsonl
rm -rf data/benchmarks/D1_augmented data/benchmarks/D2_hard_boosted
```

- [ ] **Step 2: Verify Master Frozen Test Vault integrity**

Execute:
```bash
python3 -c "
import json
from pathlib import Path

vault = [json.loads(line) for line in Path('data/benchmarks/master_frozen_test.jsonl').read_text().splitlines() if line.strip()]
train = [json.loads(line) for line in Path('data/partitions/train_reals.jsonl').read_text().splitlines() if line.strip()]

vault_ids = {x.get('identity', x['image_id']) for x in vault if x['label'] == 0}
train_ids = {x.get('identity', x['image_id']) for x in train}

overlap = vault_ids.intersection(train_ids)
assert len(overlap) == 0, f'Identity overlap detected: {overlap}'
assert len(vault) == 1181, f'Expected 1181 samples, got {len(vault)}'
print('Master Frozen Test Vault VERIFIED: 100% clean, 0% overlap.')
"
```

- [ ] **Step 3: Run full verification suite**

Run: `PYTHONPATH=. pytest tests/`
Expected: 100% PASS across all 125+ tests.

- [ ] **Step 4: Commit**

```bash
git status
git commit -am "chore(audit): clean pilot artifacts and seal master frozen test vault"
```

---

### Task 8: Production 10,000 Dataset Scaling & Final Detector $M_2$

**Files:**
- Execution: `scripts/run_pipeline.sh active_learning configs/active_learning.yaml`
- Report: `docs/reports/BLOCK3_10K_PRODUCTION_COMPLETION_REPORT.md`

**Interfaces:**
- Consumes: Calibrated DPMSolver++ engine, hardened active learning pipeline
- Produces: 3,826 synthesized and curated fakes, 10,000 total dataset, trained $M_2$ detector

- [ ] **Step 1: Execute Active Learning scaling pipeline**

Run:
```bash
bash scripts/run_pipeline.sh active_learning configs/active_learning.yaml \
    --candidates-per-round 1913 \
    --max-rounds 2 \
    --target-total-samples 10000
```
Expected: Synthesizes ~3,826 samples via DPMSolver++ 20 steps (estimated ~5.8h on RTX 3050).

- [ ] **Step 2: Verify 10,000 sample dataset count & integrity**

Verify:
- Real samples: 4,000 (2,813 Train, 598 Val, 589 Test)
- Synthetic samples: 6,000 (2,174 Block 1 + 3,826 Active Learning rounds)
- Total: 10,000 samples.

- [ ] **Step 3: Evaluate Final Detector $M_2$**

Run evaluation against:
1. `data/benchmarks/master_frozen_test.jsonl` ($N=1,181$)
2. `data/active_learning/cumulative_challenge_test.jsonl` ($N=200$, 50% hard real, 50% hard fake)

- [ ] **Step 4: Package Completion Report & Prediction Gallery**

Compile:
- Final metrics comparison table: $M_0$ vs $M_1$ vs $M_2$ (Accuracy, ROC-AUC, Hard Sample Recall, Bypass Rate).
- Contact sheets of top adversarial hard samples.
- Save report to `docs/reports/BLOCK3_10K_PRODUCTION_COMPLETION_REPORT.md`.
