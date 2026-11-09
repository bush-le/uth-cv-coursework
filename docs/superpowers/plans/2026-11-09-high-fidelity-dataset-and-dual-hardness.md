# High-Fidelity Deepfake Dataset & Proof-of-Utility Benchmark Plan
## (100% Zero Data Leakage Guaranteed via Identity-Disjoint Protocol)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Native execution chosen by user) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-fidelity synthetic facial dataset using Stable Diffusion that eliminates anatomical deformities (fooling human eyes), and execute a scientifically rigorous 3-Way Controlled Proof-of-Utility Benchmark ($D_0$ Baseline vs. $D_1 = D_0 + R$ Random Augmentation vs. $D_2 = D_0 + H_1$ Hard-Sample Augmentation) evaluated on a strictly **Identity-Disjoint Unseen Test Set** ($D_{\text{test}}$) with **100% Zero Data Leakage**.

**Architecture:**
1. **Zero-Leakage Identity-Disjoint Partitioning**: Partition the original real subjects at origin into $\text{Train\_IDs}$ ($80\%$) and $\text{Test\_IDs}$ ($20\%$) with $\text{Train\_IDs} \cap \text{Test\_IDs} = \emptyset$. All training, mining, and random augmentations are strictly generated from $\text{Train\_IDs}$. The independent evaluation set $D_{\text{test}}$ (Real + Fake) is generated solely from $\text{Test\_IDs}$.
2. **High-Fidelity Diffusion Engine**: Native $512 \times 512$ generation downsampled via Lanczos to $256 \times 256$, guided by dynamic elliptical jaw-expanded ($+18\%$) face masks with $r=16\text{px}$ Gaussian feathering to prevent duplicate mouths and boundary seams.
3. **Multi-Tier Curation Gate**: Biometric facial aspect ratio, secondary mouth/lip continuity detection, and Sobel seam gradient verification to eliminate visual anomalies before curation.
4. **Supervised Detector Bypass Mining ($H_1$)**: Define hard samples as false negatives of a trained baseline detector $M_0$: $H_1 = \{x \in \text{Synthetic Fakes} : P_{M_0}(\text{Fake} \mid x) < 0.50\}$, paired with an equal-sized random subset $R$ ($|R| = |H_1|$).
5. **3-Way Controlled Proof-of-Utility**: Train $M_0$ (on $D_0$), $M_{\text{rand}}$ (on $D_1$), and $M_{\text{hard}}$ (on $D_2$), then evaluate all three on the frozen, untouched $D_{\text{test}}$ to prove whether hard samples provide superior discriminative utility.

**Tech Stack:** PyTorch, Diffusers (Stable Diffusion v1.5), Meta DINOv2 (`facebook/dinov2-small`), OpenCV, torchvision, PIL, NumPy, Scikit-learn, Pytest.

**Spec:** `docs/PURPOSE.md`, `.agents/rules/ZERO_MOCK_DATA_RULE.md`, `configs/dataset_factory.yaml`.

---

## Global Constraints & 100% Anti-Leakage Invariants

1. **Strict Identity-Disjoint Partitioning (Zero-Leakage Invariant)**:
   - Every subject identity belongs to either $\text{Train\_IDs}$ or $\text{Test\_IDs}$.
   - **NO identity overlap**: $\text{Train\_IDs} \cap \text{Test\_IDs} = \emptyset$.
   - **No background overlap**: All Real test images come from $\text{Test\_IDs}$; all Fake test images are synthesized solely from $\text{Test\_IDs}$. The detector $M$ is NEVER exposed to any image or identity in $\text{Test\_IDs}$ during training or mining.
2. **Zero-Mock & Neural Checkpoint Integrity**:
   - Mining $H_1$ requires a verified trained checkpoint (`experiments/checkpoints/baseline_detector_best.pt`). Zero unverified random-weight or mock mining.
3. **Controlled Coursework Hardware Scale**:
   - Total dataset scale: 120–160 samples (60 Real Train, 60 Fake Train [30 initial + 30 mining pool], 20 Real Test, 20 Fake Test).
   - Generates in ~10 minutes on RTX 3050 GPU (4GB VRAM) and trains in ~2 minutes without risk of thermal throttling.
4. **AMD Ryzen CPU Affinity**:
   - Always run Python processes with `taskset -c 0-5` to avoid non-monotonic clock assertion errors (`ApproximateClock`).
5. **VRAM Hygiene**:
   - Maintain single-pipeline lifecycle in Stable Diffusion engine (`_switch_mode`) with `gc.collect()` and `torch.cuda.empty_cache()` to guarantee zero OOM.

---

## Review Focus

1. **Identity Contamination**: Verify that no subject identity present in `test.jsonl` appears in `train.jsonl` or `val.jsonl`.
   - *Test*: `test_identity_disjoint_strict_partition` in `tests/test_splits.py`.
2. **Double-Mouth Inpainting Anomaly**: Fixed inpainting boxes leaving lower open mouths unmasked.
   - *Test*: `test_dynamic_oval_mask_covers_full_jaw` in `tests/test_dynamic_masking.py`.
3. **Anatomical & Seam Defects**: Curation gate detecting horizontal seam spikes and abnormal aspect ratios.
   - *Test*: `test_curation_rejects_dual_mouth_anomaly` in `tests/test_image_curation.py`.
4. **Mining Guardrail & Checkpoint Verification**: Mining CLI rejecting untrained weights.
   - *Test*: `test_mine_hard_samples_enforces_checkpoint_guardrail` in `tests/test_pipelines.py`.
5. **Proof-of-Utility Benchmark Consistency**: Identical test set evaluation across $D_0, D_1, D_2$.
   - *Test*: `test_benchmark_stages_share_exact_test_set` in `tests/test_splits.py`.

---

## Task Decomposition

### Task 1: Strict Identity-Disjoint Partitioning & Manifest Generation

**Files:**
- Modify: `configs/dataset_factory.yaml`
- Modify: `scripts/organize_topic_dataset.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `data/raw/manifest.jsonl` (4000 real faces).
- Produces: Partitioned manifests:
  - `data/partitions/train_identities.jsonl` (60 real subjects).
  - `data/partitions/test_identities.jsonl` (20 real subjects, strictly disjoint).
- Invariant: `set(train_ids).isdisjoint(set(test_ids)) == True`.

- [ ] **Step 1: Write test verifying strict identity disjointness**
```python
def test_identity_disjoint_strict_partition():
    from src.data.splits import assign_identity_splits
    from src.data.image_schema import ImageMetadata
    recs = [
        ImageMetadata(image_id=f"r_{i}", label=0, source_real_id=f"person_{i}")
        for i in range(20)
    ]
    assignment = assign_identity_splits(recs, ratios={"train": 0.8, "val": 0.0, "test": 0.2}, seed=42)
    train_ids = {k for k, v in assignment.items() if v == "train"}
    test_ids = {k for k, v in assignment.items() if v == "test"}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) > 0 and len(test_ids) > 0
```

- [ ] **Step 2: Run test to verify it passes**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_splits.py -k test_identity_disjoint_strict_partition -v`
Expected: PASS

- [ ] **Step 3: Configure dataset factory paths and partition counts in `configs/dataset_factory.yaml`**
Update `configs/dataset_factory.yaml` with explicit partition sizes:
- `train_real_count: 60`
- `test_real_count: 20`
- `modes: ["sd_img2img", "sd_inpaint"]`

- [ ] **Step 4: Commit**
```bash
git add configs/dataset_factory.yaml tests/test_splits.py
git commit -m "feat(splits): configure strict identity-disjoint partition limits and tests"
```

---

### Task 2: High-Fidelity Diffusion Generation for Train Pool & Test Pool

**Files:**
- Modify: `src/pipelines/generate_deepfake_images.py`
- Script: `scripts/generate_identity_disjoint_data.py`
- Output:
  - `data/synthetic/train_pool/` (60 synthetic faces generated from `Train_IDs`: 30 baseline + 30 mining pool).
  - `data/synthetic/test_pool/` (20 synthetic faces generated strictly from `Test_IDs`).

**Interfaces:**
- Consumes: Real images from partitioned manifests.
- Produces: Synthesized images with metadata records containing `source_real_id`.
- Visual Quality: Native $512 \times 512$ with Lanczos downsampling + dynamic oval mask (covering jawline $+18\%$, $r=16\text{px}$ feathering).

- [ ] **Step 1: Verify dynamic oval mask inpainting test passes**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_dynamic_masking.py -v`
Expected: PASS

- [ ] **Step 2: Run High-Fidelity Generation for Train Pool (60 faces) and Test Pool (20 faces) on CUDA**
Run:
```bash
taskset -c 0-5 .venv/bin/python3 -m src.pipelines.generate_deepfake_images --config configs/dataset_factory.yaml
```
Verify generated faces have zero double-mouth artifacts and no boundary seams.

- [ ] **Step 3: Run curation gate on all generated fakes**
Run:
```bash
taskset -c 0-5 .venv/bin/python3 -m src.pipelines.curate_image_dataset --config configs/dataset_factory.yaml
```
Verify all deformities (aspect ratio, seam spikes) are caught and rejected.

- [ ] **Step 4: Commit**
```bash
git add data/synthetic/ configs/dataset_factory.yaml
git commit -m "feat(data): high-fidelity synthetic faces generated and curated with dynamic oval masking"
```

---

### Task 3: Train Baseline Detector $M_0$ on $D_0$

**Files:**
- Config: `configs/train_baseline.yaml`
- Pipeline: `src/pipelines/train.py`
- Output: `experiments/checkpoints/baseline_detector_best.pt`

**Interfaces:**
- Consumes: $D_0$ Training Split (30 Real Train + 30 Initial Synthetic Train from `Train_IDs`).
- Produces: Trained checkpoint $M_0$ evaluated on Val split.

- [ ] **Step 1: Execute training pipeline for Baseline Detector $M_0$**
Run:
```bash
taskset -c 0-5 .venv/bin/python3 -m src.pipelines.train --config configs/dataset_factory.yaml
```
Verify checkpoint `experiments/checkpoints/baseline_detector_best.pt` is generated with validation AUC $> 0.85$.

- [ ] **Step 2: Verify checkpoint integrity via Pytest**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_pipelines.py -k test_mine_hard_samples_cli_loads_checkpoint -v`
Expected: PASS

- [ ] **Step 3: Commit**
```bash
git add experiments/checkpoints/ configs/
git commit -m "feat(model): trained baseline detector M0 on D0 achieving high validation AUC"
```

---

### Task 4: Supervised Hard-Sample Mining ($H_1$) and Random Mining ($R$)

**Files:**
- Pipeline: `src/pipelines/mine_hard_samples.py`
- Script: `scripts/sample_random_augmentation.py`
- Output:
  - `data/hard_samples/hard_samples_manifest.jsonl` ($H_1$: Fake images bypassing $M_0$, $P_{M_0}(\text{Fake}) < 0.50$).
  - `data/synthetic/random_augmented_manifest.jsonl` ($R$: Randomly selected fakes from Train mining pool, $|R| = |H_1|$).

**Interfaces:**
- Consumes: 30 new synthetic fakes from `Train_IDs` mining pool + $M_0$ checkpoint.
- Produces: Mined hard samples ($H_1$) where $P(\text{Fake}) < 0.50$, and unbiased random control subset $R$.

- [ ] **Step 1: Mine hard samples using trained detector $M_0$**
Run:
```bash
taskset -c 0-5 .venv/bin/python3 -m src.pipelines.mine_hard_samples \
  --config configs/dataset_factory.yaml \
  --method detector \
  --checkpoint experiments/checkpoints/baseline_detector_best.pt \
  --threshold 0.50
```
Verify report output: total fakes evaluated, rejected count, bypassed (hard) count ($H_1$).

- [ ] **Step 2: Create matching random control subset $R$ with $|R| = |H_1|$**
Run random sampling script from the same candidate pool to guarantee size parity $|R| = |H_1|$.

- [ ] **Step 3: Commit**
```bash
git add data/hard_samples/ data/synthetic/
git commit -m "feat(mining): mined hard samples H1 via M0 bypass and created balanced random control R"
```

---

### Task 5: Package Benchmark Stages ($D_0, D_1 = D_0 + R, D_2 = D_0 + H_1$)

**Files:**
- Script: `scripts/organize_topic_dataset.py`
- Output:
  - `data/benchmarks/D0_baseline/` (`train.jsonl`, `val.jsonl`, `test.jsonl`)
  - `data/benchmarks/D1_augmented/` (`train.jsonl`, `val.jsonl`, `test.jsonl`)
  - `data/benchmarks/D2_hard_boosted/` (`train.jsonl`, `val.jsonl`, `test.jsonl`)

**Interfaces:**
- Guarantee: `test.jsonl` is IDENTICAL across $D_0, D_1, D_2$ and contains strictly unseen `Test_IDs`.
- Invariant: Zero identity overlap between any training set and `test.jsonl`.

- [ ] **Step 1: Execute dataset reorganization & benchmark packaging**
Run:
```bash
PYTHONPATH=. taskset -c 0-5 .venv/bin/python3 scripts/organize_topic_dataset.py --config configs/dataset_factory.yaml
```
Verify logs show identical test set sizes and 0% identity leakage across all 3 stages.

- [ ] **Step 2: Run verification test on benchmark splits**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_splits.py -v`
Expected: PASS (all identity-disjoint checks pass).

- [ ] **Step 3: Commit**
```bash
git add data/benchmarks/ data/dataset_statistics.json
git commit -m "feat(benchmarks): packaged D0, D1, D2 benchmarks with identical frozen unseen test split"
```

---

### Task 6: Train Augmented Detectors $M_{\text{rand}}$ (on $D_1$) and $M_{\text{hard}}$ (on $D_2$)

**Files:**
- Pipeline: `src/pipelines/train.py`
- Output:
  - `experiments/checkpoints/detector_D1_random_best.pt` ($M_{\text{rand}}$)
  - `experiments/checkpoints/detector_D2_hard_best.pt` ($M_{\text{hard}}$)

- [ ] **Step 1: Train $M_{\text{rand}}$ on $D_1$**
Run training pipeline using `data/benchmarks/D1_augmented/train.jsonl`.

- [ ] **Step 2: Train $M_{\text{hard}}$ on $D_2$**
Run training pipeline using `data/benchmarks/D2_hard_boosted/train.jsonl`.

- [ ] **Step 3: Commit**
```bash
git add experiments/checkpoints/
git commit -m "feat(models): trained M_rand on D1 and M_hard on D2"
```

---

### Task 7: 3-Way Controlled Proof-of-Utility Benchmark & Coursework Golden Table

**Files:**
- Script: `scripts/evaluate_benchmarks.py`
- Script: `scripts/build_visual_gallery.py`
- Output:
  - `experiments/reports/PROOF_OF_UTILITY_BENCHMARK.md` (Golden Table comparing $M_0$ vs $M_{\text{rand}}$ vs $M_{\text{hard}}$ on $D_{\text{test}}$).
  - `experiments/predictions/real_vs_synthetic_gallery.png` (Visual gallery showing pristine synthetic faces, inpainting overlays, and detector predictions).

**Metrics Evaluated on $D_{\text{test}}$:**
1. Accuracy ($\%$)
2. Precision ($\%$)
3. Recall / Fake Recall ($\%$)
4. False Negative Rate / Bypass Rate ($\%$)
5. False Positive Rate / Real False Alarm ($\%$)
6. ROC-AUC ($[0, 1]$)

- [ ] **Step 1: Run 3-way comparative evaluation on frozen $D_{\text{test}}$**
Run:
```bash
PYTHONPATH=. taskset -c 0-5 .venv/bin/python3 scripts/evaluate_benchmarks.py --config configs/dataset_factory.yaml
```
Verify the comparative table demonstrates superior Fake Recall and ROC-AUC for $M_{\text{hard}}$ over $M_0$ and $M_{\text{rand}}$.

- [ ] **Step 2: Generate visual prediction & comparison gallery**
Run:
```bash
PYTHONPATH=. taskset -c 0-5 .venv/bin/python3 scripts/build_visual_gallery.py --config configs/dataset_factory.yaml
PYTHONPATH=. taskset -c 0-5 .venv/bin/python3 scripts/create_coursework_notebooks.py
```

- [ ] **Step 3: Run full test suite & self-audit**
Run: `taskset -c 0-5 .venv/bin/pytest -v`
Verify all 94+ tests pass with zero errors.

- [ ] **Step 4: Commit and Push Deliverables to GitHub**
```bash
git add experiments/ docs/ data/
git commit -m "feat(eval): completed 3-way proof-of-utility benchmark with 100% zero data leakage"
git push origin baseline/initial-baseline
```
