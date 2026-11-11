# Detector-Guided Iterative Hard-Sample Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Native execution chosen by user) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, closed-loop, detector-guided iterative synthetic dataset factory that evolves synthetic deepfakes across 2 generations ($M_0 \to H_1 \to M_1 \to H_2 \to M_2$) via tri-criteria mining (Uncertainty + False Negative + DINOv2 Diversity), calibrated adaptive thresholds, and parameter/prompt adaptation—delivering a 4-way controlled proof-of-utility benchmark with 100% zero data leakage.

**Architecture:**
1. **Zero-Leakage Identity-Disjoint Partitioning**: 4000 FFHQ real faces split at origin into $\text{Train\_IDs}$ ($80\%$) and immutable $\text{Test\_IDs}$ ($20\%$). All training, mining, and prompt tuning operate strictly on $\text{Train\_IDs}$. The evaluation set $D_{\text{test}}$ is synthesized once from $\text{Test\_IDs}$ and frozen.
2. **Tri-Criteria Hard Sample Mining Engine**: Evaluates candidate pools via: (a) False-Negative Detection Bypass ($P(\text{Fake}) < \tau_t$); (b) Uncertainty Sampling ($U(x) = 1 - |2p(x) - 1| \ge 0.70$); (c) DINOv2 Face-Crop Diversity Pruning (cosine distance clustering $\le 0.95$ to reject redundant near-duplicates), passed through the biometric Curation Gate.
3. **Detector Probability Calibration**: Temperature scaling on validation logits to equalize confidence scales across generations ($M_0, M_1, M_2$), giving mathematical meaning to shrinking thresholds ($\tau_1 = 0.50 \to \tau_2 = 0.35$).
4. **Generator Feedback Adaptation**: Parameterized feedback loop: (a) Hard Trait Mining from $H_1$; (b) 70% Exploitation (lowered `img2img_strength: 0.28`, localized periocular/lip micro-inpaint masks) + 30% Exploration prompts.
5. **4-Way Controlled Proof-of-Utility Benchmark**: Evaluates $M_0$ (Baseline), $M_{\text{rand}}$ (Random Augmentation control), $M_1$ (Iter-1), and $M_2$ (Iter-2 + Feedback) on frozen $D_{\text{test}}$ to prove that hardness evolution yields superior generalizability over brute-force data scaling.

**Tech Stack:** PyTorch, Diffusers (Stable Diffusion v1.5), Meta DINOv2 (`facebook/dinov2-small`), OpenCV, torchvision, PIL, NumPy, Scikit-learn, Pytest.

**Spec:** `docs/PURPOSE.md`, `.agents/rules/ZERO_MOCK_DATA_RULE.md`, `configs/dataset_factory.yaml`.

---

## Global Constraints

- **Strict Identity-Disjoint Split (100% Anti-Leakage)**: Every identity belongs either to $\text{Train\_IDs}$ or $\text{Test\_IDs}$. $\text{Train\_IDs} \cap \text{Test\_IDs} = \emptyset$. All training pools ($D_0, H_1, H_2, R$) derive from $\text{Train\_IDs}$. $D_{\text{test}}$ derives strictly from $\text{Test\_IDs}$.
- **Zero-Mock Data & Checkpoint Guardrail**: Hard sample mining requires real trained checkpoints. Fallback to uninitialized weights is strictly forbidden.
- **AMD Ryzen CPU Affinity**: Always execute Python commands with `taskset -c 0-5` to avoid non-monotonic clock assertion errors (`ApproximateClock`).
- **GPU & VRAM Hygiene**: Maintain single-pipeline lifecycle in Stable Diffusion (`_switch_mode`), with `gc.collect()` and `torch.cuda.empty_cache()` to keep VRAM usage under 2.0GB on the RTX 3050 Laptop.
- **Coursework Feasible Scale**: Total sample budget across all stages: 160–200 images. Total generation runtime $\le 12$ minutes; total training runtime $\le 3$ minutes.

---

## Review Focus

1. **Identity Leakage Across Generations**: Verify that no subject in $D_{\text{test}}$ was used in Round 0, Round 1, or Round 2 training or mining.
   - *Test*: `test_strict_identity_isolation_across_rounds` in `tests/test_splits.py`.
2. **Confidence Miscalibration Across Models**: Verify that temperature scaling maps raw logits to monotonic, calibrated probabilities before threshold comparison.
   - *Test*: `test_temperature_scaling_calibration_improves_ece` in `tests/test_calibration.py`.
3. **Diversity Collapse in Hard Samples**: Verify that candidate faces with cosine similarity $> 0.95$ are pruned, preventing 10 identical faces from flooding $H_t$.
   - *Test*: `test_diversity_filter_prunes_near_duplicate_embeddings` in `tests/test_hardness.py`.
4. **Micro-Inpainting Anatomical Defect Bypass**: Verify that localized periocular/lip inpainting does not introduce boundary seam spikes or duplicate mouths.
   - *Test*: `test_micro_inpaint_preserves_facial_continuity` in `tests/test_dynamic_masking.py`.
5. **Replay Buffer Non-Degradation (Catastrophic Forgetting)**: Verify that $M_2$ retains $\ge 90\%$ accuracy on initial $D_0$ test subset while improving on hard samples.
   - *Test*: `test_replay_buffer_prevents_catastrophic_forgetting` in `tests/test_iterative_framework.py`.

---

## Task Decomposition

### Task 1: Strict Identity-Disjoint Partitioning & Immutable Test Vault

**Files:**
- Modify: `src/data/splits.py`
- Modify: `configs/dataset_factory.yaml`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: `data/raw/manifest.jsonl` (4000 real faces).
- Produces:
  - `data/partitions/train_identities.jsonl` (80% of identities).
  - `data/partitions/test_identities.jsonl` (20% of identities, read-only).
- Guarantees: $\text{Train\_IDs} \cap \text{Test\_IDs} = \emptyset$.

- [ ] **Step 1: Write test verifying identity disjointness and immutable partition generation**
```python
def test_strict_identity_isolation_across_rounds():
    from src.data.splits import create_identity_partitions
    from src.data.image_schema import ImageMetadata
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        records = [
            ImageMetadata(image_id=f"ffhq_{i:05d}", label=0, source_real_id=f"ffhq_{i:05d}")
            for i in range(100)
        ]
        train_p, test_p = create_identity_partitions(records, output_dir=tmp_p, train_ratio=0.8, seed=42)
        assert train_p.exists() and test_p.exists()
        import json
        train_ids = {json.loads(line)["source_real_id"] for line in open(train_p)}
        test_ids = {json.loads(line)["source_real_id"] for line in open(test_p)}
        assert len(train_ids) == 80
        assert len(test_ids) == 20
        assert train_ids.isdisjoint(test_ids)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_splits.py -k test_strict_identity_isolation_across_rounds -v`
Expected: FAIL with `ImportError: cannot import name 'create_identity_partitions'`

- [ ] **Step 3: Implement `create_identity_partitions` in `src/data/splits.py`**
Implement function that takes metadata list, groups by `get_group_key`, deterministically splits into train ($80\%$) and test ($20\%$), writes JSONL partitions, and sets read-only permissions on test partition.

- [ ] **Step 4: Run test to verify it passes**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_splits.py -k test_strict_identity_isolation_across_rounds -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/data/splits.py tests/test_splits.py
git commit -m "feat(splits): add strict identity-disjoint partition creation with immutable test vault"
```

---

### Task 2: Detector Probability Calibration Engine (Temperature Scaling)

**Files:**
- Create: `src/models/detector/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: Uncalibrated logits $z(x) \in \mathbb{R}^N$ and true labels $y \in \{0, 1\}^N$.
- Produces: `TemperatureScaler(nn.Module)` providing `calibrate_probabilities(logits: torch.Tensor) -> torch.Tensor` with optimal learned scalar $T > 0$.

- [ ] **Step 1: Write test for temperature scaling calibration**
```python
def test_temperature_scaling_calibration_improves_ece():
    import torch
    from src.models.detector.calibration import TemperatureScaler

    scaler = TemperatureScaler()
    # Simulated over-confident logits
    logits = torch.tensor([-5.0, -3.0, 3.0, 6.0])
    labels = torch.tensor([0, 0, 1, 1])
    scaler.fit(logits, labels)
    assert scaler.temperature.item() > 0.0

    calibrated_probs = scaler.calibrate_probabilities(logits)
    assert calibrated_probs.shape == (4,)
    assert (calibrated_probs >= 0.0).all() and (calibrated_probs <= 1.0).all()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.detector.calibration'`

- [ ] **Step 3: Implement `TemperatureScaler` in `src/models/detector/calibration.py`**
Implement PyTorch module optimizing $T$ via L-BFGS to minimize Binary Cross Entropy on validation logits. Add `calibrate_probabilities(logits)` returning sigmoid scaled by $1/T$.

- [ ] **Step 4: Run test to verify it passes**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/models/detector/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): temperature scaling for cross-generation detector confidence alignment"
```

---

### Task 3: Tri-Criteria Hard Sample Mining Engine with DINOv2 Diversity Filter

**Files:**
- Modify: `src/pipelines/mine_hard_samples.py`
- Modify: `src/models/evaluators/hardness.py`
- Test: `tests/test_hardness.py`

**Interfaces:**
- Consumes: Curated synthetic metadata list, trained detector checkpoint, optional calibrated temperature $T$.
- Produces: Filtered hard samples $H_t$ meeting:
  1. False Negative bypass: $P(\text{Fake}) < \tau_t$.
  2. Uncertainty sampling: $U(x) = 1 - |2p(x) - 1| \ge 0.60$.
  3. DINOv2 diversity pruning: Greedy selection rejecting samples with face-crop cosine similarity $> 0.95$ to previously chosen hard samples.

- [ ] **Step 1: Write test for DINOv2 diversity filtering**
```python
def test_diversity_filter_prunes_near_duplicate_embeddings():
    import torch
    from src.pipelines.mine_hard_samples import filter_by_embedding_diversity

    # 4 embeddings: 0 and 1 are almost identical (cos > 0.99), 2 and 3 are distinct
    e0 = torch.tensor([1.0, 0.0, 0.0])
    e1 = torch.tensor([0.999, 0.01, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    e3 = torch.tensor([0.0, 0.0, 1.0])
    embeddings = torch.stack([e0, e1, e2, e3])
    scores = [0.10, 0.12, 0.20, 0.30]  # Lower is harder

    selected_indices = filter_by_embedding_diversity(embeddings, scores, max_cosine_sim=0.95)
    # e1 must be pruned because e0 was already chosen and has lower score
    assert 0 in selected_indices
    assert 1 not in selected_indices
    assert 2 in selected_indices
    assert 3 in selected_indices
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_hardness.py -k test_diversity_filter_prunes_near_duplicate_embeddings -v`
Expected: FAIL with `cannot import name 'filter_by_embedding_diversity'`

- [ ] **Step 3: Implement `filter_by_embedding_diversity` and integrate into `mine_hard_samples.py`**
Implement greedy diversity pruning sorted by score. Integrate into `mine_hard_samples_detector` alongside uncertainty metric.

- [ ] **Step 4: Run test to verify it passes**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_hardness.py -k test_diversity_filter_prunes_near_duplicate_embeddings -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/pipelines/mine_hard_samples.py src/models/evaluators/hardness.py tests/test_hardness.py
git commit -m "feat(hardness): tri-criteria mining with false-negative, uncertainty, and DINOv2 diversity pruning"
```

---

### Task 4: Generator Adaptation Engine (Micro-Inpainting & Feedback Tuning)

**Files:**
- Modify: `src/models/diffusion/stable_diffusion_engine.py`
- Modify: `src/pipelines/generate_deepfake_images.py`
- Test: `tests/test_dynamic_masking.py`

**Interfaces:**
- Consumes: Source face tensor `(C, H, W)`, target feature region (`"face"`, `"periocular"`, or `"lips"`), `img2img_strength: float`.
- Produces: Localized anatomical mask `(1, H, W)` and high-frequency preserving generation.

- [ ] **Step 1: Write test for localized micro-inpainting mask**
```python
def test_micro_inpaint_preserves_facial_continuity():
    import torch
    from src.models.diffusion.stable_diffusion_engine import generate_dynamic_face_mask

    dummy_img = torch.zeros(3, 256, 256)
    face_bbox = (0.2, 0.2, 0.8, 0.8)
    # Periocular region mask (around eyes, upper 35-55% of face)
    mask = generate_dynamic_face_mask(dummy_img, bbox=face_bbox, target_region="periocular", blur_radius=8)
    assert mask.shape == (1, 256, 256)
    # Eye region is masked (> 0.5)
    eye_y = int((0.2 + 0.6 * 0.45) * 256)
    assert mask[:, eye_y, 128].item() > 0.5
    # Chin region remains unmasked (< 0.1)
    chin_y = int((0.2 + 0.6 * 0.90) * 256)
    assert mask[:, chin_y, 128].item() < 0.1
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_dynamic_masking.py -k test_micro_inpaint_preserves_facial_continuity -v`
Expected: FAIL with `unexpected keyword argument 'target_region'`

- [ ] **Step 3: Implement `target_region` in `generate_dynamic_face_mask` and update pipeline**
Add `target_region: str = "full"` with options `"full"`, `"periocular"`, and `"lips"`. Update `generate_deepfake_images.py` to support variable `img2img_strength` and `target_region`.

- [ ] **Step 4: Run test to verify it passes**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_dynamic_masking.py -k test_micro_inpaint_preserves_facial_continuity -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/models/diffusion/stable_diffusion_engine.py src/pipelines/generate_deepfake_images.py tests/test_dynamic_masking.py
git commit -m "feat(diffusion): localized micro-inpainting masks for periocular and lip adversarial synthesis"
```

---

### Task 5: Objective Cross-Model Hardness Verification Engine

**Files:**
- Create: `src/models/evaluators/objective_hardness.py`
- Test: `tests/test_objective_hardness.py`

**Interfaces:**
- Consumes: Mined hard sample metadata lists $H_1$, $H_2$, and reference real metadata list.
- Produces: `ObjectiveHardnessReport` computing:
  1. Mean Face-Crop DINOv2 cosine similarity to Real manifold ($\text{Sim}(H_2, \text{Real}) \text{ vs } \text{Sim}(H_1, \text{Real})$).
  2. Independent cross-model bypass rate on a zero-shot backbone (e.g. ResNet or standard ViT).

- [ ] **Step 1: Write test for objective manifold density comparison**
```python
def test_objective_manifold_density_comparison():
    import torch
    from src.models.evaluators.objective_hardness import compute_manifold_proximity_gain

    # Real embeddings (clustered around unit vector 0)
    real_emb = torch.tensor([[1.0, 0.0], [0.99, 0.05]])
    # H1 embeddings (average similarity ~0.8)
    h1_emb = torch.tensor([[0.8, 0.6], [0.85, 0.52]])
    # H2 embeddings (average similarity ~0.95 - closer to real)
    h2_emb = torch.tensor([[0.95, 0.31], [0.96, 0.28]])

    gain = compute_manifold_proximity_gain(h1_emb, h2_emb, real_emb)
    assert gain["mean_sim_h1"] < gain["mean_sim_h2"]
    assert gain["proximity_delta"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_objective_hardness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.evaluators.objective_hardness'`

- [ ] **Step 3: Implement `src/models/evaluators/objective_hardness.py`**
Implement `compute_manifold_proximity_gain` and `evaluate_cross_model_hardness` using k-NN cosine distance against reference real embeddings.

- [ ] **Step 4: Run test to verify it passes**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_objective_hardness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/models/evaluators/objective_hardness.py tests/test_objective_hardness.py
git commit -m "feat(eval): objective manifold proximity metric comparing H2 vs H1 without model bias"
```

---

### Task 6: 2-Round Iterative Framework Orchestration Pipeline

**Files:**
- Create: `src/pipelines/run_iterative_framework.py`
- Create: `tests/test_iterative_framework.py`

**Interfaces:**
- Consumes: `configs/dataset_factory.yaml`
- Orchestrates:
  - **Round 0**: Train $M_0$ on $D_0$ (30 Real + 30 Fake Base from `Train_IDs`). Calibrate $M_0$.
  - **Round 1**: Evaluate Pool 1 with calibrated $M_0$, mine $H_1$ ($\tau_1 = 0.50$, Uncertainty $\ge 0.6$, Diversity filter). Sample matching random control $R$ ($|R| = |H_1|$). Build $D_1 = D_0 \cup H_1$ and $D_{\text{rand}} = D_0 \cup R$. Train $M_1$ and $M_{\text{rand}}$.
  - **Round 2**: Analyze $H_1$ provenance, generate Pool 2 with Feedback (70% exploitation, `strength=0.28`, micro-inpainting + 30% exploration). Mine $H_2$ on calibrated $M_1$ with $\tau_2 = 0.35$. Build $D_2 = D_1 \cup H_2$. Train $M_2$.
- Guarantee: Replay buffer preserves all prior stages ($D_0 \subset D_1 \subset D_2$).

- [ ] **Step 1: Write test for iterative replay buffer and stage integrity**
```python
def test_replay_buffer_prevents_catastrophic_forgetting():
    from src.pipelines.run_iterative_framework import assemble_evolutionary_stage
    from src.data.image_schema import ImageMetadata

    d0 = [ImageMetadata(image_id=f"base_{i}", label=1, data_source="baseline") for i in range(10)]
    h1 = [ImageMetadata(image_id=f"h1_{i}", label=1, data_source="hard_sample", is_hard_sample=True) for i in range(5)]
    h2 = [ImageMetadata(image_id=f"h2_{i}", label=1, data_source="hard_sample", is_hard_sample=True) for i in range(5)]

    d1 = assemble_evolutionary_stage(d0, [h1])
    d2 = assemble_evolutionary_stage(d0, [h1, h2])

    assert len(d1) == 15
    assert len(d2) == 20
    # Baseline faces must be 100% preserved
    base_ids_in_d2 = {m.image_id for m in d2 if m.data_source == "baseline"}
    assert len(base_ids_in_d2) == 10
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_iterative_framework.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipelines.run_iterative_framework'`

- [ ] **Step 3: Implement `assemble_evolutionary_stage` and CLI in `src/pipelines/run_iterative_framework.py`**
Implement the modular stage aggregator, feedback generator hook, and CLI entry point.

- [ ] **Step 4: Run test to verify it passes**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_iterative_framework.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/pipelines/run_iterative_framework.py tests/test_iterative_framework.py
git commit -m "feat(pipeline): 2-round iterative evolutionary framework with replay buffer"
```

---

### Task 7: 4-Way Comparative Proof-of-Utility Benchmark & Deliverables

**Files:**
- Create: `scripts/evaluate_iterative_benchmarks.py`
- Modify: `scripts/build_visual_gallery.py`
- Output:
  - `experiments/reports/ITERATIVE_PROOF_OF_UTILITY.md` (Golden Table comparing $M_0$ vs $M_{\text{rand}}$ vs $M_1$ vs $M_2$).
  - `experiments/predictions/iterative_evolution_gallery.png` (Visual progression gallery).

**Metrics on Frozen $D_{\text{test}}$:**
1. Test Accuracy ($\%$)
2. Fake Recall ($\%$) — Primary Metric (Higher is better)
3. Bypass Rate / FNR ($\%$) — Primary Metric (Lower is better)
4. ROC-AUC ($[0, 1]$)
5. Real False Positive Rate ($\%$) — Controlled under $10\%$

- [ ] **Step 1: Write test verifying 4-way evaluation schema**
```python
def test_four_way_benchmark_schema():
    from scripts.evaluate_iterative_benchmarks import build_benchmark_summary_table
    metrics = {
        "M0_baseline": {"accuracy": 0.85, "fake_recall": 0.80, "bypass_rate": 0.20, "auc": 0.875, "fpr": 0.10},
        "M_rand": {"accuracy": 0.875, "fake_recall": 0.85, "bypass_rate": 0.15, "auc": 0.900, "fpr": 0.10},
        "M1_hard": {"accuracy": 0.925, "fake_recall": 0.90, "bypass_rate": 0.10, "auc": 0.940, "fpr": 0.05},
        "M2_adaptive": {"accuracy": 0.975, "fake_recall": 0.95, "bypass_rate": 0.05, "auc": 0.985, "fpr": 0.00},
    }
    markdown_table = build_benchmark_summary_table(metrics)
    assert "| Model |" in markdown_table
    assert "M2_adaptive" in markdown_table
    assert "0.985" in markdown_table
```

- [ ] **Step 2: Run test to verify it fails**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_iterative_framework.py -k test_four_way_benchmark_schema -v`
Expected: FAIL with `cannot import name 'build_benchmark_summary_table'`

- [ ] **Step 3: Implement `scripts/evaluate_iterative_benchmarks.py` and visual gallery generator**
Implement automated evaluation of checkpoints $M_0, M_{\text{rand}}, M_1, M_2$ against `data/benchmarks/frozen_test.jsonl`, generating Markdown report and visual comparison gallery.

- [ ] **Step 4: Run test to verify it passes**
Run: `taskset -c 0-5 .venv/bin/pytest tests/test_iterative_framework.py -k test_four_way_benchmark_schema -v`
Expected: PASS

- [ ] **Step 5: Run full test suite & self-audit**
Run: `taskset -c 0-5 .venv/bin/pytest -v`
Verify all 100+ tests pass with zero errors.

- [ ] **Step 6: Commit and Push Deliverables to GitHub**
```bash
git add scripts/ experiments/ tests/ configs/ src/
git commit -m "feat(eval): completed 4-way iterative proof-of-utility evaluation and visual gallery"
git push origin baseline/initial-baseline
```
