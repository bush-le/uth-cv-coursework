# PHASE_EXTERNAL_6K_BENCHMARK.md — Design Specification for External 6k Legacy Synthetic Benchmark

- **Motivation/Background**: The user provided an external independent synthetic face dataset (`synthetic_face_dataset_6k.zip`, containing 6,000 images across inpaint, img2img, and text2img) to stress-test the generalization and out-of-distribution adversarial robustness of detector models ($M_0, M_1, M_2$).
- **Purpose**: Define the technical architecture, data extraction layout, evaluation metric contracts, and visualization additions for benchmarking the 6,000 external fake images.
- **Overview Pipeline**: Unpack zip to `data/external_benchmarks/` -> Rebase manifest paths -> Batch GPU inference on $M_0, M_1, M_2$ -> Compute Recall & Bypass Rate (overall, by task, by region) -> Log metrics -> Render visual chart in `src/common/visualizer.py` & Notebook 04.
- **Detailed Plan**: §1 Scope & Intent; §2 Approaches Evaluated; §3 Architectural Specifications; §4 Metric Contracts & Output Formats; §5 Definition of Done (DoD).
- **References**: `GEMINI.md`, `configs/dataset_factory.yaml`, `src/common/visualizer.py`.
- **Created**: 2026-11-11T07:44:00+07:00
- **Last Updated**: 2026-11-11T07:44:00+07:00

---

## 1. Intent & Scope

### 1.1 Intent
Validate the cross-generator adversarial robustness of our DINOv2 deepfake detector models ($M_0$ Baseline, $M_1$ Round 1, $M_2$ Round 2) against an external, independent synthetic dataset of 6,000 images generated via Stable Diffusion / RealImage 5.1.

### 1.2 Goals
- Extract `synthetic_face_dataset_6k.zip` into an isolated workspace directory: `data/external_benchmarks/synthetic_face_dataset_6k/`.
- Rebase file paths in `fake_manifest.jsonl` to ensure 100% path resolvability.
- Evaluate $M_0$, $M_1$, and $M_2$ across all 6,000 fake images using standard evaluation transforms `(224, 224)` on GPU.
- Compute quantitative fake detection metrics:
  - **Overall Fake Recall** ($P(\text{Fake}) \ge 0.5$)
  - **Overall Bypass Rate** ($P(\text{Fake}) < 0.5$)
  - **Task Breakdown**: `inpaint` vs `img2img` vs `text2img`
  - **Regional Breakdown**: `eyes`, `mouth`, `full`
- Persist structured results to `experiments/logs/legacy_6k_benchmark_results.json`.
- Add a modular visualization function `plot_legacy_6k_benchmark` in `src/common/visualizer.py`.
- Update `notebooks/coursework/04_active_learning_production_visualizer.ipynb` with pre-rendered visual charts comparing $M_0$, $M_1$, and $M_2$ on this external benchmark.

### 1.3 Non-Goals & Anti-Goals
- **Non-Goal**: We do not mix these 6,000 images into the 10,000 production training dataset or master test vaults (strict isolation must be maintained).
- **Non-Goal**: We do not require pairing with real images (pure fake detection / stress-test metric contract as requested by the user).
- **Anti-Goal**: Never modify existing checkpoint weights or retraining pipelines during this read-only benchmark.

---

## 2. Approaches Evaluated

1. **Approach 1: Dedicated Standalone Benchmark Pipeline & Notebook Integration (Chosen)**:
   - Extract to `data/external_benchmarks/synthetic_face_dataset_6k/`.
   - Run batch inference via `scripts/benchmark_legacy_6k.py`.
   - Export structured metrics to `experiments/logs/legacy_6k_benchmark_results.json`.
   - Implement `plot_legacy_6k_benchmark` in `src/common/visualizer.py` and embed in Notebook 04.
   - *Trade-off*: Requires implementing the visualizer and re-rendering Notebook 04, but produces rich, publication-ready charts.
2. **Approach 2: Minimal CLI Evaluation Only (Rejected)**:
   - Run a quick evaluation script and print numbers to terminal.
   - *Trade-off*: Faster, but lacks visual inspection and cannot be reviewed visually in the coursework notebook.

---

## 3. Architecture & Data Layout

```
data/
└── external_benchmarks/
    └── synthetic_face_dataset_6k/
        ├── fake_manifest.jsonl      (6,000 records, paths rebased)
        ├── inpaint/                 (PNG files)
        ├── img2img/                 (PNG files)
        └── text2img/                (PNG files)
```

Manifest record contract:
```json
{
  "image_path": "data/external_benchmarks/synthetic_face_dataset_6k/inpaint/0_inpaint.png",
  "task": "inpaint",
  "sub_region": "mouth",
  "source_real_image": "0.jpg",
  "prompt": "...",
  "seed": 42
}
```

---

## 4. Metric Contracts & Output Formats

### 4.1 Evaluation Logic
For each model $m \in \{M_0, M_1, M_2\}$:
1. Pass batches of shape `(B, 3, 224, 224)` through `DeepfakeImageDetector`.
2. Compute predicted probabilities $\hat{p} = \sigma(z) \in [0.0, 1.0]$.
3. Binary decision at threshold $\tau = 0.5$:
   - $\text{Detected (Recall)} = \frac{1}{N} \sum \mathbb{I}(\hat{p} \ge 0.5)$
   - $\text{Bypass Rate} = \frac{1}{N} \sum \mathbb{I}(\hat{p} < 0.5) = 1 - \text{Recall}$
   - Mean confidence: $\mu_p = \frac{1}{N} \sum \hat{p}$

### 4.2 Output File: `experiments/logs/legacy_6k_benchmark_results.json`
```json
{
  "total_samples": 6000,
  "models": {
    "M0_baseline": {
      "overall_recall": 0.0,
      "overall_bypass_rate": 0.0,
      "mean_fake_prob": 0.0,
      "by_task": { ... },
      "by_region": { ... }
    },
    "M1_active_round1": { ... },
    "M2_active_round2": { ... }
  }
}
```

---

## 5. Definition of Done (DoD)

1. [x] Archive extracted and 6,000 images verified at `data/external_benchmarks/synthetic_face_dataset_6k/`.
2. [x] `fake_manifest.jsonl` contains valid, accessible relative paths for all 6,000 files.
3. [x] `scripts/benchmark_legacy_6k.py` completes execution on CUDA for $M_0, M_1, M_2$ with exit code 0.
4. [x] `experiments/logs/legacy_6k_benchmark_results.json` and `experiments/logs/legacy_6k_predictions.npz` generated and validated.
5. [x] `plot_legacy_6k_benchmark`, `plot_legacy_6k_distributions`, `plot_legacy_6k_gallery_inspection`, `plot_legacy_sample_inspection`, and `render_legacy_6k_markdown_table` implemented in `src/common/visualizer.py` and covered by `pytest tests/test_visualizer.py`.
6. [x] Dedicated notebook `notebooks/coursework/05_legacy_6k_out_of_distribution_benchmark.ipynb` created with 15 sections, pre-rendered charts, and interactive inspector.
7. [x] Full test suite passes: `pytest tests/` (146/146 green).
