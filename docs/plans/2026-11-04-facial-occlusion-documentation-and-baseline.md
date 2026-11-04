# 2026-11-04-facial-occlusion-documentation-and-baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the foundational academic and technical documentation suite and prepare the baseline segmentation environment for the research topic: "Diffusion-Based Synthetic Facial Occlusion Dataset Generation for Fine-Grained Occlusion Segmentation".

**Architecture:** Build a 7-document specifications framework adhering strictly to repository markdown rules (`UPPER_SNAKE_CASE.md`, 7-field metadata header, relative links), define standard data contracts (6-class occlusion taxonomy, tensor contracts `(B, C, H, W)` and `(H, W)`), and formulate the executable baseline configuration for fine-grained segmentation.

**Tech Stack:** Markdown (GitHub Flavored), PyTorch, Torchvision, PyYAML, OpenCV, SegFormer / DeepLabV3+.

**Spec:** [docs/PURPOSE.md](docs/PURPOSE.md) and Topic Brief: "Diffusion-Based Synthetic Facial Occlusion Dataset Generation for Fine-Grained Occlusion Segmentation" (Xây dựng bộ dữ liệu che khuất khuôn mặt tổng hợp bằng mô hình Diffusion phục vụ bài toán phân đoạn vùng che khuất ở mức chi tiết).

## Global Constraints

- All markdown files in `docs/` must be named in UPPERCASE (`UPPER_SNAKE_CASE.md` or `UPPERCASE.md`).
- Every markdown file must begin with the standard 7-field header: Motivation/Background, Purpose, Overview Pipeline, Detailed Plan, References, Created, Last Updated.
- No absolute system paths: Use relative markdown links only (e.g. `[config](configs/config.yaml)`).
- Zero unvectorized raw pixel loops in image processing contracts.
- Explicit tensor shapes documented in all docstrings and contracts: `(B, C, H, W)` for PyTorch, `(H, W, C)` for OpenCV.
- Color space awareness: OpenCV BGR vs. PyTorch/PIL RGB must be clearly stipulated in contracts.

## Review Focus

- Undefined label ID mapping between binary and multi-class tasks breaking downstream evaluators: Pin explicit mapping table in `TAXONOMY_AND_DATA_SPEC.md`.
- Lossy compression artifacts corrupting integer segmentation masks: Explicitly forbid JPEG for masks; require lossless 8-bit PNG or uncompressed tensors.
- Mask misalignment during diffusion synthesis due to spatial shifting: Specify SAM/AffinityNet post-processing edge refinement protocol in `PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md`.
- Class imbalance (e.g., small glasses vs large medical masks): Specify combined CE + Dice loss formulation in `PHASE_02_SEGMENTATION_BASELINE_SPEC.md`.
- Ambiguity on real evaluation benchmark: Explicitly lock FaceOcc / RealOcc test split as the official unseen benchmark in `EXPERIMENT_PROTOCOL.md`.

---

### Task 1: Lock Purpose and Problem Boundary (`docs/PURPOSE.md`)

**Files:**
- Modify: `docs/PURPOSE.md`

**Interfaces:**
- Consumes: Official coursework brief & constraints from `GEMINI.md`.
- Produces: Locked problem statement, taxonomy targets, hardware constraints, and Definition of Done.

- [ ] **Step 1: Draft the updated `docs/PURPOSE.md` content with 7-field header**

Verify that all placeholders (`[Tên đề tài...]`, `[e.g. ...]`) are replaced with concrete specifications for "Diffusion-Based Synthetic Facial Occlusion Dataset Generation for Fine-Grained Occlusion Segmentation".

- [ ] **Step 2: Verify header compliance and relative links**

Run check to ensure all 7 metadata fields exist and zero absolute paths (`/home/...`) are present.

- [ ] **Step 3: Commit `docs/PURPOSE.md`**

```bash
git add docs/PURPOSE.md
git commit -m "docs: define coursework purpose and scope for facial occlusion segmentation"
```

---

### Task 2: Literature Review and Academic Gap (`docs/coursework/LITERATURE_REVIEW.md`)

**Files:**
- Create: `docs/coursework/LITERATURE_REVIEW.md`

**Interfaces:**
- Consumes: References to NatOcc, RandOcc (CVPRW), FaceOcc, CelebAMask-HQ, DiffuMask, and DatasetDM.
- Produces: Systematic comparison matrix highlighting limitations of 2D Cut-and-Paste methods versus Diffusion synthetic generation.

- [ ] **Step 1: Write `docs/coursework/LITERATURE_REVIEW.md`**

Document existing literature across 3 pillars:
1. Facial Occlusion Datasets (CelebAMask-HQ, RealOcc, FaceOcc).
2. Synthetic Occlusion Generation (NatOcc, RandOcc, Cut-and-Paste).
3. Diffusion-based dense perception and mask synthesis (DiffuMask, DatasetDM).
Include a Markdown comparison table contrasting realism, boundary quality, lighting/shadow harmonization, and annotation cost.

- [ ] **Step 2: Verify 7-field header and formatting**

Check that file begins with the required 7 fields and contains valid relative links.

- [ ] **Step 3: Commit `docs/coursework/LITERATURE_REVIEW.md`**

```bash
git add docs/coursework/LITERATURE_REVIEW.md
git commit -m "docs(coursework): add comprehensive literature review and academic gap analysis"
```

---

### Task 3: Occlusion Taxonomy and Data Contract (`docs/coursework/TAXONOMY_AND_DATA_SPEC.md`)

**Files:**
- Create: `docs/coursework/TAXONOMY_AND_DATA_SPEC.md`
- Create: `configs/dataset/occlusion_synthetic.yaml`

**Interfaces:**
- Consumes: `docs/PURPOSE.md` taxonomy requirements.
- Produces: Concrete class ID definitions (0 to 5), RGB vs single-channel mask specifications, directory structure schemas, and YAML configuration.

- [ ] **Step 1: Write `docs/coursework/TAXONOMY_AND_DATA_SPEC.md`**

Specify:
- Class IDs:
  - 0: Background / Unoccluded Face
  - 1: Face Mask / Medical Shield
  - 2: Eyeglasses / Sunglasses
  - 3: Hands / Fingers
  - 4: Handheld Objects (Microphones, Cups, Phones)
  - 5: Organic / Miscellaneous (Scarves, Hair strands, Random items)
- Image/Mask contracts: `(H, W, 3)` RGB uint8 and `(H, W)` grayscale PNG uint8.
- Directory hierarchy: `data/synthetic/images`, `data/synthetic/masks`, `data/real_val/`, `data/real_test/`.

- [ ] **Step 2: Create matching dataset config in `configs/dataset/occlusion_synthetic.yaml`**

Include image resolution ($512 \times 512$), num_classes (6), class names list, and normalization parameters.

- [ ] **Step 3: Commit dataset specification and config**

```bash
git add docs/coursework/TAXONOMY_AND_DATA_SPEC.md configs/dataset/occlusion_synthetic.yaml
git commit -m "docs(data): specify occlusion taxonomy, data contract, and yaml config"
```

---

### Task 4: Diffusion Synthesis Pipeline Architecture (`docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md`)

**Files:**
- Create: `docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md`

**Interfaces:**
- Consumes: `TAXONOMY_AND_DATA_SPEC.md`.
- Produces: Algorithmic blueprint for the synthetic data factory: Diffusion backbone choice, prompt templates, inpainting conditioning, and automated mask extraction/refinement with SAM.

- [ ] **Step 1: Write `docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md`**

Detail the 4-stage pipeline:
1. Base Clean Face Selection (CelebA-HQ / FFHQ unoccluded subset).
2. Occlusion Region & Prompt Generation (Stochastic bounding boxes + category-specific prompt bank).
3. Latent Inpainting (Stable Diffusion v1.5 Inpainting / ControlNet).
4. Automated Mask Derivation & Boundary Refinement (Attention thresholding + SAM boundary snapping + quality filtering via CLIP/face detector).

- [ ] **Step 2: Verify pipeline consistency and mermaid architecture diagram**

Verify flowchart and ensure no raw pixel operations are prescribed.

- [ ] **Step 3: Commit Phase 1 specification**

```bash
git add docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md
git commit -m "docs(phases): establish Phase 1 diffusion synthesis architectural specification"
```

---

### Task 5: Baseline Segmentation Architecture & Metrics (`docs/phases/PHASE_02_SEGMENTATION_BASELINE_SPEC.md`)

**Files:**
- Create: `docs/phases/PHASE_02_SEGMENTATION_BASELINE_SPEC.md`
- Create: `configs/model/segformer_baseline.yaml`

**Interfaces:**
- Consumes: `TAXONOMY_AND_DATA_SPEC.md`.
- Produces: Model topology (SegFormer-B0 / DeepLabV3+ ResNet-50), loss functions (`CE + Dice`), and quantitative evaluation metrics (mIoU, Boundary IoU).

- [ ] **Step 1: Write `docs/phases/PHASE_02_SEGMENTATION_BASELINE_SPEC.md`**

Detail:
- Model architectures: SegFormer-B0 as modern baseline, DeepLabV3+ ResNet-50 as classical benchmark.
- Tensor inputs `(B, 3, 512, 512)` -> logits `(B, 6, 512, 512)`.
- Loss formulation: $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Dice}$.
- Evaluation metric formulas: Mean Intersection-over-Union (mIoU) and Boundary IoU (BIoU) with boundary dilation radius $d=3$.

- [ ] **Step 2: Create `configs/model/segformer_baseline.yaml`**

Provide complete hyperparameters: backbone, num_classes, learning rate, optimizer (AdamW), weight decay, scheduler (CosineAnnealingLR).

- [ ] **Step 3: Commit Phase 2 specification and model config**

```bash
git add docs/phases/PHASE_02_SEGMENTATION_BASELINE_SPEC.md configs/model/segformer_baseline.yaml
git commit -m "docs(phases): specify Phase 2 segmentation baseline architecture and metrics"
```

---

### Task 6: Controlled Experiment Protocol (`docs/experiments/EXPERIMENT_PROTOCOL.md`)

**Files:**
- Create: `docs/experiments/EXPERIMENT_PROTOCOL.md`

**Interfaces:**
- Consumes: Baseline specs and evaluation criteria.
- Produces: Formal 4-way comparative experimental matrix and ablation study protocol.

- [ ] **Step 1: Write `docs/experiments/EXPERIMENT_PROTOCOL.md`**

Define experimental matrix:
- Exp 0 (Lower-bound): Small real dataset training $\rightarrow$ RealOcc Test.
- Exp 1 (Heuristic Baseline): Traditional Cut-and-Paste / NatOcc $\rightarrow$ RealOcc Test.
- Exp 2 (Proposed Baseline): Pure Diffusion Synthetic Dataset $\rightarrow$ RealOcc Test.
- Exp 3 (Combined / Fine-tuned): Synthetic Pretraining + Real Fine-tuning $\rightarrow$ RealOcc Test.
- Ablations: Dataset volume scaling curve (1K, 2.5K, 5K, 10K images), Impact of SAM edge refinement.

- [ ] **Step 2: Commit experiment protocol**

```bash
git add docs/experiments/EXPERIMENT_PROTOCOL.md
git commit -m "docs(experiments): establish controlled experiment protocol and comparative matrix"
```

---

### Task 7: Project Roadmap and Milestones Tracking (`docs/progress/PROJECT_ROADMAP.md`)

**Files:**
- Create: `docs/progress/PROJECT_ROADMAP.md`

**Interfaces:**
- Consumes: All phase specs.
- Produces: 9-phase AI-First Lifecycle roadmap with explicit deliverables and DoD checkboxes for each phase.

- [ ] **Step 1: Write `docs/progress/PROJECT_ROADMAP.md`**

Map out Phase 1 through Phase 9:
- Phase 1: Environment & Project Scaffolding
- Phase 2: Data Contracts & Taxonomy Verification
- Phase 3: Diffusion Generation Data Factory
- Phase 4: Data Quality Auditing & Curation
- Phase 5: Segmentation DataModule & Pipeline Implementation
- Phase 6: Baseline Model Training (SegFormer-B0)
- Phase 7: Controlled Experiments & Comparative Evaluation
- Phase 8: Analysis, Failure Cases & Prediction Gallery
- Phase 9: Final Coursework Report & Presentation Delivery

- [ ] **Step 2: Commit project roadmap**

```bash
git add docs/progress/PROJECT_ROADMAP.md
git commit -m "docs(progress): establish project roadmap across 9-phase AI-first lifecycle"
```
