# Stable Diffusion-Based Synthetic Facial Image Dataset Generation Implementation Plan

- **Motivation/Background**: Realign the coursework workspace to the official course topic (Topic 11: "Stable diffusion for data generation" referencing Hugging Face CV Course Unit 10) focused on 2D image data generation: "Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection".
- **Purpose**: Provide a complete, test-driven, modular plan to build a 2D facial image data generation factory using Stable Diffusion, automated multi-modal curation gates, and a downstream Deepfake Image Detection benchmark.
- **Overview Pipeline**: Phase 1 Governance & Specs -> Phase 2 2D Image Data Contracts & Schema -> Phase 3 Stable Diffusion Generation Engine -> Phase 4 Automated Quality & Biometric Curation -> Phase 5 Downstream Deepfake Image Detector & Cross-Benchmark.
- **Detailed Plan**: Tasks 1 to 6 covering specifications, 2D dataset loaders, Stable Diffusion engine (Text2Img, Inpaint, Img2Img), curation filters, and detector benchmark.
- **References**: `docs/PURPOSE.md`, `docs/OVERVIEW.md`, `docs/progress/PROJECT_ROADMAP.md`, `GEMINI.md`, [Hugging Face CV Course Unit 10](https://huggingface.co/learn/computer-vision-course/unit10/datagen-diffusion).
- **Created**: 2026-11-06T23:55:00+07:00
- **Last Updated**: 2026-11-06T23:55:00+07:00

---

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor and build a modular 2D Computer Vision pipeline utilizing Stable Diffusion (Text-to-Image, Inpainting, and Img2Img) to generate a high-fidelity synthetic facial deepfake image dataset and benchmark downstream Deepfake Image Detectors against cross-generator domain shift.

**Architecture:** A 4-stage pipeline adhering to Clean Architecture: (1) 2D facial image ingestion and manipulation schema `(B, 3, H, W)`; (2) Stable Diffusion generation factory supporting Text-to-Image, Inpainting, and Img2Img with deterministic mock mode for CI/CD; (3) Automated quality curation gates measuring FID, Laplacian sharpness, and CLIP similarity; (4) Downstream deepfake image classifier (ResNet/CNN backbone) measuring ROC-AUC, Accuracy, and generalization against diffusion fakes.

**Tech Stack:** Python 3.10+, PyTorch 2.4+, Torchvision, Hugging Face Diffusers, Transformers, OpenCV, NumPy, Pytest.

**Spec:** `docs/PURPOSE.md`

## Global Constraints

- Tensor Shape Contracts: All functions processing images must document and enforce `(B, C, H, W)` or `(C, H, W)` for PyTorch tensors, and `(H, W, C)` for OpenCV/NumPy.
- Color Space Awareness: Explicit BGR <-> RGB conversions (`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`). No silent color inversions.
- Coordinate Systems: Explicit `[x_min, y_min, x_max, y_max]` (`xyxy`) pixel bounding box format.
- Strict Reproducibility: Global seeding via `src.common.seed.seed_everything(seed)`.
- GPU Hygiene: Device-agnostic code (`device = torch.device(...)`), AMP `fp16` enabled for diffusion inference and detector training, and `torch.no_grad()` on all evaluation/inference loops.
- Zero Raw Loops: Vectorized NumPy/PyTorch operations only across pixels and batch dimensions.
- Zero `print()`: Use `src.common.logger.get_logger(__name__)`.

## Review Focus

1. **Diffusion VRAM Management**: High-resolution image generation can exceed single GPU VRAM; the generation engine must support fp16 latents, attention slicing, and standardized $256 \times 256$ or $512 \times 512$ batch sizes.
2. **Missing Image Files / Corrupted Headers**: The image dataset loader must validate file existence and integrity, falling back gracefully without halting training or inference.
3. **Imbalanced Real vs Fake Splits**: The DataModule must provide balanced batches ($50\%$ Real, $50\%$ Fake) to prevent detector collapse.
4. **Offline / Headless CI Compatibility**: Diffusers model weights (several GBs) cannot be downloaded inside sandboxed CI; the engine must include a deterministic mock mode producing valid tensors in test environments.
5. **Metric Numerical Stability**: Binary classification metrics (ROC-AUC, F1) must handle edge cases (e.g., all samples predicted as single class) without division-by-zero errors.

---

### Task 1: Governance & Academic Specifications Realignment

**Files:**
- Modify: `docs/PURPOSE.md`
- Modify: `docs/OVERVIEW.md`
- Modify: `docs/progress/PROJECT_ROADMAP.md`
- Modify: `README.md`
- Create: `configs/dataset/deepfake_image.yaml`
- Create: `configs/generation/stable_diffusion.yaml`
- Create: `configs/detection/detector_baseline.yaml`

**Interfaces:**
- Consumes: Topic prompt from syllabus (Topic 11: "Stable diffusion for data generation").
- Produces: Locked problem statement, quantitative targets (FID $\le 20.0$, CLIP Score $\ge 0.30$, Detector $\text{AUC} \ge 92.0\%$), and YAML configurations.

- [x] **Step 1: Update `docs/PURPOSE.md`**
- [x] **Step 2: Update `docs/OVERVIEW.md` & `docs/progress/PROJECT_ROADMAP.md`**
- [x] **Step 3: Update `README.md`**
- [x] **Step 4: Create Configuration Files in `configs/`**
- [x] **Step 5: Verify Markdown Formatting and Links**

---

### Task 2: 2D Deepfake Image Data Schema, Manifest & DataModule

**Files:**
- Create: `src/data/image_schema.py`
- Create: `src/data/image_dataset.py`
- Create: `src/data/image_datamodule.py`
- Modify: `src/data/__init__.py`
- Test: `tests/test_image_data.py`

**Interfaces:**
- Consumes: `configs/dataset/deepfake_image.yaml`, `src.common.logger`.
- Produces: `ImageSample(image: torch.Tensor [C, H, W], label: int, metadata: ImageMetadata)`, `DeepfakeImageDataModule` yielding batched tensors `(B, 3, H, W)` and `(B,)` labels.

- [x] **Step 1: Write failing unit test in `tests/test_image_data.py`**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement `src/data/image_schema.py`**
- [x] **Step 4: Implement `src/data/image_dataset.py` & `src/data/image_datamodule.py`**
- [x] **Step 5: Run test to verify it passes**

---

### Task 3: Stable Diffusion Generation Factory

**Files:**
- Create: `src/models/diffusion/stable_diffusion_engine.py`
- Create: `src/pipelines/generate_deepfake_images.py`
- Test: `tests/test_stable_diffusion.py`

**Interfaces:**
- Consumes: Text prompt, optional source real image tensor `(C, H, W)`, optional inpaint mask.
- Produces: Synthetic deepfake image tensor `(C, H, W)` or batch `(B, C, H, W)` in $[0.0, 1.0]$.

- [x] **Step 1: Write failing unit test in `tests/test_stable_diffusion.py`**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement `StableDiffusionEngine` in `src/models/diffusion/stable_diffusion_engine.py`**
- [x] **Step 4: Implement CLI Pipeline in `src/pipelines/generate_deepfake_images.py`**
- [x] **Step 5: Run test to verify it passes**

---

### Task 4: Automated Quality & Biometric Curation Gates

**Files:**
- Modify: `src/common/metrics.py`
- Create: `src/pipelines/curate_image_dataset.py`
- Test: `tests/test_image_curation.py`

**Interfaces:**
- Consumes: Image tensor `(C, H, W)` and prompt metadata.
- Produces: `ImageCurationReport(is_valid: bool, sharpness_score: float, clip_score: float, face_detected: bool)`.

- [x] **Step 1: Write failing unit test in `tests/test_image_curation.py`**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement image quality metrics in `src/common/metrics.py`**
- [x] **Step 4: Implement `src/pipelines/curate_image_dataset.py`**
- [x] **Step 5: Run test to verify it passes**

---

### Task 5: Downstream Deepfake Image Detector & Benchmark Suite

**Files:**
- Create: `src/models/detector/image_detector.py`
- Create: `src/pipelines/train_image_detector.py`
- Create: `src/pipelines/evaluate_image_benchmark.py`
- Test: `tests/test_image_detector.py`

**Interfaces:**
- Consumes: Batched image tensors `(B, 3, H, W)`.
- Produces: Binary classification logits `(B, 1)` and benchmark evaluation metrics (Accuracy, ROC-AUC, F1).

- [x] **Step 1: Write failing unit test in `tests/test_image_detector.py`**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement `DeepfakeImageDetector` in `src/models/detector/image_detector.py`**
- [x] **Step 4: Implement Training & Benchmark Pipelines**
- [x] **Step 5: Run test to verify it passes**

---

### Task 6: Full Integration & Test Suite Verification

- [x] **Step 1: Run full test suite across the entire project**
- [x] **Step 2: Execute sanity CLI dry-runs**
- [x] **Step 3: Final self-review & ledger closure**
