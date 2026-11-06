# Project Purpose & Coursework Requirements

- **Motivation/Background**: Canonical source of truth defining the core objectives, problem statement, evaluation criteria, and operational boundaries for the UTH Computer Vision Coursework on "Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection" (Topic 11: "Stable diffusion for data generation" referencing Hugging Face Computer Vision Course Unit 10).
- **Purpose**: Provide locked project scope, 2D image data contracts, manipulation taxonomy, and quantitative success metrics so that all subsequent engineering phases remain aligned and drift-free.
- **Overview Pipeline**: Problem analysis -> 2D face image ingestion & manifest -> Stable Diffusion synthetic deepfake generation -> Multi-modal quality curation -> Downstream deepfake detector benchmark -> Coursework submission.
- **Detailed Plan**: §1 Context & Academic Scope; §2 Coursework Problem Statement; §3 Operational Boundaries & Success Metrics; §4 Hardware & Compute Constraints; §5 Required Deliverables.
- **References**: `docs/README.md`, `docs/OVERVIEW.md`, `docs/progress/PROJECT_ROADMAP.md`, `GEMINI.md`, [Hugging Face CV Course Unit 10](https://huggingface.co/learn/computer-vision-course/unit10/datagen-diffusion).
- **Created**: 2026-11-07T00:05:00+07:00
- **Last Updated**: 2026-11-07T00:05:00+07:00

---

## 1. Context & Academic Scope

- **Course**: Computer Vision (Thị giác Máy tính) — Trường Đại học Giao thông Vận tải TP.HCM (UTH).
- **Project Role**: Major Coursework Project / Đồ án môn học (Topic 11).
- **Topic Name (English)**: Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection.
- **Topic Name (Tiếng Việt)**: Xây dựng bộ dữ liệu ảnh khuôn mặt Deepfake tổng hợp bằng Stable Diffusion phục vụ bài toán phát hiện Deepfake.
- **Course Syllabus Reference**: Topic 11 — *Stable diffusion for data generation* (`https://huggingface.co/learn/computer-vision-course/unit10/datagen-diffusion`).
- **Methodology**: AI-First Engineering, Modular Clean Architecture, Strict Reproducibility.

---

## 2. Coursework Problem Statement

- **Core Academic Bottleneck**:
  - **Cross-Generator Domain Shift**: State-of-the-art Deepfake Detectors trained on legacy GAN-based datasets (e.g. ProGAN, StyleGAN, StarGAN) experience severe performance drops (AUC dropping significantly) when evaluated against modern Diffusion-generated manipulations. Stable Diffusion produces photo-realistic skin textures without the typical frequency artifacts (such as checkerboard patterns or color mismatch seams) that classical detectors rely on.
  - **The Synthetic Data Advantage (Hugging Face Unit 10)**: Real-world balanced paired data for face manipulation is scarce and requires laborious manual annotation. Stable Diffusion (via Text-to-Image, Inpainting, and Img2Img) provides an automated, scalable data factory to generate controllable synthetic facial manipulations with zero manual labeling.
- **Proposed Solution**:
  - Build an end-to-end, reproducible **Stable Diffusion Deepfake Data Generation Factory**:
    1. Ingest clean real face portraits (CelebA / FFHQ) and construct a standardized manifest schema.
    2. Employ Stable Diffusion pipelines (Text2Img, Inpainting, Img2Img) to generate photorealistic facial manipulations.
    3. Apply automated multi-modal curation gates (Laplacian sharpness, CLIP text-image alignment, face detection confidence) to prune flawed generations.
    4. Train and benchmark a downstream Deepfake Image Detector (ResNet-18 / CNN) to evaluate detection robustness against diffusion-generated faces.

---

## 3. Operational Boundaries & Success Metrics

- **Primary Quantitative Targets**:
  - **Fréchet Inception Distance (FID)**: $\le 20.0$ on curated synthetic faces relative to real face distribution.
  - **CLIP Score (Image-Text Alignment)**: $\ge 0.30$ on prompt-guided manipulations.
  - **Laplacian Sharpness Score**: Variance of Laplacian $\ge 50.0$ to ensure crisp, non-blurred facial textures.
  - **Face Detection Retention**: $\ge 95\%$ of synthetic images must maintain valid facial landmark detections.
  - **Downstream Detector Benchmark**: Detector trained/fine-tuned on the synthetic dataset achieves $\text{ROC-AUC} \ge 92.0\%$ and $\text{Accuracy} \ge 85.0\%$ on held-out test splits.
- **Qualitative DoD (Definition of Done)**:
  - 100% Vectorized processing across batches `(B, 3, H, W)`.
  - Deterministic reproducibility via global seed configuration (`src.common.seed.seed_everything(seed)`).
  - Zero raw `print()` statements in source modules (enforcing `src.common.logger.get_logger`).
  - Comprehensive technical documentation and coursework report (`docs/reports/REPORT.md`).
- **Explicit Non-Goals (Strictly Out of Scope)**:
  - *No Video Diffusion Processing*: Focus strictly on 2D images `(B, 3, H, W)` per course prompt and Unit 10.
  - *No Real-Time Mobile NPU Optimization*: Standard workstation GPU pipelines are targeted.
  - *No End-User Commercial GUI*: Deliverables are modular Python packages, CLI pipelines, and benchmark reports.

---

## 4. Hardware & Compute Constraints

- **Execution Environment**: Linux workstation / Google Colab CUDA environment.
- **Compute Budget**: Single GPU (target VRAM constraints managed via standard $256 \times 256$ or $512 \times 512$ resolutions, AMP `fp16`, and attention slicing).
- **Framework Target**: Python 3.10+, PyTorch 2.4+, Torchvision, OpenCV, Hugging Face Diffusers, Transformers.

---

## 5. Required Deliverables

1. **2D Deepfake Image Schema & DataModule**: Standardized `ImageSample`, JSONL manifest, and `DeepfakeImageDataModule`.
2. **Stable Diffusion Generation Factory**: Modular generation engine (`src/models/diffusion/stable_diffusion_engine.py`) and CLI pipeline (`src/pipelines/generate_deepfake_images.py`).
3. **Automated Multi-Modal Curation Gate**: Quality filter module (`src/pipelines/curate_image_dataset.py`) filtering images via sharpness, CLIP alignment, and face detection.
4. **Curated Deepfake Dataset Manifest**: Standardized JSONL manifest partitioning images into `train`, `val`, and `test` splits.
5. **Downstream Deepfake Image Detector Benchmark**: Trainable detector baseline and evaluation suite (`src/pipelines/train_image_detector.py`, `src/pipelines/evaluate_image_benchmark.py`).
6. **Technical Coursework Report**: Final academic report in `docs/reports/` with quantitative comparative tables.
