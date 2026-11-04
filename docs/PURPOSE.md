# Project Purpose & Coursework Requirements

- **Motivation/Background**: Canonical source of truth defining the core objectives, problem statement, evaluation criteria, and operational boundaries for the UTH Computer Vision Coursework on "Diffusion-Based Synthetic Facial Occlusion Dataset Generation for Fine-Grained Occlusion Segmentation".
- **Purpose**: Provide locked project scope, taxonomy boundaries, and quantitative success metrics so that all subsequent engineering phases remain aligned and drift-free.
- **Overview Pipeline**: Problem analysis -> Definition of quantitative targets & constraints -> Milestone execution -> Coursework submission.
- **Detailed Plan**: §1 Context & Academic Scope; §2 Coursework Problem Statement; §3 Operational Boundaries & Success Metrics; §4 Hardware & Compute Constraints; §5 Required Deliverables.
- **References**: `.agents/rules/MD_CONVENTION.md`, `.agents/AI_FIRST_LIFECYCLE.md`, `docs/README.md`, `docs/coursework/TAXONOMY_AND_DATA_SPEC.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-04T08:15:00+07:00

---

## 1. Context & Academic Scope

- **Course**: Computer Vision (Thị giác Máy tính) — Trường Đại học Giao thông Vận tải TP.HCM (UTH).
- **Project Role**: Major Coursework Project / Đồ án môn học.
- **Topic Name (English)**: Diffusion-Based Synthetic Facial Occlusion Dataset Generation for Fine-Grained Occlusion Segmentation.
- **Topic Name (Tiếng Việt)**: Xây dựng bộ dữ liệu che khuất khuôn mặt tổng hợp bằng mô hình Diffusion phục vụ bài toán phân đoạn vùng che khuất ở mức chi tiết.
- **Methodology**: AI-First Engineering, Modular Clean Architecture, Strict Reproducibility.

---

## 2. Coursework Problem Statement

- **Core Academic Bottleneck**:
  - Facial occlusion severely degrades downstream computer vision systems (facial biometric authentication, expression recognition, eye gaze tracking, and 3D facial reconstruction).
  - Annotating fine-grained pixel-level masks for occluded faces in the real world is labor-intensive, expensive, and prone to boundary bleed.
  - Traditional synthetic augmentation techniques (such as naive 2D Cut-and-Paste or random shape masking) produce severe visual artifacts, unrealistic lighting mismatch, lack of physical depth, and missing contact shadows, leading to poor domain transfer to real-world occluded faces.
- **Proposed Solution**:
  - Implement a generative data factory utilizing a latent diffusion model (Stable Diffusion / ControlNet Inpainting) to synthesize photo-realistic occlusions on clean high-resolution face images (CelebA-HQ / FFHQ).
  - Automatically extract pixel-accurate ground-truth masks using cross-attention activation map thresholding and prompt conditioning, refined with boundary snapping algorithms (SAM / morphological filtering) without requiring manual human annotation.
  - Train and benchmark fine-grained occlusion segmentation networks (SegFormer / DeepLabV3+) on the synthetic dataset, verifying domain transferability and superior generalization on real-world benchmark datasets (such as RealOcc and FaceOcc).

---

## 3. Operational Boundaries & Success Metrics

- **Primary Quantitative Targets**:
  - **Segmentation Mean IoU (mIoU)**: $\ge 0.72$ on unseen real-world occlusion benchmarks (RealOcc / FaceOcc test split).
  - **Boundary IoU (BIoU)**: $\ge 0.65$ within a 3-pixel boundary margin, demonstrating fine-grained boundary fidelity.
  - **Inference Latency**: $\le 35\text{ ms}$ per $512 \times 512$ image on target GPU (real-time capable).
  - **Synthetic Data Quality**: CLIP score $\ge 0.28$ (measuring text-image alignment) and Face Detection Retention Rate $\ge 95\%$ on generated occluded faces.
- **Qualitative DoD (Definition of Done)**:
  - 100% Vectorized processing (zero unvectorized pixel loops).
  - Deterministic reproducibility via global seed configuration (`src.common.seed.seed_everything(seed)`).
  - Zero raw `print()` statements in source modules (enforcing `src.common.logger`).
  - Comprehensive technical documentation and coursework report (`docs/reports/REPORT.md`).
- **Explicit Non-Goals (Strictly Out of Scope)**:
  - *No Face Reconstruction / De-occlusion*: The objective is segmenting where the occlusion is, not synthesizing the missing face underneath.
  - *No Biometric Identification*: Evaluating face recognition accuracy under occlusion is secondary to segmentation precision.
  - *No Interactive GUI App*: Production inference scripts and evaluation suites are required; client-facing GUIs or mobile applications are non-goals.

---

## 4. Hardware & Compute Constraints

- **Execution Environment**: Linux workstation / Google Colab CUDA environment.
- **Compute Budget**: Single GPU (target VRAM constraints adhered to via AMP `fp16` and gradient accumulation).
- **Framework Target**: Python 3.10+, PyTorch 2.4+, Torchvision, OpenCV 4.8+, Hugging Face Diffusers & Transformers.

---

## 5. Required Deliverables

1. **Automated Diffusion Generation Pipeline**: Modular generation scripts in `src/pipelines/` capable of batch-synthesizing paired RGB images and segmentation masks.
2. **Standardized Synthetic Dataset**: Curated synthetic dataset partitioned into `train`, `val`, and evaluated against standardized `test_real` benchmark.
3. **Reproducible Baseline Segmentation Codebase**: Modular dataset loaders, model architectures (SegFormer-B0 / DeepLabV3+), loss formulations (CE + Dice), and training pipelines.
4. **Experimental Results & Prediction Galleries**: Quantitative comparison tables against classical cut-and-paste baselines, ablation studies, and visual segmentation galleries stored under `experiments/`.
5. **Technical Coursework Report**: Comprehensive academic report in `docs/reports/` and final presentation deck.
