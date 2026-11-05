# Project Purpose & Coursework Requirements

- **Motivation/Background**: Canonical source of truth defining the core objectives, problem statement, evaluation criteria, and operational boundaries for the UTH Computer Vision Coursework on "Diffusion-Based Synthetic Occluded Face Dataset Generation for Facial De-occlusion".
- **Purpose**: Provide locked project scope, taxonomy boundaries, triplet data contracts, and quantitative success metrics so that all subsequent engineering phases remain aligned and drift-free.
- **Overview Pipeline**: Problem analysis -> Triplet dataset generation & taxonomy -> De-occlusion baseline modeling & identity loss -> Comprehensive multi-metric benchmark -> Coursework submission.
- **Detailed Plan**: §1 Context & Academic Scope; §2 Coursework Problem Statement; §3 Operational Boundaries & Success Metrics; §4 Hardware & Compute Constraints; §5 Required Deliverables.
- **References**: `.agents/rules/MD_CONVENTION.md`, `.agents/AI_FIRST_LIFECYCLE.md`, `docs/README.md`, `docs/coursework/TAXONOMY_AND_DATA_SPEC.md`, `docs/coursework/LITERATURE_REVIEW.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-05T18:30:00+07:00

---

## 1. Context & Academic Scope

- **Course**: Computer Vision (Thị giác Máy tính) — Trường Đại học Giao thông Vận tải TP.HCM (UTH).
- **Project Role**: Major Coursework Project / Đồ án môn học.
- **Topic Name (English)**: Diffusion-Based Synthetic Occluded Face Dataset Generation for Facial De-occlusion.
- **Topic Name (Tiếng Việt)**: Xây dựng bộ dữ liệu khuôn mặt bị che khuất tổng hợp bằng mô hình Diffusion phục vụ bài toán phục hồi khuôn mặt.
- **Methodology**: AI-First Engineering, Modular Clean Architecture, Strict Reproducibility.

---

## 2. Coursework Problem Statement

- **Core Academic Bottleneck**:
  - Facial occlusion (masks, sunglasses, hands, held objects, organic apparel) severely degrades biometric face authentication, landmark localization, emotion recognition, and 3D facial modeling.
  - **The Paired Data Dilemma**: In real-world environments, collecting paired images—the exact same subject photographed with and without occlusion under identical micro-expression, head pose, and ambient illumination—is physically impossible.
  - Traditional synthetic augmentation techniques (such as 2D Cut-and-Paste or random erasing) generate severe visual artifacts, missing contact shadows, lacking skin indentation, and lacking 3D morphological deformation. De-occlusion networks trained on 2D pasted images suffer domain collapse when tested on real-world occlusions.
- **Proposed Solution**:
  - Implement a generative data factory utilizing a Latent Diffusion Model (Stable Diffusion Inpainting) conditioned on facial anatomy prompts to synthesize photo-realistic occlusions on clean high-resolution face portraits (CelebA-HQ / FFHQ).
  - Automatically extract pixel-accurate occlusion masks using cross-attention activation maps and boundary snapping (SAM) without manual labeling, yielding clean triplets: Clean Ground Truth Face ($I_{\text{clean}}$), Occluded Face ($I_{\text{occ}}$), and Occlusion Mask ($M$).
  - Train and benchmark deep facial de-occlusion architectures (Mask-Guided Inpainting / Fourier Convolution Networks / Dual-Stage Restoration) utilizing composite reconstruction, perceptual, and ArcFace identity-preservation losses.
  - Benchmark performance across standard restoration metrics (PSNR, SSIM, LPIPS, FID) and identity retention on unseen real-world benchmarks.

---

## 3. Operational Boundaries & Success Metrics

- **Primary Quantitative Targets**:
  - **Reconstruction PSNR**: $\ge 28.0\text{ dB}$ overall; $\ge 24.5\text{ dB}$ strictly within the occluded hole region ($M = 1$).
  - **Structural Similarity (SSIM)**: $\ge 0.88$ on validation triplets.
  - **Perceptual Quality (LPIPS)**: $\le 0.090$ (VGG backbone).
  - **Identity Preservation**: Cosine similarity $\ge 0.82$ between ArcFace deep embeddings of restored face $\hat{I}_{\text{clean}}$ and unoccluded ground truth $I_{\text{clean}}$.
  - **Auxiliary Mask Segmentation (mIoU)**: $\ge 0.72$ on benchmark occlusion masks (for dual-stage / auxiliary segmentation components).
  - **Inference Latency**: $\le 45\text{ ms}$ per $512 \times 512$ image on target GPU.
  - **Synthetic Data Quality**: CLIP score $\ge 0.28$ and Face Retention Rate $\ge 95\%$ on generated occluded faces.
- **Qualitative DoD (Definition of Done)**:
  - 100% Vectorized processing (zero unvectorized pixel loops).
  - Deterministic reproducibility via global seed configuration (`src.common.seed.seed_everything(seed)`).
  - Zero raw `print()` statements in source modules (enforcing `src.common.logger`).
  - Comprehensive technical documentation and coursework report (`docs/reports/REPORT.md`).
- **Explicit Non-Goals (Strictly Out of Scope)**:
  - *No Unconstrained General Scene Inpainting*: The system focuses strictly on human facial de-occlusion and biometric identity restoration, not general background scenery.
  - *No 3D Morphable Model (3DMM) Avatar Fitting*: The core output is photorealistic 2D restored portraits, not 3D mesh rendering.
  - *No Edge/Mobile Micro-Optimization*: Real-time edge NPU deployment is out of scope; GPU-based research pipelines ($\le 45\text{ ms}$) are targeted.
  - *No Client-Facing GUI / Mobile App*: Command-line scripts, Python modules, and reproducible evaluation suites are required.

---

## 4. Hardware & Compute Constraints

- **Execution Environment**: Linux workstation / Google Colab CUDA environment.
- **Compute Budget**: Single GPU (target VRAM constraints adhered to via AMP `fp16` and gradient accumulation).
- **Framework Target**: Python 3.10+, PyTorch 2.4+, Torchvision, OpenCV 4.8+, Hugging Face Diffusers & Transformers.

---

## 5. Required Deliverables

1. **Automated Diffusion Triplet Generation Pipeline**: Modular scripts in `src/pipelines/` capable of batch-synthesizing synchronized triplets $(I_{\text{clean}}, I_{\text{occ}}, M)$.
2. **Standardized Synthetic Triplet Dataset**: Curated dataset partitioned into `train`, `val`, and real-world evaluation sets.
3. **Reproducible Facial De-occlusion Codebase**: Modular dataset loaders, deep restoration models (Mask-Guided U-Net / Inpainting baseline), composite loss functions (Reconstruction + Perceptual + ArcFace Identity), and training pipelines.
4. **Experimental Results & Prediction Galleries**: Quantitative comparison tables against classical cut-and-paste baselines, ablation studies on identity loss, and visual de-occlusion galleries stored under `experiments/`.
5. **Technical Coursework Report**: Comprehensive academic report in `docs/reports/` and final presentation deck.
