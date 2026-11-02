# Project Purpose & Coursework Requirements

- **Motivation/Background**: Canonical source of truth defining the core objectives, problem statement, evaluation criteria, and boundaries for the UTH Computer Vision Coursework.
- **Purpose**: Provide locked project scope and success metrics so that all subsequent engineering phases remain aligned and drift-free.
- **Overview Pipeline**: Problem analysis -> Definition of quantitative targets & constraints -> Milestone execution -> Coursework submission.
- **Detailed Plan**: §1 Context & Academic Scope; §2 Coursework Problem Statement; §3 Operational Boundaries & Success Metrics; §4 Hardware & Compute Constraints; §5 Required Deliverables.
- **References**: `.agents/rules/MD_CONVENTION.md`, `.agents/AI_FIRST_LIFECYCLE.md`, `docs/README.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-02T07:22:00+07:00

---

## 1. Context & Academic Scope

- **Course**: Computer Vision (Thị giác Máy tính) - Ho Chi Minh City University of Transport (UTH).
- **Project Role**: Major Coursework Project / Đồ án môn học.
- **Methodology**: AI-First Engineering, Modular Clean Architecture, Strict Reproducibility.

---

## 2. Coursework Problem Statement

<!-- Paste or define the exact coursework topic / challenge brief below -->
- **Topic Name**: [Tên đề tài đồ án môn học Thị giác máy tính]
- **Target Task**: [e.g. Object Detection, Semantic Segmentation, Multi-Object Tracking, Image Retrieval, Face Recognition]
- **Dataset**: [e.g. Custom UTH Dataset, COCO, Pascal VOC, Cityscapes, or domain-specific imagery]

---

## 3. Operational Boundaries & Success Metrics

- **Primary Quantitative Targets**:
  - Primary Metric: [e.g. mAP@50:95 >= 0.65, IoU >= 0.75, Top-1 Accuracy >= 90%]
  - Inference Latency: [e.g. <= 30ms per frame on target hardware / GPU]
  - Memory Footprint: [e.g. VRAM peak <= 6GB during batch inference]
- **Qualitative DoD (Definition of Done)**:
  - 100% Vectorized processing (zero unvectorized pixel loops).
  - Deterministic reproducibility via global seed configuration.
  - Comprehensive technical report (`REPORT.md` paired with interactive `REPORT.html`).
- **Explicit Non-Goals**:
  - Out-of-scope model architectures or tasks explicitly avoided to prevent scope bloat.

---

## 4. Hardware & Compute Constraints

- **Execution Environment**: Linux / CUDA-capable workstation.
- **Compute Budget**: Single GPU (target VRAM constraints adhered to via AMP fp16).
- **Framework Target**: Python 3.10+, PyTorch 2.4+, OpenCV 4.8+.

---

## 5. Required Deliverables

1. Well-structured, linted, tested codebase in `src/`.
2. Fully reproducible training and evaluation scripts in `scripts/` and `src/pipelines/`.
3. Model checkpoints and prediction galleries stored under `experiments/`.
4. Technical coursework report and presentation assets under `docs/`.
