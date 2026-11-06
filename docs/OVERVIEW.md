# Coursework Project Living Roadmap & Overview (`docs/OVERVIEW.md`)

- **Motivation/Background**: Track overarching coursework progress across sequential milestones, linking active specifications, status trackers, and verifiable artifacts.
- **Purpose**: Serve as the dynamic project dashboard and roadmap for the UTH Computer Vision Coursework on "Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection" (Topic 11).
- **Overview Pipeline**: Milestone 0 (Scaffold & Governance) -> Milestone 1 (Image Data Contracts & Manifest) -> Milestone 2 (Stable Diffusion Generation Engine) -> Milestone 3 (Quality & Biometric Curation) -> Milestone 4 (Detector Benchmark & Final Report).
- **Detailed Plan**: §1 Milestone Roadmap; §2 Phase Specifications Directory; §3 Active Progress Trackers; §4 Deliverables Checklist.
- **References**: `docs/PURPOSE.md`, `docs/progress/PROJECT_ROADMAP.md`, `GEMINI.md`, [Hugging Face CV Course Unit 10](https://huggingface.co/learn/computer-vision-course/unit10/datagen-diffusion).
- **Created**: 2026-11-07T00:05:00+07:00
- **Last Updated**: 2026-11-07T00:05:00+07:00

---

## 1. Milestone Roadmap

| Milestone | Phase Name | Focus & Objectives | Target DoD | Status |
| :---: | :--- | :--- | :--- | :---: |
| **M0** | Scaffold & Governance | Setup repo structure, governance, configs, tests | Scaffolding verified, test suite passing | **DONE** |
| **M1** | 2D Image Contracts & Schema | Image tensor schema `(B, 3, H, W)`, manifest JSONL, `DeepfakeImageDataModule` | DataModule verified, tests pass | **IN PROGRESS** |
| **M2** | Stable Diffusion Engine | Text2Img, Inpaint, and Img2Img generation pipelines with deterministic mock mode | Generation pipeline producing valid images | **PLANNED** |
| **M3** | Quality Curation Gates | Laplacian sharpness, CLIP score proxy, face detection retention filters | Automated curation gates pruning low-quality images | **PLANNED** |
| **M4** | Detector Benchmark & Report | Downstream Deepfake Image Detector training, ROC-AUC evaluation, academic report | Benchmark tables & final report ready | **PLANNED** |

---

## 2. Phase Specifications Directory

- `docs/plans/2026-11-06-stable-diffusion-deepfake-image-dataset.md`: Master refactoring plan.
- `docs/PURPOSE.md`: Core coursework purpose and quantitative success metrics.
- `configs/dataset/deepfake_image.yaml`: 2D image dataset specifications and tensor contracts.
- `configs/generation/stable_diffusion.yaml`: Stable Diffusion generation parameters.
- `configs/detection/detector_baseline.yaml`: Deepfake detector baseline configuration.

---

## 3. Active Progress Trackers

- `docs/progress/PROJECT_ROADMAP.md`: Master AI-First lifecycle tracker.
- `docs/plans/2026-11-06-stable-diffusion-deepfake-image-dataset.ledger.md`: Active task execution ledger.

---

## 4. Deliverables Checklist

- [x] Repository scaffolding, CI tooling, Makefile & AI Governance.
- [ ] 2D deepfake image schema, metadata manifest, and DataModule.
- [ ] Stable Diffusion synthetic deepfake generation engine and pipeline.
- [ ] Multi-modal quality curation gates (sharpness, CLIP alignment, face detection).
- [ ] Curated deepfake image dataset with standardized JSONL manifest.
- [ ] Downstream 2D deepfake image detector and evaluation benchmark.
- [ ] Comprehensive Markdown technical report and visual prediction gallery.
