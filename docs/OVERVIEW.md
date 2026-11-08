# Coursework Project Living Roadmap & Overview (`docs/OVERVIEW.md`)

- **Motivation/Background**: Track overarching coursework progress across sequential milestones for the closed-loop Stable Diffusion + `dinoViT` Hard-Sample Mining architecture.
- **Purpose**: Serve as the dynamic project dashboard and roadmap for the UTH Computer Vision Coursework on "Stable Diffusion-Based Synthetic Facial Dataset Generation with DINO ViT Hard-Sample Mining for Deepfake Detection" (Topic 11).
- **Overview Pipeline**: Milestone 0 (Scaffold & Clean Governance) -> Milestone 1 (2D Image & Hard-Sample Data Contracts) -> Milestone 2 (Phase 1: Stable Diffusion Generation & Curation) -> Milestone 3 (Phase 2: Baseline `dinoViT` & Bypass Hard-Sample Mining) -> Milestone 4 (Phase 3: Fine-Tuning on `Baseline + Syn + Hard` & Proof of Utility).
- **Detailed Plan**: §1 Milestone Roadmap; §2 Active Specifications & Configurations; §3 Deliverables Checklist.
- **References**: `docs/PURPOSE.md`, `docs/progress/PROJECT_ROADMAP.md`, `GEMINI.md`.
- **Created**: 2026-11-07T00:05:00+07:00
- **Last Updated**: 2026-11-08T17:45:00+07:00

---

## 1. Milestone Roadmap

| Milestone | Phase Name | Focus & Objectives | Target DoD | Status |
| :---: | :--- | :--- | :--- | :---: |
| **M0** | Clean Architecture & Refactor | Remove legacy occlusion/video code, lock 3-phase closed-loop governance | Clean workspace, zero legacy drift | **DONE** |
| **M1** | 2D Data & Hard-Sample Schema | `ImageMetadata` (`is_hard_sample`, `baseline_score`), Identity-Disjoint `DeepfakeImageDataModule` | DataModule & oversampling verified | **DONE** |
| **M2** | Phase 1: SD Data Factory & Gate | Text2Img, Inpaint, Img2Img synthesis + Laplacian/CLIP/Face Curation Gate | Curated synthetic DF manifest ready | **DONE** |
| **M3** | Phase 2: `dinoViT` & Hard Mining | `DinoViTBackbone` baseline evaluation, `Reject` vs `Bypass` $\rightarrow$ `hard_samples_manifest.jsonl` | Hard-sample mining pipeline verified | **DONE** |
| **M4** | Phase 3: Fine-Tune & Benchmark | Fine-tune `dinoViT` on `Baseline + Syn + Hard Samples`, prove utility via Before/After metrics | All unit tests & slide deck verified | **DONE** |

---

## 2. Active Specifications & Configurations

- `docs/PURPOSE.md`: Core coursework purpose, closed-loop design, and quantitative KPIs.
- `docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md`: Phase 1 Stable Diffusion generation and curation specification.
- `docs/phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md`: Phase 2 `dinoViT` baseline & hard-sample mining specification.
- `docs/phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md`: Phase 3 fine-tuning, forensic audit, and comparative benchmark matrix.
- `configs/dataset/deepfake_image.yaml`: 2D image & hard-sample dataset specifications.
- `configs/generation/stable_diffusion.yaml`: Stable Diffusion generation & curation thresholds.
- `configs/detection/detector_baseline.yaml`: `dinoViT` detector, bypass threshold, and fine-tuning configuration.

---

## 3. Deliverables Checklist

- [x] Clean repository structure with all legacy occlusion/segmentation files removed.
- [x] 2D deepfake image schema with `is_hard_sample` & `baseline_score` tracking and Identity-Disjoint DataModule.
- [x] Phase 1: Stable Diffusion synthetic deepfake generation engine and multi-modal curation gate.
- [x] Phase 2: Self-supervised `DinoViTBackbone` baseline detector and automated `mine_hard_samples.py` pipeline.
- [x] Phase 3: Weighted hard-sample fine-tuning (`train_image_detector.py`) and comparative evaluation suite (`evaluate_image_benchmark.py`).
- [x] Updated technical reports and 11-slide interactive HTML/Markdown presentation deck (`docs/reports/`).
