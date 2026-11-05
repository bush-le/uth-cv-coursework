# Coursework Project Living Roadmap & Overview (`docs/OVERVIEW.md`)

- **Motivation/Background**: Track overarching coursework progress across sequential milestones, linking active specifications, status trackers, and verifiable artifacts.
- **Purpose**: Serve as the dynamic project dashboard and roadmap for the UTH Computer Vision Coursework on "Diffusion-Based Synthetic Occluded Face Dataset Generation for Facial De-occlusion".
- **Overview Pipeline**: Milestone 0 (Scaffold & Governance) -> Milestone 1 (EDA & Triplet Pipeline) -> Milestone 2 (De-occlusion Baseline) -> Milestone 3 (Identity-Preserving Optimization) -> Milestone 4 (Evaluation & Final Report).
- **Detailed Plan**: §1 Milestone Roadmap; §2 Phase Specifications Directory; §3 Active Progress Trackers; §4 Deliverables Checklist.
- **References**: `.agents/rules/PROJECT_ROADMAP_TEMPLATE.md`, `.agents/AI_FIRST_LIFECYCLE.md`, `docs/PURPOSE.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-05T18:30:00+07:00

---

## 1. Milestone Roadmap

| Milestone | Phase Name | Focus & Objectives | Target DoD | Status |
| :---: | :--- | :--- | :--- | :---: |
| **M0** | Scaffold & Governance | Setup repo structure, governance, configs, tests | Scaffolding verified, make test passing | **DONE** |
| **M1** | Triplet Data Pipeline & EDA | Clean image ingestion, diffusion occlusion generation, SAM mask snapping, TripletDataModule | TripletDataModule verified, smoke tests pass | **PLANNED** |
| **M2** | De-occlusion Baseline | Build baseline facial restoration network, composite loss (L1 + Perceptual), train pipeline | Training loop running, baseline PSNR/SSIM logged | **PLANNED** |
| **M3** | Advanced Identity Optimization | ArcFace identity loss integration, boundary mask conditioning, dual-stage refinement | Identity retention $\ge 0.82$, PSNR $\ge 28\text{ dB}$ | **PLANNED** |
| **M4** | Evaluation & Reports | Comprehensive benchmarking (PSNR, SSIM, LPIPS, FID, ID-Sim), visual galleries, report | Final report & demo ready for submission | **PLANNED** |

---

## 2. Phase Specifications Directory

- `docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md`: Synthetic triplet generation, SAM mask snapping, and quality gates.
- `docs/phases/PHASE_02_DEOCCLUSION_BASELINE_SPEC.md`: Facial de-occlusion architecture, composite loss, and training loop specs.
- `docs/phases/03_ADVANCED_EXPERIMENTS.md`: Identity-preserving loss formulation, ablation studies, and architectural tuning. *(Pending M3)*
- `docs/phases/04_FINAL_EVALUATION.md`: Final benchmark protocol and report preparation. *(Pending M4)*

---

## 3. Active Progress Trackers

- `docs/progress/PROJECT_ROADMAP.md`: Master 9-phase AI-First lifecycle tracker.
- `docs/progress/00_SCAFFOLD_STATUS.md`: Initial repository initialization tracker.
- `docs/progress/01_DATA_PIPELINE_STATUS.md`: Active task tracker for data phase.

---

## 4. Deliverables Checklist

- [x] Repository scaffolding, CI tooling, Makefile & AI Governance.
- [ ] Exploratory Jupyter Notebook with initial dataset analysis and triplet visualizations.
- [ ] Modular Python package (`src/`) implementing triplet data, facial restoration model, and training pipelines.
- [ ] Reproducible experiment checkpoints under `experiments/checkpoints/` (`best_model.pt`, `last_model.pt`).
- [ ] Comprehensive Markdown technical report and interactive HTML visual report under `docs/reports/`.
