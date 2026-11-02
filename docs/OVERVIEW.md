# Coursework Project Living Roadmap & Overview (`/docs/OVERVIEW.md`)

- **Motivation/Background**: Track overarching coursework progress across sequential milestones, linking active specifications, status trackers, and verifiable artifacts.
- **Purpose**: Serve as the dynamic project dashboard and roadmap for the UTH Computer Vision Coursework.
- **Overview Pipeline**: Milestone 0 (Scaffold & Governance) -> Milestone 1 (EDA & Data Pipeline) -> Milestone 2 (Baseline Modeling) -> Milestone 3 (Advanced Exploration & Optimization) -> Milestone 4 (Evaluation & Final Report).
- **Detailed Plan**: §1 Milestone Roadmap; §2 Phase Specifications Directory; §3 Active Progress Trackers; §4 Deliverables Checklist.
- **References**: `.agents/rules/PROJECT_ROADMAP_TEMPLATE.md`, `.agents/AI_FIRST_LIFECYCLE.md`, `docs/PURPOSE.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-02T07:22:00+07:00

---

## 1. Milestone Roadmap

| Milestone | Phase Name | Focus & Objectives | Target DoD | Status |
| :---: | :--- | :--- | :--- | :---: |
| **M0** | Scaffold & Governance | Setup repo structure, governance, configs, tests | Scaffolding verified, make test passing | **DONE** |
| **M1** | Data Pipeline & EDA | Dataset ingestion, transforms, validation & DataModule | DataModule verified, smoke tests pass | **PLANNED** |
| **M2** | Baseline Modeling | Build baseline CV model architecture, train pipeline | Training loop running, baseline metrics logged | **PLANNED** |
| **M3** | Advanced Optimization | Architectural tuning, augmentations, loss refinements | Target metric improvement verified | **PLANNED** |
| **M4** | Evaluation & Reports | Comprehensive benchmarking, HTML report, slide deck | Final report & demo ready for submission | **PLANNED** |

---

## 2. Phase Specifications Directory

- `docs/phases/01_DATA_PIPELINE.md`: Data ingestion, splitting, and augmentation specs. *(Pending M1)*
- `docs/phases/02_BASELINE_MODEL.md`: Baseline model architecture and training loop specs. *(Pending M2)*
- `docs/phases/03_ADVANCED_EXPERIMENTS.md`: Exploration of advanced backbones and loss functions. *(Pending M3)*
- `docs/phases/04_FINAL_EVALUATION.md`: Final benchmark protocol and report preparation. *(Pending M4)*

---

## 3. Active Progress Trackers

- `docs/progress/00_SCAFFOLD_STATUS.md`: Initial repository initialization tracker.
- `docs/progress/01_DATA_PIPELINE_STATUS.md`: Active task tracker for data phase.

---

## 4. Deliverables Checklist

- [x] Repository scaffolding, CI tooling, Makefile & AI Governance.
- [ ] Exploratory Jupyter Notebook with initial dataset analysis.
- [ ] Modular Python package (`src/`) implementing data, model, and training pipelines.
- [ ] Reproducible experiment checkpoints under `experiments/checkpoints/`.
- [ ] Comprehensive Markdown technical report and interactive HTML visual report under `docs/reports/`.
