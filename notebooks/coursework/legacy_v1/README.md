# Legacy Coursework Notebooks Archive (v1)

> **Archival Notice**: The notebooks in this directory represent the initial exploration, experimental prototypes, and preliminary benchmark notebooks (v1) for Coursework Topic 11 (Deepfake Detection & Defense). They have been archived to preserve provenance and experimental audit trails while transitioning to the streamlined, unified 4-notebook suite in `notebooks/coursework/`.

---

## 1. Archived Notebooks Inventory

This directory contains the 10 original experimental notebooks:

| Legacy File | Original Scope & Role |
| :--- | :--- |
| `00_master_deepfake_defense_coursework.ipynb` | Initial monolithic master notebook containing preliminary experiments and summary charts. |
| `01_eda_faces_demographics.ipynb` | Initial EDA on face demographics, quality metrics, and distribution balance. |
| `02_sd_generation_and_curation.ipynb` | Exploration of Stable Diffusion synthetic face generation and initial curation filters. |
| `03_hard_samples_and_benchmark_comparison.ipynb` | Initial hard sample mining heuristics and baseline detector comparisons. |
| `04_active_learning_production_visualizer.ipynb` | Prototype visualizer for uncertainty-based active learning selection. |
| `05_synthetic_face_6k_realimage_benchmark.ipynb` | Benchmark evaluation across the 6k synthetic face dataset. |
| `06_specialization_vs_generalization_synthesis.ipynb` | Preliminary cross-generator generalization vs. specialization studies. |
| `07_universal_deepfake_benchmark.ipynb` | Multi-generator stress testing and out-of-distribution evaluation runs. |
| `08_champion_m3_dual_engine_quad_vault_benchmark.ipynb` | Early evaluation of the M3 ensemble architecture across quad-vault splits. |
| `generator_realistic_vision_kaggle.ipynb` | Kaggle notebook for Realistic Vision v5.1 batch generation workflow. |

---

## 2. Rationale for Archival & Redesign

The legacy v1 notebook suite served well during exploratory research, but presented several challenges for final submission and reproducible demonstration:

1. **Volume & Bloat**: Monolithic base64 image outputs and checkpoint dumps resulted in an aggregate size of ~49.4 MB across 10 files.
2. **Fragmentation**: Narrative and analytical workflows were dispersed across 10 disconnected files with overlapping scopes.
3. **Execution Inconsistency**: Varied conventions for data loading, device management, and visualization wrappers made full end-to-end execution difficult to maintain.

---

## 3. Transition to Unified 4-Part Coursework Suite

The coursework presentation is now consolidated into a unified, high-density, interactive 4-notebook suite located at `notebooks/coursework/`:

1. **`00_master_deepfake_defense_showcase.ipynb`**:
   - Executive dashboard, master technical narrative, Champion M3 architecture comparison, and cross-vault defense summary.
2. **`01_diffusion_dataset_and_curation.ipynb`**:
   - Data pipeline, demographic EDA, Stable Diffusion generation, and multi-tier quality curation (aesthetic, artifact, blur).
3. **`02_hard_samples_and_active_learning.ipynb`**:
   - Hard sample mining (spectral, spatial, artifact-guided), Active Learning acquisition loops, and retraining gains.
4. **`03_champion_m3_and_universal_benchmark.ipynb`**:
   - Comprehensive benchmarks, Champion M3 Dual-Engine evaluation, ROC-AUC / Confusion Matrices across 4 evaluation vaults, and failure mode analysis.

---

## 4. Usage Guidelines

- **For Examination & Grading**: Please refer to the primary notebooks in `notebooks/coursework/` (`00_` through `03_`).
- **For Historical Provenance**: The code cells and raw outputs in this `legacy_v1/` directory remain intact for audit and reference purposes.
