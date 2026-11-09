# ZERO_HARDCODE_RULE.md — Config-Driven Dataset Factory Charter

- **Motivation/Background**: Hardcoding specific sample counts (e.g. 4,000, 3,000), fixed decision thresholds (e.g. 0.45, 0.50, 0.85), or specific commercial model repository IDs into codebase defaults or research prose severely degrades architectural modularity, prevents hyperparameter exploration, and induces silent test fragility.
- **Purpose**: Mandate that all operational paths, generation settings, curation gates, hardness evaluations, and benchmark split compositions are centrally governed by `configs/dataset_factory.yaml` and dynamic reference calibration.
- **Overview Pipeline**: `configs/dataset_factory.yaml` -> `src/common/config.py` (`resolve`) -> Dynamic execution -> Auto-generated statistics exported to `paths.stats_file`.
- **Detailed Plan**: §1 Central Configuration Invariant; §2 Dynamic Calibration Invariant; §3 Zero Hardcoded Model IDs; §4 Documentation & Presentation Contracts.
- **References**: `GEMINI.md`, `configs/dataset_factory.yaml`, `src/common/config.py`, `src/data/splits.py`.
- **Created**: 2026-11-09T15:15:00+07:00
- **Last Updated**: 2026-11-09T15:15:00+07:00

---

## 1. ⚙️ Central Configuration Invariant

1. **Single Source of Truth**: All pipeline execution entrypoints (`generate_deepfake_images.py`, `curate_image_dataset.py`, `mine_hard_samples.py`, `organize_topic_dataset.py`, `build_visual_gallery.py`) must read parameters from `configs/dataset_factory.yaml`.
2. **Resolution Precedence**: Explicit command-line arguments override YAML configuration, which in turn overrides library defaults via `resolve(cli_val, cfg, dotted_key)`.
3. **No Phantom Paths**: Fallbacks to legacy paths (e.g. `data/processed/` or `data/01_raw_faces/`) are strictly forbidden. If an expected manifest is missing, raise a descriptive `FileNotFoundError`.

---

## 2. 📐 Dynamic Calibration Invariant

1. **Empirical Quantiles Over Fixed Cutoffs**:
   - Quality gates (such as Laplacian sharpness) must be empirically calibrated against real reference face distributions via `calibrate_threshold(real_sharpness, quantile=reference_quantile_sharpness)` rather than arbitrary fixed floats.
   - Hardness thresholds ($\tau$) must be derived as quantiles of real-to-real leave-one-out manifold similarities (`real_reference_scores`), ensuring the boundary reflects true data geometry.
2. **Recorded Provenance**: Every generated or mined record must log its exact parameters in `generation` and `scores` dictionaries within `ImageMetadata`.

---

## 3. 🧠 Model-Agnostic Declarations

1. **Repository IDs in Config Only**: Code signatures and docstrings must not assume or hardcode specific Hugging Face repository strings (e.g. `runwayml/...` or `facebook/...`). Instead, refer to them generically (e.g. "latent diffusion model `generation.model_id`" or "vision foundation backbone `hardness.backbone_id`").
2. **Swappable Architectures**: Backbones and diffusion pipelines must remain easily replaceable via simple configuration updates without refactoring pipeline logic.

---

## 4. 📝 Documentation & Presentation Standards

1. **Dataset-First Narrative**: Research documentation must emphasize the deliverable: the **Multimodal Synthetic Facial Dataset & Hard-Sample Adversarial Benchmark**. Downstream detector training is framed strictly as an optional proof-of-utility audit.
2. **Dynamic Stats References**: Numbers detailing dataset volume, curation retention rates, and hard-sample proportions must link to or read from `paths.stats_file` rather than being hardcoded into static Markdown documents.
