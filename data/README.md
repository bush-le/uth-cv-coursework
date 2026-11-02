# Data Layer Documentation (`/data`)

- **Motivation/Background**: Computer vision pipelines require strict data immutability, clear separation of raw data vs. processed artifacts, and protection against accidental commits of heavy datasets.
- **Purpose**: Define the 4-tier data architecture, dataset preparation standards, and guidelines for managing raw, interim, processed, and sample datasets.
- **Overview Pipeline**: Ingestion of raw images/annotations -> Validation & cleaning (interim) -> Standardized tensor/parquet formatting (processed) -> Model training.
- **Detailed Plan**: §1 Tiered Data Architecture; §2 Directory Layout; §3 Data Invariants & Rules; §4 Sample Data Guidelines; §5 Data Dictionary.
- **References**: `.agents/rules/FOLDER_STRUCTURE.md`, `.agents/rules/MD_CONVENTION.md`, `.agents/AI_REFERENCE.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-02T07:22:00+07:00

---

## 1. Tiered Data Architecture

The data layer follows the Medallion-inspired 4-tier separation:

| Tier | Directory | Git Status | Description & Rules |
| :--- | :--- | :--- | :--- |
| **Raw** | `data/raw/` | **Ignored** | Read-only canonical source data as downloaded/collected. Never edit raw files in-place. |
| **Interim** | `data/interim/` | **Ignored** | Intermediate data transformed during pipeline steps (e.g. resized crops, filtered frames). |
| **Processed** | `data/processed/` | **Ignored** | Final standardized splits, normalized datasets, and cached tensors ready for DataLoaders. |
| **Samples** | `data/samples/` | **Tracked** | Minimal synthetic or test images (< 5MB total) strictly used for unit tests and smoke tests. |

---

## 2. Directory Layout

```text
data/
├── raw/                      # Canonical untouched input images and ground truth annotations
│   └── .gitkeep
├── interim/                  # Filtered, converted, or cropped intermediate files
│   └── .gitkeep
├── processed/                # Training-ready dataset splits (train/val/test) and cached tensors
│   └── .gitkeep
├── samples/                  # Lightweight sample images for CI/CD and smoke tests
│   └── .gitkeep
└── README.md                 # This governance document
```

---

## 3. Data Invariants & Rules

1. **Immutability of `raw/`**: Raw data must never be overwritten, renamed, or modified by any preprocessing script.
2. **Deterministic Preprocessing**: Any file in `interim/` and `processed/` must be 100% reproducible via scripts in `src/data/` using explicit seeds.
3. **No Heavy Binaries in Git**: All contents of `raw/`, `interim/`, and `processed/` are gitignored. Only `.gitkeep` is tracked.
4. **Color Space Verification**: Images stored or loaded must have their color format explicitly known (`BGR` for OpenCV, `RGB` for Pillow / PyTorch).
