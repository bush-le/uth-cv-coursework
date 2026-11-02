# Jupyter Notebooks Guidelines (`/notebooks`)

- **Motivation/Background**: In data science & deep learning projects, notebooks frequently become unmaintainable when long training loops, business logic, and unsanitized states are executed ad-hoc without structure.
- **Purpose**: Enforce the "Exploration vs. Script-Only Training" rule, establish notebook headers, and provide standards for interactive analysis.
- **Overview Pipeline**: Data exploration & proof-of-concept in `notebooks/` -> Formalization into modular Python packages in `src/` -> Script-based execution in `scripts/` -> Visual analysis & post-hoc reporting in `notebooks/`.
- **Detailed Plan**: §1 Golden Rules; §2 Directory Layout; §3 Required Notebook Cell Architecture (Cell 0 & Cell 1 Roadmap Table); §4 Clean Git Hygiene.
- **References**: `.agents/rules/NOTEBOOK_HEADER_CONVENTION.md`, `.agents/rules/LOGGING_CHECKPOINT_RULES.md`, `.agents/AI_REFERENCE.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-02T07:22:00+07:00

---

## 1. Golden Rules for Notebooks

1. **NO Training Loops in Notebooks**: All heavy model training must run via Python scripts in `src/pipelines/` or `scripts/`. Notebooks are strictly for:
   - Exploratory Data Analysis (EDA).
   - Sanity checks of dataset transforms.
   - Qualitative visualization of model predictions.
   - Post-training metrics inspection and confusion matrices.
2. **Notebook vs. Report Separation**:
   - **Notebooks (`notebooks/**/*.ipynb`)**: DO NOT contain workflow diagrams (no Mermaid/SVG pipeline flowcharts) or academic references at the end. They start with Cell 0 (Objective) and Cell 1 (4-column Roadmap Table), followed by executable code cells.
   - **Reports (`docs/**/*.md` & `docs/**/*.html`)**: Contain comprehensive workflow diagrams, detailed analytical text, quantitative tables, and the 4-tier bibliography.
3. **Execution State Hygiene**: Clear all cell outputs before pushing to Git, or use pre-commit stripping hooks.

---

## 2. Directory Layout

```text
notebooks/
├── exploratory/              # Initial data inspection, distribution plots, EDA
│   └── .gitkeep
├── coursework/               # Coursework qualitative verification, visual demos
│   └── .gitkeep
└── README.md                 # This governance document
```

---

## 3. Required Notebook Architecture

Every notebook must follow this cell structure:

- **Cell 0 [Markdown]**: Title, Objective, and Prerequisites.
- **Cell 1 [Markdown]**: Roadmap Table (4 columns) & Quick Documentation Links:

| Step | Description | What it does | Import path |
| :---: | :--- | :--- | :--- |
| `1` | Environment & Imports | Initialize environment, set random seed, verify GPU | `—` |
| `2` | Load DataModule | Instantiate CVDataModule with target transforms | `[`src/data/datamodule.py`](../../src/data/datamodule.py)` |
| `3` | Visual Sanity Check | Display augmented batch samples | `[`src/common/visualizer.py`](../../src/common/visualizer.py)` |
