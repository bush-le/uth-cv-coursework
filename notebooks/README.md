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

---

## 4. Danh Mục Bộ 8 Coursework Notebooks Chuẩn Mực (`notebooks/coursework/`)

| Notebook File | Mô Tả & Nội Dung Chính | Trạng Thái Thực Thi |
| :--- | :--- | :---: |
| **[`00_master_deepfake_defense_coursework.ipynb`](coursework/00_master_deepfake_defense_coursework.ipynb)** | **Master Showcase Lecture toàn diện từ A-Z (67 cells): Đa tạp ảnh thật, SD kép, 213 mẫu khó, DINOv2, OOD 6k, Universal 20.4k, Live Inspector** | **Verified (0 blank cells)** |
| **[`01_eda_faces_demographics.ipynb`](coursework/01_eda_faces_demographics.ipynb)** | Khảo sát 10.000 ảnh thật FFHQ/CelebA-HQ, phân bố tuổi/giới tính/kính mắt, góc xoay đầu 3D, phổ màu da | **Verified (0 blank cells)** |
| **[`02_sd_generation_and_curation.ipynb`](coursework/02_sd_generation_and_curation.ipynb)** | Phân tích sinh ảnh SD 1.5 & RealisticVision v5.1, làm mềm biên Gaussian $\sigma=4.0$, phễu lọc 3 tầng | **Verified (0 blank cells)** |
| **[`03_hard_samples_and_benchmark_comparison.ipynb`](coursework/03_hard_samples_and_benchmark_comparison.ipynb)** | Khai thác 213 mẫu cực khó, t-SNE Manifold Intrusion, phân tích dị tật sinh học mắt/tai/da/răng | **Verified (0 blank cells)** |
| **[`04_active_learning_production_visualizer.ipynb`](coursework/04_active_learning_production_visualizer.ipynb)** | Tiến hóa $M_0 \to M_1 \to M_2 \to M_2^*$, sơ đồ khối DINOv2, động học hội tụ, ROC, ma trận nhầm lẫn | **Verified (0 blank cells)** |
| **[`05_synthetic_face_6k_realimage_benchmark.ipynb`](coursework/05_synthetic_face_6k_realimage_benchmark.ipynb)** | Thử thách khái quát hóa ngoại lai trên 6.000 ảnh siêu thực, bước nhảy độ chính xác $68.4\% \to 89.1\%$ | **Verified (0 blank cells)** |
| **[`06_specialization_vs_generalization_synthesis.ipynb`](coursework/06_specialization_vs_generalization_synthesis.ipynb)** | Phân tích cân bằng Pareto giữa độ chuyên biệt hóa (In-Distribution) và độ tổng quát hóa (OOD) | **Verified (0 blank cells)** |
| **[`07_universal_deepfake_benchmark.ipynb`](coursework/07_universal_deepfake_benchmark.ipynb)** | Đánh giá toàn cầu trên 20.400 ảnh, 34 thuật toán sinh, 5 họ mô hình (GAN, Diffusion, Swap, Inpaint, Talking Head) | **Verified (0 blank cells)** |

| `3` | Visual Sanity Check | Display augmented batch samples | `[`src/common/visualizer.py`](../../src/common/visualizer.py)` |
