# AGENTS.md — Global AI Agent Operating Charter & Workspace Constraints

- **Motivation/Background**: Establishes universal binding principles, quality gates, and domain invariants for AI agents (Gemini, Antigravity, Claude, etc.) operating in the `uth-cv-coursework` workspace.
- **Purpose**: Serve as the first-contact constitutional constraint file loaded automatically at the root of the workspace.
- **Overview Pipeline**: Ingestion of root constraints -> Alignment with `.agents/` second brain -> Enforcement during planning, implementation, verification, and commits.
- **Detailed Plan**: §1 Cognitive Alignment & Authority; §2 Non-Negotiable Operational Principles; §3 Computer Vision Domain Invariants; §4 Code Quality & Packaging Standards; §5 Essential Commands; §6 Contract-First Output Format.
- **References**: `.agents/README.md`, `.agents/rules/`, `.agents/AI_REFERENCE.md`, `docs/PURPOSE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T19:50:00+07:00

---

## 1. 🧠 Cognitive Alignment & Second Brain

When operating in this workspace, you are acting as a **Principal Computer Vision / MLOps Engineer & Pair Programmer**:

1. **Second Brain Anchor (`.agents/`)**: The complete technical blueprint, lifecycle specifications (9 macro phases), pair loops (7 micro steps), and templates live in [`.agents/`](.agents/README.md).
2. **Cold-Start Protocol**: Always verify your actions against the rules in [`.agents/rules/`](.agents/rules/) and domain invariants in [`.agents/AI_REFERENCE.md`](.agents/AI_REFERENCE.md).
3. **No Guesswork (Zero-Assumption)**: Never invent or assume problem requirements, student IDs, author names, or dataset splits. Read [docs/PURPOSE.md](docs/PURPOSE.md) for official specifications.

---

## 2. ⚡ Non-Negotiable Operational Principles

1. **Strict Modular Architecture & Script vs. Notebook Separation**:
   - **Modular Clean Code (`src/`)**: All domain logic, datasets, transforms, model definitions, metrics, and visualization utilities must be written as modular, reusable, type-annotated Python files in `src/`.
   - **Script-Only Training (`src/pipelines/` & `scripts/`)**:
     - **NEVER** run training loops, gradient updates, or model optimization inside Jupyter Notebooks (`notebooks/`).
     - All training and evaluation pipelines must be executable, resumable Python scripts in [`src/pipelines/`](src/pipelines/) or [`scripts/`](scripts/) that accept YAML configs and save full-state checkpoints (`best_model.pt`, `last_model.pt`).
   - **Notebooks as Visualization & Inspection Clients Only (`notebooks/`)**:
     - Notebooks are strictly reserved for: Exploratory Data Analysis (EDA), sanity checks on transforms, and visual prediction/error analysis.
     - **No monolithic code inside notebooks**: Notebooks must NEVER implement raw model classes, training loops, or complex data processing from scratch. They MUST import and call modular functions and classes from `src/` (e.g. `from src.data import FacialOcclusionDataset`, `from src.models import build_model`, `from src.common.visualizer import plot_prediction_gallery`).
2. **Strict File Separation (Agents vs Docs)**:
   - [`.agents/`](.agents/): The immutable constitution. Contains ONLY reusable rules and document templates. Never place project-specific logs or phase deliverables here.
   - [`docs/`](docs/): The evolving research memory. Phase specs go to `docs/phases/`, task tracking to `docs/progress/`, and reports to `docs/reports/`.
3. **Markdown Standards & Privacy Invariants**:
   - All `.md` files in `docs/` and `.agents/` must be named in UPPERCASE (`UPPER_SNAKE_CASE.md` or `UPPERCASE.md`).
   - Every `.md` file must begin with the standard **7-field header** (Motivation, Purpose, Overview Pipeline, Detailed Plan, References, Created, Last Updated).
   - **No absolute system paths**: Never include absolute system paths (e.g. `/home/user/...` or `C:\...`). Use relative markdown links only (e.g. `[config](configs/config.yaml)`).
4. **Zero-Mock & Neural Checkpoint Integrity**:
   - **NEVER** use mock data, noise shapes, or uninitialized random-weight neural heads for production inference, evaluation, or hard-sample mining.
   - Every mining and evaluation pipeline (`mine_hard_samples.py`, `evaluate.py`) MUST require and verify a real trained checkpoint file (`model_state_dict`). Unverified random inference is strictly forbidden. Detailed protocol in [`.agents/rules/ZERO_MOCK_DATA_RULE.md`](.agents/rules/ZERO_MOCK_DATA_RULE.md).
5. **Continuous Checkpoint & Backup Invariant**:
   - Bất cứ khi nào hoàn thành một đơn vị tác vụ và chạy đạt `make test-smoke`, Agent bắt buộc phải thực hiện Git commit nguyên tử ngay lập tức, không để dồn diff lớn nhằm bảo toàn 100% tính vẹn toàn mã nguồn.

---

## 3. 👁️ 5 Computer Vision & Deep Learning Domain Invariants

Any image processing or deep learning code MUST adhere to these 5 invariants:

1. **Tensor Shape Contracts**:
   - Every function processing images/tensors must document explicit shapes in docstrings: `(B, C, H, W)` or `(C, H, W)` for PyTorch; `(H, W, C)` or `(H, W)` for OpenCV/NumPy.
2. **Color Space Awareness**:
   - OpenCV reads and writes in **BGR**.
   - PIL, Matplotlib, and Torchvision expect **RGB**.
   - Explicitly convert color spaces using `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` when bridging libraries. Never allow silent color distortion.
3. **Bounding Box Formats**:
   - Never assume bounding box coordinate conventions. Always state and convert explicitly:
     - `xyxy`: `[x_min, y_min, x_max, y_max]` (Pascal VOC pixel format).
     - `xywh`: `[x_min, y_min, width, height]` (COCO pixel format).
     - `cxcywh`: `[center_x, center_y, width, height]` (normalized [0, 1] YOLO format).
4. **Strict Reproducibility**:
   - Always seed all random number generators at the start of scripts and tests using `src.common.seed.seed_everything(seed)`.
5. **GPU Hygiene & Clean Memory**:
   - Device-agnostic code using `device = torch.device(...)` configured via `configs/config.yaml`.
   - Use `torch.cuda.amp.autocast()` for training performance and VRAM efficiency.
   - Always wrap evaluation and inference with `model.eval()` and `torch.no_grad()`.

---

## 4. 📐 Code Quality & Clean Architecture

- **Clean Architecture (`src/`)**:
  - `src/common/`: Shared utilities (logger, seed, metrics, visualizer).
  - `src/data/`: Transforms, datasets, and DataModules.
  - `src/models/`: Model builders, classical filters, and deep architectures.
  - `src/pipelines/`: Execution entry points (`train.py`, `evaluate.py`, `infer.py`).
- **Zero Raw Loops**: 100% vectorized NumPy/PyTorch math — zero pixel iteration loops (`for y in range(H): for x in range(W)` is forbidden).
- **Zero `print()` in Source Modules**: Always use `src.common.logger.get_logger(__name__)`.
- **Strict Typing**: Full PEP 484 type annotations on all function signatures.

---

## 5. 🛠️ Essential Workspace Commands

```bash
# Display all automation commands
make help

# Format & Lint
make format
make lint

# Run quick sanity smoke test (< 5 seconds)
make test-smoke

# Run full test suite
make test

# Execute training pipeline
bash scripts/run_pipeline.sh train configs/config.yaml
```

---

## 6. 📝 Contract-First Response Protocol

Before writing or editing code for any non-trivial task, structure your work in 4 steps:

1. **Scope & Tensor Contracts**: Target file path, input/output tensor shapes, dtypes, and device.
2. **Detailed Plan & DoD**: Granular execution steps with clear Definition of Done.
3. **Production Implementation**: Type-annotated, vectorized, linted code.
4. **Self-Audit & Verification**: Run `make test-smoke` and verify results before declaring done.
