# 5-Minute Visual Define & System Design Slide Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `docs/reports/presentation_5min_define_and_design.html` into a 6-slide, 100%-visual, zero-empty-space `1920×1080` presentation focused strictly on **Define + Design System (Proposed Pipeline Intent)** with zero unverified numbers, metrics, or implementation clutter.

**Architecture:** Single-file, zero-dependency HTML5 deck combining `frontend-slides/viewport-base.css` (fixed `1920×1080` 16:9 stage scaled uniformly via JS) with the strict `Blue Professional` design system (`#fdfae7` warm cream canvas, `#1e2bfa` single cobalt blue accent, `Space Grotesk` + `Inter`). Every slide uses full-height CSS Grid/Flex (`1776×820` content stage) and bespoke inline SVG flow diagrams to communicate the proposed direction visually without dead space.

**Tech Stack:** HTML5, CSS3 Grid/Flexbox, Inline SVG Vector Diagrams, Vanilla ES6 JavaScript (`SlidePresentation` stage scaler + keyboard/touch navigation), Pytest (DOM, regex guardrail, and style-contract verification).

**Spec:** `docs/reports/2026-11-07-presentation-slide-deck-design.md`

## Global Constraints

- **Fixed 16:9 Stage:** Include `viewport-base.css` (`1920px × 1080px` `.deck-stage` + `.slide`, scaled via `transform: translate(...) scale(...)`).
- **Single-Accent `Blue Professional` Palette:** `--bg: #fdfae7`, `--primary: #1e2bfa`, `--text: #111111`, `--text-muted: #6b6b6b`, `--text-light: #9a9a9a`, `--accent-light: rgba(30, 43, 250, 0.08)`, `--accent-medium: rgba(30, 43, 250, 0.15)`, `--border: rgba(30, 43, 250, 0.2)`, `--card-bg: rgba(30, 43, 250, 0.04)`. Forbidden rainbow/dark card colors: `#d97706`, `#dc2626`, `#059669`, `#9333ea`, `#1e293b`, `#111827`.
- **Zero Unverified Numbers / Thresholds / Code Details:** Forbidden tokens in visible slide text: `95%`, `62%`, `92%`, `85%`, `50.0`, `>= 50`, `0.28`, `0.30`, `0.85`, `78.4`, `0.312`, `3,000`, `5,000`, `256`, `512`, `70%`, `15%`, `6GB`, `50 steps`, `30 steps`, `seed`, `SHA-256`, `FP16`, `1e-4`, `src/`, `data/`, `<pre>`.
- **100% Canvas Fill & Visual First:** Exactly 6 slides (`.slide#slide-1` through `.slide#slide-6`). Every slide must contain at least one `<svg>` diagram and stretch cards/grids across the full `1776×820` content stage with no `justify-content: flex-start` dead bottom space.
- **Minimal Persistent Chrome:** Only bottom-left `.slide-counter` (`1 / 6`), bottom-right circular `.nav-controls`, and bottom 3px `.progress-bar`. No floating dark HUD or speaker notes drawer.

## Review Focus

- **Leakage of unexecuted numbers or thresholds into SVG `<text>` or card copy:** Any numeric claim (`95% -> 62%`, `Laplacian >= 50`, `256x256`, `70/15/15`, `3,000-5,000`) violates the user's strict rule and must be caught by automated regex tests on extracted visible text.
- **Rainbow accent drift violating `Blue Professional`:** Using red/green/orange/purple fills or dark code blocks breaks the single-cobalt discipline (`#1e2bfa` on `#fdfae7`); tests must assert zero forbidden hex codes.
- **Vertical under-fill leaving blank cream space at the bottom of `1920×1080` slides:** Slide body containers must use `flex: 1; display: grid;` with explicit fractional rows (`grid-template-rows`) so cards and SVGs stretch to fill the entire `820px` content height.
- **Text overflow or illegible micro-fonts inside SVG diagrams:** SVG `viewBox` coordinates must match container aspect ratios and use `font-size >= 15` (`Space Grotesk` / `Inter`) so labels remain crisp and never clip box borders.
- **Slide switching visibility bug (`display: flex` overriding `.slide` visibility):** Per `frontend-slides/SKILL.md`, `.slide` visibility must be controlled via `visibility: hidden/visible` and `opacity: 0/1`, never `display: none`.

---

### Task 1: Automated Guardrail Tests & Visual Slides 01–03 (Define Problem & Scope)

**Files:**
- Create: `tests/test_presentation_deck.py`
- Modify: `docs/reports/presentation_5min_define_and_design.html`

**Interfaces:**
- Consumes: `docs/reports/2026-11-07-presentation-slide-deck-design.md`
- Produces:
  - `tests/test_presentation_deck.py` testing all 6 slides, CSS invariants, SVG presence, and forbidden token guardrails.
  - `docs/reports/presentation_5min_define_and_design.html` containing the complete CSS/JS stage engine and completed visual markup for `#slide-1`, `#slide-2`, and `#slide-3` (with placeholder shells for `#slide-4`..`#slide-6` until Task 2).

- [x] **Step 1: Write the failing guardrail test suite in `tests/test_presentation_deck.py`**

```python
from pathlib import Path
import re
import pytest

DECK_PATH = Path("docs/reports/presentation_5min_define_and_design.html")


def _extract_visible_text(html: str) -> str:
    no_style = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    no_script = re.sub(r"<script[\s\S]*?</script>", " ", no_style, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", no_script)
    return re.sub(r"\s+", " ", text).strip()


def test_deck_has_six_slides_and_fixed_stage():
    html = DECK_PATH.read_text(encoding="utf-8")
    assert "width: 1920px;" in html and "height: 1080px;" in html
    slides = re.findall(r'<section[^>]+class="[^"]*slide[^"]*"[^>]*>', html)
    assert len(slides) == 6
    for idx in range(1, 7):
        assert f'id="slide-{idx}"' in html


def test_deck_enforces_blue_professional_single_accent_palette():
    html = DECK_PATH.read_text(encoding="utf-8").lower()
    assert "#fdfae7" in html
    assert "#1e2bfa" in html
    for forbidden_hex in ["#d97706", "#dc2626", "#059669", "#9333ea", "#1e293b", "#111827"]:
        assert forbidden_hex not in html, f"Forbidden non-Blue-Professional color {forbidden_hex}"
    assert "notes-drawer" not in html
    assert "<pre" not in html


def test_deck_contains_zero_unverified_numbers_or_code_clutter():
    html = DECK_PATH.read_text(encoding="utf-8")
    visible = _extract_visible_text(html)
    forbidden_patterns = [
        r"\b95%\b", r"\b62%\b", r"\b92%\b", r"\b85%\b",
        r"\b50\.0\b", r"&\ge;\s*50", r"≥\s*50", r"\b0\.28\b", r"\b0\.30\b", r"\b0\.85\b",
        r"\b78\.4\b", r"\b0\.312\b", r"\b3,000\b", r"\b5,000\b",
        r"\b256\b", r"\b512\b", r"\b70%\b", r"\b15%\b",
        r"\b6GB\b", r"\b50 steps\b", r"\b30 steps\b", r"\bSHA-256\b", r"\bFP16\b",
        r"\b1e-4\b", r"\bsrc/", r"\bdata/",
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, visible, flags=re.I), f"Forbidden unverified detail matched: {pat}"


def test_every_slide_has_inline_svg_visual_diagram():
    html = DECK_PATH.read_text(encoding="utf-8")
    parts = re.split(r'<section[^>]+class="[^"]*slide[^"]*"[^>]*>', html)[1:]
    assert len(parts) == 6
    for idx, slide_html in enumerate(parts, start=1):
        assert "<svg" in slide_html, f"Slide {idx} is missing an inline <svg> visual diagram"
```

- [x] **Step 2: Run pytest to verify the test fails on the existing V3 HTML**

Run: `pytest tests/test_presentation_deck.py -v`
Expected: FAIL (`len(slides) == 7`, forbidden colors `#d97706`/`#dc2626`/`#059669`/`#1e293b` present, and forbidden numbers `95%`, `62%`, `256`, `0.28`, `78.4` present).

- [x] **Step 3: Implement the `Blue Professional` + `viewport-base.css` stage shell and Slides 01–03 in `docs/reports/presentation_5min_define_and_design.html`**

Build the complete HTML document in `docs/reports/presentation_5min_define_and_design.html` with:
- Full `viewport-base.css` rules + `Blue Professional` variables (`--bg: #fdfae7`, `--primary: #1e2bfa`, `--text: #111111`, `--text-muted: #6b6b6b`, `--text-light: #9a9a9a`, `--accent-light: rgba(30, 43, 250, 0.08)`, `--accent-medium: rgba(30, 43, 250, 0.15)`, `--border: rgba(30, 43, 250, 0.2)`, `--card-bg: rgba(30, 43, 250, 0.04)`).
- Full-stage layout classes where `.slide-inner` is `width: 1776px; height: 936px; display: flex; flex-direction: column;` and `.slide-body` is `flex: 1; display: grid; min-height: 0;` so every slide stretches to fill 100% of the stage.
- **Slide 01 (`#slide-1`)**: 2-column split cover (`grid-template-columns: 1.1fr 0.9fr; gap: 48px; align-items: stretch;`) with title, subtitle, presenter metadata card on the left, and a full-height SVG 4-Pillar Roadmap diagram (`01 Define Problem` $\to$ `02 Define Scope` $\to$ `03 Design Dataset & System` $\to$ `04 Design Evaluation`) on the right.
- **Slide 02 (`#slide-2`)**: `01 / PROBLEM DEFINITION` (`grid-template-rows: 1.35fr 0.85fr; gap: 24px;`) with a large SVG comparative flow diagram contrasting *Hạn Chế Của Nguồn Dữ Liệu Hiện Tại* vs. *Hướng Đi Đề Xuất Với Stable Diffusion* on top, and 3 equal-height visual cards below.
- **Slide 03 (`#slide-3`)**: `02 / OBJECTIVES & SCOPE` (`grid-template-rows: 1.35fr 0.85fr; gap: 24px;`) with 3 visual mode cards (*Text-to-Image*, *Facial Inpainting*, *Image-to-Image*), each embedding a cobalt-themed SVG flow (`Input` $\to$ `Stable Diffusion` $\to$ `Output`) on top, and a 2-column *In-Scope vs. Out-of-Scope* visual matrix below.
- Scaffold `#slide-4`, `#slide-5`, `#slide-6` with valid `<svg>` placeholders so Task 1 tests can inspect the structure.

- [x] **Step 4: Run pytest to verify Task 1 passes**

Run: `pytest tests/test_presentation_deck.py -v`
Expected: PASS (4/4 tests passing).

- [x] **Step 5: Commit Task 1**

```bash
git -C /home/bush/Projects/uth-cv-coursework add tests/test_presentation_deck.py docs/reports/presentation_5min_define_and_design.html docs/reports/2026-11-07-presentation-slide-deck-design.md
git -C /home/bush/Projects/uth-cv-coursework commit -m "feat(slides): add deck guardrail tests and visual Define slides 1-3"
```

---

### Task 2: Visual System Design & Evaluation Slides 04–06 + Speaker Script Sync

**Files:**
- Modify: `docs/reports/presentation_5min_define_and_design.html`
- Modify: `tests/test_presentation_deck.py`
- Modify: `docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md`
- Modify: `docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md`

**Interfaces:**
- Consumes: `docs/reports/presentation_5min_define_and_design.html` (from Task 1), `docs/reports/2026-11-07-presentation-slide-deck-design.md`
- Produces:
  - Completed visual markup for `#slide-4` (Dataset Design), `#slide-5` (Hero 6-Stage System Architecture Pipeline), and `#slide-6` (Evaluation Design & Expected Outcomes).
  - Content-completeness assertions in `tests/test_presentation_deck.py` verifying all 6 pipeline stages and 4 deliverables.
  - Synchronized speaker notes in `docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md` and `docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md`.

- [x] **Step 1: Add content-completeness assertions for Slides 04–06 in `tests/test_presentation_deck.py`**

```python
def test_slides_4_to_6_cover_taxonomy_six_pipeline_stages_and_evaluation():
    html = DECK_PATH.read_text(encoding="utf-8")
    required_phrases = [
        "Real Faces",
        "Deepfake Faces",
        "SD Synthetic Faces",
        "Data Acquisition",
        "Face Preprocessing",
        "Stable Diffusion Generation",
        "Quality Control",
        "Labeling &amp; Metadata",
        "Dataset Export",
        "Baseline",
        "Proposed",
        "Held-Out Test Set",
    ]
    for phrase in required_phrases:
        assert phrase in html, f"Missing required System Design visual label: {phrase}"
```

- [x] **Step 2: Run pytest to verify the new test fails on placeholder Slides 04–06**

Run: `pytest tests/test_presentation_deck.py::test_slides_4_to_6_cover_taxonomy_six_pipeline_stages_and_evaluation -v`
Expected: FAIL with missing required phrases on Slides 04–06.

- [x] **Step 3: Implement Slides 04, 05 (Hero Architecture Pipeline), and 06 in `docs/reports/presentation_5min_define_and_design.html` and sync Markdown guides**

Populate `#slide-4`, `#slide-5`, and `#slide-6` in `docs/reports/presentation_5min_define_and_design.html`:
- **Slide 04 (`#slide-4`) — `03 / DATASET DESIGN`**:
  - 2-column full-height layout (`grid-template-columns: 0.95fr 1.05fr; gap: 28px;`).
  - Left column: 3 stacked visual cards for `01. Real Faces`, `02. Deepfake Faces`, `03. SD Synthetic Faces` + a bottom provenance callout card.
  - Right column: Top panel with an inline SVG diagram of **Identity-Disjoint Partitioning** (`Nhân Dạng Gốc A + Ảnh Biến Thể A` $\to$ `Cố Định Trong Tập Train` | `Rào Chắn Chống Rò Rỉ` | `Nhân Dạng Gốc B + Ảnh Biến Thể B` $\to$ `Cố Định Trong Tập Test`); Bottom panel with a 6-cell visual metadata attribute grid (`Mã Định Danh Ảnh`, `Nhãn Phân Loại`, `Nhóm Nguồn Gốc`, `Chế Độ Sinh Ảnh`, `Liên Kết Nhân Dạng Gốc`, `Mô Tả Prompt`).
- **Slide 05 (`#slide-5`) — `04 / SYSTEM ARCHITECTURE` (Hero Slide)**:
  - Full-canvas 2-row layout (`grid-template-rows: 1.55fr 0.65fr; gap: 22px;`).
  - Top row: Large inline SVG (`viewBox="0 0 1720 520"`) depicting the **6-Stage Proposed Pipeline Flow** in clean `Blue Professional` cobalt geometry:
    1. `01. Data Acquisition` (*Nguồn ảnh mặt thật & Deepfake*)
    2. `02. Face Preprocessing` (*Phát hiện, căn chỉnh & chuẩn hóa vùng mặt*)
    3. `03. Stable Diffusion Generation` (*Tạo sinh có điều kiện: T2I / Inpaint / Img2Img*)
    4. `04. Quality Control` (*Sàng lọc độ nét, cấu trúc mặt & khớp mô tả* $\to$ nhánh *Đạt chuẩn* đi tiếp, nhánh *Lỗi / Biến dạng* loại bỏ)
    5. `05. Labeling & Metadata` (*Gán nhãn phân nhóm & lưu vết nguồn gốc*)
    6. `06. Dataset Export` (*Phân tập Train / Val / Test theo danh tính*)
    $\implies$ Bottom conduit into **Downstream Task: Deepfake Detection & Evaluation**.
  - Bottom row: 3 equal-width Architectural Intent cards stretching across the full width.
- **Slide 06 (`#slide-6`) — `05 / EVALUATION & EXPECTED OUTCOMES`**:
  - Full-canvas 2-row layout (`grid-template-rows: 1.2fr 0.9fr; gap: 22px;`).
  - Top row: Inline SVG diagram of the **2-Branch Comparative Experiment**:
    - `Nhánh Baseline (Real + Deepfake)` vs. `Nhánh Proposed (Real + Deepfake + SD Synthetic)` $\to$ `Cùng Kiến Trúc Mô Hình Phát Hiện (CNN / ViT)` $\to$ `Đánh Giá Trên Cùng Held-Out Test Set` $\to$ `Đo Lường Độ Cải Thiện (Accuracy, Precision, Recall, F1, ROC-AUC & Độ Tổng Quát Hóa)`.
  - Bottom row: 4 equal-width visual deliverable cards (`01. Bộ Dữ Liệu Khuôn Mặt Có Cấu Trúc`, `02. Quy Trình Sinh & Lọc Tự Động`, `03. Thực Nghiệm Đối Chứng Phát Hiện Deepfake`, `04. Báo Cáo Đánh Giá Thực Nghiệm`).
- Update `docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md` and `docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md` to remove unverified numbers and match the 6 visual slides.

- [x] **Step 4: Run full pytest suite to verify all checks pass**

Run: `pytest tests/test_presentation_deck.py -v`
Expected: PASS (5/5 tests passing).

- [x] **Step 5: Commit Task 2**

```bash
git -C /home/bush/Projects/uth-cv-coursework add docs/reports/presentation_5min_define_and_design.html docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md tests/test_presentation_deck.py
git -C /home/bush/Projects/uth-cv-coursework commit -m "feat(slides): complete visual System Design & Evaluation slides 4-6"
```
