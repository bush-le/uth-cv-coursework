# 11-Slide Unified 2D Image + SD-Video Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the SD-Video extension (`sd-video:docs/reports/SD_VIDEO_DEFINE_AND_DESIGN_SLIDES.html`) into the 2D baseline presentation (`/home/bush/Projects/uth-cv-coursework/docs/reports/presentation_5min_define_and_design.html` and companion guides) as an **11-slide unified ultra-visual presentation** with **100% Clean UI** (no `N` speaker-notes drawer, no `R` references modal), explicit `Nhóm thực hiện: Nhóm 4` attribution, zero unexecuted metrics, and zero Headless Chrome geometry/overflow defects.

**Architecture:** Single self-contained HTML5 presentation (`presentation_5min_define_and_design.html`) with fixed `1920×1080` stage (`viewport-base.css`), strict `Blue Professional` single-accent palette (`#fdfae7` background, `#1e2bfa` cobalt accent), reusable SVG `<defs>` (markers, gradients, clip paths, stylized facial portrait templates, temporal `t1-t2-t3` filmstrip vectors, hand-occlusion overlays), CSS keyframe animations (`@keyframes flowPulse`, `@keyframes scanLine`, `@keyframes pulseNode`, `@keyframes playhead`), and an automated Headless Chrome CDP + Pytest verification suite (`tests/test_presentation_deck.py`).

**Tech Stack:** HTML5, CSS3 (`viewport-base.css` + CSS Grid + CSS Keyframe Animations), Inline SVG (`<defs>`, `<use>`, layered facial portrait vectors, temporal video filmstrips, CV diagrams), Vanilla JS (`SlidePresentation` class without overlay modals), Python 3.12 + `pytest` + Headless Google Chrome CDP (`websocket-client`).

**Spec:** `/home/bush/Projects/uth-cv-coursework/docs/reports/2026-11-07-presentation-slide-deck-design.md`

## Global Constraints

- Fixed `1920×1080` 16:9 canvas with `.slide-counter`, `.nav-controls`, and `.progress-bar` positioned inside `#deck-stage` (`position: absolute`).
- **100% Clean UI Invariant**: Strictly NO `#overlay`, `.notes-drawer`, `.ref-modal`, or keyboard `N`/`R` popups.
- Exact 11 slides (`#slide-1` through `#slide-11`), each fitting within `1920×1080` (`scrollHeight <= 1080`, `scrollWidth <= 1920`) and filling the `820px` content stage with `<= 15%` vertical slack.
- Strict `Blue Professional` palette: `--bg: #fdfae7`, `--primary: #1e2bfa`, `--text: #111111`, `--text-muted: #6b6b6b`. Forbidden hex codes: `#d97706`, `#dc2626`, `#059669`, `#9333ea`, `#1e293b`, `#111827`.
- Strictly ZERO unexecuted metrics, thresholds, or code clutter in visible slide text: `95%`, `62%`, `92%`, `85%`, `50.0`, `>= 50`, `0.28`, `0.30`, `0.85`, `78.4`, `0.312`, `3,000`, `5,000`, `256`, `512`, `70%`, `15%`, `6GB`, `50 steps`, `30 steps`, `SHA-256`, `FP16`, `1e-4`, `src/`, `data/`, `<pre>`.
- Prominent `Nhóm thực hiện: Nhóm 4` attribution on Slide 01 (`#slide-1`), Slide 03 (`#slide-3`), Slide 04 (`#slide-4`), and Slide 10 (`#slide-10`).
- Zero SVG text-to-box overflow (`SVG_TEXT_OVERFLOWS_RECT = 0`) and matched SVG `viewBox` aspect ratios (`SVG_ASPECT_RATIO_MISMATCH <= 22%`) across all 11 slides in Headless Chrome at `1920×1080`.

## Review Focus

- **SVG text clipping inside narrow diagram boxes (`SVG_TEXT_OVERFLOWS_RECT`)**: Vietnamese uppercase labels and multi-word stage titles must be concise enough (`<= 58` chars) that rendered `getBBox().width` never exceeds the enclosing `<rect>` width in Headless Chrome.
- **SVG aspect-ratio distortion (`SVG_ASPECT_RATIO_MISMATCH`)**: Every `<svg viewBox="0 0 W H">` inside a flex/grid container must have its `W/H` ratio matched to the container's rendered `clientWidth / clientHeight` within `<= 22%`.
- **Clean UI & Zero forbidden numeric substrings**: Ensure no `N`/`R` overlays exist and that `512 × 512` / `50 frame` from the video branch are converted into conceptual labels (`Khung mặt chuẩn hóa`, `Chuỗi khung hình ngắn`).
- **Complete SD-Video integration**: Ensure all core concepts from `sd-video` are preserved (`Ảnh A + Video B → Video A←B`, `t1-t2-t3` filmstrip with hand occlusion, 4 models `Face-Adapter`, `REFace`, `VFace`, `VividFace` in 2 families, Pilot Matrix, 3-axis evaluation `Identity / Motion / Temporal`, connected-component lock on both `Identity` and `Parent Video`, `D0 / D1 / D2` detector regimes, and `M0 → M7` milestones).

---

### Task 1: Upgrade Automated Test Suite for the 11-Slide Unified 2D + SD-Video Specification (TDD Red)

**Files:**
- Modify: `/home/bush/Projects/uth-cv-coursework/tests/test_presentation_deck.py`

**Interfaces:**
- Consumes: `/home/bush/Projects/uth-cv-coursework/docs/reports/presentation_5min_define_and_design.html`, `/home/bush/Projects/uth-cv-coursework/docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md`, `/home/bush/Projects/uth-cv-coursework/docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md`
- Produces: Pytest verification suite asserting 11 slides (`#slide-1`..`#slide-11`), Clean UI (`#overlay` and `.notes-drawer` absent), 2D + Video content markers (`Face-Adapter`, `REFace`, `VFace`, `VividFace`, `Video A←B`, `D0`, `D1`, `D2`, `M0`), `Nhóm thực hiện: Nhóm 4` on Slides 1, 3, 4, and 10, zero forbidden tokens/colors, and zero Headless Chrome layout/SVG defects across all 11 slides.

- [x] **Step 1: Update `tests/test_presentation_deck.py` to enforce the 11-slide unified specification**
  - Expected: `tests/test_presentation_deck.py` checks for 11 slides (`01 / 11`), Clean UI, SD-Video concepts, and Headless Chrome geometry across slides 1..11.
- [x] **Step 2: Run `/home/bush/Projects/uth-cv-coursework/.venv/bin/pytest tests/test_presentation_deck.py -v` and confirm RED failure on the 8-slide HTML**
  - Expected: Fails because `presentation_5min_define_and_design.html` currently has 8 slides instead of 11 and lacks the SD-Video slides.

---

### Task 2: Build the 11-Slide Unified HTML Deck (`presentation_5min_define_and_design.html`) & Companion Docs (TDD Green)

**Files:**
- Modify: `/home/bush/Projects/uth-cv-coursework/docs/reports/presentation_5min_define_and_design.html`
- Modify: `/home/bush/Projects/uth-cv-coursework/docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md`
- Modify: `/home/bush/Projects/uth-cv-coursework/docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md`

**Interfaces:**
- Consumes: Design spec `/home/bush/Projects/uth-cv-coursework/docs/reports/2026-11-07-presentation-slide-deck-design.md` and `tests/test_presentation_deck.py`
- Produces: Self-contained 11-slide HTML presentation (Clean UI, 2D + SD-Video unified) and synchronized Markdown guides passing all static and Headless Chrome geometry tests.

- [x] **Step 1: Implement all 11 unified slides (`#slide-1` through `#slide-11`) with Clean UI, temporal video filmstrips (`t1-t2-t3`), 4 SD-Video backends, 3-axis QA gate, connected-component vault, and `D0/D1/D2` evaluation in `presentation_5min_define_and_design.html`**
- [x] **Step 2: Synchronize `PRESENTATION_5MIN_DEFINE_AND_DESIGN.md` and `SLIDE_PRESENTATION_STRUCTURE_5MIN.md` with the 11-slide unified structure**
- [x] **Step 3: Run Headless Chrome visual & geometry audit (`audit_slides.py`) + inspect rendered PNG screenshots (`slide_1.png`..`slide_11.png`) and run `pytest` until 100% GREEN**
  - Expected: `0` layout/SVG defects across all 11 slides and `100%` pytest pass.
- [x] **Step 4: Commit all updated artifacts after pre-commit audit**
