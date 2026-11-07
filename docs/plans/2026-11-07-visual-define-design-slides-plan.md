# 8-Slide Ultra-Visual Define & System Design Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `/home/bush/Projects/uth-cv-coursework/docs/reports/presentation_5min_define_and_design.html` (and its companion markdown guides) into an 8-slide ultra-visual presentation combining Computer Vision vector illustrations (Approach 1), stylized facial portrait thumbnails with technical overlays (Approach 2), and pure CSS/SVG data-flow & scanning animations (Approach 3) while enforcing zero unexecuted metrics, explicit `Nhóm thực hiện: Nhóm 4` attribution, and zero Headless Chrome geometry/overflow bugs.

**Architecture:** Single self-contained HTML5 presentation (`presentation_5min_define_and_design.html`) with fixed `1920×1080` stage (`viewport-base.css`), strict `Blue Professional` single-accent palette (`#fdfae7` background, `#1e2bfa` cobalt accent), reusable SVG `<defs>` (markers, gradients, clip paths, and stylized facial portrait templates), CSS keyframe animations (`@keyframes flowPulse`, `@keyframes scanLine`, `@keyframes pulseNode`), and an automated Headless Chrome CDP + Pytest verification suite (`tests/test_presentation_deck.py`).

**Tech Stack:** HTML5, CSS3 (`viewport-base.css` + CSS Grid + CSS Keyframe Animations), Inline SVG (`<defs>`, `<use>`, layered facial portrait vectors, CV diagrams), Vanilla JS (`SlidePresentation` class), Python 3.12 + `pytest` + Headless Google Chrome CDP (`websocket-client`).

**Spec:** `/home/bush/Projects/uth-cv-coursework/docs/reports/2026-11-07-presentation-slide-deck-design.md`

## Global Constraints

- Fixed `1920×1080` 16:9 canvas with `.slide-counter`, `.nav-controls`, and `.progress-bar` positioned inside `#deck-stage` (`position: absolute`).
- Exact 8 slides (`#slide-1` through `#slide-8`), each fitting within `1920×1080` (`scrollHeight <= 1080`, `scrollWidth <= 1920`) and filling the `820px` content stage with `<= 15%` vertical slack.
- Strict `Blue Professional` palette: `--bg: #fdfae7`, `--primary: #1e2bfa`, `--text: #111111`, `--text-muted: #6b6b6b`. Forbidden hex codes: `#d97706`, `#dc2626`, `#059669`, `#9333ea`, `#1e293b`, `#111827`.
- Strictly ZERO unexecuted metrics, thresholds, or code clutter in visible slide text: `95%`, `62%`, `92%`, `85%`, `50.0`, `>= 50`, `0.28`, `0.30`, `0.85`, `78.4`, `0.312`, `3,000`, `5,000`, `256`, `512`, `70%`, `15%`, `6GB`, `50 steps`, `30 steps`, `SHA-256`, `FP16`, `1e-4`, `src/`, `data/`, `<pre>`.
- Prominent `Nhóm thực hiện: Nhóm 4` attribution on Slide 01 (`#slide-1`), Slide 04 (`#slide-4`), and Slide 05 (`#slide-5`).
- Zero SVG text-to-box overflow (`SVG_TEXT_OVERFLOWS_RECT = 0`) and matched SVG `viewBox` aspect ratios (`SVG_ASPECT_RATIO_MISMATCH <= 22%`) across all 8 slides in Headless Chrome at `1920×1080`.

## Review Focus

- **SVG text clipping inside narrow diagram boxes (`SVG_TEXT_OVERFLOWS_RECT`)**: Vietnamese uppercase labels and multi-word stage titles must be concise enough that rendered `getBBox().width` never exceeds the enclosing `<rect>` width in Headless Chrome.
- **SVG aspect-ratio distortion (`SVG_ASPECT_RATIO_MISMATCH`)**: Every `<svg viewBox="0 0 W H">` inside a flex/grid container must have its `W/H` ratio matched to the container's rendered `clientWidth / clientHeight` within `<= 22%`.
- **Accidental inclusion of forbidden numeric substrings**: Even in SVG coordinates or labels, visible text nodes must never contain forbidden substrings (`256`, `512`, `0.28`, `50.0`, `95%`, etc.).
- **Conceptual illustration honesty**: Every embedded facial portrait thumbnail must carry a clear conceptual label (`MINH HỌA Ý TƯỞNG` / `CONCEPTUAL ILLUSTRATION`) so viewers immediately understand it illustrates the intended mechanism rather than claiming pre-run output.
- **Multi-line card header or title orphan wrapping**: `.slide-title` must stay on 1 line (`clientHeight <= 56px`) and card `h3` headings must stay on 1 line (`clientHeight <= 30px`) across all 8 slides.

---

### Task 1: Upgrade Automated Test Suite for the 8-Slide Ultra-Visual Specification (TDD Red)

**Files:**
- Modify: `/home/bush/Projects/uth-cv-coursework/tests/test_presentation_deck.py`

**Interfaces:**
- Consumes: `/home/bush/Projects/uth-cv-coursework/docs/reports/presentation_5min_define_and_design.html`, `/home/bush/Projects/uth-cv-coursework/docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md`, `/home/bush/Projects/uth-cv-coursework/docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md`
- Produces: Pytest verification suite asserting 8 slides (`#slide-1`..`#slide-8`), Approach 1 CV diagrams (Decision Boundary, ROC Curve, 3-Layer Funnel, 6-Stage Pipeline), Approach 2 Portrait Thumbnails & Overlays (`portrait-`, `MINH HỌA Ý TƯỞNG`), Approach 3 CSS/SVG Animations (`@keyframes flowPulse`, `@keyframes scanLine`, `@keyframes pulseNode`), `Nhóm thực hiện: Nhóm 4` on Slides 1, 4, and 5, zero forbidden tokens/colors, and zero Headless Chrome layout/SVG defects across all 8 slides.

- [x] **Step 1: Update `tests/test_presentation_deck.py` to enforce the 8-slide specification**
- [x] **Step 2: Run `/home/bush/Projects/uth-cv-coursework/.venv/bin/pytest tests/test_presentation_deck.py -v` and confirm RED failure on the old 6-slide HTML**

---

### Task 2: Build the 8-Slide Ultra-Visual HTML Deck (`presentation_5min_define_and_design.html`) & Companion Docs (TDD Green)

**Files:**
- Modify: `/home/bush/Projects/uth-cv-coursework/docs/reports/presentation_5min_define_and_design.html`
- Modify: `/home/bush/Projects/uth-cv-coursework/docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md`
- Modify: `/home/bush/Projects/uth-cv-coursework/docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md`

**Interfaces:**
- Consumes: Design spec `/home/bush/Projects/uth-cv-coursework/docs/reports/2026-11-07-presentation-slide-deck-design.md`
- Produces: Self-contained 8-slide HTML presentation and synchronized Markdown speaker script / structure guides passing all static and Headless Chrome geometry tests.

- [x] **Step 1: Implement reusable SVG `<defs>` (stylized face portrait vector symbols, landmark wireframes, binary masks, noise patterns, arrow markers) and CSS `@keyframes` (`flowPulse`, `scanLine`, `pulseNode`) plus all 8 slides (`#slide-1` through `#slide-8`) in `presentation_5min_define_and_design.html`**
- [x] **Step 2: Synchronize `PRESENTATION_5MIN_DEFINE_AND_DESIGN.md` and `SLIDE_PRESENTATION_STRUCTURE_5MIN.md` with the 8-slide structure**
- [x] **Step 3: Run Headless Chrome visual & geometry audit (`audit_slides.py`) + inspect rendered PNG screenshots (`slide_1.png`..`slide_8.png`) and run `pytest` until 100% GREEN**
- [x] **Step 4: Commit all updated artifacts after pre-commit audit**
