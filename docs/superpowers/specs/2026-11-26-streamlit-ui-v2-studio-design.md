# Design Spec: Streamlit Hybrid UI v2 — Studio Workspace, Pipeline Flow & Auto Batch Generation

**Date:** 2026-11-26  
**Status:** Updated with Auto Batch (`N`) & End-to-End Pipeline Stepper (Pending Final Approval)  
**Target Branch:** `main` (modifying `src/pipelines/live_sd_worker.py`, `src/analysis/ui_showcase.py`, `scripts/app_streamlit.py`, `tests/test_live_sd_worker.py`, `tests/test_ui_showcase.py`)

---

## 1. Intent & Why

Real browser testing of UI v1 (`3873784`–`86c87b8`) uncovered major UX friction and missing interactive capabilities:

1. **Vertical Scroll Fatigue & Lost Context:** Header controls, 12 large gallery cards, and the sandbox pushed the X-Ray Inspector below the fold, and the active gallery card had no visual highlight.
2. **Incomplete Pipeline Storytelling:** Individual columns existed, but there was no unified **end-to-end pipeline flow indicator** (`Input & Sinh Ảnh ➔ Data Gate ➔ M0 Mining ➔ M3 Verdict & Explainability`) showing how a sample progresses (or gets blocked) from start to finish.
3. **Misleading Progress Colors & Cramped Heatmaps:** Default blue `st.progress` bars made failed metrics (`✕`) and `FAKE` predictions look identical to passing/`REAL` ones; `M0 vs M3` heatmaps were squeezed into ~1/6 screen width; `force_detector_checkbox` was hidden inside a collapsed upload expander.
4. **No Random or Auto Batch (`N`) Generation on UI:** Despite `PromptGrammarEngine` existing in `src/models/diffusion/prompt_engine.py`, the Live SD form had only 1 static prompt (`seed=42`) and could only generate 1 image at a time — no **Random** button and no **Auto Batch (`N` samples)** mode with a live funnel summary.
5. **Sluggish Reruns & Log Noise:** Gallery/slider clicks recomputed heavy YuNet/CLIP/DINOv2 gradients, `sd_source` showed raw file paths without face thumbnails, and `use_container_width=True` spammed Streamlit `1.67` deprecation warnings.

---

## 2. Goals, Non-Goals & Anti-Goals

### Goals
- **Above-the-Fold Studio Workspace (Approach A):** Place 4 Mode buttons, 4 Golden Presets, threshold sliders, and Global Scorecard in `st.sidebar`; keep a compact 3-mode source bar (`Kho Held-Out` | `Sinh Live / Auto Batch SD 1.5` | `Upload Ảnh Lạ`) at the top of the main canvas right above the X-Ray Inspector.
- **Complete End-to-End Visual Pipeline Flow (`Input ➔ Output`):** Render a 4-stage live **Pipeline Stepper / Ribbon** at the top of the X-Ray Inspector showing exact live status at each stage:
  1. `① Input & Synthesis` (Seed / FFHQ Source / Mask / Mode / Gen Latency)
  2. `② Data Curation Gate` (`✓ Đạt 6/6` or `✕ Loại tại [lý do]`)
  3. `③ M0 Hard-Sample Mining` (`🔥 M0 Bỏ Lọt (< τ)` / `✓ M0 Bắt Ngay` / `Skipped`)
  4. `④ Detector M3 & Explainability` (`M3 P(fake)` + `M0→M3` rescue verdict + spatial heatmap)
- **Random & Auto Batch (`N` samples) Live Pipeline:**
  - Wire `PromptGrammarEngine` (`src/models/diffusion/prompt_engine.py`) into `sample_random_live_config()` and `sample_random_batch_configs()`.
  - Support **1-Click Random single**, **Random config fill**, and **Auto Batch (`N = 1..12`)** across fixed modes (`text2img`, `img2img`, `inpaint`) or **`mixed`** (cycling all 3 modes), running in **one isolated subprocess** (`live_sd_worker.py --batch-manifest`) to load pipelines efficiently on 4GB VRAM, then automatically running the full **Data Gate ➔ M0 Mining ➔ M3 Evaluation** funnel over all `N` generated samples and displaying a live batch funnel + clickable thumbnail strip.
- **Semantic Visuals & Fast Caching:** Color-coded HTML metric bars (Green = Pass/REAL, Red/Amber = Fail/FAKE, Gray = N/A), inline `force_detector_checkbox` inside Column C when rejected, enlarged full-width Row 2 `M0 vs M3` heatmaps, FFHQ face thumbnail preview in Live SD, image-hash caching for heavy CV/DINOv2 outputs, and `width="stretch"` cleanup.

### Non-Goals
- Do not modify trained checkpoints (`M0`, `M2*`, `M3`), dataset splits, or `configs/dataset_factory.yaml`.
- Do not hold Stable Diffusion pipelines and DINOv2 detectors on GPU simultaneously (`4096 MiB` RTX 3050 limit).

### Anti-Goals
- Never spawn `N` separate heavyweight SD worker subprocesses for a batch of `N` images (reloading SD weights `N` times wastes ~4s × `N`).
- Never break existing `AppTest` and unit test contracts (`_metric_progress_ratio`, 4 mode buttons, 4 Golden Presets, Scorecard expander, slider keys, `Sinh thử 1 ảnh Live (SD 1.5)` submit flow).

---

## 3. Architecture & Data Flow

```
Sidebar (Controls)                  Main Workspace (Top -> Bottom)
┌──────────────────────┐           ┌───────────────────────────────────────────────┐
│ Device & VRAM Pill   │           │ Source Switch: Held-Out | Live/Auto | Upload  │
│ 4 Mode Buttons       │           │ (Active Live/Batch/Upload status banner)      │
│ 4 Golden Presets     │           ├───────────────────────────────────────────────┤
│ Sliders (Sharpness,  │           │ Strip: 6 cards/page OR N Auto-Batch results + │
│  Cheek FFT, CLIP, τ) │           │        Batch Funnel (Gen -> Gate -> M0 -> M3) │
│ Global Scorecard     │           ├───────────────────────────────────────────────┤
└──────────────────────┘           │ 4-Stage Pipeline Flow Ribbon:                 │
                                   │ [① Input] ➔ [② Gate] ➔ [③ M0 Mine] ➔ [④ M3]  │
                                   ├───────────────┬───────────────┬───────────────┤
                                   │ Col A: Input  │ Col B: 6 Gate │ Col C: M0..M3 │
                                   │ -> Output     │ Semantic Bars │ + Force Toggle│
                                   ├───────────────┴───────────────┴───────────────┤
                                   │ Row 2 (100% Width): M0 vs M3 DINOv2 Heatmaps  │
                                   └───────────────────────────────────────────────┘
```

### 3.1 Batch-Capable Isolated SD Worker (`src/pipelines/live_sd_worker.py`)
- Add `run_batch_live_generation(configs: Sequence[Dict[str, Any]], output_dir: Path, *, local_files_only: bool = False, cpu_offload: bool = True, device: Optional[str] = None, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]` and CLI `--batch-manifest <path.json>`.
- **VRAM & Speed Optimization:** Group/order execution inside a single `StableDiffusionEngine` instance so base SD 1.5 tasks (`text2img`, `img2img`) run before switching to `inpaint` (or switching cleanly via `engine._switch_mode()`, which explicitly `del`s the old pipeline and calls `gc.collect()` + `torch.cuda.empty_cache()`). Always wrap in `try ... finally: engine.close()`.
- Write each item to `output_dir / f"item_{idx:03d}"` (`output.png`, optional `input.png` & `mask.png`, `metadata.json`) and write `batch_manifest.json` with per-item paths + latencies.
- Preserve existing `run_single_live_generation()` and single-image CLI untouched for backward compatibility.

### 3.2 Random Config & Batch Launcher (`src/analysis/ui_showcase.py`)
- `sample_random_live_config(catalog: Dict[str, List[ShowcaseCatalogEntry]], *, mode: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]`:
  - Uses `PromptGrammarEngine(seed=resolved_seed).generate_prompt(mode=resolved_mode, seed_offset=...)`.
  - Picks random `seed` (`0..999999`), random `mask_region` (`"lips"`, `"periocular"`, `"full"`), and random FFHQ source path from catalog (for `img2img`/`inpaint`).
- `sample_random_batch_configs(catalog: Dict[str, List[ShowcaseCatalogEntry]], *, count: int, mode: str = "mixed", base_seed: Optional[int] = None) -> List[Dict[str, Any]]`:
  - Generates `count` (`1 <= count <= 12`) diverse configs; when `mode == "mixed"`, cycles `("text2img", "img2img", "inpaint")` evenly.
- `launch_live_sd_batch_worker(configs: Sequence[Dict[str, Any]], *, output_root: Path = Path("experiments/demo_runs/streamlit_live_batch"), timeout_s: int = 360) -> Dict[str, Any]`:
  - Writes temporary `request.json`, invokes `python -m src.pipelines.live_sd_worker --batch-manifest ...`, handles timeout/non-zero exit/corrupt JSON with contextual `RuntimeError` & cleanup.
- `evaluate_batch_with_bundle(items: Sequence[Dict[str, Any]], *, bundle: Optional[DetectorBundle], min_sharpness: float, min_cheek_fft: float, min_clip: Optional[float], tau: float, clip_scorer: Optional[Any] = None) -> Dict[str, Any]`:
  - Runs `inspect_pipeline_sample(..., compute_heatmaps=False)` across all `N` generated items (fast, no gradient overhead until user clicks an item!) and returns `items` + batch funnel counts (`generated`, `gate_passed`, `gate_rejected`, `hard_samples`, `m3_rescued`, `m3_caught`).

### 3.3 Studio UI & Inspection Caching (`scripts/app_streamlit.py`)
1. **Sidebar (`st.sidebar`):**
   - Device/VRAM caption, 4 mode buttons, 4 Golden Presets + secondary contrast button, 4 threshold sliders (`Min Sharpness`, `Min Cheek FFT`, `Min CLIP`, `Ngưỡng τ`), and Global Scorecard expander.
   - When `active_tab == "ood_and_real"`, place `Ngưỡng τ` first and tuck the 3 curation sliders inside a secondary sidebar sub-expander (`Ngưỡng tham chiếu Curation (tùy chọn)`) while keeping their widget keys mounted so state stays clean.
2. **Top Workspace Source Bar via `st.tabs(["📂 Kho Held-Out", "⚡ Sinh Live & Auto Batch (SD 1.5)", "📤 Upload Ảnh Lạ"])`:**
   - Using `st.tabs` keeps only the active source visible on screen while keeping widget keys mounted across interactions. Above the X-Ray Inspector, show an explicit **Active Source Banner** whenever viewing a Live, Auto-Batch, or Upload target (with a `Quay lại Kho Held-Out` reset button).
   - **`Kho Held-Out`**: funnel metrics, radio filter, 1-row 6-card horizontal strip with `▶ ĐANG SOI` badge & `type="primary"` button on the selected card, and `< Trước` / `Sau >` pagination.
   - **`Sinh Live / Auto SD 1.5`**:
     - Mode selectbox (`text2img`, `img2img`, `inpaint`, plus `mixed (cả 3 chế độ)` when `N > 1` or auto batch).
     - FFHQ thumbnail preview `(128×128)` next to source dropdown (for `img2img` / `inpaint`).
     - Action buttons: **`🎲 Random cấu hình`**, **`⚡ Sinh ngẫu nhiên ngay (1-Click)`**, plus form submit **`Sinh thử 1 ảnh Live (SD 1.5)`**, AND an **Auto Batch** section (`Số lượng mẫu N` slider/number `1..12` + **`🚀 Auto Sinh & Kiểm Định Lô (N mẫu)`**).
     - After an Auto Batch completes, show its **Phễu Pipeline Lô (`N Sinh ➔ Đạt Gate ➔ M0 Bỏ Lọt ➔ M3 Bắt Đúng`)** and clickable thumbnail strip right below so the user can inspect any of the `N` generated samples in X-Ray.
   - **`Upload Ảnh Lạ`**: target + optional paired source uploaders + CLIP prompt + **`Phân tích ảnh upload`**.
3. **4-Stage Pipeline Flow Ribbon + 2-Tier X-Ray Inspector:**
   - **Pipeline Ribbon:** 4 connected stage cards (`① Input & Sinh Ảnh` ➔ `② Data Curation Gate` ➔ `③ M0 Hard Mining` ➔ `④ M3 Verdict & XAI`) styled cleanly via HTML/CSS with green/red/amber state pills.
   - **Row 1 (3 Columns):**
     - **Col A:** Single non-duplicated visual stage (interactive Before↔After wipe slider when paired & overlay off, or Input+Mask ↔ Output+Overlay) + prompt & metadata (`Seed`, `Steps`, `CFG`, `Strength`, `Latency`).
     - **Col B:** Colored semantic HTML bars for all 6 Data Gate metrics (`_metric_progress_ratio` drives width `[0..1]`; Green = Pass, Red = Fail, Gray = N/A).
     - **Col C:** Inline `force_detector_checkbox` directly inside the Data-Gate-rejected banner; colored semantic `M0 / M2* / M3` probability bars (`Red/Amber` if `>= tau`, `Green` if `< tau`); prominent `M0 → M3` rescue / True Negative / False Positive verdict card.
   - **Row 2 (Full Width `[1, 1]`):** Enlarged side-by-side **`M0 (Baseline)` vs `M3 (Champion)` DINOv2 Patch Sensitivity Heatmaps** + latency caption.
4. **Fast Inspection Cache:** Cache expensive per-image outputs (`measured`, `overlays`, `detector_scores`, `heatmaps`, `latencies_ms`) by `(image_sha256, mode, prompt, mask_region, force_det, bundle_device)` inside `st.session_state["_xray_cache"]` (bounded LRU, max 24 entries) and re-evaluate `apply_interactive_thresholds` + `face_audit` overlay + verdict badges instantaneously when sliders move.

---

## 4. Verification Plan

1. **Unit & Integration Tests (`tests/test_live_sd_worker.py`, `tests/test_ui_showcase.py`):**
   - Test `run_batch_live_generation()` with `model_id="mock_sd"` across `mixed` modes (`text2img`, `img2img`, `inpaint`) verifying single-worker execution, per-item artifacts, and `batch_manifest.json`.
   - Test `sample_random_live_config()`, `sample_random_batch_configs(count=4, mode="mixed")`, and `evaluate_batch_with_bundle()`.
   - Verify all existing UI/AppTest checks pass plus new `AppTest` checks for sidebar layout, `🎲 Random cấu hình`, Auto Batch submission (mocked worker), pipeline ribbon presence, and inline `force_detector_checkbox`.
2. **Repo Quality Gates:**
   - Zero `print()` in `src/`.
   - `make lint && make type-check && make test-smoke && make test`.
   - Restart live Streamlit server and confirm zero `use_container_width` warnings in server logs.
