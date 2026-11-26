# Streamlit Hybrid UI v2 (Studio Workspace, Pipeline Flow & Auto Batch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Streamlit Showcase into a compact Studio Workspace + Sidebar with a 4-stage visual Pipeline Flow Ribbon, color-coded Data Gate & Detector bars, enlarged full-width `M0 vs M3` heatmaps, fast inspection caching, `PromptGrammarEngine`-powered 1-Click Random sampling, and a 4GB-VRAM-safe Auto Batch (`N = 1..12`) live generation & funnel evaluation mode.

**Architecture:** `src/pipelines/live_sd_worker.py` gains `run_batch_live_generation()` + `--batch-manifest` CLI to synthesize `N` samples inside a single isolated subprocess (`cpu_offload=True`, mode-ordered to minimize pipeline swaps). `src/analysis/ui_showcase.py` adds `PromptGrammarEngine`-backed random config/batch samplers (`sample_random_live_config`, `sample_random_batch_configs`), `launch_live_sd_batch_worker()`, `evaluate_batch_with_bundle()`, and an optional `precomputed` cache hook on `inspect_pipeline_sample()`. `scripts/app_streamlit.py` reorganizes controls into `st.sidebar`, top `st.tabs` (`Kho Held-Out`, `Sinh Live & Auto Batch (SD 1.5)`, `Upload Ảnh Lạ`), an end-to-end 4-stage Pipeline Flow Ribbon, semantic HTML bars, inline `force_detector_checkbox`, and full-width Row 2 heatmaps.

**Tech Stack:** Python 3.12+, PyTorch 2.9 (`dinov2_small`, `torch.amp.autocast`), OpenCV YuNet, Diffusers (`StableDiffusionEngine`), `PromptGrammarEngine`, Streamlit (`>=1.38,<2`, `width="stretch"`), pytest (`AppTest` + unit/smoke tests).

**Spec:** [2026-11-26-streamlit-ui-v2-studio-design.md](../specs/2026-11-26-streamlit-ui-v2-studio-design.md)

## Global Constraints

- Hardware ceiling: `NVIDIA GeForce RTX 3050 Laptop GPU` (`4096 MiB VRAM`); never hold SD 1.5 pipelines and DINOv2 detectors on GPU simultaneously. Always clear `@st.cache_resource` (`get_detector_bundle.clear()`) and call `unload_detector_bundle(bundle)` before spawning single or batch SD 1.5 workers.
- No `print()` in `src/`: always use `src.common.logger.get_logger(__name__)`.
- Keep `scripts/app_streamlit.py` free of raw CV/PyTorch tensor math and avoid `use_container_width` (use `width="stretch"`).
- Preserve existing `AppTest` and unit-test contracts: `_metric_progress_ratio`, 4 mode buttons (`Text2Img`, `Img2Img`, `Inpaint`, `OOD RV5.1 & Real FFHQ`), 4 Golden Preset buttons (`Ca 1:`..`Ca 4:`), Scorecard expander, slider labels (`Min Sharpness`, `Min Cheek FFT`, `Min CLIP`, `Ngưỡng τ`), and `Sinh thử 1 ảnh Live (SD 1.5)` submit flow.
- No absolute personal home paths in Markdown docs or UI labels.

## Review Focus

- **Auto Batch `mixed` mode ordering & 4GB VRAM cleanup (`run_batch_live_generation`)**: Must group `text2img`/`img2img` before `inpaint` inside a single `StableDiffusionEngine` instance, call `engine.close()` in `finally:`, and fail cleanly if `configs` is empty or an `img2img`/`inpaint` entry lacks a valid `source_image`.
- **Slider drag latency & stale `precomputed` cache**: Dragging `Min Sharpness`, `Min Cheek FFT`, `Min CLIP`, or `Ngưỡng τ` must immediately update `gate_passed`, red/green `face_audit` bounding box color, and `is_hard_sample` without reusing stale `gate_passed` or `run_detector_if_rejected` states from a previous cache hit (`run_detector_if_rejected=True` after an initial rejected inspection must compute missing `detector_scores` + `heatmaps` rather than returning empty dicts).
- **Random / Batch UI state transitions in Streamlit**: Clicking `🎲 Random cấu hình` or `🚀 Auto Sinh & Kiểm Định Lô` must never assign widget-bound `st.session_state` keys after those widgets are instantiated (always route programmatic updates through `_pending_*` keys applied at the top of `_init_session_state`).
- **Inline `force_detector_checkbox` on rejected samples**: Checking the inline toggle in Column C must update the active inspection immediately for catalog, uploaded, single-live, and auto-batch items without losing `custom_target` or active batch item selection.
- **Zero-FFHQ-source fallback**: If `catalog` has no local FFHQ paths on disk, `sample_random_live_config(catalog, mode="text2img")` must still succeed and `sample_random_batch_configs(..., mode="mixed")` must gracefully fall back to `text2img` (or raise a clear contextual `ValueError` when `img2img`/`inpaint` is explicitly requested without FFHQ images).

---

### Task 1: Batch-Capable Isolated Live SD Worker (`src/pipelines/live_sd_worker.py`)

**Files:**
- Modify: `src/pipelines/live_sd_worker.py`
- Modify: `tests/test_live_sd_worker.py`

**Interfaces:**
- Consumes: `StableDiffusionEngine`, `generate_dynamic_face_mask`, `detect_face_yunet`, `seed_everything`, `_load_generation_defaults`, `_normalize_source_image`, `resolve_cached_sd_model_id`
- Produces:
  - `run_batch_live_generation(configs: Sequence[Dict[str, Any]], output_dir: Path, *, local_files_only: bool = False, cpu_offload: bool = True, device: Optional[str] = None, config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]`
  - CLI `--batch-manifest <json_path>` support in `parse_args` / `main` writing `item_{idx:03d}/output.png` (+ optional `input.png`, `mask.png`, `metadata.json`) and `batch_manifest.json` (preserving original `item_index` order even when execution reorders modes to minimize pipeline swaps).

- [ ] **Step 1: Write failing tests for batch live generation & CLI `--batch-manifest`**

Append tests to `tests/test_live_sd_worker.py`:
- `test_run_batch_live_generation_mixed_modes_mock_sd(tmp_path)`: create an RGBA `(640, 480, 4)` source PNG, call `run_batch_live_generation` with 3 configs (`inpaint`, `text2img`, `img2img` — intentionally placing `inpaint` first to verify execution handles mixed mode order while returning items sorted by original index `0, 1, 2`) using `model_id="mock_sd"`, and assert `batch_manifest.json` exists, `len(result["items"]) == 3`, items preserve indices `[0, 1, 2]`, `item_000/mask.png` & `item_002/input.png` exist, and all `output.png` files are `(512, 512)` RGB.
- `test_run_batch_live_generation_validates_empty_and_missing_source(tmp_path)`: assert `ValueError` on `configs=[]` and on an `img2img` config without `source_image`.
- `test_live_worker_cli_batch_manifest_executes(tmp_path)`: write a JSON manifest file with 2 `text2img` `mock_sd` configs, invoke `main(["--batch-manifest", str(manifest_path), "--output-dir", str(out_dir)])`, assert exit code `0` and `(out_dir / "batch_manifest.json").is_file()`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_live_sd_worker.py -v`
Expected: FAIL (`ImportError: cannot import name 'run_batch_live_generation'`).

- [ ] **Step 3: Implement `run_batch_live_generation` and `--batch-manifest` CLI in `src/pipelines/live_sd_worker.py`**

Implementation rules:
- Extract shared single-sample synthesis helper `_synthesize_item_with_engine(engine: StableDiffusionEngine, *, mode: str, prompt: str, out_dir: Path, source_path: Optional[Path], mask_region: str, steps: int, guidance_scale: float, strength: float, seed: int, resolved_model_id: str) -> Dict[str, Any]` reused by both `run_single_live_generation` and `run_batch_live_generation` (zero duplicated synthesis logic).
- In `run_batch_live_generation`: validate all configs up front (`configs` non-empty, valid `mode in VALID_MODES`, existing `source_image` for `img2img`/`inpaint`), create a single `StableDiffusionEngine` (`cpu_offload=cpu_offload`, `local_files_only=local_files_only`), execute indices ordered by `(1 if mode == "inpaint" else 0, idx)` so base SD 1.5 modes share one pipeline before switching to `inpaint`, close `engine` in `finally:`, sort results back by `item_index`, and write `output_dir / "batch_manifest.json"`.
- Update `parse_args` so `--batch-manifest` is mutually compatible (either `--batch-manifest` + `--output-dir` OR `--mode` + `--prompt` + `--output-dir`), validating required fields cleanly via `parser.error(...)`.

- [ ] **Step 4: Run worker tests to verify they pass**

Run: `.venv/bin/pytest tests/test_live_sd_worker.py -v`
Expected: `8 passed` (5 existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/live_sd_worker.py tests/test_live_sd_worker.py
git commit -m "feat(pipelines): add batch live generation mode to isolated SD worker"
```

---

### Task 2: Random Config/Batch Samplers, Batch Worker Launcher & `precomputed` Inspection Cache Hook (`src/analysis/ui_showcase.py`)

**Files:**
- Modify: `src/analysis/ui_showcase.py`
- Modify: `tests/test_ui_showcase.py`

**Interfaces:**
- Consumes: `src.models.diffusion.prompt_engine.PromptGrammarEngine`, `src.pipelines.live_sd_worker`, `inspect_pipeline_sample`, `apply_interactive_thresholds`
- Produces:
  - `sample_random_live_config(catalog: Dict[str, List[ShowcaseCatalogEntry]], *, mode: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]`
  - `sample_random_batch_configs(catalog: Dict[str, List[ShowcaseCatalogEntry]], *, count: int, mode: str = "mixed", base_seed: Optional[int] = None) -> List[Dict[str, Any]]`
  - `launch_live_sd_batch_worker(configs: Sequence[Dict[str, Any]], *, output_root: Path = Path("experiments/demo_runs/streamlit_live_batch"), timeout_s: int = 360) -> Dict[str, Any]`
  - `evaluate_batch_with_bundle(items: Sequence[Dict[str, Any]], *, bundle: Optional[DetectorBundle], min_sharpness: float, min_cheek_fft: float, min_clip: Optional[float], tau: float, clip_scorer: Optional[Any] = None) -> Dict[str, Any]`
  - `inspect_pipeline_sample(..., precomputed: Optional[Dict[str, Any]] = None) -> InspectionResult` (when `precomputed` is supplied, skips redundant YuNet/CLIP/audit/detector/heatmap calls for items already present in `precomputed`, while ALWAYS recomputing `apply_interactive_thresholds`, `CurationMetricItem`s, `face_audit` red/green overlay, `is_hard_sample`, and running `detector_scores`/`heatmaps` on demand if `should_run_detector` flips from False to True).

- [ ] **Step 1: Write failing unit tests for random samplers, batch launcher, batch funnel evaluation, and `precomputed` cache equivalence**

Append tests to `tests/test_ui_showcase.py` (`@pytest.mark.smoke` where mock/lightweight):
- `test_sample_random_live_and_batch_configs_uses_prompt_grammar_and_handles_missing_ffhq(tmp_path)`:
  - Call `sample_random_live_config(catalog, mode="text2img", seed=123)` and verify keys `{"mode", "prompt", "seed", "source_image", "mask_region"}`, `prompt` non-empty, `seed == 123`.
  - Call `sample_random_batch_configs(catalog, count=5, mode="mixed", base_seed=77)` (when `catalog` has valid FFHQ sources, or with a temp PNG injected into `catalog["ood_and_real"]`) and verify 5 items cycling modes (`text2img`, `img2img`, `inpaint`, `text2img`, `img2img`) with valid `source_image` on paired modes.
  - With an empty catalog (`{}`), verify `sample_random_live_config({}, mode="text2img")` succeeds, `sample_random_batch_configs({}, count=3, mode="mixed")` falls back cleanly to `"text2img"`, and `sample_random_live_config({}, mode="img2img")` raises `ValueError`.
- `test_launch_live_sd_batch_worker_mock_and_error_cleanup(monkeypatch, tmp_path)`:
  - Verify `launch_live_sd_batch_worker` invokes `live_sd_worker` with `--batch-manifest`, returns parsed `batch_manifest.json`, and raises contextual `RuntimeError` + cleans `run_dir` on non-zero exit code.
- `test_evaluate_batch_with_bundle_computes_funnel_counts_and_precomputed_cache_matches(tmp_path)`:
  - Evaluate 2 synthetic PNGs via `evaluate_batch_with_bundle(..., bundle=None, min_sharpness=1.3, min_cheek_fft=0.005, min_clip=0.12, tau=0.50)` and assert returned `funnel` keys `{"generated", "gate_passed", "gate_rejected", "hard_samples", "m3_rescued"}` sum consistently (`generated == 2`, `gate_passed + gate_rejected == 2`).
  - Run `inspect_pipeline_sample` once (`compute_heatmaps=False`) on a `(512, 512, 3)` face/synthetic image to extract a `precomputed` payload, re-run with altered `min_sharpness=999.0` passing `precomputed=...`, and assert `gate_passed is False`, `metrics` reflect `999.0`, and `detector_scores` were preserved cleanly without re-running detector.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ui_showcase.py -k "sample_random or batch_worker or evaluate_batch" -v`
Expected: FAIL (`ImportError: cannot import name 'sample_random_live_config'`).

- [ ] **Step 3: Implement helpers and `precomputed` hook in `src/analysis/ui_showcase.py`**

Details:
- Collect valid FFHQ paths cleanly in `_collect_valid_ffhq_paths(catalog)` (checking `Path(...).is_file()`).
- In `sample_random_live_config`: instantiate `PromptGrammarEngine(seed=resolved_seed)` and call `generate_prompt(mode=resolved_mode, seed_offset=resolved_seed % 10000)`.
- In `inspect_pipeline_sample(..., precomputed: Optional[Dict[str, Any]] = None)`:
  - Allow `precomputed` dict to supply cached `rgb_512`, `source_512`, `mask_512`, `yunet_face`, `cheek_boxes`, `biometrics`, `measured_base` (raw measured values before threshold evaluation), `source_with_mask_overlay`, `before_after_html`, `detector_scores`, `heatmaps`, `warnings`.
  - Always recompute `apply_interactive_thresholds`, `_build_curation_metric_items`, `face_audit` overlay (green vs red), `is_hard_sample` wrt current `tau`, and run `score_image_with_bundle` / `compute_dinov2_patch_heatmap` if `should_run_detector` is True and `detector_scores` (or `heatmaps` when `compute_heatmaps=True`) are missing in `precomputed`.

- [ ] **Step 4: Run `tests/test_ui_showcase.py` to verify all pass**

Run: `.venv/bin/pytest tests/test_ui_showcase.py -v`
Expected: All existing + 3 new tests PASS (`18 passed`).

- [ ] **Step 5: Commit**

```bash
git add src/analysis/ui_showcase.py tests/test_ui_showcase.py
git commit -m "feat(analysis): add random/batch live samplers, batch launcher, and inspection cache hook"
```

---

### Task 3: Studio Workspace + Sidebar, Pipeline Flow Ribbon, Semantic Bars & Auto Batch UI (`scripts/app_streamlit.py`)

**Files:**
- Modify: `scripts/app_streamlit.py`
- Modify: `tests/test_ui_showcase.py`

**Interfaces:**
- Consumes: `src.analysis.ui_showcase.*`, `src.analysis.coursework_metrics.collect_live_coursework_metrics`
- Produces: Redesigned `scripts/app_streamlit.py` with:
  - Left `st.sidebar` controls (Device/VRAM pill, 4 Mode buttons, 4 Golden Presets + secondary contrast button, 4 Sliders, Global Scorecard expander).
  - Top `st.tabs(["📂 Kho Held-Out", "⚡ Sinh Live & Auto Batch (SD 1.5)", "📤 Upload Ảnh Lạ"])` source bar:
    - Held-Out horizontal 6-card strip (`1` row + `< Trước` / `Sau >` pagination) with `▶ ĐANG SOI` active highlight.
    - Live & Auto Batch tab: FFHQ thumbnail `(128×128)` preview for paired modes, **`🎲 Random cấu hình`**, **`⚡ Sinh ngẫu nhiên ngay (1-Click)`**, deterministic **`Sinh thử 1 ảnh Live (SD 1.5)`**, plus **Auto Batch (`N = 1..12`, fixed or `mixed`)** with **`🚀 Auto Sinh & Kiểm Định Lô (N mẫu)`** + live batch funnel metrics + clickable `N`-card strip.
    - Upload tab: target & optional paired source uploaders + CLIP prompt + **`Phân tích ảnh upload`**.
  - Active Source Banner above X-Ray Inspector when viewing `live_sd`, `batch_item`, or `upload` (with `Quay lại Kho Held-Out`).
  - **4-Stage Pipeline Flow Ribbon** (`① Input & Synthesis ➔ ② Data Curation Gate ➔ ③ M0 Hard Mining ➔ ④ Detector M3 & Explainability`).
  - **Row 1 (3 Columns)**: Col A (non-duplicated Input↔Output / Wipe stage + metadata), Col B (colored HTML semantic bars via `_render_semantic_bar` + `_metric_progress_ratio`), Col C (inline `force_detector_checkbox` right inside Data-Gate-rejected alert + colored `M0`/`M2*`/`M3` probability bars + rescue/verdict badge).
  - **Row 2 (Full Width `[1, 1]`):** Enlarged `M0` vs `M3` DINOv2 patch sensitivity heatmaps + stage latency caption.
  - Session-keyed LRU cache (`_xray_cache`, max 24 entries) passing `precomputed` into `inspect_pipeline_sample`, `_pending_*` state keys applied cleanly in `_init_session_state`, and `width="stretch"` replacing all `use_container_width=True`.

- [ ] **Step 1: Write failing `AppTest` checks for Studio Sidebar, Random config button, Auto Batch submission, Pipeline Ribbon & inline force toggle**

Append tests to `tests/test_ui_showcase.py`:
- `test_streamlit_studio_v2_random_config_auto_batch_and_pipeline_flow(monkeypatch, tmp_path)`:
  - Mock `launch_live_sd_batch_worker` returning 2 synthetic `(512, 512)` images (`item_000` text2img, `item_001` img2img with source).
  - Boot `AppTest` (`default_timeout=30`), assert `not at.exception`.
  - Click **`🎲 Random cấu hình`** button -> assert `not at.exception` and `live_sd_prompt` / `live_sd_seed` updated cleanly without `StreamlitAPIException`.
  - Click **`🚀 Auto Sinh & Kiểm Định Lô (N mẫu)`** button -> assert `not at.exception`, `at.session_state["batch_results"]` populated with 2 items, active target switched to `batch_item`, and markdown/HTML output contains the 4-stage pipeline flow labels (`Input` / `Data Gate` / `M0` / `M3`).
  - Also verify `_render_semantic_bar_html(...)` returns green (`#16a34a`) for pass, red (`#dc2626`) for fail, and gray (`#64748b`) for `N/A`.

- [ ] **Step 2: Run new `AppTest` to verify it fails**

Run: `.venv/bin/pytest tests/test_ui_showcase.py::test_streamlit_studio_v2_random_config_auto_batch_and_pipeline_flow -v`
Expected: FAIL (`StopIteration` / missing button & `_render_semantic_bar_html`).

- [ ] **Step 3: Implement Studio UI v2 in `scripts/app_streamlit.py`**

Carefully verify all widget keys and `_pending_*` transitions:
- In `_init_session_state()`: handle `_pending_min_clip`, `_pending_live_sd_mode`, `_pending_live_sd_prompt`, `_pending_live_sd_seed`, `_pending_live_sd_source`, `_pending_live_sd_mask_region`, `_pending_force_detector`, plus `gallery_page = 0`, `batch_results = None`, `_xray_cache = {}`.
- Keep `force_detector_checkbox` inside Column C when `not inspection.gate_passed` (using `on_change` or direct `key="force_detector_checkbox"` synced with `st.session_state["force_detector_flag"]`), AND keep a lightweight mirror or default `force_detector_flag = False` when gate passed so state persists cleanly across reruns.
- Replace every `use_container_width=True` with `width="stretch"`.
- Ensure zero `print()` statements and zero raw CV/PyTorch math in `scripts/app_streamlit.py`.

- [ ] **Step 4: Run all UI tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ui_showcase.py -v`
Expected: All tests in `tests/test_ui_showcase.py` PASS (both existing AppTests and new Studio v2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/app_streamlit.py tests/test_ui_showcase.py
git commit -m "feat(ui): redesign Streamlit Studio Workspace v2 with pipeline ribbon, random/batch SD, and semantic bars"
```

---

### Task 4: Full Verification, Live Server Smoke Check & Clean Log Audit

**Files:**
- Verify: `src/`, `scripts/`, `tests/`

- [ ] **Step 1: Audit `src/` for zero `print()` and run full repo quality & test gates**

Run:
```bash
.venv/bin/python -c "import pathlib, re; hits=[str(p) for p in pathlib.Path('src').rglob('*.py') if re.search(r'(?m)^\s*print\(', p.read_text())]; assert not hits, hits; print('Zero print() in src/ verified')" && make lint && make type-check && make test-smoke && make test
```
Expected: All syntax/import checks pass, all smoke tests and full pytest suite pass with 0 failures.

- [ ] **Step 2: Boot live Streamlit server and confirm HTTP 200 with zero `use_container_width` deprecation warnings**

Start `.venv/bin/python -m streamlit run scripts/app_streamlit.py --server.headless true`, fetch `http://localhost:8501`, verify zero `use_container_width` warnings in server output, and leave server running at `http://localhost:8501` for user verification.

---

## Plan Self-Review Notes
- Checked against [2026-11-26-streamlit-ui-v2-studio-design.md](../specs/2026-11-26-streamlit-ui-v2-studio-design.md): Studio Sidebar + top `st.tabs`, 4-stage Pipeline Flow Ribbon, semantic colored bars (`_metric_progress_ratio` preserved), inline `force_detector_checkbox`, full-width `M0 vs M3` Row 2 heatmaps, `PromptGrammarEngine` random & auto-batch (`N = 1..12` with `mixed`), single-worker batch execution, FFHQ thumbnail preview, image-hash `precomputed` cache, and `width="stretch"` are all covered with exact signatures and owning tests.
- Every Review Focus failure mode has explicit test coverage in Tasks 1–3.
