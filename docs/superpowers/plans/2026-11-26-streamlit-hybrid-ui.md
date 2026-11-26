# Streamlit Hybrid Visual UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 1-screen Streamlit Hybrid visual inspection and curation showcase across all 3 generation modes (`Text2Img`, `Img2Img`, `Inpaint`) plus Tab 4 (`OOD RV5.1 & Real FFHQ`), backed by pure modular CV/DL helpers in `src/`, held-out test presets, DINOv2 patch explainability, and an isolated 4GB-VRAM-safe live Stable Diffusion worker.

**Architecture:** All ML/CV domain logic lives in `src/analysis/ui_showcase.py` (catalog loading, detector cache lifecycle, `ImageCurationGate` + separate `cheek_fft_ratio`, YuNet/ROI visual overlays, DINOv2 spatial patch heatmap, before/after HTML wipe builder) and `src/pipelines/live_sd_worker.py` (isolated subprocess for single-image SD 1.5 generation). `scripts/app_streamlit.py` is a thin presentation layer reading live global metrics from `src/analysis/coursework_metrics.py`, pre-synced catalog scores from `experiments/demo_runs/ui_showcase_catalog.json`, and cached PyTorch checkpoints via `@st.cache_resource`.

**Tech Stack:** Python 3.12, PyTorch 2.9 (`dinov2_small`, `torch.amp.autocast`), OpenCV YuNet, Diffusers (`StableDiffusionEngine`), Streamlit (`>=1.38,<2`), pytest (`AppTest` + unit/smoke tests).

**Spec:** [2026-11-26-streamlit-hybrid-ui-design.md](../specs/2026-11-26-streamlit-hybrid-ui-design.md)

## Global Constraints

- Hardware ceiling: `NVIDIA GeForce RTX 3050 Laptop GPU` (`4096 MiB VRAM`, ~3.78 GiB free); never hold SD 1.5 pipelines and DINOv2 detectors on GPU simultaneously.
- No `print()` in `src/`: always use `src.common.logger.get_logger(__name__)`.
- Document `(B, C, H, W)` / `(H, W, C)` shapes and convert OpenCV `BGR` <-> `RGB` explicitly via `cv2.cvtColor`.
- Wrap all detector inference in `model.eval()`, `torch.no_grad()`, and `torch.amp.autocast("cuda", enabled=...)` on `(B, 3, 512, 512)` float32 RGB `[0.0, 1.0]` inputs resized with `Image.Resampling.LANCZOS`.
- Zero mock data or uninitialized random weights in production UI inference (`baseline_d0_best.pt`, `active_learning_m2_best.pt`, `active_learning_m3_best.pt` loaded with `strict=True`).
- Keep `requirements/notebook.txt` untouched; isolate Streamlit dependency in `requirements/ui.txt`.
- No absolute personal home paths in Markdown docs or UI labels.

## Review Focus

- **Non-face / occluded image upload (`detect_face_yunet() is None`)**: `inspect_pipeline_sample` must return `gate_passed=False` with clear reason, render overlay safely without crashing on missing landmarks/cheek boxes, and allow optional forced detector run when `run_detector_if_rejected=True`.
- **Browser uploads with EXIF rotation, RGBA alpha, or arbitrary resolution (e.g. `2048×1536 RGBA PNG`)**: input normalization must apply `ImageOps.exif_transpose()`, convert `.convert("RGB")`, and resize cleanly via LANCZOS to `(512, 512, 3)` `uint8`.
- **Offline / Hub-blocked execution when CLIP or remote SD config lookup fails**: `inspect_pipeline_sample` must catch CLIP load failures gracefully (`clip_score=None`, warning note) and `StableDiffusionEngine` / `live_sd_worker` must retry `local_files_only=True` and resolve cached SD repo aliases (`runwayml/stable-diffusion-v1-5` <-> `stable-diffusion-v1-5/stable-diffusion-v1-5`).
- **Missing paired `source_image` on `OOD RV5.1` or un-paired `img2img`/`inpaint` upload**: inspector must skip paired `ΔE`/`Seam` metrics (`None` / `"N/A"`) and omit the before/after wipe slider rather than crashing or reporting false `0.0` drift.
- **Failed or timed-out `live_sd_worker` subprocess**: `launch_live_sd_worker` must handle non-zero exit codes / timeouts cleanly, raise contextual `RuntimeError` caught by UI banner, and never corrupt Streamlit session state or leak uncollected CUDA tensors.

---

### Task 1: Offline-Safe `StableDiffusionEngine` Cache Fallback & Isolated `live_sd_worker.py`

**Files:**
- Modify: `src/models/diffusion/stable_diffusion_engine.py:313-378`
- Create: `src/pipelines/live_sd_worker.py`
- Test: `tests/test_live_sd_worker.py`

**Interfaces:**
- Consumes: `src.models.diffusion.stable_diffusion_engine.StableDiffusionEngine`, `generate_dynamic_face_mask`, `src.common.metrics.detect_face_yunet`, `src.common.seed.seed_everything`
- Produces:
  - `StableDiffusionEngine._from_pretrained_safe(self, pipe_cls: Any, model_id: str) -> Any` (handles `local_files_only`, cached repo ID fallback, and `cpu_offload`)
  - `resolve_cached_sd_model_id(configured_id: str, mode: str) -> str`
  - `run_single_live_generation(mode: str, prompt: str, output_dir: Path, *, source_image: Optional[Path] = None, mask_region: str = "lips", model_id: Optional[str] = None, steps: int = 20, guidance_scale: float = 6.2, strength: float = 0.30, seed: int = 42, local_files_only: bool = False) -> Dict[str, Any]`
  - CLI `python -m src.pipelines.live_sd_worker` writing `input.png` (for `img2img`/`inpaint`), `mask.png` (for `inpaint`), `output.png`, and `metadata.json`.

- [ ] **Step 1: Write failing tests for offline cache fallback, `inpaint` masking, EXIF/RGBA source loading, and CLI worker execution**

Write `tests/test_live_sd_worker.py` covering:
- `test_load_pipeline_passes_local_files_only_and_handles_cpu_offload(monkeypatch)`: monkeypatch `StableDiffusionPipeline.from_pretrained` inside `StableDiffusionEngine("stable-diffusion-v1-5/stable-diffusion-v1-5", device="cpu", local_files_only=True, cpu_offload=True)` to verify `local_files_only=True` is forwarded and `.enable_model_cpu_offload()` is called when available without raising on CPU.
- `test_live_worker_text2img_img2img_inpaint_mock_artifacts(tmp_path)`: create an RGBA `(640, 480, 4)` PNG source image, invoke `run_single_live_generation(...)` with `model_id="mock_sd"` for each mode (`text2img`, `img2img`, `inpaint` with `mask_region="lips"`), verify `output.png` is `(512, 512)` RGB, `input.png` exists for `img2img`/`inpaint`, `mask.png` exists for `inpaint`, and `metadata.json` records exact keys (`mode`, `model_id`, `prompt`, `seed`, `steps`, `guidance_scale`, `strength`, `mask_region`, `latency_ms`).
- `test_live_worker_cli_requires_source_for_paired_modes(tmp_path)`: verify CLI `parse_args` / `main` fails cleanly with non-empty error when `--mode img2img` omits `--source-image`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_live_sd_worker.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.pipelines.live_sd_worker'` / missing offline fallback).

- [ ] **Step 3: Implement offline/cache fallback & `cpu_offload` in `src/models/diffusion/stable_diffusion_engine.py`**

In `StableDiffusionEngine._switch_mode()` (`src/models/diffusion/stable_diffusion_engine.py:313-378`):
- Add helper `_from_pretrained_safe(self, pipe_cls: Any, model_id: str) -> Any` passing `torch_dtype=self.torch_dtype`, `safety_checker=None`, and `local_files_only=self.local_files_only`; if Hub download fails when `local_files_only=False`, retry once with `local_files_only=True` across alias pair `("stable-diffusion-v1-5/stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5")` for base SD 1.5 before raising.
- When `self.cpu_offload` and `self.target_device.type == "cuda"` and `hasattr(pipe, "enable_model_cpu_offload")`, call `pipe.enable_model_cpu_offload()` instead of `.to(self.target_device)`; otherwise `.to(self.target_device)`.

- [ ] **Step 4: Implement `src/pipelines/live_sd_worker.py`**

Implement `resolve_cached_sd_model_id`, `run_single_live_generation`, `parse_args`, and `main`:
- Read defaults (`steps`, `guidance_scale`, `img2img_strength`, `inpaint_model_id`, `generation.model_id`) from `configs/dataset_factory.yaml` when optional args are omitted; fall back cleanly to `"runwayml/stable-diffusion-v1-5"` if only `models--runwayml--stable-diffusion-v1-5` is present in local Hugging Face cache.
- Normalize uploaded/selected `--source-image` via `ImageOps.exif_transpose(im).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)`.
- For `inpaint`: call `detect_face_yunet(source_rgb)` to get `bbox_xyxy`, pass to `generate_dynamic_face_mask(source_tensor, bbox=bbox, target_region=mask_region)`, save grayscale `(512, 512)` `mask.png`, run `engine.generate_inpaint(...)`.
- Ensure `engine.close()` runs inside a `finally:` block.

- [ ] **Step 5: Run worker tests to verify they pass**

Run: `.venv/bin/pytest tests/test_live_sd_worker.py -v`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/models/diffusion/stable_diffusion_engine.py src/pipelines/live_sd_worker.py tests/test_live_sd_worker.py
git commit -m "feat(pipelines): add isolated live SD worker with offline cache fallback"
```

---

### Task 2: Visual Overlays, Threshold Recalibration & DINOv2 Patch Heatmap in `src/analysis/ui_showcase.py`

**Files:**
- Create: `src/analysis/ui_showcase.py`
- Create: `tests/test_ui_showcase.py`

**Interfaces:**
- Produces:
  - `CurationMetricItem`, `ShowcaseCatalogEntry`, `DetectorBundle`, `InspectionResult` dataclasses
  - `compute_cheek_roi_boxes(yunet_face: Dict[str, Any], width: int = 512, height: int = 512) -> List[Tuple[int, int, int, int]]`
  - `render_face_audit_overlay(image_rgb: np.ndarray, *, yunet_face: Optional[Dict[str, Any]], passed: bool, cheek_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None, mask: Optional[np.ndarray] = None) -> np.ndarray`
  - `render_latent_noise_preview(seed: int, size: int = 256) -> np.ndarray`
  - `build_before_after_wipe_html(before_rgb: np.ndarray, after_rgb: np.ndarray, left_label: str = "INPUT FFHQ", right_label: str = "OUTPUT", height_px: int = 320) -> str`
  - `compute_dinov2_patch_heatmap(detector: torch.nn.Module, image_512_rgb: np.ndarray, device: torch.device) -> Dict[str, Any]` returning `{"prob": float, "heatmap_rgb": np.ndarray, "raw_map": np.ndarray}`
  - `apply_interactive_thresholds(measured: Dict[str, Any], *, mode: str, min_sharpness: float, min_cheek_fft: float, min_clip: float, max_delta_e: float = 18.0, max_seam: float = 35.0) -> Tuple[bool, List[str]]`

- [ ] **Step 1: Write failing smoke tests for overlays, noise preview, wipe HTML, threshold evaluation, and DINOv2 patch heatmap**

Write initial tests in `tests/test_ui_showcase.py` (`@pytest.mark.smoke`):
- `test_render_overlays_handles_yunet_none_and_explicit_rgb_contract()`: test `render_face_audit_overlay` both with `yunet_face=None` (must return `(512, 512, 3)` `uint8` without raising) and with valid `yunet_face` + `mask` (must draw distinct green/red box matching `passed` flag without RGB/BGR channel inversion on pure red/blue landmarks).
- `test_render_noise_and_wipe_html_determinism()`: verify `render_latent_noise_preview(42)` is deterministic, differs from seed `99`, has shape `(256, 256, 3)` `uint8`; verify `build_before_after_wipe_html` embeds two base64 JPEG tags, `clip-path`, and labels cleanly.
- `test_apply_interactive_thresholds_separates_cheek_fft_and_clip_by_mode()`: verify `apply_interactive_thresholds` fails when `cheek_fft_ratio=0.0048 < 0.0050` even if `fft_high_freq_ratio=0.04`, treats missing `delta_e_lab=None` / `seam_diff=None` cleanly without failing un-paired modes (`rv51_ood` / `sd_text2img`), and applies caller's `min_clip` directly on active mode (e.g. rejects `clip_score=0.18` if user drags `min_clip=0.20` on `sd_img2img`).
- `test_compute_dinov2_patch_heatmap_produces_valid_overlay_map()`: build `DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=False)` (or load `baseline_d0_best.pt` if present locally), pass `(512, 512, 3)` `uint8` RGB array, assert returned `heatmap_rgb` has shape `(512, 512, 3)` `uint8`, `0.0 <= prob <= 1.0`, `raw_map.shape == (512, 512)` in `[0.0, 1.0]`, and `model.training is False` afterward.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ui_showcase.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.analysis.ui_showcase'`).

- [ ] **Step 3: Implement dataclasses, visual overlay helpers, threshold logic, and `compute_dinov2_patch_heatmap` in `src/analysis/ui_showcase.py`**

Implement:
- `compute_cheek_roi_boxes`: mirror exact geometry from `src.common.metrics.compute_cheek_skin_fft`.
- `render_face_audit_overlay`: convert RGB->BGR only inside `cv2` draw block and convert back BGR->RGB before returning `(H, W, 3)` `uint8`, drawing bbox, 5 YuNet landmarks, cheek ROI boxes, and optional semi-transparent mask contours.
- `compute_dinov2_patch_heatmap`: resize input RGB `(H, W, 3)` via `Image.Resampling.LANCZOS` to `(512, 512)`, interpolate normalized float tensor to `(1, 3, 504, 504)` (`ImageNet` mean/std), pass `backbone(pixel_values=...)` with gradients enabled on `patch_tokens = outputs.last_hidden_state[:, 1:, :]` (shape `(1, 1296, embed_dim)`) -> normalize CLS & pass through `detector.classifier` -> compute `|∂logit / ∂patch_tokens|` L2 norm across `embed_dim`, reshape `(36, 36)`, bilinear upsample to `(512, 512)`, apply `cv2.applyColorMap(..., cv2.COLORMAP_JET)` (`BGR->RGB`) blended over original RGB (`0.55 * img + 0.45 * jet`); restore `model.zero_grad(set_to_none=True)` and `eval()`.
- Ensure zero `print()` calls (only `logger = get_logger(__name__)`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ui_showcase.py -v`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/ui_showcase.py tests/test_ui_showcase.py
git commit -m "feat(analysis): add UI overlay renderers and DINOv2 patch heatmap helpers"
```

---

### Task 3: Detector Lifecycle, Live `inspect_pipeline_sample`, and `launch_live_sd_worker`

**Files:**
- Modify: `src/analysis/ui_showcase.py`
- Modify: `tests/test_ui_showcase.py`

**Interfaces:**
- Consumes: `src.pipelines.curate_image_dataset.ImageCurationGate`, `src.common.metrics.{detect_face_yunet, extract_face_biometrics_rgb, audit_image_extended, compute_cheek_skin_fft, ClipScorer, resolve_effective_min_clip}`, `src.models.detector.image_detector.DeepfakeImageDetector`
- Produces:
  - `load_detector_bundle(config_path: Union[str, Path] = "configs/config.yaml", checkpoint_map: Optional[Dict[str, str]] = None, preferred_device: str = "auto") -> DetectorBundle`
  - `unload_detector_bundle(bundle: Optional[DetectorBundle]) -> None`
  - `score_image_with_bundle(bundle: DetectorBundle, image_512_rgb: np.ndarray) -> Dict[str, float]`
  - `inspect_pipeline_sample(target_image: Union[Path, Image.Image, np.ndarray], *, bundle: Optional[DetectorBundle] = None, source_image: Optional[Union[Path, Image.Image, np.ndarray]] = None, mask: Optional[np.ndarray] = None, mask_region: Optional[str] = None, mode: str = "sd_img2img", prompt: str = "", seed: Optional[int] = None, min_sharpness: float = 1.3, min_cheek_fft: float = 0.005, min_clip: Optional[float] = None, max_delta_e: float = 18.0, max_seam: float = 35.0, tau: float = 0.50, ground_truth_label: Optional[int] = None, run_detector_if_rejected: bool = False, clip_scorer: Optional[Any] = None, compute_heatmaps: bool = True) -> InspectionResult`
  - `launch_live_sd_worker(mode: str, prompt: str, *, source_image: Optional[Path] = None, mask_region: str = "lips", model_id: Optional[str] = None, steps: int = 20, guidance_scale: float = 6.2, strength: float = 0.30, seed: int = 42, output_root: Path = Path("experiments/demo_runs/streamlit_live"), timeout_s: int = 180) -> Dict[str, Any]`

- [ ] **Step 1: Write failing tests for real checkpoint inspection, non-face rejection, RGBA upload normalization, offline CLIP fallback, un-paired `rv51_ood`, and worker failure handling**

Append tests to `tests/test_ui_showcase.py`:
- `test_load_real_detector_bundle_and_inspect_real_ffhq_and_hard_inpaint()`: skip only if checkpoints/FFHQ missing; load real `DetectorBundle`, inspect real `data/raw/faces/ffhq/00820.png` (`mode="real_ffhq"`, `ground_truth_label=0`), then `fake_ffhq_00820_3.png` paired with `00820.png` (`mode="sd_inpaint"`, `mask_region="lips"`), verify `gate_passed is True`, `cheek_fft_ratio is not None`, `M3` probability returned, `heatmaps["M0"]` and `heatmaps["M3"]` valid `(512, 512, 3)` `uint8`, and `latencies_ms` populated.
- `test_inspect_non_face_rgba_upload_fails_gate_and_honors_force_flag(tmp_path)`: pass random RGBA `(160, 240, 4)` image; verify `gate_passed is False`, `"No valid face"` or geometric failure reason present, `detector_scores` is empty when `run_detector_if_rejected=False`, and `detector_scores` has all 3 keys when `run_detector_if_rejected=True` (with real checkpoints).
- `test_inspect_offline_clip_and_unpaired_rv51_graceful_handling()`: monkeypatch `ClipScorer` load failure so `clip_score is None` without raising, and run `mode="rv51_ood"` with `source_image=None` verifying `delta_e_lab is None`, `seam_diff is None`, `before_after_html is None`.
- `test_launch_live_sd_worker_raises_contextual_error_on_nonzero_exit(monkeypatch, tmp_path)`: mock `subprocess.run` returning exit code `1` with stderr message and assert `launch_live_sd_worker` raises `RuntimeError` containing stderr context rather than silently failing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ui_showcase.py -v`
Expected: FAIL (`ImportError: cannot import name 'load_detector_bundle'`).

- [ ] **Step 3: Implement detector lifecycle, `inspect_pipeline_sample`, and `launch_live_sd_worker` in `src/analysis/ui_showcase.py`**

Key implementation constraints:
- Normalize all image inputs via `_to_512_rgb_u8()` (`ImageOps.exif_transpose` + `.convert("RGB")` + `Image.Resampling.LANCZOS` `(512, 512)`).
- Auto GPU VRAM guard in `load_detector_bundle()`: query `torch.cuda.mem_get_info()` when `cuda` is available; use `cuda` if free memory `>= 900 * 1024 * 1024` bytes, else clean `cpu` fallback so 3 checkpoints never cause 4GB OOM.
- Compute `cheek_fft_ratio` explicitly via `compute_cheek_skin_fft(rgb_512, yunet_face=yunet_res, min_cheek_ratio=min_cheek_fft)` and merge with `audit_image_extended(...)`. Keep `ClipScorer` on CPU inside `try/except Exception` with warning log on offline cache miss.
- `launch_live_sd_worker` checks stderr/returncode on `subprocess.run(..., check=False, timeout=timeout_s)`, clears temp/corrupt dir on failure, reads `metadata.json` on success.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ui_showcase.py -v`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/ui_showcase.py tests/test_ui_showcase.py
git commit -m "feat(analysis): implement live pipeline inspector and VRAM-safe worker launcher"
```

---

### Task 4: Held-Out Catalog Builder, Golden Presets & `sd15_pregen_200` Checkpoint Resync

**Files:**
- Modify: `src/analysis/ui_showcase.py`
- Modify: `tests/test_ui_showcase.py`
- Generate/Update artifacts: `experiments/demo_runs/ui_showcase_catalog.json`, `experiments/demo_runs/sd15_pregen_200/{scored_manifest.jsonl,hard_samples_manifest.jsonl,e2e_summary.json}`

**Interfaces:**
- Produces:
  - `resync_demo_run_scores(demo_dir: Path = Path("experiments/demo_runs/sd15_pregen_200"), threshold: float = 0.50) -> Dict[str, Any]` (re-scores ALL items in `fake_manifest.jsonl` — both curated & rejected — using current `M0/M2*/M3` checkpoints, updates `scored_manifest.jsonl`, `hard_samples_manifest.jsonl`, `e2e_summary.json`)
  - `build_ui_showcase_catalog(output_path: Path = Path("experiments/demo_runs/ui_showcase_catalog.json"), per_mode_pass: int = 18, per_mode_fail: int = 12, ood_count: int = 18, real_count: int = 12) -> Dict[str, Any]`
  - `load_curated_showcase_catalog(catalog_path: Path = Path("experiments/demo_runs/ui_showcase_catalog.json")) -> Dict[str, List[ShowcaseCatalogEntry]]`
  - `get_golden_presets(catalog: Optional[Dict[str, List[ShowcaseCatalogEntry]]] = None) -> List[Dict[str, Any]]`
  - CLI `python -m src.analysis.ui_showcase --build-catalog --resync-demo`

- [ ] **Step 1: Write failing tests for held-out provenance, balanced 4-tab catalog, `cheek_fft_ratio` presence, and `sd15_pregen_200` checkpoint sync**

Append tests to `tests/test_ui_showcase.py`:
- `test_catalog_entries_passed_are_strictly_held_out_and_cover_all_tabs()`: load `ui_showcase_catalog.json` via `load_curated_showcase_catalog()`, assert keys `{"sd_text2img", "sd_img2img", "sd_inpaint", "ood_and_real"}`, each SD 1.5 tab contains both `gate_passed=True` (`split_provenance in {"master_frozen_test", "cumulative_challenge_test"}` — never `d0_train_images`) and `gate_passed=False` entries, `ood_and_real` contains both `rv51_ood` (`unseen_rv51_test`) and `real_ffhq` (`label == 0`), every entry has valid `scores` (`M0`, `M2*`, `M3`), non-null `cheek_fft_ratio` when face detected, and existing image file paths.
- `test_golden_presets_cover_all_four_presentation_scenarios()`: verify `get_golden_presets()` returns 4 presets (`Text2Img`, `Img2Img`, `Inpaint`, `OOD RV5.1 & Real FFHQ`) referencing existing catalog entries and live checkpoint metrics.
- `test_sd15_pregen_200_scores_match_live_checkpoints()`: pick sample `fake_ffhq_00500_4` and 1 rejected sample (`fake_ffhq_00032_1`) from `sd15_pregen_200/scored_manifest.jsonl`, confirm all 200 items exist in `scored_manifest.jsonl` and `fake_ffhq_00500_4`'s recorded `M0/M2*/M3` scores match `score_image_with_bundle` within `1e-3`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ui_showcase.py::test_catalog_entries_passed_are_strictly_held_out_and_cover_all_tabs -v`
Expected: FAIL (`ImportError` / missing catalog).

- [ ] **Step 3: Implement `resync_demo_run_scores`, `build_ui_showcase_catalog`, `load_curated_showcase_catalog`, and `get_golden_presets` in `src/analysis/ui_showcase.py`**

Details:
- `build_ui_showcase_catalog`:
  - Sample held-out PASSED fake records from `data/benchmarks/master_frozen_test.jsonl` across `sd_text2img`, `sd_img2img`, `sd_inpaint` (prioritizing genuine `M0 < 0.5 and M3 >= 0.5` hard cases first so Golden Presets showcase real `M0 -> M3` rescues!), plus failed records from `experiments/demo_runs/sd15_pregen_200/curation_report.jsonl` (unseen by any detector train split).
  - Run `audit_image_extended(..., face_backend="yunet")` + `compute_cheek_skin_fft()` + `ClipScorer` (CPU offline-safe) + `score_image_with_bundle()` using the live `M0`, `M2*`, `M3` checkpoints so catalog metrics, `cheek_fft_ratio`, and detector scores are 100% consistent.
  - Sample `RV5.1 OOD` from `data/benchmarks/D3_clean/unseen_rv51_test.jsonl` (`rv51_img2img` + `rv51_inpaint`) and held-out `FFHQ Real` (`label == 0`) from `master_frozen_test.jsonl` for `ood_and_real`.
- `resync_demo_run_scores`:
  - Re-score ALL items in `sd15_pregen_200/fake_manifest.jsonl` (both passed & rejected, storing `"is_curated": bool`), rewrite `scored_manifest.jsonl`, `hard_samples_manifest.jsonl` (subset of curated with `M0 < threshold`), and update `model_metrics` inside `e2e_summary.json`.

- [ ] **Step 4: Execute catalog build & demo resync via CLI, then run tests**

Run:
```bash
.venv/bin/python -m src.analysis.ui_showcase --resync-demo --build-catalog
.venv/bin/pytest tests/test_ui_showcase.py -v
```
Expected: Catalog & resync finish cleanly; all tests in `tests/test_ui_showcase.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/ui_showcase.py tests/test_ui_showcase.py experiments/demo_runs/sd15_pregen_200/scored_manifest.jsonl experiments/demo_runs/sd15_pregen_200/hard_samples_manifest.jsonl experiments/demo_runs/sd15_pregen_200/e2e_summary.json experiments/demo_runs/ui_showcase_catalog.json
git commit -m "feat(analysis): build held-out UI showcase catalog and resync demo scores"
```
*(Note: confirm whether `experiments/` tracked files match `.gitignore` before `git add`; only commit tracked/intended JSON/JSONL artifacts per `.gitignore` rules.)*

---

### Task 5: Streamlit Environment (`requirements/ui.txt`, `Makefile`) & 1-Screen UI (`scripts/app_streamlit.py`)

**Files:**
- Create: `requirements/ui.txt`
- Modify: `Makefile`
- Create: `scripts/app_streamlit.py`
- Modify: `tests/test_ui_showcase.py`

**Interfaces:**
- Consumes: `src.analysis.ui_showcase.*`, `src.analysis.coursework_metrics.collect_live_coursework_metrics`
- Produces: Streamlit application `scripts/app_streamlit.py` runnable via `make ui`.

- [ ] **Step 1: Add `requirements/ui.txt` and `make ui` target, and install Streamlit into `.venv`**

Create `requirements/ui.txt`:
```text
# Streamlit Hybrid Presentation UI Dependencies (kept separate from requirements/notebook.txt)
streamlit>=1.38.0,<2.0.0
```
Update `Makefile` `.PHONY` & add:
```makefile
ui:
	$(PYTHON) -m streamlit run scripts/app_streamlit.py
```
Install into `.venv`:
Run: `.venv/bin/pip install -r requirements/ui.txt`

- [ ] **Step 2: Write failing Streamlit headless `AppTest` smoke test in `tests/test_ui_showcase.py`**

Add `test_streamlit_app_boots_renders_4_modes_and_handles_sliders()` using `streamlit.testing.v1.AppTest.from_file("scripts/app_streamlit.py", default_timeout=30)`:
- Run `.run()`, verify `not at.exception`.
- Verify 4 mode buttons/tabs exist (`Text2Img`, `Img2Img`, `Inpaint`, `OOD RV5.1 & Real FFHQ`), global Proof-of-Utility expander/table loads live numbers, Golden Presets buttons exist, and changing slider values (`Min Sharpness`, `Min Cheek FFT`, `τ`) re-runs cleanly with `not at.exception`.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ui_showcase.py::test_streamlit_app_boots_renders_4_modes_and_handles_sliders -v`
Expected: FAIL (`FileNotFoundError: scripts/app_streamlit.py`).

- [ ] **Step 4: Implement `scripts/app_streamlit.py`**

Keep UI thin (zero raw CV/PyTorch tensor math in `scripts/app_streamlit.py`):
1. `@st.cache_resource` wrapper for `load_detector_bundle()` + `@st.cache_data` for `load_curated_showcase_catalog()` and `collect_live_coursework_metrics()`.
2. **Presenter Header & Top Bar**:
   - Title + live GPU/CPU & Latency chip (`st.caption` / badge).
   - **4 Golden Preset 1-Click buttons** (`Ca 1: Text2Img`, `Ca 2: Img2Img`, `Ca 3: Inpaint`, `Ca 4: OOD RV5.1 & Real FFHQ`).
   - Collapsible expander **"Scorecard Toàn Cục M0 → M1 → M2* → M3 (Proof of Utility)"** rendering Quad-Vault & Universal 20.4K metrics directly from `collect_live_coursework_metrics()`.
3. **Khối 1 — Mode Selector, Curation Sliders & Visual Gallery + Live Upload / SD 1.5 Sandbox**:
   - Segmented mode switch (`1. Text2Img` · `2. Img2Img` · `3. Inpaint` · `4. OOD RV5.1 & Real FFHQ`), resetting `Min CLIP` slider default smoothly when switching between Text2Img (`0.20`) and Img2Img/Inpaint (`0.12`).
   - Live threshold sliders (`Min Sharpness`, `Min Cheek FFT`, `Min CLIP`, `Ngưỡng τ`) dynamically re-evaluating catalog pass/reject status via `apply_interactive_thresholds()`.
   - Visual batch funnel metrics + filter (`Tất cả` / `✓ Vượt qua` / `✕ Bị loại`, or `RV5.1 OOD` / `FFHQ Real` in Tab 4) + thumbnail grid with clickable `Chọn soi X-Ray` buttons.
   - Collapsible **Upload Ảnh Lạ / Sinh thử 1 ảnh Live (SD 1.5)**:
     - File uploader (`png`, `jpg`, `jpeg`, `webp`) + optional paired source uploader + checkbox *"Chạy thử Detector kể cả khi ảnh bị loại"*.
     - Live SD 1.5 generator form: Mode (`text2img`, `img2img`, `inpaint`), FFHQ source selector, `mask_region` (`lips`, `periocular`, `full`), prompt input, seed. On submit: clear `@st.cache_resource` (`load_detector_bundle.clear()`) + `unload_detector_bundle()` -> call `launch_live_sd_worker()` inside `st.spinner(...)` -> inspect generated output -> re-populate X-Ray inspector below (handling errors gracefully via `st.error`).
4. **Khối 2 — 4-Stage X-Ray Inspector**:
   - **Cột A (Tầng 1 — Mode-Native Input → Output)**: Renders mode-specific input (`Latent Noise + Prompt + CFG/Steps` for `Text2Img`; `FFHQ Source + Strength` for `Img2Img`; `FFHQ Source + Mask Overlay` for `Inpaint`; single OOD/Real source note for Tab 4), toggleable YuNet/ROI/Mask overlay on output, and embedded interactive `build_before_after_wipe_html()` when paired `source_image` exists.
   - **Cột B (Tầng 2 — Phiếu Thẩm Định Data Gate)**: Green/Red verdict banner + 6 progress bars with numeric threshold indicators (`Sharpness`, `Cheek Skin FFT`, `CLIP`, `YuNet Geometry`, `CIELab ΔE`, `Seam / Highlight`).
   - **Cột C (Tầng 3 & 4 — M0 Mining → M0..M3 & Explainability)**: Gate-blocked message if failed (unless force-run checked), False-Positive check if `FFHQ Real (label=0)`, else `M0` Hard-Sample badge (`M0 < τ`), 3-checkpoint probability comparison (`M0`, `M2*`, `M3`), and side-by-side **`M0` vs `M3` DINOv2 Patch Sensitivity Heatmaps** + latency breakdown.

- [ ] **Step 5: Run `AppTest` and live Streamlit boot smoke test**

Run:
```bash
.venv/bin/pytest tests/test_ui_showcase.py -v
```
Also verify real Streamlit server starts cleanly in headless mode and responds HTTP `200` within 10s, then stops cleanly.

- [ ] **Step 6: Commit**

```bash
git add requirements/ui.txt Makefile scripts/app_streamlit.py tests/test_ui_showcase.py
git commit -m "feat(ui): implement Streamlit Hybrid 4-mode visual showcase app"
```

---

### Task 6: Full Repo Verification & Quality Gate

**Files:**
- Verify: `src/`, `scripts/`, `tests/`, `docs/`

- [ ] **Step 1: Audit `src/` for zero `print()` statements and check Markdown/repo invariants**

Run: `.venv/bin/python -c "import pathlib, re; hits=[str(p) for p in pathlib.Path('src').rglob('*.py') if re.search(r'(?m)^\s*print\(', p.read_text())]; assert not hits, hits; print('Zero print() in src/ verified')"`
Expected: `Zero print() in src/ verified`.

- [ ] **Step 2: Run full project quality & regression gates**

Run:
```bash
make lint && make type-check && make test-smoke && make test
```
Expected: All syntax/import checks pass, all smoke tests (`74+`) and full pytest suite (`240+` existing plus new UI & worker tests) pass with 0 failures.

---

## Plan Self-Review Notes
- Verified against spec sections §1–§6: all 4 modes, held-out presets, `cheek_fft_ratio` separation, mode-aware `Min CLIP`, `M0 vs M3` DINOv2 heatmap, HTML before/after wipe slider, isolated 4GB-VRAM `live_sd_worker`, global `coursework_metrics` scorecard, and all 5 Review Focus edge cases have explicit owning tasks and assertions.
- No placeholder steps or inconsistent helper names across tasks.
