# 2026_11_09_DATASET_FIRST_ZERO_HARDCODE_MASTER_PLAN.md — Kế Hoạch Tổng: Nhà Máy Dataset Chuẩn, Không Mock, Không Hardcode

- **Motivation/Background**: Sau khi rà soát toàn bộ repo, đề tài được chốt lại: sản phẩm chính là **bộ dataset khuôn mặt giả sinh bằng mô hình khuếch tán**, kèm tập Mẫu Khó có khả năng vượt mặt mô hình thị giác. Fine-tune downstream chỉ là thử nghiệm phụ để kiểm tra hiệu quả dataset, không thuộc bài toán chính. Rà soát cũng phát hiện các lỗi thật trong pipeline hiện tại (ảnh trùng lặp, metric CLIP giả, fallback ảnh nhiễu, ngưỡng chọn sau khi đã nhìn dữ liệu, tài liệu ghi cứng số liệu và tên model).
- **Purpose**: Kế hoạch thực thi từng task để đưa repo về đúng hướng Dataset-First, mọi tham số nằm trong một file cấu hình duy nhất, không có số liệu hay tên model ghi cứng trong code mặc định và tài liệu.
- **Overview Pipeline**: Task 1 (cấu hình tập trung) → Task 2 (gỡ mọi fallback mock) → Task 3 (sinh ảnh đa dạng, chống trùng, lưu provenance) → Task 4 (Curation Gate dùng metric thật, ngưỡng hiệu chỉnh từ ảnh thật) → Task 5 (đánh giá độ khó thiết kế lại) → Task 6 (đóng gói dataset + file thống kê tự sinh) → Task 7 (gallery & notebooks đọc từ config/stats) → Task 8 (viết lại tài liệu theo hướng Dataset-First, không hardcode) → Task 9 (sinh lại dataset end-to-end và nghiệm thu).
- **Detailed Plan**: §0 Quyết định đã chốt; §1 Lỗi đã xác minh; §2 Cấu trúc file; §3 Task 1–9; §4 Quyết định còn mở.
- **References**: [../PURPOSE.md](../PURPOSE.md), [../../GEMINI.md](../../GEMINI.md), [../../.agents/rules/ZERO_MOCK_DATA_RULE.md](../../.agents/rules/ZERO_MOCK_DATA_RULE.md), [2026_11_09_DATASET_FACTORY_AND_DINOV2_MINING_PLAN.md](2026_11_09_DATASET_FACTORY_AND_DINOV2_MINING_PLAN.md) (được thay thế bởi kế hoạch này).
- **Created**: 2026-11-09T14:35:00+07:00
- **Last Updated**: 2026-11-09T14:35:00+07:00

---

# Dataset-First Zero-Hardcode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Biến repo thành một nhà máy dataset có thể tái lập: ảnh thật → sinh ảnh giả khuếch tán (3 chế độ) → lọc chất lượng bằng metric thật → chấm độ khó bằng thước đo đã kiểm chứng → đóng gói dataset kèm thống kê tự sinh; mọi giá trị đều lấy từ cấu hình.

**Architecture:** Một file cấu hình duy nhất `configs/dataset_factory.yaml`, đọc qua `src/common/config.py`. Logic thuần (seed, dedup, hiệu chỉnh ngưỡng, chấm độ khó) nằm trong `src/`, có test; các pipeline trong `src/pipelines/` chỉ nối các khối lại. CLI chỉ dùng để ghi đè cấu hình, không chứa giá trị mặc định dạng số hay tên model.

**Tech Stack:** Python ≥3.10, PyTorch, Hugging Face `diffusers` + `transformers`, OpenCV, PyYAML, pytest.

**Spec:** §0 bên dưới (các quyết định người dùng đã chốt trong phiên này) + [../../GEMINI.md](../../GEMINI.md).

## §0. Quyết Định Đã Chốt (Người Dùng)

1. **Dataset-First 100%**: sản phẩm là bộ dataset. Fine-tune D0/D1/D2 chỉ là thử nghiệm phụ kiểm tra hiệu quả dataset, nằm ngoài kế hoạch này (sẽ có kế hoạch riêng).
2. **Mẫu Khó** = ảnh giả đủ thật để vượt mặt mô hình thị giác. Không bao giờ được chấm bằng mô hình có trọng số ngẫu nhiên.
3. **Zero-Hardcode**: không ghi cứng số liệu (số ảnh, ngưỡng, tỷ lệ) hay tên model cụ thể trong tài liệu và giá trị mặc định của code. Người dùng có thể đổi mọi thứ.
4. **Zero-Mock**: không dùng dữ liệu giả, ảnh nhiễu hay metric giả trong bất kỳ đường chạy nào ra dataset thật.
5. Giữ nguyên `identity_disjoint=True` theo `source_real_id`.

## §1. Lỗi Đã Xác Minh Trong Code Hiện Tại

| # | Lỗi | Bằng chứng | Task sửa |
| :-- | :-- | :-- | :-- |
| B1 | Ảnh `text2image` trùng hệt nhau (cùng prompt + cùng seed) | `md5sum`: 4 nhóm trùng, 1 nhóm 3 file giống nhau | 3 |
| B2 | `compute_clip_similarity_proxy` dùng vector ngẫu nhiên seed theo `hash(prompt)` làm "embedding văn bản", tức metric giả | `src/common/metrics.py` | 4 |
| B3 | Curation giữ lại bản ghi khi file ảnh **không tồn tại** | nhánh `else` trong `curate_image_manifest` | 2 |
| B4 | Không có kiểm tra khuôn mặt thật, `face_detected` luôn là `True` | `ImageCurationGate.evaluate_image` | 4 |
| B5 | Thiếu ảnh nguồn thì generator dùng `torch.rand` làm ảnh gốc | `generate_deepfake_images.py` | 2 |
| B6 | Dataset trả về tensor toàn số 0 khi thiếu file | `image_dataset.py` | 2 |
| B7 | Thiếu manifest thì miner tự tạo metadata giả `syn_df_{i}` | `mine_hard_samples.py` | 2 |
| B8 | Ngưỡng Mẫu Khó bị ghi cứng thành giá trị chọn **sau khi đã xem** phân bố của chính các ảnh đó (data snooping) | `mine_hard_samples.py` | 5 |
| B9 | Độ khó đo bằng độ giống **ảnh nguồn**, tức đo mức chỉnh sửa ít hay nhiều chứ không đo độ thật. Thiên vị inpaint, gần như loại bỏ text2img | 6/6 Mẫu Khó đều là `sd_inpaint` | 5 |
| B10 | Nạp checkpoint với `strict=False`, head ngẫu nhiên vẫn chạy được mà không báo lỗi | `mine_hard_samples.py` | 5 |
| B11 | Ngưỡng độ nét không khớp thang đo (ảnh thật ra điểm thấp hơn ảnh giả) và bị chỉnh tay khi chạy | log lần chạy curation | 4 |
| B12 | D0 gán nhãn lại ảnh SD thành `baseline`. D0/D1/D2 có cùng số mẫu. Mỗi stage tự chia split riêng nên tập test không cố định | `organize_topic_dataset.py` | 6 |
| B13 | Mirror sang `data/processed`, ảnh bị sao chép phẳng vào gốc `data/synthetic`, ghi cứng `max_real_samples` | nhiều file | 2, 3, 6 |
| B14 | Tài liệu ghi cứng số lượng ảnh, ngưỡng, tên model. Mặt nạ inpaint thực chất là hình chữ nhật cố định nhưng tài liệu lại mô tả là "5 điểm mốc" | `docs/`, `README.md` | 8 |

## Global Constraints

- Mọi giá trị số, đường dẫn dữ liệu và ID model chỉ nằm trong `configs/dataset_factory.yaml`. Code đọc cấu hình. `argparse` mặc định là `None`, nghĩa là "lấy từ config".
- Thiếu key cấu hình bắt buộc → `KeyError` nêu tên key. Không có giá trị dự phòng âm thầm.
- Zero-Mock theo [ZERO_MOCK_DATA_RULE.md](../../.agents/rules/ZERO_MOCK_DATA_RULE.md). `mock_sd` chỉ được dùng khi có `--mock` **và** thư mục output nằm ngoài `data/`.
- Tài liệu trong `docs/`, `README.md`, `data/README.md` nhắc đến tham số bằng **tên key** (vd. `generation.model_id`), không ghi giá trị. Con số chỉ xuất hiện trong file thống kê tự sinh.
- Tuân theo GEMINI.md: type hint PEP 484, `get_logger` thay `print`, `seed_everything`, docstring ghi shape tensor, cấm vòng lặp pixel, notebook không train, tên `.md` viết HOA, đủ header 7 trường, không dùng đường dẫn tuyệt đối.
- `make test` phải pass 100% sau mỗi task.

## Review Focus

1. **Thiếu hoặc hỏng ảnh nguồn khi chạy img2img/inpaint**: bỏ qua mẫu đó, ghi log lý do, không bao giờ thay bằng nhiễu → test ở Task 2.
2. **Prompt và metadata nhân khẩu học giống nhau** giữa nhiều mẫu text2img: vẫn phải ra ảnh khác nhau → test ở Task 3.
3. **Resume sau khi đổi cấu hình sinh ảnh** (đổi model, steps, strength): ảnh cũ sinh với tham số khác không được tái sử dụng → test ở Task 3.
4. **Ảnh giả có ảnh nguồn nằm trong tập tham chiếu ảnh thật**: điểm độ khó bị thổi phồng vì so với chính ảnh gốc → test loại trừ nguồn ở Task 5.
5. **Checkpoint thiếu trọng số head**: phải báo lỗi, không chạy với head ngẫu nhiên → test ở Task 5.

---

## §2. Cấu Trúc File

| File | Trách nhiệm |
| :-- | :-- |
| Create `configs/dataset_factory.yaml` | Nguồn duy nhất cho mọi tham số của nhà máy dữ liệu |
| Create `src/common/config.py` | Nạp và kiểm tra config, gộp giá trị ghi đè từ CLI |
| Create `src/data/provenance.py` | Seed riêng từng mẫu, metadata sinh ảnh, hash nội dung để chống trùng |
| Create `src/models/evaluators/hardness.py` | Trích embedding, điểm real-manifold, hiệu chỉnh ngưỡng, nạp checkpoint đã kiểm chứng |
| Create `src/data/splits.py` | Chia split theo danh tính một lần, dùng chung cho mọi stage |
| Create `.agents/rules/ZERO_HARDCODE_RULE.md` | Quy tắc cấm ghi cứng giá trị |
| Create `tests/test_config.py`, `tests/test_provenance.py`, `tests/test_hardness.py`, `tests/test_splits.py`, `tests/test_docs_zero_hardcode.py` | Test |
| Modify `src/data/image_schema.py` | Thêm trường `generation`, `scores` (tùy chọn, tương thích ngược) |
| Modify `src/pipelines/generate_deepfake_images.py`, `curate_image_dataset.py`, `mine_hard_samples.py`, `evaluate.py` | Đọc config, gỡ mock |
| Modify `src/common/metrics.py`, `src/data/image_dataset.py` | CLIP thật, không trả tensor 0 |
| Modify `scripts/organize_topic_dataset.py`, `build_visual_gallery.py`, `create_coursework_notebooks.py` | Đọc config, xuất file stats |
| Remove keys trùng lặp khỏi `configs/generation/stable_diffusion.yaml` và mục `hard_sample_mining` của `configs/detection/detector_baseline.yaml` | Tránh có hai nguồn cấu hình |

---

### Task 1: Cấu Hình Tập Trung

**Files:** Create `configs/dataset_factory.yaml`, `src/common/config.py`, `tests/test_config.py`

**Interfaces — Produces:**
- `load_config(path: Path) -> Dict[str, Any]`
- `cfg_get(cfg: Dict[str, Any], dotted_key: str) -> Any`: thiếu key thì raise `KeyError(dotted_key)`
- `resolve(cli_value: Optional[Any], cfg: Dict[str, Any], dotted_key: str) -> Any`: giá trị CLI khác `None` thì dùng CLI, ngược lại lấy từ config

**Các key cấu hình (giá trị khởi tạo copy từ default CLI và YAML hiện tại; với key mới, người dùng điền khi review):**
`paths.{raw_manifest, synthetic_dir, fake_manifest, curated_manifest, curation_report, hard_dir, hard_manifest, annotated_manifest, benchmarks_dir, stats_file, gallery_image}` · `seed` · `generation.{model_id, inpaint_model_id, modes, num_per_real, max_real_samples, steps, guidance_scale, img2img_strength, image_size, negative_prompt, inpaint_mask_box}` · `curation.{reference_quantile_sharpness, clip.enabled, clip.model_id, clip.min_score, face_check.backend, dedup}` · `hardness.{method, backbone_id, k_neighbors, reference_quantile, detector_checkpoint, detector_tau, batch_size}` · `benchmarks.{split_ratios, stages}`

- [ ] **Step 1: Viết test fail**

```python
def test_cfg_get_missing_key_raises_named_keyerror(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text("a:\n  b: 1\n")
    cfg = load_config(p)
    assert cfg_get(cfg, "a.b") == 1
    with pytest.raises(KeyError, match="a.c"):
        cfg_get(cfg, "a.c")

def test_resolve_prefers_cli_over_config(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text("x: 3\n")
    cfg = load_config(p)
    assert resolve(None, cfg, "x") == 3
    assert resolve(7, cfg, "x") == 7

def test_repo_factory_config_has_all_required_keys():
    cfg = load_config(Path("configs/dataset_factory.yaml"))
    for key in REQUIRED_KEYS:  # REQUIRED_KEYS là hằng số định nghĩa trong src/common/config.py
        cfg_get(cfg, key)
```

- [ ] **Step 2:** `pytest tests/test_config.py -v` → FAIL (ImportError)
- [ ] **Step 3:** Viết code cho `src/common/config.py` (dùng `yaml.safe_load`, `REQUIRED_KEYS: Tuple[str, ...]` liệt kê toàn bộ key ở trên) và tạo `configs/dataset_factory.yaml`. Xóa các key trùng lặp trong hai file YAML cũ.
- [ ] **Step 4:** `pytest tests/test_config.py -v` → PASS. `make test` → PASS.
- [ ] **Step 5:** `git commit -m "feat(config): add single-source dataset factory config loader"`

---

### Task 2: Gỡ Mọi Fallback Mock (B3, B5, B6, B7, B13)

**Files:** Modify `src/pipelines/generate_deepfake_images.py`, `src/pipelines/curate_image_dataset.py`, `src/pipelines/mine_hard_samples.py`, `src/pipelines/evaluate.py`, `src/data/image_dataset.py`. Test: `tests/test_pipelines.py`, `tests/test_image_data.py`, `tests/test_image_curation.py`

**Interfaces — Produces:** `MissingSourceImageError(FileNotFoundError)` trong `src/pipelines/generate_deepfake_images.py`. Dataset raise `FileNotFoundError` khi đọc từ đĩa mà thiếu file (vẫn đọc từ `cache` được để test).

- [ ] **Step 1: Viết test fail**

```python
def test_dataset_raises_on_missing_file():
    ds = DeepfakeImageDataset([ImageMetadata(image_id="x", label=0, file_path="nope.png")])
    with pytest.raises(FileNotFoundError):
        ds[0]

def test_curation_rejects_missing_file(tmp_path):
    m = tmp_path / "in.jsonl"
    m.write_text(json.dumps({"image_id": "a", "label": 1, "file_path": "nope.png"}) + "\n")
    summary = curate_image_manifest(m, tmp_path / "out.jsonl", ImageCurationGate(...))
    assert summary["kept"] == 0 and summary["rejected"] == 1

def test_generator_skips_img2img_when_source_missing(tmp_path):
    # manifest có 1 bản ghi file_path không tồn tại, mode="img2img", model mock, output ngoài data/
    count = run_image_generation(..., mode="img2img")
    assert count == 0  # bị bỏ qua và log lý do, không sinh từ nhiễu

def test_mock_refuses_output_inside_data_dir(tmp_path):
    with pytest.raises(ValueError, match="mock"):
        run_image_generation(..., output_dir=Path("data/synthetic"), use_mock_fallback=True)

def test_miner_raises_when_manifest_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "--input-manifest", str(tmp_path / "none.jsonl")])
    with pytest.raises(FileNotFoundError):
        mine_main()
```

- [ ] **Step 2:** Chạy các test trên → FAIL
- [ ] **Step 3:** Xóa nhánh `torch.rand`, tensor 0, metadata `syn_df_{i}` và dummy batch trong `evaluate.py`. Xóa toàn bộ mirror sang `data/processed` cùng các fallback đọc từ `data/processed`. Curation từ chối file thiếu với lý do `"missing_file"`.
- [ ] **Step 4:** `make test` → PASS (sửa các test cũ đang dựa vào fallback, chuyển sang dùng `cache`)
- [ ] **Step 5:** `git commit -m "fix(zero-mock): remove noise/zero/fabricated fallbacks across pipelines"`

---

### Task 3: Sinh Ảnh Đa Dạng, Chống Trùng, Lưu Provenance (B1, B13)

**Files:** Create `src/data/provenance.py`, `tests/test_provenance.py`. Modify `src/data/image_schema.py`, `src/pipelines/generate_deepfake_images.py`

**Interfaces — Produces:**
- `sample_seed(base_seed: int, image_id: str, variant: int) -> int`: tất định, dùng `zlib.crc32`, không dùng `hash()` (bị salt theo process)
- `generation_params(cfg: Dict[str, Any], mode: str) -> Dict[str, Any]`: model id, steps, guidance, strength, kích thước, negative prompt
- `file_sha256(path: Path) -> str`
- `ImageMetadata.generation: Optional[Dict[str, Any]] = None`, `ImageMetadata.scores: Optional[Dict[str, float]] = None` (`from_dict` vẫn đọc được bản ghi cũ)

- [ ] **Step 1: Viết test fail**

```python
def test_sample_seed_differs_per_image_and_is_stable():
    assert sample_seed(1, "a", 0) == sample_seed(1, "a", 0)
    assert sample_seed(1, "a", 0) != sample_seed(1, "b", 0)

def test_text2img_same_prompt_yields_distinct_images(tmp_path):
    # 3 bản ghi thật có demographics giống nhau, mode="text2img", model mock, output trong tmp
    run_image_generation(...)
    hashes = {file_sha256(p) for p in (tmp_path / "out" / "text2image").glob("*.png")}
    assert len(hashes) == 3

def test_resume_regenerates_when_generation_params_change(tmp_path):
    run_image_generation(..., steps=2)   # lần 1
    run_image_generation(..., steps=3)   # lần 2, cùng image_id
    rec = json.loads((tmp_path / "fake.jsonl").read_text().splitlines()[0])
    assert rec["generation"]["steps"] == 3

def test_metadata_roundtrip_keeps_generation_and_scores():
    m = ImageMetadata(image_id="a", label=1, generation={"seed": 5}, scores={"s": 0.1})
    assert ImageMetadata.from_dict(m.to_dict()) == m
```

- [ ] **Step 2:** → FAIL
- [ ] **Step 3:** Generator lấy seed từ `sample_seed(...)`, ghi `generation_params` vào từng bản ghi. Resume chỉ tái sử dụng ảnh khi file tồn tại **và** `record.generation` trùng với tham số hiện tại. Bỏ việc sao chép phẳng vào gốc `data/synthetic`. Vùng mặt nạ inpaint lấy từ `generation.inpaint_mask_box` (tỷ lệ theo kích thước ảnh). Mọi tham số còn lại lấy qua `resolve(...)`.
- [ ] **Step 4:** `make test` → PASS
- [ ] **Step 5:** `git commit -m "feat(generation): per-sample seeds, provenance-aware resume, no duplicate outputs"`

---

### Task 4: Curation Gate Dùng Metric Thật, Ngưỡng Hiệu Chỉnh Từ Ảnh Thật (B2, B4, B11)

**Files:** Modify `src/common/metrics.py`, `src/pipelines/curate_image_dataset.py`. Test: `tests/test_image_curation.py`, `tests/test_metrics.py`

**Interfaces — Produces:**
- `calibrate_threshold(reference_scores: Sequence[float], quantile: float) -> float` (trong `src/common/metrics.py`, dùng `numpy.quantile`)
- `ClipScorer(model_id: str, device: str).score(image: torch.Tensor (3,H,W), prompt: str) -> float`: cosine thật từ `transformers.CLIPModel`
- `detect_faces(image_rgb: np.ndarray (H,W,3) uint8, backend: str) -> int`: backend `"opencv_haar"` (`cv2.data.haarcascades`) hoặc `"none"`
- Xóa hẳn `compute_clip_similarity_proxy`
- Curation ghi `paths.curation_report` (một dòng JSON mỗi ảnh: điểm, lý do). Ngưỡng độ nét = `calibrate_threshold(độ nét của ảnh thật cùng kích thước, curation.reference_quantile_sharpness)`

- [ ] **Step 1: Viết test fail**

```python
def test_calibrate_threshold_is_quantile():
    assert calibrate_threshold([1, 2, 3, 4, 5], 0.0) == 1
    assert calibrate_threshold([1, 2, 3, 4, 5], 1.0) == 5

def test_clip_disabled_records_none_not_fake_score(tmp_path):
    report = run_curation_on_one_image(clip_enabled=False)
    assert report["clip_score"] is None and "clip" not in " ".join(report["rejection_reasons"])

def test_duplicate_content_rejected(tmp_path):
    # 2 bản ghi trỏ tới 2 file có nội dung giống hệt nhau
    summary = curate_image_manifest(...)
    assert summary["rejected"] == 1

def test_face_check_none_backend_skips_and_haar_backend_rejects_blank():
    blank = np.zeros((64, 64, 3), np.uint8)
    assert detect_faces(blank, "opencv_haar") == 0
```

- [ ] **Step 2:** → FAIL
- [ ] **Step 3:** Viết code. Chỉ nạp CLIP khi `curation.clip.enabled`. Chấm độ nét ảnh thật và ảnh giả ở cùng `generation.image_size`.
- [ ] **Step 4:** `make test` → PASS
- [ ] **Step 5:** `git commit -m "fix(curation): real CLIP and face checks, reference-calibrated sharpness, dedup"`

---

### Task 5: Đánh Giá Độ Khó Thiết Kế Lại (B8, B9, B10)

**Định nghĩa** (thay hẳn kiểu "giống ảnh nguồn"):
- `real_manifold`: điểm của ảnh giả = trung bình cosine top-k với **tập ảnh thật tham chiếu**, đã loại ảnh nguồn của chính nó. Ngưỡng τ = quantile `hardness.reference_quantile` của phân bố điểm leave-one-out giữa các ảnh thật với nhau. Nghĩa là ảnh giả được tính là khó khi nó nằm gần đa tạp ảnh thật giống như một ảnh thật bình thường. τ được tính lại mỗi lần chạy, không ghi cứng, không chỉnh theo kết quả.
- `detector`: P(fake) < `hardness.detector_tau` từ checkpoint đã được kiểm chứng.
- `source_similarity` chỉ lưu vào `scores` để chẩn đoán (img2img/inpaint), **không dùng để quyết định**.

**Files:** Create `src/models/evaluators/__init__.py`, `src/models/evaluators/hardness.py`, `tests/test_hardness.py`. Modify `src/pipelines/mine_hard_samples.py` (gỡ `mine_hard_samples_by_feature_similarity` và các default 0.45/0.85)

**Interfaces — Produces:**
- `embed_images(extractor: nn.Module, metas: List[ImageMetadata], batch_size: int, device: str, cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor`: shape `(N, D)`, chuẩn hóa L2, chạy dưới `eval()` + `no_grad()`
- `real_manifold_scores(fake_emb: Tensor (N,D), real_emb: Tensor (M,D), k: int, exclude_idx: List[Optional[int]]) -> Tensor (N,)`: vector hóa, che `exclude_idx` bằng `-inf`
- `real_reference_scores(real_emb: Tensor (M,D), k: int) -> Tensor (M,)`: leave-one-out
- `load_verified_checkpoint(model: nn.Module, path: Path, device: str) -> None`: raise `ValueError` nếu không có `model_state_dict` hoặc thiếu bất kỳ key nào của head
- `HardnessReport` (dataclass): `method, tau, tau_source, total, hard_count, hard_rate_by_mode: Dict[str, float], hard_rate_by_group: Dict[str, Dict[str, float]], hard_ids: List[str]`
- `build_report(metas: List[ImageMetadata], scores: torch.Tensor (N,), tau: float, tau_source: str, method: str, higher_is_harder: bool) -> HardnessReport`

- [ ] **Step 1: Viết test fail**

```python
def test_real_manifold_excludes_own_source():
    real = F.normalize(torch.tensor([[1., 0.], [0., 1.]]), dim=-1)
    fake = real[:1].clone()                       # giống hệt ảnh thật số 0
    s = real_manifold_scores(fake, real, k=1, exclude_idx=[0])
    assert torch.isclose(s[0], torch.tensor(0.0)) # chỉ còn so được với ảnh thật số 1

def test_reference_scores_leave_one_out():
    real = F.normalize(torch.eye(3), dim=-1)
    assert torch.allclose(real_reference_scores(real, k=1), torch.zeros(3))

def test_load_verified_checkpoint_rejects_missing_head(tmp_path):
    model = DeepfakeImageDetector(backbone="dinov2_small", pretrained=False)
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("head")}
    torch.save({"model_state_dict": sd}, tmp_path / "c.pt")
    with pytest.raises(ValueError, match="head"):
        load_verified_checkpoint(model, tmp_path / "c.pt", "cpu")

def test_report_breaks_down_by_mode():
    # 2 ảnh giả inpaint + 2 ảnh text2img, embedding dựng sẵn để đúng 1 ảnh mỗi chế độ là khó
    rep = build_report(...)
    assert rep.hard_rate_by_mode == {"sd_inpaint": 0.5, "sd_text2img": 0.5}
```

(Kiểm tra lại tên tiền tố key của head trong `DeepfakeImageDetector` trước khi viết test thứ ba.)

- [ ] **Step 2:** → FAIL
- [ ] **Step 3:** Viết `hardness.py`. Viết lại `mine_hard_samples.main()`: đọc `hardness.*` qua `resolve`, nạp toàn bộ ảnh thật từ `paths.raw_manifest` làm tập tham chiếu, ghi `scores` (`real_manifold`, `source_similarity`) vào bản ghi, ghi `HardnessReport` vào `paths.hard_dir/audit_reports/`. `baseline_score` chỉ dùng cho xác suất của detector.
- [ ] **Step 4:** `make test` → PASS
- [ ] **Step 5:** `git commit -m "feat(hardness): real-manifold scoring with calibrated tau and verified checkpoints"`

---

### Task 6: Đóng Gói Dataset + Thống Kê Tự Sinh (B12, B13)

**Files:** Create `src/data/splits.py`, `tests/test_splits.py`. Modify `scripts/organize_topic_dataset.py`

**Interfaces — Produces:**
- `assign_identity_splits(records: List[ImageMetadata], ratios: Dict[str, float], seed: int) -> Dict[str, str]`: map `group_key → split`, tính **một lần** trên toàn bộ bản ghi, phân tầng theo nhóm có chứa ảnh giả hoặc chỉ có ảnh thật (tái sử dụng logic `_partition_group_keys`)
- `build_stage(records, assignment, stage_spec: Dict[str, Any]) -> Dict[str, List[ImageMetadata]]`: `stage_spec` lấy từ `benchmarks.stages[i]` (`name`, `train_sources`, `train_modes`, `oversample_hard`). Test set **cố định và giống nhau** ở mọi stage
- `compute_dataset_stats(cfg) -> Dict[str, Any]` → ghi ra `paths.stats_file`: số lượng theo nguồn/chế độ/nhóm nhân khẩu học, tỷ lệ qua curation, tỷ lệ Mẫu Khó theo chế độ, τ cùng cách tính, các model id lấy từ provenance

- [ ] **Step 1: Viết test fail**

```python
def test_test_split_identical_across_stages(records_fixture):
    a = assign_identity_splits(records_fixture, {"train": .7, "val": .15, "test": .15}, 0)
    s0 = build_stage(records_fixture, a, {"name": "S0", "train_sources": ["baseline"], ...})
    s2 = build_stage(records_fixture, a, {"name": "S2", "train_sources": ["baseline", "synthetic", "hard_sample"], ...})
    assert [m.image_id for m in s0["test"]] == [m.image_id for m in s2["test"]]

def test_no_identity_crosses_splits(records_fixture):
    a = assign_identity_splits(records_fixture, {...}, 0)
    # mọi bản ghi cùng source_real_id phải cùng split
    ...

def test_stage_never_relabels_data_source(records_fixture):
    s = build_stage(...)
    assert all(m.data_source != "baseline" for m in s["train"] if m.label == 1)

def test_stats_contains_no_inputs_only_computed_counts(tmp_path):
    stats = compute_dataset_stats(cfg_fixture)
    assert set(stats) >= {"counts_by_source", "counts_by_mode", "curation_pass_rate", "hard_rate_by_mode", "tau"}
```

- [ ] **Step 2:** → FAIL
- [ ] **Step 3:** Viết code. Bỏ `max_real_samples` ghi cứng, lấy từ `generation.max_real_samples` (giá trị `null` nghĩa là lấy tất cả). Stage nào có train chỉ một lớp thì log WARNING.
- [ ] **Step 4:** `make test` → PASS
- [ ] **Step 5:** `git commit -m "feat(package): identity splits computed once, config-driven stages, generated stats"`

---

### Task 7: Gallery & Notebooks Đọc Từ Config/Stats

**Files:** Modify `scripts/build_visual_gallery.py`, `scripts/create_coursework_notebooks.py`. Test: `tests/test_visualizer.py`

- [ ] **Step 1: Viết test fail**: `test_notebook_generator_has_no_processed_paths_or_literals`: đọc 3 notebook vừa sinh, assert không chứa `data/processed` và có gọi `load_config(`
- [ ] **Step 2:** → FAIL
- [ ] **Step 3:** Gallery lấy đường dẫn từ `paths.*`, cột Mẫu Khó chọn theo điểm cao nhất của từng chế độ. Notebook chỉ import từ `src/`, đọc `paths.stats_file` để hiển thị, không ghi cứng con số.
- [ ] **Step 4:** `make test` → PASS
- [ ] **Step 5:** `git commit -m "refactor(viz): gallery and notebooks driven by config and generated stats"`

---

### Task 8: Viết Lại Tài Liệu Theo Hướng Dataset-First, Không Hardcode (B14)

**Files:** Rewrite `docs/PURPOSE.md`, `docs/OVERVIEW.md`, `data/README.md`, `README.md`, `docs/coursework/TAXONOMY_AND_DATA_SPEC.md`, `docs/phases/PHASE_01_*.md`, `docs/phases/PHASE_02_*.md`, `docs/phases/PHASE_03_*.md`, `docs/experiments/EXPERIMENT_PROTOCOL.md` (đổi vai trò thành "Phụ lục: Kiểm tra hiệu quả dataset"), `docs/reports/*.md`, `docs/reports/presentation_5min_define_and_design.html`. Create `.agents/rules/ZERO_HARDCODE_RULE.md`, `tests/test_docs_zero_hardcode.py`. Mark các kế hoạch cũ trong `docs/plans/` là "Superseded" (không sửa nội dung lịch sử).

> [!NOTE]
> Các thay đổi chưa commit hiện có trong `docs/PURPOSE.md`, `docs/OVERVIEW.md`, `data/README.md` vẫn còn nhắc tên model cụ thể. Task này viết đè lên chúng.

- [ ] **Step 1: Viết test fail**

```python
DOC_GLOBS = ["docs/*.md", "docs/coursework/*.md", "docs/phases/*.md", "docs/experiments/*.md",
             "docs/reports/*.md", "docs/reports/*.html", "README.md", "data/README.md"]
FORBIDDEN = [r"runwayml/", r"stabilityai/", r"facebook/dinov2", r"stable-diffusion-v\d",
             r"\b\d{1,3}\.\d{3}\s*(ảnh|images)", r"(τ|tau|threshold|ngưỡng)\s*[=≥<>]\s*0?\.\d",
             r"5 điểm mốc"]

def test_docs_contain_no_hardcoded_values_or_model_ids():
    for path in iter_files(DOC_GLOBS):
        text = path.read_text(encoding="utf-8")
        for pat in FORBIDDEN:
            assert not re.search(pat, text), f"{path}: {pat}"

def test_docs_follow_dataset_first_framing():
    text = Path("docs/PURPOSE.md").read_text(encoding="utf-8")
    assert "dataset" in text.lower() and "Proof of Utility" in text
```

- [ ] **Step 2:** → FAIL
- [ ] **Step 3:** Viết lại. Nguyên tắc: sản phẩm = dataset. "Stable Diffusion" chỉ là tên họ mô hình (có trong tên đề tài), model cụ thể ghi là `generation.model_id`. Thước đo độ khó ghi là "mô hình nền tảng thị giác (`hardness.backbone_id`)". Mọi con số thay bằng link tới `paths.stats_file`. Mặt nạ inpaint mô tả đúng với code (`generation.inpaint_mask_box`). Slide HTML vẫn phải giữ đủ các ràng buộc của `tests/test_presentation_deck.py` (11 slide, bảng màu, không chứa số liệu chưa kiểm chứng).
- [ ] **Step 4:** `make test` → PASS
- [ ] **Step 5:** `git commit -m "docs: realign to dataset-first framing with zero hardcoded values"`

---

### Task 9: Sinh Lại Dataset End-to-End & Nghiệm Thu

> [!CAUTION]
> Bước 1 xóa toàn bộ dữ liệu sinh ra trước đây (`data/synthetic/`, `data/hard_samples/`, `data/benchmarks/`). `data/raw/` không bị đụng tới. Cần người dùng xác nhận trước khi chạy.

- [ ] **Step 1:** Sao lưu ra `experiments/runs/<timestamp>_pre_regen/` rồi mới xóa 3 thư mục trên
- [ ] **Step 2:** Người dùng điền `generation.max_real_samples` và các key mới trong `configs/dataset_factory.yaml`
- [ ] **Step 3:** Chạy lần lượt (mọi lệnh chỉ có `--config configs/dataset_factory.yaml`):
  `python3 -m src.pipelines.generate_deepfake_images` → `python3 -m src.pipelines.curate_image_dataset` → `python3 -m src.pipelines.mine_hard_samples` → `python3 -m scripts.organize_topic_dataset` → `python3 -m scripts.build_visual_gallery` → `python3 scripts/create_coursework_notebooks.py`
- [ ] **Step 4: Nghiệm thu** (mọi tiêu chí kiểm tra bằng lệnh, không ước lượng bằng mắt):
  - `md5sum data/synthetic/*/*.png | awk '{print $1}' | sort | uniq -d | wc -l` → `0`
  - `paths.stats_file` tồn tại và có đủ 3 chế độ trong `counts_by_mode`
  - `hard_rate_by_mode` được báo cáo đầy đủ (không yêu cầu phải cân bằng, đây là kết quả đo được)
  - Mọi bản ghi trong manifest đều có `generation` và file ảnh tồn tại
  - `make test-smoke` và `make test` → PASS 100%
- [ ] **Step 5:** `git commit -m "data: regenerate dataset with zero-mock zero-hardcode pipeline"` rồi `git push origin baseline/initial-baseline`

---

## §4. Quyết Định Còn Mở (Người Dùng Chốt — Không Tự Suy Diễn)

1. **Thành phần train của D0** cho thử nghiệm phụ: repo chỉ có ảnh thật, nên D0 không có lớp "fake" nếu không bổ sung nguồn ảnh giả bên ngoài. Task 6 cho phép khai báo stage qua `benchmarks.stages`, việc chọn thành phần cho từng stage để người dùng quyết định khi lập kế hoạch Proof of Utility.
2. **Phương pháp độ khó mặc định**: `real_manifold` (không cần detector đã train) hay `detector` (cần checkpoint thật). Có thể chạy cả hai và lưu cả hai điểm vào `scores`.
3. **Mặt nạ inpaint theo landmark** thay cho hình chữ nhật cấu hình được: nằm ngoài phạm vi kế hoạch này trừ khi người dùng yêu cầu.
