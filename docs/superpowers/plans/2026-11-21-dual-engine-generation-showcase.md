# Section 03b: Cơ Chế Sinh Ảnh Dual-Engine (SD 1.5 vs RV5.1) — Kế Hoạch Triển Khai

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bổ sung Mục `03b / CƠ CHẾ SINH ẢNH DUAL-ENGINE` vào [presentation_final_acceptance.html](../../reports/presentation_final_acceptance.html) thông qua [build_acceptance_report_vertical.py](../../../scripts/build_acceptance_report_vertical.py), trình bày trực quan toàn bộ cơ chế 3 chế độ (`Text2Img`, `Img2Img`, `Inpainting`), khác biệt kỹ thuật giữa Bộ A (`SD v1.5 Base`) và Bộ B (`RealisticVision v5.1`), cấu trúc Prompt/Negative Prompt thực tế và minh họa hình học Mask.

**Architecture:** Tách quy trình dựng tài nguyên hình ảnh mới thành một script chuyên biệt tạo 3 biểu đồ/ảnh tổng hợp từ dữ liệu & hàm thật trên đĩa vào `docs/reports/assets/extracted_nb/`, sau đó tích hợp Section `#sec-generation-mechanics` (`03b`) cùng mục lục Sidebar, luồng điều hướng `Alt+ArrowUp/Down`, sơ đồ kiến trúc SVG nội tuyến và các thẻ Prompt Card phân tách màu vào `scripts/build_acceptance_report_vertical.py`. Toàn bộ kiểm chứng đi qua `tests/test_acceptance_presentation.py` (TDD) và `make test` + `make test-smoke` + `make lint`.

**Tech Stack:** Python 3.14, Matplotlib/PIL/OpenCV (tận dụng trực tiếp `generate_dynamic_face_mask` trong `src/models/diffusion/stable_diffusion_engine.py` và `create_mask` trong `scripts/run_generation.py`), HTML5 + CSS Grid + Inline SVG, Pytest + BeautifulSoup4.

**Spec:** Phương án 1 đã được chốt cùng người dùng — trình bày thành mục độc lập trên mạch dọc báo cáo (`#sec-generation-mechanics` ngay sau `#sec-dataset`), hiển thị sẵn toàn bộ phục vụ trình chiếu và in PDF.

## Global Constraints

- Mọi thông số (`steps`, `guidance_scale`, `img2img_strength`, `blur_radius`, `exploration_floor`, số mẫu từng mode) phải khớp 100% với `configs/dataset_factory.yaml`, `src/models/diffusion/stable_diffusion_engine.py`, `src/models/diffusion/prompt_engine.py`, `src/pipelines/generate_deepfake_images.py` và `scripts/run_generation.py`.
- Không dùng ảnh giả lập hay dữ liệu bịa đặt; toàn bộ ảnh ví dụ và thống kê phân bổ mode phải trích từ manifest & thư mục ảnh thật (`data/synthetic/`, `data/active_learning/`, `data/external_benchmarks/synthetic_face_dataset_6k/`).
- Tuân thủ `tests/test_docs_zero_hardcode.py` (không ghi pattern cấm trong tài liệu Markdown) và quy tắc không chứa đường dẫn tuyệt đối nội bộ trong tài liệu `.md`.
- Giữ nguyên chuẩn in `@media print`, kích thước chữ/responsive, chức năng Lightbox phóng to ảnh, và cập nhật danh sách link Sidebar (`nav-link`) đồng bộ để phím `▲ Trước` / `▼ Tiếp` hoạt động đúng thứ tự.

## Review Focus

- Ảnh ghép minh họa mask (`dual_engine_mask_anatomy.png`) không bị cắt mất tiêu đề phụ, nhãn màu hay thanh chú thích opacity alpha `[0, 1]`.
- Đối chiếu trực quan SD 1.5 vs RV5.1 (`dual_engine_visual_3modes.png`) hiển thị rõ sự khác biệt kết cấu da (`visible pores`, ánh sáng RAW ở RV5.1 vs độ mịn/nét tổng quát ở SD 1.5) kèm nhãn tên file gốc & siêu tham số khớp từng engine.
- Sơ đồ SVG luồng 3 chế độ (Text2Img / Img2Img / Inpaint) co giãn đúng trên cả desktop (`1400px`) lẫn màn hình nhỏ (`<= 900px`) và bản in PDF.
- Bảng đối chiếu 1-1 phân biệt rạch ròi điểm chung (Inpaint đều gọi `stable-diffusion-v1-5/stable-diffusion-inpainting`) và điểm khác (`BASE_MODEL`, `steps`, `CFG`, `strength`, `PromptGrammarEngine 7-slot + Softmax Error-Steering` vs `5-D Cartesian Prompt`, `Soft-Mask` vs `Binary Ellipse`).
- Prompt Card trích nguyên văn prompt thực tế trong manifest `curated_manifest.jsonl` / `fake_manifest.jsonl`, không bị tràn khung ngang (`.prompt-code-box` wrap dòng sạch trên mobile).

---

### Task 1: Thêm Kiểm Thử Tự Động Cho Section `03b` Trong `tests/test_acceptance_presentation.py` (TDD)

**Files:**
- Modify: `tests/test_acceptance_presentation.py:22-87`

**Interfaces:**
- Consumes: `docs/reports/presentation_final_acceptance.html`
- Produces: Các assertion mới trong `test_acceptance_presentation_vertical_structure` và `test_acceptance_presentation_scientific_metrics` yêu cầu sự hiện diện của `#sec-generation-mechanics`, 3 ảnh trực quan mới, từ khóa 3 mode (`Text-to-Image`, `Img2Img`, `Inpainting`), `PromptGrammarEngine`, `DPMSolverMultistepScheduler`, và đối chiếu `SD 1.5` vs `RealisticVision v5.1`.

- [ ] **Step 1: Viết bổ sung assertion kiểm tra mục `03b` vào `tests/test_acceptance_presentation.py`**

```python
    # Verify Section 03b Dual-Engine Generation Mechanics
    gen_sec = soup.find("section", id="sec-generation-mechanics")
    assert gen_sec is not None, "Missing #sec-generation-mechanics (03b) section"
    assert len(nav_links) >= 12, f"Expected at least 12 nav links including 03b, found {len(nav_links)}"
```

Và trong `test_acceptance_presentation_scientific_metrics`:

```python
    for term in (
        "PromptGrammarEngine",
        "DPMSolverMultistepScheduler",
        "CALIBRATED_NEGATIVE_PROMPT",
        "2.000",
        "blur_radius",
    ):
        assert term in content, f"Missing generation mechanics technical detail: {term}"
```

- [ ] **Step 2: Chạy pytest để xác nhận test đang FAIL (Red)**

Run: `.venv/bin/pytest tests/test_acceptance_presentation.py -v`
Expected: FAIL với thông báo thiếu `#sec-generation-mechanics` và `Expected at least 12 nav links`.

---

### Task 2: Viết Script Sinh 3 Hình Trực Quan Kỹ Thuật Cho Mục `03b`

**Files:**
- Create: `scripts/generate_dual_engine_mechanics_visuals.py`
- Output:
  - `docs/reports/assets/extracted_nb/dual_engine_mask_anatomy.png`
  - `docs/reports/assets/extracted_nb/dual_engine_visual_3modes.png`
  - `docs/reports/assets/extracted_nb/dual_engine_prompt_quota_evolution.png`

**Interfaces:**
- Consumes:
  - `generate_dynamic_face_mask` (`src/models/diffusion/stable_diffusion_engine.py`)
  - `create_mask` (`scripts/run_generation.py`)
  - Manifests: `data/synthetic/curated_manifest.jsonl`, `data/active_learning/round_1/candidates_raw.jsonl`, `data/active_learning/round_2/candidates_raw.jsonl`, `data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl`
  - Real sample PNGs: `data/synthetic/d0_train_images/` và `data/external_benchmarks/synthetic_face_dataset_6k/`, cùng ảnh thật nguồn từ `docs/reports/assets/samples/real_0.jpg` (hoặc `data/raw/`).
- Produces: 3 ảnh PNG chất lượng cao (`dpi=200`, `bbox_inches='tight'`) tự động kiểm tra không rỗng (> 20 KB) vào `docs/reports/assets/extracted_nb/`.

- [ ] **Step 1: Tạo `scripts/generate_dual_engine_mechanics_visuals.py` gồm 3 hàm dựng hình độc lập**
  - `render_mask_anatomy(out_path: Path)`: Lấy `docs/reports/assets/samples/real_0.jpg`, chạy trực tiếp `generate_dynamic_face_mask(..., blur_radius=18, target_region=r)` cho 3 vùng (`periocular`, `lips`, `full`) và `create_mask((512, 512), r)` cho 2 vùng RV5.1 (`mouth`, `eyes`). Vẽ lưới 2 hàng (Hàng trên: heatmap mask `[0, 1]` + biên Gaussian mờ vs nhị phân sắc cạnh; Hàng dưới: overlay bán trong suốt lên khuôn mặt thật + đường cắt profile cường độ alpha ngang mặt cắt để thấy rõ độ dốc biên của Soft-Mask vs bước nhảy bậc thang của Binary Ellipse).
  - `render_visual_3modes_comparison(out_path: Path)`: Chọn 1 cặp mẫu thật từ `data/synthetic/d0_train_images/` (`sd_inpaint`, `sd_img2img`, `sd_text2img`) và 1 cặp mẫu thật từ `data/external_benchmarks/synthetic_face_dataset_6k/` (`inpaint`, `img2img`, `text2img`), kèm ảnh thật tham chiếu + crop phóng đại 2× vùng kết cấu da/viền môi-mắt để quan sát bằng mắt thường độ mịn của SD 1.5 (`steps=20, CFG=6.2`) so với lỗ chân lông/ánh sáng `RAW photo` của RV5.1 (`steps=25, CFG=7.0`).
  - `render_prompt_and_quota_evolution(out_path: Path)`: Đếm trực tiếp từ 4 manifest (`D0 Curated`, `AL R1 Raw`, `AL R2 Raw`, `RV5.1 6k`) ra biểu đồ 2 panel:
    - Panel trái: Cơ cấu phân bổ 3 chế độ (`Img2Img` / `Inpaint` / `Text2Img`) theo từng giai đoạn — làm nổi bật hiện tượng `PromptGrammarEngine` + Active Learning ở R2 tự dồn `47.2%` (`897/1900`) sang `sd_inpaint` nơi M₀/M₁ yếu nhất, trong khi RV5.1 chia đều `2.000 / 2.000 / 2.000`.
    - Panel phải: Sơ đồ giải phẫu cấu trúc **7 Slot Tổ Hợp** (`subject_age`, `subject_gender`, `optics`, `lighting`, `expression`, `occlusion`, `imperfections`) + `Softmax Error-Steering (floor=0.15)` của SD 1.5 đặt cạnh **Lưới Descartes 5 Chiều** (`10 × 2 × 5 × 5 × 4 = 2.000`) của RV5.1 T2I.

- [ ] **Step 2: Chạy script sinh 3 ảnh PNG**

Run: `.venv/bin/python scripts/generate_dual_engine_mechanics_visuals.py`
Expected: Tạo đủ 3 file PNG trong `docs/reports/assets/extracted_nb/`, mỗi file > 50 KB, không có cảnh báo lỗi layout.

- [ ] **Step 3: Kiểm tra trực quan 3 ảnh vừa sinh bằng `read_file` (image viewer)**

Xác nhận không có chữ đè chữ, không bị cắt lề, số liệu hiển thị trên biểu đồ khớp 100% số đếm từ manifest.

---

### Task 3: Bổ Sung CSS, Sidebar Link Và Section `03b` Vào `scripts/build_acceptance_report_vertical.py`

**Files:**
- Modify: `scripts/build_acceptance_report_vertical.py` (CSS layout + `<nav class="sidebar-nav">` + Section `#sec-generation-mechanics` ngay sau `#sec-dataset`)

**Interfaces:**
- Consumes: 3 ảnh PNG tạo ở Task 2 trong `docs/reports/assets/extracted_nb/` và các mẫu prompt trích từ manifest.
- Produces: `docs/reports/presentation_final_acceptance.html` hoàn chỉnh với tổng cộng 12 section (`01` đến `11` + `03b`).

- [ ] **Step 1: Thêm các class CSS chuyên dụng cho Mục `03b` vào khối `<style>` của `scripts/build_acceptance_report_vertical.py`**
  - `.mode-trio-grid` (grid 3 cột responsive cho `Text-to-Image`, `Image-to-Image`, `Inpainting`).
  - `.engine-diff-badge` (nhãn `SD 1.5 Base` vs `RV5.1 Tả Thực`).
  - `.prompt-anatomy-grid` & `.prompt-code-box` với chip màu theo vai trò từ khóa (`.tok-prefix`, `.tok-demo`, `.tok-optic`, `.tok-light`, `.tok-skin`, `.tok-neg`) để nhìn vào prompt là hiểu ngay cấu trúc ghép từng phần.
  - Đảm bảo quy tắc `@media (max-width: 900px)` và `@media print` áp dụng gọn cho `.mode-trio-grid` và `.prompt-anatomy-grid`.

- [ ] **Step 2: Cập nhật `<nav class="sidebar-nav">` chèn link `3b. Cơ Chế Dual-Engine` vào ngay sau `3. Bộ Dữ Liệu 12.545 Ảnh`**

Kiểm tra script scroll-spy & nút `▲ Trước` / `▼ Tiếp` tự động nhận dạng section thứ tự mới nhờ duyệt DOM `.nav-link` & `.report-section`.

- [ ] **Step 3: Chèn nội dung chi tiết `<section id="sec-generation-mechanics" class="report-section">` ngay sau `</section>` của `#sec-dataset`**

Bao gồm 5 khối trực quan theo mạch logic từ kiến trúc → siêu tham số → Mask → Prompt → hình ảnh:
  1. **Sơ đồ SVG nội tuyến — Nguyên lý 3 Luồng Denoising Trong Không Gian Ẩn `(4, 64, 64)`**:
     - Luồng `Text2Img`: Gaussian Noise -> Cross-Attention Prompt -> 20/25 bước DPMSolver++ Karras -> VAE Decoder (`512×512`).
     - Luồng `Img2Img`: Real Image -> VAE Encoder -> Bơm nhiễu theo `strength` (`0.30` SD1.5 / `0.55` RV5.1) -> Denoise `steps × strength` bước cuối -> VAE Decoder.
     - Luồng `Inpainting`: Real Image + Mask (`1=vùng sửa`) -> `StableDiffusionInpaintPipeline` (`9-channel UNet`) -> Soft-Mask Blend (SD 1.5) / Composite (RV5.1).
  2. **3 Thẻ Cơ Chế Chi Tiết (`.mode-trio-grid`)**: Tóm tắt đầu vào, công thức giữ danh tính / biến đổi, và mục đích đóng góp vào bài toán phát hiện Deepfake của từng mode.
  3. **Bảng Đối Chiếu Toàn Diện 1-1: Bộ A (`SD v1.5 Base`) vs Bộ B (`RealisticVision v5.1`)**:
     - Checkpoint Base vs Inpaint (`stable-diffusion-v1-5/stable-diffusion-inpainting` dùng chung cho Inpaint ở cả 2 nhánh!).
     - Siêu tham số: `steps` (`20` vs `25`), `CFG` (`6.2` vs `7.0`), `img2img_strength` (`0.30`, hạ `<=0.25` nếu `0-6 tuổi`/đeo kính vs `0.55`), Scheduler (`DPMSolverMultistepScheduler` `dpmsolver++` + `karras_sigmas=True`), Seed (`sample_seed` CRC32 vs `seed + idx`), Masking (`generate_dynamic_face_mask` Haar + 3 vùng + `blur_radius=18` vs `create_mask` ellipse nhị phân 2 vùng `mouth`/`eyes`), Chiến lược Prompt & Negative Prompt.
  4. **Trực Quan Hóa Hình Học Mask, So Sánh Da/Biên Thực Tế & Phân Bổ Prompt/Quota**:
     - Nhúng lần lượt 3 `.artifact-row-card` với `dual_engine_mask_anatomy.png`, `dual_engine_visual_3modes.png`, `dual_engine_prompt_quota_evolution.png` kèm chú giải kỹ thuật.
  5. **Phòng Thí Nghiệm Prompt & Negative Prompt Thực Tế (6 Prompt Cards Trích Nguyên Văn từ Manifest)**:
     - Hiển thị từng cặp Prompt thực tế của `Text2Img`, `Img2Img`, `Inpaint` cho cả `SD 1.5` (`PromptGrammarEngine`) và `RV5.1` (`generate_prompts` 5-D / template cố định), tô màu giải phẫu từng cụm từ + hiển thị đầy đủ 2 chuỗi `Negative Prompt` thực tế của từng nhánh.

- [ ] **Step 4: Chạy build lại HTML từ script generator**

Run: `.venv/bin/python scripts/build_acceptance_report_vertical.py`
Expected: Xuất ra `docs/reports/presentation_final_acceptance.html` sạch, in tổng dung lượng KB hợp lệ (`20 KB < size < 500 KB`).

---

### Task 4: Kiểm Định Toàn Diện Sau Hoàn Thiện (Verification Gate)

**Files:**
- Verify: `docs/reports/presentation_final_acceptance.html`, `tests/test_acceptance_presentation.py`, toàn bộ suite test dự án.

- [ ] **Step 1: Chạy test Mục `03b` & smoke test giao diện**

Run: `.venv/bin/pytest tests/test_acceptance_presentation.py -v`
Expected: `4 passed` (Green!).

- [ ] **Step 2: Kiểm tra toàn bộ liên kết cục bộ trong HTML và chạy full gate**

Run: `make test && make test-smoke && make lint`
Expected: Toàn bộ 198 unit tests pass (0 failures), 41 smoke tests pass, AST lint pass.

- [ ] **Step 3: Kiểm tra hiển thị trực tiếp hoặc mở bằng headless browser (nếu khả dụng) / rà soát DOM**

Xác nhận không tràn ngang, tất cả link nội bộ hoạt động đúng và cập nhật ghi chú nghiệm thu.
