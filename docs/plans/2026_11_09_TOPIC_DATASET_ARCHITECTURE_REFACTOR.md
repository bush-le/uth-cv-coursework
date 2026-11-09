# TOPIC_DATASET_ARCHITECTURE_REFACTOR.md — Tái Cấu Trúc Toàn Bộ Kiến Trúc Dữ Liệu & Workspace Khớp 100% Với Đề Tài 11

- **Motivation/Background**: Cấu trúc thư mục `data/` và một số module trong `src/` hiện vẫn mang tính generic theo mẫu Data Engineering truyền thống (raw/interim/processed), chưa phản ánh đúng vòng đời 3 Phase của Đề tài 11 (Stable Diffusion sinh ảnh giả -> Baseline DINOv2 bóc tách Mẫu Khó -> Đóng gói 3 tập đối chứng $D_0, D_1, D_2$).
- **Purpose**: Thiết lập kế hoạch chi tiết từng bước để tái cấu trúc `data/`, dọn dẹp mã thừa trong `src/`, xây dựng 3 Jupyter Notebooks trực quan cho môn học, và đảm bảo 100% kiểm thử tự động pass.
- **Overview Pipeline**: Chuẩn hóa cấu trúc 5 tầng chuẩn Topic 11 (`raw`, `synthetic`, `hard_samples`, `benchmarks`, `samples`) -> Tối ưu pipeline sinh đa chế độ (img2img, inpaint, text2image) -> DINOv2 bóc tách Mẫu Khó -> Đóng gói $D_0, D_1, D_2$ -> 3 Notebooks -> Kiểm thử toàn bộ 55 tests -> Commit.
- **Detailed Plan**: §1 Task 1: Thiết kế cấu trúc thư mục 5 tầng tinh gọn (loại bỏ symlink trùng lặp); §2 Task 2: Script tổ chức dữ liệu tự động; §3 Task 3: Cập nhật config & pipelines sinh đa chế độ; §4 Task 4: Dọn dẹp dead code trong `src/`; §5 Task 5: Xây dựng 3 Notebooks; §6 Task 6: Kiểm định toàn diện & tài liệu hóa; §7 Task 7: Tinh chỉnh kiến trúc dữ liệu thực tế (Canonical 5-Tier Pure Structure).
- **References**: [../PURPOSE.md](../PURPOSE.md), [../phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md](../phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md), [../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md](../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md), [../phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md](../phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md).
- **Created**: 2026-11-09T12:12:00+07:00
- **Last Updated**: 2026-11-09T12:45:00+07:00

---

## Danh Sách Nhiệm Vụ Chi Tiết

### Task 1: Khởi Tạo Cấu Trúc Thư Mục Dữ Liệu 5 Tầng & Nạp Sample Data
- Tạo các thư mục:
  - `data/raw/` (ffhq, celeba_hq, metadata)
  - `data/synthetic/` (text2image, inpainting, img2img, fake_manifest.jsonl, curated_manifest.jsonl)
  - `data/hard_samples/` (images, audit_reports, hard_samples_manifest.jsonl)
  - `data/benchmarks/` (D0_baseline, D1_augmented, D2_hard_boosted)
  - `data/samples/` (real, synthetic, hard_samples, sample_manifest.jsonl)
- Nạp 4 ảnh thật, 4 ảnh giả và 2 ảnh mẫu khó vào `data/samples/` (< 1.5MB) phục vụ kiểm thử nhanh.

### Task 2: Viết Script Tổ Chức Dữ Liệu Tự Động (`scripts/organize_topic_dataset.py`)
- Quét manifest, tự động phân loại ảnh giả theo 3 thư mục `text2image/`, `inpainting/`, `img2img/`.
- Sao chép các ảnh mẫu khó vào `data/hard_samples/images/` kèm báo cáo nhân khẩu học `audit_reports/hard_sample_demographics.json`.
- Xuất các file JSONL phân chia train/val/test cho $D_0$, $D_1$, $D_2$ trong `data/benchmarks/`.

### Task 3: Cập Nhật Config & Pipelines Đồng Bộ Với Cấu Trúc Mới
- Cập nhật `configs/config.yaml` và `configs/dataset/deepfake_image.yaml` với các đường dẫn mới.
- Cập nhật các pipeline `generate_deepfake_images.py`, `curate_image_dataset.py`, `mine_hard_samples.py`, `train.py` để ghi vào cấu trúc mới.

### Task 4: Dọn Dẹp Mã Thừa & Tinh Gọn `src/`
- Loại bỏ các module generic cũ không dùng đến: `src/models/classical/`, `src/models/deep/SimpleCVCNN`, `src/data/dataset.py`, `src/data/datamodule.py`.
- Chuẩn hóa lại export trong `src/data/__init__.py` và `src/models/__init__.py`.
- Cập nhật các unit test liên quan (`test_data.py`, `test_models.py`) để kiểm tra đúng các lớp chính thức (`DeepfakeImageDataset`, `DeepfakeImageDataModule`, `DeepfakeImageDetector`).

### Task 5: Xây Dựng 3 Jupyter Notebooks Phục Vụ Báo Cáo Môn Học
- `notebooks/coursework/01_eda_faces_demographics.ipynb`: Phân tích thống kê 4.000 khuôn mặt thật (tuổi tác, giới tính, kính, góc chụp).
- `notebooks/coursework/02_sd_generation_and_curation.ipynb`: Trực quan hóa ảnh sinh từ 3 chế độ Stable Diffusion và kiểm tra ngưỡng lọc Curation Gate.
- `notebooks/coursework/03_hard_samples_and_benchmark_comparison.ipynb`: Khảo sát trực quan các mẫu khó bypass dinoViT và so sánh kết quả 3 giai đoạn $D_0 \to D_1 \to D_2$.

### Task 6: Kiểm Định Toàn Diện & Cập Nhật Tài Liệu
- Chạy `pytest tests/ -v` xác nhận 100% tests pass (55/55 passed).
- Chạy `python3 -m py_compile` xác nhận không có lỗi cú pháp.
- Cập nhật `data/README.md` với sơ đồ kiến trúc dữ liệu hoàn chỉnh.

### Task 7: Tinh Chỉnh & Làm Sạch Kiến Trúc Thư Mục Dữ Liệu Thực Tế (Hoàn tất)
- Loại bỏ hoàn toàn các thư mục tiền tố số và symlink trùng lặp (`01_raw_faces`, `02_synthetic_sd`, `03_hard_samples`, `04_benchmarks`, `interim`).
- Chuẩn hóa cấu trúc vật lý sạch: `data/raw/`, `data/synthetic/{text2image,inpainting,img2img}`, `data/hard_samples/{images,audit_reports}`, `data/benchmarks/{D0_baseline,D1_augmented,D2_hard_boosted}`, `data/samples/`.
- Tạo sinh ảnh Stable Diffusion v1.5 cân bằng đủ cả 3 chế độ (`img2img`, `inpaint`, `text2img`).
- Cập nhật thư viện ảnh so sánh trực quan `experiments/predictions/real_vs_synthetic_gallery.png`.
- Commit và push lên Git branch `origin/baseline/initial-baseline`.
