# PLAN — Photorealistic SD Generation, Real-Manifold Hard Mining & 300-Sample Verification

- **Motivation/Background**: Người dùng phản ánh ảnh sinh ra còn trông rất giả (fake, nhựa, méo viền) và tập Mẫu Khó (Hard Samples) bị chọn nhầm các ảnh kém chất lượng do sử dụng detector chưa hội tụ thay vì đa tạp thật. Đồng thời cần mở rộng quy mô từ 30 mẫu lên 300 mẫu để kiểm chứng thực chất toàn bộ pipeline trước khi mở rộng lên 10k+.
- **Purpose**: Cung cấp bản kế hoạch chi tiết, có thể thực thi theo quy trình TDD để: (1) Nâng cấp chất lượng ảnh sinh chân thực cao (Photorealistic Prompting & Soft Feathered Masking); (2) Chuyển đổi phương pháp khai thác Mẫu Khó sang Đa Tạp Thật Khách Quan (`Real-Manifold Proximity`); (3) Thực thi toàn bộ chặng kiểm chứng 300 mẫu (150 thật + 150 giả); (4) Đo lường Proof of Utility với số liệu ROC-AUC và Recall tăng trưởng rõ rệt.
- **Overview Pipeline**: `configs/dataset_factory.yaml` (tuning prompts, negative prompts, manifold method) -> `src/pipelines/generate_deepfake_images.py` -> `src/pipelines/curate_image_dataset.py` -> `src/pipelines/mine_hard_samples.py` -> `scripts/organize_topic_dataset.py` -> `src/pipelines/train.py` -> `evaluate.py` -> `experiments/predictions/real_vs_synthetic_gallery.png`.
- **Detailed Plan**: §1 Phân tích lỗi gốc rễ; §2 Bốn trụ cột nâng cấp kỹ thuật; §3 Năm (05) nhiệm vụ TDD chi tiết; §4 Tiêu chuẩn nghiệm thu (Definition of Done).
- **References**: [docs/PURPOSE.md](docs/PURPOSE.md), [docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md](docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md), [docs/phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md](docs/phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md).
- **Created**: 2026-11-09T18:20:00+07:00
- **Last Updated**: 2026-11-09T18:20:00+07:00

---

## 1. Phân Tích Lỗi Gốc Rễ (Root Cause Analysis)

### Vấn Đề 1: Tại Sao Ảnh Sinh Ra Lại "Rất Rất Tệ & Trông Rất Fake"?
1. **Từ khóa Prompt phản tác dụng**: Trong `generate_deepfake_images.py`, prompt mặc định dùng cụm từ `"photorealistic portrait of a synthetic person"`. Stable Diffusion 1.5 hiểu từ `"synthetic person"` là ma-nơ-canh, búp bê sáp, robot hoặc người nhân tạo, tạo ra làn da bóng nhựa, mắt đờ đẫn, vô hồn.
2. **Từ khóa Img2Img tiêu cực**: Dùng `"subtle facial manipulation"`. Trong tập huấn luyện LAION của SD, từ `"manipulation"` gắn liền với ảnh cắt ghép lỗi, biến dạng, khiến SD sinh thêm nhiễu méo mặt.
3. **Mặt nạ Inpainting dạng khối hộp cứng (`inpaint_mask_box`)**: Hộp chữ nhật tĩnh `[0.3, 0.25, 0.7, 0.75]` che toàn bộ vùng giữa mặt. Khi vẽ lại với biên hòa trộn mỏng, nó tạo ra vết cắt da lệch tông màu lộ rõ giữa vùng trong hộp và vùng tóc/tai ngoài hộp.
4. **Negative Prompt quá sơ sài**: Chưa chặn các token đặc trưng của ảnh 3D, CGI, ma-nơ-canh nhựa (`plastic, doll, 3d render, cartoon, uncanny valley`).

### Vấn Đề 2: Tại Sao Dữ Liệu "Hard Sample Lại Rất Tệ"?
1. **Dùng phương pháp `detector` chưa hội tụ**: Trong `configs/dataset_factory.yaml`, cấu hình đang để `hardness.method: "detector"`. Vì detector baseline chưa được huấn luyện đủ dữ liệu, xác suất dự đoán $P(\text{fake})$ chỉ dao động quanh $0.44 - 0.49$.
2. **Hậu quả**: Bất kỳ ảnh nào vô tình có $P(\text{fake}) < 0.50$ đều bị gắn cờ là Mẫu Khó, kể cả ảnh đó có chất lượng xấu, mờ hoặc méo mặt!
3. **Giải pháp chuẩn khoa học**: Chuyển ngay sang **`Real-Manifold Proximity`** (dùng embedding DINOv2 đo khoảng cách tới toàn bộ ảnh thật tham chiếu, loại trừ ảnh nguồn). Một bức ảnh giả CHỈ được coi là Mẫu Khó khi vector biểu diễn sâu của nó nằm chen lẫn vào đa tạp ảnh người thật ($s_i \ge \tau_{\text{manifold}}$). Điều này bảo đảm 100% mẫu khó phải là ảnh chân thực cao.

---

## 2. Bốn (04) Trụ Cột Nâng Cấp Kỹ Thuật

1. **Photorealistic Prompt Engineering & Anti-Mannequin Negative Prompts**:
   * *T2I*: `"RAW photo, authentic portrait of a person, sharp focus, natural skin texture, natural pores, high fidelity, 8k uhd, dslr, soft studio lighting, 35mm photograph"` kèm thuộc tính nhân khẩu học (tuổi, giới tính, kính).
   * *Inpaint*: `"seamless natural facial features, authentic detailed eyes, natural smile, photorealistic skin texture, highly detailed portrait"`.
   * *Img2Img*: `"high resolution authentic portrait, natural skin texture, detailed facial features, realistic lighting"`, với `strength` cân bằng ở mức `0.32 - 0.36` (giữ nguyên hình khối khuôn mặt, chỉ biến đổi vi mô da).
   * *Negative Prompt*: `"(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, plastic, doll, 3d render, cartoon, anime, artificial, mannequin, fake looking, oversaturated, painting, drawing, sketch"`.
2. **Làm Mịn Biên Mặt Nạ Inpainting (Gaussian / Multi-pass Feathering)**:
   * Thu nhỏ tỷ lệ hộp inpaint tập trung vào vùng đặc trưng (`[0.32, 0.28, 0.68, 0.72]`) và áp dụng làm mịn biên Gaussian/Feathering bán kính rộng (15–20px) để triệt tiêu hoàn toàn đường viền nối.
3. **Khai Thác Mẫu Khó Khách Quan Bằng Đa Tạp DINOv2 (`real_manifold`)**:
   * Thiết lập `hardness.method: "real_manifold"` trong `configs/dataset_factory.yaml`.
   * Trích xuất đặc trưng DINOv2 chuẩn hóa L2, tính cosine similarity top-$k$ tới toàn bộ ảnh thật (loại bỏ ảnh gốc).
   * Hiệu chỉnh ngưỡng $\tau$ tự động qua Leave-One-Out phân vị 50th/60th của tập ảnh thật.
4. **Thực Nghiệm Kiểm Chứng 300 Mẫu (150 Thật + 150 Giả)**:
   * Quy mô: 150 ảnh thật (75 FFHQ, 75 CelebA-HQ) + 150 ảnh giả (50 inpaint, 50 img2img, 50 text2img).
   * Split 70/15/15: Train ~210, Val ~45, Test ~45.
   * Đánh giá Proof of Utility: So sánh $D_0, D_1, D_2$ để chứng minh mức tăng `ROC-AUC` và `Hard-Sample Recall`.

---

## 3. Kế Hoạch Thực Hiện Chi Tiết (TDD Tasks)

### Task 1: Nâng Cấp Bộ Từ Khóa Chân Thực & Mặt Nạ Mịn Trong SD Engine
- **Files**:
  - Sửa: `configs/dataset_factory.yaml`
  - Sửa: `src/pipelines/generate_deepfake_images.py`
  - Kiểm thử: `tests/test_stable_diffusion.py`
- **Chi tiết**:
  - Cập nhật `base_prompt`, `_build_demographic_prompt` và `negative_prompt`.
  - Cấu hình `generation.steps: 15`, `generation.img2img_strength: 0.35`, `generation.max_real_samples: 150`.
  - Đảm bảo mặt nạ inpainting được làm mịn biên (feathering) mềm mại.

### Task 2: Chuyển Đổi Sang Khai Thác Mẫu Khó Bằng Đa Tạp Thật (`real_manifold`)
- **Files**:
  - Sửa: `configs/dataset_factory.yaml` (thiết lập `hardness.method: "real_manifold"`, `reference_quantile: 0.50`)
  - Sửa: `src/pipelines/mine_hard_samples.py`
  - Kiểm thử: `tests/test_hardness.py`, `tests/test_objective_hardness.py`
- **Chi tiết**:
  - Kích hoạt thuật toán `real_manifold_scores` và `real_reference_scores`.
  - Lưu trữ báo cáo chi tiết kèm điểm tương đồng đa tạp `real_manifold` vào `ImageMetadata.scores`.

### Task 3: Chạy Pipeline Sinh 150 Ảnh Giả & Curation Gate
- **Files**:
  - Sinh ảnh: `data/synthetic/img2img/`, `data/synthetic/inpainting/`, `data/synthetic/text2image/`
  - Manifest: `data/synthetic/fake_manifest.jsonl`, `data/synthetic/curated_manifest.jsonl`
  - Kiểm thử: `tests/test_image_curation.py`, `tests/test_image_data.py`
- **Chi tiết**:
  - Chạy `python -m src.pipelines.generate_deepfake_images --config configs/dataset_factory.yaml`.
  - Chạy `python -m src.pipelines.curate_image_dataset --config configs/dataset_factory.yaml`.
  - Xác nhận tỷ lệ duyệt curation $\ge 80\%$.

### Task 4: Khai Thác Mẫu Khó & Đóng Gói 3 Bộ Benchmark ($D_0, D_1, D_2$)
- **Files**:
  - Mẫu khó: `data/hard_samples/hard_samples_manifest.jsonl`
  - Benchmark: `data/benchmarks/D0_baseline/`, `data/benchmarks/D1_augmented/`, `data/benchmarks/D2_hard_boosted/`
  - Thống kê: `data/dataset_statistics.json`
- **Chi tiết**:
  - Chạy `python -m src.pipelines.mine_hard_samples --config configs/dataset_factory.yaml`.
  - Sửa `scripts/organize_topic_dataset.py` để không cắt ngắn tập ảnh thật sai lệch, đóng gói trọn vẹn 150 thật + 150 giả.
  - Phân chia `identity-disjoint` tạo tập test 45 mẫu cố định.

### Task 5: Huấn Luyện Lại & Đo Lường Proof of Utility
- **Files**:
  - Checkpoints: `experiments/checkpoints/`
  - Kết quả so sánh: `experiments/logs/benchmark_comparison.json`
  - Gallery đối chứng: `experiments/predictions/real_vs_synthetic_gallery.png`
- **Chi tiết**:
  - Huấn luyện detector DINOv2 trên $D_0$, $D_1$, $D_2$.
  - Đánh giá trên tập test 45 mẫu độc lập với hiệu chỉnh ngưỡng validation.
  - Xác nhận: $ROC-AUC(D_2) > ROC-AUC(D_0)$ và `Hard-Sample Recall` tăng trưởng rõ rệt.
  - Kết xuất ảnh gallery 16 mẫu thực tế để người dùng kiểm tra thủ công.

---

## 4. Tiêu Chuẩn Nghiệm Thu (Definition of Done)

1. **Chất lượng ảnh sinh**: Ảnh có kết cấu da tự nhiên, không bị bóng nhựa, mắt rõ nét và biên inpaint không bị vết cắt chữ nhật.
2. **Chất lượng Mẫu Khó**: Mẫu khó được chọn dựa trên độ gần đa tạp DINOv2 ($s_i \ge \tau$), loại bỏ hoàn toàn các mẫu dị tật/xấu.
3. **Quy mô dữ liệu**: Đạt đúng **300 mẫu** (150 thật + 150 giả).
4. **Tính hợp lệ của Metrics**:
   - `test.jsonl` có 45 mẫu (không còn là 9 mẫu đồ chơi).
   - `ROC-AUC`, `Precision`, `Recall`, `F1` đều có giá trị thực chất ($> 0.70$).
   - Mức tăng trưởng Proof of Utility: $ROC-AUC(D_2) \ge ROC-AUC(D_1) \ge ROC-AUC(D_0)$.
5. **Toàn bộ 102+ unit test tự động**: Đều vượt qua hoàn toàn (`pytest`).
