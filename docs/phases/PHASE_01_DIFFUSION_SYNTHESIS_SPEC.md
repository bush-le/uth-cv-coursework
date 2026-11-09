# PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md — Giai Đoạn 1: Sinh Ảnh Giả Bằng Stable Diffusion & Lọc Ảnh Lỗi

- **Motivation/Background**: Triển khai giai đoạn khởi đầu của luồng Dataset-First: từ nguồn ảnh khuôn mặt thật tham chiếu, sử dụng Stable Diffusion để tạo ra tập dữ liệu ảnh giả đa phương thức, kiểm soát chất lượng qua bộ lọc tự động và lưu trữ nguồn gốc sinh chi tiết.
- **Purpose**: Quy định đặc tả kỹ thuật cho 3 chế độ sinh ảnh Stable Diffusion (`Text-to-Image`, `Inpainting`, `Image-to-Image`), quy trình sinh seed tất định chống trùng lặp và bộ lọc chất lượng tự động `ImageCurationGate`.
- **Overview Pipeline**: `data/raw/` -> `src/pipelines/generate_deepfake_images.py` -> `src/pipelines/curate_image_dataset.py` -> `data/synthetic/curated_manifest.jsonl`.
- **Detailed Plan**: §1 Chuẩn bị ảnh đầu vào; §2 Ba cơ chế sinh ảnh đa phương thức; §3 Cơ chế seed tất định & Provenance; §4 Bộ lọc chất lượng tự động `ImageCurationGate`.
- **References**: [../PURPOSE.md](../PURPOSE.md), [../OVERVIEW.md](../OVERVIEW.md), `configs/dataset_factory.yaml`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-09T15:15:00+07:00

---

## 1. Chuẩn Bị Ảnh Khuôn Mặt Đầu Vào

Ảnh mặt thật tham chiếu từ FFHQ và CelebA-HQ được chuẩn hóa kích thước theo `generation.image_size` (cấu hình trong `configs/dataset_factory.yaml`) và chuyển đổi không gian màu RGB:
1. **Trích xuất thông tin nhân khẩu học**: Đọc độ tuổi, giới tính và phụ kiện che mặt (kính, mũ) từ metadata gốc để xây dựng câu lệnh mô tả (prompt conditioning).
2. **Khởi tạo vùng mặt nạ inpainting**: Tạo mặt nạ nhị phân tỷ lệ theo kích thước ảnh dựa trên `generation.inpaint_mask_box` và làm mịn biên bằng toán tử 2D pooling nhằm triệt tiêu đường viền nhân tạo.

---

## 2. Ba Cơ Chế Sinh Ảnh Đa Phương Thức

Hệ thống kết hợp cân bằng 3 chế độ sinh ảnh để tập dữ liệu có độ đa dạng hình thái cao nhất:

| Chế độ sinh | Mã nhãn (`manipulation_type`) | Dữ liệu đầu vào | Đặc tính kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Text-to-Image** | `sd_text2img` | Prompt nhân khẩu học | Khởi tạo hoàn toàn từ vector nhiễu ngẫu nhiên trong không gian tiềm ẩn, tạo diện mạo độc lập. |
| **Facial Inpainting** | `sd_inpaint` | Ảnh thật + Mặt nạ + Prompt | Bảo tồn toàn bộ nền và viền đầu, tái tạo chi tiết vùng khuôn mặt mục tiêu. Dạng giả mạo biên giới tinh vi nhất. |
| **Image-to-Image** | `sd_img2img` | Ảnh thật + Prompt + Strength | Giữ khung khuôn mặt thật nhưng biến đổi kết cấu da, biểu cảm và đặc điểm nhân khẩu học. |

---

## 3. Cơ Chế Seed Tất Định & Provenance Tracking

Để đảm bảo khả năng tái lập 100% và ngăn ngừa trùng lặp file:
- Mọi mẫu sinh đều nhận seed riêng biệt tính từ hàm băm CRC32: `sample_seed(base_seed, image_id, variant)`. Tuyệt đối không dùng `hash()` của Python do bị salt ngẫu nhiên giữa các tiến trình.
- Toàn bộ siêu tham số sinh ảnh (`model_id`, `steps`, `guidance_scale`, `img2img_strength`, `seed`) được lưu trực tiếp vào trường `generation` của bản ghi `ImageMetadata`.
- Cơ chế `resume` chỉ bỏ qua khi tệp ảnh tồn tại trên đĩa VÀ tham số `generation` trong manifest trùng khớp hoàn toàn.

---

## 4. Bộ Lọc Chất Lượng Tự Động (`ImageCurationGate`)

Mọi ảnh giả sinh ra đều bắt buộc vượt qua cổng kiểm soát chất lượng tự động:
1. **Lọc độ sắc nét (Laplacian Sharpness)**: Tính phương sai toán tử Laplacian trên ảnh xám. Ngưỡng tối thiểu được tự động hiệu chỉnh động qua quantile thực nghiệm của tập ảnh thật (`curation.reference_quantile_sharpness`).
2. **Kiểm tra diện mạo khuôn mặt (`face_check`)**: Xác thực sự hiện diện của cấu trúc khuôn mặt hợp lệ qua bộ dò `curation.face_check.backend`.
3. **Chống trùng lặp nội dung (`dedup`)**: Tính toán hàm băm cryptographic SHA-256 (`file_sha256`) trên từng tệp ảnh; từ chối ngay lập tức các tệp có nội dung trùng lặp.
4. **Biên bản kiểm định**: Toàn bộ kết quả đánh giá từng ảnh (điểm số, lý do từ chối nếu có) được ghi nhận chi tiết vào `paths.curation_report`.
