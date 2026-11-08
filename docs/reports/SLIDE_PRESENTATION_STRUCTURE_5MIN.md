# SLIDE_PRESENTATION_STRUCTURE_5MIN.md — 11-Slide Visual Blueprint & Layout Specification

- **Motivation/Background**: Đặc tả cấu trúc bố cục, hệ thống lưới, sơ đồ SVG và luồng lập luận của bộ 11 slides thuyết trình 5 phút sau khi cập nhật theo yêu cầu giảng viên (`Stable Diffusion -> Baseline dinoViT -> Reject/Bypass Hard Sample -> Closed-Loop Fine-Tuning`).
- **Purpose**: Làm bản thiết kế chuẩn đối chiếu trực tiếp với tệp trình chiếu [presentation_5min_define_and_design.html](presentation_5min_define_and_design.html) và kịch bản thuyết trình [PRESENTATION_5MIN_DEFINE_AND_DESIGN.md](PRESENTATION_5MIN_DEFINE_AND_DESIGN.md).
- **Overview Pipeline**: Khung trình chiếu `1920x1080` cố định -> Bảng màu Blue Professional (`#fdfae7`, `#1e2bfa`, `#111111`) -> Chuỗi 11 slides chia thành 4 chặng lập luận khép kín.
- **Detailed Plan**: Đặc tả hệ thống thiết kế (§1) và cấu trúc chi tiết từ `Slide 01` đến `Slide 11` (§2).
- **References**: [presentation_5min_define_and_design.html](presentation_5min_define_and_design.html), [PRESENTATION_5MIN_DEFINE_AND_DESIGN.md](PRESENTATION_5MIN_DEFINE_AND_DESIGN.md), [PART_01_PROBLEM_DEFINITION.md](PART_01_PROBLEM_DEFINITION.md), [PART_02_SYSTEM_ARCHITECTURE.md](PART_02_SYSTEM_ARCHITECTURE.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:22:00+07:00

---

## 1. Hệ Thống Thiết Kế & Bất Biến Hình Học

1. **Khung Khấu Độ Cố Định**: Toàn bộ `Slide 01` đến `Slide 11` nằm trên khung `1920x1080` tự động co giãn theo tỷ lệ màn hình, tuyệt đối không cuộn dọc hay tràn chữ.
2. **Bảng Màu Học Thuật**: Nền kem sáng `#fdfae7`, chữ chính `#111111`, điểm nhấn xanh lam `#1e2bfa` và nền thẻ phụ `#eef0ff`.
3. **Định Danh Nhóm**: Nhãn `"Nhóm thực hiện: Nhóm 4"` xuất hiện nổi bật tại `Slide 01`, `Slide 03`, `Slide 04`, `Slide 10` và chân lộ trình tại `Slide 11`.
4. **Không Chứa Số Liệu Chưa Kiểm Chứng**: Toàn bộ sơ đồ chỉ trình bày kiến trúc định tính, luồng dữ liệu và giao thức thực nghiệm đối chứng minh bạch.

---

## 2. Cấu Trúc Chi Tiết 11 Slides (`Slide 01` – `Slide 11`)

### Slide 01 — Cover & Closed-Loop Vision (Nhóm thực hiện: Nhóm 4)
- **Bố cục**: Chia 2 cột (`cover-grid`).
- **Trái**: Tiêu đề đề tài, định danh **Nhóm thực hiện: Nhóm 4**, và dải SVG 4 nút (`Phase 1: Sinh Dữ Liệu SD` -> `Phase 2: Baseline dinoViT (Reject vs. Bypass)` -> `Gộp Dữ Liệu Base + Syn + Hard` -> `Fine-Tune D0 / D1 / D2`).
- **Phải**: Lộ trình dọc 4 chặng bao quát trọn vẹn 11 slides.

### Slide 02 — 01 / Problem Definition: Sự Dịch Chuyển Miền Tạo Sinh & Khoảng Trống Biên Quyết Định
- **Bố cục**: Chia 2 cột trên (`split-2col`) + 3 thẻ tóm tắt dưới (`card-grid-3`).
- **Trái**: Đối chiếu trực quan giữa ghép mặt cục bộ truyền thống và khử nhiễu toàn cục từ không gian tiềm ẩn của Stable Diffusion.
- **Phải**: Sơ đồ không gian đặc trưng 2D minh họa vùng mù của biên quyết định cũ và sự dịch chuyển sang biên quyết định mở rộng.

### Slide 03 — 02 / Closed-Loop Scope & Boundaries (Nhóm thực hiện: Nhóm 4)
- **Bố cục**: Ma trận đối xứng 2 cột (`split-2col`).
- **Trái (`In-Scope`)**: 4 khối cam kết thực thi của **Nhóm thực hiện: Nhóm 4** (Pha 1 Tạo sinh ảnh 2D SD, Pha 2 `dinoViT` & `Hard Sample`, Khóa bất biến danh tính `source_real_id`, và Tinh chỉnh đối chứng `D0 / D1 / D2`).
- **Phải (`Out-of-Scope`)**: 4 giới hạn loại trừ rõ ràng (`X1` Không sinh âm thanh, `X2` Không huấn luyện lại mạng nền từ đầu, `X3` Không mạo danh cá nhân thật, `X4` Không tối ưu thiết bị biên).

### Slide 04 — 03 / 2D Generation Mechanisms: Ba Chế Độ Tạo Sinh Khuôn Mặt (Nhóm thực hiện: Nhóm 4)
- **Bố cục**: 3 cột ngang bằng nhau (`split-3col-full`).
- **Nội dung**: Minh họa 3 cơ chế tạo sinh bổ trợ của Pha 1:
  1. `01. Text-to-Image (T2I)` (`sd_text2img`) — sinh danh tính nhân tạo mới từ đặc tả văn bản.
  2. `02. Facial Inpainting` (`sd_inpainting`) — chỉnh sửa cục bộ vùng mắt, mũi, miệng qua mặt nạ nhị phân.
  3. `03. Image-to-Image (Img2Img)` (`sd_img2img`) — biến đổi diện mạo trên khung hình học 5 điểm mốc gốc.

### Slide 05 — 04 / 2D System Architecture: Đường Ống 6 Giai Đoạn Dự Kiến
- **Bố cục**: Sơ đồ luồng toàn chiều ngang (`svg viewBox="0 0 1700 490"`) + 3 thẻ nguyên tắc thiết kế bên dưới.
- **Nội dung**: Luồng một chiều từ `Stage 01: Data Acquisition` -> `Stage 02: Face Preprocessing` -> `Stage 03: Stable Diffusion Generation` -> `Stage 04: Quality Control` -> `Stage 05: Labeling & Metadata` -> `Stage 06: Dataset Export` kết nối trực tiếp xuống tác vụ phát hiện hạ nguồn.

### Slide 06 — 05 / 2D Automated Curation Gate: Tiền Xử Lý Hình Học & Phễu Sàng Lọc 3 Tầng
- **Bố cục**: Chia 2 cột (`split-2col`).
- **Trái**: Quy trình `Stage 02` gồm quét khung mặt, căn chỉnh hình học 5 điểm mốc và tách mặt nạ vùng nhạy cảm.
- **Phải**: Phễu lọc tự động `Stage 04` gồm 3 tầng kiểm định (độ sắc nét, cấu trúc giải phẫu khuôn mặt, và độ khớp ngữ nghĩa).

### Slide 07 — 06 / Phase 2: Baseline dinoViT Benchmark & Phân Nhánh Reject / Bypass
- **Bố cục**: Sơ đồ kiến trúc toàn chiều ngang + 3 thẻ cơ chế Pha 2 bên dưới.
- **Nội dung**: Minh họa quá trình đưa ảnh giả từ Pha 1 qua **Baseline `dinoViT`** (chia lưới patch, trích xuất đặc trưng `[CLS]` tự giám sát và tính điểm nghi ngờ giả mạo), sau đó phân nhánh tự động tại cổng quyết định thành `REJECT` (bị phát hiện) và `BYPASS` (vượt rào -> đóng dấu `Hard Sample`).

### Slide 08 — 07 / Hard-Sample Mining Engine: Khai Phá Điểm Mù & Tái Định Hình Biên Quyết Định
- **Bố cục**: Chia 2 cột trên (`split-2col`) + 3 thẻ chiến lược khai thác mẫu khó bên dưới.
- **Trái**: Đồ thị phân bố mật độ điểm dự đoán của Baseline `dinoViT`, làm nổi bật vùng `Bypass Zone` nơi các mẫu giả tinh vi đánh lừa mô hình gốc.
- **Phải**: Sơ đồ dịch chuyển biên quyết định sau khi học từ `Hard Sample` cùng 2 kịch bản mẫu khó tiêu biểu (`Inpainting` cục bộ và `Img2Img` hòa trộn tiềm ẩn).

### Slide 09 — 08 / Phase 3: Closed-Loop Fine-Tuning & Ba Trục Tối Ưu Hóa
- **Bố cục**: Sơ đồ luồng 6 bước kết hợp 3 trục tối ưu toàn chiều ngang + 3 thẻ quản trị thực nghiệm bên dưới.
- **Nội dung**: Quy trình gộp `Data Baseline + Synthetic Data + Hard Sample` và tinh chỉnh `dinoViT` theo 3 trục: `TRỤC 01 / UNFREEZE` (mở khóa các khối Transformer cuối), `TRỤC 02 / SAMPLING` (nhân bản ưu tiên nhóm `Hard Sample`), và `TRỤC 03 / LOSS` (hàm mất mát có trọng số phạt cao cho mẫu `Bypass`).

### Slide 10 — 09 / Dataset Schema & Zero Leakage Vault (Nhóm thực hiện: Nhóm 4)
- **Bố cục**: Chia 2 cột (`split-2col`).
- **Trái**: Phân loại 3 nguồn dữ liệu (`baseline`, `synthetic`, `hard_sample`) và thẻ căn cước mẫu chuẩn hóa gồm 6 trường truy vết (`image_id`, nhãn & chế độ SD, `data_source`, `source_real_id`, điểm Baseline, cờ `is_hard_sample`).
- **Phải**: Sơ đồ Két khóa danh tính (`ZERO SOURCE ID LEAK`) của **Nhóm thực hiện: Nhóm 4**, khóa mọi biến thể chung `source_real_id` vào cùng một phân tập.

### Slide 11 — 10 / Proof of Utility (D0 / D1 / D2) & Roadmap M0->M7
- **Bố cục**: Sơ đồ thực nghiệm đối chứng toàn chiều ngang + 4 thẻ cột mốc `M0 → M7` bên dưới (`card-grid-4`).
- **Nội dung**: Ba chế độ huấn luyện `D0` (`Data Baseline`), `D1` (`Data Baseline + Synthetic Data`), `D2` (`Data Baseline + Synthetic Data + Hard Sample`) hội tụ vào cùng kiến trúc `dinoViT Detector`, đồ thị minh họa kỳ vọng `ROC-AUC` cùng các thước đo chứng minh tác dụng (`Hard-Sample Recall`, mức giảm `Bypass Rate`).
