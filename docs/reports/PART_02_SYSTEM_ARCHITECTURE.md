# PART_02_SYSTEM_ARCHITECTURE.md — Báo Cáo Phần 2: Thiết Kế Hệ Thống 3 Bước (Nhóm 4)

- **Motivation/Background**: Giải thích chi tiết nhưng đơn giản nhất về cách thiết kế từng bước trong hệ thống của Nhóm 4.
- **Purpose**: Làm nội dung Phần 2 của báo cáo môn học (tương ứng từ `Slide 04` đến `Slide 11`).
- **Overview Pipeline**: Bước 1 (3 cách tạo ảnh SD & 3 bước lọc ảnh lỗi) -> Bước 2 (`dinoViT` tách `Reject` và `Bypass` thành `Hard Sample`) -> Bước 3 (Gộp dữ liệu, khóa không trùng người & so sánh `D0 / D1 / D2`).
- **Detailed Plan**: §1 Thiết kế Bước 1 (Sinh và lọc ảnh SD); §2 Thiết kế Bước 2 (`dinoViT` và Mẫu khó); §3 Thiết kế Bước 3 (Huấn luyện lại và chứng minh `D0/D1/D2`).
- **References**: [ PART_01_PROBLEM_DEFINITION.md ](PART_01_PROBLEM_DEFINITION.md), [ SYSTEM_DEFINE_AND_DESIGN_REPORT.md ](SYSTEM_DEFINE_AND_DESIGN_REPORT.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:40:00+07:00

---

## 1. Thiết Kế Bước 1: Tạo Ảnh Giả Bằng Stable Diffusion & Lọc Ảnh Lỗi

1. **Ba cách tạo ảnh khuôn mặt giả (`Slide 04`)**:
   - **Cách 1 — `Text-to-Image` (`sd_text2img`)**: Từ câu chữ mô tả, vẽ ra một khuôn mặt nhân tạo hoàn toàn mới.
   - **Cách 2 — `Facial Inpainting` (`sd_inpaint`)**: Che vùng mắt hoặc miệng trên ảnh thật rồi vẽ lại vùng đó (nền và khung mặt thật giữ nguyên nên rất khó nhận ra).
   - **Cách 3 — `Image-to-Image` (`sd_img2img`)**: Dựa trên 5 điểm mốc mắt-mũi-miệng của ảnh thật, biến đổi làn da, độ tuổi hoặc ánh sáng.

2. **Đường ống 6 giai đoạn & Lọc ảnh lỗi 3 bước (`Slide 05` & `Slide 06`)**:
   - Quy trình đi thẳng qua 6 khâu: Thu thập ảnh gốc $\rightarrow$ Căn chỉnh mặt & tạo mặt nạ che $\rightarrow$ Sinh ảnh bằng Stable Diffusion $\rightarrow$ Lọc bỏ ảnh lỗi $\rightarrow$ Gán nhãn lưu thông tin $\rightarrow$ Xuất dữ liệu.
   - Bộ lọc tự động loại bỏ ngay những ảnh bị mờ nhòe, ảnh méo mặt (sai lệch 5 điểm mốc) hoặc ảnh không khớp câu mô tả.

---

## 2. Thiết Kế Bước 2: Đưa Qua Baseline `dinoViT` & Tách Mẫu Khó (`Slide 07` & `Slide 08`)

1. **Cách Baseline `dinoViT` kiểm tra ảnh giả**:
   - Chia ảnh mặt thành lưới các ô vuông nhỏ, dùng cơ chế `Self-Attention` để nhìn tổng thể toàn mặt và chấm điểm nghi ngờ giả mạo $P(\text{fake}) \in [0, 1]$.
2. **Hai nhánh kết quả rõ ràng**:
   - **Nhánh 1 — Bị bắt giữ (`Reject`, $P(\text{fake}) \ge 0.5$)**: Mô hình gốc nhận ra được đây là ảnh giả $\rightarrow$ xếp vào nhóm ảnh giả thông thường (`data_source = "synthetic"`).
   - **Nhánh 2 — Vượt rào (`Bypass`, $P(\text{fake}) < 0.5$)**: Ảnh giả tinh vi đánh lừa được `dinoViT` gốc khiến nó đoán nhầm là ảnh thật $\rightarrow$ tự động gom riêng vào nhóm **Mẫu khó (`is_hard_sample = True`, `data_source = "hard_sample"`)**.

---

## 3. Thiết Kế Bước 3: Huấn Luyện Lại & Chứng Minh Tác Dụng (`Slide 09`, `Slide 10`, `Slide 11`)

1. **Gộp 3 nguồn dữ liệu và ưu tiên học Mẫu khó (`Slide 09`)**:
   - Gộp chung: **Dữ liệu gốc (`Data Baseline`) + Ảnh sinh mới (`Syn Data`) + Mẫu khó (`Hard Sample`)**.
   - Khi huấn luyện lại `dinoViT`, nhóm kết hợp 3 cách: (1) Mở khóa các khối cuối của `dinoViT`, (2) Nhân bản cho **Mẫu khó** xuất hiện nhiều hơn trong tập học, và (3) Phạt nặng hơn trong hàm tính lỗi nếu mô hình đoán sai **Mẫu khó**.
2. **Chia tập Học / Thi không trùng người (`Slide 10`)**:
   - Mọi ảnh thật, ảnh `Reject` và ảnh `Bypass` phát sinh từ cùng một người thật (`source_real_id`) được khóa cứng trong cùng một ngăn để không bao giờ lộ mặt người ở tập Học sang tập Thi.
3. **So sánh 3 chế độ `D0 / D1 / D2` để chứng minh tác dụng (`Slide 11`)**:
   - So sánh mô hình học trên `D0` (chỉ dữ liệu gốc), `D1` (+ ảnh mới thường) và `D2` (+ cả mẫu khó). Khi `D2` bắt giữ lại được các mẫu từng vượt rào và đạt `ROC-AUC` cao nhất, tác dụng của dữ liệu được chứng minh hoàn toàn.
