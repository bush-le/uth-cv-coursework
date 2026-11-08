# SYSTEM_DEFINE_AND_DESIGN_REPORT.md — Báo Cáo Tổng Hợp Đề Tài 11 (Nhóm 4)

- **Motivation/Background**: Tổng hợp toàn bộ nội dung đề tài của Nhóm 4 trong một văn bản duy nhất, viết bằng ngôn ngữ đơn giản, mạch lạc và đúng theo yêu cầu mới của giảng viên.
- **Purpose**: Dùng làm báo cáo chính thức nộp cho giảng viên và làm tài liệu ôn tập trước khi thuyết trình.
- **Overview Pipeline**: Đặt vấn đề -> Bước 1: Sinh ảnh giả bằng Stable Diffusion -> Bước 2: Dùng `dinoViT` gốc tìm Mẫu khó (`Bypass`) -> Bước 3: Gộp dữ liệu & Huấn luyện lại `dinoViT` (`D0 / D1 / D2`) để chứng minh tác dụng.
- **Detailed Plan**: §1 Tóm tắt đề tài; §2 Chi tiết 3 bước thực hiện; §3 Quản lý dữ liệu không trùng người; §4 Cách chấm điểm chứng minh tác dụng.
- **References**: [ ../PURPOSE.md ](../PURPOSE.md), [ PART_01_PROBLEM_DEFINITION.md ](PART_01_PROBLEM_DEFINITION.md), [ PART_02_SYSTEM_ARCHITECTURE.md ](PART_02_SYSTEM_ARCHITECTURE.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:40:00+07:00

---

## 1. Tóm Tắt Đề Tài & Bộ Ba Luận Cứ (Problem ➔ Business ➔ Technical)

- **Tên đề tài**: Tạo bộ dữ liệu khuôn mặt giả bằng Stable Diffusion phục vụ phát hiện Deepfake (*Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*).
- **Mục tiêu cốt lõi**: Không chỉ tạo ra ảnh giả mà phải **chứng minh được bộ dữ liệu tạo ra có tác dụng thực tế** thông qua việc tìm ra những bức ảnh đánh lừa được mô hình gốc `dinoViT` (**Mẫu khó / `Hard Sample`**), sau đó dạy lại cho mô hình để nó bắt được những mẫu khó đó.

### 1.1. Phân Loại Deepfake (What Is Deepfake?)
1. **Deepfake Cũ (Ghép mặt / Face-Swap)**: Cắt dán vùng mặt người này lên người khác, để lại **vết cắt nối mờ hoặc lệch màu da** quanh viền mặt. Các mô hình phát hiện cũ chỉ cần tìm đường viền này là bắt được.
2. **Deepfake Mới (Stable Diffusion)**: Khử nhiễu toàn diện từ không gian tiềm ẩn (`Latent Space`), tái tạo mượt mà cấu trúc da, ánh sáng và chi tiết mặt. **Không có vết cắt dán**, tạo ra "vùng mù" khiến các mô hình phát hiện cũ bị đánh lừa.

### 1.2. Stable Diffusion Là Gì & Tạo Dữ Liệu Dạng Nào?
Stable Diffusion là mô hình khuếch tán tiềm ẩn, hỗ trợ tạo **3 dạng dữ liệu khuôn mặt**:
- **Dạng 1 (`sd_text2img`)**: Sinh mặt người mới 100% từ mô tả chữ (Prompt), mở rộng độ phong phú danh tính.
- **Dạng 2 (`sd_inpainting`)**: Chỉ sửa cục bộ mắt/miệng trên mặt thật, giữ nguyên đầu và tóc. Dạng này tinh vi nhất, đánh trúng điểm mù của mô hình.
- **Dạng 3 (`sd_img2img`)**: Biến đổi diện mạo trên khung 5 điểm mốc của ảnh gốc, tạo cặp đối chứng Gốc – Biến đổi.

### 1.3. Bộ 3 Tầng Luận Cứ
- **Problem (Vấn đề)**: Sự dịch chuyển từ ảnh cắt dán sang ảnh khuếch tán khiến các mô hình phát hiện cũ bị "mù".
- **Business (Nghiệp vụ)**: Đe dọa bảo mật cổng xác thực khuôn mặt **eKYC ngân hàng**, mở tài khoản từ xa và ví điện tử. Doanh nghiệp cần giải pháp chủ động sinh dữ liệu để huấn luyện phòng thủ từ sớm.
- **Technical (Kỹ thuật)**: Chu trình 3 bước khép kín của Nhóm 4: Sinh ảnh SD ➔ Qua dinoViT tìm Mẫu khó (Hard Sample) ➔ Huấn luyện lại trên tập gộp để chứng minh tác dụng thực tế.

---

## 2. Chi Tiết Quy Trình 3 Bước Khép Kín

### Bước 1: Sinh dữ liệu khuôn mặt giả bằng Stable Diffusion (`Phase 1`)
1. Từ ảnh khuôn mặt thật, hệ thống tìm 5 điểm mốc (2 mắt, mũi, 2 khóe miệng) để căn chỉnh khuôn mặt và tạo vùng mặt nạ che mắt/miệng.
2. Dùng **Stable Diffusion** sinh ra ảnh giả theo 3 cách:
   - **`Text-to-Image` (`sd_text2img`)**: Vẽ khuôn mặt mới hoàn toàn từ câu chữ mô tả.
   - **`Facial Inpainting` (`sd_inpaint`)**: Chỉ sửa vùng mắt hoặc miệng trên nền khuôn mặt thật.
   - **`Image-to-Image` (`sd_img2img`)**: Giữ nguyên khung mặt thật nhưng đổi làn da, ánh sáng hoặc biểu cảm.
3. Chạy qua **bộ lọc 3 bước** để loại bỏ ảnh bị mờ, ảnh méo mặt và ảnh sai mô tả.

### Bước 2: Kiểm tra qua Baseline `dinoViT` để tách Mẫu khó (`Phase 2`)
1. Đưa các ảnh giả đạt chuẩn ở Bước 1 vào mô hình phát hiện gốc **Baseline `dinoViT`**.
2. Dựa trên điểm nghi ngờ ảnh giả $P(\text{fake}) \in [0, 1]$ với ngưỡng $0.5$:
   - **Bắt được (`Reject`, $P(\text{fake}) \ge 0.5$)**: Mô hình gốc nhận diện đúng là ảnh giả $\rightarrow$ lưu vào nhóm ảnh giả thông thường (`data_source = "synthetic"`).
   - **Bỏ lọt (`Bypass`, $P(\text{fake}) < 0.5$)**: Ảnh giả đánh lừa được `dinoViT` gốc khiến nó tưởng là ảnh thật $\rightarrow$ đánh dấu ngay là **Mẫu khó (`is_hard_sample = True`, `data_source = "hard_sample"`)**.

### Bước 3: Gộp dữ liệu, Huấn luyện lại (`Fine-Tune`) & Chứng minh tác dụng (`Phase 3`)
1. Gộp cả 3 nguồn dữ liệu lại: **Dữ liệu gốc (`Data Baseline`) + Ảnh sinh mới (`Syn Data`) + Mẫu khó (`Hard Sample`)**.
2. Huấn luyện lại mô hình `dinoViT` với 3 cải tiến:
   - Mở khóa các tầng Transformer cuối cùng của `dinoViT` để học đặc trưng chi tiết mới.
   - Nhân bản tăng số lần xuất hiện của **Mẫu khó** trong tập huấn luyện.
   - Đánh trọng số phạt cao hơn trong hàm tính lỗi nếu mô hình đoán sai **Mẫu khó**.

---

## 3. Quản Lý Dữ Liệu Công Bằng (Không Trùng Người)

- Mỗi bức ảnh đều lưu kèm mã người thật gốc (`source_real_id`).
- Tất cả ảnh thật, ảnh `Reject` và ảnh `Hard Sample` sinh ra từ cùng một người luôn được xếp chung vào một tập (hoặc cùng ở tập Học, hoặc cùng ở tập Thi) để đảm bảo tập Kiểm thử hoàn toàn độc lập.

---

## 4. Cách So Sánh `D0 / D1 / D2` Để Chứng Minh Tác Dụng

Nhóm so sánh công bằng 3 chế độ trên cùng kiến trúc `dinoViT` và cùng tập Kiểm thử độc lập:
- **`D0`**: Chỉ học trên **Dữ liệu gốc (`Data Baseline`)**.
- **`D1`**: Học trên **Dữ liệu gốc + Ảnh sinh mới thông thường (`Syn Data`)**.
- **`D2`**: Học trên **Dữ liệu gốc + Ảnh sinh mới + Mẫu khó (`Hard Sample`)**.
- **Kết luận**: Khi mô hình ở chế độ `D2` bắt giữ lại được các mẫu từng vượt rào (tăng mạnh `Hard-Sample Recall`, giảm `Bypass Rate`) và nâng cao `ROC-AUC`, tác dụng của bộ dữ liệu được chứng minh trọn vẹn.
