# PRESENTATION_5MIN_DEFINE_AND_DESIGN.md — 5-Minute Speaker Script & Presentation Guide (11 Slides)

- **Motivation/Background**: Cung cấp kịch bản thuyết trình 5 phút cho Nhóm thực hiện: Nhóm 4 (Đề tài 11), bám sát định hướng mới của giảng viên về chu trình khép kín 3 pha: Sinh dữ liệu khuôn mặt bằng Stable Diffusion -> Đánh giá qua Baseline `dinoViT` để khai phá mẫu vượt rào (`Bypass` / `Hard Sample`) -> Tinh chỉnh (`Fine-Tune`) trên tập gộp `Data Baseline + Synthetic Data + Hard Sample` để chứng minh tác dụng thực tế.
- **Purpose**: Hướng dẫn trình bày chi tiết từng slide từ `Slide 01` đến `Slide 11`, đồng bộ tuyệt đối với bộ slide trực quan [presentation_5min_define_and_design.html](presentation_5min_define_and_design.html).
- **Overview Pipeline**: Giới thiệu & tầm nhìn khép kín (`Slide 01`–`Slide 03`) -> Pha 1: Nhà máy dữ liệu Stable Diffusion & phễu lọc (`Slide 04`–`Slide 06`) -> Pha 2 & Pha 3: Baseline `dinoViT`, khai phá `Hard Sample` & tinh chỉnh khép kín (`Slide 07`–`Slide 09`) -> Két khóa danh tính & chứng minh tác dụng `D0 / D1 / D2` (`Slide 10`–`Slide 11`).
- **Detailed Plan**: Phân bổ thời lượng, nội dung trình bày miệng (Speaker Script) và điểm nhấn thị giác cho toàn bộ 11 slides.
- **References**: [presentation_5min_define_and_design.html](presentation_5min_define_and_design.html), [SLIDE_PRESENTATION_STRUCTURE_5MIN.md](SLIDE_PRESENTATION_STRUCTURE_5MIN.md), [SYSTEM_DEFINE_AND_DESIGN_REPORT.md](SYSTEM_DEFINE_AND_DESIGN_REPORT.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:20:00+07:00

---

## 1. Tổng Quan Phân Bổ Thời Lượng (11 Slides — 5 Phút)

| Chặng | Slide | Chủ đề cốt lõi | Trọng tâm thông điệp |
| :--- | :--- | :--- | :--- |
| **Chặng 01** | `Slide 01` – `Slide 03` | Bài toán & Phạm vi 3 Pha Khép Kín | Dịch chuyển miền tạo sinh & phạm vi của **Nhóm thực hiện: Nhóm 4** |
| **Chặng 02** | `Slide 04` – `Slide 06` | Phase 1: Sinh Dữ Liệu Khuôn Mặt SD | Ba chế độ `Text-to-Image`, `Facial Inpainting`, `Image-to-Image` & phễu lọc 3 tầng |
| **Chặng 03** | `Slide 07` – `Slide 09` | Phase 2 & 3: `dinoViT` & `Hard Sample` | Kiểm định `Reject` vs. `Bypass` qua `dinoViT`, thu hoạch `Hard Sample` & tinh chỉnh |
| **Chặng 04** | `Slide 10` – `Slide 11` | Két Khóa Danh Tính & Chứng Minh `D0/D1/D2` | Chống rò rỉ theo `source_real_id` & bắt giữ mẫu `Bypass` để chứng minh tác dụng |

---

## 2. Kịch Bản Thuyết Trình Chi Tiết Từng Slide

### Slide 01 — Cover & Closed-Loop Vision (Nhóm thực hiện: Nhóm 4)
- **Điểm nhấn thị giác**: Dải luồng 4 nút ở trung tâm (`Phase 1: Sinh Dữ Liệu SD` -> `Phase 2: Baseline dinoViT (Reject vs. Bypass)` -> `Gộp Dữ Liệu Base + Syn + Hard` -> `Fine-Tune D0 / D1 / D2`) cùng lộ trình 4 chặng bên phải.
- **Lời thoại gợi ý**:
  > "Xin chào thầy và các bạn. **Nhóm thực hiện: Nhóm 4** xin trình bày Đề tài 11: *Tạo sinh bộ dữ liệu khuôn mặt tổng hợp dựa trên Stable Diffusion phục vụ phát hiện Deepfake*. Tiếp thu ý kiến định hướng của thầy, nhóm thiết kế hệ thống theo chu trình khép kín 3 pha: Pha 1 sinh dữ liệu giả mạo bằng Stable Diffusion; Pha 2 đưa qua mô hình chuẩn `dinoViT` để phân tách mẫu bị bắt (`Reject`) và mẫu vượt rào (`Bypass` — tức `Hard Sample`); Pha 3 kết hợp `Data Baseline + Synthetic Data + Hard Sample` để tinh chỉnh lại `dinoViT`, từ đó chứng minh trực tiếp tác dụng của bộ dữ liệu."

### Slide 02 — 01 / Problem Definition: Phân Loại Deepfake & Bộ Ba Luận Cứ Problem – Business – Technical
- **Điểm nhấn thị giác**: Phân loại Deepfake trực quan giữa ghép mặt cũ và Stable Diffusion mới, biểu đồ vùng mù biên quyết định, cùng 3 thẻ tóm tắt chuẩn Problem ➔ Business ➔ Technical ở chân slide.
- **Lời thoại gợi ý**:
  > "Tại `Slide 02`, nhóm đặt bài toán qua 3 tầng luận cứ rõ ràng:
  > Thứ nhất là **Problem**: Phân loại Deepfake hiện nay đã chuyển từ thế hệ cắt dán lộ viền sang thế hệ Stable Diffusion vẽ mượt toàn bộ, tạo ra 'vùng mù' khiến các mô hình phát hiện cũ bị đánh lừa.
  > Thứ hai là **Business**: Nguy cơ này đe dọa trực tiếp các cổng xác thực khuôn mặt eKYC ngân hàng, ví điện tử và an ninh danh tính cá nhân. Doanh nghiệp cần giải pháp phòng thủ chủ động để đón đầu hiểm họa.
  > Thứ ba là **Technical**: Nhóm 4 giải quyết bằng chu trình 3 bước khép kín: Dùng Stable Diffusion tạo ảnh giả ➔ Đưa qua mô hình chuẩn dinoViT để tìm Mẫu khó (Hard Sample) ➔ Huấn luyện lại trên tập gộp để chứng minh dữ liệu có tác dụng thực tế."

### Slide 03 — 02 / Closed-Loop Scope & Boundaries (Nhóm thực hiện: Nhóm 4)
- **Điểm nhấn thị giác**: Ma trận đối xứng 2 cột: Cột trái nêu 4 cam kết thực thi của **Nhóm thực hiện: Nhóm 4**; Cột phải khóa chặt 4 giới hạn loại trừ (`Out-of-Scope`).
- **Lời thoại gợi ý**:
  > "Sang `Slide 03`, **Nhóm thực hiện: Nhóm 4** xác lập rõ phạm vi thực hiện gồm: tạo sinh ảnh tĩnh 2D đa chế độ, khai phá mẫu vượt rào qua Baseline `dinoViT`, khóa bất biến danh tính gốc và thực nghiệm đối chứng `D0 / D1 / D2`. Đồng thời, nhóm chủ động loại trừ sinh âm thanh, không huấn luyện lại mạng nền từ đầu và tuân thủ nghiêm ngặt đạo đức dữ liệu."

### Slide 04 — 03 / 2D Generation Mechanisms: Stable Diffusion & 3 Dạng Dữ Liệu Khuôn Mặt (Nhóm thực hiện: Nhóm 4)
- **Điểm nhấn thị giác**: Ba cột song song minh họa 3 dạng dữ liệu mà Stable Diffusion hỗ trợ: `Text-to-Image (T2I)` (sinh mặt mới từ chữ), `Facial Inpainting` (sửa cục bộ mắt/miệng) và `Image-to-Image (Img2Img)` (biến đổi diện mạo giữ khung 5 điểm mốc).
- **Lời thoại gợi ý**:
  > "Ở `Slide 04`, Stable Diffusion hoạt động bằng cách khử nhiễu từ không gian tiềm ẩn (Latent Space). **Nhóm thực hiện: Nhóm 4** khai thác mô hình này để tạo ra 3 dạng dữ liệu khuôn mặt: Dạng 1 là `Text-to-Image` sinh danh tính mới hoàn toàn từ mô tả văn bản để mở rộng tập mẫu; Dạng 2 là `Facial Inpainting` chỉ sửa cục bộ mắt hoặc miệng trên mặt thật để tạo ra mẫu giả tinh vi nhất; và Dạng 3 là `Image-to-Image` biến đổi diện mạo nhưng giữ nguyên khung 5 điểm mốc hình học để tạo các cặp ảnh đối chứng."

### Slide 05 — 04 / 2D System Architecture: Đường Ống 6 Giai Đoạn Dự Kiến
- **Điểm nhấn thị giác**: Sơ đồ luồng ngang 6 giai đoạn từ `Stage 01: Data Acquisition` đến `Stage 06: Dataset Export` bàn giao xuống bộ phát hiện hạ nguồn.
- **Lời thoại gợi ý**:
  > "`Slide 05` trình bày kiến trúc đường ống 6 giai đoạn mô-đun hóa: từ thu thập nguồn dữ liệu, tiền xử lý căn chỉnh khuôn mặt, tạo sinh đa chế độ bằng Stable Diffusion, lọc chất lượng tự động, gán nhãn lưu vết đến xuất bản phân tập phục vụ đánh giá và tinh chỉnh mô hình phát hiện."

### Slide 06 — 05 / 2D Automated Curation Gate: Tiền Xử Lý Hình Học & Phễu Sàng Lọc 3 Tầng
- **Điểm nhấn thị giác**: Cột trái mô tả quy trình phát hiện 5 điểm mốc và tách mặt nạ; cột phải là phễu lọc 3 tầng (độ sắc nét, cấu trúc giải phẫu mặt, độ khớp ngữ nghĩa).
- **Lời thoại gợi ý**:
  > "Để ngăn ngừa ảnh lỗi làm nhiễu mô hình hạ nguồn, `Slide 06` thiết lập cổng kiểm duyệt tự động 3 tầng. Mọi ảnh sinh ra phải vượt qua bài kiểm tra độ sắc nét, kiểm tra hình học 5 điểm mốc khuôn mặt và độ tương đồng ngữ nghĩa trước khi được đưa vào Pha 2."

### Slide 07 — 06 / Phase 2: Baseline dinoViT Benchmark & Phân Nhánh Reject / Bypass
- **Điểm nhấn thị giác**: Kiến trúc `dinoViT` trích xuất token `[CLS]` từ lưới patch khuôn mặt và cổng quyết định phân nhánh `REJECT` (bị bắt giữ) vs. `BYPASS` (vượt rào -> `Hard Sample`).
- **Lời thoại gợi ý**:
  > "Bước sang `Slide 07` — trái tim của Pha 2 theo yêu cầu của thầy. Tập ảnh giả đạt chuẩn từ Pha 1 được đưa vào mô hình chuẩn **Baseline `dinoViT`**. Nếu `dinoViT` phát hiện được là ảnh giả (`Reject`), mẫu đó được xếp vào nhóm dữ liệu tổng hợp thông thường. Ngược lại, nếu mẫu đánh lừa được `dinoViT` (`Bypass`), hệ thống lập tức thu hoạch và đóng dấu đó là một **Hard Sample**."

### Slide 08 — 07 / Hard-Sample Mining Engine: Khai Phá Điểm Mù & Tái Định Hình Biên Quyết Định
- **Điểm nhấn thị giác**: Đồ thị mật độ xác suất dự đoán với vùng `Bypass Zone` chứa các mẫu khó và sơ đồ dịch chuyển biên quyết định của `dinoViT` sau khi học từ `Hard Sample`.
- **Lời thoại gợi ý**:
  > "`Slide 08` giải thích vì sao các mẫu `Bypass` lại có giá trị cao nhất. Những mẫu `Inpainting` nội vùng hoặc `Img2Img` tinh vi giữ nguyên cấu trúc toàn cục nên dễ rơi vào vùng điểm mù của cơ chế `Self-Attention` gốc. Khi khai phá tự động nhóm `Hard Sample` này và đưa vào huấn luyện với trọng số phạt cao, chúng ta buộc các khối cuối của `dinoViT` phải mở rộng biên quyết định."

### Slide 09 — 08 / Phase 3: Closed-Loop Fine-Tuning & Ba Trục Tối Ưu Hóa
- **Điểm nhấn thị giác**: Luồng thực thi 6 bước của Pha 3 kết hợp `Data Baseline + Synthetic Data + Hard Sample` cùng 3 trục tối ưu (`Unfreeze`, `Sampling`, `Weighted Loss`).
- **Lời thoại gợi ý**:
  > "Tại `Slide 09`, Pha 3 thực hiện tinh chỉnh khép kín trên tập dữ liệu gộp từ 3 nguồn: `Data Baseline`, `Synthetic Data` và `Hard Sample`. Nhóm áp dụng đồng thời 3 kỹ thuật: mở khóa có chọn lọc các khối Transformer cuối của `dinoViT`, nhân bản tăng tần suất nhóm `Hard Sample` trong tập huấn luyện, và đánh trọng số phạt cao trong hàm mất mát để mô hình bắt giữ triệt để các mẫu từng vượt rào."

### Slide 10 — 09 / Dataset Schema & Zero Leakage Vault (Nhóm thực hiện: Nhóm 4)
- **Điểm nhấn thị giác**: Ba nguồn dữ liệu (`baseline`, `synthetic`, `hard_sample`), thẻ căn cước 6 trường truy vết và Két khóa danh tính `ZERO SOURCE ID LEAK`.
- **Lời thoại gợi ý**:
  > "Để phép chứng minh hoàn toàn minh bạch, `Slide 10` của **Nhóm thực hiện: Nhóm 4** thiết kế sổ cái 3 nguồn kèm thẻ căn cước 6 trường truy vết. Đặc biệt, mọi ảnh thật, ảnh `Reject` và ảnh `Hard Sample` có chung danh tính gốc `source_real_id` đều bị khóa cứng trong cùng một phân tập, triệt tiêu hoàn toàn hiện tượng rò rỉ danh tính giữa tập huấn luyện và tập kiểm thử."

### Slide 11 — 10 / Proof of Utility (D0 / D1 / D2) & Roadmap M0->M7
- **Điểm nhấn thị giác**: Sơ đồ đối chứng 3 chế độ `D0` (`Data Baseline`), `D1` (`Baseline + Syn Data`), `D2` (`Baseline + Syn Data + Hard Sample`) hội tụ vào `dinoViT Detector`, đồ thị kỳ vọng `ROC-AUC` và lộ trình `M0 → M7`.
- **Lời thoại gợi ý**:
  > "Cuối cùng, tại `Slide 11`, nhóm chứng minh tác dụng của dữ liệu thông qua 3 chế độ đối chứng công bằng trên cùng kiến trúc `dinoViT`: `D0` chỉ dùng `Data Baseline`, `D1` bổ sung `Synthetic Data` thường, và `D2` tích hợp trọn vẹn `Data Baseline + Synthetic Data + Hard Sample`. Khi mô hình tinh chỉnh ở chế độ `D2` bắt giữ lại được các mẫu từng `Bypass` và nâng cao đường cong `ROC-AUC` trên tập kiểm thử độc lập, tác dụng của dữ liệu tổng hợp được chứng minh trọn vẹn. Xin cảm ơn thầy và các bạn!"
