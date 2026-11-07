# Kịch Bản Thuyết Trình 5 Phút: Define & Design System (8 Slides — Đề Tài 11)

- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection*
- **Nhóm thực hiện**: **Nhóm 4**
- **Bản trình chiếu tương tác**: `docs/reports/presentation_5min_define_and_design.html` (8 Slides siêu trực quan, chuẩn khung hình `1920×1080`, giao diện `Blue Professional`)
- **Nguyên tắc học thuật**: Trình bày thuần túy **Định nghĩa bài toán (Problem Definition)** và **Thiết kế hệ thống dự kiến (System Design)** kết hợp minh họa trực quan (`Minh họa ý tưởng`), tuyệt đối không đưa số liệu thực nghiệm chưa chạy.

---

## Phân Bổ Thời Lượng 5 Phút Trên 8 Slide

| Slide | Tiêu đề Slide | Nội dung Trực quan Trọng tâm | Thời lượng gợi ý |
| :--- | :--- | :--- | :--- |
| **Slide 01** | **Cover & Core System Vision** | Tên đề tài, ghi danh **Nhóm thực hiện: Nhóm 4**, dải minh họa *Chân dung $\to$ Stable Diffusion $\to$ Phễu Lọc $\to$ Detector*, và lộ trình 4 trụ cột. | ~25 giây |
| **Slide 02** | **01 / Problem Definition** | Đối chiếu trực quan *Deepfake cắt ghép cục bộ* vs. *Tạo sinh khử nhiễu toàn cục* và đồ thị không gian đặc trưng 2D với sự dịch chuyển **Decision Boundary**. | ~40 giây |
| **Slide 03** | **02 / Generation Mechanisms** | Gallery trực quan 3 cột cho 3 chế độ tạo sinh khuôn mặt: **Text-to-Image**, **Facial Inpainting** (mặt nạ vùng mắt/miệng), và **Image-to-Image** (giữ khung 5 điểm mốc). | ~45 giây |
| **Slide 04** | **03 / Scope & Boundaries** | Huy hiệu **Nhóm thực hiện: Nhóm 4** và ma trận đối xứng 2 cột: **Phạm vi thực hiện trọng tâm (In-Scope)** vs. **Giới hạn loại trừ (Out-of-Scope)**. | ~35 giây |
| **Slide 05** | **04 / Dataset Design** | Ba nhóm nguồn gốc dữ liệu (*Real, Deepfake, SD Synthetic*), thẻ siêu dữ liệu 6 trường, và sơ đồ **Két khóa độc lập danh tính (Zero Identity Leakage)**. | ~40 giây |
| **Slide 06** | **05 / System Architecture** | Sơ đồ **Đường ống 6 giai đoạn dự kiến** có hiệu ứng dòng chảy dữ liệu động (`Stage 01` $\to$ `Stage 06`) kết nối trực tiếp xuống tác vụ **Deepfake Detection**. | ~45 giây |
| **Slide 07** | **06 / Automated Curation Gate** | Sơ đồ chuyên sâu **Stage 02** (Quét khung mặt, 5 điểm mốc, căn chỉnh hình học, tách mặt nạ) và **Stage 04** (Phễu lọc chất lượng 3 tầng tách nhánh Đạt chuẩn vs. Loại bỏ). | ~40 giây |
| **Slide 08** | **07 / Evaluation & Deliverables** | Thiết kế thực nghiệm đối chứng 2 nhánh (**Baseline** vs. **Proposed**), đồ thị minh họa kỳ vọng nâng đường cong **ROC-AUC**, và 4 sản phẩm bàn giao. | ~30 giây |

---

## Lời Thoại Gợi Ý Theo Từng Slide

### Slide 01 — Cover & Core System Vision (`Nhóm thực hiện: Nhóm 4`)
> "Xin chào Thầy/Cô và các bạn. Em đại diện **Nhóm 4** trình bày phần Định nghĩa Bài toán và Thiết kế Hệ thống cho Đề tài 11: *Xây dựng bộ dữ liệu ảnh khuôn mặt tổng hợp bằng Stable Diffusion phục vụ phát hiện Deepfake*. Ý tưởng cốt lõi của hệ thống là chủ động tạo sinh khuôn mặt đa chế độ, sàng lọc tự động qua phễu kiểm định 3 tầng để mở rộng biên phát hiện của mô hình Deepfake Detector."

### Slide 02 — 01 / Problem Definition: Dịch Chuyển Miền & Biên Quyết Định
> "Về bối cảnh bài toán: các bộ phát hiện truyền thống chủ yếu học dấu vết cắt ghép cục bộ quanh viền mặt của Deepfake cổ điển. Ngược lại, Stable Diffusion tổng hợp toàn bộ khuôn mặt từ nhiễu tiềm ẩn nên bề mặt da và ánh sáng rất liền mạch. Như minh họa trên đồ thị không gian đặc trưng 2D bên phải, các mẫu Diffusion nằm ở vùng mù của biên quyết định cũ; do đó chúng ta cần chủ động sinh dữ liệu để dịch chuyển biên quyết định bao phủ miền tạo sinh mới."

### Slide 03 — 02 / Generation Mechanisms: Ba Chế Độ Tạo Sinh Khuôn Mặt
> "Để bao phủ đa dạng kịch bản giả mạo, hệ thống khai thác 3 chế độ tạo sinh bổ trợ nhau: thứ nhất là **Text-to-Image** sinh mới hoàn toàn khuôn mặt từ văn bản; thứ hai là **Facial Inpainting** dùng mặt nạ nhị phân chỉ tái tạo cục bộ vùng mắt, mũi, miệng trên ảnh thật; và thứ ba là **Image-to-Image** biến đổi diện mạo và ánh sáng nhưng bảo toàn khung hình học 5 điểm mốc mặt gốc."

### Slide 04 — 03 / Scope & Boundaries: Phạm Vi Thực Hiện & Giới Hạn (`Nhóm thực hiện: Nhóm 4`)
> "Trên Slide 04, **Nhóm thực hiện: Nhóm 4** xác lập rõ phạm vi đề tài: tập trung vào ảnh tĩnh khuôn mặt 2D, xây dựng đường ống tự động 3 chế độ, phân tập độc lập danh tính và thực nghiệm đối chứng Detector. Đồng thời nhóm loại trừ rõ việc xử lý video/audio thời gian thực, không huấn luyện lại Stable Diffusion từ đầu, không mạo danh cá nhân có thật và không tối ưu phần cứng biên."

### Slide 05 — 04 / Dataset Design: Ba Nhóm Nguồn Gốc & Két Khóa Danh Tính (`Nhóm thực hiện: Nhóm 4`)
> "Về thiết kế bộ dữ liệu, **Nhóm 4** tổ chức 3 nhóm nguồn gốc gồm `Real Faces`, `Deepfake Faces` và `SD Synthetic Faces` với thẻ siêu dữ liệu 6 trường bắt buộc. Đặc biệt, sơ đồ Két khóa sinh trắc học bên phải minh họa nguyên tắc **Zero Identity Leakage**: mọi biến thể sinh ra từ cùng một khuôn mặt gốc đều bị khóa chặt trong cùng một phân tập, tuyệt đối không để rò rỉ danh tính sang tập kiểm thử `Held-Out Test`."

### Slide 06 — 05 / System Architecture: Đường Ống 6 Giai Đoạn Dự Kiến
> "Slide 06 thể hiện toàn cảnh đường ống 6 giai đoạn dự kiến với luồng dữ liệu một chiều: từ `Stage 01 Thu thập nguồn`, `Stage 02 Tiền xử lý khuôn mặt`, `Stage 03 Tạo sinh đa chế độ bằng Stable Diffusion`, đi qua `Stage 04 Cổng kiểm định chất lượng`, `Stage 05 Gán nhãn và lưu vết siêu dữ liệu`, đến `Stage 06 Xuất bản bộ dữ liệu` bàn giao trực tiếp cho tác vụ huấn luyện và đánh giá mô hình phát hiện Deepfake ở hạ nguồn."

### Slide 07 — 06 / Automated Curation Gate: Tiền Xử Lý & Phễu Sàng Lọc 3 Tầng
> "Slide 07 đi sâu vào hai chốt chặn kỹ thuật quan trọng nhất của đường ống. Bên trái là `Stage 02`: tự động phát hiện khung mặt và 5 điểm mốc để căn chỉnh hình học chuẩn, đồng thời trích xuất mặt nạ vùng nhạy cảm cho Inpainting. Bên phải là `Stage 04`: phễu sàng lọc tự động 3 tầng kiểm tra độ sắc nét, cấu trúc giải phẫu mặt và độ khớp ngữ nghĩa để cách ly các mẫu mờ hoặc biến dạng trước khi gán nhãn."

### Slide 08 — 07 / Evaluation & Expected Outcomes: Thực Nghiệm Đối Chứng & Đầu Ra
> "Cuối cùng, trên Slide 08 là thiết kế thực nghiệm đối chứng 2 nhánh: nhánh **Baseline** chỉ huấn luyện trên ảnh thật và Deepfake cổ điển, còn nhánh **Proposed** bổ sung thêm dữ liệu tổng hợp từ Stable Diffusion. Cả hai cùng huấn luyện trên một kiến trúc mạng phát hiện cố định và đo lường trên cùng tập kiểm thử độc lập thông qua kỳ vọng nâng đường cong **ROC-AUC**, đi kèm 4 sản phẩm bàn giao hoàn chỉnh của nhóm. Em xin cảm ơn Thầy/Cô và các bạn đã lắng nghe."
