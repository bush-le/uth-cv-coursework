# Kịch Bản Thuyết Trình 5 Phút: Define & Design System (Đề Tài 11)

- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection*
- **File trình chiếu trực quan**: [`docs/reports/presentation_5min_define_and_design.html`](./presentation_5min_define_and_design.html)
- **Phong cách thiết kế**: `Blue Professional` (`#fdfae7` Warm Cream Canvas + `#1e2bfa` Single Electric Cobalt Accent) trên khung chuẩn `1920×1080` (`viewport-base.css`).
- **Nguyên tắc cốt lõi**: Trực quan hóa 100% ý định bài toán (**Define**) và kiến trúc đường ống đề xuất (**Design System**); tuyệt đối không đưa các con số thực nghiệm chưa đo đạc hay chi tiết mã nguồn vụn vặt lên slide.

---

## Slide 01 — Cover & Lộ Trình 4 Trụ Cột (00:00 – 00:30)
- **Trọng tâm hiển thị**:
  - Cột trái: Tên đề tài, định hướng kết hợp giữa khởi tạo dữ liệu khuôn mặt tổng hợp bằng Stable Diffusion và bài toán phát hiện Deepfake.
  - Cột phải: Sơ đồ lộ trình 4 trụ cột (`01 Define Problem` $\to$ `02 Define Scope` $\to$ `03 Design Dataset & System` $\to$ `04 Design Evaluation`).
- **Lời thoại gợi ý**:
  > "Xin chào Thầy/Cô và các bạn. Nhóm chúng em trình bày Đề tài 11: Xây dựng bộ dữ liệu ảnh khuôn mặt tổng hợp bằng Stable Diffusion phục vụ phát hiện Deepfake. Bài trình bày 5 phút hôm nay tập trung vào hai nội dung cốt lõi: Định nghĩa bài toán và Thiết kế kiến trúc hệ thống đường ống dự kiến."

---

## Slide 02 — `01 / PROBLEM DEFINITION`: Đặt Vấn Đề & Động Lực (00:30 – 01:20)
- **Trọng tâm hiển thị**:
  - Sơ đồ đối chiếu trực quan giữa **Hiện trạng khai thác tập dữ liệu tĩnh thụ động** và **Hướng đi tổng hợp dữ liệu khuôn mặt chủ động bằng Stable Diffusion**.
  - 3 thẻ ý định cốt lõi: *Sự dịch chuyển miền tạo sinh (Domain Shift)*, *Nhu cầu kiểm soát biến thể (Controllability)*, và *Ý định thiết kế đường ống khép kín (End-to-End Pipeline)*.
- **Lời thoại gợi ý**:
  > "Về vấn đề nghiên cứu: các tập dữ liệu Deepfake truyền thống chủ yếu cố định về bối cảnh và dựa trên kỹ thuật ghép mặt cũ. Khi đối mặt với ảnh sinh từ các mô hình khuếch tán như Stable Diffusion, bộ phát hiện dễ gặp hiện tượng lệch miền phân phối. Do đó, hướng đi của nhóm là chuyển từ khai thác dữ liệu tĩnh sang chủ động tạo sinh các biến thể khuôn mặt có kiểm soát, đi kèm cổng lọc chất lượng tự động để mở rộng biên quyết định cho mô hình phát hiện."

---

## Slide 03 — `02 / OBJECTIVES & SCOPE`: Ba Chế Độ Tạo Sinh & Phạm Vi (01:20 – 02:10)
- **Trọng tâm hiển thị**:
  - Hàng trên: 3 thẻ sơ đồ luồng trực quan cho 3 chế độ Stable Diffusion:
    1. **Text-to-Image**: Tổng hợp khuôn mặt hoàn toàn mới từ văn bản mô tả.
    2. **Facial Inpainting**: Chỉnh sửa cục bộ vùng mắt, mũi, miệng trên khuôn mặt thật thông qua mặt nạ.
    3. **Image-to-Image**: Biến đổi thuộc tính và phong cách dựa trên ảnh khuôn mặt gốc.
  - Hàng dưới: Ma trận đối chiếu rõ ràng giữa **Trong phạm vi (In-Scope)** và **Giới hạn loại trừ (Out-of-Scope)**.
- **Lời thoại gợi ý**:
  > "Để bao phủ đa dạng các kịch bản giả mạo trong miền ảnh tĩnh 2D, hệ thống kết hợp 3 chế độ tạo sinh của Stable Diffusion: Text-to-Image để tạo mẫu tổng hợp toàn phần, Facial Inpainting để mô phỏng giả mạo tinh vi từng vùng trên khuôn mặt, và Image-to-Image để biến đổi thuộc tính từ ảnh gốc. Nhóm cũng xác định rõ phạm vi chỉ tập trung vào ảnh tĩnh khuôn mặt 2D và sử dụng mô hình khuếch tán tiền huấn luyện, không xử lý video thời gian thực hay huấn luyện lại Stable Diffusion từ đầu."

---

## Slide 04 — `03 / DATASET DESIGN`: Cấu Trúc Bộ Dữ Liệu & Phân Tập Độc Lập (02:10 – 03:00)
- **Trọng tâm hiển thị**:
  - Cột trái: 3 nhóm nguồn gốc dữ liệu (`01. Real Faces`, `02. Deepfake Faces`, `03. SD Synthetic Faces`) cùng nguyên tắc quản trị nhãn kép.
  - Cột phải trên: Sơ đồ **Phân tập độc lập theo danh tính (Identity-Disjoint Split)** với rào chắn ngăn rò rỉ khuôn mặt gốc giữa tập Train/Val và tập Held-Out Test.
  - Cột phải dưới: Lưới 6 thuộc tính Metadata chuẩn hóa cho từng mẫu ảnh.
- **Lời thoại gợi ý**:
  > "Về thiết kế bộ dữ liệu, chúng em tổ chức thành 3 nhóm nguồn gốc rõ ràng: ảnh mặt thật, ảnh giả mạo truyền thống, và ảnh tổng hợp từ Stable Diffusion. Điểm quan trọng nhất trong thiết kế dữ liệu là cơ chế phân tập độc lập theo danh tính gốc: mọi ảnh biến thể sinh ra từ cùng một khuôn mặt gốc đều được khóa cố định trong cùng một phân tập, đảm bảo không có bất kỳ sự trùng lặp danh tính nào giữa tập huấn luyện và tập kiểm thử."

---

## Slide 05 — `04 / SYSTEM ARCHITECTURE`: Luồng Pipeline 6 Giai Đoạn Dự Kiến (03:00 – 04:10)
- **Trọng tâm hiển thị**:
  - Sơ đồ kiến trúc trung tâm gồm **6 giai đoạn tuần tự**:
    1. `STAGE 01 — Data Acquisition`: Thu thập nguồn ảnh mặt thật, mẫu giả mạo đối chứng và kịch bản mô tả.
    2. `STAGE 02 — Face Preprocessing`: Phát hiện vùng mặt, căn chỉnh hình học và tạo mặt nạ vùng chỉnh sửa.
    3. `STAGE 03 — Stable Diffusion Generation`: Tạo sinh khuôn mặt đa chế độ (Text-to-Image, Facial Inpainting, Image-to-Image).
    4. `STAGE 04 — Quality Control`: Cổng kiểm định tự động sàng lọc độ sắc nét, cấu trúc khuôn mặt và độ khớp mô tả (loại bỏ mẫu lỗi/biến dạng).
    5. `STAGE 05 — Labeling & Metadata`: Gán nhãn phân lớp, ghi nhận phương thức sinh và khóa danh tính gốc.
    6. `STAGE 06 — Dataset Export`: Đóng gói bộ dữ liệu chuẩn hóa theo các tập Train / Validation / Test.
  - Đường dẫn nối trực tiếp xuống **Tác vụ hạ nguồn (Downstream Task: Deepfake Detection & Evaluation)**.
- **Lời thoại gợi ý**:
  > "Đây là kiến trúc tổng thể của hệ thống gồm 6 giai đoạn tuần tự. Bắt đầu từ thu thập dữ liệu đầu vào ở Giai đoạn 1 và tiền xử lý chuẩn hóa khuôn mặt ở Giai đoạn 2. Giai đoạn 3 là lõi tạo sinh đa chế độ bằng Stable Diffusion. Vì mô hình tạo sinh có thể tạo ra các mẫu lỗi giải phẫu hoặc mờ nhòe, Giai đoạn 4 đóng vai trò chốt chặn kiểm định chất lượng tự động để loại bỏ mẫu hỏng. Các mẫu đạt chuẩn sẽ được gán nhãn và lưu vết metadata ở Giai đoạn 5, sau đó xuất thành bộ dữ liệu phân tập ở Giai đoạn 6 để cấp trực tiếp cho tác vụ huấn luyện và đánh giá bộ phát hiện Deepfake hạ nguồn."

---

## Slide 06 — `05 / EVALUATION & EXPECTED OUTCOMES`: Thiết Kế Đánh Giá & Đầu Ra (04:10 – 05:00)
- **Trọng tâm hiển thị**:
  - Hàng trên: Sơ đồ thực nghiệm đối chứng 2 nhánh (**Nhánh Baseline** chỉ gồm `Real + Deepfake` so sánh với **Nhánh Proposed** gồm `Real + Deepfake + SD Synthetic`) đi qua **Cùng kiến trúc phát hiện** và đánh giá trên **Cùng tập Held-Out Test Set**.
  - Hàng dưới: 4 thẻ sản phẩm đầu ra dự kiến (*Bộ dữ liệu khuôn mặt chuẩn hóa*, *Đường ống tạo sinh & kiểm định*, *Thực nghiệm phát hiện đối chứng*, *Báo cáo học thuật*).
- **Lời thoại gợi ý**:
  > "Cuối cùng, để đánh giá khách quan đóng góp của dữ liệu tổng hợp, nhóm thiết kế thực nghiệm đối chứng hai nhánh: một nhánh huấn luyện chỉ với dữ liệu truyền thống (Baseline) và một nhánh bổ sung dữ liệu tổng hợp từ Stable Diffusion (Proposed). Cả hai nhánh sử dụng cùng một kiến trúc mô hình phát hiện và được kiểm thử trên cùng tập Held-Out Test độc lập danh tính. Kết quả kỳ vọng là đóng gói hoàn chỉnh bộ dữ liệu, đường ống tự động hóa và báo cáo thực nghiệm đối chứng. Em xin cảm ơn Thầy/Cô và các bạn đã lắng nghe."
