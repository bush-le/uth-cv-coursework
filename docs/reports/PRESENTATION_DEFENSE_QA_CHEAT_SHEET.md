# BỘ CÂU HỎI & TRẢ LỜI NHANH BẢO VỆ ĐỒ ÁN (PRESENTATION DEFENSE Q&A CHEAT SHEET)

- **Motivation/Background**: Khi bảo vệ đồ án chuyên sâu về Thị giác Máy tính & Phát hiện Deepfake trước hội đồng giảng viên UTH, nhóm cần một tài liệu tra cứu nhanh các câu hỏi phản biện kỹ thuật then chốt, cơ sở toán học và số liệu thực nghiệm.
- **Purpose**: Cung cấp bộ câu hỏi và câu trả lời súc tích, chính xác tuyệt đối về mặt kỹ thuật cho Đề tài 11.
- **Overview Pipeline**: Tiếp nhận câu hỏi hội đồng -> Định vị phương diện kỹ thuật -> Trả lời theo cấu trúc luận điểm khoa học kèm số liệu thực chứng.
- **Detailed Plan**: §1 Kiến trúc mô hình & DINOv2; §2 Kỹ thuật sinh dữ liệu & Soft Feathering; §3 Active Learning & Hard Sample Mining; §4 Benchmark Phổ Quát & Năng Lực Khái Quát Hóa; §5 Số liệu vàng & Invariants.
- **References**: `../../GEMINI.md`, `SYSTEM_DEFINE_AND_DESIGN_REPORT.md`, `ACTIVE_LEARNING_M2_REFINED_REPORT.md`.
- **Created**: 2026-11-11T23:25:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00

---

## 1. Nhóm Câu Hỏi Về Kiến Trúc Bộ Dò & DINOv2

### Q1: Tại sao nhóm chọn DINOv2-ViT-S/14 thay vì ResNet-50 hoặc EfficientNet truyền thống?
**Trả lời:**
- Các mạng CNN truyền thống tiền huấn luyện có giám sát (Supervised ImageNet) có xu hướng dựa vào **ngữ nghĩa vật thể cấp cao (Semantic Bias)** như hình dáng mắt, mũi, miệng.
- **DINOv2** được huấn luyện tự giám sát (Self-Supervised Learning) trên 142 triệu ảnh bằng cơ chế DINO + iBOT. Nhờ đó, không gian đặc trưng của DINOv2 bảo tồn nguyên vẹn các thông tin **kết cấu vi thể (micro-texture)** ở cấp độ patch $14 \times 14$, cực kỳ nhạy cảm với các dị thường tần số cao (high-frequency artifacts) đặc trưng của quá trình khử nhiễu khuếch tán mà mắt thường không thấy được.
- DINOv2-ViT-S/14 chỉ có **22.06 triệu tham số**, cho tốc độ suy luận ~18.5 ms/ảnh trên GPU, tối ưu cho triển khai thời gian thực.

### Q2: Cơ chế trích xuất đặc trưng của bộ dò hoạt động như thế nào?
**Trả lời:**
- Đầu vào ảnh được đưa về kích thước chuẩn hóa $(3, 224, 224)$.
- DINOv2 chia ảnh thành $16 \times 16 = 256$ patches kích thước $14 \times 14$.
- Chúng tôi kết hợp cả token đại diện toàn cục `[CLS]` (384 chiều) và trung bình cộng các patch tokens (Average Pooling, 384 chiều) thành vector 768 chiều.
- Đầu phân loại (Classification Head) gồm: `LayerNorm(768) -> Linear(768, 256) -> GeLU -> Dropout(0.3) -> Linear(256, 1)` dự đoán xác suất logit nhị phân.

---

## 2. Nhóm Câu Hỏi Về Dữ Liệu Sinh & Động Cơ Khuếch Tán Kép

### Q3: Hiện tượng "học tủ biên ghép" (Boundary Artifact Overfitting) là gì và nhóm giải quyết ra sao?
**Trả lời:**
- Khi dùng Inpainting thông thường với mặt nạ nhị phân cứng (Hard Binary Mask), ranh giới giữa vùng ảnh gốc và vùng sinh mới xuất hiện một đường cắt sắc nhọn về giá trị gradient pixel. Bộ dò nông cạn sẽ dễ dàng phát hiện deepfake chỉ bằng cách tìm đường biên này thay vì học bản chất khuôn mặt giả.
- Nhóm giải quyết bằng kỹ thuật **Làm Mịn Biên Gaussian Mềm (Soft-Mask Feathering $\sigma=4.0$)**:
  $$M_{\text{feathered}} = \text{GaussianBlur}(M_{\text{binary}}, k=(15, 15), \sigma=4.0)$$
  $$I_{\text{composite}} = I_{\text{inpaint}} \odot M_{\text{feathered}} + I_{\text{real}} \odot (1.0 - M_{\text{feathered}})$$
  Điều này tạo ra một dải chuyển tiếp mượt mà từ 0.0 đến 1.0, xóa bỏ hoàn toàn vết cắt nhân tạo, buộc DINOv2 phải học sâu vào các dị tật sinh học vi thể bên trong khuôn mặt.

### Q4: Động cơ khuếch tán kép (Dual-Engine) gồm những gì và tại sao cần kết hợp?
**Trả lời:**
- **Engine A: Stable Diffusion v1.5 Inpainting** — Đại diện cho mô hình sinh phổ thông mã nguồn mở, cho khuôn mặt cân đối nhưng đôi khi thiếu chi tiết da siêu thực.
- **Engine B: RealisticVision v5.1** — Đại diện cho mô hình khuếch tán tinh chỉnh cao cấp, mô phỏng siêu thực từng lỗ chân lông và phản chiếu giác mạc.
- Việc kết hợp hai động cơ giúp bộ dò không bị "học tủ" phong cách sinh của riêng một mô hình (generator-specific overfitting).

---

## 3. Nhóm Câu Hỏi Về Active Learning & Hard Sample Mining

### Q5: Tiêu chuẩn toán học nào được dùng để xác định một "Mẫu Cực Khó" (Hard Sample)?
**Trả lời:**
Một mẫu dữ liệu được phân loại là Mẫu Khó theo 2 tiêu chí khắt khe:
1. **Mẫu ranh giới mơ hồ (Margin Ambiguity):** Xác suất dự đoán nằm sát ngưỡng quyết định 0.50:
   $$|P(\text{Fake} \mid x) - 0.50| \le 0.15 \iff P(\text{Fake}) \in [0.35, 0.65]$$
2. **Mẫu vi phạm nghiêm trọng (High-Confidence Inversion):**
   - False Negative: Ảnh deepfake thật sự nhưng bộ dò dự đoán $P(\text{Fake}) < 0.20$.
   - False Positive: Ảnh người thật nhưng bộ dò dự đoán $P(\text{Fake}) > 0.80$.
- Nhóm đã cô lập được **213 mẫu cực khó** vào tập kiểm định thách thức (Challenge Vault).

### Q6: Hiệu quả thực chứng của vòng lặp Active Learning được chứng minh như thế nào?
**Trả lời:**
Trên tập 213 mẫu cực khó (Challenge Vault):
- **$M_0$ (Baseline D0):** Chỉ đạt độ chính xác **52.1%** (gần như đoán mò ngẫu nhiên).
- **$M_1$ (Active Learning Round 1):** Tăng vọt lên **81.7%** (+29.6%).
- **$M_2^{\text{cold}}$ (Cold-Start Retrain):** Đạt **88.3%**.
- **$M_2^*$ (Fine-tuned Champion):** Đạt **94.8%** (+42.7% so với $M_0$), đồng thời diện tích dưới đường cong AUC tăng từ **0.612 lên 0.988**.

---

## 4. Nhóm Câu Hỏi Về Benchmark Phổ Quát & Khả Năng Tổng Quát Hóa

### Q7: Bộ dò hoạt động như thế nào khi gặp các thuật toán sinh chưa từng học (Out-of-Distribution)?
**Trả lời:**
Nhóm đã thực hiện 2 bài kiểm tra mù quy mô lớn:
1. **Blind Test 6.000 ảnh RealisticVision v5.1:**
   - $M_0$ Baseline chỉ đạt **68.4%**.
   - $M_2^*$ Champion đạt **89.1%** (+20.7% tuyệt đối mà không cần huấn luyện toàn bộ kho ảnh).
2. **Universal Benchmark 20.400 ảnh (34 thuật toán, 5 họ mô hình):**
   - Đánh giá trên GAN, Diffusion, Face Swap, Commercial Inpainting, và Talking Heads.
   - $M_0$: Accuracy 64.2%, AUC 0.712.
   - $M_2^*$: Accuracy **82.4%**, AUC **0.897**.

### Q8: Trong ứng dụng an ninh ngân hàng (eKYC), tỷ lệ báo động giả (FPR) được kiểm soát ra sao?
**Trả lời:**
- Khi vận hành thực tế, việc chặn nhầm khách hàng thật (False Positive) gây thiệt hại trải nghiệm nghiêm trọng.
- Nhóm đã thực hiện quét ngưỡng $\tau \in [0.10, 0.90]$.
- Tại ngưỡng nghiêm ngặt **$\text{FPR} \le 1.0\%$** (chỉ cho phép tối đa 1% người thật bị nghi ngờ), mô hình $M_2^*$ vẫn duy trì được **Fake Recall đạt 74.2%** trên Universal Benchmark, vượt xa mức 31.8% của mô hình cơ sở $M_0$.

---

## 5. Bảng Số Liệu Vàng Tra Cứu Nhanh (Golden Key Facts)

| Chỉ Số / Hạng Mục | $M_0$ (Baseline) | $M_1$ (Active R1) | $M_2^{\text{cold}}$ (Cold) | $M_2^*$ (Champion) | Ghi Chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Val Accuracy** | 88.47% | 90.35% | 89.97% | **91.20%** | Tập kiểm định D0/D1/D2 |
| **Val AUC** | 0.9327 | 0.9445 | 0.9430 | **0.9580** | Khả năng phân tách nhị phân |
| **Master Frozen Test Acc** | 82.10% | 84.50% | 80.81% | **85.30%** | 1.178 ảnh kiểm thử đông kết |
| **Challenge Vault Acc** | 52.11% | 81.69% | 88.26% | **94.84%** | 213 mẫu cực khó (+42.7%) |
| **Challenge Vault AUC** | 0.6124 | 0.8842 | 0.9430 | **0.9875** | Tăng vọt độ tự tin phân loại |
| **RealisticVision 6k Acc** | 68.40% | 82.60% | 84.10% | **89.10%** | 6.000 ảnh kiểm thử mù OOD |
| **Universal 20.4k Acc** | 64.20% | 73.50% | 76.80% | **82.40%** | 34 thuật toán, 5 họ mô hình |
| **Universal 20.4k AUC** | 0.7120 | 0.8050 | 0.8390 | **0.8970** | Toàn bộ 20.400 ảnh benchmark |
