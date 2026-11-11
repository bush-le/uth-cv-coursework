# PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md — Giai Đoạn 2: Khai Thác Mẫu Khó Đối Kháng (Hard-Sample Mining)

- **Motivation/Background**: Quy định cơ chế khai thác Mẫu Khó (`Hard Samples`) — luận điểm khoa học cốt lõi của đề tài. Mẫu khó là những khuôn mặt giả mạo tinh vi có khả năng vượt qua hàng rào phát hiện và nằm sát phân bố đa tạp ảnh thật (`Real-Manifold Proximity`).
- **Purpose**: Đặc tả thuật toán đo độ khó bằng không gian biểu diễn sâu, hiệu chỉnh ngưỡng động Tau ($\tau$) và quản lý mẫu khó phục vụ đóng gói benchmark.
- **Overview Pipeline**: `data/synthetic/curated_manifest.jsonl` + `data/raw/manifest.jsonl` -> `src/models/evaluators/hardness.py` -> `src/pipelines/mine_hard_samples.py` -> `data/hard_samples/`.
- **Detailed Plan**: §1 Khái niệm & bản chất Mẫu Khó; §2 Thuật toán Real-Manifold Proximity; §3 Hiệu chỉnh ngưỡng Tau tự động qua Leave-One-Out; §4 Phương pháp Baseline Detector Bypass; §5 Xuất bản & Lưu trữ Mẫu Khó.
- **References**: [../PURPOSE.md](../PURPOSE.md), [../OVERVIEW.md](../OVERVIEW.md), `configs/dataset_factory.yaml`, `.agents/rules/ZERO_HARDCODE_RULE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-09T15:15:00+07:00

---

## 1. Khái Niệm & Bản Chất Của Mẫu Khó (`Hard Samples`)

Trong bài toán phát hiện Deepfake, không phải mọi ảnh giả đều có giá trị đóng góp như nhau:
- **Ảnh giả dễ nhận biết (`Easy/Typical Fakes`)**: Mang các đặc trưng sai lệch rõ rệt (mất cân đối cấu trúc, biên mờ), dễ dàng bị các bộ trích xuất đặc trưng phát hiện.
- **Mẫu khó đối kháng (`Hard Samples`)**: Là những ảnh giả mà đặc trưng biểu diễn sâu của chúng nằm chen lẫn hoặc tiệm cận sát đa tạp biểu diễn của ảnh khuôn mặt thật, tạo ra các điểm mù nguy hiểm trong không gian phân định (decision boundary).

---

## 2. Thuật Toán Khai Thác Khách Quan: Real-Manifold Proximity

Thay vì so sánh ảnh giả với chính ảnh gốc sinh ra nó (vốn chỉ đo mức độ can thiệp và làm lệch kết quả về phía inpainting), hệ thống đo khoảng cách hình học tới **toàn bộ đa tạp ảnh thật tham chiếu**:

1. **Trích xuất biểu diễn sâu chuẩn hóa (`embed_images`)**: Chiếu ảnh thật và ảnh giả qua mô hình nền tảng thị giác (`hardness.backbone_id`), chuẩn hóa L2 trên mặt cầu đơn vị $||\mathbf{v}||_2 = 1.0$.
2. **Loại trừ ảnh nguồn gốc (`exclude_idx`)**: Khi tính toán khoảng cách của một ảnh giả $i$, chỉ số của ảnh thật nguồn (`source_real_id`) bị che bằng $-\infty$ để triệt tiêu ảnh hưởng tự tương quan.
3. **Độ tương đồng lân cận gần nhất (`real_manifold_scores`)**:
   $$s_i = \frac{1}{k} \sum_{j \in \text{Top-}k(\mathcal{M}_{\text{real}} \setminus \{ \text{source}_i \})} \cos(\mathbf{e}_i^{\text{fake}}, \mathbf{e}_j^{\text{real}})$$
   với $k = \text{hardness.k\_neighbors}$ (cấu hình trong YAML).

---

## 3. Hiệu Chỉnh Ngưỡng Tau ($\tau$) Tự Động Qua Leave-One-Out

Ngưỡng phân định Mẫu Khó $\tau$ **tuyệt đối không được gán cứng**, mà được xác định dựa trên hình thái thực nghiệm của chính tập ảnh thật:
1. **Phân bố kiểm tra chéo (`real_reference_scores`)**: Với mỗi ảnh thật trong tập tham chiếu, tính độ tương đồng trung bình tới $k$ ảnh thật gần nhất khác (loại bỏ chính nó).
2. **Xác lập phân vị chuẩn (`calibrate_threshold`)**:
   $$\tau = \text{Quantile}(\{ s_m^{\text{ref}} \}_{m=1}^M, q = \text{hardness.reference_quantile})$$
   Một ảnh giả được xác nhận là **Mẫu Khó (`is_hard_sample = True`)** khi và chỉ khi điểm số $s_i \ge \tau$, tức nó nằm gần đa tạp ảnh thật giống như một bức ảnh thật điển hình.

---

## 4. Phương Pháp Baseline Detector Bypass (Tùy chọn)

Hệ thống cũng hỗ trợ phương pháp khai thác truyền thống qua mô hình phân loại đường cơ sở (`hardness.method = "detector"`):
- Yêu cầu bắt buộc nạp checkpoint đã qua huấn luyện kiểm chứng thông qua `load_verified_checkpoint()`. Nghiêm cấm chạy suy luận với trọng số ngẫu nhiên theo `ZERO_MOCK_DATA_RULE`.
- Ảnh giả bị detector đánh giá có xác suất $P(\text{fake}) < \tau_{\text{detector}}$ (bị đoán nhầm là Thật) sẽ được gắn cờ `is_hard_sample = True`.

---

## 5. Xuất Bản & Lưu Trữ Mẫu Khó

- Danh mục Mẫu Khó được lưu tại `paths.hard_manifest` (`data/hard_samples/hard_samples_manifest.jsonl`).
- Các tệp ảnh được lưu độc lập tại `data/hard_samples/images/` phục vụ trực quan hóa và kiểm tra.
- Biên bản thống kê chi tiết theo chế độ sinh và nhóm nhân khẩu học được xuất tự động tại `data/hard_samples/audit_reports/hardness_report.json`.
