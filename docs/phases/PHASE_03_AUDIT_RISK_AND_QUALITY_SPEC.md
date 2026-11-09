# PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md — Phụ Lục: Đóng Gói Benchmark & Thực Nghiệm Proof of Utility

- **Motivation/Background**: Quy định phương thức đóng gói bộ benchmark đối chứng ($D_0, D_1, D_2$) và giao thức thực nghiệm **Proof of Utility** nhằm kiểm chứng giá trị thực tế của bộ dữ liệu đối với các mô hình phát hiện Deepfake.
- **Purpose**: Đặc tả cấu trúc các bộ dữ liệu so sánh, cơ chế oversampling Mẫu Khó, phân chia `identity-disjoint` và thước đo cải thiện năng lực phân loại.
- **Overview Pipeline**: `data/benchmarks/` ($D_0, D_1, D_2$) -> Huấn luyện detector đối chứng -> Đánh giá độc lập trên tập test cố định (`test.jsonl`) -> Đo lường mức cải thiện `ROC-AUC` và `Hard-Sample Recall`.
- **Detailed Plan**: §1 Cấu trúc 3 bộ benchmark đối chứng; §2 Cơ chế tập trung vào Mẫu Khó; §3 Giao thức kiểm thử Proof of Utility không rò rỉ danh tính.
- **References**: [../PURPOSE.md](../PURPOSE.md), [../OVERVIEW.md](../OVERVIEW.md), `configs/dataset_factory.yaml`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-09T15:15:00+07:00

---

## 1. Cấu Trúc 3 Bộ Benchmark Đối Chứng ($D_0, D_1, D_2$)

Để chứng minh luận điểm nghiên cứu: *"Tập trung vào Mẫu Khó giúp mô hình phát hiện Deepfake hiệu quả hơn hẳn việc tăng dữ liệu ồ ạt"*, hệ thống đóng gói 3 tập benchmark có chung tập test cố định:

1. **`D0_baseline`**:
   - Dữ liệu huấn luyện: Chỉ gồm ảnh cơ sở (`baseline`).
   - Mục đích: Xác lập đường cơ sở (Baseline accuracy và ROC-AUC) của mô hình.
2. **`D1_augmented`**:
   - Dữ liệu huấn luyện: Ảnh cơ sở + Toàn bộ ảnh mặt giả Stable Diffusion đã qua lọc curation (`baseline` + `synthetic`).
   - Mục đích: Trả lời câu hỏi nghiên cứu: *"Nếu chỉ bổ sung thêm ảnh tổng hợp thông thường thì mô hình cải thiện được bao nhiêu?"*.
3. **`D2_hard_boosted`**:
   - Dữ liệu huấn luyện: Ảnh cơ sở + Ảnh giả + Ưu tiên trọng số Mẫu Khó (`oversample_hard` theo cấu hình YAML).
   - Mục đích: Chứng minh đóng góp chính của đề tài: Huấn luyện có định hướng Mẫu Khó giúp mô hình vá triệt để các lỗ hổng nhận diện.

---

## 2. Kỹ Thuật Tăng Cường Mẫu Khó Trong Benchmark

Khi xuất tập huấn luyện cho giai đoạn $D_2$:
1. **Hard-Sample Oversampling**: Các bản ghi mang cờ `is_hard_sample = True` được lặp lại theo tỷ số `benchmarks.stages[D2].oversample_hard` (mặc định cấu hình trong `configs/dataset_factory.yaml`), đảm bảo mô hình có đủ tần suất học tập các ca biên giới.
2. **Bảo tồn nguồn gốc (`data_source`)**: Mọi bản ghi giữ nguyên trường `data_source` tương ứng (`baseline`, `synthetic`, `hard_sample`), tuyệt đối không bị ghi đè nhãn nguồn gốc.

---

## 3. Giao Thức Proof of Utility (Đánh Giá Không Rò Rỉ)

1. **Tập Test cố định tuyệt đối**: Cả 3 giai đoạn $D_0, D_1, D_2$ đều sử dụng chung một tệp `test.jsonl` duy nhất, không bị thay đổi giữa các stage.
2. **Bảo vệ danh tính (`identity-disjoint`)**: Nhóm danh tính thật và các biến thể sinh ra từ nó bắt buộc thuộc cùng một split, loại bỏ rủi ro mô hình ghi nhớ khuôn mặt thay vì học đặc trưng giả mạo.
3. **Thước đo tác dụng**: Đo đạc sự thay đổi của:
   - Mức tăng diện tích dưới đường cong ROC (`ROC-AUC Gain`).
   - Mức tăng tỷ lệ nhận diện Mẫu Khó (`Hard-Sample Recall Gain`).
   - Mức giảm tỷ lệ bỏ lọt (`Bypass Rate Reduction`).
