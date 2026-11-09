# EXPERIMENT_PROTOCOL.md — Phụ Lục: Giao Thức Thực Nghiệm Proof of Utility ($D_0 \to D_1 \to D_2$)

- **Motivation/Background**: Thiết lập giao thức thực nghiệm khoa học, minh bạch và có tính lặp lại cao nhằm đo lường giá trị thực tế của bộ dữ liệu đối với downstream detectors.
- **Purpose**: Định nghĩa 3 giai đoạn đối chứng benchmark ($D_0, D_1, D_2$), 4 tiêu chuẩn đo lường và quy chuẩn kết luận tác dụng của bộ dữ liệu.
- **Overview Pipeline**: `data/benchmarks/` -> Downstream Detector -> Đánh giá trên tập test cố định -> Đo lường delta metrics (`ROC-AUC`, `Hard-Sample Recall`, `Bypass Rate`).
- **Detailed Plan**: §1 Ba chế độ thực nghiệm đối chứng ($D_0, D_1, D_2$); §2 Bốn thước đo đánh giá định lượng; §3 Tiêu chuẩn kiểm chứng tác dụng của bộ dữ liệu.
- **References**: [../PURPOSE.md](../PURPOSE.md), [../OVERVIEW.md](../OVERVIEW.md), `configs/dataset_factory.yaml`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-09T15:15:00+07:00

---

## 1. Ba Chế Độ Thực Nghiệm Đối Chứng ($D_0, D_1, D_2$)

Cả 3 chế độ đều sử dụng chung một kiến trúc mô hình, cùng điều kiện siêu tham số huấn luyện và được chấm điểm trên **cùng một tập kiểm thử độc lập cố định (`test.jsonl`)**:

| Chế độ | Dữ liệu huấn luyện (`train.jsonl`) | Ý nghĩa khoa học |
| :--- | :--- | :--- |
| **Chế độ $D_0$** | Chỉ gồm **Ảnh cơ sở (`baseline`)** | Đo lường năng lực ban đầu của mô hình trước khi tiếp cận dữ liệu tổng hợp phái sinh. |
| **Chế độ $D_1$** | **Ảnh cơ sở + Ảnh giả thông thường (`synthetic`)** | Đo lường mức cải thiện khi chỉ bổ sung ảnh giả diện rộng không qua lọc độ khó. |
| **Chế độ $D_2$** | **Ảnh cơ sở + Ảnh giả + Nhân đôi Mẫu Khó (`hard_sample`)** | Đo lường mức độ tối ưu hóa biên phân định khi tập trung tăng cường các Mẫu Khó sát đa tạp ảnh thật. |

---

## 2. Bốn Thước Đo Đánh Giá Định Lượng

1. **Tỷ lệ nhận diện Mẫu Khó (`Hard-Sample Recall` — Càng cao càng tốt)**:
   - Đo lường khả năng bắt lại các mẫu khó đối kháng từng nằm sát đa tạp ảnh thật.
   - Đây là chỉ số trực tiếp nhất chứng minh mô hình đã vá được điểm mù sau khi học trên tập $D_2$.
2. **Tỷ lệ bỏ lọt (`Bypass Rate` — Càng thấp càng tốt)**:
   - Tỷ lệ các mẫu giả mạo đánh lừa được mô hình và bị phân loại nhầm thành ảnh thật.
3. **Chỉ số phân tách tổng thể (`ROC-AUC` — Trong [0.0, 1.0])**:
   - Diện tích dưới đường cong ROC đo lường năng lực phân biệt ảnh thật và ảnh giả độc lập với việc chọn ngưỡng quyết định.
4. **Độ chính xác và độ cân bằng (`Accuracy` & `F1-Score`)**:
   - Đảm bảo việc tăng độ nhạy trên ảnh giả không làm tăng tỷ lệ báo động giả trên ảnh người thật (False Positives).

---

## 3. Tiêu Chuẩn Kết Luận Tác Dụng Của Bộ Dữ Liệu

Bộ dữ liệu do pipeline của Nhóm 4 tạo ra được kết luận là **có giá trị thực tiễn cao** khi thỏa mãn:
1. **Khắc phục điểm mù**: Điểm `Hard-Sample Recall` của $D_2$ tăng vượt bậc so với $D_0$ và $D_1$.
2. **Giảm thiểu rủi ro bảo mật**: `Bypass Rate` giảm đều đặn theo chiều $D_0 \to D_1 \to D_2$.
3. **Nâng cao năng lực tổng thể**: Điểm số `ROC-AUC` của $D_2$ đạt cao nhất trong cả 3 chế độ trên cùng tập test độc lập.
