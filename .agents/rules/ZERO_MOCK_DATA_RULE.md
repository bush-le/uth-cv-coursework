# ZERO_MOCK_DATA_RULE.md — Zero-Mock & Verified Neural Weights Operational Invariant

- **Motivation/Background**: Nghiêm cấm tuyệt đối việc sử dụng dữ liệu giả lập (mock data), toán học hình oval/noise, hoặc mô hình nơ-ron chưa nạp checkpoint (untrained random weights) để tạo ra kết quả phân loại hay mẫu khó giả mạo trong môi trường sản xuất và báo cáo nghiên cứu.
- **Purpose**: Thiết lập quy tắc hiến pháp bắt buộc kiểm tra tính hợp lệ của checkpoint và dữ liệu nạp vào pipeline.
- **Overview Pipeline**: Khởi tạo pipeline -> Kiểm tra checkpoint nơ-ron tồn tại & hợp lệ -> Xác thực nhãn và phân tầng dữ liệu -> Chặn đứng mọi hành vi tạo mẫu ngẫu nhiên -> Báo cáo khoa học chính xác.
- **Detailed Plan**: §1 Điều răn Zero-Mock; §2 Ràng buộc kiểm tra Checkpoint trước khi suy luận; §3 Ràng buộc phân tầng dữ liệu không thoái hóa; §4 Cơ chế tự kiểm tra tự động (Self-Audit Assertions).
- **References**: [../../GEMINI.md](../../GEMINI.md), [../AI_REFERENCE.md](../AI_REFERENCE.md), [../../docs/PURPOSE.md](../../docs/PURPOSE.md).
- **Created**: 2026-11-09T13:05:00+07:00
- **Last Updated**: 2026-11-09T13:05:00+07:00

---

## 1. Điều Răn Tuyệt Đối: Cấm Mock Data & Mô Hình Trọng Số Ngẫu Nhiên

1. **Không Mock Data trong Pipelines Sản Xuất**:
   - Mọi ảnh khuôn mặt phải đến từ tệp ảnh thật (`data/raw/`) hoặc ảnh do mạng khuếch tán thật sinh ra (`runwayml/stable-diffusion-v1-5`).
   - Cấm sử dụng các hàm vẽ hình oval toán học, noise giả lập để thay thế ảnh thật/ảnh giả.
2. **Cấm Suy Luận Với Đầu Phân Loại Ngẫu Nhiên (Random Weights)**:
   - Trong pipeline khai thác Mẫu Khó (`mine_hard_samples.py`) hoặc đánh giá (`evaluate.py`), **BẮT BUỘC** phải nạp checkpoint trọng số đã qua huấn luyện (`model_state_dict`).
   - Cấm tuyệt đối việc chỉ nạp backbone pretrained rồi để đầu phân loại (`self.classifier`) ở trạng thái khởi tạo ngẫu nhiên để chấm điểm hay bóc tách mẫu khó.
   - Nếu không có checkpoint hoặc checkpoint không tồn tại, script bắt buộc phải dừng lại và ném ngoại lệ `FileNotFoundError` hoặc `ValueError`, cấm âm thầm chạy với trọng số ngẫu nhiên.

---

## 2. Ràng Buộc Phân Tầng Dữ Liệu (Stratified Invariant)

Mọi phân vùng dữ liệu (`train`, `val`, `test`) trong bài toán phát hiện ảnh giả phải thỏa mãn:
1. **Không Rò Rỉ Danh Tính (`identity_disjoint=True`)**: Toàn bộ ảnh phát sinh từ cùng một người thật (`source_real_id`) phải nằm chung một phân vùng.
2. **Không Rỗng Nhãn (Non-Empty Label Contract)**: Cả tập Train, Val và Test **bắt buộc phải có cả nhãn 0 (Real) và nhãn 1 (Fake)**.
3. Tuyệt đối không cho phép tạo ra tập Test có $100\%$ nhãn 0 dẫn đến các chỉ số $F_1$, Precision, Recall bị thoái hóa về $0.0$.

---

## 3. Cơ Chế Tự Kiểm Tra (Self-Audit Assertions)

Mọi pipeline suy luận và huấn luyện phải tích hợp assertion:

```python
# Kiểm tra bắt buộc trước khi suy luận mẫu khó
if checkpoint_path is None or not Path(checkpoint_path).exists():
    raise FileNotFoundError(
        f"Critical Rule Violation: Cannot mine hard samples without a verified trained checkpoint! "
        f"Provided path: {checkpoint_path}"
    )
```
