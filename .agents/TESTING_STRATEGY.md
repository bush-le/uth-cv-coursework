# TESTING STRATEGY - CHIẾN LƯỢC KIỂM THỬ HAI CHẾ ĐỘ

- **Motivation/Background**: Deep learning codebases without fast, deterministic testing degrade rapidly through silent regression.
- **Purpose**: Define the testing pyramid, smoke test gate (< 5s), numerical tolerance thresholds, and mock isolation.
- **Overview Pipeline**: Unit Tests -> Smoke Tests (< 5s) -> Integration Pipelines -> Checkpoint Verification.
- **Detailed Plan**: §1 Testing Pyramid; §2 Fast Smoke Test Gate; §3 Tensor Shape & Numerical Assertions; §4 CI/CD Makefile Targets.
- **References**: `../GEMINI.md`, `rules/TEST_BEFORE_CODE_RULE.md`, `rules/ZERO_MOCK_DATA_RULE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00

---

Áp dụng Chiến lược Kiểm thử 2 Chế độ: Smoke Test nhanh (< 5s) trước, Normal Test đầy đủ sau.

---

## 1. Chế độ 1: Smoke Test (Kiểm thử Nhanh Sanity)

### Mục đích:
Xác nhận nhanh trong vòng **dưới 5 giây** xem mã nguồn có khả thi để chạy tiếp hay bị crash ngay từ đầu.

### Phạm vi Kiểm tra:
- **Cú pháp (Syntax Check)**: Code có bị lỗi thụt dòng (indentation), thiếu dấu hai chấm hoặc cú pháp không hợp lệ không.
- **Imports & Dependencies**: Các lệnh `import` có thành công không, có bị thiếu thư viện hoặc circular import không.
- **Khởi tạo Đối tượng (Instantiation Check)**: Class có khởi tạo được với dummy arguments không.
- **Zero-Crash Forward**: Truyền 1 tensor giả lập kích thước cực nhỏ (ví dụ `(1, 3, 32, 32)`) để đảm bảo không văng exception.

### Lệnh Thực thi:
```bash
# Chạy suite smoke tests được đánh dấu @pytest.mark.smoke
make test-smoke
```

---

## 2. Chế độ 2: Normal Test (Kiểm thử Toàn diện)

### Mục đích:
Thẩm định tính chính xác toán học, độ ổn định của thuật toán, kiểm tra rò rỉ bộ nhớ và đánh giá hiệu năng (Performance).

### Phạm vi Kiểm tra:
- **Unit Test Chi tiết**: Từng hàm xử lý hình ảnh, hàm tính metrics được kiểm tra với các trường hợp biết trước kết quả (Ground Truth).
- **Kích thước Ma trận (Shape Assertions)**: Kiểm tra shape đầu ra qua từng bước biến đổi không gian.
- **Bất biến Số học (Numerical Invariants)**:
  - Xác suất Softmax luôn có tổng bằng 1.0.
  - Chỉ số $0.0 \le \text{IoU} \le 1.0$, $0.0 \le \text{Accuracy} \le 1.0$.
  - PSNR không bị âm hoặc NaN.
- **Trường hợp Biên (Edge Cases)**: Xử lý mảng rỗng, ảnh toàn màu đen, ảnh toàn màu trắng, bounding box diện tích bằng 0.
- **Hiệu năng & Rò rỉ Bộ nhớ (Performance & Leak Check)**: Kiểm tra thời gian thực thi (Latency ms) và không làm phình RAM/VRAM sau nhiều vòng lặp.

### Lệnh Thực thi:
```bash
# Chạy toàn bộ test suite và đo lường độ bao phủ mã nguồn
make test
```

---

## 3. Quy trình Đối chiếu Unit Test với Detail Plan

Sau khi hoàn thành một module:
1. **Viết Unit Test bám sát Detailed Plan**: Mỗi bước trong kế hoạch ở `.agents/TASK_DESCRIPTION.md` phải có ít nhất 1 test case tương ứng.
2. **Chạy Unit Test**: Chạy `pytest` trên file test vừa viết.
3. **Đối chiếu Tiêu chuẩn Nghiệm thu**:
   - Nếu test fail: Kích hoạt quy trình phân tích nguyên nhân gốc rễ tại [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) để sửa code tận gốc; tuyệt đối không đoán mò hay dùng try-except che giấu lỗi.
   - Nếu test pass nhưng không phản ánh đúng yêu cầu trong Detail Plan: Sửa lại test case cho chặt chẽ hơn.
4. **Không bao giờ xóa bỏ test case để làm cho test pass**: Nếu test bị lỗi do thiết kế ban đầu sai, phải cập nhật lại tài liệu Design trước khi sửa test.

---

## 4. Mẫu Cấu trúc File Test Chuẩn (`tests/test_template.py`)

Mọi file test trong thư mục `tests/` phải phân tách rõ giữa Smoke Test và Normal Test:

```python
"""Template kiểm thử 2 chế độ: Smoke Test vs Normal Test."""

import pytest
import torch


# ==========================================
# SMOKE TEST SUITE (< 5s, Sanity Check)
# ==========================================
@pytest.mark.smoke
def test_module_import_and_instantiation():
    """Kiểm tra import và khởi tạo module không crash."""
    from src.models.classical import ClassicalFilter  # minh họa

    filter_instance = ClassicalFilter()
    assert filter_instance is not None


@pytest.mark.smoke
def test_dummy_forward_zero_crash():
    """Kiểm tra forward pass với dummy batch kích thước nhỏ không văng lỗi."""
    dummy_input = torch.randn(2, 3, 32, 32)
    # output = model(dummy_input)
    # assert output is not None


# ==========================================
# NORMAL TEST SUITE (Logic & Invariants)
# ==========================================
def test_output_tensor_shapes():
    """Kiểm tra tensor shape qua các phép biến đổi không gian."""
    b, c, h, w = 4, 3, 64, 64
    x = torch.randn(b, c, h, w)
    # out = model(x)
    # assert out.shape == (b, NUM_CLASSES)


def test_numerical_invariants():
    """Kiểm tra các ràng buộc toán học bất biến."""
    # Ví dụ: probabilities phải có tổng bằng 1.0
    # probs = torch.softmax(logits, dim=-1)
    # assert torch.allclose(probs.sum(dim=-1), torch.ones(probs.shape[0]))
    pass


def test_edge_cases_and_boundaries():
    """Kiểm tra các trường hợp biên: tensor toàn 0, kích thước 1x1, v.v."""
    zero_tensor = torch.zeros(1, 3, 32, 32)
    # output = model(zero_tensor)
    # assert not torch.isnan(output).any()
    pass
```
