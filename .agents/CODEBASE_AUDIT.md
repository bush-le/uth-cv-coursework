# CODEBASE AUDIT - QUY TRÌNH KIỂM TOÁN MÃ NGUỒN SINH RA

Code do AI sinh ra không bao giờ được commit trực tiếp. Kỹ sư phải thực hiện Codebase Audit qua 3 tầng thẩm định nghiêm ngặt.

---

## Sơ đồ Luồng Kiểm toán (Audit Flowchart)

```text
Generated Code từ AI Agent
        │
        ▼
[ TẦNG 1: KIỂM TOÁN KIẾN TRÚC ]
(Kiểm tra Clean Architecture, ranh giới module, không circular import)
        │
        ▼
[ TẦNG 2: KIỂM TOÁN DETAILED PLAN ]
(Kiểm tra có bám sát từng bước trong kế hoạch không, có tự ý bịa thêm tính năng không)
        │
        ▼
[ TẦNG 3: KIỂM TOÁN CODING CONVENTION ]
(Kiểm tra Naming, Type hints, Tensor shapes docstring, Không dùng print/global)
        │
        ▼
ĐẠT CHUẨN AUDIT ──► Chuyển sang Bước Viết Unit Test & Smoke Test
```

---

## Chi tiết 3 Tầng Kiểm toán

### Tầng 1: Audit Kiến trúc (Architecture Audit)
- Code có đặt đúng thư mục được chỉ định trong `src/` không?
- Có bị vi phạm nguyên tắc phân tách trách nhiệm không? (Ví dụ: DataLoader không được gọi logic tính loss).
- Có bị circular import giữa các module trong `src/` không?
- Các tham số có được tách ra file `configs/` thay vì viết cứng trong code không?

### Tầng 2: Audit Kế hoạch (Detailed Plan Audit)
- Code có giải quyết đúng mục tiêu được đề ra trong `.agents/TASK_DESCRIPTION.md` không?
- **Kiểm tra tính trung thực (Faithfulness Check)**: AI có tự ý "sáng tạo" thêm các class, hàm hoặc tham số ngoài phạm vi mô tả không? Nếu có, phải yêu cầu cắt bỏ ngay.
- Các trường hợp biên (Edge cases, zero division, missing keys) đã được xử lý như cam kết trong plan chưa?

### Tầng 3: Audit Quy chuẩn Mã nguồn (Convention Audit)
- Tên hàm, biến, class có tuân thủ đúng quy định trong `.agents/AI_REFERENCE.md` không?
- Có đầy đủ Type Hints theo chuẩn PEP 484 cho toàn bộ arguments và return type không?
- Có Google-style docstring ghi rõ shapes của tensor `[B, C, H, W]` không?
- Chạy lệnh kiểm tra linter:
  ```bash
  make lint
  ```
  Nếu có bất kỳ cảnh báo Ruff hoặc Black nào, code chưa đạt chuẩn Audit.

---

## 4. Bộ Tiêu chí Audit Chuyên sâu cho Computer Vision & Deep Learning

Khi kiểm toán code trong `src/`, kỹ sư và AI phải rà soát 5 bất biến kỹ thuật:

| Tiêu chí | Quy định Kỹ thuật | Cách phát hiện Vi phạm |
| :--- | :--- | :--- |
| **Vectorization Mandate** | Cấm duyệt pixel bằng vòng lặp `for` lồng nhau. Mọi phép toán phải vector hóa bằng NumPy broadcasting hoặc PyTorch ops. | Tìm thấy `for y in range(H): for x in range(W):` trong code. |
| **Tensor Shape Contracts** | Mọi phép toán trên batch phải xử lý linh hoạt kích thước batch B (không hardcode batch size). Docstring phải ghi rõ `(B, C, H, W)` hoặc `(H, W, C)`. | Thiếu docstring shape hoặc code fail khi batch size thay đổi. |
| **Device & Precision Safety** | Không hardcode `.cuda()` hoặc `'cuda:0'`. Luôn nhận `device: torch.device` từ config hoặc tự động nhận diện. | Xuất hiện `.cuda()` hardcode trực tiếp trong class module. |
| **Memory Leak Prevention** | Khi tính toán và ghi log metrics/loss, bắt buộc dùng `loss.item()` hoặc `.detach()` để cắt đứt computation graph. | Biến loss PyTorch bị append nguyên vẹn vào danh sách lưu trữ qua từng epoch. |
| **Numerical Stability** | Phải cộng thêm hằng số ε = 1e-7 vào mẫu số trước khi chia, tính log hoặc sqrt để chống văng NaN/Inf. | Biểu thức `a / b` mà `b` có thể bằng 0 không có epsilon hoặc clip. |

---

## Các Dấu hiệu Đỏ (Red Flags) Cần Reject Ngay Lập Tức
1. Code sinh ra có chứa `import` các thư viện lạ không có trong dự án.
2. Có các hàm `print()` đặt rải rác để debug thay vì dùng logger.
3. Code dài hàng trăm dòng trong một hàm duy nhất (Monolithic Function).
4. Tự ý thay đổi cấu trúc thư mục hoặc tạo file ngoài danh mục đã duyệt.
5. Vòng lặp duyệt pixel bằng Python thuần chạy quá chậm.
6. Sử dụng `try-except pass` để nuốt lỗi runtime thay vì xử lý tận gốc theo [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
