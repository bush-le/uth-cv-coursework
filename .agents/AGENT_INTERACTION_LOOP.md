# AGENT INTERACTION LOOP - CHU TRÌNH TƯƠNG TÁC VI MÔ 7 BƯỚC

Nhịp điệu làm việc giữa Kỹ sư và AI Agent cho mỗi tác vụ lập trình cụ thể:

`Describe → Generate → Review → Run → Test → Refactor → Commit`

---

## Chi tiết Từng Bước Thực hiện

```text
[ 1. DESCRIBE ] ──► Kỹ sư mô tả input, output, constraints, file target theo Behavioral Schema
       │
       ▼
[ 2. GENERATE ] ──► AI Agent sinh code tập trung, có type hints & docstring, không thêm code thừa
       │
       ▼
[ 3. REVIEW ]   ──► Kỹ sư & Agent rà soát Codebase Audit (Kiến trúc, Detailed Plan, Conventions)
       │
       ▼
[ 4. RUN ]      ──► Chạy thử nghiệm ngay trên terminal với dummy input (Sanity Check)
       │
       ▼
[ 5. TEST ]     ──► Chạy Smoke Test rồi đến Normal Test qua pytest, đối chiếu kết quả mong đợi
       │
       ▼
[ 6. REFACTOR ] ──► Tinh gọn code, vector hóa, chạy format/lint (Ruff, Black, Mypy)
       │
       ▼
[ 7. COMMIT ]   ──► Kiểm tra git status, commit nguyên tử theo chuẩn Conventional Commits
```

---

## Mẫu Đối thoại Chuẩn mực giữa Kỹ sư & Agent

### Bước 1: Kỹ sư DESCRIBE
```text
"Hãy triển khai hàm `calculate_iou(box1, box2)` vào file `src/common/metrics.py`.
- Input: box1, box2 dạng mảng 4 phần tử [x1, y1, x2, y2].
- Output: float trong khoảng [0.0, 1.0].
- Ràng buộc: Xử lý an toàn khi union_area <= 0 (trả về 0.0). Có PEP 484 type hints và docstring."
```

### Bước 2: Agent GENERATE
- Agent đọc Knowledge Store từ `.agents/AI_REFERENCE.md`.
- Sinh code chính xác theo yêu cầu, không tự ý tạo thêm các hàm không liên quan.

### Bước 3: REVIEW (Audit)
- Kỹ sư kiểm tra:
  - Code có đúng chữ ký hàm không?
  - Có bị lỗi logic chia cho 0 không?
  - Có tuân thủ naming convention không?

### Bước 4: RUN (Sanity Execution)
- Agent chạy thử nghiệm trực tiếp qua terminal:
  ```bash
  python -c "from src.common.metrics import calculate_iou; print(calculate_iou([0,0,10,10], [5,5,15,15]))"
  ```
- Kết quả in ra số thực hợp lệ, không crash.

### Bước 5: TEST (Automated Verification)
- Viết test trong `tests/test_metrics.py`:
  - Test trường hợp 2 box trùng nhau (IoU = 1.0).
  - Test trường hợp 2 box tách rời nhau (IoU = 0.0).
- Chạy `pytest tests/test_metrics.py -v`.

### Bước 6: REFACTOR & QUALITY CHECK
- Chạy công cụ tự động hóa:
  ```bash
  make format
  make lint
  ```
- Đảm bảo 0 warning, 0 error.

### Bước 7: COMMIT
- Lưu vết thay đổi nguyên tử:
  ```bash
  git add src/common/metrics.py tests/test_metrics.py
  git commit -m "feat(metrics): implement IoU calculation with boundary validation"
  ```
- Chuyển sang task tiếp theo trong Sprint Backlog.
