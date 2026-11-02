# TROUBLESHOOTING - QUY TRÌNH XỬ LÝ SỰ CỐ & PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ

Quy trình chuẩn hóa xử lý lỗi runtime, crash và sai lệch số học trong Computer Vision & MLOps. Nghiêm cấm sửa mò (trial-and-error) hoặc dùng try-except để che giấu lỗi.

---

## 1. Quy trình Phân tích Nguyên nhân Gốc rễ 5 Bước (5-Step Root-Cause Analysis)

Mỗi khi mã nguồn gặp lỗi crash, văng exception hoặc trả về kết quả bất thường, AI Agent và Kỹ sư bắt buộc thực hiện tuần tự 5 bước:

```text
[ Bước 1: Reproduce ] ──► Tái hiện lỗi với dummy batch nhỏ nhất có thể
          │
          ▼
[ Bước 2: Trace Shapes & Dtypes ] ──► In kích thước, kiểu dữ liệu và device qua từng layer
          │
          ▼
[ Bước 3: Check Invariants ] ──► Kiểm tra ranh giới toán học: NaN, Inf, chia cho 0, giá trị âm
          │
          ▼
[ Bước 4: Isolate Memory & Graph ] ──► Rà soát rò rỉ GPU VRAM và giữ computational graph
          │
          ▼
[ Bước 5: Root Fix & Smoke Test ] ──► Sửa tận gốc nguyên nhân và xác nhận qua make test-smoke
```

### Bước 1: Tái hiện Lỗi Tối giản (Reproduce with Minimal Input)
- Không cố gắng chạy lại toàn bộ dataset lớn hoặc full epoch.
- Tạo một script kiểm tra cô lập hoặc test case với dummy tensor kích thước nhỏ nhất có thể (ví dụ: `torch.randn(1, 3, 32, 32)`).
- Xác nhận lỗi tái hiện 100% trong môi trường tối giản.

### Bước 2: Truy vết Chiều & Kiểu dữ liệu (Trace Shapes & Dtypes)
- In chính xác 3 thuộc tính tại vị trí nghi vấn:
  ```python
  logger.debug(f"Tensor shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")
  ```
- Kiểm tra tính tương thích giữa các layer tích chập, pooling và fully-connected.
- Đảm bảo tensor không bị chuyển đổi ngầm giữa CPU và CUDA.

### Bước 3: Kiểm tra Ổn định Số học (Check Numerical Stability)
- Kiểm tra sự xuất hiện của giá trị vô định hoặc vô cực:
  ```python
  assert not torch.isnan(loss).any(), "Loss contains NaN"
  assert not torch.isinf(loss).any(), "Loss contains Inf"
  ```
- Rà soát các phép toán nhạy cảm: chia (division), logarithm (`torch.log`), căn bậc hai (`torch.sqrt`), softmax. Đảm bảo luôn có hằng số an toàn ε = 1e-7.

### Bước 4: Cô lập Bộ nhớ & Đồ thị Tính toán (Isolate Memory & Graph)
- Kiểm tra các biến tích lũy trong vòng lặp huấn luyện:
  - Cắt đứt computation graph: bắt buộc dùng `loss.item()` thay vì giữ nguyên tensor loss.
  - Sử dụng `torch.cuda.empty_cache()` khi chuyển đổi giữa train loop và validation loop.

### Bước 5: Khắc phục Tận gốc & Kiểm chứng (Root Fix & Smoke Test)
- Tuyệt đối không thêm `try ... except: pass` để dập tắt exception.
- Triển khai giải pháp xử lý triệt để nguyên nhân gốc rễ.
- Chạy lệnh kiểm thử nhanh:
  ```bash
  make test-smoke
  ```
- Đảm bảo 100% test cases vượt qua mà không làm vỡ các module khác.

---

## 2. Cẩm nang Xử lý 6 Sự cố Kinh điển trong Computer Vision

| Sự cố Kỹ thuật | Nguyên nhân Gốc rễ | Giải pháp Xử lý Chuẩn mực |
| :--- | :--- | :--- |
| **CUDA Out Of Memory (OOM)** | 1. Append tensor PyTorch vào list trong training loop mà không gọi `.item()`.<br/>2. Batch size quá lớn vượt dung lượng VRAM.<br/>3. Quên gọi `optimizer.zero_grad()`. | 1. Đổi `history.append(loss)` thành `history.append(loss.item())`.<br/>2. Giảm `batch_size` trong file `configs/` tương ứng.<br/>3. Kích hoạt mixed precision `torch.cuda.amp.autocast()`. |
| **Loss = NaN / Inf** | 1. Phép chia cho 0 trong tính IoU / Focal Loss.<br/>2. Log của 0 trong Cross-Entropy (`torch.log(0)`).<br/>3. Gradient exploding do learning rate quá cao. | 1. Cộng thêm hằng số `eps = 1e-7` vào mẫu số.<br/>2. Dùng `torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)`.<br/>3. Bổ sung `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`. |
| **Shape Mismatch Across Layers** | Lệch chiều không gian khi chuyển từ Conv2D sang Linear layer (`mat1 and mat2 shapes cannot be multiplied`). | 1. Sử dụng `nn.AdaptiveAvgPool2d((1, 1))` trước layer phân loại để cố định kích thước feature map.<br/>2. Dùng hàm forward dummy để suy luận tự động chiều in_features. |
| **DataLoader Deadlock / Crash** | Xung đột đa tiến trình giữa thư viện OpenCV C++ (`cv2`) và cơ chế fork multiprocessing của PyTorch trên Linux. | Thêm lệnh vô hiệu hóa đa luồng nội bộ OpenCV vào hàm khởi tạo worker của DataLoader:<br/>`cv2.setNumThreads(0)`<br/>`cv2.ocl.setUseOpenCL(False)` |
| **Color Space / Bounding Box Error** | 1. Đọc ảnh bằng OpenCV (BGR) nhưng nạp vào model huấn luyện trên RGB.<br/>2. Đảo lộn định dạng box `[x1, y1, x2, y2]` với `[cx, cy, w, h]`. | 1. Chuyển đổi tường minh `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` ngay sau khi đọc ảnh.<br/>2. Kiểm tra điều kiện hình học: `assert (x2 >= x1).all()` và `assert (y2 >= y1).all()`. |
| **Metric Bất thường / Fake Performance** | 1. Data leakage giữa tập Train và tập Val/Test.<br/>2. Quên chuyển mô hình sang `model.eval()`.<br/>3. Quên bọc code đánh giá trong `torch.no_grad()`. | 1. Kiểm tra hash checksum danh sách file giữa các tập split.<br/>2. Đảm bảo mọi pipeline evaluate và infer đều chạy dưới bối cảnh:<br/>`model.eval()` kèm `with torch.no_grad():`. |

---

## 3. Danh sách Điều Cấm Kỵ khi Xử lý Sự cố (Anti-Patterns)

1. **Cấm dùng `try-except pass`**: Nuốt exception chỉ làm lỗi tiềm ẩn và bùng phát ở các tầng sâu hơn với chi phí sửa chữa đắt hơn gấp nhiều lần.
2. **Cấm sửa mò ngẫu nhiên (Random Guessing)**: Mọi thay đổi mã nguồn phải có giả thuyết kỹ thuật rõ ràng và giải thích được nguyên nhân.
3. **Cấm hardcode tham số ma thuật để pass test**: Không được ép giá trị cố định vào code chỉ để qua mặt một test case cụ thể.
4. **Cấm sửa test để che giấu lỗi code**: Nếu test case phản ánh đúng hợp đồng kỹ thuật mà code chạy sai, phải sửa code chứ không được hạ thấp tiêu chuẩn của test.
