# PIPELINE SEPARATION - NGUYÊN TẮC PHÂN TÁCH PIPELINE ĐỘC LẬP

Hệ thống bắt buộc chia tách thành các pipeline độc lập. Không nhồi nhét toàn bộ logic vào một prompt hoặc một file duy nhất.

---

## 5 Pipeline Tiêu chuẩn trong Computer Vision

```text
[ Data Preparation ] ──► [ Model Definition ] ──► [ Training Loop ] ──► [ Evaluation & Benchmark ]
                                                            │
                                                            ▼
                                                [ Inference & Deployment ]
```

Mỗi pipeline phải có **3 yếu tố độc lập**:
1. **Prompt riêng**: Chỉ tập trung vào phạm vi kỹ thuật của pipeline đó.
2. **Tài liệu riêng**: Đặc tả input, output và cấu hình YAML riêng trong `configs/`.
3. **Checklist riêng**: Bộ tiêu chí kiểm thử độc lập trước khi tích hợp.

---

### 1. Data Preparation Pipeline
- **Trách nhiệm**: Tải dữ liệu, làm sạch, chia split (train/val/test), áp dụng augmentation, tạo DataLoader.
- **Thư mục phụ trách**: `src/data/`, `configs/dataset/`, `data/`.
- **Checklist độc lập**:
  - [ ] Kiểm tra không bị data leakage giữa train set và test set.
  - [ ] Batch tensor sinh ra đúng kích thước `[B, C, H, W]`.
  - [ ] Tốc độ nạp dữ liệu không làm nghẽn tiến trình.

### 2. Model Architecture Pipeline
- **Trách nhiệm**: Xây dựng kiến trúc mô hình (CNN, Transformer, YOLO, Classical filters).
- **Thư mục phụ trách**: `src/models/`, `configs/model/`.
- **Checklist độc lập**:
  - [ ] Forward pass hoạt động chính xác với dummy input.
  - [ ] Tách rời hoàn toàn khỏi training loop (không chứa optimizer hay loss function bên trong class Model).
  - [ ] Đếm số lượng tham số (Parameters) và kích thước mô hình.

### 3. Training Pipeline
- **Trách nhiệm**: Điều phối vòng lặp huấn luyện, tính loss, cập nhật gradient, scheduler, checkpointing.
- **Thư mục phụ trách**: `src/pipelines/train.py`, `configs/config.yaml`.
- **Checklist độc lập**:
  - [ ] Có lưu checkpoint `best.pt` và `last.pt`.
  - [ ] Có cơ chế Early Stopping chống overfitting.
  - [ ] Ghi nhận đầy đủ log Loss và Metrics qua TensorBoard / Wandb.

### 4. Evaluation & Benchmark Pipeline
- **Trách nhiệm**: Đánh giá trên tập test, đo lường các chỉ số định lượng (Accuracy, mAP, IoU, PSNR) và trực quan hóa kết quả.
- **Thư mục phụ trách**: `src/pipelines/evaluate.py`, `src/common/metrics.py`, `src/common/visualizer.py`.
- **Checklist độc lập**:
  - [ ] Chạy hoàn toàn trên chế độ `model.eval()` và `torch.no_grad()`.
  - [ ] Xuất báo cáo bảng số liệu rõ ràng.
  - [ ] Lưu ảnh minh họa vào `experiments/predictions/`.

### 5. Inference & Deployment Pipeline (Nếu có)
- **Trách nhiệm**: Suy luận trên ảnh/video thực tế từ webcam hoặc file đơn lẻ với độ trễ thấp.
- **Thư mục phụ trách**: `src/pipelines/infer.py`.
- **Checklist độc lập**:
  - [ ] Đo lường độ trễ (latency ms/frame) và FPS.
  - [ ] Tối ưu hóa tiền xử lý và hậu xử lý (NMS, thresholding).
