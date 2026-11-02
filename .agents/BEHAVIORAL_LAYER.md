# BEHAVIORAL LAYER - ĐIỀU KHIỂN HÀNH VI QUA PROMPTING SCHEMA

Toàn bộ hành vi AI được điều khiển qua các file Markdown có cấu trúc đồng nhất thay vì câu lệnh tùy hứng.

---

## Cấu trúc Chuẩn của một File Hành vi (Behavioral Markdown Schema)

Mọi file hướng dẫn hoặc prompt đặc tả một chức năng/pipeline đều phải tuân theo 7 trường thông tin:

```text
Name:        [Tên ngắn gọn của chức năng / workflow]
Description: [Mô tả tổng quan về tác vụ]
Purpose:     [Mục đích kỹ thuật - Tại sao phải thực hiện]
Input:       [Dữ liệu đầu vào: kiểu dữ liệu, shapes, đường dẫn file]
Output:      [Dữ liệu đầu ra: kiểu dữ liệu, artifacts, vị trí lưu]
How to do:   [Quy trình từng bước thực hiện (Workflow)]
Examples:    [Mẫu input/output hoặc lệnh gọi minh họa]
```

---

## Các Mẫu Điển hình trong Computer Vision

### Mẫu 1: Data Preparation Pipeline
```markdown
Name:
Data Preparation Pipeline

Description:
Tải, xác thực và tiền xử lý dữ liệu ảnh thô thành dữ liệu sẵn sàng nạp vào DataLoader.

Purpose:
Đảm bảo tính toàn vẹn dữ liệu, loại bỏ ảnh hỏng và chuẩn hóa kích thước trước khi huấn luyện.

Input:
- Thư mục ảnh thô: `data/raw/`
- Tỷ lệ phân chia: train (0.8), val (0.1), test (0.1)
- Kích thước ảnh mục tiêu: (224, 224)

Output:
- Thư mục dữ liệu chuẩn hóa: `data/processed/train/`, `val/`, `test/`
- File thống kê số lượng mẫu và metadata: `data/processed/metadata.json`

How to do:
1. Quét toàn bộ file ảnh trong `data/raw/` và kiểm tra tính toàn vẹn qua PIL/OpenCV.
2. Loại bỏ các file có kích thước 0 bytes hoặc corrupt headers.
3. Chia ngẫu nhiên danh sách file theo tỷ lệ định sẵn với seed cố định.
4. Resize và lưu ảnh vào thư mục tương ứng trong `data/processed/`.
5. Ghi log tổng kết số lượng mẫu của từng class.

Examples:
python -m src.data.datamodule --raw-dir data/raw --output-dir data/processed --split 0.8 0.1 0.1
```

---

### Mẫu 2: Model Evaluation Pipeline
```markdown
Name:
Model Evaluation & Benchmarking

Description:
Chạy đánh giá mô hình trên tập kiểm thử (Test Split) và xuất báo cáo định lượng.

Purpose:
Đo lường độ chính xác và tốc độ suy luận của mô hình để kiểm chứng giả thuyết thực nghiệm.

Input:
- Model Checkpoint: `experiments/checkpoints/best.pt`
- Test DataLoader: tập dữ liệu từ `data/processed/test/`
- File cấu hình: `configs/experiment/template.yaml`

Output:
- Bảng metrics JSON: `experiments/logs/eval_metrics.json`
- Ảnh trực quan hóa dự đoán: `experiments/predictions/`

How to do:
1. Nạp weights từ checkpoint vào mô hình và chuyển sang chế độ `model.eval()`.
2. Tắt tính toán gradient (`torch.no_grad()`).
3. Lặp qua từng batch trong Test DataLoader và ghi nhận predictions.
4. Tính toán các chỉ số mAP, Accuracy, F1-Score qua `src/common/metrics.py`.
5. Vẽ bounding boxes / masks lên ảnh và lưu vào `experiments/predictions/`.
6. Xuất bảng tổng kết số liệu ra log console và file JSON.

Examples:
python -m src.pipelines.evaluate experiment=template
```
