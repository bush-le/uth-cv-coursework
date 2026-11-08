# PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md — Bước 2: Đưa Qua Baseline `dinoViT` Để Tìm Mẫu Khó (`Hard Sample`)

- **Motivation/Background**: Thực hiện đúng yêu cầu trọng tâm của giảng viên ở Bước 2 (`baseline model dinoViT (benchmark) -> reject/bypass DF? nếu bypass thì sẽ có hard sample`).
- **Purpose**: Giải thích đơn giản cách mô hình gốc `dinoViT` chấm điểm ảnh giả và cách tách các ảnh vượt rào (`Bypass`) thành tập Mẫu khó (`Hard Sample`).
- **Overview Pipeline**: Tập ảnh giả từ Bước 1 -> Đưa vào mô hình gốc `dinoViT` -> Chấm điểm nghi ngờ giả $P(\text{fake})$ -> Nếu $P(\text{fake}) \ge 0.5$ thì Bắt được (`Reject`), nếu $P(\text{fake}) < 0.5$ thì Bỏ lọt (`Bypass`) -> Lưu nhóm `Bypass` thành `Hard Sample`.
- **Detailed Plan**: §1 Mô hình gốc `dinoViT` hoạt động thế nào?; §2 Quy tắc tách `Reject` và `Bypass`; §3 Vì sao các ảnh `Bypass` lại là Mẫu khó quý giá nhất?
- **References**: [ ../PURPOSE.md ](../PURPOSE.md), [ PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md ](PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md), [ PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md ](PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:39:00+07:00

---

## 1. Mô Hình Gốc `dinoViT` Hoạt Động Thế Nào?

Trong hệ thống của nhóm (`src/models/detector/image_detector.py`), mô hình **Baseline `dinoViT`** xử lý một bức ảnh khuôn mặt theo 4 bước rất dễ hiểu:

1. **Chia ảnh thành lưới ô vuông nhỏ (Patch)**: Cắt bức ảnh khuôn mặt thành các ô vuông nhỏ $16 \times 16$ điểm ảnh.
2. **So sánh toàn bộ khuôn mặt (Self-Attention)**: Các tầng Transformer đối chiếu sự ăn khớp giữa mắt, mũi, miệng, làn da và ánh sáng xung quanh.
3. **Rút trích vector tổng hợp (`[CLS]` Token)**: Tóm tắt toàn bộ đặc điểm của khuôn mặt thành một vector chuẩn hóa.
4. **Chấm điểm nghi ngờ ảnh giả $P(\text{fake}) \in [0, 1]$**:
   - Điểm gần `1.0`: Mô hình rất chắc chắn đây là **ảnh giả**.
   - Điểm gần `0.0`: Mô hình tin rằng đây là **ảnh thật**.

---

## 2. Quy Tắc Tách Nhánh `Reject` (Bắt Được) vs. `Bypass` (Bỏ Lọt $\rightarrow$ Mẫu Khó)

Tất cả các bức ảnh đưa vào kiểm tra ở Bước 2 đều là **ảnh giả 100%** (nhãn thật sự `label = 1`) do Stable Diffusion sinh ra ở Bước 1. Khi chạy qua **Baseline `dinoViT`** (với ngưỡng quyết định $\tau = 0.5$), hệ thống chia chúng thành 2 nhánh:

```text
                       ┌───► Điểm P(fake) >= 0.5  ──►  REJECT (Bị bắt giữ)
                       │     • Mô hình gốc đã nhận ra được đây là ảnh giả.
                       │     • Lưu làm ảnh giả thông thường (data_source = "synthetic").
Ảnh giả SD (label=1) ──┤
  qua Baseline dinoViT │
                       └───► Điểm P(fake) < 0.5   ──►  BYPASS (Lọt lưới / Đánh lừa được dinoViT!)
                             • Là ảnh giả nhưng mô hình gốc lại đoán nhầm là ảnh thật.
                             • Đánh dấu ngay: is_hard_sample = True (data_source = "hard_sample").
```

---

## 3. Tại Sao Các Mẫu `Bypass` Lại Là "Mẫu Khó" (`Hard Sample`) Quan Trọng Nhất?

- Những bức ảnh bị bắt giữ (`Reject`) thường có dấu vết giả mạo khá rõ nên mô hình gốc đã nhận biết được rồi.
- Ngược lại, những bức ảnh **vượt rào (`Bypass`)** — đặc biệt là ảnh chỉ sửa nhẹ vùng mắt/miệng (`Inpainting`) hoặc hòa trộn mượt mà từ ảnh gốc (`Img2Img`) — chính là **lỗ hổng thực sự** của mô hình `dinoViT` gốc.
- Pipeline `src/pipelines/mine_hard_samples.py` tự động gom toàn bộ các mẫu `Bypass` này vào file `hard_samples_manifest.jsonl` để ưu tiên dạy lại cho mô hình ở Bước 3.
