# 2026_11_09_BASELINE_AUDIT_AND_CORRECTION_PLAN.md — Báo Cáo Thẩm Định Lỗi Baseline & Kế Hoạch Chuẩn Hóa Khoa Học

- **Motivation/Background**: Người dùng phản ánh gay gắt và hoàn toàn chính xác về việc xuất hiện "Mẫu Khó" (`Hard Samples`) trong thư mục dữ liệu khi chưa hề có mô hình Baseline nào được huấn luyện thực tế. Qua thẩm định toàn diện codebase, phát hiện `mine_hard_samples.py` đã dùng `DeepfakeImageDetector` với đầu phân loại ngẫu nhiên (`random initialized weights`), chưa nạp bất kỳ checkpoint nào, dẫn đến kết quả phân loại và mẫu khó hoàn toàn là ngẫu nhiên, vi phạm nguyên tắc "Zero-Assumption" và chuẩn mực nghiên cứu khoa học.
- **Purpose**: Bản báo cáo công khai toàn bộ các lỗi nghiêm trọng trong baseline hiện tại và vạch ra kế hoạch từng bước (Step-by-step Execution Plan) để sửa chữa triệt để, thiết lập quy trình khoa học khép kín từ huấn luyện Baseline $D_0 \to$ Khai thác Mẫu khó thật $\to$ Huấn luyện đối chứng $D_1, D_2 \to$ Kiểm chứng tác dụng.
- **Overview Pipeline**: Dọn dẹp dữ liệu giả lập trong `data/hard_samples/` -> Khắc phục lỗi phân tầng `identity_disjoint` trong `DeepfakeImageDataModule` -> Cải tổ `mine_hard_samples.py` bắt buộc có checkpoint thật -> Huấn luyện Baseline $D_0$ chuẩn mực -> Đào Mẫu khó thật từ mô hình $D_0$ -> Huấn luyện đối chứng $D_1, D_2$ -> Đánh giá độc lập & cập nhật tài liệu.
- **Detailed Plan**: §1 Báo cáo thẩm định các lỗi nghiêm trọng đã tìm thấy; §2 Chu trình 3 bước chuẩn theo yêu cầu giảng viên; §3 Kế hoạch thực thi chi tiết 6 nhiệm vụ; §4 Định nghĩa hoàn thành (DoD).
- **References**: [../PURPOSE.md](../PURPOSE.md), [../OVERVIEW.md](../OVERVIEW.md), [../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md](../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md), [../experiments/EXPERIMENT_PROTOCOL.md](../experiments/EXPERIMENT_PROTOCOL.md).
- **Created**: 2026-11-09T13:05:00+07:00
- **Last Updated**: 2026-11-09T13:05:00+07:00

---

## 1. 🚨 Báo Cáo Thẩm Định Các Lỗi Nghiêm Trọng Trong Baseline

Qua quá trình rà soát toàn bộ mã nguồn, cấu hình và dữ liệu, chúng tôi xác nhận các lỗi nghiêm trọng sau:

| Mã lỗi | Vị trí (File & Dòng) | Bản chất lỗi | Hậu quả khoa học |
| :--- | :--- | :--- | :--- |
| **BUG-01** (Chí tử) | `src/pipelines/mine_hard_samples.py:233-248` | `DeepfakeImageDetector` được khởi tạo trực tiếp và đem đi dự đoán mà **KHÔNG nạp checkpoint** (`--checkpoint` thậm chí không có trong parser). Đầu phân lớp `self.classifier` là trọng số ngẫu nhiên của PyTorch. | Điểm $P(\text{fake})$ hoàn toàn là ngẫu nhiên. Mẫu nào rơi vào $< 0.511$ bị gán bừa là "Mẫu khó", sau đó copy vào `data/hard_samples/images/`. |
| **BUG-02** (Quy trình) | `src/pipelines/train.py` & `scripts/run_pipeline.sh` | Lỗi "Con gà - Quả trứng": `mine_hard_samples.py` (Phase 2) cần mô hình Baseline đã học $D_0$ để chấm điểm, nhưng `train.py` lại gộp cả $D_0, D_1, D_2$ chạy một thể và đòi hỏi phải có `hard_samples_manifest.jsonl` mới chạy được $D_2$. | Đảo lộn vòng đời nghiên cứu, dẫn đến việc lấy mẫu khó giả lập ở Bước 2 trước khi có mô hình được huấn luyện. |
| **BUG-03** (Thoái hóa) | `src/data/image_datamodule.py:95-122` | Phân chia `identity_disjoint=True` không có phân tầng nhãn (Stratification). Toàn bộ nhóm ảnh giả ít ỏi bị dồn vào tập Train, khiến tập Test có **0 ảnh giả** (toàn bộ nhãn 0). | Báo cáo đánh giá `experiments/logs/benchmark_comparison.json` bị thoái hóa: `Precision=0, Recall=0, F1=0`, mọi delta tăng trưởng giữa $D_0, D_1, D_2$ đều bằng `0.0`. |
| **BUG-04** (Ghi đè) | `src/pipelines/train_image_detector.py:156-170` | Checkpoint chỉ lưu cố định tên `best_model.pt` và `last_model.pt`. Khi chạy $D_0 \to D_1 \to D_2$, mô hình sau ghi đè lên mô hình trước, không lưu vết `baseline_d0.pt` để đem đi đào mẫu khó. | Mất dấu trọng số của từng giai đoạn đối chứng. |
| **BUG-05** (Trùng lặp $D_0, D_1$) | `scripts/organize_topic_dataset.py:165-190` | $D_0$ được tạo bằng cách lấy toàn bộ ảnh thật + toàn bộ ảnh giả đã sinh (gán nhãn `baseline`), trong khi $D_1$ cũng lấy toàn bộ ảnh thật + toàn bộ ảnh giả đã sinh. | $D_0$ và $D_1$ có tập dữ liệu y hệt nhau, không có ý nghĩa đối chứng thực nghiệm. |

---

## 2. 🎯 Luồng Nghiệp Vụ Chuẩn Khoa Học (Topic 11 - Nhóm 4)

Để tuân thủ 100% [docs/PURPOSE.md](../PURPOSE.md), chu trình phải vận hành theo đúng 4 bước tuần tự:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Bước 1: Chuẩn bị Dữ liệu Huấn luyện & Sinh ảnh giả Stable Diffusion (Phase 1) │
│ - Ảnh thật FFHQ + CelebA-HQ (label=0).                                       │
│ - Sinh ảnh giả đa chế độ (text2image, inpainting, img2img) (label=1).         │
│ - Phân chia độc lập danh tính: một phần làm Baseline Fakes (D0),             │
│   phần còn lại để dành cho Bước 2 thử thách mô hình.                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Bước 2: Huấn luyện Mô hình Gốc Baseline D0 (Stage D0)                         │
│ - Huấn luyện DeepfakeImageDetector trên tập D0 (Real + Baseline Fake).      │
│ - Lưu checkpoint độc lập: experiments/checkpoints/baseline_d0_best.pt.      │
│ - Đảm bảo mô hình đạt AUC > 0.85 trên tập Val của D0.                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Bước 3: Đưa ảnh SD mới qua Baseline D0 để Bóc Tách Mẫu Khó (Phase 2)        │
│ - Nạp ĐÚNG checkpoint baseline_d0_best.pt vào mine_hard_samples.py.         │
│ - Chấm điểm tập ảnh giả thử thách mới:                                      │
│   + Bị bắt (P(fake) >= 0.5): Ảnh giả thông thường (Synthetic).              │
│   + Bỏ lọt (P(fake) < 0.5): Ảnh giả tinh vi lừa được Baseline D0            │
│     ==> ĐÂY MỚI LÀ MẪU KHÓ THẬT SỰ (Hard Samples)!                          │
│ - Lưu danh mục và copy ảnh vào data/hard_samples/.                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Bước 4: Huấn luyện Đối chứng D1, D2 & Kiểm chứng Tác dụng (Phase 3)          │
│ - Huấn luyện D1: D0 + Toàn bộ ảnh giả SD ==> lưu augmented_d1_best.pt.       │
│ - Huấn luyện D2: D0 + Toàn bộ ảnh SD + Mẫu khó (oversample x2, loss x2)      │
│   ==> lưu hard_boosted_d2_best.pt.                                           │
│ - Đánh giá cả 3 mô hình trên cùng tập Test độc lập (có cả Real & Fake):      │
│   Chứng minh D2 tăng ROC-AUC, tăng Hard-Sample Recall, giảm Bypass Rate!    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 📋 Kế Hoạch Thực Thi Từng Bước (Execution Tasks)

### Task 1: Dọn Dẹp Dữ Liệu Chưa Qua Kiểm Chứng
- Xóa các tệp ảnh và manifest ngẫu nhiên trong `data/hard_samples/images/` và `data/hard_samples/hard_samples_manifest.jsonl`.
- Đảm bảo `data/hard_samples/` chỉ chứa `.gitkeep` cho đến khi mô hình $D_0$ thật hoàn thành việc bóc tách.

### Task 2: Khắc Phục Lỗi Phân Tầng Trong `DeepfakeImageDataModule`
- Sửa hàm `_partition_splits()`: Thực hiện phân chia có phân tầng (Stratified Group Partitioning) theo `source_real_id`.
- Đảm bảo cả 3 tập `train`, `val`, `test` luôn có đủ cả ảnh nhãn 0 (Real) và ảnh nhãn 1 (Fake), không bao giờ để tập test bị rỗng nhãn 1.
- Bổ sung unit test kiểm tra hợp đồng phân tầng không rỗng.

### Task 3: Cải Tổ `src/pipelines/mine_hard_samples.py`
- Bắt buộc phải truyền `--checkpoint` (nếu không có checkpoint hoặc checkpoint không tồn tại, script phải dừng lại báo lỗi thay vì âm thầm chạy ngẫu nhiên).
- Nạp chính xác `model_state_dict` vào `DeepfakeImageDetector` trước khi thực hiện chấm điểm suy luận.
- Ghi rõ độ tin cậy và nguồn checkpoint trong báo cáo `HardSampleMiningReport`.

### Task 4: Tách Bạch Checkpoint & Quy Trình Huấn Luyện Trong `train.py`
- Bổ sung tham số lưu checkpoint theo từng stage:
  - Stage D0: Lưu `experiments/checkpoints/baseline_d0_best.pt`.
  - Stage D1: Lưu `experiments/checkpoints/augmented_d1_best.pt`.
  - Stage D2: Lưu `experiments/checkpoints/hard_boosted_d2_best.pt`.
- Cho phép chạy từng giai đoạn độc lập qua CLI:
  `bash scripts/run_pipeline.sh train configs/config.yaml --stage D0`

### Task 5: Thực Thi Chu Trình Khoa Học Hoàn Chỉnh
1. Huấn luyện Baseline $D_0 \to$ Thu được `baseline_d0_best.pt`.
2. Chạy `mine_hard_samples.py --checkpoint experiments/checkpoints/baseline_d0_best.pt` trên tập ảnh SD $\to$ Thu được Mẫu Khó THẬT SỰ.
3. Huấn luyện $D_1$ và $D_2$ $\to$ Thu được `augmented_d1_best.pt` và `hard_boosted_d2_best.pt`.
4. Chạy `evaluate.py` so sánh 3 mô hình trên tập Test chuẩn $\to$ Xuất báo cáo số liệu `experiments/logs/benchmark_comparison.json`.

### Task 6: Cập Nhật Visual Gallery, Coursework Notebooks & Tài Liệu
- Cập nhật [experiments/predictions/real_vs_synthetic_gallery.png](experiments/predictions/real_vs_synthetic_gallery.png) với các mẫu khó thật sự đã vượt qua $D_0$.
- Cập nhật 3 Notebooks trong [notebooks/coursework/](notebooks/coursework/) phản ánh chính xác kết quả huấn luyện.
- Chạy toàn bộ test suite (`pytest tests/ -v`, `make test-smoke`) xác nhận 100% pass.
