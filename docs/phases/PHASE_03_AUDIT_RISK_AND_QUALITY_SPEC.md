# PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md — Bước 3: Huấn Luyện Lại (`Fine-Tune`) & Kiểm Chứng Tác Dụng

- **Motivation/Background**: Thực hiện bước cuối cùng trong yêu cầu của giảng viên (`data baseline + syn data + hard sample -> finetuned up -> nếu bắt được -> có tác dụng`).
- **Purpose**: Giải thích đơn giản cách gộp 3 nguồn dữ liệu, 3 kỹ thuật giúp `dinoViT` học tốt các mẫu khó và cách kiểm tra không để rò rỉ dữ liệu.
- **Overview Pipeline**: Gộp `Data Baseline + Syn Data + Hard Sample` -> Khóa danh tính người gốc -> Tinh chỉnh `dinoViT` (Mở khóa các tầng cuối + Nhân bản mẫu khó + Phạt nặng khi đoán sai mẫu khó) -> Kiểm tra bắt lại mẫu khó trên tập Test.
- **Detailed Plan**: §1 Gộp 3 nguồn dữ liệu; §2 Ba cách giúp `dinoViT` học thuộc đặc điểm của Mẫu khó; §3 Kiểm tra an toàn không trùng người giữa tập Học và tập Thi.
- **References**: [ ../PURPOSE.md ](../PURPOSE.md), [ PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md ](PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md), [ ../experiments/EXPERIMENT_PROTOCOL.md ](../experiments/EXPERIMENT_PROTOCOL.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:39:00+07:00

---

## 1. Gộp 3 Nguồn Dữ Liệu Để Huấn Luyện Lại

Ở Bước 3 (`src/pipelines/train_image_detector.py`), hàm `combine_training_manifests()` gộp chung 3 nguồn dữ liệu lại thành một bộ dữ liệu thống nhất:

1. **`Data Baseline` (`baseline`)**: Tập ảnh thật và ảnh giả đối chứng ban đầu (giúp mô hình không quên kiến thức gốc).
2. **`Synthetic Data` (`synthetic`)**: Các ảnh giả Stable Diffusion đã bị `dinoViT` bắt được ở Bước 2 (giúp giữ sự đa dạng của ảnh tổng hợp).
3. **`Hard Samples` (`hard_sample`)**: Các ảnh giả tinh vi từng đánh lừa được `dinoViT` ở Bước 2 (giúp vá đúng lỗ hổng của mô hình).

---

## 2. Ba Kỹ Thuật Giúp `dinoViT` Bắt Được Mẫu Khó Khi Học Lại (`Fine-Tune`)

Để mô hình sau khi huấn luyện lại thực sự bắt được các mẫu khó mà không bị học vẹt, nhóm áp dụng 3 kỹ thuật đơn giản nhưng hiệu quả:

1. **Mở khóa các tầng cuối của `dinoViT` (`Partial Unfreeze`)**:
   - Giữ nguyên các tầng đầu của `dinoViT` (để bảo toàn khả năng nhìn hình dạng khuôn mặt cơ bản), chỉ mở khóa các khối Transformer cuối cùng và bộ phân lớp để tinh chỉnh theo dữ liệu mới.
2. **Cho Mẫu khó xuất hiện nhiều hơn trong lúc học (`Hard-Sample Oversampling`)**:
   - Trong tập huấn luyện (`Train`), các bức ảnh có cờ `is_hard_sample = True` được nhân bản để mô hình được nhìn thấy và ôn luyện thường xuyên hơn.
3. **Tăng điểm phạt nếu đoán sai Mẫu khó (`Weighted BCE Loss`)**:
   - Trong hàm tính lỗi (Loss), nếu mô hình vẫn đoán nhầm một **Mẫu khó** thì mức phạt sẽ được nhân cao hơn (ví dụ gấp đôi so với mẫu thường), buộc mô hình phải tập trung sửa sai ở những bức ảnh này.

---

## 3. Kiểm Soát Chất Lượng & Chống Gian Lận Dữ Liệu

- **Không trùng người giữa tập Học và tập Thi**: Tất cả ảnh thật, ảnh giả thường và mẫu khó sinh ra từ cùng một người thật (`source_real_id`) bắt buộc phải nằm chung một tập. Tập Kiểm thử (`Test`) hoàn toàn là những người mới.
- **Chứng minh rõ ràng bằng con số**: Hàm `compare_baseline_vs_finetuned()` chấm điểm cả mô hình gốc lẫn mô hình đã huấn luyện lại trên cùng tập Test để chỉ ra mức tăng của `ROC-AUC`, mức tăng tỷ lệ bắt mẫu khó (`hard_sample_recall_gain`) và mức giảm tỷ lệ lọt lưới (`bypass_rate_reduction`).
