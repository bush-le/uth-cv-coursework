# EXPERIMENT_PROTOCOL.md — Quy Trình Thử Nghiệm & Chứng Minh Tác Dụng

- **Motivation/Background**: Đặt ra cách so sánh rõ ràng, công bằng để chứng minh rằng việc thêm dữ liệu tổng hợp Stable Diffusion và các Mẫu khó (`Hard Sample`) thực sự giúp mô hình `dinoViT` phát hiện ảnh giả tốt hơn.
- **Purpose**: Quy định 3 chế độ huấn luyện (`D0`, `D1`, `D2`) và các con số dùng để chấm điểm mô hình.
- **Overview Pipeline**: Cố định tập kiểm thử độc lập -> Huấn luyện `dinoViT` trên 3 chế độ (`D0`, `D1`, `D2`) -> Chấm điểm `ROC-AUC`, `Hard-Sample Recall`, `Bypass Rate` -> Kết luận tác dụng của dữ liệu.
- **Detailed Plan**: §1 Ba chế độ huấn luyện đối chứng (`D0`, `D1`, `D2`); §2 Bốn chỉ số chấm điểm dễ hiểu; §3 Khi nào thì kết luận dữ liệu có tác dụng?
- **References**: [ ../PURPOSE.md ](../PURPOSE.md), [ ../coursework/TAXONOMY_AND_DATA_SPEC.md ](../coursework/TAXONOMY_AND_DATA_SPEC.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:38:00+07:00

---

## 1. Ba Chế Độ Huấn Luyện Để So Sánh (`D0`, `D1`, `D2`)

Cả 3 chế độ đều dùng **cùng một mô hình `dinoViT`**, **cùng số vòng lặp học (epochs)** và được chấm điểm trên **cùng một tập kiểm thử độc lập (`Held-Out Test Set`)**:

| Chế độ | Tập dữ liệu dùng để dạy `dinoViT` | Ý nghĩa thực tế |
| :--- | :--- | :--- |
| **Chế độ `D0`** | Chỉ dùng **Dữ liệu gốc (`Data Baseline`)** | Đây là mô hình gốc ban đầu trước khi có dữ liệu Stable Diffusion của nhóm. |
| **Chế độ `D1`** | **Dữ liệu gốc + Ảnh sinh mới thông thường (`Syn Data`)** | Mô hình được học thêm ảnh giả Stable Diffusion nhưng chưa tập trung vào các mẫu khó. |
| **Chế độ `D2`** | **Dữ liệu gốc + Ảnh sinh mới + Mẫu khó (`Hard Sample`)** | Mô hình được học đầy đủ cả ảnh sinh mới lẫn các **Mẫu khó từng đánh lừa `D0`** (kèm nhân bản và trọng số phạt cao). |

---

## 2. Bốn Chỉ Số Chấm Điểm Dễ Hiểu

Thay vì chỉ nhìn độ chính xác chung chung, nhóm chấm điểm bằng 4 chỉ số trực tiếp trả lời câu hỏi của giảng viên:

1. **Tỷ lệ lọt lưới (`Bypass Rate` — càng thấp càng tốt)**:
   - Trong số các bức ảnh giả, có bao nhiêu phần trăm đánh lừa được mô hình (được chấm điểm giả $< 0.5$)?
   - Trước khi học lại (`D0`), tỷ lệ lọt lưới cao; sau khi học lại (`D2`), tỷ lệ lọt lưới phải giảm mạnh.

2. **Tỷ lệ bắt được Mẫu khó (`Hard-Sample Recall` — càng cao càng tốt)**:
   - Trong số các **Mẫu khó (`Hard Sample`)** từng vượt qua mô hình gốc, nay mô hình mới đã bắt lại được bao nhiêu phần trăm?
   - Đây là con số trực tiếp nhất chứng minh mô hình đã vá được lỗ hổng của mình.

3. **Điểm phân biệt tổng thể (`ROC-AUC` — từ 0 đến 1, càng gần 1 càng tốt)**:
   - Đo khả năng phân biệt rạch ròi giữa ảnh thật và ảnh giả ở mọi mức ngưỡng.

4. **Độ chính xác và điểm cân bằng (`Accuracy` & `F1-Score`)**:
   - Đảm bảo khi mô hình bắt thêm được ảnh giả khó thì vẫn không đoán nhầm ảnh người thật thành ảnh giả.

---

## 3. Khi Nào Kết Luận Bộ Dữ Liệu Có Tác Dụng?

Bộ dữ liệu tổng hợp Stable Diffusion và quy trình khai phá Mẫu khó của **Nhóm 4** được chứng minh là **có tác dụng rõ rệt** khi kết quả trên tập Kiểm thử thỏa mãn:
1. **Bắt lại được Mẫu khó**: Điểm `Hard-Sample Recall` của chế độ `D2` tăng vọt so với mô hình gốc `D0`.
2. **Giảm mạnh tỷ lệ lọt lưới**: `Bypass Rate` giảm rõ rệt từ `D0` $\rightarrow$ `D1` $\rightarrow$ `D2`.
3. **Nâng cao năng lực tổng thể**: Đường cong `ROC-AUC` của `D2` nằm cao hơn `D1` và `D0`, đồng thời không làm giảm độ chính xác trên các ảnh thật ban đầu.
