# PURPOSE.md — Đề Bài & Mục Tiêu Đề Tài 11 (Nhóm 4)

- **Motivation/Background**: Xác định rõ mục tiêu của đề tài theo định hướng Dataset-First: sản phẩm trọng tâm là bộ dataset khuôn mặt giả sinh bằng mô hình khuếch tán Stable Diffusion, kèm tập Mẫu Khó (Hard Samples) được kiểm chứng. Thử nghiệm huấn luyện downstream đóng vai trò là Proof of Utility (kiểm tra tính hữu dụng của dữ liệu).
- **Purpose**: Là tài liệu gốc quy định bài toán, luồng 3 bước thực hiện, cơ chế Proof of Utility và phạm vi nghiên cứu của **Nhóm 4**.
- **Overview Pipeline**: Bước 1 (Ảnh thật + Stable Diffusion -> Ảnh giả mới) -> Bước 2 (Chạy qua mô hình nền tảng / dinoViT gốc -> Bắt được hay Bỏ lọt? Nếu bỏ lọt thì lấy làm Mẫu khó) -> Bước 3 (Gộp dữ liệu gốc + ảnh sinh mới + mẫu khó -> Thử nghiệm Proof of Utility chứng minh dữ liệu thực sự có tác dụng).
- **Detailed Plan**: §1 Thông tin đề tài; §2 Luồng 3 bước theo yêu cầu giảng viên; §3 Cơ chế Proof of Utility; §4 Những phần làm và không làm.
- **References**: [OVERVIEW.md](OVERVIEW.md), [PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md](phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md), [PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md](phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-09T14:40:00+07:00

---

## 1. Thông Tin Đề Tài

- **Môn học**: Thị Giác Máy Tính (Computer Vision) — Trường Đại học Giao thông Vận tải TP.HCM (UTH)
- **Đề tài số**: Topic 11
- **Tên đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection* (Tạo bộ dữ liệu khuôn mặt giả bằng Stable Diffusion phục vụ phát hiện Deepfake)
- **Nhóm thực hiện**: **Nhóm 4**
- **Sản phẩm trọng tâm (Dataset-First)**: Bộ dữ liệu khuôn mặt giả chuẩn hóa có truy xuất nguồn gốc (provenance), kèm tập Mẫu Khó (Hard Samples) khách quan và báo cáo thống kê tự sinh.

---

## 2. Yêu Cầu Của Giảng Viên (Luồng 3 Bước Cốt Lõi)

Sau buổi báo cáo, giảng viên đã chốt lại hướng đi rất rõ ràng cho nhóm như sau:

```text
Bước 1: Ảnh gốc (img) + Stable Diffusion  ──►  Ảnh giả (Synthetic DF img)
                                                        │
                                                        ▼
Bước 2: Đưa qua mô hình gốc dinoViT (Benchmark)  ──►  Bắt được (Reject) hay Bỏ lọt (Bypass)?
                                                        └── Nếu Bypass (bỏ lọt) ──► Thu được MẪU KHÓ (Hard Sample)
                                                        │
                                                        ▼
Bước 3: Dữ liệu gốc + Ảnh sinh mới + Mẫu khó  ──►  Proof of Utility (Fine-Tune dinoViT)
                                                        └── Nếu bắt được mẫu khó ──► CHỨNG MINH ĐƯỢC TÁC DỤNG!
```

### Giải thích dễ hiểu từng bước:
1. **Bước 1 — Sinh dữ liệu ảnh giả (`Phase 1`)**:
   - Từ ảnh khuôn mặt thật và câu lệnh mô tả (Prompt), dùng **Stable Diffusion** tạo ra các bức ảnh khuôn mặt giả (`Synthetic DF img`) bằng 3 cách: tạo mới hoàn toàn (`Text-to-Image`), sửa vùng mắt/miệng (`Inpainting`), và biến đổi từ ảnh gốc (`Image-to-Image`).
   - Sau đó lọc bỏ những ảnh bị mờ hoặc méo mặt bằng Curation Gate để chỉ giữ lại các ảnh giả tự nhiên, rõ nét.

2. **Bước 2 — Kiểm tra bằng mô hình thị giác để tìm Mẫu khó (`Phase 2`)**:
   - Dùng mô hình **Baseline `dinoViT`** hoặc đa tạp ảnh thật (Real Manifold) để chấm điểm từng bức ảnh giả vừa tạo.
   - Có 2 trường hợp xảy ra:
     - **Bắt được (`Reject`)**: Mô hình nhận diện được ảnh giả $\rightarrow$ lưu làm ảnh giả thông thường (`Synthetic Data`).
     - **Bỏ lọt (`Bypass`)**: Ảnh giả có độ chân thực cao, đánh lừa được mô hình gốc $\rightarrow$ tách riêng thành **Mẫu khó (`Hard Sample`)**.

3. **Bước 3 — Thử nghiệm Proof of Utility chứng minh tác dụng (`Phase 3`)**:
   - Gộp cả 3 nguồn dữ liệu lại với nhau: **Dữ liệu gốc (`Data Baseline`) + Ảnh sinh mới (`Syn Data`) + Mẫu khó (`Hard Sample`)**.
   - Dùng tập gộp này để thực hiện thử nghiệm **Proof of Utility**: Huấn luyện tinh chỉnh (`Fine-Tune`) lại mô hình phát hiện.
   - **Cách chứng minh dữ liệu có tác dụng**: Kiểm tra lại trên tập kiểm thử độc lập (identity-disjoint); nếu mô hình sau khi học thêm dữ liệu mới **bắt được những mẫu khó trước đó từng bị bỏ lọt** và tăng điểm `ROC-AUC`, ta hoàn thành việc chứng minh bộ dữ liệu Stable Diffusion thực sự có ích.

---

## 3. Những Phần Nhóm Tập Trung Làm & Không Làm

| Trong phạm vi thực hiện (Có làm) | Ngoài phạm vi (Không làm) |
| :--- | :--- |
| • Tạo ảnh khuôn mặt giả 2D bằng 3 chế độ Stable Diffusion (`Text-to-Image`, `Inpainting`, `Img2Img`). | • Không làm giả giọng nói hay âm thanh (chỉ xử lý hình ảnh khuôn mặt). |
| • Lọc bỏ ảnh lỗi tự động (kiểm tra độ nét, bộ lọc phát hiện khuôn mặt và độ khớp ngữ nghĩa). | • Không huấn luyện lại mạng Stable Diffusion từ đầu (dùng mô hình có sẵn để tiết kiệm tài nguyên). |
| • Dùng mô hình gốc `dinoViT` / đa tạp thị giác để tách ảnh giả thành nhóm bị bắt (`Reject`) và nhóm vượt rào (`Bypass` / `Hard Sample`). | • Không ghép mặt người nổi tiếng hay cá nhân có thật ngoài đời (tuân thủ đạo đức nghiên cứu). |
| • Chia tập Huấn luyện / Kiểm thử không trùng người (`source_real_id`) và so sánh thực nghiệm Proof of Utility `D0`, `D1`, `D2`. | • Không tối ưu mô hình chạy trên điện thoại hay thiết bị nhúng. |
