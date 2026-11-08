# PROJECT_ROADMAP.md — Lộ Trình Thực Hiện Dự Án (M0 -> M7)

- **Motivation/Background**: Chia nhỏ toàn bộ công việc của Nhóm 4 thành các mốc rõ ràng từ lúc chốt lại đề tài đến lúc hoàn thành báo cáo và slide thuyết trình.
- **Purpose**: Theo dõi tiến độ từng bước thực hiện của đề tài.
- **Overview Pipeline**: M0 (Chốt đề bài mới) -> M1–M2 (Sinh & lọc ảnh Stable Diffusion) -> M3–M5 (Chạy `dinoViT` tìm mẫu khó & đóng gói dữ liệu) -> M6–M7 (Huấn luyện lại `D0/D1/D2` & viết báo cáo).
- **Detailed Plan**: §1 Bảng các mốc công việc `M0` đến `M7`; §2 Cách kiểm tra hoàn thành từng mốc.
- **References**: [ ../PURPOSE.md ](../PURPOSE.md), [ ../OVERVIEW.md ](../OVERVIEW.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:37:00+07:00

---

## 1. Bảng Lộ Trình 8 Mốc Công Việc (`M0` $\rightarrow$ `M7`)

| Mốc | Tên công việc | Nội dung thực hiện đơn giản | Kết quả đầu ra | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **`M0`** | Chốt đề bài & thiết kế | Cập nhật lại toàn bộ hướng đi theo góp ý của giảng viên (SD $\rightarrow$ `dinoViT` $\rightarrow$ `Hard Sample` $\rightarrow$ Fine-Tune). | Tài liệu thiết kế & bộ 11 Slide thuyết trình | `[x]` Hoàn tất |
| **`M1`** | Chuẩn bị ảnh gốc & khuôn mặt | Thu thập ảnh mặt thật, phát hiện 5 điểm mốc (2 mắt, mũi, 2 khóe miệng) và tạo vùng che mắt/miệng để sửa ảnh. | Tập ảnh mặt thật đã căn chỉnh ngay ngắn | `[x]` Hoàn tất |
| **`M2`** | Sinh ảnh giả bằng Stable Diffusion | Chạy 3 chế độ sinh ảnh giả (`sd_text2img`, `sd_inpaint`, `sd_img2img`) để tạo ra nhiều kiểu giả mạo khác nhau. | Tập ảnh giả thô ban đầu | `[x]` Hoàn tất |
| **`M3`** | Lọc bỏ ảnh lỗi tự động | Dùng bộ lọc 3 bước loại bỏ ảnh bị mờ nhòe, ảnh méo mặt và ảnh không khớp nội dung mô tả. | Tập ảnh giả đạt chuẩn (`curated_manifest.jsonl`) | `[x]` Hoàn tất |
| **`M4`** | Đưa qua `dinoViT` tìm Mẫu khó | Chạy mô hình gốc `dinoViT` trên các ảnh giả: ảnh nào bị bắt giữ là `Reject`, ảnh nào lọt lưới là `Bypass` (`Hard Sample`). | File danh sách Mẫu khó (`hard_samples_manifest.jsonl`) | `[x]` Hoàn tất |
| **`M5`** | Gộp dữ liệu & khóa danh tính | Gộp `Dữ liệu gốc + Ảnh sinh mới + Mẫu khó` và chia tập Train/Val/Test sao cho cùng một người không bị lẫn giữa Train và Test. | Bộ dữ liệu gộp chuẩn hóa | `[x]` Hoàn tất |
| **`M6`** | Huấn luyện lại `D0 / D1 / D2` | Huấn luyện lại `dinoViT` trên 3 chế độ `D0`, `D1`, `D2` và chấm xem mô hình có bắt lại được các Mẫu khó hay không. | Bảng điểm so sánh & mô hình tốt nhất (`best_model.pt`) | `[x]` Hoàn tất |
| **`M7`** | Hoàn thiện báo cáo & Slide | Chỉnh sửa toàn bộ tài liệu và bộ 11 Slide bằng ngôn ngữ dễ hiểu nhất, sẵn sàng bảo vệ. | Báo cáo hoàn chỉnh & bộ Slide 5 phút | `[x]` Hoàn tất |

---

## 2. Điều Kiện Kiểm Tra Hoàn Thành

1. Toàn bộ lệnh kiểm thử nhanh `make test-smoke` và kiểm thử đầy đủ `make test` chạy qua 100% không có lỗi.
2. Đường ống chạy tự động đủ cả 5 bước: `generate` (sinh ảnh) $\rightarrow$ `curate` (lọc ảnh lỗi) $\rightarrow$ `mine` (tìm mẫu khó qua `dinoViT`) $\rightarrow$ `train` (huấn luyện lại) $\rightarrow$ `eval` (chấm điểm so sánh).
3. Bộ 11 Slide HTML hiển thị rõ ràng, không tràn chữ, dùng từ ngữ đơn giản, dễ trình bày trong 5 phút.
