# PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md — Bước 1: Sinh Ảnh Giả Bằng Stable Diffusion & Lọc Ảnh Lỗi

- **Motivation/Background**: Thực hiện vế đầu tiên trong yêu cầu của giảng viên (`img + Stable Diffusion / Phase 1 sinh dữ liệu -> synthetic img (DF img)`), tạo ra nguồn ảnh khuôn mặt giả tự nhiên để thử thách mô hình phát hiện.
- **Purpose**: Giải thích đơn giản cách chuẩn bị ảnh mặt thật, 3 cách tạo ảnh giả bằng Stable Diffusion và 3 bước lọc bỏ ảnh lỗi.
- **Overview Pipeline**: Ảnh mặt thật -> Tìm 5 điểm mốc & tạo vùng che mắt/miệng -> Sinh ảnh bằng 3 chế độ Stable Diffusion -> Lọc bỏ ảnh mờ/lỗi qua 3 bước -> Xuất tập ảnh giả đạt chuẩn.
- **Detailed Plan**: §1 Chuẩn bị khuôn mặt đầu vào; §2 Ba cách tạo ảnh giả bằng Stable Diffusion; §3 Ba bước lọc bỏ ảnh lỗi tự động.
- **References**: [ ../PURPOSE.md ](../PURPOSE.md), [ PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md ](PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:39:00+07:00

---

## 1. Chuẩn Bị Ảnh Khuôn Mặt Đầu Vào

Trước khi đưa vào Stable Diffusion, mỗi bức ảnh người thật được xử lý qua 3 thao tác cơ bản:
1. **Tìm khung mặt (Bounding Box)**: Xác định đúng vị trí khuôn mặt trong ảnh.
2. **Tìm 5 điểm mốc chính trên mặt**: Xác định mắt trái, mắt phải, đầu mũi, khóe miệng trái và khóe miệng phải để xoay khuôn mặt cho ngay ngắn.
3. **Tạo mặt nạ vùng mắt / miệng (Binary Mask)**: Tô đánh dấu vùng mắt hoặc miệng để dùng riêng cho chế độ sửa một phần khuôn mặt (`Inpainting`).

---

## 2. Ba Cách Tạo Ảnh Giả Bằng Stable Diffusion

Hệ thống kết hợp 3 cách tạo ảnh khác nhau để bộ dữ liệu vừa có ảnh giả toàn mặt, vừa có ảnh chỉ sửa một chi tiết nhỏ:

| Cách tạo ảnh | Mã nhãn trong code | Đầu vào | Cách hoạt động dễ hiểu |
| :--- | :--- | :--- | :--- |
| **1. Tạo mặt mới từ chữ (`Text-to-Image`)** | `sd_text2img` | Câu mô tả (Prompt) | Từ câu chữ mô tả (độ tuổi, ánh sáng, biểu cảm), Stable Diffusion tự vẽ ra một khuôn mặt người mới hoàn toàn. |
| **2. Sửa vùng mắt / miệng (`Facial Inpainting`)** | `sd_inpaint` | Ảnh thật + Mặt nạ che + Prompt | Giữ nguyên khuôn mặt thật, chỉ xóa vùng mắt hoặc miệng rồi nhờ Stable Diffusion vẽ lại biểu cảm mới vào đúng chỗ đó. |
| **3. Biến đổi từ ảnh thật (`Image-to-Image`)** | `sd_img2img` | Ảnh thật + 5 điểm mốc + Prompt | Giữ nguyên tư thế đầu và vị trí mắt-mũi-miệng của ảnh thật, nhưng vẽ lại kết cấu da và ánh sáng mới. |

---

## 3. Ba Bước Lọc Bỏ Ảnh Lỗi Tự Động (`ImageCurationGate`)

Khi sinh ảnh hàng loạt bằng AI, đôi khi sẽ có ảnh bị nhòe hoặc méo mặt. Vì vậy, mọi bức ảnh sinh ra phải đi qua 3 bước kiểm tra tự động:

1. **Bước lọc 1 — Kiểm tra độ nét**: Tính độ sắc nét của ảnh; nếu ảnh bị mờ nhòe thì loại bỏ ngay.
2. **Bước lọc 2 — Kiểm tra cấu trúc mặt**: Kiểm tra xem trên ảnh mới tạo có tìm lại được đủ 5 điểm mốc mắt-mũi-miệng hợp lý hay không; nếu mặt bị biến dạng thì loại bỏ.
3. **Bước lọc 3 — Kiểm tra độ khớp câu lệnh**: Kiểm tra xem bức ảnh vẽ ra có đúng với câu mô tả ban đầu hay không.

Chỉ những bức ảnh **vượt qua cả 3 bước lọc** mới được lưu vào tập ảnh giả đạt chuẩn (`curated_manifest.jsonl`) để chuyển sang Bước 2.
