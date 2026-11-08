# TAXONOMY_AND_DATA_SPEC.md — Quy Định Về Dữ Liệu & Nhãn (Nhóm 4)

- **Motivation/Background**: Quy định rõ 3 nhóm dữ liệu trong dự án (`Dữ liệu gốc`, `Ảnh sinh mới thông thường`, `Mẫu khó vượt rào`) và các thông tin đi kèm mỗi bức ảnh để quản lý dữ liệu sạch sẽ, không bị lẫn lộn.
- **Purpose**: Làm chuẩn chung cho code đọc dữ liệu (`src/data/`) và cho việc chia tập huấn luyện / kiểm thử.
- **Overview Pipeline**: 3 nguồn dữ liệu (`baseline`, `synthetic`, `hard_sample`) -> 3 chế độ tạo ảnh SD (`sd_text2img`, `sd_inpaint`, `sd_img2img`) -> 6 thông tin lưu kèm mỗi bức ảnh -> Quy tắc chia tập không trùng người.
- **Detailed Plan**: §1 Ba nhóm dữ liệu chính; §2 Ba cách tạo ảnh giả bằng Stable Diffusion; §3 Sáu thông tin lưu kèm mỗi mẫu ảnh; §4 Quy tắc khóa danh tính không trùng người.
- **References**: [ ../PURPOSE.md ](../PURPOSE.md), [ ../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md ](../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:38:00+07:00

---

## 1. Ba Nhóm Dữ Liệu Chính Trong Hệ Thống

Mỗi bức ảnh trong hệ thống thuộc đúng 1 trong 3 nhóm nguồn gốc (`data_source`):

| STT | Tên nhóm (`data_source`) | Nhãn (`label`) | Cờ mẫu khó (`is_hard_sample`) | Giải thích đơn giản |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **`baseline`** (Dữ liệu gốc) | `0` (Thật) hoặc `1` (Giả) | `False` | Tập ảnh mặt thật và ảnh giả có sẵn ban đầu dùng để dạy và đo mô hình gốc. |
| **02** | **`synthetic`** (Ảnh giả bị bắt) | `1` (Giả) | `False` | Ảnh giả tạo bởi Stable Diffusion đã bị mô hình gốc `dinoViT` phát hiện ra (**Reject**, điểm giả $\ge 0.5$). |
| **03** | **`hard_sample`** (Mẫu khó vượt rào) | `1` (Giả) | `True` | Ảnh giả tạo bởi Stable Diffusion nhưng **đánh lừa được `dinoViT` gốc** (**Bypass**, điểm giả $< 0.5$). |

---

## 2. Ba Chế Độ Tạo Ảnh Giả Bằng Stable Diffusion (`manipulation_type`)

1. **`sd_text2img` (Tạo mặt mới từ chữ)**: Nhập câu mô tả khuôn mặt để Stable Diffusion tự vẽ ra một khuôn mặt hoàn toàn mới.
2. **`sd_inpaint` (Sửa vùng mắt / mũi / miệng)**: Lấy ảnh người thật, che vùng mắt hoặc miệng lại rồi cho Stable Diffusion vẽ lại vùng đó (nền và khung mặt giữ nguyên nên rất khó phát hiện).
3. **`sd_img2img` (Biến đổi từ ảnh thật)**: Lấy ảnh người thật làm khung, giữ nguyên vị trí mắt-mũi-miệng nhưng dùng Stable Diffusion đổi ánh sáng, làn da hoặc biểu cảm.

---

## 3. Sáu Thông Tin Lưu Kèm Mỗi Bức Ảnh (Thẻ Căn Cước Mẫu)

Mỗi bức ảnh được lưu một dòng trong file `.jsonl` (`ImageMetadata`) với 6 thông tin cốt lõi:

1. **`image_id`**: Mã số riêng của bức ảnh (ví dụ: `syn_0042`).
2. **`label` & `manipulation_type`**: Nhãn thật/giả (`0` hoặc `1`) và cách tạo ảnh (`real`, `sd_text2img`, `sd_inpaint`, `sd_img2img`).
3. **`data_source`**: Nguồn dữ liệu (`baseline`, `synthetic`, hoặc `hard_sample`).
4. **`source_real_id`**: Mã số của người thật gốc dùng để tạo ra bức ảnh đó (dùng để khóa không cho cùng một người vừa ở tập học vừa ở tập thi).
5. **`baseline_score`**: Điểm xác suất giả do mô hình gốc `dinoViT` chấm ở Bước 2.
6. **`is_hard_sample`**: Đánh dấu `True` nếu bức ảnh này đánh lừa được `dinoViT` gốc (`Bypass`), ngược lại là `False`.

---

## 4. Quy Tắc Chia Tập Không Trùng Người (Chống Học Vẹt)

- Giả sử từ bức ảnh thật của người **#A1**, ta tạo ra 3 bức ảnh giả (cả ảnh bị bắt lẫn mẫu khó).
- Hệ thống bắt buộc **gom tất cả ảnh có chung `source_real_id = #A1` vào cùng một ngăn** (hoặc cùng ở tập Huấn luyện, hoặc cùng ở tập Kiểm thử).
- Nhờ vậy, tập Kiểm thử (`Held-Out Test`) luôn gồm những khuôn mặt hoàn toàn mới mà mô hình chưa từng nhìn thấy lúc học.
