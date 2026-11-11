# TAXONOMY_AND_DATA_SPEC.md — Quy Định Về Dữ Liệu & Nhãn (Nhóm 4)

- **Motivation/Background**: Quy định rõ các nhóm dữ liệu trong dự án (`Dữ liệu gốc`, `Ảnh sinh mới thông thường`, `Mẫu khó đối kháng`) và cấu trúc metadata chuẩn hóa đi kèm mỗi bức ảnh.
- **Purpose**: Làm chuẩn chung cho mã nguồn quản lý dữ liệu (`src/data/`), hợp đồng lưu trữ manifest và quy tắc chia tập benchmark không trùng danh tính.
- **Overview Pipeline**: 3 nguồn dữ liệu (`baseline`, `synthetic`, `hard_sample`) -> 3 chế độ tạo ảnh SD (`sd_text2img`, `sd_inpaint`, `sd_img2img`) -> Hợp đồng `ImageMetadata` -> Quy tắc chia tập `identity-disjoint`.
- **Detailed Plan**: §1 Các nhóm dữ liệu chính; §2 Ba cách tạo ảnh giả bằng Stable Diffusion; §3 Cấu trúc bản ghi `ImageMetadata`; §4 Quy tắc khóa danh tính `identity-disjoint`.
- **References**: [../PURPOSE.md](../PURPOSE.md), [../OVERVIEW.md](../OVERVIEW.md), `configs/dataset_factory.yaml`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-09T15:15:00+07:00

---

## 1. Các Nhóm Dữ Liệu Chính Trong Hệ Thống

Mỗi bức ảnh trong hệ thống thuộc đúng 1 trong 3 nhóm nguồn gốc (`data_source`):

| STT | Tên nhóm (`data_source`) | Nhãn (`label`) | Cờ mẫu khó (`is_hard_sample`) | Giải thích khoa học |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **`baseline`** (Ảnh thật tham chiếu) | `0` (Thật) | `False` | Tập ảnh khuôn mặt thật chuẩn hóa từ FFHQ và CelebA-HQ. |
| **02** | **`synthetic`** (Ảnh giả thông thường) | `1` (Giả) | `False` | Ảnh giả tạo bởi Stable Diffusion đã qua bộ lọc Curation Gate, nằm ngoài ranh giới Mẫu Khó. |
| **03** | **`hard_sample`** (Mẫu khó đối kháng) | `1` (Giả) | `True` | Ảnh giả có biểu diễn sâu nằm sát đa tạp ảnh thật (`Real-Manifold Proximity` $\ge \tau$), có khả năng đánh lừa bộ phân loại. |

---

## 2. Ba Chế Độ Tạo Ảnh Giả Bằng Stable Diffusion (`manipulation_type`)

1. **`sd_text2img` (Chân dung tạo từ văn bản)**: Nhập câu mô tả nhân khẩu học để Stable Diffusion tự vẽ ra một diện mạo hoàn toàn mới.
2. **`sd_inpaint` (Chỉnh sửa cục bộ khuôn mặt)**: Giữ nguyên nền và viền đầu, tái tạo vùng mắt/mũi/miệng hoặc vật che với đường biên hòa trộn mềm.
3. **`sd_img2img` (Biến đổi diện mạo từ ảnh thật)**: Giữ nguyên cấu trúc khuôn mặt gốc nhưng biến đổi làn da, ánh sáng, độ tuổi hoặc biểu cảm.

---

## 3. Cấu Trúc Bản Ghi Hợp Đồng (`ImageMetadata`)

Mỗi bức ảnh được định danh bằng một dòng JSON trong tệp `.jsonl` với cấu trúc chuẩn:

```json
{
  "image_id": "fake_ffhq_00001_0",
  "label": 1,
  "manipulation_type": "sd_inpaint",
  "prompt": "altered facial expression and eyes, wearing eyeglasses with realistic frame reflections",
  "source_real_id": "ffhq_00001",
  "file_path": "data/synthetic/inpainting/fake_ffhq_00001_0.png",
  "bbox": null,
  "is_hard_sample": true,
  "baseline_score": 0.884,
  "data_source": "hard_sample",
  "demographics": {
    "age_group": "adult",
    "gender": "female",
    "glasses": "Normal"
  },
  "generation": {
    "model_id": "<generation.inpaint_model_id>",
    "mode": "sd_inpaint",
    "steps": 20,
    "guidance_scale": 7.5,
    "seed": 91823712
  },
  "scores": {
    "sharpness": 142.1,
    "real_manifold": 0.884,
    "source_similarity": 0.945
  }
}
```

---

## 4. Quy Tắc Khóa Danh Tính (`identity_disjoint`)

- Từ ảnh thật của một cá nhân (`source_real_id = #ID`), mọi biến thể giả mạo sinh ra từ người này (inpaint, img2img) đều chia sẻ chung một khóa nhóm danh tính (`group_key`).
- Thuật toán `assign_identity_splits` phân phối toàn bộ nhóm vào cùng một tập phân vùng (`train`, `val`, hoặc `test`).
- Nhờ vậy, tập kiểm thử độc lập (`Held-Out Test Set`) luôn gồm những danh tính hoàn toàn mới mà các mô hình downstream chưa từng tiếp cận khi huấn luyện.
