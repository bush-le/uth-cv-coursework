# DATA.md — Data Layer Architecture & Dataset Governance (`data/`)

- **Motivation/Background**: Thiết lập cấu trúc dữ liệu chuẩn hóa bám sát 100% vòng đời của Đề tài 11 (Topic 11: Sinh dữ liệu khuôn mặt giả bằng Stable Diffusion & Khai thác mẫu khó phục vụ phát hiện Deepfake), đảm bảo tính bất biến của ảnh thật, phân loại rõ 3 cơ chế sinh ảnh, bảo tồn Mẫu Khó và cung cấp 3 bộ dữ liệu đối chứng $D_0, D_1, D_2$.
- **Purpose**: Định nghĩa chi tiết các tầng dữ liệu, tiêu chuẩn manifest, quản lý Mẫu Khó (`Hard Samples`) và các bộ benchmark splits.
- **Overview Pipeline**: `raw/` (Ảnh thật tham chiếu) -> `synthetic/` (t2i, inpaint, img2img + Curation Gate) -> `hard_samples/` (Mẫu Khó sát đa tạp thật) -> `benchmarks/` ($D_0, D_1, D_2$ identity-disjoint splits) -> `paths.stats_file`.
- **Detailed Plan**: §1 Cấu trúc phân tầng; §2 Tiêu chuẩn Manifest JSONL & Provenance; §3 Ba bộ dữ liệu benchmark đối chứng ($D_0, D_1, D_2$); §4 Quy tắc quản trị dữ liệu.
- **References**: [../docs/PURPOSE.md](../docs/PURPOSE.md), [../docs/OVERVIEW.md](../docs/OVERVIEW.md), `../configs/dataset_factory.yaml`, `../.agents/rules/ZERO_HARDCODE_RULE.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-09T15:15:00+07:00

---

## 1. Cấu Trúc Dữ Liệu Chuẩn Hóa Theo Đề Tài 11

```text
data/
├── raw/                              # [Input] Nguồn ảnh khuôn mặt thật tham chiếu (Bất biến)
│   ├── faces/
│   │   ├── ffhq/                     # Ảnh khuôn mặt FFHQ
│   │   └── celeba_hq/                # Ảnh khuôn mặt CelebA-HQ
│   ├── metadata/                     # Thuộc tính nhân khẩu học gốc (tuổi, giới tính, kính)
│   └── manifest.jsonl                # Danh mục ảnh thật chuẩn hóa kèm demographics
│
├── synthetic/                        # [Phase 1] Bộ dữ liệu mặt giả sinh bởi Stable Diffusion
│   ├── text2image/                   # Chế độ 1: Sinh mới chân dung từ prompt nhân khẩu học
│   ├── inpainting/                   # Chế độ 2: Sửa vùng mặt và vật che (occlusion)
│   ├── img2img/                      # Chế độ 3: Biến đổi kết cấu da & biểu cảm từ ảnh thật
│   ├── fake_manifest.jsonl           # Toàn bộ ảnh giả đã sinh
│   ├── curation_report.jsonl         # Báo cáo kiểm định chất lượng từng bức ảnh
│   └── curated_manifest.jsonl        # Ảnh giả ĐẠT CHUẨN sau khi qua ImageCurationGate
│
├── hard_samples/                     # [Phase 2] Mẫu Khó Đối Kháng (Sát đa tạp ảnh thật)
│   ├── images/                       # Các ảnh giả tinh vi lọt qua ngưỡng quyết định Tau (τ)
│   ├── audit_reports/                # Thống kê phân tích: tỷ lệ mẫu khó theo chế độ và nhóm nhân khẩu học
│   └── hard_samples_manifest.jsonl   # Danh mục mẫu khó kèm điểm số manifold và is_hard_sample=True
│
├── benchmarks/                       # [Phase 3] 3 Bộ Dữ Liệu Đối Chứng Cho Huấn Luyện & Đánh Giá
│   ├── D0_baseline/                  # D0: Mốc cơ sở (train / val / test)
│   ├── D1_augmented/                 # D1: Tăng cường diện rộng (train / val / test)
│   ├── D2_hard_boosted/              # D2: Tăng cường trọng tâm Mẫu Khó (oversample x2)
│   └── dataset_stats.json            # paths.stats_file: Tổng hợp số liệu tự sinh toàn diện
│
├── samples/                          # Dữ liệu kiểm thử siêu nhẹ cho CI/CD và Smoke Test
│   ├── real/                         # Ảnh thật mẫu
│   ├── synthetic/                    # Ảnh giả mẫu
│   └── hard_samples/                 # Mẫu khó mẫu
│
└── README.md                         # Tài liệu kiến trúc dữ liệu này
```

---

## 2. Tiêu Chuẩn Bản Ghi JSONL (`ImageMetadata`)

Mọi tệp manifest trong `data/` đều tuân thủ cấu trúc hợp đồng chuẩn, lưu trữ nguồn gốc sinh (`generation`) và điểm chất lượng (`scores`):

```json
{
  "image_id": "fake_ffhq_00000_0",
  "label": 1,
  "manipulation_type": "sd_img2img",
  "prompt": "subtle facial manipulation, realistic skin texture, young child portrait, soft natural skin",
  "source_real_id": "ffhq_00000",
  "file_path": "data/synthetic/img2img/fake_ffhq_00000_0.png",
  "bbox": null,
  "is_hard_sample": true,
  "baseline_score": 0.865,
  "data_source": "hard_sample",
  "demographics": {
    "dataset": "ffhq",
    "age_group": "young",
    "gender": "male",
    "glasses": "None"
  },
  "generation": {
    "model_id": "<generation.model_id>",
    "mode": "sd_img2img",
    "steps": 20,
    "guidance_scale": 7.5,
    "img2img_strength": 0.50,
    "seed": 28419512
  },
  "scores": {
    "sharpness": 128.4,
    "real_manifold": 0.865,
    "source_similarity": 0.912
  }
}
```

---

## 3. Ba Bộ Dữ Liệu Thực Nghiệm Đối Chứng ($D_0, D_1, D_2$)

Nhóm xây dựng sẵn 3 bộ phân chia (`benchmarks/`) phục vụ kiểm chứng thực nghiệm độc lập:
1. **$D_0$ (Baseline Benchmark Split)**:
   - Đo đường cơ sở ban đầu của mô hình.
2. **$D_1$ (Augmented Benchmark Split)**:
   - Kiểm tra hiệu quả khi tăng cường toàn bộ ảnh mặt giả Stable Diffusion đã qua lọc curation.
3. **$D_2$ (Hard Boosted Benchmark Split)**:
   - Chứng minh đóng góp chính của đề tài: Tập trung vào các Mẫu Khó (`is_hard_sample=True`) giúp tối ưu khả năng phát hiện Deepfake.

---

## 4. Quy Tắc Quản Trị Dữ Liệu Bắt Buộc

1. **Khóa chặt ảnh thật (`raw/`)**: Thư mục `data/raw/` là bất biến, chỉ đọc, không bao giờ được ghi đè hay chỉnh sửa bởi bất kỳ pipeline nào.
2. **Bảo vệ danh tính tuyệt đối (`identity_disjoint`)**: Ảnh thật và ảnh giả phái sinh từ cùng một người (`source_real_id`) bắt buộc nằm chung trong một split (cùng Train hoặc cùng Test), triệt tiêu rò rỉ danh tính.
3. **Thống kê tự sinh**: Mọi số liệu chính xác về số lượng mẫu, tỷ lệ thông qua bộ lọc, và tỷ lệ mẫu khó được lưu tự động tại `paths.stats_file`, xem chi tiết tại `configs/dataset_factory.yaml`.
