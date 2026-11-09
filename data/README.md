# DATA.md — Data Layer Architecture & Dataset Governance (`data/`)

- **Motivation/Background**: Thiết lập cấu trúc dữ liệu 5 tầng chuẩn hóa bám sát 100% vòng đời của Đề tài 11 (Topic 11: Sinh dữ liệu khuôn mặt giả bằng Stable Diffusion & Khai thác mẫu khó phục vụ phát hiện Deepfake), đảm bảo tính bất biến của ảnh thật, phân loại rõ 3 cơ chế sinh ảnh, bảo tồn Mẫu Khó và cung cấp 3 bộ dữ liệu đối chứng $D_0, D_1, D_2$.
- **Purpose**: Định nghĩa chi tiết các tầng dữ liệu, tiêu chuẩn manifest, quản lý Mẫu Khó (`Hard Samples`) và các bộ benchmark splits.
- **Overview Pipeline**: `raw` (FFHQ + CelebA-HQ) -> `synthetic` (t2i, inpaint, img2img + Curation Gate) -> `hard_samples` (bypassed deepfakes) -> `benchmarks` (D0, D1, D2 splits) -> `samples` (CI/CD quick tests).
- **Detailed Plan**: §1 Cấu trúc 5 tầng Topic 11; §2 Chi tiết các phân vùng; §3 Tiêu chuẩn Manifest JSONL; §4 Dữ liệu đối chứng $D_0, D_1, D_2$; §5 Quy tắc quản lý dữ liệu.
- **References**: [../docs/PURPOSE.md](../docs/PURPOSE.md), [../docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md](../docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md), [../docs/phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md](../docs/phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md).
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-09T12:18:00+07:00

---

## 1. Cấu Trúc Dữ Liệu Chuẩn Hóa Theo Đề Tài 11

```text
data/
├── raw/                              # [Input] Nguồn ảnh khuôn mặt thật nguyên bản (Bất biến)
│   ├── faces/
│   │   ├── ffhq/                     # 2.000 ảnh FFHQ (00000.png -> 01999.png)
│   │   └── celeba_hq/                # 2.000 ảnh CelebA-HQ (0.jpg -> 1999.jpg)
│   ├── metadata/                     # Nhãn nhân khẩu học gốc (tuổi, giới tính, kính, pose)
│   │   ├── ffhq_aging_labels.csv
│   │   ├── identity_CelebA.txt
│   │   ├── CelebAMask-HQ-attribute-anno.txt
│   │   └── ffhq-dataset-v2.json
│   └── manifest.jsonl                # 4.000 bản ghi ảnh thật chuẩn hóa có demographics
│
├── synthetic/                        # [Phase 1] Bộ dữ liệu mặt giả sinh bởi Stable Diffusion v1.5
│   ├── text2image/                   # Chế độ 1: Sinh mới chân dung từ prompt nhân khẩu học
│   ├── inpainting/                   # Chế độ 2: Sửa vùng mắt/miệng/vật che (occlusion)
│   ├── img2img/                      # Chế độ 3: Biến đổi kết cấu da & biểu cảm từ ảnh thật
│   ├── fake_manifest.jsonl           # Toàn bộ ảnh giả đã sinh
│   └── curated_manifest.jsonl        # Ảnh giả ĐẠT CHUẨN sau khi qua ImageCurationGate
│
├── hard_samples/                     # [Phase 2] Mẫu Khó vượt qua được Baseline dinoViT
│   ├── images/                       # Các ảnh giả tinh vi đánh lừa được dinoViT (P(fake) < 0.5)
│   ├── audit_reports/                # Thống kê phân tích: nhóm tuổi/phụ kiện nào dễ lừa mô hình nhất
│   └── hard_samples_manifest.jsonl   # Danh mục mẫu khó kèm điểm số P(fake) và is_hard_sample=True
│
├── benchmarks/                       # [Phase 3] 3 Bộ Dữ Liệu Đối Chứng Cho Huấn Luyện & Đánh Giá
│   ├── D0_baseline/                  # D0: Real Faces + Baseline Fake (train / val / test)
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── D1_augmented/                 # D1: D0 + Toàn bộ ảnh giả Curated SD (train / val / test)
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── D2_hard_boosted/              # D2: D0 + Curated SD + Mẫu Khó (phạt loss x2, oversample x2)
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── samples/                          # Dữ liệu kiểm thử siêu nhẹ (< 1.5MB) cho CI/CD và Smoke Test (<3s)
│   ├── real/                         # 4 ảnh thật mẫu (real_01.png -> real_04.png)
│   ├── synthetic/                    # 4 ảnh giả mẫu (synth_01.png -> synth_04.png)
│   ├── hard_samples/                 # 2 mẫu khó mẫu (hard_01.png -> hard_02.png)
│   └── sample_manifest.jsonl         # Danh mục mẫu thử nghiệm
│
└── README.md                         # Tài liệu kiến trúc dữ liệu này
```

---

## 2. Tiêu Chuẩn Bản Ghi JSONL (`ImageMetadata`)

Mọi tệp manifest trong `data/` đều tuân thủ cấu trúc hợp đồng chuẩn:

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
  "baseline_score": 0.5098,
  "data_source": "hard_sample",
  "demographics": {
    "dataset": "ffhq",
    "age_group": "0-2",
    "gender": "male",
    "glasses": "None",
    "yaw": -9.359,
    "pitch": 4.644,
    "roll": 2.180
  }
}
```

---

## 3. Ba Bộ Dữ Liệu Thực Nghiệm Đối Chứng ($D_0, D_1, D_2$)

Nhóm 4 xây dựng sẵn 3 bộ phân chia (`benchmarks/`) phục vụ kiểm chứng thực nghiệm độc lập:
1. **$D_0$ (Baseline Benchmark Split)**:
   - Bao gồm toàn bộ ảnh thật và ảnh giả chuẩn ban đầu.
   - Dùng để đo đường cơ sở ban đầu của mô hình.
2. **$D_1$ (Augmented Benchmark Split)**:
   - $D_0$ + Bổ sung toàn bộ ảnh giả sinh bởi Stable Diffusion đã qua lọc Curation Gate.
   - Đo lường khả năng mở rộng dữ liệu thông thường.
3. **$D_2$ (Hard-Boosted Benchmark Split)**:
   - $D_0$ + Ảnh giả SD + Mẫu Khó (`Hard Samples`).
   - Mẫu khó được gán cờ `is_hard_sample=True`, kích hoạt trọng số phạt `hard_sample_loss_weight=2.0` và nhân bản mẫu trong tập train (`hard_sample_oversample_factor=2`).

Tất cả các tập phân chia đều tuân thủ nghiêm ngặt **`identity_disjoint=True`**: mọi ảnh thật và ảnh giả sinh ra từ cùng một người thật (`source_real_id`) đều nằm trọn vẹn trong cùng 1 phân vùng (Train, Val hoặc Test), đảm bảo không xảy ra rò rỉ danh tính.

---

## 4. Lệnh Tự Động Hóa Dữ Liệu

```bash
# Tổ chức lại toàn bộ dữ liệu vào cấu trúc 5 tầng chuẩn Topic 11
python3 -m scripts.organize_topic_dataset

# Xuất lưới ảnh đối chứng trực quan
python3 -m scripts.build_visual_gallery

# Chạy kiểm thử nhanh trên tập samples (< 3 giây)
make test-smoke
```
