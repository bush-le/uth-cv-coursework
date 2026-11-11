# UTH Computer Vision Coursework — Topic 11 (Nhóm 4)
**Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection**

- **Motivation/Background**: Dự án môn học Thị giác Máy tính (Topic 11) của Nhóm 4, thực hiện theo định hướng **Dataset-First**: Xây dựng bộ dữ liệu khuôn mặt giả chất lượng cao bằng Stable Diffusion -> Khai thác Mẫu Khó (`Hard Samples`) thông qua khoảng cách đa tạp ảnh thật (`Real-Manifold Proximity`) -> Đóng gói bộ dữ liệu benchmark đối chứng ($D_0, D_1, D_2$) -> Thực hiện thực nghiệm Proof of Utility.
- **Purpose**: Cung cấp tài liệu tổng quan, kiến trúc dự án, hợp đồng dữ liệu và hướng dẫn chạy thực nghiệm cho giảng viên và hội đồng nghiệm thu.
- **Overview Pipeline**: `data/raw/` -> `src.pipelines.generate_deepfake_images` -> `src.pipelines.curate_image_dataset` -> `src.pipelines.mine_hard_samples` -> `scripts.organize_topic_dataset` -> `data/benchmarks/`.
- **Detailed Plan**: §1 Ý Tưởng Cốt Lõi (Dataset-First Framing); §2 Cấu Trúc Dự Án; §3 Quy Trình Chạy Pipeline; §4 Bộ Kiểm Thử Chất Lượng; §5 Tài Liệu Tham Khảo.
- **References**: [docs/PURPOSE.md](docs/PURPOSE.md), [docs/OVERVIEW.md](docs/OVERVIEW.md), `configs/dataset_factory.yaml`, `.agents/rules/ZERO_HARDCODE_RULE.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-09T15:15:00+07:00

---

## 1. Ý Tưởng Cốt Lõi & Định Hướng Nghiên Cứu (Dataset-First)

### 1.1. Bối cảnh bài toán: Deepfake thế hệ mới (Diffusion-Based Synthesis)
- **Deepfake Cắt Dán (Face-Swap)**: Tạo vết ghép quanh biên mặt hoặc sai lệch ánh sáng cục bộ. Các bộ phân loại truyền thống dễ dàng phát hiện dựa trên artifacts biên.
- **Deepfake Khuếch Tán (Stable Diffusion)**: Khử nhiễu trực tiếp trong không gian tiềm ẩn (`Latent Space`) tạo nên cấu trúc da, ánh sáng và biên giới mượt mà, gây tê liệt nhiều mô hình phát hiện Deepfake hiện hành.

### 1.2. Ba phương thức sinh dữ liệu bổ trợ (Multimodal Synthesis)
1. **`Text-to-Image (T2I)`**: Sinh khuôn mặt nhân tạo từ câu lệnh mô tả nhân khẩu học (độ tuổi, giới tính, phụ kiện).
2. **`Facial Inpainting`**: Can thiệp cục bộ vùng mắt, miệng hoặc vật che trên khuôn mặt thật, bảo tồn tối đa nền gốc.
3. **`Image-to-Image (Img2Img)`**: Biến đổi phong cách, tuổi tác hoặc biểu cảm nhưng bảo lưu cấu trúc diện mạo gốc.

### 1.3. Luồng Pipeline Khép Kín Của Đề Tài

```text
[Giai đoạn 1: Sinh dữ liệu]
Ảnh mặt thật + Stable Diffusion  ───►  Ảnh giả mới (Deterministic Seeds & Provenance)
                                                │
                                                ▼
[Giai đoạn 2: Lọc chất lượng tự động]
Curation Gate (Độ nét Laplacian hiệu chỉnh động + Face Integrity + Hash Dedup)
                                                │
                                                ▼
[Giai đoạn 3: Khai thác Mẫu Khó Đối Kháng]
Khoảng cách đa tạp ảnh thật (Real-Manifold Distance) với ngưỡng Tau (τ) tự động
                                                │
                                                ▼
[Giai đoạn 4: Đóng gói Benchmark]
Đóng gói 3 tập dữ liệu đối chứng (D0, D1, D2) chuẩn Identity-Disjoint
                                                │
                                                ▼
[Giai đoạn 5: Proof of Utility (Thực nghiệm phụ)]
Huấn luyện đối chứng D0 -> D1 -> D2 chứng minh hiệu quả tăng cường bảo vệ
```

---

## 2. Cấu Trúc Thư Mục Chuẩn Hóa

```text
uth-cv-coursework/
├── configs/                        # Cấu hình tập trung (Single Source of Truth)
│   └── dataset_factory.yaml        # Mọi tham số sinh, lọc, khai thác mẫu khó và đóng gói
├── data/                           # Dữ liệu phân tầng chuẩn Topic 11
│   ├── raw/                        # Ảnh thật tham chiếu (FFHQ & CelebA-HQ, bất biến)
│   ├── synthetic/                  # Ảnh giả đã sinh và danh mục lọc chất lượng
│   ├── hard_samples/               # Mẫu khó sát đa tạp ảnh thật và biên bản audit
│   └── benchmarks/                 # 3 bộ dữ liệu đối chứng D0, D1, D2
├── docs/                           # Tài liệu nghiên cứu chi tiết & Slide báo cáo
│   ├── PURPOSE.md                  # Bản tuyên ngôn mục tiêu đề tài (Dataset-First)
│   ├── OVERVIEW.md                 # Kiến trúc hệ thống chi tiết
│   ├── coursework/                 # Phân loại học thuật & quy chuẩn dữ liệu
│   └── reports/                    # Báo cáo thực nghiệm & slide thuyết trình
├── scripts/                        # Kịch bản thực thi tự động
│   ├── organize_topic_dataset.py   # Đóng gói benchmark & xuất thống kê tự sinh
│   ├── build_visual_gallery.py     # Tạo lưới ảnh đối chứng trực quan
│   └── create_coursework_notebooks.py # Tự sinh 3 notebook kiểm tra độc lập
├── src/                            # Mã nguồn chuẩn Clean Architecture
│   ├── common/                     # Config loader, logger, seed, metrics
│   ├── data/                       # Datasets, DataModules, Provenance, Splits
│   ├── models/                     # Stable Diffusion engine, Detectors, Hardness evaluators
│   └── pipelines/                  # Pipeline sinh ảnh, curation gate, hard-sample miner
└── tests/                          # Bộ kiểm thử tự động toàn diện (80+ test cases)
```

---

## 3. Các Lệnh Chạy Thực Nghiệm

Toàn bộ quy trình được vận hành qua cấu hình tập trung `configs/dataset_factory.yaml`:

```bash
# 1. Chạy toàn bộ bài kiểm thử tự động
pytest -v

# 2. Quy trình sinh dữ liệu và đóng gói Dataset-First:
python3 -m src.pipelines.generate_deepfake_images --config configs/dataset_factory.yaml
python3 -m src.pipelines.curate_image_dataset --config configs/dataset_factory.yaml
python3 -m src.pipelines.mine_hard_samples --config configs/dataset_factory.yaml
python3 -m scripts.organize_topic_dataset --config configs/dataset_factory.yaml
python3 -m scripts.build_visual_gallery --config configs/dataset_factory.yaml
python3 scripts/create_coursework_notebooks.py
```

## Kaggle image pilot

[Notebook](notebooks/coursework/celeba-hq-df.ipynb) · [Guide](docs/reports/KAGGLE_SD_IMAGE_DEMO.md) · [HTML report](docs/reports/KAGGLE_SD_IMAGE_DEMO.html). Real T2I / img2img / inpainting; detector evaluation remains separate.
