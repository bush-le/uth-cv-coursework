# UTH Computer Vision Coursework — Topic 11 (Nhóm 4)
**Sinh ảnh khuôn mặt giả bằng Stable Diffusion, tìm mẫu khó (Hard Sample) qua `dinoViT` và huấn luyện lại để chứng minh tác dụng**

- **Motivation/Background**: Dự án môn học Thị giác Máy tính (Topic 11) của Nhóm 4, thực hiện luồng nghiên cứu khép kín 3 bước: Dùng Stable Diffusion tạo ảnh giả khuôn mặt -> Đưa qua mô hình chuẩn `dinoViT` để tìm Mẫu khó (`Hard Sample`) -> Huấn luyện lại `dinoViT` trên 3 nguồn dữ liệu để chứng minh dữ liệu sinh ra có tác dụng thực tế.
- **Purpose**: Đóng vai trò tài liệu hướng dẫn tổng quan, cung cấp luồng bài toán đơn giản, cấu trúc dự án và các lệnh chạy nhanh cho giảng viên và sinh viên.
- **Overview Pipeline**: Bước 1: Tạo & lọc ảnh giả SD -> Bước 2: Chạy qua `dinoViT` tìm Mẫu khó (`Bypass`) -> Bước 3: Huấn luyện lại `dinoViT` trên `Data Baseline + Syn Data + Hard Sample` và đo kết quả.
- **Detailed Plan**: §1 Ý Tưởng Cốt Lõi (Luồng 3 Bước); §2 Cấu Trúc Thư Mục; §3 Ba Bước Chạy Thực Nghiệm Nhanh; §4 Kiểm Tra Mã Nguồn; §5 Tài Liệu Tham Khảo.
- **References**: [docs/PURPOSE.md](docs/PURPOSE.md), [docs/OVERVIEW.md](docs/OVERVIEW.md), [docs/reports/presentation_5min_define_and_design.html](docs/reports/presentation_5min_define_and_design.html).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-09T13:19:40+07:00

---

## 1. Ý Tưởng Cốt Lõi (Problem ➔ Business ➔ Technical)

### 1.1. Phân Loại Deepfake (What Is Deepfake?)
- **Deepfake Cũ (Ghép mặt / Face-Swap)**: Cắt khuôn mặt người này dán đè lên người khác, để lại **đường viền cắt dán mờ quanh mặt** hoặc lệch màu da. Các mô hình phát hiện truyền thống chỉ cần dựa vào vết ranh giới này là phát hiện được.
- **Deepfake Mới (Stable Diffusion)**: Sử dụng mô hình khuếch tán trong không gian tiềm ẩn (`Latent Space`) để vẽ lại mượt mà toàn bộ khuôn mặt hoặc từng chi tiết. Ảnh sinh ra **không còn vết cắt ghép**, ánh sáng tự nhiên, khiến các mô hình phát hiện cũ bị "mù" và đoán nhầm là ảnh thật.

### 1.2. Stable Diffusion Là Gì & Tạo Dữ Liệu Dạng Nào?
Stable Diffusion khử nhiễu từ không gian tiềm ẩn, hỗ trợ tạo **3 dạng dữ liệu khuôn mặt bổ trợ**:
1. **`Text-to-Image (T2I)`**: Sinh khuôn mặt nhân tạo mới hoàn toàn từ mô tả văn bản (Prompt).
2. **`Facial Inpainting`**: Chỉ chỉnh sửa cục bộ vùng mắt hoặc miệng trên khuôn mặt thật, giữ nguyên viền đầu và tóc. Đây là dạng giả mạo tinh vi nhất.
3. **`Image-to-Image (Img2Img)`**: Biến đổi diện mạo (tuổi tác, biểu cảm) nhưng giữ nguyên khung 5 điểm mốc hình học của ảnh gốc.

### 1.3. Bộ 3 Tầng Luận Cứ Chuẩn
- **Problem (Vấn đề)**: Sự dịch chuyển từ ảnh cắt dán sang ảnh khuếch tán khiến các mô hình phát hiện cũ bị đánh lừa.
- **Business (Nghiệp vụ & Ứng dụng)**: Đe dọa trực tiếp các cổng định danh điện tử **eKYC ngân hàng**, mở tài khoản từ xa và ví điện tử. Doanh nghiệp cần giải pháp phòng vệ chủ động: tự sinh dữ liệu để huấn luyện trước hàng rào bảo mật.
- **Technical (Giải pháp kỹ thuật khép kín của Nhóm 4)**: Luồng 3 bước bên dưới:

```text
[Bước 1: Sinh dữ liệu]
Ảnh mặt thật + Stable Diffusion  ───►  Ảnh giả mới (Synthetic DF Image)
                                                │
                                                ▼
[Bước 2: Thử thách mô hình gốc (Baseline dinoViT)]
Đưa ảnh giả mới qua mô hình dinoViT gốc:
  ├── Nếu dinoViT BẮT ĐƯỢC (Reject)  ───►  Ảnh giả thông thường (Synthetic Data)
  └── Nếu dinoViT BỊ LỪA (Bypass)   ───►  Lấy làm MẪU KHÓ (Hard Sample)
                                                │
                                                ▼
[Bước 3: Huấn luyện lại (Fine-Tune) & Chứng minh tác dụng]
Gộp (Dữ liệu gốc + Ảnh sinh mới + Mẫu khó) ───► Huấn luyện lại dinoViT
  └── Kết quả: Mô hình mới bắt được các Mẫu khó trước đó ───► CHỨNG MINH DỮ LIỆU CÓ TÁC DỤNG!
```

1. **Bước 1 — Tạo ảnh giả bằng Stable Diffusion**: Dùng 3 cách tạo ảnh (`Text-to-Image` tạo mặt mới từ chữ, `Inpainting` sửa vùng mắt/miệng, `Img2Img` biến đổi từ ảnh thật) rồi lọc bỏ ảnh mờ, méo mặt.
2. **Bước 2 — Tìm mẫu khó bằng mô hình gốc `dinoViT`**: Đưa các ảnh giả vừa tạo vào mô hình phát hiện gốc `dinoViT`:
   - Ảnh nào bị `dinoViT` phát hiện là giả (**Reject**) $\rightarrow$ xếp vào nhóm ảnh giả thông thường.
   - Ảnh nào đánh lừa được `dinoViT` khiến nó tưởng là ảnh thật (**Bypass**) $\rightarrow$ gom lại thành nhóm **Mẫu khó (`Hard Sample`)**.
3. **Bước 3 — Huấn luyện lại và chứng minh tác dụng**: Gộp chung **Dữ liệu gốc (`Data Baseline`) + Ảnh sinh mới (`Syn Data`) + Mẫu khó (`Hard Sample`)** để huấn luyện lại (`Fine-Tune`) mô hình `dinoViT`. Nếu sau khi học xong, mô hình bắt được các mẫu khó từng lọt lưới thì chứng minh được dữ liệu sinh ra thực sự có tác dụng.

---

## 2. Cấu Trúc Thư Mục

```text
uth-cv-coursework/
├── configs/                        # Các file cấu hình chạy thực nghiệm
│   ├── config.yaml                 # Cấu hình chung toàn hệ thống
│   ├── dataset/
│   │   └── deepfake_image.yaml     # Cấu hình dữ liệu (Gốc + Sinh mới + Mẫu khó)
│   └── detection/
│       └── detector_baseline.yaml  # Cấu hình mô hình dinoViT & ngưỡng lọc Bypass
├── docs/                           # Toàn bộ tài liệu & Slide thuyết trình 5 phút
│   ├── PURPOSE.md                  # Đề bài & mục tiêu của Nhóm 4
│   ├── OVERVIEW.md                 # Giải thích tổng quan hệ thống
│   ├── coursework/                 # Tổng quan lý thuyết & cách chia dữ liệu
│   ├── experiments/                # Cách so sánh 3 chế độ D0 / D1 / D2
│   ├── phases/                     # Chi tiết Bước 1, Bước 2, Bước 3
│   ├── progress/                   # Lộ trình thực hiện M0 -> M7
│   └── reports/                    # Báo cáo & Bộ 11 Slide HTML thuyết trình
├── scripts/
│   └── run_pipeline.sh             # Lệnh chạy nhanh từng bước
├── src/                            # Mã nguồn chính
│   ├── common/                     # Các hàm tính điểm (ROC-AUC, Bypass Rate, Hard Recall)
│   ├── data/                       # Đọc dữ liệu, lọc ảnh lỗi & chia tập không trùng người
│   ├── models/                     # Mô hình tạo ảnh SD & mô hình phát hiện dinoViT
│   └── pipelines/                  # Các bước chạy: sinh ảnh -> lọc -> tìm mẫu khó -> train -> đánh giá
└── tests/                          # Bộ kiểm thử tự động (pytest)
```

---

## 3. Các Lệnh Chạy Nhanh

```bash
# 1. Chạy kiểm thử nhanh (< 5 giây)
make test-smoke

# 2. Chạy toàn bộ bài kiểm thử
make test

# 3. Chạy lần lượt 3 bước của đề tài:
bash scripts/run_pipeline.sh generate configs/config.yaml   # Bước 1a: Sinh ảnh giả bằng SD
bash scripts/run_pipeline.sh curate configs/config.yaml     # Bước 1b: Lọc bỏ ảnh lỗi
bash scripts/run_pipeline.sh mine configs/config.yaml       # Bước 2: Đưa qua dinoViT tìm Mẫu khó (Bypass)
bash scripts/run_pipeline.sh train configs/config.yaml      # Bước 3a: Fine-tune lại dinoViT trên tập gộp
bash scripts/run_pipeline.sh eval configs/config.yaml       # Bước 3b: Chấm điểm chứng minh tác dụng
```

## Kaggle image pilot

[Notebook](notebooks/SD_IMAGE_DEMO_KAGGLE.ipynb) ? [Guide](docs/reports/KAGGLE_SD_IMAGE_DEMO.md) ? [HTML report](docs/reports/KAGGLE_SD_IMAGE_DEMO.html). Real T2I / img2img / inpainting; detector evaluation remains separate.
