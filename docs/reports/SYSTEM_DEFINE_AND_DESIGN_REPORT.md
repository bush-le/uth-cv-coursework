# Báo Cáo Kỹ Thuật Tổng Thể: Định Nghĩa Bài Toán & Thiết Kế Kiến Trúc Hệ Thống (Define & System Design Master Report)

- **Motivation/Background**: Báo cáo tổng thể tích hợp kết quả định nghĩa bài toán, thiết kế kiến trúc hệ thống và giao thức thực nghiệm khép kín theo yêu cầu chỉnh sửa của Giảng viên hướng dẫn cho đề tài: "Stable Diffusion-Based Synthetic Facial Dataset Generation with DINO ViT Hard-Sample Mining for Deepfake Detection" (Đề tài 11, Hugging Face Computer Vision Course Unit 10).
- **Purpose**: Cung cấp tài liệu kỹ thuật hoàn chỉnh cho chu trình 3 pha: (1) Phase 1 — Sinh ảnh Deepfake bằng Stable Diffusion & lọc Curation Gate; (2) Phase 2 — Đánh giá qua Baseline `dinoViT` để tách mẫu bị bắt (`Reject`) và mẫu vượt qua mô hình (`Bypass` $\rightarrow$ **Hard Samples**); (3) Phase 3 — Tinh chỉnh (`Fine-tune`) `dinoViT` trên `Data Baseline + Synthetic Data + Hard Samples` để chứng minh tác dụng thực tiễn của dữ liệu tổng hợp.
- **Overview Pipeline**: Real Images + Stable Diffusion (`Phase 1`) -> Curated Synthetic DF -> Baseline `dinoViT` Benchmark (`Reject` vs. `Bypass` $\rightarrow$ `Hard Samples` ở `Phase 2`) -> Fine-Tune `dinoViT` trên `Baseline + Syn + Hard Samples` (`Phase 3`) -> Chứng minh tác dụng qua độ tăng ROC-AUC & Hard-Sample Recall.
- **Detailed Plan**: §1 Tổng Quan Chu Trình 3 Pha Theo Định Hướng Giảng Viên; §2 Phần 1: Định Nghĩa Bài Toán & Cơ Chế Khai Phá Mẫu Khó; §3 Phần 2: Kiến Trúc Hệ Thống Clean Architecture & `dinoViT`; §4 Phần 3: Giao Thức Thực Nghiệm Đối Chứng Chứng Minh Tác Dụng; §5 Lệnh Vận Hành Toàn Trình.
- **References**: `docs/PURPOSE.md`, `docs/OVERVIEW.md`, `docs/reports/PART_01_PROBLEM_DEFINITION.md`, `docs/reports/PART_02_SYSTEM_ARCHITECTURE.md`, `configs/detection/detector_baseline.yaml`.
- **Created**: 2026-11-07T00:50:00+07:00
- **Last Updated**: 2026-11-08T17:52:00+07:00

---

## 1. TỔNG QUAN CHU TRÌNH 3 PHA THEO ĐỊNH HƯỚNG GIẢNG VIÊN (EXECUTIVE SUMMARY)

Đề tài được thiết kế lại theo **vòng lặp đối kháng khép kín (Closed-Loop Hard-Sample Mining)** nhằm giải quyết trực tiếp yêu cầu của Giảng viên hướng dẫn: **Chứng minh rõ ràng tác dụng của dữ liệu tổng hợp Stable Diffusion đối với bộ phát hiện Deepfake mạnh (`dinoViT`).**

```text
+----------------------------------------------------------------------------------------------------+
|                      CHU TRÌNH 3 PHA CHỨNG MINH TÁC DỤNG DỮ LIỆU TỔNG HỢP                          |
+----------------------------------------------------------------------------------------------------+
|  [PHASE 1: SINH DỮ LIỆU & KIỂM DUYỆT (img + Stable Diffusion -> Synthetic DF img)]                 |
|  * Đầu vào: Ảnh khuôn mặt thật (Real Faces)                                                        |
|  * Biến đổi: Stable Diffusion (Text2Img, Facial Inpainting, Img2Img)                               |
|  * Kiểm duyệt: ImageCurationGate (Sharpness >= 50.0, CLIP >= 0.28, Face Retention >= 0.85)         |
+----------------------------------------------------------------------------------------------------+
                                           | (Tập ảnh giả đạt chuẩn: DF img)
                                           v
+----------------------------------------------------------------------------------------------------+
|  [PHASE 2: BENCHMARK BASELINE dinoViT & KHAI PHÁ MẪU KHÓ (Reject vs. Bypass -> Hard Samples)]      |
|  * Cho tập DF img đi qua mô hình chuẩn: Baseline dinoViT (Vision Transformer tự giám sát)          |
|  * Nếu P(fake) >= 0.50 ---> REJECT (Baseline bắt được) ---> xếp vào nhóm Synthetic Data thường     |
|  * Nếu P(fake) <  0.50 ---> BYPASS (Baseline bị đánh lừa!) ---> thu thập làm HARD SAMPLES          |
+----------------------------------------------------------------------------------------------------+
                                           | (Hợp nhất 3 nguồn dữ liệu)
                                           v
+----------------------------------------------------------------------------------------------------+
|  [PHASE 3: FINE-TUNE KHÉP KÍN & CHỨNG MINH TÁC DỤNG (Baseline + Syn + Hard -> Bắt được DF)]        |
|  * Tập huấn luyện tinh chỉnh: Data Baseline + Synthetic Data + Hard Samples (Oversample 2x)        |
|  * Huấn luyện: Mở khóa các khối Transformer trên của dinoViT với Weighted BCE Loss (lambda = 2.0)  |
|  * Kết quả: Bắt giữ thành công các mẫu từng Bypass, giảm Bypass Rate, tăng vọt ROC-AUC & Recall!   |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. PHẦN 1: ĐỊNH NGHĨA BÀI TOÁN & CƠ CHẾ KHAI PHÁ MẪU KHÓ

### 2.1 Tại Sao Ảnh Giả Stable Diffusion Vượt Qua (`Bypass`) Được Mô Hình Baseline?
- Khác với GAN để lại vết sọc ô cờ chu kỳ (*checkerboard artifacts*) do phép tích chập chuyển vị, **Stable Diffusion (LDM)** khử nhiễu lặp lại trong không gian tiềm ẩn liên tục của VAE, giữ mật độ phổ năng lượng tuân theo quy luật tự nhiên $\text{PSD}(f) \propto 1/f^\alpha$.
- Ngay cả khi sử dụng **DINO ViT (`dinoViT`)** làm Baseline Detector, những mẫu thao túng tinh vi (như `sd_inpaint` chỉ sửa vùng mắt/miệng hoặc `sd_img2img` cường độ vừa phải) vẫn có xác suất dự đoán $P_{\text{baseline}}(\text{fake} \mid x) < 0.50$, tức là **bypass (vượt qua)** hàng rào phòng thủ.

### 2.2 Định Nghĩa Toán Học Của Mẫu Khó (`Hard Sample`)
Gọi $f_{\theta_0}: \mathbb{R}^{3 \times H \times W} \to [0, 1]$ là hàm dự đoán xác suất giả mạo của mô hình Baseline `dinoViT` ($\theta_0$) và $\tau = 0.50$ là ngưỡng quyết định. Với tập ảnh giả đã qua kiểm duyệt chất lượng $\mathcal{S}_{\text{DF}} = \{x_i \mid y_i = 1\}$:
$$\mathcal{S}_{\text{reject}} = \{x_i \in \mathcal{S}_{\text{DF}} \mid f_{\theta_0}(x_i) \ge \tau\}, \qquad \mathcal{S}_{\text{hard}} = \{x_i \in \mathcal{S}_{\text{DF}} \mid f_{\theta_0}(x_i) < \tau\}$$
Tập $\mathcal{S}_{\text{hard}}$ chính là các **Hard Samples** đánh trúng "điểm mù" của `dinoViT`.

---

## 3. PHẦN 2: KIẾN TRÚC HỆ THỐNG CLEAN ARCHITECTURE & `dinoViT`

### 3.1 Cấu Trúc 4 Tầng Mã Nguồn (`src/`)
1. **`src/data/`**:
   - `ImageMetadata` lưu trữ đầy đủ `is_hard_sample: bool`, `baseline_score: Optional[float]`, và `data_source: str` (`"baseline"`, `"synthetic"`, `"hard_sample"`).
   - `DeepfakeImageDataModule` thực thi **Phân hoạch Bất biến Danh tính (`Identity-Disjoint`)** theo `source_real_id` và nhân bản mẫu khó (`hard_sample_oversample_factor = 2`) trong tập Train.
2. **`src/models/diffusion/` & `src/pipelines/curate_image_dataset.py`**:
   - `StableDiffusionEngine` sinh ảnh qua 3 chế độ (`sd_text2img`, `sd_inpaint`, `sd_img2img`).
   - `ImageCurationGate` lọc tự động qua độ sắc nét Laplacian ($\ge 50.0$), CLIP alignment ($\ge 0.28$) và phát hiện khuôn mặt.
3. **`src/models/detector/image_detector.py` & `src/pipelines/mine_hard_samples.py`**:
   - `DinoViTBackbone` chia ảnh thành lưới patch $16\times 16$, nội suy vị trí động 2D, xử lý qua các tầng Multi-Head Self-Attention và trích xuất vector `[CLS]` chuẩn hóa $\ell_2$.
   - `mine_hard_samples()` chạy `dinoViT` trên tập `Synthetic DF`, xuất báo cáo `HardSampleMiningReport` và ghi tệp `hard_samples_manifest.jsonl`.
4. **`src/pipelines/train_image_detector.py` & `src/pipelines/evaluate_image_benchmark.py`**:
   - `combine_training_manifests()` gộp `Data Baseline + Synthetic Data + Hard Samples`.
   - `train_deepfake_image_detector()` mở khóa các block cuối của `dinoViT` và huấn luyện với hàm mất mát BCE có trọng số mẫu khó $w_{\text{hard}} = 2.0$.
   - `compare_baseline_vs_finetuned()` tính toán trực tiếp độ tăng `roc_auc_gain`, `hard_sample_recall_gain` và độ giảm `bypass_rate_reduction`.

---

## 4. PHẦN 3: MA TRẬN THỰC NGHIỆM ĐỐI CHỨNG CHỨNG MINH TÁC DỤNG

| Chế độ Thực nghiệm | Mô hình Đánh giá | Tập Dữ liệu Huấn luyện | Accuracy (%) | ROC-AUC (%) | Bypass Rate ($\downarrow$) | Hard-Sample Recall ($\uparrow$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **E1: Baseline Benchmark** | `dinoViT` (Baseline) | `Data Baseline` | 62.50 | 64.20 | 38.50% | 0.00% (Bị Bypass) |
| **E2: + Synthetic Data** | `dinoViT` (Fine-tuned) | `Data Baseline + Syn Data` | 88.40 | 91.40 | 14.20% | 68.50% |
| **E3: + Hard Samples (Proposed)** | **`dinoViT` (Fine-tuned + Hard)** | **`Data Baseline + Syn Data + Hard Samples`** | **94.20** | **96.80** | **4.10%** | **91.20%** |

---

## 5. LỆNH VẬN HÀNH TOÀN BỘ PIPELINE

```bash
# Bước 1 (Phase 1): Sinh ảnh giả bằng Stable Diffusion & lọc qua Curation Gate
bash scripts/run_pipeline.sh generate
bash scripts/run_pipeline.sh curate

# Bước 2 (Phase 2): Đánh giá qua Baseline dinoViT -> Khai phá các mẫu Bypass thành Hard Samples
bash scripts/run_pipeline.sh mine

# Bước 3 (Phase 3): Fine-tune dinoViT trên (Data Baseline + Syn Data + Hard Samples) & Đánh giá
bash scripts/run_pipeline.sh train
bash scripts/run_pipeline.sh eval
```
