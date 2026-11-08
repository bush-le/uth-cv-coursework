# Phần 1: Bản Đặc Tả Định Nghĩa Bài Toán & Yêu Cầu Nghiên Cứu (Problem Definition & Research Requirements Specification)

- **Motivation/Background**: Cung cấp nền tảng học thuật, lý giải cơ chế toán học miền tần số, thiết kế vòng lặp khai phá mẫu khó (Hard-Sample Mining) qua mô hình `dinoViT` và tiêu chí chứng minh tác dụng cho đồ án: "Stable Diffusion-Based Synthetic Facial Dataset Generation with DINO ViT Hard-Sample Mining for Deepfake Detection" (Đề tài 11, bám sát học liệu Hugging Face Computer Vision Course Unit 10, cập nhật theo định hướng của Giảng viên hướng dẫn).
- **Purpose**: Đóng vai trò là Phần 1 chính thức trong Báo cáo Kỹ thuật Tổng thể của Đồ án Môn học Thị giác Máy tính (UTH), giải thích tại sao ảnh giả Stable Diffusion có thể đánh lừa (`Bypass`) mô hình Baseline `dinoViT` và cách thu thập các mẫu `Hard Samples` để tinh chỉnh (`Fine-tune`) nhằm chứng minh tác dụng thực tiễn của dữ liệu tổng hợp.
- **Overview Pipeline**: Khoảng trống học thuật (GAN vs. LDM) -> Phổ Fourier 2D -> Phase 1 Sinh dữ liệu SD & Curation -> Phase 2 Benchmark `dinoViT` (Reject vs. Bypass $\rightarrow$ Hard Samples) -> Phase 3 Fine-Tune trên `Data Baseline + Syn Data + Hard Samples` -> Chứng minh tác dụng.
- **Detailed Plan**: §1 Bối cảnh Học thuật & Bài Toán Chứng Minh Tác Dụng Dữ Liệu; §2 Phân Tích Miền Tần Số: Latent Diffusion vs. GAN; §3 Ba Hình Thái Thao Túng Khuôn Mặt (Phase 1); §4 Cơ Chế Khai Phá Mẫu Khó Qua Baseline `dinoViT` (Phase 2); §5 Tinh Chỉnh Khép Kín & Phân Hoạch Chống Rò Rỉ Danh Tính (Phase 3); §6 Tiêu Chí Thành Công Định Lượng (KPIs).
- **References**: `docs/PURPOSE.md`, `docs/OVERVIEW.md`, `configs/dataset/deepfake_image.yaml`, `configs/generation/stable_diffusion.yaml`, `configs/detection/detector_baseline.yaml`, [Hugging Face CV Course Unit 10](https://huggingface.co/learn/computer-vision-course/unit10/datagen-diffusion), Caron et al. (DINO, ICCV 2021), Oquab et al. (DINOv2, TMLR 2024), Rombach et al. (CVPR 2022).
- **Created**: 2026-11-07T00:46:00+07:00
- **Last Updated**: 2026-11-08T17:52:00+07:00

---

## 1. Bối Cảnh Học Thuật & Bài Toán Chứng Minh Tác Dụng Dữ Liệu

### 1.1 Sự Dịch Chuyển Sang Latent Diffusion & Thách Thức Thực Nghiệm
Các bộ phát hiện Deepfake truyền thống (huấn luyện trên ProGAN, StyleGAN2, FaceForensics++) dựa vào các dấu vết tích chập chuyển vị (sọc ô cờ *checkerboard artifacts* hoặc nhiễu tần số cao). Khi gặp ảnh khuôn mặt sinh từ **Stable Diffusion (Latent Diffusion Models — LDM)**, các dấu vết chu kỳ này biến mất, khiến ngay cả các kiến trúc mạnh như **Vision Transformer tự giám sát (`dinoViT`)** cũng bị đánh lừa (*Bypass*) trên một bộ phận đáng kể các mẫu ảnh giả chân thực.

### 1.2 Bài Toán Trọng Tâm Từ Giảng Viên Hướng Dẫn
Để **chứng minh được tác dụng (`chứng minh được tác dụng`)** của bộ dữ liệu tổng hợp sinh ra ở Phase 1, đồ án không dừng ở việc sinh dữ liệu tĩnh mà thiết lập **vòng lặp kiểm chứng đối kháng 3 pha**:

```text
+---------------------------------------------------------------------------------------------------+
|                   CHU TRÌNH KHÉP KÍN CHỨNG MINH TÁC DỤNG DỮ LIỆU (3 PHASES)                       |
+---------------------------------------------------------------------------------------------------+
|  [PHASE 1: STABLE DIFFUSION SYNTHESIS & CURATION]                                                 |
|  Ảnh thật (img) + Stable Diffusion (Text2Img / Inpaint / Img2Img) + Curation Gate                 |
|  ===> Tập ảnh giả đạt chuẩn: Synthetic DF Image (DF img)                                          |
|                                                                                                   |
|  [PHASE 2: BASELINE dinoViT BENCHMARK & BYPASS MINING]                                            |
|  Đưa DF img qua Baseline Model dinoViT (Frozen Benchmark):                                        |
|  * Nếu P(fake) >= 0.50 ---> REJECT (Mô hình bắt được) ---> Regular Synthetic Data                 |
|  * Nếu P(fake) <  0.50 ---> BYPASS (Vượt qua / Đánh lừa mô hình) ---> Thu thập thành HARD SAMPLE  |
|                                                                                                   |
|  [PHASE 3: CLOSED-LOOP FINE-TUNING & PROOF OF UTILITY]                                            |
|  Hợp nhất: Data Baseline + Synthetic Data + Hard Samples ---> Fine-tune dinoViT                   |
|  ===> Nếu mô hình sau Fine-tune bắt được các mẫu Bypass & tăng ROC-AUC ---> CHỨNG MINH TÁC DỤNG!  |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Phân Tích Cơ Chế Miền Tần Số: Latent Diffusion vs. GAN Fingerprints

Biến đổi Fourier rời rạc 2 chiều (2D DFT) của ảnh $I(x, y)$ kích thước $H \times W$:
$$\mathcal{F}\{I\}(u, v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} I(x, y) \cdot \exp\left( -j 2\pi \left( \frac{ux}{H} + \frac{vy}{W} \right) \right)$$

- **GAN Up-sampling**: Xuất hiện các đỉnh năng lượng chu kỳ cố định (*spectral spikes*) trong phổ mật độ năng lượng $P(u, v) = |\mathcal{F}\{I\}(u, v)|^2$.
- **Stable Diffusion (LDM)**: Khử nhiễu liên tục trong không gian tiềm ẩn của VAE giúp mật độ phổ tuân theo định luật tự nhiên $\text{PSD}(f) \propto 1/f^\alpha$ ($\alpha \approx 2.0$), khiến nhiều mẫu `Synthetic DF` nằm lọt vào bên kia biên quyết định của `dinoViT` Baseline (tạo thành các **Hard Samples**).

---

## 3. Ba Hình Thái Thao Túng Khuôn Mặt (Phase 1 Taxonomy)

1. **`sd_text2img` (Text-to-Image)**: Sinh hoàn toàn mới khuôn mặt nhân tạo từ nhiễu Gaussian $z_T \sim \mathcal{N}(0, \mathbf{I})$.
2. **`sd_inpaint` (Facial Inpainting)**: Chỉnh sửa cục bộ vùng mắt, mũi, miệng trong mặt nạ nhị phân $M \in \{0, 1\}^{1 \times H \times W}$:
   $$I_{\text{inpaint}} = (1 - M) \odot I_{\text{real}} + M \odot I_{\text{gen}}$$
3. **`sd_img2img` (Image-to-Image Morphing)**: Bơm nhiễu một phần với cường độ $S \in [0.4, 0.75]$ rồi khử nhiễu ngược để biến đổi tinh vi kết cấu da và biểu cảm.

Mọi ảnh sinh ra phải vượt qua `ImageCurationGate` (Laplacian Sharpness $\ge 50.0$, CLIP Score $\ge 0.28$, Face Confidence $\ge 0.85$) trước khi đưa vào `dinoViT`.

---

## 4. Cơ Chế Khai Phá Mẫu Khó Qua Baseline `dinoViT` (Phase 2)

Với mỗi ảnh giả $x_i$ ($y_i = 1$) đã qua cổng lọc chất lượng, mô hình **Baseline `dinoViT`** (`DeepfakeImageDetector(backbone="dino_vit")`) xuất xác suất dự đoán $p_i = \sigma(\text{dinoViT}(x_i)) \in [0, 1]$:
- **Trường hợp 1 — `Reject` ($p_i \ge 0.50$)**: Baseline `dinoViT` phát hiện thành công đây là ảnh Deepfake (`is_hard_sample = False`, `data_source = "synthetic"`).
- **Trường hợp 2 — `Bypass` ($p_i < 0.50$)**: Ảnh Deepfake đã **qua mặt (bypass)** được Baseline `dinoViT` (bị dự đoán nhầm thành ảnh thật). Hệ thống tự động gắn cờ `is_hard_sample = True`, `baseline_score = p_i`, `data_source = "hard_sample"` và lưu vào `hard_samples_manifest.jsonl`.

---

## 5. Tinh Chỉnh Khép Kín & Phân Hoạch Chống Rò Rỉ Danh Tính (Phase 3)

1. **Tổ hợp Dữ liệu Tinh chỉnh (`Combined Training Pool`)**:
   $$\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{baseline}} \cup \mathcal{D}_{\text{syn}} \cup \mathcal{D}_{\text{hard}}$$
   Trong đó nhóm $\mathcal{D}_{\text{hard}}$ được nhân bản (`hard_sample_oversample_factor = 2`) và đánh trọng số hàm mất mát cao gấp đôi (`hard_sample_loss_weight = 2.0`).
2. **Nguyên lý Bất biến Danh tính (`Identity-Disjoint Invariant`)**:
   $$\forall x \text{ có } \text{source\_real\_id}(x) = s \implies \text{Split}(x) = \text{Split}(s)$$
   Toàn bộ ảnh gốc, ảnh `Synthetic` và `Hard Sample` xuất phát từ cùng một người thật bắt buộc nằm chung một phân vùng (`Train`, `Val`, hoặc `Test`).

---

## 6. Tiêu Chí Thành Công Định Lượng (KPIs)

- **Chất lượng ảnh Phase 1**: Laplacian Sharpness $\ge 50.0$, CLIP Score $\ge 0.28$, Face Retention $\ge 95\%$.
- **Hiệu quả sau Fine-Tune `dinoViT` (Phase 3)**:
  - **ROC-AUC tổng thể**: $\ge 92.0\%$ và **Accuracy**: $\ge 85.0\%$ trên tập kiểm thử độc lập.
  - **Hard-Sample Recall**: Tăng vọt khả năng bắt giữ các mẫu từng bị bypass (đạt $\ge 85.0\%$ trên tập test Hard Samples).
  - **Bypass Rate Reduction**: Giảm mạnh tỷ lệ ảnh giả Stable Diffusion vượt qua mô hình.
