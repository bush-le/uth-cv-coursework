# BỘ CÂU HỎI & TRẢ LỜI NHANH BẢO VỆ ĐỒ ÁN (PRESENTATION DEFENSE Q&A CHEAT SHEET)

- **Motivation/Background**: Khi bảo vệ đồ án chuyên sâu về Thị giác Máy tính & Phát hiện Deepfake trước hội đồng giảng viên UTH, nhóm cần một tài liệu tra cứu nhanh các câu hỏi phản biện kỹ thuật then chốt, cơ sở toán học và số liệu thực nghiệm đã kiểm chứng.
- **Purpose**: Cung cấp bộ câu hỏi và câu trả lời chính xác tuyệt đối với mã nguồn (`src/`) và log thực nghiệm (`experiments/`) cho Đề tài 11.
- **Overview Pipeline**: Tiếp nhận câu hỏi hội đồng -> Định vị phương diện kỹ thuật -> Trả lời theo cấu trúc luận điểm khoa học kèm số liệu thực chứng chuẩn `512→504px`.
- **Detailed Plan**: §1 Kiến trúc mô hình & DINOv2; §2 Kỹ thuật sinh dữ liệu & Soft Feathering; §3 Active Learning & Hard Sample Mining; §4 Benchmark Phổ Quát & Năng Lực Khái Quát Hóa; §5 Bảng Số Liệu Vàng (Verified Golden Key Facts).
- **References**: `../../AGENTS.md`, `SYSTEM_DEFINE_AND_DESIGN_REPORT.md`, `ACTIVE_LEARNING_M2_REFINED_REPORT.md`, `M3_DUAL_ENGINE_CHAMPION_REPORT.md`.
- **Created**: 2026-11-11T23:25:00+07:00
- **Last Updated**: 2026-11-21T18:10:00+07:00

---

## 1. Nhóm Câu Hỏi Về Kiến Trúc Bộ Dò & DINOv2

### Q1: Tại sao nhóm chọn DINOv2-ViT-S/14 thay vì ResNet-50 hoặc EfficientNet truyền thống?
**Trả lời:**
- Các mạng CNN tiền huấn luyện có giám sát (Supervised ImageNet) thường dựa vào **ngữ nghĩa vật thể cấp cao (Semantic Bias)** như hình dáng tổng thể mắt, mũi, miệng.
- **DINOv2** được huấn luyện tự giám sát (Self-Supervised Learning) trên 142 triệu ảnh bằng DINO + iBOT, bảo tồn nguyên vẹn đặc trưng **kết cấu vi thể (micro-texture)** ở cấp độ patch $14 \times 14$, nhạy hơn với các dị thường tần số cao của quá trình khử nhiễu khuếch tán.
- DINOv2-ViT-S/14 + Head chỉ có **22.06 triệu tham số**, đạt tốc độ suy luận ~18.5 ms/ảnh trên GPU.

### Q2: Cơ chế trích xuất đặc trưng và đầu phân loại thực tế của `DeepfakeImageDetector` hoạt động như thế nào?
**Trả lời (`src/models/detector/image_detector.py`):**
- Ảnh đầu vào RGB kích thước $(3, 512, 512)$ đi qua `MetaDinoV2Wrapper` và được snap theo bội 14 (`(h // 14) * 14`) về **$(3, 504, 504)$**, rồi chuẩn hóa theo thống kê ImageNet.
- DINOv2 chia lưới thành **$36 \times 36 = 1.296$ patch tokens** kích thước $14 \times 14$ px + 1 token `[CLS]`.
- Toàn bộ 11 tầng đầu (`Blocks 0–10`) được đóng băng (`91.94%` tham số); chỉ mở khóa **Block 11 + LayerNorm + Classifier Head** (`1.777.153` tham số trainable = **8.06%**).
- Vector `[CLS]` (`384` chiều, chuẩn hóa L2) đi trực tiếp qua đầu phân loại: `LayerNorm(384) -> Dropout(0.3) -> Linear(384, 1)` (`1.153` tham số) để xuất logit nhị phân $z$, sau đó hiệu chuẩn nhiệt độ $P(\text{Fake}) = \sigma(z / T)$.

---

## 2. Nhóm Câu Hỏi Về Dữ Liệu Sinh & Động Cơ Khuếch Tán Kép

### Q3: Hiện tượng "học tủ biên ghép" (Boundary Artifact Overfitting) là gì và nhóm giải quyết ra sao?
**Trả lời:**
- Khi dùng Inpainting thông thường với mặt nạ nhị phân cứng (Hard Binary Mask), đường ranh giới giữa vùng ảnh gốc và vùng sinh mới tạo bước nhảy gradient pixel sắc nhọn khiến detector chỉ "soi đường cắt" thay vì học cấu trúc khuôn mặt giả.
- Nhóm áp dụng **Làm Mịn Biên Gaussian Mềm (Soft-Mask Feathering)** trên mặt nạ elip giải phẫu (`blur_radius = 18–20` trong `stable_diffusion_engine.py`, hình minh họa trực quan dùng $\sigma=4.0$):
  $$I_{\text{composite}} = I_{\text{inpaint}} \odot M_{\text{feathered}} + I_{\text{real}} \odot (1.0 - M_{\text{feathered}})$$
  tạo dải chuyển tiếp mượt, triệt tiêu vết cắt nhân tạo ở biên mặt nạ.

### Q4: Động cơ khuếch tán kép (Dual-Engine) gồm những gì và tại sao cần kết hợp?
**Trả lời:**
- **Engine A (`SD v1.5`):** Sinh `6.545` ảnh thô (`5.251` ảnh sạch sau `CurationGate`) phục vụ khởi tạo $D_0$ và 2 vòng Active Learning nội miền (`740` mẫu khó).
- **Engine B (`RealisticVision v5.1`):** Kho `6.000` ảnh siêu thực OOD (`2.000 text2img`, `2.000 inpaint`, `2.000 img2img`), cách ly tuyệt đối **626 ảnh trùng danh tính test** sang `unseen_rv51_test.jsonl` và trích **1.384 ảnh sạch danh tính** vào tập $D_3$ để huấn luyện $M_3$ mà **không có Zero Identity Leakage**.

---

## 3. Nhóm Câu Hỏi Về Active Learning & Hard Sample Mining

### Q5: Tiêu chuẩn toán học nào được dùng để xác định một "Mẫu Cực Khó" (Hard Sample)?
**Trả lời (`src/models/evaluators/calibrated_mining.py` & `configs/active_learning.yaml`):**
1. **Mẫu bỏ lọt (False Negative):** Ảnh giả vượt qua detector ($P(\text{Fake} \mid x) < \tau = 0.50$).
2. **Mẫu ranh giới mơ hồ (Uncertain / Margin Ambiguity):** Xác suất sau hiệu chuẩn Temperature Scaling ($T = 1.9831$ ở R2) rơi sát ngưỡng quyết định ($|P(\text{Fake} \mid x) - 0.50| \le 0.10$ khi mining vòng AL, hoặc $\le 0.15$ khi profiling OOD).
- Nhóm khai phá được **740 mẫu khó SD 1.5**: `629` mẫu vào tập train ($D_2 / D_3$) và khóa cứng **111 mẫu giả khó** cùng **102 ảnh thật** thành **213 Challenge Vault** độc lập.

### Q6: Hiệu quả thực chứng của Active Learning & Warm-Start được chứng minh ra sao?
**Trả lời (`experiments/benchmarks/tables/benchmark_5models_quad_vault_summary.json`):**
Trên **213 Challenge Vault** (111 mẫu giả khó + 102 ảnh thật):
- **$M_0$ (Baseline $D_0$):** Fake Recall **11.71%** (Acc **53.99%**, ROC-AUC **0.8713**).
- **$M_1$ (Active R1):** Fake Recall tăng lên **70.27%** (Acc **83.10%**, ROC-AUC **0.9655**).
- **$M_2$ (Cold-Start Scratch):** Challenge Recall đạt **86.49%**, nhưng bị quên kiến thức trên Unseen RV5.1 (tụt xuống **56.55%**).
- **$M_2^*$ (Refined Warm-Start):** Challenge Recall đạt **91.89%** (**+80.18%** so với $M_0$, Acc **93.90%**, ROC-AUC **0.9875**), bảo toàn độ nhạy trên kho ngoài miền.
- **$M_3$ (Dual-Engine Champion):** Giữ nguyên **91.89%** Challenge Vault đồng thời nâng Master Frozen Recall lên **69.44%** và Unseen RV5.1 Recall lên **93.13%** (Acc **94.07%**, ROC-AUC **0.9828**).

---

## 4. Nhóm Câu Hỏi Về Benchmark Phổ Quát & Khả Năng Tổng Quát Hóa

### Q7: Bộ dò hoạt động ra sao khi đánh giá ngoại miền (RV5.1 6k & Universal 20.4k @ `512→504px LANCZOS`)?
**Trả lời:**
1. **Kho 6.000 ảnh RealisticVision v5.1 (`experiments/logs/legacy_6k_benchmark_results.json` @ `504px`):**
   - $M_0$: **75.20%** Recall (bỏ lọt 24.80%; nếu bị ép nén nhầm `224px` tụt xuống 2.00%).
   - $M_2^*$: **80.13%** Recall (1.192 ảnh bỏ lọt — chủ yếu ở `inpaint` 43.10% & `img2img` 16.40%).
   - $M_3$: **94.37%** Recall (chỉ còn 5.63% bypass; riêng `text2img` 100.00%, `img2img` 96.60%, `inpaint` 86.50%).
2. **Universal Deepfake Benchmark (`20.400` ảnh, `34` phương pháp, `8` họ công nghệ @ `504px`):**
   - $M_3$ dẫn đầu toàn diện ở ngưỡng mặc định $\tau=0.50$: Fake Recall **21.15%** (tăng **3.85×** so với mức `5.49%` khi bị nén `224px`), Accuracy **57.95%**, Precision **80.13%**, F1-Score **33.46%**, ROC-AUC **0.6101**, PR-AUC **0.6365**.
   - Theo họ công nghệ ($M_3$): **GAN Architectures 60.00%** (`StyleGAN2` 100.00%), **SD 2.1 54.00%**, **GAN Inversion 53.67%** (`e4e` 99.67%), **AutoEncoders 35.33%**, **DDIM 31.33%**, **DiT 17.67%** (`pixart` 60.67%).
   - **Đánh giá trung thực khoảng trống miền:** Vì $M_3$ chỉ học ảnh tĩnh Diffusion (SD 1.5 + RV5.1), Recall trên ảnh trích xuất từ video nén `256×256` thuộc họ `Face Swapping` (**5.37%**) và `Reenactment / Talking Head` (**6.55%**) vẫn thấp — đây là giới hạn Zero-Shot được báo cáo minh bạch.

### Q8: Trong ứng dụng thực tế (ví dụ kiểm duyệt eKYC), tỷ lệ báo động giả (FPR) được kiểm soát ra sao?
**Trả lời (`experiments/benchmarks/logs/benchmark_5models_predictions.npz`):**
- Tại ngưỡng mặc định $\tau = 0.50$, $M_3$ đạt **Real Specificity 94.75%** (FPR chung `5.25%`; riêng trên video nén H.264 FaceForensics++ C23 là **5.14%** = 85/1.655 khung hình).
- Nếu siết ngưỡng lên **$\tau = 0.85$**, FPR trên video FF++ C23 giảm một nửa xuống **2.54%**.
- Ở cực hạn bảo thủ **$\text{FPR} \le 1.0\%$** ($\tau = 0.971$), Zero-Shot Fake Recall của $M_3$ đạt **13.38%** (với $M_0$ là **13.07%** tại $\tau = 0.573$). Do đó trong thực chiến eKYC, mô hình phù hợp nhất cho bài toán **chặn ảnh giả Diffusion / GAN độ phân giải chuẩn (`≥ 512px`)** hơn là thay thế hoàn toàn bộ phát hiện video Face-Swap nén thấp.

---

## 5. Bảng Số Liệu Vàng Tra Cứu Nhanh (Verified Golden Key Facts)

| Chỉ Số / Hạng Mục | $M_0$ (Baseline) | $M_1$ (Active R1) | $M_2$ (Cold) | $M_2^*$ (Refined) | $M_3$ (Champion) | Nguồn Xác Thực |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Master Frozen Test Recall / AUC** | 49.07% / 0.8850 | 62.14% / 0.9161 | 52.63% / 0.8910 | 66.89% / 0.9177 | **69.44% / 0.9206** | `benchmark_5models_quad_vault_summary.json` |
| **Challenge Vault Recall / AUC** | 11.71% / 0.8713 | 70.27% / 0.9655 | 86.49% / 0.9937 | **91.89%** / 0.9875 | **91.89% / 0.9875** | `benchmark_5models_quad_vault_summary.json` |
| **Unseen RV5.1 Vault Recall / AUC** | 66.45% / 0.9162 | 72.04% / 0.9170 | 56.55% / 0.8804 | 71.88% / 0.9158 | **93.13% / 0.9828** | `benchmark_5models_quad_vault_summary.json` |
| **Legacy RV5.1 6k Recall (`504px`)** | 75.20% | 80.03% | 70.13% | 80.13% | **94.37%** | `legacy_6k_benchmark_results.json` |
| **Universal 20.4k Fake Recall (`504px`)** | 13.48% | 17.94% | 10.68% | 16.48% | **21.15%** | `benchmark_5models_global_summary.csv` |
| **Universal 20.4k Real Spec (`504px`)** | **98.87%** | 94.91% | 96.49% | 94.57% | 94.75% | `benchmark_5models_global_summary.csv` |
| **Universal 20.4k F1 / ROC-AUC** | 23.52% / **0.6431** | 29.17% / 0.5552 | 18.70% / 0.5597 | 27.04% / 0.5619 | **33.46%** / 0.6101 | `benchmark_5models_global_summary.csv` |
| **Universal SD 2.1 / GAN Recall** | 23.67% / 48.47% | 44.33% / 53.87% | 27.33% / 33.13% | 44.33% / 51.20% | **54.00% / 60.00%** | `benchmark_5models_family_summary.csv` |
