# Phần 1: Bản Đặc Tả Định Nghĩa Bài Toán & Yêu Cầu Nghiên Cứu (Problem Definition & Research Requirements Specification)

- **Motivation/Background**: Cung cấp nền tảng học thuật, lý giải cơ chế toán học, phân loại hình thái can thiệp sâu (manipulation taxonomy), thiết lập hợp đồng dữ liệu kỹ thuật và tiêu chí định lượng thành công cho đồ án nghiên cứu: "Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection" (Đề tài 11, bám sát học liệu Hugging Face Computer Vision Course Unit 10: "Stable diffusion for data generation").
- **Purpose**: Đóng vai trò là Phần 1 chính thức trong Báo cáo Kỹ thuật Tổng thể của Đồ án Môn học Thị giác Máy tính (UTH), làm cơ sở lý luận chặt chẽ giải thích nguyên nhân các bộ phát hiện Deepfake tiền nhiệm suy giảm hiệu năng trước Diffusion Models và chứng minh phương pháp luận sinh dữ liệu tổng hợp chủ động là giải pháp cốt lõi cho an ninh sinh trắc học khuôn mặt.
- **Overview Pipeline**: Bối cảnh & Khoảng trống học thuật -> Cơ chế khử nhiễu LDM vs. Up-sampling GAN trên miền tần số Fourier -> Phương pháp luận sinh dữ liệu tổng hợp (Hugging Face Unit 10) -> Phân loại 3 hình thái can thiệp khuôn mặt -> Hợp đồng dữ liệu & Manifest JSONL -> Phân hoạch dữ liệu chống rò rỉ danh tính -> Tiêu chí định lượng KPIs & Definition of Done.
- **Detailed Plan**: §1 Bối cảnh Học thuật & Khoảng Trống Nghiên Cứu; §2 Phân Tích Cơ Chế Miền Tần Số: Latent Diffusion vs. GAN Fingerprints; §3 Giải Pháp Sinh Dữ Liệu Tổng Hợp (Hugging Face Unit 10); §4 Phân Loại Hình Thái Can Thiệp Khuôn Mặt (Taxonomy); §5 Hợp Đồng Dữ Liệu & Bản Đặc Tả Kỹ Thuật Manifest; §6 Chiến Lược Phân Hoạch Chống Rò Rỉ Danh Tính (Anti-Identity Leakage); §7 Tiêu Chí Thành Công Định Lượng (KPIs) & Definition of Done.
- **References**: `docs/PURPOSE.md`, `docs/OVERVIEW.md`, `configs/dataset/deepfake_image.yaml`, `configs/generation/stable_diffusion.yaml`, `configs/detection/detector_baseline.yaml`, [Hugging Face CV Course Unit 10](https://huggingface.co/learn/computer-vision-course/unit10/datagen-diffusion), Frank et al. (ICML 2020), Durall et al. (CVPR 2020), Rombach et al. (CVPR 2022).
- **Created**: 2026-11-07T00:46:00+07:00
- **Last Updated**: 2026-11-07T00:46:00+07:00

---

## 1. Bối Cảnh Học Thuật & Khoảng Trống Nghiên Cứu (Academic Gap & Research Motivation)

### 1.1 Sự Dịch Chuyển Mô Hình Sinh Dữ Liệu (Generative Paradigm Shift)

Trong hơn một thập kỷ qua, lĩnh vực tổng hợp ảnh khuôn mặt nhân tạo (Facial Media Synthesis) bị chi phối chủ yếu bởi các kiến trúc **Generative Adversarial Networks (GANs)** như ProGAN (Karras et al., 2017), StyleGAN v1/v2/v3 (Karras et al., 2019-2021), StarGAN (Choi et al., 2018), cùng các kỹ thuật tráo đổi mặt truyền thống (FaceSwap, DeepFakes, Face2Face trong bộ benchmark FaceForensics++). 

Song hành với sự phát triển của GAN, các hệ thống phòng thủ Deepfake (Deepfake Detectors) dựa trên Mạng Nơ-ron Tích chập (CNN) và Vision Transformer (ViT) đã đạt được độ chính xác rất cao trong điều kiện cùng phân phối dữ liệu (in-distribution accuracy $\ge 98\%$). Hầu hết các bộ phát hiện này khai thác các dấu vết nhân tạo đặc thù do kiến trúc sinh của GAN để lại, chẳng hạn như vết sọc ô cờ (checkerboard artifacts), sự suy giảm phổ năng lượng cao tần, hoặc sự bất thường trong ma trận đồng xuất hiện mức xám (GLCM).

Tuy nhiên, sự bùng nổ của **Diffusion Models**, đặc biệt là **Latent Diffusion Models (LDM)** và đại diện tiêu biểu là **Stable Diffusion** (Rombach et al., CVPR 2022), đã tạo nên một bước ngoặt căn bản trong công nghệ sinh ảnh. Không giống như GAN sinh toàn bộ bức ảnh qua một lần lan truyền tiến (single forward pass), Stable Diffusion tiến hành khử nhiễu lặp lại (iterative stochastic denoising) trong không gian tiềm ẩn được nén bởi bộ mã hóa tự động tri giác (Perceptual Autoencoder / VAE).

```
+---------------------------------------------------------------------------------------------------+
|                        SỰ DỊCH CHUYỂN KHOẢNG TRỐNG HỌC THUẬT (ACADEMIC GAP)                       |
+---------------------------------------------------------------------------------------------------+
|  [THẾ HỆ CŨ: GAN GENERATOR]                                                                       |
|  Latent z ---> [Transposed Convolutions / Interpolation] ---> peccable spatial fingerprints        |
|                * Checkerboard patterns trong pixel domain                                         |
|                * Radial frequency spikes trong 2D Fourier spectrum                                |
|                * Bộ phát hiện CNN đạt AUC >= 98% (overfitting vào GAN fingerprint)                |
|                                                                                                   |
|  [THẾ HỆ MỚI: LATENT DIFFUSION MODELS]                                                            |
|  Noise z_T ---> [U-Net Reverse Denoising qua 30-50 steps] ---> [Continuous Latent z_0]           |
|                 * Phổ năng lượng tuân theo quy luật tự nhiên 1/f^alpha                            |
|                 * Triệt tiêu hoàn toàn sọc chu kỳ cấu trúc của GAN                                |
|                 * Suy giảm khái quát hóa: GAN Detectors tụt AUC xuống 50% - 65% (Near-Random)     |
+---------------------------------------------------------------------------------------------------+
```

### 1.2 Hiện Tượng Suy Giảm Khái Quát Hóa Xuyên Bộ Sinh (Cross-Generator Generalization Collapse)

Khi đối mặt với các khuôn mặt do Stable Diffusion tạo ra hoặc can thiệp, các mô hình phát hiện Deepfake huấn luyện trên GAN gặp phải hiện tượng **suy giảm khái quát hóa nghiêm trọng** (cross-generator generalization collapse):
1. **Mất Dấu Vết Đặc Trưng (Fingerprint Extinction)**: Các mạng nơ-ron học sâu thường tận dụng các lối tắt học tập (shortcut learning) bằng cách ghi nhớ các "vân tay bộ sinh" (generator fingerprints) cố định của GAN thay vì học bản chất của việc thao túng ngữ nghĩa khuôn mặt. Khi chuyển sang LDM, toàn bộ các vân tay tích chập này biến mất.
2. **Độ Chân Thực Tri Giác Vượt Trội (Perceptual Hyper-Realism)**: Stable Diffusion mô phỏng xuất sắc cấu trúc vi mô của da (lỗ chân lông, nếp nhăn nhỏ, ánh sáng khuếch tán bề mặt dưới da - subsurface scattering) và các đặc trưng phức tạp như sợi tóc hay ánh sáng mắt, đánh lừa các bộ phân loại hình thái truyền thống.
3. **Sự Thiếu Hụt Benchmark Huấn Luyện Phòng Thủ Chuyên Biệt**: Phần lớn các tập dữ liệu mở phục vụ nghiên cứu Deepfake (FaceForensics++, Celeb-DF, DFDC) hoàn toàn vắng bóng các biến thể can thiệp bằng Diffusion Models ở quy mô chuẩn hóa.

Do đó, một nhu cầu học thuật cấp thiết được đặt ra: **Xây dựng một quy trình sinh dữ liệu ảnh khuôn mặt tổng hợp bằng Stable Diffusion có kiểm soát, đi kèm hệ thống nhãn siêu dữ liệu (metadata) chính xác tuyệt đối, nhằm huấn luyện và đánh giá lại độ vững chắc (robustness) của các bộ phát hiện Deepfake thế hệ mới.**

---

## 2. Phân Tích Cơ Chế Miền Tần Số: Latent Diffusion Model vs. GAN Up-sampling

Để lý giải tận gốc tính ưu việt về mặt pháp y của Stable Diffusion, ta phân tích sự khác biệt bản chất giữa cơ chế sinh của GAN và LDM trên **Miền Tần Số 2 Chiều (2D Fourier Spectrum)**.

### 2.1 Cơ Chế Up-sampling & Dị Thường Phổ Tần Số Trong GAN

Kiến trúc Generator của GAN (điển hình như DCGAN, ProGAN, StyleGAN) ánh xạ một vector tiềm ẩn ngẫu nhiên $z \in \mathbb{R}^d$ lên không gian ảnh độ phân giải cao thông qua chuỗi các lớp tích chập chuyển vị (`ConvTranspose2d`) hoặc nội suy song tuyến (bilinear/nearest interpolation) kết hợp tích chập chuẩn:

$$I_{\text{GAN}} = G(z) = (f_K \circ f_{K-1} \circ \dots \circ f_1)(z)$$

Thao tác up-sampling này tạo ra hiện tượng **bất đối xứng chu kỳ không gian (spatial periodicity)**:
1. **Vết Sọc Ô Cờ (Checkerboard Artifacts)**: Khi kernel size không chia hết cho stride trong phép tích chập chuyển vị, sự chồng lấn không đồng đều của các tích vô hướng tạo ra các bước nhảy biên độ có tính chu kỳ trên lưới pixel.
2. **Đỉnh Dị Thường Trong Phổ Fourier (Spectral Peaks / Spikes)**:
   Biến đổi Fourier rời rạc 2 chiều (2D Discrete Fourier Transform - 2D DFT) của ảnh $I(x, y)$ kích thước $H \times W$ được tính bằng:
   $$\mathcal{F}\{I\}(u, v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} I(x, y) \cdot \exp\left( -j 2\pi \left( \frac{ux}{H} + \frac{vy}{W} \right) \right)$$
   Mật độ phổ năng lượng (Power Spectral Density - PSD) $P(u, v) = |\mathcal{F}\{I\}(u, v)|^2$ của ảnh GAN xuất hiện các đốm sáng bất thường hình chữ thập và các vòng tròn đồng tâm tại các tần số tương ứng với kích thước stride up-sampling (Frank et al., ICML 2020; Durall et al., CVPR 2020).
3. **Sự Thiếu Hụt Năng Lượng Cao Tần**: GAN gặp khó khăn trong việc khớp phân phối phổ ở các dải tần cực cao, dẫn đến sự suy giảm năng lượng phổ dốc đứng ở vùng rìa tần số so với ảnh chụp thực tế. Các mô hình như F3-Net (Frequency-aware Deepfake Detection) khai thác trực tiếp sự sai lệch này để bóc trần ảnh giả mạo.

```
       GAN: Peaky Fourier Spectrum                LDM: Smooth Natural Spectrum
             (Spurious Spikes)                           (1/f^alpha Law)
             
           v ^                                         v ^
             |       *                                   |       .  :  .
             |   *   |   *                               |    :  :  :  :
             | * * * O * * *                             |  . : : : O : : .
             |   *   |   *                               |    :  :  :  :
             |       *                                   |       .  :  .
             +--------------> u                          +--------------> u
      (Đỉnh tần số chu kỳ cố định)               (Phân rã liên tục tự nhiên)
```

### 2.2 Cơ Chế Khử Nhiễu Động Học Của Latent Diffusion Models (LDM)

Khác biệt hoàn toàn với GAN, Latent Diffusion Models (LDM) phân rã bài toán sinh ảnh thành hai giai đoạn độc lập:
1. **Nén Tri Giác (Perceptual Compression)**: Một mạng Autoencoder ($\mathcal{E}, \mathcal{D}$) đã được huấn luyện trước ánh xạ ảnh gốc $x \in \mathbb{R}^{H \times W \times 3}$ thành biểu diễn tiềm ẩn $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$, giảm chiều không gian theo tỉ lệ $f = H/h = 8$, loại bỏ nhiễu cao tần vô nghĩa mà vẫn bảo toàn cấu trúc ngữ nghĩa.
2. **Khuếch Tán & Khử Nhiễu Tiềm Ẩn (Latent Denoising)**:
   - **Forward Diffusion Process**: Thêm dần nhiễu Gauss qua $T$ bước:
     $$q(z_t | z_0) = \mathcal{N}\left( z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) \mathbf{I} \right)$$
   - **Reverse Denoising Process**: Một mạng U-Net $\epsilon_\theta(z_t, t, c)$ có cơ chế Cross-Attention được tối ưu hóa để dự đoán thành phần nhiễu tại mỗi bước thời gian $t$ dưới điều kiện văn bản/ngữ cảnh $c$:
     $$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0, \mathbf{I}), t, c} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, c) \right\|_2^2 \right]$$

#### Tính Tự Nhiên Tuyệt Đối Trong Miền Tần Số Của LDM:
- **Tích Lũy Chi Tiết Qua Chuỗi Markov**: Quá trình khử nhiễu ngược (reverse sampling qua DDIM / Euler Ancestral) mô phỏng động học Langevin stochastic. Cấu trúc ảnh lớn được hình thành ở các bước thời gian đầu ($t \approx T$), trong khi các chi tiết vi mô (da, tóc) được kết tinh dần dần ở các bước cuối ($t \to 0$). Thao tác này hoàn toàn không có bước kéo dãn đột ngột (abrupt spatial stretching).
- **Tuân Thủ Quy Luật Phổ Tự Nhiên ($1/f^\alpha$)**: Ảnh giải mã từ VAE $\hat{x} = \mathcal{D}(z_0)$ sở hữu mật độ phổ năng lượng trung bình hướng tâm (Radially Averaged Power Spectrum) tuân thủ gần như hoàn hảo định luật vật lý của ảnh quang học tự nhiên:
  $$\text{PSD}(f) \propto \frac{1}{f^\alpha}, \quad \text{với } \alpha \approx 2.0$$
- **Không Còn Sọc Chu Kỳ (No Spectral Spikes)**: Do không dựa vào các lớp `ConvTranspose2d` với stride lớn để mở rộng không gian trong suốt quá trình khử nhiễu, phổ Fourier của ảnh Stable Diffusion không xuất hiện các đỉnh năng lượng nhân tạo. Khi đưa ảnh này vào các bộ phát hiện dựa trên tần số (như F3-Net), bộ lọc không thể bắt được bất kỳ tín hiệu dị thường nào, khiến tỷ lệ **Bỏ Sót (False Negative Rate)** tăng vọt.

---

## 3. Giải Pháp Sinh Dữ Liệu Tổng Hợp (Hugging Face Course Unit 10)

Sự suy giảm hiệu năng của các hệ thống an ninh AI trước Diffusion Models bắt nguồn từ sự mất cân đối dữ liệu: ta dùng hệ thống phòng thủ thời đại GAN để chống lại kẻ tấn công thời đại Diffusion. Để giải quyết triệt để vấn đề này, đồ án áp dụng phương pháp luận từ **Hugging Face Computer Vision Course Unit 10: *Stable Diffusion for Data Generation***.

### 3.1 Bế Tắc Của Phương Pháp Thu Thập Dữ Liệu Truyền Thống
1. **Rào Cản Pháp Lý & Quyền Riêng Tư Sinh Trắc Học**: Quy định bảo vệ dữ liệu cá nhân (GDPR của Liên minh Châu Âu, CCPA, Luật An ninh Mạng) cấm nghiêm ngặt việc thu thập, lưu trữ và tái phân phối hàng loạt ảnh khuôn mặt người thật trên internet mà không có sự đồng thuận rõ ràng.
2. **Chi Phí Gán Nhãn Thủ Công Không Tưởng**: Để huấn luyện bộ phát hiện phân vùng can thiệp (manipulation localization), kỹ sư cần nhãn mặt nạ điểm ảnh (pixel-level mask) xác định chính xác khu vực bị sửa đổi. Việc vẽ mask thủ công trên hàng vạn ảnh là bất khả thi về mặt tài chính và thời gian.
3. **Sự Thiên Lệch Nhân Khẩu Học (Demographic Bias)**: Dữ liệu khuôn mặt thực tế cào từ mạng xã hội bị lệch nặng nề về độ tuổi trẻ, giới tính nữ và chủng tộc da trắng, dẫn đến hiện tượng suy giảm công bằng sinh trắc học (biometric disparity).

### 3.2 Sinh Dữ Liệu Tổng Hợp Chủ Động (Proactive Synthetic Data Generation)
Thay vì chờ đợi các chiến dịch Deepfake độc hại xuất hiện trên thực tế rồi mới thu thập mẫu phòng thủ một cách bị động, phương pháp sinh dữ liệu tổng hợp đóng vai trò là một **Nhà Máy Sản Xuất Dữ Liệu Tự Động (Automated Data Factory)**:

```
+---------------------------------------------------------------------------------------------------+
|                        MÔ HÌNH NHÀ MÁY SINH DỮ LIỆU TỔNG HỢP CHỦ ĐỘNG                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Điều Khiển Ngữ Nghĩa]   +   [Điều Khiển Không Gian]   +   [Kiểm Soát Độ Mạnh Biến Đổi]          |
|  Prompt: Tuổi, Biểu Cảm       Landmark Mask: Mắt, Miệng     Denoising Strength S in [0.0, 1.0]    |
|               |                           |                              |                        |
|               +---------------------------+------------------------------+                        |
|                                           |                                                       |
|                                           v                                                       |
|                     [STABLE DIFFUSION GENERATION FACTORY]                                         |
|                     * Text-to-Image (T2I): Khuôn mặt mới 100%                                     |
|                     * Inpainting: Can thiệp biểu cảm / thuộc tính                                 |
|                     * Img2Img: Biến đổi danh tính / kết cấu da                                    |
|                                           |                                                       |
|                                           v                                                       |
|                     [CỔNG KIỂM DUYỆT CHẤT LƯỢNG ĐA PHƯƠNG THỨC]                                   |
|                     * Độ sắc nét Laplacian >= 50.0                                                |
|                     * Tương đồng ngữ nghĩa CLIP Score >= 0.30                                     |
|                     * Giữ lại cấu trúc khuôn mặt Face Retention >= 95%                            |
|                                           |                                                       |
|                                           v                                                       |
|                     [BỘ DỮ LIỆU NHÃN SẠCH & MANIFEST JSONL CHUẨN HÓA]                             |
|                     (Sẵn sàng huấn luyện Downstream Deepfake Detectors)                           |
+---------------------------------------------------------------------------------------------------+
```

#### Lợi Ích Vượt Trội Của Phương Pháp Này:
- **Tự Động Xuất Nhãn Đầy Đủ (Ground-Truth by Design)**: Mỗi ảnh sinh ra đều có siêu dữ liệu tuyệt đối chính xác: tọa độ bounding box, prompt điều kiện, mặt nạ can thiệp pixel-level, mã nhận diện ảnh gốc, và thông số khử nhiễu mà không tốn 1 giờ lao động thủ công.
- **Tuân Thủ Tuyệt Đối Quyền Riêng Tư (Privacy-by-Design)**: Các khuôn mặt sinh từ Text-to-Image là thực thể hoàn toàn nhân tạo (non-existent humans), loại bỏ 100% nguy cơ xâm phạm quyền riêng tư của cá nhân thực.
- **Chủ Động Điều Khiển Độ Khó Của Mẫu Thử Nghiệm**: Bằng cách tinh chỉnh bước khử nhiễu ($N$), hệ số dẫn dắt văn bản ($s_{\text{cfg}}$), và độ mạnh khử nhiễu ($S_{\text{strength}}$), hệ thống có thể tạo ra các mẫu Deepfake từ mức độ biến dạng rõ ràng đến mức độ tinh vi mắt thường không thể phân biệt, giúp xây dựng các bộ dữ liệu kiểm thử có độ phân hóa cao.

---

## 4. Phân Loại Hình Thái Can Thiệp Khuôn Mặt (Facial Manipulation Taxonomy)

Đồ án phân loại và triển khai 3 hình thái can thiệp khuôn mặt chủ chốt tương ứng với 3 vector tấn công Deepfake nguy hiểm nhất hiện nay bằng Stable Diffusion:

| Nhóm Can Thiệp | Kỹ Thuật Sinh | Điều Kiện Đầu Vào | Mô Hình Mối Đe Dọa (Threat Model) | Dấu Vết Pháp Y (Forensic Signature) |
| :--- | :--- | :--- | :--- | :--- |
| **Nhóm 1: T2I** | Text-to-Image Entire Synthesis | Khởi tạo từ nhiễu trắng $z_T \sim \mathcal{N}(0, \mathbf{I})$ + Prompt văn bản | Danh tính ảo (phantom personas), tài khoản giả mạo KYC, bot lừa đảo viễn thông | Bất đối xứng vi mô trong phản xạ đồng tử mắt (corneal reflection), hoa văn bokeh nền mờ phi tự nhiên |
| **Nhóm 2: Inpaint** | Facial Inpainting Local Editing | Ảnh thật $I_{\text{real}}$ + Mặt nạ landmark $M \in \{0, 1\}$ + Prompt | Sửa đổi biểu cảm nhân chứng, nhép môi giả mạo, thêm/bớt đặc điểm nhận dạng pháp lý | Ranh giới chuyển tiếp (stitching seam), sự lệch pha gradient màu sắc và độ phân giải lỗ chân lông |
| **Nhóm 3: Img2Img** | Img2Img Identity & Skin Morphing | Ảnh thật $I_{\text{real}}$ + Nhiễu bước $t_{\text{start}}$ ($S \in [0.4, 0.75]$) + Prompt | Rửa danh tính (identity laundering), giả mạo nhận diện sinh trắc học, hoán đổi sắc tộc/độ tuổi | Bề mặt da bị làm mịn bất thường (loss of epidermal micro-texture), biến dạng nhẹ tỷ lệ nhân trắc học |

### 4.1 Nhóm 1: Text-to-Image (T2I) — Sinh Toàn Bộ Khuôn Mặt Nhân Tạo

- **Định Nghĩa Toán Học**: 
  Khởi tạo vector tiềm ẩn $z_T \sim \mathcal{N}(0, \mathbf{I})$ tại bước $t = T$. Dưới sự dẫn dắt của embedding văn bản $c = \tau(y)$ (với $y$ là prompt như *"a photorealistic headshot portrait of a person, studio lighting, high resolution, 8k"*), mạng U-Net thực hiện quá trình lấy mẫu ngược qua $N$ bước khử nhiễu để thu về $z_0$:
  $$z_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( z_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \tilde{\epsilon}_\theta(z_t, t, c) \right) + \sigma_t \mathbf{\epsilon}$$
  Trong đó $\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + s \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))$ là ước lượng nhiễu sau khi áp dụng kỹ thuật Classifier-Free Guidance (CFG) với hệ số $s \ge 1.0$. Cuối cùng, ảnh không gian được giải mã bởi VAE decoder: $I_{\text{fake}} = \mathcal{D}(z_0)$.
- **Nguy Cơ An Ninh**: Tạo dựng danh tính ảo hoàn toàn không có thực trong cơ sở dữ liệu dân cư, qua mặt các hệ thống xác thực khuôn mặt từ xa (eKYC) chưa tích hợp kiểm tra độ sống (liveness detection).

### 4.2 Nhóm 2: Facial Inpainting — Chỉnh Sửa Cục Bộ Thuộc Tính & Biểu Cảm

- **Định Nghĩa Toán Học**:
  Cho ảnh khuôn mặt gốc $x_{\text{real}}$ và mặt nạ nhị phân $M \in \{0, 1\}^{H \times W}$ bao phủ các vùng giải phẫu cụ thể (mắt, miệng, mũi) thu được từ thuật toán trích xuất facial landmark. Mặt nạ được downsample về không gian tiềm ẩn $M_{\text{latent}}$. Tại mỗi bước khử nhiễu $t$, biểu diễn tiềm ẩn được nội suy kết hợp:
  $$z_{t-1}^{\text{combined}} = (1 - M_{\text{latent}}) \odot z_{t-1}^{\text{real}} + M_{\text{latent}} \odot z_{t-1}^{\text{synth}}$$
  Trong đó $z_{t-1}^{\text{real}}$ là thành phần tiềm ẩn của ảnh gốc đã được thêm nhiễu tại bước $t-1$, còn $z_{t-1}^{\text{synth}}$ là thành phần khử nhiễu sinh ra từ U-Net dưới điều kiện prompt thay đổi biểu cảm (ví dụ: *"a realistic smiling mouth with natural teeth and lips"*).
- **Nguy Cơ An Ninh**: Làm sai lệch ngữ cảnh cảm xúc của đối tượng trong các video thẩm vấn, tạo chứng cứ giả mạo, hoặc ghép các phát ngôn phản cảm vào khuôn mặt nhân vật của công chúng.

### 4.3 Nhóm 3: Img2Img — Biến Đổi Danh Tính & Kết Cấu Biểu Bì

- **Định Nghĩa Toán Học**:
  Mã hóa ảnh thực $x_{\text{real}}$ thành latent $z_0 = \mathcal{E}(x_{\text{real}})$. Thêm nhiễu trực tiếp tới một bước thời gian trung gian $t_{\text{start}} = \lfloor S \cdot T \rfloor$, với $S \in [0.0, 1.0]$ là độ mạnh khử nhiễu (Denoising Strength):
  $$z_{t_{\text{start}}} = \sqrt{\bar{\alpha}_{t_{\text{start}}}} z_0 + \sqrt{1 - \bar{\alpha}_{t_{\text{start}}}} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
  Quá trình khử nhiễu ngược chỉ chạy từ $t_{\text{start}}$ đếm ngược về $0$. Nếu $S$ nhỏ ($S \le 0.3$), cấu trúc ảnh gốc gần như được giữ nguyên; nếu $S$ lớn ($S \ge 0.7$), ảnh sẽ biến đổi danh tính mạnh mẽ nhưng vẫn giữ lại bố cục tư thế và ánh sáng tổng thể.
- **Nguy Cơ An Ninh**: "Rửa danh tính" (Identity Laundering), tạo ra biến thể khuôn mặt đánh lừa các mô hình trích xuất đặc trưng sinh trắc học (như ArcFace / CosFace) khiến khoảng cách cosine giữa vector khuôn mặt thật và giả vượt qua ngưỡng chấp nhận xác thực.

---

## 5. Hợp Đồng Dữ Liệu & Bản Đặc Tả Kỹ Thuật Manifest

Hệ thống tuân thủ nghiêm ngặt các quy chuẩn kỹ thuật nhằm đảm bảo khả năng tích hợp thông suốt giữa các module sinh dữ liệu, sàng lọc chất lượng và huấn luyện detector.

### 5.1 Cấu Trúc Thư Mục Dữ Liệu Chuẩn Hóa

```
data/
|-- raw/
|   `-- faces/                       # Nguồn ảnh chân dung thực tế (CelebA-HQ / FFHQ)
|       |-- real_000001.png
|       `-- real_000002.png
`-- processed/
    |-- manifest.jsonl               # File chỉ mục JSON Lines chuẩn hóa toàn bộ tập dữ liệu
    `-- images/
        |-- real/                    # Ảnh thật đã chuẩn hóa kích thước (Nhãn 0)
        |   |-- real_000001.png
        |   `-- real_000002.png
        `-- fake/                    # Ảnh giả mạo tổng hợp từ Stable Diffusion (Nhãn 1)
            |-- sd_t2i_000001.png
            |-- sd_inp_000002.png
            `-- sd_i2i_000003.png
```

### 5.2 Hợp Đồng Định Dạng Tensor & Không Gian Màu

Nhằm loại trừ triệt để lỗi đảo ngược kênh màu (RGB vs. BGR) và sai lệch chiều ma trận, hệ thống thiết lập các bất biến kỹ thuật:

1. **Tensor Shape Contracts**:
   - PyTorch Input Batch: Luôn có shape `(B, C, H, W)` với kiểu dữ liệu `torch.float32`, nằm trong dải $[0.0, 1.0]$.
   - Single Image Tensor: Shape `(C, H, W)` với $C = 3, H = 256, W = 256$ (hoặc $512 \times 512$).
   - OpenCV / NumPy Array: Shape `(H, W, C)` hoặc `(H, W)`.
2. **Quy Chuẩn Không Gian Màu (Color Space Invariant)**:
   - OpenCV đọc và ghi mặc định ở hệ **BGR**.
   - Torchvision, Matplotlib, Diffusers và Transformers mặc định ở hệ **RGB**.
   - Mọi thao tác tải ảnh từ đĩa bằng OpenCV phải được chuyển đổi tường minh bằng `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` trước khi chuyển thành Tensor PyTorch.
3. **Quy Chuẩn Bounding Box**:
   - Bắt buộc tuân thủ định dạng pixel tuyệt đối Pascal VOC: `xyxy` = `[x_min, y_min, x_max, y_max]`.
4. **Chuẩn Hóa Phân Phối Chuẩn (ImageNet Normalization)**:
   - Mean: $\mu = [0.485, 0.456, 0.406]$
   - Standard Deviation: $\sigma = [0.229, 0.224, 0.225]$

### 5.3 Lược Đồ Manifest JSON Lines (JSONL Schema)

File chỉ mục trung tâm `data/processed/manifest.jsonl` bao gồm các dòng JSON độc lập, mỗi dòng tương ứng với một mẫu ảnh theo chuẩn `ImageMetadata` (`src/data/image_schema.py`):

```json
{
  "image_id": "sd_inp_000104",
  "label": 1,
  "manipulation_type": "sd_inpaint",
  "prompt": "a realistic smiling mouth with natural teeth and lips",
  "source_real_id": "real_000042",
  "file_path": "data/processed/images/fake/sd_inp_000104.png",
  "bbox": [58, 62, 198, 204]
}
```

```json
{
  "image_id": "real_000042",
  "label": 0,
  "manipulation_type": "real",
  "prompt": null,
  "source_real_id": null,
  "file_path": "data/processed/images/real/real_000042.png",
  "bbox": [56, 60, 200, 206]
}
```

#### Bảng Đặc Tả Trường Dữ Liệu:

| Tên Trường | Kiểu Dữ Liệu | Bắt Buộc | Mô Tả Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :---: | :--- |
| `image_id` | `str` | Có | Mã định danh duy nhất của bức ảnh trong tập dữ liệu (khóa chính). |
| `label` | `int` | Có | Nhãn phân loại nhị phân ground-truth: `0` cho Ảnh Thật (Real), `1` cho Ảnh Giả (Fake). |
| `manipulation_type`| `str` | Có | Phương pháp sinh: `"real"`, `"sd_text2img"`, `"sd_inpaint"`, hoặc `"sd_img2img"`. |
| `prompt` | `str \| null` | Không | Câu lệnh văn bản điều khiển sinh ảnh (bằng `null` đối với ảnh thật). |
| `source_real_id` | `str \| null` | Không | Mã định danh của ảnh thật nguyên bản nếu mẫu này sinh từ Inpainting hoặc Img2Img. |
| `file_path` | `str` | Có | Đường dẫn tương đối từ gốc workspace tới tệp tin ảnh trên đĩa cứng. |
| `bbox` | `List[int] \| null`| Không | Tọa độ hộp bao khuôn mặt theo chuẩn `[x1, y1, x2, y2]`. |

---

## 6. Chiến Lược Phân Hoạch Chống Rò Rỉ Danh Tính (Anti-Identity Leakage Partitioning)

Trong các bài toán thị giác máy tính về khuôn mặt, **Rò rỉ danh tính (Identity Leakage)** là một sai lầm phương pháp luận phổ biến dẫn đến đánh giá hiệu năng bị thổi phồng quá mức (over-optimistic performance).

### 6.1 Nguy Cơ Rò Rỉ Danh Tính Giữa Train/Val/Test
Nếu ảnh thật của Đối tượng A nằm trong tập huấn luyện (Train), trong khi ảnh can thiệp Inpainting của chính Đối tượng A lại nằm trong tập kiểm thử (Test), mô hình phân loại sẽ học được các đặc trưng danh tính (như cấu trúc xương mặt, vị trí nốt ruồi, màu mắt) thay vì học cách phát hiện vết cắt biên hoặc dấu vết khử nhiễu. Khi đó, mô hình sẽ phân loại sai hoặc ghi nhớ danh tính (identity memorization).

### 6.2 Nguyên Tắc Phân Hoạch Bất Biến (Identity-Disjoint Invariant)

```
[Danh Sách Ảnh Thực Tế Gốc (Unique Identities)]
               |
               v
    +--------------------------------------------------------------+
    | Phân Hoạch Cụm Danh Tính Theo Tỷ Lệ (Seed=42 Cố Định)        |
    | - Cụm Train: 70% Số Danh Tính Độc Lập                        |
    | - Cụm Validation: 15% Số Danh Tính Độc Lập                   |
    | - Cụm Test: 15% Số Danh Tính Độc Lập                         |
    +--------------------------------------------------------------+
               |
               |-----> [Tập Train]: Tất cả ảnh Real(ID_A) + Inpaint(ID_A) + Img2Img(ID_A)
               |-----> [Tập Val]:   Tất cả ảnh Real(ID_B) + Inpaint(ID_B) + Img2Img(ID_B)
               `-----> [Tập Test]:  Tất cả ảnh Real(ID_C) + Inpaint(ID_C) + Img2Img(ID_C)
               
[Tập Ảnh Sinh Text-to-Image (Không Có Danh Tính Gốc)]:
Phân bổ ngẫu nhiên phân tầng (Stratified Random Assignment) bảo đảm tỷ lệ 70/15/15.
```

1. **Quy Tắc Phân Cụm Danh Tính Gốc (Source Identity Grouping)**:
   $$\forall x \text{ sao cho } \text{source\_real\_id}(x) = s \implies \text{Split}(x) = \text{Split}(s)$$
   Tất cả các ảnh phái sinh từ một ảnh gốc bắt buộc phải cùng nằm trong một phân hoạch với ảnh gốc đó.
2. **Cân Bằng Phân Phối Nhãn (Class Balance)**:
   - Trong mỗi tập phân hoạch (Train, Val, Test), tỷ lệ giữa ảnh Thật và ảnh Giả luôn được giữ ở mức $1:1$ ($50\%$ Real, $50\%$ Fake).
   - Trong tập ảnh Giả, số lượng được chia đều cho 3 hình thái can thiệp ($33.3\%$ T2I, $33.3\%$ Inpainting, $33.3\%$ Img2Img) nhằm tránh thiên lệch bộ phân loại về một kỹ thuật sinh cụ thể.

---

## 7. Tiêu Chí Thành Công Định Lượng (KPIs) & Definition of Done (DoD)

Nhằm đảm bảo đồ án đạt chất lượng học thuật cao nhất, các chỉ số đo lường hiệu năng định lượng và tiêu chuẩn hoàn thành kỹ thuật (DoD) được thiết lập với các ngưỡng chặn nghiêm ngặt:

### 7.1 Bảng Chỉ Số Hiệu Năng Định Lượng (Quantitative KPIs)

| Chỉ Số Đánh Giá | Ngưỡng Tối Thiểu (Target Threshold) | Công Thức & Giao Thức Đánh Giá | Ý Nghĩa Học Thuật & Kiểm Soát Chất Lượng |
| :--- | :---: | :--- | :--- |
| **Fréchet Inception Distance (FID)** | $\le \mathbf{20.0}$ | $\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)$<br>Trích xuất vector đặc trưng Inception-v3 pool3 (2048 chiều) giữa tập ảnh sinh và tập ảnh thật CelebA-HQ/FFHQ. | Đo lường độ chân thực và tính đa dạng của phân phối ảnh sinh; đảm bảo ảnh không bị mode collapse hoặc biến dạng dị thường. |
| **CLIP Score (Image-Text Alignment)** | $\ge \mathbf{0.30}$ | $S_{\text{CLIP}}(I, y) = \max\left( \cos\left(\mathbf{e}_I, \mathbf{e}_T\right), 0 \right)$<br>Sử dụng mô hình `openai/clip-vit-base-patch32` với $\mathbf{e}_I = \mathcal{E}_{\text{img}}(I)$ và $\mathbf{e}_T = \mathcal{E}_{\text{text}}(y)$. | Đảm bảo ảnh sinh bám sát chính xác câu lệnh prompt điều khiển ngữ nghĩa (ví dụ: mô tả biểu cảm, tuổi tác, ánh sáng). |
| **Laplacian Sharpness Score** | $\ge \mathbf{50.0}$ | $\text{Var}(\nabla^2 I) = \frac{1}{HW}\sum_{x,y} \left( L(x, y) - \bar{L} \right)^2$<br>Toán tử vi phân bậc hai Laplacian trên ảnh xám. | Tự động loại bỏ các ảnh sinh bị nhòe mờ (blur), sương mù (hazy) hoặc lỗi giải mã của VAE decoder. |
| **Face Detection Retention Rate** | $\ge \mathbf{95.0\%}$ | $R_{\text{face}} = \frac{N_{\text{detected}}}{N_{\text{total}}} \times 100\%$<br>Độ tin cậy phát hiện khuôn mặt $\ge 0.85$ từ bộ phát hiện khuôn mặt Haar Cascade / MTCNN. | Đảm bảo ảnh sinh giữ được cấu trúc giải phẫu học khuôn mặt người bình thường, không bị biến dạng kỳ quái (disfigured faces). |
| **Downstream Detector ROC-AUC** | $\ge \mathbf{92.0\%}$ | Diện tích dưới đường cong ROC (Area Under the Receiver Operating Characteristic Curve) trên tập kiểm thử độc lập (held-out test set). | Chứng minh tập dữ liệu sinh ra có giá trị huấn luyện cao, giúp mô hình phân loại nhị phân đạt khả năng phân biệt vượt trội. |
| **Downstream Detector Accuracy** | $\ge \mathbf{85.0\%}$ | Độ chính xác phân loại nhị phân ($0=\text{Real}, 1=\text{Fake}$) tại ngưỡng quyết định tối ưu $\tau = 0.5$ trên tập test cân bằng. | Khẳng định tính hiệu quả và độ tin cậy của mô hình phòng thủ khi triển khai thực tế. |

### 7.2 Định Nghĩa Hoàn Thành Kỹ Thuật (Definition of Done - DoD)

Mọi mã nguồn và hạng mục chuyển giao của đồ án chỉ được phê duyệt khi đáp ứng đầy đủ 5 tiêu chuẩn:

1. **Tuân Thủ Thiết Kế Phần Mềm Chuẩn Mực (Clean Architecture)**:
   - Toàn bộ thuật toán được đóng gói thành các module Python có type annotation đầy đủ (PEP 484) trong thư mục `src/`.
   - Tuyệt đối không dùng vòng lặp thô (raw loop) duyệt từng điểm ảnh; $100\%$ tính toán được vector hóa trên PyTorch và NumPy.
   - Tuyệt đối không dùng lệnh `print()` tùy tiện trong module thư viện; toàn bộ thông tin được ghi vết qua `src.common.logger.get_logger(__name__)`.
2. **Khả Năng Tái Lập Xác Thực (Strict Deterministic Reproducibility)**:
   - Toàn bộ quy trình sinh số ngẫu nhiên được neo bằng hạt giống `src.common.seed.seed_everything(seed=42)`.
   - Kết quả phân hoạch dữ liệu và trọng số mô hình khi chạy lại cho kết quả hoàn toàn trùng khớp.
3. **Chế Độ Kiểm Thử Giả Lập Ngoại Tuyến (Deterministic Mocking Mode for CI/CD)**:
   - Động cơ sinh ảnh `StableDiffusionEngine` có cờ kiểm thử `is_mock=True` cho phép tạo tensor hợp lệ ngay trong môi trường kiểm thử tự động mà không cần tải hàng gigabyte trọng số từ Hugging Face Hub.
4. **Bộ Kiểm Thử Đạt Chuẩn (Automated Testing Suite)**:
   - Các bài kiểm tra nhanh (`make test-smoke`) hoàn thành trong $< 5$ giây.
   - Bộ unit test kiểm tra toàn bộ luồng dữ liệu (`tests/test_image_data.py`, `tests/test_stable_diffusion.py`) vượt qua $100\%$ không lỗi.
5. **Tài Liệu Học Thuật Khép Kín (Documentation Integrity)**:
   - Toàn bộ tệp markdown tuân thủ quy tắc đặt tên chữ hoa, có phần đầu 7 trường thông tin chuẩn mực.
   - Tuyệt đối không sử dụng đường dẫn tuyệt đối của máy cá nhân; sử dụng liên kết tương đối chuẩn mực.
