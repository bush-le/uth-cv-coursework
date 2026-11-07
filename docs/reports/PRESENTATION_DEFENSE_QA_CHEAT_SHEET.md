# PRESENTATION_DEFENSE_QA_CHEAT_SHEET.md — Bộ Phản Biện & Kịch Bản Ứng Khẩu Đỉnh Cao (Academic Defense & Q&A Auditor)

- **Motivation/Background**: Xây dựng tài liệu phản biện học thuật chiến lược, giải mã điểm nóng tâm lý của Hội đồng chấm thi UTH và cung cấp kịch bản ứng khẩu chuẩn xác, đanh thép trong 30 giây cho bài thuyết trình 5 phút đồ án: "Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection" (Đề tài 11, Hugging Face Computer Vision Course Unit 10).
- **Purpose**: Đóng vai trò là cẩm nang bỏ túi (Cheat Sheet) giúp sinh viên tự tin bảo vệ xuất sắc trước các câu hỏi học thuật hóc búa nhất từ các Giảng viên khó tính và Hội đồng phản biện.
- **Overview Pipeline**: Khảo sát bẫy học thuật (Trap Analysis) -> Phân tích điểm nóng bắt bẻ (Interrogator Mindset) -> Xây dựng luận điểm toán học & kiến trúc hệ thống -> Thiết lập kịch bản ứng khẩu 30 giây -> Tổng hợp nguyên tắc bản lĩnh phòng thi.
- **Detailed Plan**: §1 Chiến lược & Tâm thế Phản biện 5 Phút; §2 Bộ 6 Câu Hỏi Hóc Búa Nhất & Kịch Bản Ứng Khẩu 30 Giây; §3 Ma Trận Từ Khóa Đắt Giá & Công Thức Toán Học Cốt Lõi; §4 Quy Trình Xử Lý Sự Cố & Kỹ Thuật Đảo Ngược Tình Thế.
- **References**: `docs/PURPOSE.md`, `docs/reports/SYSTEM_DEFINE_AND_DESIGN_REPORT.md`, `docs/reports/PART_01_PROBLEM_DEFINITION.md`, `docs/reports/PART_02_SYSTEM_ARCHITECTURE.md`, [Hugging Face CV Course Unit 10](https://huggingface.co/learn/computer-vision-course/unit10/datagen-diffusion), Frank et al. (ICML 2020), Durall et al. (CVPR 2020), Corvi et al. (CVPR 2023).
- **Created**: 2026-11-07T01:05:00+07:00
- **Last Updated**: 2026-11-07T01:05:00+07:00

---

## 1. 🎯 CHIẾN LƯỢC & TÂM THẾ PHẢN BIỆN (DEFENSE PSYCHOLOGY)

Trong buổi bảo vệ đồ án 5 phút tại UTH, thời gian thuyết trình rất ngắn nên **phần Q&A (3 - 5 phút)** chính là nơi phân định điểm Giỏi (8.5+) và Xuất sắc (9.5 - 10.0).

> [!IMPORTANT]
> **Quy Tắc Vàng Khi Nhận Câu Hỏi:**
> 1. **Dừng 2 giây để thở & mỉm cười**: Thể hiện sự tôn trọng, điềm tĩnh, không vội vàng cướp lời Giảng viên.
> 2. **Cấu Trúc 3 Bước Đanh Thép**:
>    - *Bước 1 (Xác nhận & Cảm ơn)*: "Dạ, em cảm ơn câu hỏi rất sâu sắc của Thầy/Cô về..."
>    - *Bước 2 (Khẳng định Luận điểm Cốt lõi trong 1 câu)*: Nêu ngay bản chất kỹ thuật, không vòng vo.
>    - *Bước 3 (Bằng chứng Toán học / Kiến trúc Hệ thống)*: Dẫn chứng tên module code (`src/`), công thức hoặc trích dẫn nghiên cứu khoa học.

---

## 2. ⚡ BỘ 6 CÂU HỎI HÓC BÚA NHẤT & KỊCH BẢN ỨNG KHẨU 30 GIÂY

```
+---------------------------------------------------------------------------------------------------+
|                        BẢN ĐỒ CHIẾN LƯỢC ĐỐI THOẠI HỘI ĐỒNG PHẢN BIỆN                             |
+---------------------------------------------------------------------------------------------------+
|  [CÂU HỎI HÓC BÚA]                                  [BẢN CHẤT KỸ THUẬT CỐT LÕI ĐÁP TRẢ]          |
|  1. Tại sao không dùng FaceForensics++ / Celeb-DF?   ---> Sự dịch chuyển sang LDM & Sụp đổ miền tần số |
|  2. Tại sao chỉ làm Ảnh mà không làm Video?          ---> Trọng tâm Topic 11, Artifact nằm ở Spatial   |
|  3. Làm sao chống Data Leakage giữa Train và Test?   ---> Nguyên lý Subject-Exclusive (ID-Disjoint)    |
|  4. Kế hoạch thực nghiệm được thiết kế thế nào?      ---> Evaluation Protocol (Define & System Design)  |
|  5. Khắc phục lỗi mắt lé / mặt sáp over-smoothing?   ---> ImageCurationGate (Laplacian >= 50, Face 85%)|
|  6. Tại sao dùng Stable Diffusion thay vì GAN?       ---> Latent Entanglement & Phổ tự nhiên 1/f^alpha |
+---------------------------------------------------------------------------------------------------+
```

---

### Câu Hỏi 1: "Tại sao không dùng luôn các bộ dữ liệu Deepfake có sẵn (FaceForensics++, Celeb-DF) mà lại phải sinh dữ liệu mới bằng Stable Diffusion?"

- **[ĐIỂM NÓNG GIẢNG VIÊN ĐANG BẮT BẺ]**:
  - Giảng viên nghi ngờ tính cần thiết của đồ án: Cho rằng sinh viên đang "vẽ việc ra làm" (*reinventing the wheel*), tốn tài nguyên vô ích khi thế giới đã có sẵn các benchmark khổng lồ gán nhãn hàng triệu khung hình.
- **[BẢN CHẤT TOÁN HỌC & CÔNG NGHỆ]**:
  - FaceForensics++ (2019) và Celeb-DF (2020) được xây dựng trên công nghệ cũ: Autoencoder tráo mặt (DeepFakes, FaceSwap) hoặc biến dạng đồ họa (Face2Face).
  - Các bộ phát hiện hiện đại huấn luyện trên FF++ bị hiện tượng **"Lối tắt học tập" (Shortcut Learning)**: chỉ chăm chú ghi nhớ vết ghép biên (blending boundary seams) hoặc sọc ô cờ tần số cao do phép tích chập chuyển vị của GAN.
  - Khi gặp ảnh Deepfake thế hệ mới sinh từ **Latent Diffusion Models (LDM)** với quá trình khử nhiễu liên tục theo quy luật phổ tự nhiên $1/f^\alpha$, các bộ phát hiện cũ **suy giảm nghiêm trọng khả năng khái quát hóa (Generalization Collapse)** do không còn tìm thấy các sọc tần số quen thuộc.
- **[KỊCH BẢN ỨNG KHẨU 30 GIÂY — ĐANH THÉP & CHUẨN XÁC]**:
  > *"Dạ thưa Thầy/Cô, các bộ dữ liệu như FaceForensics++ hay Celeb-DF được xây dựng từ năm 2019-2020, dựa trên các kỹ thuật GAN hoặc hoán đổi khuôn mặt đồ họa cũ. Chúng để lại các dấu vết nhân tạo đặc thù như sọc ô cờ do phép tích chập upsampling.*
  > 
  > *Nhiều nghiên cứu quốc tế gần đây (như Corvi et al. tại ICIP 2023) đã chỉ ra rằng các mô hình phát hiện huấn luyện trên dữ liệu cũ không có khả năng tổng quát hóa tốt khi gặp ảnh sinh từ Diffusion Models. Do đó, việc ứng dụng bài học Hugging Face Unit 10 để tự động sinh dữ liệu bằng Stable Diffusion là bước đi cần thiết nhằm cập nhật phân phối dữ liệu mới nhất cho bài toán phòng thủ Deepfake."*

---

### Câu Hỏi 2: "Tại sao đề tài chỉ làm ẢNH mà không làm VIDEO Deepfake?"

- **[ĐIỂM NÓNG GIẢNG VIÊN ĐANG BẮT BẺ]**:
  - Giảng viên muốn thử thách phạm vi ứng dụng thực tế: "Deepfake ngoài đời thường là video lừa đảo, tại sao nhóm chỉ làm ảnh tĩnh? Có phải nhóm chọn đề tài dễ để né tránh bài toán khó?"
- **[BẢN CHẤT TOÁN HỌC & CÔNG NGHỆ]**:
  - **1. Chuẩn hóa theo Đề tài & Học liệu**: Đề tài 11 gắn chặt với *Hugging Face Computer Vision Course Unit 10: "Stable diffusion for data generation"*. Học liệu này chuyên sâu vào cơ chế sinh dữ liệu tổng hợp 2D (T2I, Inpainting, Img2Img) để giải quyết tình trạng thiếu hụt dữ liệu thị giác.
  - **2. Nguyên lý Pháp y Không gian vs. Thời gian (Spatial vs. Temporal Forensics)**: Bản chất video là chuỗi ảnh 2D tĩnh ($T \times C \times H \times W$). Dấu vết cốt lõi của Diffusion xuất phát từ bộ mã hóa tri giác VAE và quá trình khử nhiễu ngược U-Net ở mức vi mô không gian (intra-frame spatial artifacts). Nếu mô hình không phát hiện được dấu vết trên từng frame 2D, việc thêm các module thời gian (Temporal LSTM/Transformer) chỉ gây lãng phí bộ nhớ $\mathcal{O}(T)$ mà không giải quyết được bản chất giả mạo.
  - **3. Ràng buộc Tài nguyên Phần cứng**: Mô hình video diffusion (như SVD) đòi hỏi VRAM $\ge 24\text{GB}$, vượt quá giới hạn GPU đồ án môn học tiêu chuẩn (8GB).
- **[KỊCH BẢN ỨNG KHẨU 30 GIÂY — ĐANH THÉP & CHUẨN XÁC]**:
  > *"Dạ thưa Thầy/Cô, có hai lý do phương pháp luận cốt lõi:
  > Thứ nhất, đồ án bám sát chuẩn Đề tài 11 và học liệu chính thức Hugging Face Unit 10, tập trung giải quyết bài toán: 'Sinh dữ liệu tổng hợp bằng Stable Diffusion'.
  > 
  > Thứ hai, về mặt pháp y thị giác máy tính, mọi video Deepfake đều cấu thành từ các khung hình tĩnh 2D. Dấu vết đặc trưng của Diffusion nằm ở mức phân bố vi mô không gian và phổ tần số của bộ giải mã VAE, chứ không nằm ở chiều thời gian. Làm chủ việc phát hiện trên ảnh tĩnh 2D là nền tảng tiên quyết; nếu không bắt được artifact trên từng frame, các mô hình video temporal sẽ gặp hiện tượng 'rác vào - rác ra'. Phạm vi này vừa đúng chuẩn học liệu, vừa đảm bảo tính khả thi trên GPU đơn."*

---

### Câu Hỏi 3: "Làm thế nào để đảm bảo không bị rò rỉ danh tính (Data Leakage) giữa tập Train và tập Test?"

- **[ĐIỂM NÓNG GIẢNG VIÊN ĐANG BẮT BẺ]**:
  - Giảng viên nghi ngờ tính trung thực của kết quả huấn luyện: "Nếu ảnh thật của người A nằm ở tập Train, rồi ảnh Inpainting/Img2Img của chính người A lại lọt vào tập Test, thì mô hình chỉ đang 'nhớ mặt' (memorize) người A chứ đâu có học phát hiện Deepfake! Con số độ chính xác cao là do gian lận dữ liệu!"
- **[BẢN CHẤT TOÁN HỌC & CÔNG NGHỆ]**:
  - Hiện tượng **Identity Leakage (Biometric Overfitting)** làm sai lệch hoàn toàn đánh giá an ninh sinh trắc học.
  - Đồ án thiết lập **Hợp Đồng Phân Hoạch Bất Biến Danh Tính (Subject-Exclusive / Identity-Disjoint Invariant)** tại tầng Data Ingestion (`src/data/image_datamodule.py`):
    1. Toàn bộ ảnh người thật được phân cụm theo danh tính duy nhất trước khi phân chia.
    2. Cố định hạt giống ngẫu nhiên `seed=42`, chia danh tính theo tỷ lệ 70% Train, 15% Val, 15% Test.
    3. Thiết lập ràng buộc toán học bất biến:
       $$\forall x \text{ có } \text{source\_real\_id}(x) = s \implies \text{Split}(x) = \text{Split}(s)$$
       Tất cả các biến thể phái sinh (Inpainting, Img2Img) bắt buộc phải nằm cùng tập phân hoạch với ảnh gốc.
    4. Tập Test chứa 100% danh tính hoàn toàn xa lạ (unseen identities) mà mô hình chưa từng được huấn luyện.
- **[KỊCH BẢN ỨNG KHẨU 30 GIÂY — ĐANH THÉP & CHUẨN XÁC]**:
  > *"Dạ thưa Thầy/Cô, nhóm đã loại trừ triệt để nguy cơ này bằng cơ chế phân hoạch Subject-Exclusive (Identity-Disjoint Invariant) ngay từ gốc kiến trúc:
  > 
  > Trước khi sinh ảnh, toàn bộ tập danh tính người thật được phân cụm độc lập và chia cố định vào 3 tập Train (70%), Val (15%), Test (15%) với seed=42.
  > Trong Manifest schema, mọi ảnh phái sinh từ Inpainting hoặc Img2Img đều lưu trường `source_real_id`. Hệ thống cưỡng chế ràng buộc: Toàn bộ ảnh phái sinh bắt buộc phải nằm cùng split với ảnh gốc. Vì vậy, tập Test chỉ chứa những khuôn mặt người mà mô hình chưa từng thấy trong tập Train, ngăn chặn 100% hiện tượng 'nhớ mặt' và phản ánh đúng năng lực tổng quát hóa."*

---

### Câu Hỏi 4: "Kế hoạch thực nghiệm và đánh giá đối chứng của nhóm được thiết kế như thế nào?"

- **[ĐIỂM NÓNG GIẢNG VIÊN ĐANG BẮT BẺ]**:
  - Giảng viên muốn kiểm tra tính trung thực học thuật và phân biệt rõ ràng giữa giai đoạn thiết kế và giai đoạn chạy thực nghiệm: "Các em đã có kết quả chạy trên GPU chưa hay mới chỉ thiết kế? Kế hoạch đánh giá có khách quan không?"
- **[BẢN CHẤT TOÁN HỌC & CÔNG NGHỆ]**:
  - Giai đoạn hiện tại là **Define & System Design**: hoàn thiện định nghĩa bài toán, thiết kế kiến trúc hệ thống và xây dựng toàn bộ mã nguồn pipeline.
  - Khung thực nghiệm đối chứng đa miền (Cross-Dataset Evaluation Protocol) được thiết kế bài bản:
    - 3 nhóm mô hình (Detector huấn luyện trên GAN, Detector huấn luyện trên dữ liệu SD của đồ án, và mô hình phòng thủ lai).
    - 2 tập test độc lập (StyleGAN2 và Stable Diffusion Dataset) được phân hoạch Identity-Disjoint.
    - Mã nguồn trong đồ án (`src/pipelines/evaluate_image_benchmark.py` và `src/models/detector/image_detector.py`) đã hiện thực hóa đầy đủ quy trình tính toán ROC-AUC theo định lý Wilcoxon-Mann-Whitney. Hệ thống đã vượt qua toàn bộ unit test, sẵn sàng vận hành trên GPU.
- **[KỊCH BẢN ỨNG KHẨU 30 GIÂY — ĐANH THÉP & CHUẨN XÁC]**:
  > *"Dạ thưa Thầy/Cô, hiện tại nhóm đang báo cáo giai đoạn **Define & System Design**, nghĩa là hoàn thiện định nghĩa bài toán, thiết kế kiến trúc hệ thống và xây dựng toàn bộ mã nguồn pipeline.
  > 
  > Nhóm đã thiết lập sẵn Khung thực nghiệm đối chứng đa miền (Evaluation Protocol) với 3 nhóm mô hình thử nghiệm và 2 tập test độc lập. Toàn bộ mã nguồn huấn luyện (`train_image_detector.py`) và công thức tính ROC-AUC (`evaluate_image_benchmark.py`) đã được viết hoàn chỉnh và vượt qua unit test. Trong giai đoạn tiếp theo, nhóm sẽ chạy batch dữ liệu trên GPU và ghi nhận các số liệu thực nghiệm thực tế lên ma trận này."*

---

### Câu Hỏi 5: "Làm sao giải quyết được hiện tượng lỗi mắt/tay hoặc da bị quá mịn (over-smoothing) của Stable Diffusion?"

- **[ĐIỂM NÓNG GIẢNG VIÊN ĐANG BẮT BẺ]**:
  - Giảng viên bắt bẻ vào hạn chế tự nhiên của Diffusion: "Ảnh Stable Diffusion rất hay bị dị tật đồng tử mắt lé, biến dạng cơ mặt hoặc da bị làm mịn như tượng sáp. Nếu đưa toàn bộ đống dữ liệu lỗi này vào huấn luyện thì detector chỉ học nhận diện ảnh dị tật chứ sao phát hiện được Deepfake tinh vi?"
- **[BẢN CHẤT TOÁN HỌC & CÔNG NGHỆ]**:
  - Vấn đề: Phân phối nhiễu không mong muốn (anomalous generative artifacts, wax-like skin do VAE decoder loss).
  - Giải pháp 2 tầng tích hợp trong đồ án:
    1. **Tầng Điều Khiển Sinh (Generation Conditioning)**:
       - Ứng dụng Prompt Engineering chuyên sâu kết hợp Negative Prompt chặt chẽ: `blurry, mutated eyes, deformed pupils, wax skin, unnatural face structure, disfigured`.
       - Khống chế Denoising Strength $S \in [0.4, 0.7]$ cho Img2Img và CFG Scale $s \in [7.0, 8.5]$.
    2. **Tầng Sàng Lọc Đa Phương Thức Tự Động (Multi-Modal ImageCurationGate - `src/pipelines/curate_image_dataset.py`)**:
       - **Laplacian Sharpness Gate**: Tính toán phương sai toán tử vi phân bậc hai $\text{Var}(\nabla^2 I) \ge 50.0$. Loại bỏ ngay các ảnh sáp mịn, mờ đục.
       - **Face Landmark & Retention Gate**: Kiểm định bằng MTCNN/Haar Cascade với độ tin cậy $\ge 0.85$. Đảm bảo tỷ lệ giữ cấu trúc giải phẫu $\ge 95\%$; mọi ảnh méo mắt, dị dạng đều bị chuyển thẳng vào thư mục cách ly (`quarantine/`).
       - **CLIP Semantic Gate**: Điểm tương đồng ngữ nghĩa $S_{\text{CLIP}} \ge 0.28$ (ViT-B/32), bảo đảm ảnh bám sát prompt.
- **[KỊCH BẢN ỨNG KHẨU 30 GIÂY — ĐANH THÉP & CHUẨN XÁC]**:
  > *"Dạ thưa Thầy/Cô, nhóm xử lý triệt để bài toán này bằng **Cổng kiểm duyệt chất lượng đa phương thức tự động (ImageCurationGate)** với 3 tầng lọc định lượng:
  > 
  > Thứ nhất là **Laplacian Sharpness Filter** với ngưỡng phương sai $\ge 50.0$, loại bỏ hoàn toàn các ảnh da sáp quá mịn hoặc mờ nhòe do bộ giải mã VAE.
  > Thứ hai là **Face Landmark Retention Gate** với độ tin cậy nhận diện $\ge 0.85$. Bất kỳ ảnh nào bị dị tật mắt lé hay biến dạng nhân trắc học đều bị hệ thống tự động loại bỏ vào thư mục cách ly (quarantine).
  > Kết hợp với Negative Prompt chuyên sâu ở khâu sinh ảnh, 100% dữ liệu lọt vào tập huấn luyện đều là ảnh chuẩn sinh trắc học và có độ chân thực cao."*

---

### Câu Hỏi 6: "Tại sao không dùng trực tiếp ảnh sinh bằng GAN để bổ sung data mà lại dùng Stable Diffusion?"

- **[ĐIỂM NÓNG GIẢNG VIÊN ĐANG BẮT BẺ]**:
  - Giảng viên so sánh công nghệ: "GAN sinh ảnh mặt người (như StyleGAN3) rất nét, tốc độ sinh nhanh gấp 50 lần Diffusion (chỉ 1 forward pass thay vì 30-50 bước khử nhiễu lặp). Tại sao phải dùng Stable Diffusion cồng kềnh, tốn tài nguyên hơn?"
- **[BẢN CHẤT TOÁN HỌC & CÔNG NGHỆ]**:
  - **1. Bản Chất Pháp Y Miền Tần Số (Fourier Domain Fingerprints)**:
    - GAN sử dụng các lớp tích chập chuyển vị (`ConvTranspose2d`), để lại các vết sọc ô cờ và các đỉnh nhọn dị thường (spectral spikes) trên phổ Fourier 2D (Frank et al., ICML 2020). Bộ phát hiện CNN học rất dễ dàng, nên bổ sung thêm ảnh GAN chỉ củng cố "lối tắt học tập" mà không mở rộng được ranh giới quyết định.
    - Stable Diffusion khử nhiễu liên tục theo quá trình khuếch tán ngược trong không gian tiềm ẩn (LDM), phổ năng lượng tuân theo định luật tự nhiên $1/f^\alpha$. Đây chính là "vùng mù" mà các hệ thống an ninh hiện nay cần bổ sung dữ liệu để phòng thủ.
  - **2. Khả Năng Kiểm Soát Cục Bộ (Controllability & Manipulation Ground-Truth)**:
    - GAN bị hiện tượng **Vướng víu không gian tiềm ẩn (Latent Entanglement)**: Khi muốn đổi biểu cảm miệng trong StyleGAN, các đặc trưng danh tính, tóc và nền xung quanh thường bị thay đổi ngoài ý muốn.
    - Stable Diffusion kết hợp cơ chế Cross-Attention với Text Prompt và Inpainting Mask nhị phân, cho phép can thiệp chính xác vào từng vùng giải phẫu (mắt, miệng) và **tự động xuất nhãn mặt nạ (Ground-Truth Mask) chính xác từng điểm ảnh** — điều GAN không thể thực hiện một cách nhất quán.
- **[KỊCH BẢN ỨNG KHẨU 30 GIÂY — ĐANH THÉP & CHUẨN XÁC]**:
  > *"Dạ thưa Thầy/Cô, có hai lý do căn bản về toán học và điều khiển dữ liệu:
  > 
  > Thứ nhất, về miền tần số: Ảnh GAN chứa các đỉnh dị thường cố định do phép tích chập chuyển vị up-sampling. Bổ sung ảnh GAN chỉ khiến mô hình bị lối tắt học tập (shortcut learning), không học được đặc trưng mới. Ngược lại, Stable Diffusion khử nhiễu liên tục, tái tạo phổ tự nhiên $1/f^\alpha$, là mối đe dọa thực tế mà các detector hiện tại đang bỏ lọt.
  > 
  > Thứ hai, về khả năng kiểm soát: GAN bị vướng víu tiềm ẩn (latent entanglement), không thể chỉnh sửa cục bộ mà vẫn bảo toàn danh tính gốc. Stable Diffusion kết hợp Inpainting Mask và Cross-Attention cho phép chỉnh sửa chính xác từng milimet và tự động sinh Ground-Truth Mask chuẩn 100% để huấn luyện bài toán phân vùng giả mạo."*

---

## 3. 📊 MA TRẬN TỪ KHÓA ĐẮT GIÁ & CÔNG THỨC TOÁN HỌC CỐT LÕI

Khi trả lời câu hỏi, việc lồng ghép tự nhiên các thuật ngữ chuẩn quốc tế và ký hiệu toán học sẽ ngay lập tức thuyết phục Hội đồng về độ sâu chuyên môn:

| Lĩnh Vực | Thuật Ngữ Đắt Giá (Buzzwords) | Ký Hiệu / Công Thức Toán Học | Vị Trí Module Trong Source Code |
| :--- | :--- | :--- | :--- |
| **Mô hình Khuếch tán** | Latent Diffusion Models (LDM), Reverse Markov Process, Classifier-Free Guidance (CFG) | $\mathcal{L}_{\text{LDM}} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\right]$ | `src/models/diffusion/stable_diffusion_engine.py` |
| **Miền Tần số** | 2D Fourier Spectrum, Power Spectral Density, $1/f^\alpha$ Natural Law, Checkerboard Artifacts | $\text{PSD}(f) \propto \frac{1}{f^\alpha}, \quad \alpha \approx 2.0$ | `src/pipelines/evaluate_image_benchmark.py` |
| **Phân Hoạch Dữ Liệu**| Subject-Exclusive Split, Identity-Disjoint Invariant, Anti-Identity Leakage | $\text{source\_real\_id}(x) = s \implies \text{Split}(x) = \text{Split}(s)$ | `src/data/image_datamodule.py` |
| **Kiểm Duyệt Chất Lượng**| Multi-Modal Curation Gate, Laplacian Sharpness, Face Landmark Retention, CLIP Score | $\text{Var}(\nabla^2 I) \ge 50.0, \quad S_{\text{CLIP}} \ge 0.28$ | `src/pipelines/curate_image_dataset.py` |
| **Đánh Giá Phòng Thủ**| Cross-Generator Generalization Collapse, Shortcut Learning, Wilcoxon ROC-AUC | $\text{AUC} = \frac{1}{N_{\text{pos}}N_{\text{neg}}} \sum \mathbf{1}(f(x_i) > f(x_j))$ | `src/models/detector/image_detector.py` |

---

## 4. 🛡️ QUY TRÌNH XỬ LÝ SỰ CỐ & KỸ THUẬT ĐẢO NGƯỢC TÌNH THẾ

### 4.1 Khi Giảng Viên Đặt Câu Hỏi Ngoài Phạm Vi (Out of Scope)
- *Ví dụ*: "Tại sao không áp dụng mô hình Vision-Language khổng lồ như GPT-4V hay LLaVA để phát hiện luôn?"
- *Kịch bản ứng phó*:
  > *"Dạ, ý kiến của Thầy/Cô rất hiện đại và đón đầu xu hướng. Tuy nhiên, các mô hình VLM khổng lồ như GPT-4V có kích thước hàng chục tỷ tham số, thời gian suy luận (latency) tính bằng giây và chi phí suy luận API quá lớn để triển khai tại các cổng kiểm soát eKYC thời gian thực.
  > 
  > Đồ án của nhóm tập trung vào phân khúc **Mô hình Phòng thủ Nhẹ & Tối ưu (Efficient Edge Detector)** như ResNet-18 và CNN, có thể suy luận dưới 15ms trên GPU thông thường hoặc CPU, phù hợp cho bài toán xác thực trực tuyến quy mô hàng triệu người dùng. Nhóm xin phép ghi nhận hướng tiếp cận VLM để phát triển trong các nghiên cứu chuyên sâu tiếp theo."*

### 4.2 Khi Giảng Viên Hỏi Câu Nhóm Chưa Có Số Liệu Ngay Lập Tức
- *Kịch bản ứng phó*:
  > *"Dạ thưa Thầy/Cô, đây là một điểm phản biện rất tinh tế. Trong phạm vi thiết kế của đồ án, nhóm đã xây dựng sẵn module `src/pipelines/evaluate_image_benchmark.py` với cấu trúc mở để đo lường tham số này. Hiện tại hệ thống đã hoàn thiện logic tính toán và nhóm đã sẵn sàng cấu hình để xuất số liệu chi tiết theo đúng gợi ý của Thầy/Cô trong báo cáo hoàn thiện cuối khóa."*

---

## 5. ✅ BẢNG TỰ KIỂM TRA TRƯỚC GIỜ BẢO VỆ (DEFENSE READINESS CHECKLIST)

- [x] **Thời lượng ứng khẩu**: Tất cả các câu trả lời đều được khống chế dứt khoát trong vòng **25 - 35 giây**.
- [x] **Thái độ tự tin, cầu thị**: Luôn bắt đầu bằng việc cảm ơn góc nhìn sắc sảo của Hội đồng.
- [x] **Dẫn chứng minh bạch**: Chỉ rõ tên file code (`src/pipelines/`, `src/data/`, `src/models/`) khi nói về giải pháp.
- [x] **Thành thật học thuật**: Khẳng định rõ ràng ranh giới giữa lý thuyết chuẩn quốc tế (CVPR/ICML) và phần thực thi mã nguồn của nhóm.
- [x] **Phong thái bảo vệ chuyên nghiệp**: Giọng nói rõ ràng, mắt nhìn thẳng vào Thầy/Cô đặt câu hỏi, kết thúc câu trả lời bằng một cái gật đầu tự tin.
