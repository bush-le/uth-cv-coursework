# Technical Design Spec: 5-Minute Visual Define & System Design Slide Deck
## Topic 11: Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection

- **Status**: Approved Direction (Visual-First, Zero Unverified Numbers, Intent-Focused)
- **Target Deliverable**: `docs/reports/presentation_5min_define_and_design.html`
- **Design System**: `Blue Professional` (`#fdfae7` warm cream canvas, `#1e2bfa` single electric cobalt accent, `Space Grotesk` + `Inter`) + `frontend-slides` (`viewport-base.css` fixed `1920×1080` 16:9 stage)
- **Duration**: 5 Minutes (6 High-Impact Visual Slides)

---

## 1. Core Intent & Strict Content Guardrails

### 1.1 Presentation Goal
Trình bày **Define (Định nghĩa Bài toán, Phạm vi, Cấu trúc Dataset)** và **Design System (Luồng Pipeline Dự Kiến & Thiết Kế Thực Nghiệm Đánh Giá)** trong đúng 5 phút bằng ngôn ngữ thị giác (sơ đồ khối SVG, luồng mũi tên, thẻ trực quan lấp đầy 100% khung hình `1920×1080`).

### 1.2 Strict Content Exclusions (TUYỆT ĐỐI KHÔNG ĐƯA VÀO SLIDE)
1. **Không đưa bất kỳ con số/số liệu nào chưa thực hiện hoặc chưa kiểm chứng**:
   - Cấm tuyệt đối các con số giả định/chưa chạy như `95% -> 62%`, `>= 92%`, `>= 85%`, `Laplacian >= 50`, `CLIP >= 0.28`, `0.85`, `78.4`, `0.312`, `3,000 - 5,000`, `256x256`, `512x512`, `70% / 15% / 15%`, `< 6GB VRAM`, `50 steps`, `seed 42`, `alpha = 0.4`, `lr = 1e-4`.
2. **Không đưa chi tiết kỹ thuật vụn vặt / đường dẫn code / JSON thô**:
   - Không hiển thị tên file code (`src/...`, `data/...`), không dùng khối `<pre><code>` JSON đen, không nêu `SHA-256`, `FP16`, `AdamW`, `BCEWithLogitsLoss`, hay giải thích toán học U-Net/VAE/Fourier.
3. **Không dùng màu sắc cầu vồng hoặc thanh công cụ thừa**:
   - Loại bỏ toàn bộ màu cam (`#d97706`), đỏ (`#dc2626`), xanh lá (`#059669`), tím (`#9333ea`) trên viền/nền thẻ và loại bỏ thanh HUD đen + ngăn kéo Speaker Notes.
   - Tuân thủ 100% kỷ luật đơn sắc của `Blue Professional`: nền kem `#fdfae7`, điểm nhấn duy nhất xanh cobalt `#1e2bfa`, thẻ phủ sắc xanh 4% (`rgba(30, 43, 250, 0.04)`) với viền mờ 20% (`rgba(30, 43, 250, 0.2)`).
4. **Không để bất kỳ khoảng trống chết nào trên khung hình `1920×1080`**:
   - Mọi slide phải dùng bố cục CSS Grid / Flex `stretch` lấp đầy toàn bộ vùng nội dung `1776×820`, chữ to rõ từ xa (`18px–60px`).

---

## 2. Visual Slide-by-Slide Architecture (6 Slides)

### Slide 01: Title & 4-Pillar Visual Roadmap (`layout-cover-split`)
- **Left Column (54% width)**:
  - Eyebrow & Tag Pill: `COMPUTER VISION COURSEWORK • TOPIC 11`
  - Accent Line (`60×4px` cobalt)
  - `H1`: *Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection*
  - Subtitle: *Định Nghĩa Bài Toán & Thiết Kế Hệ Thống Sinh Dữ Liệu Khuôn Mặt Tổng Hợp*
  - Bottom Presenter Card: Thông tin sinh viên thực hiện, trường UTH, và định hướng báo cáo.
- **Right Column (46% width, full height)**:
  - Khối sơ đồ 4 Trụ Cột Luận Chứng kết nối trực quan bằng mũi tên SVG:
    1. `01 / Define Problem` — Khoảng trống dữ liệu trong phát hiện Deepfake
    2. `02 / Define Scope` — Mục tiêu & 3 chế độ sinh ảnh khuôn mặt 2D
    3. `03 / Design Dataset & System` — Phân nhóm dữ liệu & Pipeline 6 bước
    4. `04 / Design Evaluation` — Thực nghiệm đối chứng & Kết quả kỳ vọng

### Slide 02: 01 / Define the Problem (`layout-problem`)
- **Header**: Eyebrow `01 / PROBLEM DEFINITION` | Tag `Research Motivation` | `H2`: *Tại Sao Cần Sinh Dữ Liệu Khuôn Mặt Tổng Hợp Cho Bài Toán Phát Hiện Deepfake?*
- **Top Visual Zone (62% height) — SVG Contrast Diagram**:
  - **Nhánh Trên (Hạn Chế Của Hướng Tiếp Cận Dữ Liệu Hiện Tại)**:
    `Thu Thập Ảnh Thật & Deepfake Sẵn Có` $\to$ `Rào Cản Riêng Tư, Chi Phí Gán Nhãn & Kém Đa Dạng` $\to$ `Mô Hình Phát Hiện Khó Tổng Quát Hóa Trước Kỹ Thuật Sinh Mới`
  - **Nhánh Dưới (Hướng Đi Đề Xuất Với Stable Diffusion)**:
    `Chủ Động Sinh Khuôn Mặt Tổng Hợp Có Kiểm Soát` $\to$ `Đa Dạng Hóa Thuộc Tính & Tự Động Hóa Nhãn Nguồn Gốc` $\to$ `Bổ Sung Dữ Liệu Giúp Cải Thiện Mô Hình Phát Hiện Deepfake`
- **Bottom Zone (38% height) — 3 Equal Visual Cards**:
  1. `01. Sự Phụ Thuộc Vào Dữ Liệu`: Hiệu năng của các mô hình Deepfake Detection phụ thuộc trực tiếp vào độ đa dạng và chất lượng của dữ liệu huấn luyện.
  2. `02. Điểm Nghẽn Thu Thập & Gán Nhãn`: Dữ liệu khuôn mặt thực tế bị giới hạn bởi quyền riêng tư sinh trắc học và chi phí gán nhãn thủ công.
  3. `03. Câu Hỏi Nghiên Cứu Trọng Tâm`: Liệu dữ liệu khuôn mặt tổng hợp từ Stable Diffusion có giúp mở rộng độ đa dạng và nâng cao hiệu quả phát hiện Deepfake?

### Slide 03: 02 / Define Objectives & Scope (`layout-scope`)
- **Header**: Eyebrow `02 / OBJECTIVES & SCOPE` | Tag `System Boundaries` | `H2`: *Mục Tiêu Hệ Thống & Phạm Vi Ba Chế Độ Tạo Sinh Khuôn Mặt*
- **Top Visual Zone (64% height) — 3 Visual Mode Cards (Each with SVG Input → Process → Output Diagram)**:
  1. **Text-to-Image (T2I)**:
     - SVG Flow: `[Văn Bản Mô Tả (Prompt)]` $\to$ `[Stable Diffusion]` $\to$ `[Khuôn Mặt Tổng Hợp Mới]`
     - Ý định: Tạo các nhân dạng khuôn mặt mới hoàn toàn nhằm tăng độ đa dạng cho tập dữ liệu.
  2. **Facial Inpainting**:
     - SVG Flow: `[Ảnh Mặt Thật + Vùng Chọn]` $\to$ `[Stable Diffusion]` $\to$ `[Khuôn Mặt Chỉnh Sửa Cục Bộ]`
     - Ý định: Thay đổi cục bộ từng vùng trên khuôn mặt (như mắt, miệng, biểu cảm) trên nền ảnh thật.
  3. **Image-to-Image (Img2Img)**:
     - SVG Flow: `[Ảnh Mặt Thật + Điều Kiện]` $\to$ `[Stable Diffusion]` $\to$ `[Khuôn Mặt Biến Đổi Bề Mặt]`
     - Ý định: Biến đổi diện mạo, ánh sáng hoặc thuộc tính khuôn mặt trong khi giữ bố cục gốc.
- **Bottom Zone (36% height) — 2-Column Visual Scope Matrix**:
  - **Trong Phạm Vi Nghiên Cứu (In-Scope)**: Tập trung vào **ảnh khuôn mặt tĩnh 2D**; xây dựng quy trình sinh, lọc và đóng gói dữ liệu bằng Stable Diffusion; đánh giá tác động trên mô hình phát hiện Deepfake.
  - **Ngoài Phạm Vi Nghiên Cứu (Out-of-Scope)**: **Không** xử lý Video Diffusion hay chuỗi thời gian; **không** huấn luyện mới mô hình Diffusion từ đầu; **không** đi sâu mổ xẻ kiến trúc nội bộ U-Net/VAE.

### Slide 04: 03 / Dataset Design & Provenance (`layout-dataset`)
- **Header**: Eyebrow `03 / DATASET DESIGN` | Tag `Taxonomy & Split Integrity` | `H2`: *Thiết Kế Ba Nhóm Dữ Liệu Độc Lập & Nguyên Tắc Phân Tập Chống Rò Rỉ*
- **Left Column (48% width, full height) — Visual 3-Group Taxonomy Stack**:
  - Sơ đồ 3 khối phân biệt rõ bản chất dữ liệu (không đánh đồng ảnh SD với Deepfake truyền thống):
    1. **`01. Real Faces` (Ảnh Mặt Thật)**: Ảnh chân dung người thật nguyên bản làm chuẩn đối chiếu.
    2. **`02. Deepfake Faces` (Ảnh Mặt Bị Thao Túng)**: Khuôn mặt thật bị hoán đổi hoặc làm giả bằng kỹ thuật Deepfake.
    3. **`03. SD Synthetic Faces` (Ảnh Sinh Từ Stable Diffusion)**: Khuôn mặt tổng hợp hoặc biến đổi có kiểm soát từ pipeline đề xuất.
  - Callout Box: Giữ 3 nhóm tách biệt trong metadata để bảo toàn nguồn gốc (*provenance*) và hỗ trợ phân tích độc lập từng loại.
- **Right Column (52% width, full height) — 2 Visual Architecture Panels**:
  - **Top Panel — SVG Identity-Disjoint Split Diagram**:
    Minh họa trực quan: Toàn bộ ảnh gốc và các ảnh biến thể sinh ra từ cùng một nhân dạng bắt buộc đi vào cùng một tập (`Train`, `Validation`, hoặc `Test`), ngăn chặn mô hình học thuộc khuôn mặt.
  - **Bottom Panel — Visual Metadata Schema Map**:
    6 khối thẻ thuộc tính trực quan (`Mã Định Danh Ảnh`, `Nhãn Phân Loại`, `Nhóm Nguồn Gốc`, `Chế Độ Sinh Ảnh`, `Liên Kết Ảnh Gốc`, `Câu Lệnh Prompt`) thay cho code JSON thô.

### Slide 05: 04 / System Architecture — Proposed Pipeline (`layout-architecture` — HERO SLIDE)
- **Header**: Eyebrow `04 / SYSTEM ARCHITECTURE` | Tag `Proposed End-to-End Pipeline` | `H2`: *Kiến Trúc Luồng Hệ Thống Dự Kiến Qua Sáu Giai Đoạn Khép Kín*
- **Main Visual Zone (72% height) — Full-Canvas 6-Stage Pipeline SVG (`1776×620`)**:
  Sơ đồ luồng trực quan lớn, rõ nét, thể hiện ý định xuyên suốt từ đầu vào đến đầu ra:
  - **Stage 01: Data Acquisition** (*Thu thập nguồn ảnh mặt thật & Deepfake*)
    $\to$ **Stage 02: Face Preprocessing** (*Phát hiện khuôn mặt, căn chỉnh & cắt chuẩn hóa*)
    $\to$ **Stage 03: SD Generation** (*Sinh ảnh tổng hợp qua 3 chế độ T2I, Inpaint, Img2Img*)
    $\to$ **Stage 04: Quality Control** (*Kiểm tra độ nét, cấu trúc khuôn mặt & khớp mô tả* — nhánh **Đạt** đi tiếp, nhánh **Không đạt** loại bỏ)
    $\to$ **Stage 05: Labeling & Metadata** (*Gán nhãn nhóm dữ liệu & ghi nhận thông tin nguồn gốc*)
    $\to$ **Stage 06: Dataset Export** (*Tổ chức tập dữ liệu Train / Val / Test theo danh tính*)
    $\implies$ **Downstream Task: Deepfake Detection & Evaluation** (*Đưa vào huấn luyện & đánh giá mô hình phát hiện*)
- **Bottom Zone (28% height) — 3 Architecture Intent Cards**:
  1. `Thiết Kế Module Hóa`: Tách biệt rõ khâu tiền xử lý, sinh ảnh, lọc chất lượng và đóng gói dữ liệu.
  2. `Kiểm Soát Chất Lượng Tự Động`: Ngăn chặn các mẫu ảnh sinh bị lỗi biến dạng khuôn mặt hoặc sai lệch mô tả lọt vào dataset.
  3. `Sẵn Sàng Cho Huấn Luyện Hạ Nguồn`: Xuất bản dữ liệu có cấu trúc đồng nhất kèm metadata đầy đủ cho mô hình phát hiện.

### Slide 06: 05 / Evaluation Design & Expected Outcomes (`layout-evaluation`)
- **Header**: Eyebrow `05 / EVALUATION & OUTCOMES` | Tag `Validation & Deliverables` | `H2`: *Thiết Kế Thực Nghiệm Đối Chứng & Bốn Kết Quả Đầu Ra Dự Kiến*
- **Top Visual Zone (56% height) — SVG A/B Comparative Experiment Diagram**:
  - **Nhánh Baseline**: `[Dữ Liệu: Real + Deepfake]` $\to$ `[Huấn Luyện Mô Hình Phát Hiện (CNN / ViT)]`
  - **Nhánh Proposed**: `[Dữ Liệu: Real + Deepfake + SD Synthetic]` $\to$ `[Huấn Luyện Cùng Kiến Trúc Mô Hình]`
  - Hai nhánh hội tụ vào: **`[Đánh Giá Trên Cùng Tập Test Độc Lập]`** $\to$ **`[So Sánh: Accuracy, Precision, Recall, F1, ROC-AUC & Độ Tổng Quát Hóa]`**
- **Bottom Zone (44% height) — 4 Visual Deliverable Cards**:
  1. `01. Structured Facial Dataset`: Bộ dữ liệu ảnh khuôn mặt phân tách rõ 3 nhóm kèm metadata nguồn gốc.
  2. `02. Automated Generation Pipeline`: Quy trình tự động từ tiền xử lý, sinh ảnh Stable Diffusion đến lọc chất lượng.
  3. `03. Detection Benchmark Setup`: Mô hình cơ sở và kịch bản thực nghiệm đối chứng đánh giá giá trị dữ liệu.
  4. `04. Empirical Evaluation Report`: Phân tích thực nghiệm về đóng góp của dữ liệu tổng hợp cho phát hiện Deepfake.
