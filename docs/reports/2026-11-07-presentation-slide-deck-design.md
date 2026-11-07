# 5-Minute Ultra-Visual Define & System Design Slide Deck Specification (11-Slide Unified 2D Image + SD-Video Edition)

- **Project**: `uth-cv-coursework` — Topic 11: *Stable Diffusion-Based Synthetic Facial Image & Video Dataset Generation for Deepfake Detection*
- **Group Attribution**: `Nhóm thực hiện: Nhóm 4`
- **Target Artifact**: `docs/reports/presentation_5min_define_and_design.html` (with companion guides `docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md` and `docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md`)
- **Design System**: `Blue Professional` (`#fdfae7` warm cream canvas, `#1e2bfa` single electric cobalt blue accent) + `frontend-slides/viewport-base.css` (`1920×1080` fixed 16:9 stage, 100% Clean UI without `N` speaker-notes drawer or `R` references modal).

---

## 1. Intent & Why

The presentation unifies two complementary branches of **Group 4's (`Nhóm 4`)** work into one cohesive, ultra-visual slide deck:
1. **Track 1 — 2D Static Image Baseline (`baseline/initial-baseline`)**: Problem definition of generative domain shift, 3 spatial generation modes (`Text-to-Image`, `Facial Inpainting`, `Image-to-Image`), the 6-stage reproducible curation pipeline, and the 3-layer spatial quality gate.
2. **Track 2 — Temporal Video Extension (`sd-video`)**: Extending the pipeline to short identity-swapped video clips (`Ảnh danh tính A + Video chuyển động B → Video hoán đổi A←B`), benchmarking 4 pretrained Stable Diffusion video backends across 2 model families (`Face-Adapter`, `REFace`, `VFace`, `VividFace`), evaluating along 3 video axes (`Identity`, `Motion`, `Temporal Continuity` with occlusion handling), locking connected components (`Source Identity + Parent Video`) against data leakage, and evaluating downstream detector utility across 3 controlled regimes (`D0`, `D1`, `D2`).

To ensure maximum visual impact during live presentation, the deck enforces a **100% Clean UI** (no speaker-notes drawer `N`, no references overlay `R`, no clutter) and combines domain-specific Computer Vision vector graphics, conceptual portrait/filmstrip thumbnails (`Minh họa ý tưởng`), and subtle CSS/SVG data-flow animations (`@keyframes flowPulse`, `@keyframes scanLine`, `@keyframes playhead`).

---

## 2. Goals, Non-Goals, Anti-Goals & Hard Constraints

- **Goals**:
  - Expand the unified deck to **11 slides (`#slide-1` through `#slide-11`)** structured in 3 logical acts:
    - **Act I (Slides 01–06)**: Unified Vision, 2-Tier Scope, & 2D Spatial Image Foundation.
    - **Act II (Slides 07–09)**: Temporal Video Extension (`Ảnh A + Video B → Video A←B`, 4 Backends in 2 Families, Pilot Matrix, & 3-Axis Video QA Gate).
    - **Act III (Slides 10–11)**: Unified Dataset Passport, Connected-Component Zero-Leakage Vault (`Identity + Parent Video`), 3-Tier Detector Benchmark (`D0 / D1 / D2`), & Roadmap (`M0 → M7`).
  - Ensure 100% canvas fill (`1920×1080` stage, `1776×820` content stage) with zero dead space, zero text clipping (`SVG_TEXT_OVERFLOWS_RECT = 0`), and matched SVG `viewBox` aspect ratios (`SVG_ASPECT_RATIO_MISMATCH <= 22%`).
  - Display **`Nhóm thực hiện: Nhóm 4`** prominently on Slide 01 (Cover), Slide 03 (Unified Scope), Slide 04 (2D Mechanisms), and Slide 10 (Unified Dataset Design).
  - Keep the stage interface **100% Clean**: only the minimal bottom slide counter (`01 / 11`), prev/next arrows, 11 navigation dots, and bottom progress bar anchored inside `#deck-stage`.
- **Non-Goals & Anti-Goals**:
  - **No `N` / `R` Overlays or Popups**: Strictly omit speaker-notes drawers (`#overlay`, `.notes-drawer`) and reference modals so the presentation interface stays distraction-free.
  - **Zero Unverified Numbers / Thresholds / Code Clutter**: Forbidden in visible text: `95%`, `62%`, `92%`, `85%`, `50.0`, `>= 50`, `0.28`, `0.30`, `0.85`, `78.4`, `0.312`, `3,000`, `5,000`, `256`, `512`, `70%`, `15%`, `6GB`, `50 steps`, `30 steps`, `SHA-256`, `FP16`, `1e-4`, `src/`, `data/`, `<pre>`. (Convert `512 × 512` and `50 frame` from the video branch into pure conceptual labels such as `Khung mặt chuẩn hóa` and `Chuỗi khung hình ngắn`).
  - **Zero Rainbow / Dark Colors**: Strictly enforce `--bg: #fdfae7`, `--primary: #1e2bfa`, `--text: #111111`, `--text-muted: #6b6b6b`. Forbidden hex codes: `#d97706`, `#dc2626`, `#059669`, `#9333ea`, `#1e293b`, `#111827`.

---

## 3. Approaches Considered & Selected

| Approach | Summary | Trade-off | Decision |
| :--- | :--- | :--- | :--- |
| **Phương án 1: 11-Slide Unified Progression (2D Foundation $\to$ Temporal Video Extension $\to$ Unified Vault & `D0/D1/D2` Evaluation) with Clean UI** | Dedicates Slides 01–06 to Unified Scope & 2D Foundation, Slides 07–09 to Video Extension, and Slides 10–11 to Unified Dataset & Detector Evaluation, with no `N`/`R` overlays. | Expands deck to 11 slides, giving every diagram (2D portraits, `t1-t2-t3` filmstrips, 4 video backends, connected-component vault) full breathing room. | **Selected by User** |
| **Phương án 2: 9-Slide Side-by-Side Compression** | Places 2D Image and Video pipelines side-by-side inside the same slides. | High visual density; compresses the 4 video model families (`Face-Adapter`, `REFace`, `VFace`, `VividFace`) and `t1-t2-t3` filmstrips. | Rejected |
| **Phương án 3: 10-Slide Sequential Concatenation** | Appends Video slides after the 8-slide 2D deck without merging Dataset & Detector Evaluation. | Duplicates the Identity-Disjoint split and Detector Evaluation concepts across two separate endings. | Rejected |

---

## 4. 11-Slide Unified Visual Architecture Specification

### Slide 01 (`#slide-1`) — Cover & Unified 2-Track Vision (`Nhóm thực hiện: Nhóm 4`)
- **Left Column**:
  - Top badge: `COMPUTER VISION COURSEWORK • TOPIC 11 • NHÓM THỰC HIỆN: NHÓM 4`.
  - Headline: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*.
  - Middle Visual Panel: **Unified 2D-to-Video Thesis Strip (`viewBox="0 0 880 310"`)** illustrating **Chặng 1: Ảnh tĩnh 2D** (Portrait + Landmarks) $\to$ **Chặng 2: Mở rộng Video `A←B`** (`t1, t2, t3` filmstrip) $\to$ **Cổng Kiểm Định Đa Tầng** $\to$ **Deepfake Detector (`D0 / D1 / D2`)**.
  - Bottom 2×2 Metadata Grid (`Nhóm thực hiện: Nhóm 4 — Đề tài 11`, `Phạm vi: Ảnh Tĩnh 2D & Mở Rộng Video Ngắn`).
- **Right Column**:
  - Full-height **4-Pillar Unified Roadmap** (`01 Define Problem & 2-Tier Scope` $\to$ `02 2D Spatial Synthesis & Curation` $\to$ `03 Temporal Video Extension (4 Backends)` $\to$ `04 Unified Vault & D0/D1/D2 Evaluation`).

### Slide 02 (`#slide-2`) — `01 / PROBLEM DEFINITION`: Dịch Chuyển Miền & Biên Quyết Định Không Gian - Thời Gian
- **Top Row (2-Column Visual Split)**:
  - **Left Panel (`Visual Artifact Contrast`)**: Side-by-side face portrait comparison (`viewBox="0 0 800 380"`) contrasting *Classic Face-Swap Artifacts (Vết ghép biên mặt cục bộ)* vs. *Stable Diffusion Synthesis (Khử nhiễu toàn cục mượt từ Latent)* with animated scan-line overlays.
  - **Right Panel (`Conceptual Feature Space & Decision Boundary`)**: 2D feature-space diagram (`viewBox="0 0 800 380"`) showing clusters of `Real Faces`, `Classic Deepfakes`, and `Unseen Diffusion Samples (2D & Video)` crossing the old static decision boundary vs. the expanded boundary achieved by active Stable Diffusion data generation.
- **Bottom Row**: 3 concise visual cards (*01. Dịch Chuyển Miền Tạo Sinh*, *02. Kiểm Soát Thuộc Tính & Chuyển Động*, *03. Mở Rộng Biên Quyết Định*).

### Slide 03 (`#slide-3`) — `02 / UNIFIED SCOPE & BOUNDARIES`: Phạm Vi Thực Hiện 2 Tầng & Giới Hạn (`Nhóm thực hiện: Nhóm 4`)
- **Header**: Includes prominent **`Nhóm thực hiện: Nhóm 4`** cobalt badge.
- **Body (2-Column Visual Matrix)**:
  - **Left Panel (`Phạm Vi Trọng Tâm 2 Tầng — In-Scope`)**: Large visual diagram (`viewBox="0 0 800 420"`) showing **Tầng 1 (Nền tảng Ảnh Tĩnh 2D: T2I, Inpainting, Img2Img)** + **Tầng 2 (Mở rộng Video Ngắn `A←B`: 4 Backend Pretrained & Kiểm định thời gian)** + 4 compact scope cards.
  - **Right Panel (`Giới Hạn Loại Trừ — Out-of-Scope`)**: Large visual boundary diagram (`viewBox="0 0 800 420"`) + 4 compact exclusion cards (*Không sinh âm thanh / đồng bộ môi Audio-Voice*, *Không Pre-train lại Stable Diffusion từ đầu*, *Không mạo danh cá nhân thật ngoài bộ công khai*, *Không tối ưu phần cứng biên Edge*).

### Slide 04 (`#slide-4`) — `03 / 2D GENERATION MECHANISMS`: 3 Chế Độ Tạo Sinh Khuôn Mặt 2D (`Nhóm thực hiện: Nhóm 4`)
- **Header**: Includes **`Nhóm thực hiện: Nhóm 4`** badge.
- **Full-Stage 3-Column Visual Gallery** (`grid-template-columns: repeat(3, 1fr)`):
  - **Column 1 (`01. Text-to-Image`)**: Full-height SVG (`viewBox="0 0 510 560"`) showing structured prompt attribute tags $\to$ Gaussian noise grid ($z_T$) $\to$ animated denoising progression $\to$ **Synthesized Portrait Thumbnail**.
  - **Column 2 (`02. Facial Inpainting`)**: Full-height SVG (`viewBox="0 0 510 560"`) showing **Real Portrait Thumbnail + Binary Mask Overlay on Eyes/Mouth** $\to$ Masked Latent Denoising $\to$ **Locally Manipulated Portrait Thumbnail**.
  - **Column 3 (`03. Image-to-Image`)**: Full-height SVG (`viewBox="0 0 510 560"`) showing **Source Portrait Thumbnail + 5-Point Landmark Wireframe** $\to$ Guided Attribute Transfer $\to$ **Translated Portrait Thumbnail**.

### Slide 05 (`#slide-5`) — `04 / 2D SYSTEM ARCHITECTURE`: Đường Ống 6 Giai Đoạn Trên Miền Ảnh Tĩnh
- **Top Row**: Full-canvas **6-Stage Animated Pipeline SVG (`viewBox="0 0 1700 465"`)**:
  - `STAGE 01: Data Acquisition` $\to$ `STAGE 02: Face Preprocessing` $\to$ `STAGE 03: Stable Diffusion Generation` $\to$ `STAGE 04: Quality Control` $\to$ `STAGE 05: Labeling & Metadata` $\to$ `STAGE 06: Dataset Export` $\implies$ Full-width **`TÁC VỤ HẠ NGUỒN (DOWNSTREAM): Deepfake Detection & Evaluation`**.
- **Bottom Row**: 3 Architectural Intent cards (*Thiết Kế Mô-Đun Hóa*, *Chốt Chặn Kiểm Định Tự Động*, *Khớp Nối Trực Tiếp Tác Vụ Phát Hiện*).

### Slide 06 (`#slide-6`) — `05 / 2D AUTOMATED CURATION GATE`: Tiền Xử Lý Hình Học & Phễu Sàng Lọc 3 Tầng
- **Left Panel (`Stage 02 Deep-Dive: Tiền Xử Lý & Căn Chỉnh Khuôn Mặt`)**:
  - Visual SVG (`viewBox="0 0 800 560"`) illustrating: (1) *Bounding Box + 5 Điểm Mốc* with `@keyframes scanLine`, (2) *Căn chỉnh hình học (Affine Alignment)*, and (3) *Trích xuất mặt nạ ngữ nghĩa (Semantic Mask)*.
- **Right Panel (`Stage 04 Deep-Dive: Phễu Kiểm Định Chất Lượng 3 Tầng`)**:
  - Visual Funnel SVG (`viewBox="0 0 800 560"`) showing raw synthesized portraits entering a 3-layer sieve (*Tầng 1: Độ Sắc Nét*, *Tầng 2: Cấu Trúc Giải Phẫu Mặt*, *Tầng 3: Độ Khớp Ngữ Nghĩa*) with **Nhánh Loại Bỏ** vs. **Nhánh Đạt Chuẩn**.

### Slide 07 (`#slide-7`) — `06 / VIDEO EXTENSION: DEFINE`: Bài Toán Hoán Đổi Khuôn Mặt Trên Video (`Ảnh A + Video B → Video A←B`)
- **Top Row (`Visual Identity + Motion Equation`)**:
  - Full-width SVG (`viewBox="0 0 1680 410"`) adapted from `SD_VIDEO_DEFINE_AND_DESIGN_SLIDES.html`:
    1. **Card 1 (`Đầu vào 01: Ảnh Danh Tính A`)**: Source identity portrait `Người A` (cobalt identity features) with badge `Nguồn danh tính`.
    2. **`+` Operator** $\to$ **Card 2 (`Đầu vào 02: Video Chuyển Động B`)**: 3-frame temporal filmstrip (`t1`, `t2`, `t3`) showing head rotation (`-7°`, `0°`, `+7°`) and a **hand-occlusion path** crossing the lower face at `t3` (`Nguồn góc đầu, biểu cảm, che khuất`).
    3. **Animated SD Arrow (`Stable Diffusion Video`)** $\to$ **Card 3 (`Đầu ra: Video Hoán Đổi A←B`)**: 3-frame output filmstrip (`t1`, `t2`, `t3`) combining **Identity A** (cobalt features) with **Motion & Occlusion of B**, plus an animated playback timeline bar (`@keyframes playhead`).
- **Bottom Row (`3 Yêu Cầu Kỹ Thuật Cốt Lõi & Câu Hỏi Nghiên Cứu RQ1`)**:
  - **01. Bảo toàn Danh tính**: Khớp đặc trưng nhận dạng với `Ảnh A`, thay thế danh tính `B`.
  - **02. Bảo toàn Diễn xuất**: Giữ đúng góc đầu, biểu cảm và nhịp chuyển động của `Video B`.
  - **03. Liên tục Thời gian & Che khuất**: Chống nhấp nháy (`flicker`), chống lộ viền ghép (`seam`) và không vẽ đè khi tay/vật cản che mặt ở `t3`.

### Slide 08 (`#slide-8`) — `07 / VIDEO MODEL CANDIDATES`: 4 Mô Hình SD Video Trong 2 Họ Kiến Trúc & Ma Trận Pilot
- **Top Row (2-Column Visual Comparison of Model Families)**:
  - **Left Panel (`Họ 01 — Xử Lý Theo Từng Frame (Frame-by-Frame Wrapper)`)**:
    - Visual SVG (`viewBox="0 0 800 380"`) showing `Ảnh A` + independent frames `t1`, `t2`, `t3` processed separately (`infer t1`, `infer t2`, `infer t3`) with warning callout (`Dễ lệch nhẹ giữa các frame liên tiếp`).
    - Models: **`Face-Adapter`** (Pretrained adapter, kiểm tra đường chạy cơ sở) & **`REFace`** (Tập trung độ thật từng frame và giữ danh tính).
  - **Right Panel (`Họ 02 — Có Thành Phần Liên Kết Thời Gian (Temporal-Aware)`)**:
    - Visual SVG (`viewBox="0 0 800 380"`) showing `Ảnh A` + temporal links (`t1 ↔ t2 ↔ t3`) and landmark/bbox guidance (`Liên kết thời gian giảm nhấp nháy`).
    - Models: **`VFace`** (Mở rộng trực tiếp từ `REFace` với thành phần thời gian — không tính là họ độc lập) & **`VividFace`** (Điều kiện hóa theo `bbox / landmarks`, xử lý theo cụm frame có chồng lấp).
- **Bottom Row (`Ma Trận Thử Nghiệm Pilot & Nguyên Tắc So Sánh Công Bằng`)**:
  - Visual Pilot Matrix (`Tập cặp đầu vào đại diện (Ảnh A, Video B)` $\times$ `4 Backend Pretrained: Face-Adapter, REFace, VFace, VividFace`) để kiểm tra tính khả thi và lỗi điển hình trước khi sinh hàng loạt; giữ riêng bước tiền xử lý đặc thù của từng họ mô hình.

### Slide 09 (`#slide-9`) — `08 / VIDEO PIPELINE & 3-AXIS QA`: Đường Ống Sinh Video Tái Lập & Cổng Kiểm Định 3 Trục
- **Top Row (`6-Stage Reproducible Video Pipeline SVG`)**:
  - Full-width SVG (`viewBox="0 0 1680 280"`):
    `01 Đầu vào (Ảnh A + Video B)` $\to$ `02 Manifest (Mã cặp & Cụm liên thông)` $\to$ `03 4 Backend SD (Chạy Pretrained)` $\to$ `04 Đánh giá 3 trục + Cổng QA` $\to$ `05 Sinh Batch (Hàng đợi & Khôi phục mẫu lỗi)` $\to$ `06 Bộ dữ liệu + Hồ sơ Dataset Card` (with feedback loop `Chọn cấu hình trên Development → Khóa trước Holdout`).
- **Bottom Row (`Ba Trục Đánh Giá Video Tự Động + Cổng QA`)**:
  - **Trục 01 — Danh tính (`Identity`)**: Độ tương đồng vector nhận dạng khuôn mặt (`Output ↔ Ảnh A`) có đối chiếu với `Người B`.
  - **Trục 02 — Diễn xuất (`Motion & Pose`)**: Độ lệch góc đầu và biểu cảm so với `Video B` (chống hiện tượng mặt đơ cứng).
  - **Trục 03 — Thời gian (`Temporal & Occlusion`)**: Căn chỉnh bù chuyển động (`motion-compensated warping`) trên vùng mặt hợp lệ (`valid mask`) kết hợp mặt nạ che khuất (`occlusion mask`).
  - **Cổng QA (`Pass / Quarantine`)**: Đạt chuẩn $\to$ Đưa vào Dataset chính thức; Không đạt $\to$ Cách ly kèm nhãn nguyên nhân lỗi.

### Slide 10 (`#slide-10`) — `09 / UNIFIED DATASET & ZERO LEAKAGE VAULT`: Ba Nhóm Nguồn Gốc & Két Khóa Cụm Liên Thông (`Nhóm thực hiện: Nhóm 4`)
- **Header**: Includes **`Nhóm thực hiện: Nhóm 4`** badge.
- **Left Column**:
  - 3 Unified Provenance Taxonomy Cards (`01. Real Faces & Videos`, `02. Classic Deepfakes`, `03. SD Synthetic (2D & Video A←B)`) with dual-label badges.
  - Bottom **Unified Sample Passport Card** showing metadata keys for both 2D & Video samples (`Mã mẫu / pair_id`, `Nhãn Nhị Phân`, `Nhóm Nguồn Gốc & Backend`, `Khóa Danh Tính A`, `Khóa Video Cha B`, `Trạng Thái Cổng QA`).
- **Right Column**:
  - Full-height **Connected-Component Biometric Vault SVG (`viewBox="0 0 880 680`)**:
    - Left Vault (`Tập Train & Development`): Clusters locked by **Danh tính gốc A1** AND **Video cha B1**.
    - Center Animated Barrier: **`RÀO CHẮN CỤM LIÊN THÔNG (ZERO IDENTITY & PARENT-VIDEO LEAKAGE)`** ensuring neither source identity nor parent driving video crosses splits.
    - Right Vault (`Tập Kiểm Thử Độc Lập — Held-Out Test`): Disjoint **Danh tính gốc A2** AND **Video cha B2**.

### Slide 11 (`#slide-11`) — `10 / DETECTOR EVALUATION & ROADMAP`: Đối Chứng 3 Chế Độ (`D0 / D1 / D2`) & Lộ Trình Thực Thi
- **Top Row (`3-Condition Downstream Detector Evaluation & Conceptual ROC Curve`)**:
  - Full-width SVG (`viewBox="0 0 1680 390"`) illustrating:
    - **3 Controlled Training Regimes**:
      - **`D0` (Baseline Gốc)**: Chỉ dữ liệu gốc ban đầu.
      - **`D1` (+ Augmentation Thường)**: Bổ sung biến đổi ảnh/video cổ điển (đối chứng công bằng cùng ngân sách dữ liệu).
      - **`D2` (+ SD Synthetic Đạt Chuẩn QA)**: Bổ sung dữ liệu tổng hợp Stable Diffusion (Ảnh 2D & Video `A←B` đã qua cổng QA).
    - **Shared Detector & Held-Out Benchmark**: Cùng kiến trúc Detector, cùng ngân sách huấn luyện $\to$ Đánh giá trên tập `Held-Out Test` độc lập $\to$ **Conceptual Generalization Curve (`ROC-AUC`, `AP`, `TPR @ FPR`)** showing progression `D0` (dotted) $\to$ `D1` (dashed) $\to$ `D2` (solid cobalt).
- **Bottom Row (`Lộ Trình M0 → M7 & Sản Phẩm Đầu Ra`)**:
  - 4 Milestone & Deliverable Cards:
    - **01. `M0` — Define & Design**: Chốt đặc tả bài toán 2D + Video, ranh giới phạm vi và giao thức đánh giá.
    - **02. `M1–M2` — Input & Pilot Feasibility**: Chuẩn hóa manifest cụm liên thông và chạy thử nghiệm Pilot trên các chế độ/backend.
    - **03. `M3–M5` — Benchmark, QA Gate & Batch**: Đánh giá đa trục, khóa cấu hình trên tập Development và sinh dữ liệu hàng loạt có kiểm soát.
    - **04. `M6–M7` — Detector `D0/D1/D2` & Dataset Card**: Thực nghiệm Detector hạ nguồn, xuất bản bộ dữ liệu chuẩn hóa và báo cáo học thuật.

---

## 5. Verification & Automated Geometry Guardrails

- **Automated Geometry & Quality Test Suite (`tests/test_presentation_deck.py`)**:
  - Assert total slide count is `11` (`#slide-1` through `#slide-11`).
  - Assert `Nhóm thực hiện: Nhóm 4` appears on Slide 01, Slide 03, Slide 04, and Slide 10.
  - Assert **Clean UI Invariant**: zero `#overlay`, `.notes-drawer`, or speaker-notes popup elements in DOM.
  - Assert zero forbidden numeric tokens or code blocks across all visible text nodes.
  - Assert Headless Chrome `1920×1080` inspection passes across all 11 slides with `0` horizontal/vertical overflow, `0` `SVG_TEXT_OVERFLOWS_RECT`, and `0` `SVG_ASPECT_RATIO_MISMATCH`.
