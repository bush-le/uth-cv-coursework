# 5-Minute Ultra-Visual Define & System Design Slide Deck Specification (8-Slide Edition)

- **Project**: `uth-cv-coursework` — Topic 11: *Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection*
- **Group Attribution**: `Nhóm thực hiện: Nhóm 4`
- **Target Artifact**: `docs/reports/presentation_5min_define_and_design.html` (with companion guides `docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md` and `docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md`)
- **Design System**: `Blue Professional` (`#fdfae7` warm cream canvas, `#1e2bfa` single electric cobalt blue accent) + `frontend-slides/viewport-base.css` (`1920×1080` fixed 16:9 stage).

---

## 1. Intent & Why

The presentation must communicate **Problem Definition (Define)** and **Proposed System Architecture (Design System)** in 5 minutes with maximum visual impact. To eliminate the "text-in-boxes" feel while strictly avoiding any unexecuted numbers or fabricated experimental results, this design combines **three visual layers** across an expanded **8-slide structure**:

1. **Approach 1 — Domain-Specific Computer Vision Vector Graphics & 50% Text Reduction**:
   - Replace wordy bullet paragraphs and plain text rectangles with recognizable Computer Vision illustrations: 2D feature-space decision boundaries, Gaussian latent noise grids ($z_T \to z_0$), 5-point facial landmarks, face crop bounding boxes, binary inpainting masks, biometric identity vaults, a 3-stage quality control sieve/funnel, neural network detector blocks (`CNN / ViT`), and a conceptual ROC curve schematic.
2. **Approach 2 — Embedded Illustrative Face Thumbnails & Technical Overlays**:
   - Embed self-contained, stylized facial portrait thumbnails (constructed via high-craft layered SVG portraits with realistic shading, facial contours, landmark nodes, mask overlays, and noise patterns, explicitly badged as `MINH HỌA Ý TƯỞNG / CONCEPTUAL ILLUSTRATION`) inside Slides 01, 02, 03, 05, and 07 so viewers immediately see how faces are aligned, masked, synthesized, grouped by identity, and filtered.
3. **Approach 3 — Animated Data-Flow & Visual Scanning Pulses**:
   - Add pure CSS/SVG keyframe animations (`@keyframes flowPulse` on pipeline connectors, `@keyframes scanLine` on facial landmark detection boxes, and `@keyframes pulseNode` on active synthesis/gate nodes) to guide the audience's eye along the pipeline flow during live presentation.

---

## 2. Goals, Non-Goals, Anti-Goals & Hard Constraints

- **Goals**:
  - Expand the deck to **8 slides (`#slide-1` through `#slide-8`)** so each slide has one unmistakable visual centerpiece and zero visual crowding.
  - Ensure 100% canvas fill (`1920×1080` stage, `1776×820` content stage) with zero dead space, zero text clipping (`SVG_TEXT_OVERFLOWS_RECT = 0`), and matched SVG `viewBox` aspect ratios (`SVG_ASPECT_RATIO_MISMATCH <= 22%`).
  - Display **`Nhóm thực hiện: Nhóm 4`** prominently on Slide 01 (Cover), Slide 04 (Scope), and Slide 05 (Dataset Design).
- **Non-Goals & Anti-Goals**:
  - **Zero Unverified Numbers / Thresholds / Code Clutter**: Forbidden in visible text: `95%`, `62%`, `92%`, `85%`, `50.0`, `>= 50`, `0.28`, `0.30`, `0.85`, `78.4`, `0.312`, `3,000`, `5,000`, `256`, `512`, `70%`, `15%`, `6GB`, `50 steps`, `30 steps`, `SHA-256`, `FP16`, `1e-4`, `src/`, `data/`, `<pre>`.
  - **Zero Rainbow / Dark Colors**: Strictly enforce `--bg: #fdfae7`, `--primary: #1e2bfa`, `--text: #111111`, `--text-muted: #6b6b6b`. Forbidden hex codes: `#d97706`, `#dc2626`, `#059669`, `#9333ea`, `#1e293b`, `#111827`.

---

## 3. Approaches Considered & Selected

| Approach | Summary | Trade-off | Decision |
| :--- | :--- | :--- | :--- |
| **1. CV Vector Illustrations + 50% Text Reduction** | Replace text boxes with CV diagrams (landmarks, latent grids, decision boundaries, funnels). | Crisp and self-contained, but needs portrait thumbnails for immediate human-face recognition. | **Selected (Combined)** |
| **2. Embedded Illustrative Face Thumbnails + Overlays** | Show visual face portraits for Real, Deepfake, T2I, Inpainting Mask, Img2Img, and QC Rejection. | High visual engagement; must be clearly badged as conceptual illustrations (`Minh họa ý tưởng`) to uphold academic honesty. | **Selected (Combined)** |
| **3. Animated Pipeline Flow & Scanning Pulses** | Animate arrows (`stroke-dashoffset`), landmark scan lines, and quality gate pulses. | Guides attention dynamically during live defense; best layered on top of Approaches 1 & 2. | **Selected (Combined)** |

---

## 4. 8-Slide Visual Architecture Specification

### Slide 01 (`#slide-1`) — Cover & Core System Vision (`Nhóm thực hiện: Nhóm 4`)
- **Left Column**:
  - Top badge: `COMPUTER VISION COURSEWORK • TOPIC 11 • NHÓM THỰC HIỆN: NHÓM 4`.
  - Headline: *Stable Diffusion-Based Synthetic Facial Image Dataset Generation for Deepfake Detection*.
  - Middle Visual Panel: **Visual Face-to-Detector Thesis Strip** (`viewBox="0 0 880 310"`) featuring an illustrated source portrait with landmark nodes $\to$ animated Stable Diffusion latent prism $\to$ Quality Gate shield $\to$ Deepfake Detector network.
  - Bottom 2×2 Metadata Grid (`Nhóm thực hiện: Nhóm 4 — Đề tài 11`).
- **Right Column**:
  - Full-height **4-Pillar Visual Roadmap** (`01 Define Problem` $\to$ `02 Define Mechanisms & Scope` $\to$ `03 Design Dataset & Architecture` $\to$ `04 Design Evaluation`) with animated spine connectors.

### Slide 02 (`#slide-2`) — `01 / PROBLEM DEFINITION`: Dịch Chuyển Miền & Biên Quyết Định
- **Top Row (2-Column Visual Split)**:
  - **Left Panel (`Visual Artifact Contrast`)**: Side-by-side face portrait comparison (`viewBox="0 0 800 380"`) contrasting *Classic Face-Swap Artifacts (Vết ghép biên mặt cục bộ)* vs. *Stable Diffusion Synthesis (Khử nhiễu toàn cục mượt từ Latent)* with animated scan-line overlays.
  - **Right Panel (`Conceptual Feature Space & Decision Boundary`)**: 2D feature-space diagram (`viewBox="0 0 800 380"`) showing clusters of `Real Faces`, `Classic Deepfakes`, and `Unseen Diffusion Samples` crossing the old static decision boundary vs. the expanded boundary achieved by active Stable Diffusion data generation.
- **Bottom Row**: 3 concise visual cards (*01. Dịch Chuyển Miền Tạo Sinh*, *02. Kiểm Soát Thuộc Tính Chủ Động*, *03. Mở Rộng Biên Quyết Định*).

### Slide 03 (`#slide-3`) — `02 / GENERATION MECHANISMS`: 3 Chế Độ Tạo Sinh Khuôn Mặt (Visual Showcase)
- **Full-Stage 3-Column Visual Gallery** (`grid-template-columns: repeat(3, 1fr)`):
  - **Column 1 (`01. Text-to-Image`)**: Full-height SVG (`viewBox="0 0 510 560"`) showing structured prompt attribute tags $\to$ Gaussian noise grid ($z_T$) $\to$ animated denoising progression $\to$ **Synthesized Portrait Thumbnail** with clean facial geometry.
  - **Column 2 (`02. Facial Inpainting`)**: Full-height SVG (`viewBox="0 0 510 560"`) showing **Real Portrait Thumbnail + Binary Mask Overlay on Eyes/Mouth** $\to$ Masked Latent Denoising $\to$ **Locally Manipulated Portrait Thumbnail** preserving outer facial contour.
  - **Column 3 (`03. Image-to-Image`)**: Full-height SVG (`viewBox="0 0 510 560"`) showing **Source Portrait Thumbnail + 5-Point Landmark Wireframe** $\to$ Guided Attribute Transfer $\to$ **Translated Portrait Thumbnail** preserving pose geometry with altered lighting/expression.

### Slide 04 (`#slide-4`) — `03 / SCOPE & BOUNDARIES`: Phạm Vi Thực Hiện & Giới Hạn (`Nhóm thực hiện: Nhóm 4`)
- **Header**: Includes prominent **`Nhóm thực hiện: Nhóm 4`** cobalt badge.
- **Body (2-Column Visual Matrix)**:
  - **Left Panel (`Phạm Vi Trọng Tâm — In-Scope`)**: Large visual diagram (`viewBox="0 0 800 420"`) + 4 compact cards with custom CV icons (*Miền ảnh tĩnh khuôn mặt 2D*, *Đường ống đa chế độ & Lọc tự động*, *Phân tập độc lập danh tính*, *Thực nghiệm đối chứng Detector*).
  - **Right Panel (`Giới Hạn Loại Trừ — Out-of-Scope`)**: Large visual boundary diagram (`viewBox="0 0 800 420"`) + 4 compact exclusion cards (*Không xử lý Video / Audio thời gian thực*, *Không Pre-train lại Stable Diffusion*, *Không mạo danh cá nhân thật*, *Không tối ưu phần cứng biên Edge*).

### Slide 05 (`#slide-5`) — `04 / DATASET DESIGN`: Ba Nhóm Nguồn Gốc & Két Khóa Danh Tính (`Nhóm thực hiện: Nhóm 4`)
- **Header**: Includes **`Nhóm thực hiện: Nhóm 4`** badge.
- **Left Column**:
  - 3 Visual Taxonomy Cards (`01. Real Faces`, `02. Deepfake Faces`, `03. SD Synthetic Faces`) with portrait avatar icons and dual-label badges (`Real vs. Fake` + `Sub-group Provenance`).
  - Bottom **Visual Sample ID Card** illustrating the 6 metadata attributes (`Mã Định Danh`, `Nhãn Nhị Phân`, `Nhóm Nguồn Gốc`, `Chế Độ Sinh`, `Khóa Danh Tính Gốc`, `Đặc Tả Prompt`) as a visual ID badge.
- **Right Column**:
  - Full-height **Identity-Disjoint Biometric Vault SVG (`viewBox="0 0 880 680`)**:
    - Left Vault (`Cụm Danh Tính Gốc A`): Portrait A + 3 variant thumbnails locked inside `Tập Train & Validation`.
    - Center Animated Barrier: **`RÀO CHẮN ĐỘC LẬP DANH TÍNH (ZERO IDENTITY LEAKAGE)`** with shield icon.
    - Right Vault (`Cụm Danh Tính Gốc B`): Portrait B + 3 variant thumbnails locked inside `Tập Kiểm Thử (Held-Out Test)`.

### Slide 06 (`#slide-6`) — `05 / SYSTEM ARCHITECTURE`: Đường Ống 6 Giai Đoạn Dự Kiến (Hero Animated Pipeline)
- **Top Row**: Full-canvas **6-Stage Animated Pipeline SVG (`viewBox="0 0 1700 465"`)**:
  - `STAGE 01: Data Acquisition` $\to$ `STAGE 02: Face Preprocessing` $\to$ `STAGE 03: Stable Diffusion Generation` $\to$ `STAGE 04: Quality Control` $\to$ `STAGE 05: Labeling & Metadata` $\to$ `STAGE 06: Dataset Export` $\implies$ Full-width **`TÁC VỤ HẠ NGUỒN (DOWNSTREAM): Deepfake Detection & Evaluation`**.
  - Every connector arrow features an animated data-flow pulse (`@keyframes flowPulse`), and each stage includes a custom visual icon alongside concise labels.
- **Bottom Row**: 3 Architectural Intent cards (*Thiết Kế Mô-Đun Hóa*, *Chốt Chặn Kiểm Định Tự Động*, *Khớp Nối Trực Tiếp Tác Vụ Phát Hiện*).

### Slide 07 (`#slide-7`) — `06 / AUTOMATED CURATION GATE`: Cơ Chế Tiền Xử Lý & Phễu Sàng Lọc Tự Động (Visual Deep-Dive)
- **Left Panel (`Stage 02 Deep-Dive: Tiền Xử Lý & Căn Chỉnh Khuôn Mặt`)**:
  - Visual SVG (`viewBox="0 0 800 560"`) illustrating:
    1. *Ảnh đầu vào* với khung quét **Bounding Box + 5 Điểm Mốc (Mắt, Mũi, Miệng)** có hiệu ứng tia quét (`@keyframes scanLine`).
    2. *Căn chỉnh hình học (Affine Alignment)* đưa khuôn mặt về trục chuẩn.
    3. *Trích xuất mặt nạ ngữ nghĩa (Semantic Mask)* khoanh vùng mắt, mũi, miệng phục vụ Inpainting.
- **Right Panel (`Stage 04 Deep-Dive: Phễu Kiểm Định Chất Lượng 3 Tầng`)**:
  - Visual Funnel SVG (`viewBox="0 0 800 560"`) showing raw synthesized portraits entering a 3-layer sieve:
    1. *Tầng 1 — Độ Sắc Nét (Sharpness Gate)*: Loại bỏ ảnh mờ nhòe.
    2. *Tầng 2 — Cấu Trúc Giải Phẫu Mặt (Landmark Integrity)*: Loại bỏ ảnh méo mặt / biến dạng.
    3. *Tầng 3 — Độ Khớp Ngữ Nghĩa (Prompt-Image Alignment)*: Loại bỏ ảnh lệch mô tả.
  - Bottom split: **Nhánh Loại Bỏ (Mẫu mờ / biến dạng)** vs. **Nhánh Đạt Chuẩn (Chuyển tiếp sang Stage 05 Gán nhãn)**.

### Slide 08 (`#slide-8`) — `07 / EVALUATION & EXPECTED OUTCOMES`: Thiết Kế Thực Nghiệm Đối Chứng & Sản Phẩm Đầu Ra
- **Top Row**: Full-width SVG (`viewBox="0 0 1680 340"`) of the **2-Branch Comparative Experiment**:
  - `Nhánh Baseline (Real + Deepfake)` vs. `Nhánh Proposed (Real + Deepfake + SD Synthetic)` $\to$ **Shared `Deepfake Detector` Network Diagram** $\to$ `Held-Out Test Set` $\to$ **Conceptual ROC / Generalization Curve Plot** visually showing the expected upward curve shift from Baseline (dashed) to Proposed (solid cobalt) alongside key evaluation criteria (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`).
- **Bottom Row**: 4 Equal-Width Deliverable Cards (*01. Bộ Dữ Liệu Chuẩn Hóa*, *02. Đường Ống Sinh & Lọc*, *03. Thực Nghiệm Đối Chứng*, *04. Báo Cáo Học Thuật*).
