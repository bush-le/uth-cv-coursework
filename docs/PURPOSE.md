# Project Purpose & Coursework Requirements (`docs/PURPOSE.md`)

- **Motivation/Background**: Canonical source of truth defining the core objectives, closed-loop problem statement, evaluation criteria, and operational boundaries for the UTH Computer Vision Coursework on "Stable Diffusion-Based Synthetic Facial Image Dataset Generation, `dinoViT` Hard-Sample Mining, and Detector Fine-Tuning" (Topic 11: "Stable diffusion for data generation" referencing Hugging Face Computer Vision Course Unit 10, updated per instructor feedback).
- **Purpose**: Provide locked project scope, 2D image data contracts, the 3-phase closed-loop experimental design (`SD Data Generation -> Baseline dinoViT Bypass/Hard-Sample Mining -> Fine-Tuning & Proof of Utility`), and quantitative success metrics.
- **Overview Pipeline**: Real face ingestion -> Phase 1 Stable Diffusion synthetic deepfake generation & quality curation -> Phase 2 Baseline `dinoViT` evaluation (Reject vs. Bypass $\rightarrow$ Hard Sample extraction) -> Phase 3 Fine-tuning `dinoViT` on `Baseline Data + Synthetic Data + Hard Samples` -> Benchmark proof of utility.
- **Detailed Plan**: §1 Context & Academic Scope; §2 Instructor-Refined Problem Statement & Closed-Loop Design; §3 Operational Boundaries & Quantitative Success Metrics; §4 Hardware & Compute Constraints; §5 Required Deliverables.
- **References**: `docs/README.md`, `docs/OVERVIEW.md`, `docs/progress/PROJECT_ROADMAP.md`, `GEMINI.md`, [Hugging Face CV Course Unit 10](https://huggingface.co/learn/computer-vision-course/unit10/datagen-diffusion).
- **Created**: 2026-11-07T00:05:00+07:00
- **Last Updated**: 2026-11-08T17:45:00+07:00

---

## 1. Context & Academic Scope

- **Course**: Computer Vision (Thị giác Máy tính) — Trường Đại học Giao thông Vận tải TP.HCM (UTH).
- **Project Role**: Major Coursework Project / Đồ án môn học (Topic 11 — Nhóm 4).
- **Topic Name (English)**: Stable Diffusion-Based Synthetic Facial Dataset Generation with DINO ViT Hard-Sample Mining for Deepfake Detection.
- **Topic Name (Tiếng Việt)**: Sinh dữ liệu khuôn mặt Deepfake bằng Stable Diffusion kết hợp Khai phá Mẫu khó (Hard-Sample Mining) và Tinh chỉnh mô hình DINO ViT (`dinoViT`) phục vụ phát hiện Deepfake.
- **Course Syllabus Reference**: Topic 11 — *Stable diffusion for data generation* (`https://huggingface.co/learn/computer-vision-course/unit10/datagen-diffusion`).
- **Methodology**: Closed-Loop Adversarial Hard-Sample Mining, Modular Clean Architecture, Strict Reproducibility.

---

## 2. Instructor-Refined Problem Statement & Closed-Loop Design

- **Core Academic Question (Yêu cầu trọng tâm từ Giảng viên)**:
  - *Làm thế nào để chứng minh trực tiếp tác dụng của bộ dữ liệu ảnh giả (`Synthetic DF`) sinh ra từ Stable Diffusion đối với một mô hình phát hiện mạnh như `dinoViT`?*
  - Nếu chỉ sinh dữ liệu giả ngẫu nhiên mà không kiểm chứng xem mô hình Baseline có bị đánh lừa hay không, ta không chứng minh được giá trị thực tiễn của nhà máy dữ liệu.
- **Proposed 3-Phase Closed-Loop Solution**:
  1. **Phase 1 — Sinh dữ liệu & Lọc chất lượng (`img + Stable Diffusion -> Synthetic DF img`)**:
     - Từ ảnh khuôn mặt thật (`CelebA / FFHQ`), sử dụng [`StableDiffusionEngine`](../src/models/diffusion/stable_diffusion_engine.py) (`sd_text2img`, `sd_inpaint`, `sd_img2img`) để sinh ảnh khuôn mặt Deepfake (`DF img`).
     - Áp dụng cổng kiểm duyệt đa phương thức ([`ImageCurationGate`](../src/pipelines/curate_image_dataset.py): Laplacian Sharpness, CLIP Score, Face Landmark Retention) để đảm bảo ảnh giả có chất lượng thị giác chân thực.
  2. **Phase 2 — Benchmark Baseline `dinoViT` & Khai phá Mẫu khó (`Baseline dinoViT -> Reject / Bypass -> Hard Samples`)**:
     - Sử dụng mô hình **DINO ViT (`dinoViT`)** ([`DeepfakeImageDetector(backbone="dino_vit")`](../src/models/detector/image_detector.py)) làm Baseline Detector để đánh giá toàn bộ tập ảnh `Synthetic DF`:
       - **Reject (Bắt được)**: Các mẫu ảnh giả có $P(\text{fake}) \ge \tau$ ($\tau = 0.50$) $\rightarrow$ Baseline đã nhận diện được, xếp vào tập `Regular Synthetic Data`.
       - **Bypass (Vượt qua / Đánh lừa được Baseline)**: Các mẫu ảnh giả có $P(\text{fake}) < \tau$ (bị phân loại nhầm thành ảnh thật — False Negative) $\rightarrow$ được trích xuất tự động qua [`mine_hard_samples.py`](../src/pipelines/mine_hard_samples.py) thành tập **Hard Samples** (`is_hard_sample = True`).
  3. **Phase 3 — Tinh chỉnh (`Fine-Tune`) & Chứng minh Tác dụng (`Data Baseline + Syn Data + Hard Samples -> Catch Bypassed DF`)**:
     - Hợp nhất 3 nguồn dữ liệu: **`Data Baseline` + `Synthetic Data` + `Hard Samples`** (kèm cơ chế *Hard-Sample Oversampling* và *Weighted BCE Loss* $\lambda_{\text{hard}} = 2.0$).
     - Thực hiện Fine-tune các khối Transformer phía trên của `dinoViT` ([`train_image_detector.py`](../src/pipelines/train_image_detector.py)).
     - **Chứng minh tác dụng**: Đánh giá đối chứng trước và sau khi Fine-tune ([`evaluate_image_benchmark.py`](../src/pipelines/evaluate_image_benchmark.py)). Mô hình sau khi Fine-tune bắt được các mẫu từng bị bypass (`Hard-Sample Recall` tăng vọt, `Bypass Rate` giảm mạnh, `ROC-AUC` tổng thể tăng cao trên tập kiểm thử độc lập).

---

## 3. Operational Boundaries & Quantitative Success Metrics

- **Primary Quantitative Targets**:
  - **Phase 1 Generation Quality**:
    - **Laplacian Sharpness Score**: $\ge 50.0$ (loại bỏ ảnh mờ do VAE).
    - **CLIP Score (Image-Text Alignment)**: $\ge 0.28$ trên các mẫu điều khiển bằng prompt.
    - **Face Detection Retention**: $\ge 95\%$ ảnh giả giữ nguyên cấu trúc giải phẫu khuôn mặt.
  - **Phase 2 Hard-Sample Mining (`dinoViT` Baseline)**:
    - Xác định rõ tỷ lệ `Bypass Rate` của tập `Synthetic DF` trước mô hình `dinoViT` chưa tinh chỉnh và xuất tự động `hard_samples_manifest.jsonl`.
  - **Phase 3 Downstream Fine-Tuned `dinoViT` Benchmark (Proof of Utility)**:
    - **Overall ROC-AUC**: $\ge 92.0\%$ và **Accuracy**: $\ge 85.0\%$ trên tập test độc lập (Identity-Disjoint).
    - **Hard-Sample Recall Gain**: Tăng mạnh khả năng bắt giữ trên nhóm mẫu khó (từ $0\%$ ở thời điểm bị bypass trên Baseline lên $\ge 85.0\%$ trên tập kiểm thử Hard Samples sau khi Fine-tune).
    - **Bypass Rate Reduction**: Giảm mạnh tỷ lệ lọt lưới của ảnh giả Stable Diffusion.
- **Explicit Non-Goals (Strictly Out of Scope)**:
  - *No Video Diffusion or Facial De-occlusion/Segmentation*: Dự án tập trung 100% vào bài toán ảnh tĩnh 2D `(B, 3, H, W)` theo đúng yêu cầu chỉnh sửa của giảng viên.

---

## 4. Hardware & Compute Constraints

- **Execution Environment**: Linux workstation / Google Colab CUDA environment.
- **Compute Budget**: Single GPU ($256 \times 256$ hoặc $512 \times 512$, AMP `fp16`, DINO ViT partial-block fine-tuning).
- **Framework Target**: Python 3.10+, PyTorch 2.2+, Torchvision, Pillow, NumPy, Hugging Face Diffusers.

---

## 5. Required Deliverables

1. **2D Deepfake & Hard-Sample Data Schema**: `ImageMetadata` (hỗ trợ `is_hard_sample`, `baseline_score`, `data_source`), `ImageSample`, và `DeepfakeImageDataModule` (hỗ trợ *Identity-Disjoint Splitting* và *Hard-Sample Oversampling*).
2. **Phase 1 — Stable Diffusion Generation & Curation Engine**: `src/models/diffusion/stable_diffusion_engine.py`, `src/pipelines/generate_deepfake_images.py`, và `src/pipelines/curate_image_dataset.py`.
3. **Phase 2 — `dinoViT` Baseline Detector & Hard-Sample Miner**: `DinoViTBackbone` & `DeepfakeImageDetector` (`src/models/detector/image_detector.py`) cùng pipeline khai phá mẫu vượt rào `src/pipelines/mine_hard_samples.py`.
4. **Phase 3 — Fine-Tuning & Comparative Benchmark Suite**: `src/pipelines/train_image_detector.py` và `src/pipelines/evaluate_image_benchmark.py` (`compare_baseline_vs_finetuned`).
5. **Technical Coursework Report & Slide Deck**: Bộ báo cáo học thuật và slide thuyết trình 11 slides tại `docs/reports/`.
