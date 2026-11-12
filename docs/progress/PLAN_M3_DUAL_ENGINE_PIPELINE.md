# PLAN_M3_DUAL_ENGINE_PIPELINE.md — Plan Huấn Luyện Champion Model M3 và Đối Chuẩn Quad-Vault

- **Motivation/Background**: Xây dựng mô hình dò tìm Deepfake $M_3$ Dual-Engine kết hợp hai bộ sinh Stable Diffusion 1.5 và RealisticVision v5.1 (kho 6k ảnh), giải quyết dứt điểm câu hỏi về khả năng khái quát hóa liên bộ sinh (Cross-Generator Generalization) và tối ưu hóa ROC-AUC trên Universal Benchmark mà không gây rò rỉ dữ liệu.
- **Purpose**: Bản kế hoạch hành động chi tiết chuẩn bị dữ liệu sạch $D_3$, huấn luyện mô hình $M_3$, đánh giá tứ giác kiểm thử (Quad-Vault Benchmark) và cập nhật hệ thống báo cáo/notebooks của dự án UTH.
- **Overview Pipeline**: Phân vùng dữ liệu $D_3$ sạch (chống rò rỉ 626 ảnh test) -> Huấn luyện $M_3$ Warm-Start từ $M_2^*$ (Max 6 epochs + Early Stopping + Differential LR) -> Đánh giá đối chuẩn 5 mô hình trên 4 kho kiểm thử -> Cập nhật trực quan hóa & Báo cáo bảo vệ đồ án.
- **Detailed Plan**: 
  - §1 Mục tiêu & Ràng buộc toàn cục (Global Constraints & Non-Negotiable Invariants);
  - §2 Cấu trúc phân vùng dữ liệu $D_3$ sạch & Kho OOD kiểm thử mù;
  - §3 Thiết kế kiến trúc huấn luyện $M_3$ (Differential LR, Optimizer, Scheduler);
  - §4 Kế hoạch kiểm thử & Đánh giá đối chuẩn Quad-Vault 5 mô hình;
  - §5 Danh mục nhiệm vụ theo dõi (Task Checklist with DoD);
  - §6 Kế hoạch Commit nguyên tử & An toàn mã nguồn.
- **References**: [GEMINI.md](file:///home/bush/Projects/uth-cv-coursework/GEMINI.md), [docs/PURPOSE.md](file:///home/bush/Projects/uth-cv-coursework/docs/PURPOSE.md), [scripts/train_m2_refined.py](file:///home/bush/Projects/uth-cv-coursework/scripts/train_m2_refined.py).
- **Created**: 2026-11-12T11:10:00+07:00
- **Last Updated**: 2026-11-12T11:10:00+07:00

---

## 1. 🎯 Mục Tiêu & Ràng Buộc Toàn Cục (Global Constraints)

1. **Mục tiêu**:
   - Xây dựng tập $D_3$ sạch cân bằng 1:1 nhãn Real/Fake ($\approx 5.626$ mẫu).
   - Huấn luyện mô hình $M_3$ (Dual-Engine Champion) kế thừa từ $M_2^*$, nâng cao Fake Recall trên RealisticVision v5.1 lên $\ge 85\%$ và cải thiện nhận diện Latent Diffusion trên Universal Benchmark.
   - Bảo toàn tính trung thực học thuật tuyệt đối: **Zero Identity Leakage**.
2. **Ràng buộc phần cứng**:
   - GPU: NVIDIA GeForce RTX 3050 Laptop (3.73 GB VRAM).
   - Yêu cầu bộ nhớ: `batch_size = 16`, `torch.cuda.amp.autocast()` với `bfloat16`/`float16`.
3. **Ràng buộc mã nguồn**:
   - Không chạy training trong Jupyter Notebook. Tất cả pipeline phải nằm trong `scripts/` hoặc `src/pipelines/`.
   - Commit Git nguyên tử sau mỗi đơn vị tác vụ hoàn thành.

---

## 2. 🗂️ Cấu Trúc Phân Vùng Dữ Liệu $D_3$ Sạch

### 2.1. Kiểm soát 6.000 ảnh RealisticVision v5.1
- **Kho nguồn**: `data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl`
- **Bộ lọc danh tính nghiêm ngặt**:
  - `626 ảnh test-derived`: **LOẠI TRỪ 100% KHỎI TẬP TRAIN** $\to$ Đưa vào `data/benchmarks/unseen_rv51_test.jsonl` (OOD Test Vault).
  - `600 ảnh val-derived`: **LOẠI TRỪ 100% KHỎI TẬP TRAIN** $\to$ Đưa vào tập `val.jsonl`.
  - `2.774 ảnh train-derived` + `2.000 ảnh text2img` sạch: **4.774 ảnh khả dụng để train**.
- **Tập Fake $D_3$ Train được chọn mẫu phân tầng (Tổng: 2.813 ảnh)**:
  - $800$ ảnh SD 1.5 ngẫu nhiên từ $D_0$.
  - $629$ ảnh Hard Samples khai thác từ $D_2$.
  - $1.384$ ảnh RV5.1 sạch (484 Text-to-Image + 450 Inpaint + 450 Img2Img).
- **Tập Real $D_3$ Train**: Đúng $2.813$ ảnh từ `data/partitions/train_reals.jsonl`.

---

## 3. ⚙️ Kiến Trúc Huấn Luyện Mô Hình $M_3$

1. **Khởi tạo**: Warm-Start từ `experiments/checkpoints/active_learning_m2_best.pt`.
2. **Cấu hình Unfreeze**:
   - Đóng băng Layers 0 → 10 (20.01M tham số, $91.94\%$).
   - Mở khóa Layer 11 + Classifier Head (1.78M tham số, $8.06\%$).
3. **Tối ưu hóa (Optimizer & Scheduler)**:
   - AdamW với Weight Decay $1 \times 10^{-4}$.
   - Tốc độ học phân tầng (Differential LR):
     - Classifier Head: $\eta = 3 \times 10^{-5}$.
     - Backbone Layer 11: $\eta = 3 \times 10^{-6}$ (nhỏ gấp 10 lần để chống quên đặc trưng).
   - Scheduler: `CosineAnnealingLR` với `T_max = max_epochs`.
4. **Hàm mất mát (Loss Function)**:
   - `BCEWithLogitsLoss` có trọng số mẫu khó $\lambda_{\text{hard}} = 1.10$.
5. **Thời lượng & Dừng sớm**:
   - `max_epochs = 6`.
   - Đánh giá Validation sau mỗi epoch, tự động lưu `active_learning_m3_best.pt` theo `val_auc` / `val_loss`.
   - Early Stopping với patience = 2 epochs nếu validation suy giảm.

---

## 4. 📊 Kế Hoạch Đánh Giá Quad-Vault 5 Mô Hình

Đánh giá 5 mô hình ($M_0, M_1, M_2^{\text{cold}}, M_2^*, M_3$) trên 4 kho kiểm thử:
1. **Master Frozen Test** (1.178 ảnh SD 1.5): Đo độ bền bỉ nội miền.
2. **Cumulative Challenge Test** (213 ảnh Hard Samples): Đo khả năng chế ngự mẫu khó.
3. **Unseen RV5.1 Test Vault** (626 ảnh): Đo khả năng tổng quát hóa trên người lạ với RealisticVision v5.1.
4. **Universal Benchmark** (20.400 ảnh): Đo độ nhạy trên 27 phương pháp sinh ảnh toàn cầu.

---

## 5. 📋 Danh Mục Nhiệm Vụ Chi Tiết (Checklist)

- [x] **Task 1: Xây dựng script phân vùng $D_3$ sạch**
  - File: `scripts/build_d3_clean_dataset.py`
  - Output: `data/benchmarks/D3_clean/train.jsonl`, `val.jsonl`, `unseen_rv51_test.jsonl`.
  - Verification: Đạt chuẩn 1:1 (2.813 Real, 2.813 Fake), kiểm tra giao thoa ID với `test_reals.jsonl` = 0 leaked IDs.
- [x] **Task 2: Lập trình Pipeline huấn luyện $M_3$ Dual-Engine**
  - File: `scripts/train_m3_dual_engine.py`
  - Output: Lưu checkpoint `experiments/checkpoints/active_learning_m3_best.pt`.
  - Verification: 6 epochs hội tụ mượt mà, Val AUC đạt đỉnh 0.9846, Train loss giảm xuống 0.0831.
- [x] **Task 3: Thực thi huấn luyện $M_3$ trên GPU**
  - Đã chạy hoàn tất trong 1483.3s trên GPU RTX 3050 Laptop, lưu full-state checkpoint `active_learning_m3_best.pt`.
- [x] **Task 4: Chạy Đánh giá Đối Chuẩn Quad-Vault & Universal 5 Mô Hình**
  - Files: `scripts/evaluate_quad_vault_5models.py`, `scripts/evaluate_universal_benchmark_5models.py`.
  - Outputs: Bảng `benchmark_5models_quad_vault_summary.csv`, `benchmark_5models_global_summary.csv`, biểu đồ `benchmark_5models_vault_recall_comparison.png`, `benchmark_5models_roc_curves.png`, `benchmark_5models_family_recall.png`.
- [x] **Task 5: Tổng hợp Báo Cáo & Cập Nhật Tài Liệu Dự Án**
  - File: `docs/reports/M3_DUAL_ENGINE_CHAMPION_REPORT.md`.
  - Verification: Chạy `make test-smoke` đạt 12/12 passing.
- [x] **Task 6: Hoàn Tất Mục Tiêu & Git Commit Nguyên Tử**
