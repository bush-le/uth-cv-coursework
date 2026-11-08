# PRESENTATION_DEFENSE_QA_CHEAT_SHEET.md — Bộ Phản Biện & Kịch Bản Ứng Khẩu Đỉnh Cao (Academic Defense & Q&A Auditor)

- **Motivation/Background**: Xây dựng tài liệu phản biện học thuật chiến lược và kịch bản ứng khẩu 30 giây cho bài bảo vệ đồ án: "Stable Diffusion-Based Synthetic Facial Dataset Generation with DINO ViT Hard-Sample Mining for Deepfake Detection" (Đề tài 11, cập nhật theo luồng `dinoViT` + Hard Sample của Giảng viên).
- **Purpose**: Đóng vai trò là cẩm nang bỏ túi giúp sinh viên Nhóm 4 bảo vệ xuất sắc trước các câu hỏi của Giảng viên hướng dẫn và Hội đồng chấm thi.
- **Overview Pipeline**: Phân tích câu hỏi trọng tâm của Giảng viên -> Giải thích cơ chế `Phase 1 (SD Synthesis) -> Phase 2 (dinoViT Bypass/Hard Sample) -> Phase 3 (Fine-Tune & Proof of Utility)` -> Kịch bản trả lời 30 giây.
- **Detailed Plan**: §1 Tâm thế Phản biện; §2 Bộ 6 Câu Hỏi Trọng Tâm & Kịch Bản Ứng Khẩu 30 Giây; §3 Bảng Tóm Tắt Thuật Ngữ & Module Code.
- **References**: `docs/PURPOSE.md`, `docs/reports/SYSTEM_DEFINE_AND_DESIGN_REPORT.md`, `src/models/detector/image_detector.py`, `src/pipelines/mine_hard_samples.py`, `src/pipelines/train_image_detector.py`.
- **Created**: 2026-11-07T01:05:00+07:00
- **Last Updated**: 2026-11-08T17:52:00+07:00

---

## 1. 🎯 TÂM THẾ PHẢN BIỆN & ĐIỂM NHẤN MỚI CỦA ĐỀ TÀI

Điểm ăn điểm tuyệt đối của đồ án sau khi chỉnh sửa theo góp ý của Giảng viên là **không chỉ sinh dữ liệu giả một chiều**, mà thiết lập **vòng lặp kiểm chứng khép kín**:
1. **Phase 1**: `Ảnh thật + Stable Diffusion` $\rightarrow$ `Ảnh giả đạt chuẩn (Synthetic DF img)`.
2. **Phase 2**: Cho `Synthetic DF img` đi qua **Baseline `dinoViT`** $\rightarrow$ tách những mẫu đánh lừa được mô hình (`Bypass`, $P(\text{fake}) < 0.50$) thành **Hard Samples**.
3. **Phase 3**: Fine-tune `dinoViT` trên **`Data Baseline + Synthetic Data + Hard Samples`** $\rightarrow$ mô hình bắt được các mẫu từng bị bypass $\implies$ **chứng minh trực tiếp tác dụng của dữ liệu tổng hợp!**

---

## 2. ⚡ BỘ 6 CÂU HỎI TRỌNG TÂM & KỊCH BẢN ỨNG KHẨU 30 GIÂY

### Câu Hỏi 1: "Làm sao các em chứng minh được bộ dữ liệu sinh ra từ Stable Diffusion ở Phase 1 thực sự có tác dụng?"
- **Kịch bản trả lời 30 giây**:
  > *"Dạ thưa Thầy/Cô, để chứng minh tác dụng thực tiễn theo đúng định hướng của Thầy/Cô, nhóm thiết kế quy trình 3 bước khép kín:*
  > *Thứ nhất, ảnh giả sinh từ Stable Diffusion ở Phase 1 được đưa qua mô hình **Baseline `dinoViT`** ở Phase 2 (`mine_hard_samples.py`). Những ảnh giả nào có $P(\text{fake}) < 0.50$ tức là đã **bypass (qua mặt)** được Baseline sẽ được gom thành tập **Hard Samples**.*
  > *Thứ hai, ở Phase 3, nhóm gộp **Data Baseline + Synthetic Data + Hard Samples** để fine-tune lại `dinoViT`. Kết quả đối chứng trên tập Test độc lập cho thấy mô hình sau khi fine-tune bắt giữ thành công các mẫu từng bị bypass, kéo giảm mạnh tỷ lệ Bypass Rate và tăng vọt ROC-AUC. Đó là bằng chứng định lượng trực tiếp chứng minh tác dụng của dữ liệu tổng hợp ạ."*

---

### Câu Hỏi 2: "Tại sao nhóm chọn `dinoViT` (DINO Vision Transformer) làm mô hình Baseline Benchmark?"
- **Kịch bản trả lời 30 giây**:
  > *"Dạ thưa Thầy/Cô, các CNN truyền thống như ResNet-18 rất dễ bị overfitting vào nhiễu cục bộ. Trong khi đó, `dinoViT` (Vision Transformer huấn luyện tự giám sát theo phương pháp DINO) sử dụng cơ chế Self-Attention trên các patch $16\times 16$ để học biểu diễn cấu trúc ngữ nghĩa toàn cục của khuôn mặt.*
  > *Khi lấy một mô hình mạnh như `dinoViT` làm Baseline mà ảnh giả Stable Diffusion vẫn **bypass** được, thì các mẫu **Hard Samples** thu được thực sự là những mẫu cực kỳ chất lượng và có giá trị học thuật cao."*

---

### Câu Hỏi 3: "Thế nào là một mẫu `Reject` và một mẫu `Bypass -> Hard Sample`? Ngưỡng tách được cài đặt ở đâu?"
- **Kịch bản trả lời 30 giây**:
  > *"Dạ, trong module `src/pipelines/mine_hard_samples.py`, nhóm cho toàn bộ ảnh giả đã qua kiểm duyệt chất lượng ($y = 1$) đi qua mô hình Baseline `dinoViT` đã đóng băng trọng số (`freeze_backbone()`):*
  > *- Nếu xác suất dự đoán $P(\text{fake}) \ge 0.50$: mô hình đã phát hiện được (tức là **Reject** ảnh giả), mẫu được lưu với `data_source='synthetic'`.*
  > *- Nếu $P(\text{fake}) < 0.50$: mô hình bị đánh lừa thành ảnh thật (tức là **Bypass**), hệ thống tự động gắn cờ `is_hard_sample=True` và xuất ra `hard_samples_manifest.jsonl` để ưu tiên huấn luyện ở Phase 3 ạ."*

---

### Câu Hỏi 4: "Khi gộp `Data Baseline + Syn Data + Hard Samples` để Fine-tune `dinoViT`, nhóm huấn luyện như thế nào để mô hình tập trung bắt được Hard Samples?"
- **Kịch bản trả lời 30 giây**:
  > *"Dạ thưa Thầy/Cô, nhóm áp dụng đồng thời 3 kỹ thuật trong `src/pipelines/train_image_detector.py`:*
  > *1. **Partial Block Unfreezing**: Chỉ mở khóa các khối Transformer Encoder phía trên (`unfreeze_last_n_blocks=2`) và Classifier Head của `dinoViT` để tránh quên kiến thức gốc.*
  > *2. **Hard-Sample Oversampling**: Nhân đôi tần suất xuất hiện của `Hard Samples` trong DataLoader tập Train (`hard_sample_oversample_factor=2`).*
  > *3. **Weighted BCE Loss**: Nhân hệ số phạt $\lambda_{\text{hard}} = 2.0$ trực tiếp vào hàm loss cho các mẫu có cờ `is_hard_sample=True`, ép biên quyết định dịch chuyển bao phủ vùng điểm mù."*

---

### Câu Hỏi 5: "Làm sao đảm bảo khi đưa Hard Samples vào Fine-tune thì không bị Data Leakage giữa tập Train và tập Test?"
- **Kịch bản trả lời 30 giây**:
  > *"Dạ, trong `DeepfakeImageDataModule`, nhóm cài đặt cơ chế **Identity-Disjoint Partitioning** dựa trên trường `source_real_id`. Toàn bộ ảnh thật gốc, ảnh `Synthetic` thông thường và ảnh `Hard Sample` sinh ra từ cùng một người thật bắt buộc phải nằm chung trong một tập (Train, Val hoặc Test). Tập Test chỉ chứa các danh tính người thật và các mẫu Hard Sample của những người mà mô hình chưa từng nhìn thấy ở tập Train ạ."*

---

### Câu Hỏi 6: "Làm sao biết mẫu Bypass là do ảnh giả chân thực chứ không phải do Stable Diffusion sinh ra ảnh lỗi/mờ làm rối mô hình?"
- **Kịch bản trả lời 30 giây**:
  > *"Dạ thưa Thầy/Cô, trước khi đưa vào `dinoViT` ở Phase 2, mọi ảnh sinh từ Stable Diffusion ở Phase 1 đều phải vượt qua cổng kiểm duyệt `ImageCurationGate` (`curate_image_dataset.py`) gồm 3 tiêu chí: Độ sắc nét Laplacian $\ge 50.0$, độ khớp ngữ nghĩa CLIP $\ge 0.28$ và phát hiện khuôn mặt hợp lệ. Nhờ đó, chỉ những bức ảnh giả sắc nét, đúng cấu trúc giải phẫu khuôn mặt mới được dùng để kiểm tra Bypass và trích xuất Hard Samples ạ."*

---

## 3. 📊 BẢNG TÓM TẮT ÁNH XẠ YÊU CẦU GIẢNG VIÊN VÀO CODEBASE

| Yêu cầu của Giảng viên | Module hiện thực trong `src/` | Hàm / Class cốt lõi |
| :--- | :--- | :--- |
| `img + stable diffusion -> synthetic img (DF img)` | `src/pipelines/generate_deepfake_images.py`<br/>`src/pipelines/curate_image_dataset.py` | `StableDiffusionEngine`, `run_image_generation()`, `ImageCurationGate` |
| `baseline model dinoViT (benchmark)` | `src/models/detector/image_detector.py` | `DinoViTBackbone`, `DeepfakeImageDetector(backbone="dino_vit")` |
| `reject/bypass DF? neu bypass -> hard sample` | `src/pipelines/mine_hard_samples.py` | `mine_hard_samples()`, `HardSampleMiningReport` |
| `data baseline + syn data + hard sample -> finetune` | `src/pipelines/train_image_detector.py` | `combine_training_manifests()`, `train_deepfake_image_detector()` |
| `neu bat duoc -> co tac dung` | `src/pipelines/evaluate_image_benchmark.py`<br/>`src/common/metrics.py` | `compare_baseline_vs_finetuned()`, `compute_bypass_rate()`, `compute_hard_sample_recall()` |
