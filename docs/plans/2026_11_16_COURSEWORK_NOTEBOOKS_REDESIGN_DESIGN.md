# DESIGN DOC: TÁI CẤU TRÚC BỘ NOTEBOOK COURSEWORK ĐỒ ÁN 11 THÀNH CHUỖI 4 CHƯƠNG TÍCH HỢP

- **Motivation/Background**: Hệ thống 9 notebook coursework hiện tại (~50MB) đang gặp vấn đề nghiêm trọng về trải nghiệm người dùng: các code cell bị đóng hộp đen trong các hàm vẽ tĩnh (`plot_xyz(); plt.show()`), dung lượng phình to do nhúng base64 làm đơ giật giao diện, và nội dung bị xé nhỏ rời rạc qua 9 file khiến giảng viên và hội đồng khó theo dõi được toàn bộ câu chuyện khoa học của Đề tài 11.
- **Purpose**: Thiết kế bản đặc tả kiến trúc kỹ thuật toàn diện để tái cấu trúc và chuẩn hóa bộ notebook thành chuỗi 4 chương tích hợp (The 4-Part Unified Narrative Suite), đảm bảo 100% code tương tác thực tế, trực quan sinh động, bao quát đầy đủ mọi thành quả của dự án mà không bỏ sót bất kỳ chi tiết nào.
- **Overview Pipeline**: Khảo sát hiện trạng & Bóc tách tài nguyên -> Thiết kế cấu trúc 4 Notebook -> Chuẩn hóa mã tương tác (DataModule, Tensor, DINOv2, Checkpoints) -> Tích hợp bảng đối chiếu chuẩn vàng ($M_0 \to M_3$, 6k OOD, 20.4k Universal) -> Kiểm thử thực thi tự động (Zero error, Zero blank cells).
- **Detailed Plan**: §1 Bối cảnh & Ý định thiết kế; §2 Mục tiêu, Phi mục tiêu & Điều cấm; §3 Các ràng buộc kỹ thuật; §4 So sánh các phương án & Lý do lựa chọn; §5 Đặc tả chi tiết 4 Notebook; §6 Quản lý dữ liệu & Lưu trữ phiên bản cũ; §7 Trách nhiệm của kỹ sư thực thi (Builder Scope).
- **References**: [GEMINI.md](../../AGENTS.md), [docs/PURPOSE.md](../PURPOSE.md), [docs/OVERVIEW.md](../OVERVIEW.md), [MASTER_PROJECT_CROSSCHECK_AND_FACTCHECK_REPORT.md](../reports/MASTER_PROJECT_CROSSCHECK_AND_FACTCHECK_REPORT.md), [M3_DUAL_ENGINE_CHAMPION_REPORT.md](../reports/M3_DUAL_ENGINE_CHAMPION_REPORT.md).
- **Created**: 2026-11-16T21:40:00+07:00
- **Last Updated**: 2026-11-16T21:40:00+07:00

---

## 1. Bối Cảnh & Ý Định Thiết Kế (Intent & The Why)

Đề tài 11 (*Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*) là một công trình nghiên cứu sâu sắc theo định hướng **Dataset-First**, kết hợp giữa:
1. **Đa tạp ảnh thật (Real Manifold)**: 4.000 chân dung FFHQ/CelebA-HQ phân vùng chuẩn Identity-Disjoint.
2. **Động cơ khuếch tán kép (Dual-Engine Diffusion)**: SD 1.5 Base (2.200 ảnh lọc chuẩn) + RealisticVision v5.1 SOTA Photorealistic (6.000 ảnh).
3. **Phễu lọc chất lượng 3 tầng (CurationGate)**: RetinaFace, Laplacian biến thiên động, CLIP Aesthetic.
4. **Khai thác Mẫu Khó sinh học (Hard-Sample Mining)**: 740 ca biên mơ hồ ($|P-0.5| \le 0.15$), 213 Challenge Vault cô lập.
5. **Tiến hóa bộ dò DINOv2 qua Active Learning**: $M_0 \to M_1 \to M_2 \to M_2^*$ và đỉnh cao là Champion $M_3$ Dual-Engine (loại trừ rò rỉ 626 ảnh test).
6. **Sát hạch toàn cầu Universal Benchmark**: 20.400 ảnh trên 34 thuật toán sinh, 8 họ công nghệ, quét ngưỡng quyết định $\tau$.

Tuy nhiên, các notebook hiện có không phản ánh được tầm vóc này do bị che giấu đằng sau các hàm vẽ biểu đồ đóng hộp kín, khiến người đọc có cảm giác như đang xem một slideshow tĩnh hơn là một đồ án Thị giác Máy tính tương tác. Mục đích của thiết kế này là mang lại trải nghiệm thực hành mã nguồn chân thực, hiển thị rõ luồng biến đổi tensor, phân tích sai số sinh học trực quan và dẫn dắt người xem qua một câu chuyện khoa học liền mạch từ đầu đến cuối.

---

## 2. Mục Tiêu, Phi Mục Tiêu & Điều Cấm (Goals, Non-Goals, Anti-Goals)

### 2.1. Mục Tiêu (Goals)
- **Chuẩn hóa cấu trúc 4 Notebook**:
  - `00_master_deepfake_defense_showcase.ipynb`: Bản thuyết trình Master Showcase từ A-Z, tích hợp Live Forensic Inspector.
  - `01_diffusion_dataset_and_curation.ipynb`: Chuyên sâu Giai đoạn 1 (Đa tạp ảnh thật, SD kép, Soft-mask Gaussian $\sigma=4.0$, CurationGate).
  - `02_hard_samples_and_active_learning.ipynb`: Chuyên sâu Giai đoạn 2 (DINOv2 backbone, 740 mẫu khó, 213 Challenge Vault, tiến hóa $M_0 \to M_2^*$).
  - `03_champion_m3_and_universal_benchmark.ipynb`: Chuyên sâu Giai đoạn Mở rộng (Phân vùng $D_3$ sạch, Champion $M_3$, OOD 6.000 ảnh RV5.1, Universal Benchmark 20.400 ảnh).
- **Mã nguồn tương tác thực tế (Hands-on Interactive Code)**:
  - Nạp dữ liệu thực tế bằng `CVDataModule` và `ImageDataset`.
  - Hiển thị kích thước tensor tường minh `(B, C, H, W)` và kiểu dữ liệu `torch.float32`.
  - Tải checkpoint thực tế (`active_learning_m2_best.pt`, `active_learning_m3_best.pt`) và thực hiện suy luận live trên một mini-batch.
  - Tính toán trực tiếp ma trận nhầm lẫn, độ đo ROC-AUC và Fake Recall ngay trong notebook.
- **Tối ưu dung lượng & Trải nghiệm (Clean UX)**:
  - Giảm dung lượng file xuống dưới 2.5MB mỗi notebook bằng cách lưu đồ thị sạch, tối ưu hóa DPI và layout.
  - Tuyệt đối không có cell rỗng, không có cell chỉ chứa `plt.show()`.
  - Tuân thủ nghiêm ngặt Notebook Blueprint trong `.agents/AI_REFERENCE.md`: Cell 0 (Header & Mục tiêu), Cell 1 (Bảng Roadmap 4 cột chuẩn có link trỏ trực tiếp tới module `src/`).

### 2.2. Phi Mục Tiêu (Non-Goals)
- Không huấn luyện lại các mô hình từ đầu (tận dụng toàn bộ 15 checkpoint PyTorch đã được huấn luyện và kiểm chứng).
- Không sinh lại tập dữ liệu (toàn bộ 12.500+ ảnh và manifest JSONL đã có sẵn trên đĩa).
- Không sửa đổi mã nguồn thư viện trong `src/models/` hoặc `src/pipelines/` trừ khi phát hiện lỗi logic nghiêm trọng.

### 2.3. Điều Cấm (Anti-Goals)
- **Cấm mã giả (Zero Mock Data)**: Tuyệt đối không dùng tensor ngẫu nhiên không có checkpoint để giả lập kết quả.
- **Cấm hàm vẽ hộp đen (No Opaque Wrapper Code)**: Không viết các cell chỉ gồm một dòng gọi hàm từ `src/common/visualizer.py` mà không hiển thị luồng chuẩn bị dữ liệu.
- **Cấm rò rỉ đường dẫn tuyệt đối**: 100% đường dẫn trong code và markdown phải là đường dẫn tương đối.
- **Cấm xóa bỏ lịch sử**: Các notebook cũ (01-08) phải được lưu trữ dự phòng an toàn vào `notebooks/coursework/legacy_v1/` trước khi triển khai bộ mới.

---

## 3. Ràng Buộc Kỹ Thuật (Constraints)

1. **Môi trường thực thi**: Python 3.14.8 tại `.venv`, PyTorch 2.x, torchvision, albumentations, matplotlib, opencv-python.
2. **Quy chuẩn miền Thị giác Máy tính (5 CV Domain Invariants)**:
   - Tensor shape: `(B, C, H, W)` cho PyTorch; `(H, W, C)` cho OpenCV/NumPy.
   - Không gian màu: OpenCV đọc `BGR`, phải chuyển đổi tường minh qua `RGB` khi hiển thị với Matplotlib.
   - Quản lý tài nguyên GPU: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
   - Tính tái lập: Khóa seed bằng `seed_everything(42)`.
3. **Quy chuẩn tài liệu**:
   - File markdown viết hoa toàn bộ (`UPPER_SNAKE_CASE.md`), có 7 trường header chuẩn.
   - Liên kết bấm được trực tiếp theo chuẩn relative Markdown links.

---

## 4. So Sánh Các Phương Án & Lý Do Lựa Chọn (Approaches & Tradeoffs)

| Tiêu Chí | Phương Án 1 (Đã chọn): Bộ 4 Chương Tích Hợp | Phương Án 2: Giữ Nguyên Danh Mục 9 Notebook |
| :--- | :--- | :--- |
| **Cấu trúc tập tin** | 4 files: 00 (Master), 01 (Data), 02 (Active Learning), 03 (Champion & Universal) | 9 files: 00 đến 08 theo danh mục cũ |
| **Mạch kể chuyện** | Liền mạch, sư phạm, dẫn dắt theo đúng 3 giai đoạn cốt lõi của đề tài | Bị phân mảnh, người đọc phải mở chuyển đổi qua lại liên tục |
| **Trùng lặp số liệu** | Triệt tiêu hoàn toàn sự trùng lặp giữa các benchmark | Trùng lặp bảng số liệu và biểu đồ giữa 05, 06, 07 và 08 |
| **Dung lượng file** | Siêu nhẹ (< 2.5MB/file), mở nhanh tức thì trên VS Code và GitHub | Nặng nề (~50MB tổng), dễ làm đơ trình duyệt và kernel |
| **Tính tương tác** | Mỗi notebook có code nạp DataModule, chạy DINOv2 và tính toán thực tế | Đa số là các hàm vẽ tĩnh được gán sẵn |
| **Lý do lựa chọn** | **Người dùng đã duyệt chọn Phương án 1**: Tối ưu tối đa độ trực quan và khả năng truyền tải đồ án. | Bị từ chối do phân mảnh và khó duy trì chất lượng đồng đều. |

---

## 5. Đặc Tả Chi Tiết 4 Notebook Coursework Mới

### 5.1. Notebook 00: `00_master_deepfake_defense_showcase.ipynb`
- **Vai trò**: Đại bản doanh trình diễn toàn bộ đồ án (Executive Showcase & Live Forensic Lecture).
- **Cấu trúc nội dung**:
  1. *Cell 0 & Cell 1*: Tiêu đề chuẩn UTH, tóm tắt điều hành, Bảng Roadmap 4 cột và liên kết nhanh tới các báo cáo HTML/MD.
  2. *Chương 1: Bối cảnh & Điểm mù của bộ dò truyền thống*: Trực quan hóa sự khác biệt giữa Face-Swap (cắt dán biên) và Diffusion Latent Inpainting.
  3. *Chương 2: Đa tạp ảnh thật & Động cơ khuếch tán kép*: Lưới ảnh so sánh đối xứng giữa FFHQ, CelebA-HQ, SD 1.5 và RealisticVision v5.1.
  4. *Chương 3: Khai thác Mẫu Khó & Phân tích Dị tật Sinh học*: Không gian nhúng t-SNE, 213 mẫu Challenge Vault, vi vết bất thường ở đồng tử, khóe mắt và răng.
  5. *Chương 4: Vòng lặp Active Learning & Tiến hóa $M_0 \to M_2^* \to M_3$*: Bảng tổng hợp các chỉ số vàng (Golden Scorecard), đường cong ROC/PR và động học di chuyển mẫu.
  6. *Chương 5: Sát hạch toàn cầu Universal Benchmark 20.400 ảnh*: Nạp và trực quan hóa bảng tổng hợp đối chuẩn thực nghiệm (`benchmark_5models_global_summary.csv`, `family_summary.csv`), hiển thị Radar chart 8 họ công nghệ, ma trận nhiệt 34 thuật toán sinh.
  7. *Chương 6: Live Forensic Inspector (Tương tác thực tế)*: Nạp mô hình Champion $M_3$, chạy suy luận trên ảnh chỉ định, trích xuất và hiển thị bản đồ chú ý (Spatial Attention Map qua DINOv2 self-attention weights) và điểm xác suất Deepfake.

### 5.2. Notebook 01: `01_diffusion_dataset_and_curation.ipynb`
- **Vai trò**: Chuyên sâu Giai đoạn 1 — Xây dựng Đa Tạp Ảnh Thật & Động Cơ Sinh Dữ Liệu Khuếch Tán Kép.
- **Cấu trúc nội dung**:
  1. *Roadmap Table*: Liên kết mã nguồn `src/data/raw_ingestion.py`, `src/models/diffusion/`, `src/pipelines/curate_image_dataset.py`.
  2. *Thực hành Đa tạp ảnh thật (4.000 ảnh)*: Nạp dữ liệu qua `DeepfakeImageDataModule`, kiểm tra phân bố giới tính, nhóm tuổi, phụ kiện kính mắt và tính toán phổ màu da thực tế từ các mẫu ảnh thật.
  3. *Thực hành Sinh ảnh Khuếch tán Kép*:
     - Phân tích cơ chế Soft-mask Feathering Gaussian $\sigma=4.0$ trên 68 điểm mốc khuôn mặt.
     - So sánh 3 chế độ sinh: Text-to-Image (T2I), Facial Inpainting, Image-to-Image (Img2Img).
  4. *Thực hành Phễu lọc chất lượng CurationGate*:
     - Đo độ sắc nét Laplacian động bằng `ImageCurationGate` trên các mẫu ảnh thật và ảnh sinh.
     - Nạp biên bản kiểm định thực tế `d0_train_curation_report_att0.json` để bóc tách 545 ảnh bị đào thải (do diện tích khuôn mặt dị thường, mờ nhòe quang học) mà không cần cài đặt thêm thư viện ngoài.

### 5.3. Notebook 02: `02_hard_samples_and_active_learning.ipynb`
- **Vai trò**: Chuyên sâu Giai đoạn 2 — Khai Thác Mẫu Khó & Vòng Lặp Huấn Luyện Chủ Động DINOv2.
- **Cấu trúc nội dung**:
  1. *Roadmap Table*: Liên kết mã nguồn `src/models/detector/image_detector.py`, `src/models/evaluators/hardness.py`, `src/pipelines/run_closed_loop_active_learning.py`.
  2. *Kiến trúc Bộ Dò DINOv2-ViT-S/14*: Nạp mô hình `DeepfakeImageDetector`, cấu hình Multi-Scale Attention Head, hiển thị kích thước tensor đầu vào/đầu ra và các tham số đóng băng.
  3. *Hình học Đa tạp & Khai thác Mẫu Khó*:
     - Đo khoảng cách cosine tới đa tạp ảnh thật (Real-Manifold Proximity qua `src/models/evaluators/hardness.py`).
     - Lọc các mẫu nằm trong dải biên mơ hồ ($|P - 0.5| \le 0.15$).
     - Trực quan hóa 213 mẫu Challenge Vault (102 thật khó + 111 giả tinh vi).
  4. *Khám nghiệm Dị tật Sinh học Vi thể*: Lưới phóng to vùng mắt (mất đối xứng phản chiếu giác mạc), vùng răng (mờ rãnh kẽ răng), vùng tai (sụn biến dạng).
  5. *Vòng lặp Active Learning $M_0 \to M_1 \to M_2 \to M_2^*$*:
     - Nạp checkpoint $M_0$ và $M_2^*$ (`experiments/checkpoints/active_learning_m2_best.pt`), chạy suy luận live trên toàn bộ 213 mẫu Challenge Vault (~2s trên GPU/CPU).
     - Tính toán trực tiếp Confusion Matrix, Fake Recall (11.71% $\to$ 91.89%) và ROC-AUC live để chứng minh bước nhảy vọt khớp 100% báo cáo kiểm toán.

### 5.4. Notebook 03: `03_champion_m3_and_universal_benchmark.ipynb`
- **Vai trò**: Chuyên sâu Giai đoạn 3 & Mở Rộng Đột Phá — Mô Hình Champion $M_3$ & Đối Chuẩn Toàn Cầu.
- **Cấu trúc nội dung**:
  1. *Roadmap Table*: Liên kết mã nguồn `scripts/train_m3_dual_engine.py`, `scripts/evaluate_quad_vault_5models.py`, `scripts/evaluate_universal_benchmark_5models.py`.
  2. *Kiểm toán Liêm chính & Phân vùng $D_3$ Sạch (Zero Identity Leakage)*:
     - Minh chứng cách ly 626 ảnh test RV5.1 sang `unseen_rv51_test.jsonl`.
     - Nạp `data/benchmarks/D3_clean/train.jsonl` qua `DeepfakeImageDataModule` (2.813 thật vs 2.813 giả).
  3. *Huấn luyện & Động học Hội tụ Champion $M_3$*:
     - Phân tích tiến trình 6 epochs với Warm-Start từ $M_2^*$ và Cosine Annealing Differential LR, đạt Validation AUC 0.9846.
  4. *Sát hạch Tứ Giác Thực Nghiệm (Quad-Vault Evaluation)*:
     - Nạp checkpoint $M_3$ (`experiments/checkpoints/active_learning_m3_best.pt`), thực hiện suy luận live trên tập mẫu kiểm tra của Unseen RV5.1 Vault, hiển thị Fake Recall đạt 93.13% (Bypass rate 6.87%).
     - So sánh 5 mô hình qua bảng tổng hợp `benchmark_5models_quad_vault_summary.csv`.
  5. *Sát hạch Toàn cầu Universal Benchmark (20.400 ảnh)*:
     - Nạp và trực quan hóa các bảng đối chuẩn toàn diện (`benchmark_5models_global_summary.csv`, `benchmark_5models_family_summary.csv`) trên 34 phương pháp sinh và 8 họ công nghệ.
     - Quét ngưỡng quyết định $\tau \in [0.005, 0.800]$ và kiểm chứng khả năng kháng nhiễu nén video (FF++ C23 FPR 1.45%).

---

## 6. Các Quyết Định Kỹ Thuật Đã Được Thẩm Định (Builder Decisions Settled)

Sau vòng phản biện độc lập (Builder Check), các chi tiết kỹ thuật sau đã được chốt hạ chính thức:
1. **Phạm vi suy luận Live vs Nạp số liệu vĩ mô**: Đối với tập Universal Benchmark 20.400 ảnh và Quad-Vault 5 mô hình, notebook sẽ nạp các bảng tổng kết và file tensor dự đoán đã có sẵn (`experiments/benchmarks/tables/` và `logs/`) để vẽ radar, heatmap và đồ thị ROC. Suy luận live bằng PyTorch checkpoint sẽ thực hiện trên mini-batch và toàn bộ 213 mẫu Challenge Vault để đảm bảo thời gian chạy của notebook chỉ mất 1-2 phút, không gây nghẽn kernel.
2. **Triển khai CurationGate**: Sử dụng bộ lọc Laplacian và hình học khuôn mặt có sẵn trong `ImageCurationGate`, kết hợp phân tích trực tiếp biên bản lưu trữ `d0_train_curation_report_att0.json` cho 545 mẫu lỗi nhằm bảo đảm tính độc lập, không phụ thuộc vào tải gói ngoài.
3. **Mô-đun nạp dữ liệu chuẩn**: Sử dụng trực tiếp `DeepfakeImageDataModule` và `DeepfakeImageDataset` từ `src.data` (thay cho lớp khung `CVDataModule`) để đồng bộ hoàn toàn với kiến trúc phân vùng của đồ án.
4. **Bản đồ chú ý cho Live Forensic Inspector**: Tận dụng cơ chế attention weights từ `AutoModel` của DINOv2 (`output_attentions=True` với chế độ eager) hoặc patch hook để sinh heatmap không gian trực quan cho các vùng mắt/miệng.
5. **Lưu trữ toàn bộ notebook cũ**: Toàn bộ 9 file notebook cũ (từ `00` đến `08` và `generator_realistic_vision_kaggle.ipynb`) được di chuyển vào `notebooks/coursework/legacy_v1/`.

---

## 6. Quản Lý Dữ Liệu & Lưu Trữ Dự Phòng (Data & Legacy Archiving)

1. **Lưu trữ an toàn các notebook cũ**:
   - Di chuyển toàn bộ các notebook hiện có (`01` đến `08`, `generator_realistic_vision_kaggle.ipynb`) vào thư mục `notebooks/coursework/legacy_v1/`.
   - Giữ lại bản ghi đầy đủ trong `notebooks/coursework/legacy_v1/README.md` giải thích lý do chuyển đổi.
2. **Triển khai bộ 4 notebook mới**:
   - Vị trí: `notebooks/coursework/`
   - Đặt tên chuẩn:
     + `00_master_deepfake_defense_showcase.ipynb`
     + `01_diffusion_dataset_and_curation.ipynb`
     + `02_hard_samples_and_active_learning.ipynb`
     + `03_champion_m3_and_universal_benchmark.ipynb`
3. **Cập nhật tài liệu quản trị**:
   - Cập nhật [notebooks/README.md](../../notebooks/README.md) để phản ánh cấu trúc 4 chương chuẩn mực.
   - Bổ sung hướng dẫn chạy nhanh cho sinh viên và hội đồng chấm điểm.

---

## 7. Trách Nhiệm Của Kỹ Sư Thực Thi (What's Left To The Builder)

1. **Xây dựng kịch bản tự động hóa sinh và thực thi notebook**:
   - Viết kịch bản `scripts/build_unified_coursework_suite.py` để sinh ra 4 file notebook hoàn chỉnh, tích hợp mã nguồn tương tác sạch và chạy live qua IPython InteractiveShell để đảm bảo 100% cell xanh và output chuẩn xác.
2. **Kiểm tra định dạng & Độ mượt mà**:
   - Đảm bảo mỗi biểu đồ được render ở độ phân giải chuẩn (120-150 DPI), không lưu ảnh kích thước lớn thừa thãi làm phình to JSON.
   - Xác minh toàn bộ các liên kết markdown trong Cell 1 trỏ đúng vào các file hiện hữu trong `src/` và `docs/`.
3. **Kiểm thử khói & Thẩm định toàn vẹn**:
   - Chạy `make test-smoke` để bảo đảm không có xung đột mã nguồn.
   - Chạy thử nghiệm load model trên GPU/CPU trong từng notebook để xác nhận tính khả thi độc lập.
