# AI-FIRST LIFECYCLE - VÒNG ĐỜI KỸ THUẬT AI-FIRST TOÀN DIỆN

AI hỗ trợ từ phân tích bài toán, sau đó lần lượt triển khai dữ liệu, mô hình và tích hợp hệ thống qua 9 giai đoạn kỹ thuật.

---

## Sơ đồ Tiến trình AI-First (Progression Roadmap)

```text
[ 1. Problem ]
      │
      ▼
[ 2. AI Solution Design ] ────► (DEFINE & DESIGN: Xác định kiến trúc, rủi ro, WBS)
      │
      ▼
[ 3. Data Pipeline ]      ────► (Xây dựng ETL, tiền xử lý, kiểm tra rò rỉ dữ liệu)
      │
      ▼
[ 4. Model Architecture ] ────► (Xây dựng backbone, loss function, kiểm tra shapes)
      │
      ▼
[ 5. Training Loop ]      ────► (Điều phối tối ưu hóa, early stopping, checkpointing)
      │
      ▼
[ 6. Evaluation & Bench ] ────► (Đo lường định lượng mAP, PSNR, FPS, phân tích lỗi)
      │
      ▼
[ 7. Full Integration ]   ────► (Tích hợp pipelines, hoàn thiện báo cáo và commit)
```

---

## 9 Giai đoạn Kỹ thuật Chi tiết (Macro Engineering Phases)

### 1. DEFINE (Định nghĩa Bài toán & Phạm vi)
Mục tiêu: Làm rõ bài toán trước khi viết bất kỳ dòng mã nào. Bản đặc tả DEFINE bao gồm 7 trụ cột cốt lõi:

| Trụ cột | Mô tả Yêu cầu |
| :--- | :--- |
| **Project Name** | Tên dự án / bài lab chuẩn hóa (ví dụ: `Object Detection with Feature Pyramid Networks`). |
| **Problem Statement** | Mô tả rõ ràng bài toán cần giải quyết, bối cảnh thực tế và lý do giải pháp hiện tại chưa đáp ứng. |
| **Target Users** | Người thụ hưởng hoặc hệ thống tiếp nhận (sinh viên, giảng viên, hệ thống nhúng, edge device). |
| **Intended Output** | Đầu ra mong đợi cụ thể: Tensor bounding boxes, segmentation masks, classification labels, file JSON metrics. |
| **Core Features** | Danh sách 3-5 tính năng kỹ thuật cốt lõi bắt buộc phải có trong phiên bản hoàn thiện. |
| **Technical Constraints** | Ràng buộc về phần cứng (VRAM giới hạn, CPU-only), giới hạn thư viện, latency tối đa cho phép (ms/frame). |
| **Evaluation Metrics** | Bộ chỉ số định lượng đánh giá thành công: Accuracy, Top-5, mAP@[.5:.95], IoU, F1-Score, PSNR, SSIM, FPS. |

---

### 2. DESIGN (Thiết kế Kiến trúc Hệ thống)
Mục tiêu: Chuyển đổi định nghĩa bài toán thành bản thiết kế kỹ thuật khả thi gồm 7 thành phần cấu trúc:

| Thành phần | Đặc tả Kỹ thuật |
| :--- | :--- |
| **Architecture System** | Sơ đồ khối tổng thể thể hiện tương tác giữa Data, Model, Training và Evaluation. |
| **Modules** | Phân chia trách nhiệm thành các package độc lập trong `src/`: `common`, `data`, `models`, `pipelines`. |
| **Folder Structure** | Ánh xạ chính xác các file mã nguồn và cấu hình theo chuẩn cấu trúc của dự án. |
| **Dataflow** | Luồng biến đổi dữ liệu: Raw Image → Tensor BCHW → Augmentation → Backbone Features → Prediction → Loss/Metrics. |
| **Interfaces / API** | Chữ ký hàm (function signatures), class signatures, tensor shapes `(B, C, H, W)` và kiểu dữ liệu đầu vào/đầu ra. |
| **Milestones** | Lộ trình chia theo mốc: M1 (Data & Baseline), M2 (Model Improvement), M3 (Evaluation & Benchmark). |
| **Technical Risks & Mitigation** | Nhận diện rủi ro kỹ thuật (overfitting, VRAM OOM, data leakage, class imbalance) và giải pháp dự phòng tương ứng. |

---

### 3. BREAKDOWN (Phân rã Công việc - WBS)
- Lập Work Breakdown Structure (WBS) chia nhỏ bài toán thành các Task độc lập.
- Thời lượng mỗi Task dao động từ **30 phút đến 4 giờ** (không quá 1 ngày làm việc).
- Mỗi task phải có **Definition of Done (DoD)** rõ ràng:
  - Code đã viết và pass `make lint`.
  - Có unit test với độ bao phủ xác thực.
  - Không vi phạm các điều cấm kỵ (Negative Constraints).

---

### 4. BUILD MVP (Xây dựng Bản Mẫu Tối giản - Tracer Bullet)
- Triển khai Minimal Viable Pipeline từ đầu đến cuối (End-to-End) theo nguyên tắc *Tracer Bullet*.
- Chạy thông 1 batch hoặc 1 epoch với dummy dataset giả lập kích thước nhỏ để xác nhận luồng:
  `DataModule → Model Forward → Loss Backward → Metric Log`
- Đảm bảo toàn bộ hạ tầng phần cứng, GPU CUDA, logger hoạt động trơn tru trước khi huấn luyện thực tế.

---

### 5. IMPLEMENT (Triển khai Sản xuất Chuẩn mực)
- Áp dụng chu trình vi mô 7 bước: [`AGENT_INTERACTION_LOOP.md`](AGENT_INTERACTION_LOOP.md).
- Mã nguồn tuân thủ:
  - PEP 484 Type Hints đầy đủ cho mọi hàm và phương thức.
  - Google-style docstrings ghi rõ tensor shapes `[B, C, H, W]`.
  - Lập trình phòng thủ (Fail-Fast): kiểm tra ranh giới, kích thước mảng trước khi tính toán.
  - Không hardcode tham số, tách toàn bộ vào `configs/`.

---

### 6. TEST (Kiểm thử 2 Chế độ)
- Tuân thủ chiến lược kiểm thử tại [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md):
  - **Smoke Test (`make test-smoke`)**: Kiểm tra cú pháp, import, khởi tạo đối tượng, chạy forward dummy dưới 5 giây.
  - **Normal Test (`make test`)**: Kiểm tra logic toán học, tính bảo toàn kích thước, bất biến số học (Softmax sum = 1.0, IoU ∈ [0, 1]), edge cases và rò rỉ bộ nhớ.
- Đối chiếu từng test case với Detailed Execution Plan.

---

### 7. REFACTOR (Tối ưu hóa & Làm sạch Mã nguồn)
- **Vectorization**: Loại bỏ toàn bộ vòng lặp duyệt pixel bằng Python thuần; chuyển sang NumPy broadcasting hoặc PyTorch tensor operations.
- **Config Decoupling**: Đưa toàn bộ hằng số ma thuật (magic numbers) vào file YAML tương ứng trong `configs/`.
- **Memory Optimization**: Sử dụng `.detach()` và `.item()` khi ghi log loss để giải phóng computational graph trên GPU.
- **Tự động chuẩn hóa**: Chạy `make format` (Black, Ruff) để định dạng mã nguồn đồng nhất.

---

### 8. COMMIT (Lưu vết Phiên bản Nguyên tử)
- Đảm bảo toàn bộ pre-commit hooks và test suites đều đạt trạng thái Pass:
  ```bash
  make lint && make test
  ```
- Thực hiện commit nguyên tử (Atomic Commit) tuân theo chuẩn Conventional Commits quy định tại [`AI_REFERENCE.md`](AI_REFERENCE.md):
  ```bash
  git commit -m "feat(data): implement albumentations pipeline for image segmentation"
  ```

---

### 9. DOCUMENT (Tài liệu hóa & Báo cáo Thí nghiệm)
- Hoàn thiện báo cáo kỹ thuật dựa trên file mẫu quy chuẩn của dự án:
  - **Báo cáo chuyên sâu (`*.md` & `*.html`)**: Chứa đầy đủ Sơ đồ Luồng Công việc Toàn diện (Mermaid code + hình ảnh vector) và Mục Tài Liệu Tham Khảo (References) 4 nhóm chuẩn.
  - **Chuẩn hóa Jupyter Notebook (`notebooks/**/*.ipynb`)**: Mở đầu bằng Cell 0 (Mục tiêu) và Cell 1 gồm Bảng Lộ trình Thực thi 4 cột chuẩn (`| Step | Description | What it does | Import path |`) kèm ngay phía dưới là Mục Liên kết Nhanh Tài Liệu & Báo Cáo liên quan. Tuyệt đối KHÔNG chèn sơ đồ luồng workflow hay references học thuật vào notebook để giữ notebook gọn nhẹ, tương tác mượt mà theo quy chuẩn tại [`AI_REFERENCE.md`](AI_REFERENCE.md).
  - **Bảo mật & Điều hướng (`Ctrl + Click`)**: 100% liên kết mã nguồn và tài liệu phải sử dụng đường dẫn tương đối chuẩn (`../../src/...`), tuyệt đối KHÔNG hardcode đường dẫn tuyệt đối chứa username hệ thống (`file:///home/...`).
- Ghi nhận đầy đủ:
  - Bảng số liệu định lượng (Quantitative Results): Loss, Accuracy, mAP, FPS.
  - Hình ảnh trực quan hóa định tính (Qualitative Visualizations) từ `experiments/predictions/`.
  - Seed và lệnh chạy CLI chính xác để người khác có thể tái lập kết quả (Reproducibility).
