# AI REFERENCE - BỘ QUY CHUẨN THAM CHIẾU BẮT BUỘC CHO AI AGENT

AI Agent phải tuyệt đối tuân thủ toàn bộ quy chuẩn trong tài liệu này. Bất kỳ sai lệch nào sẽ bị từ chối ở bước Codebase Audit.

---

## 1. Quy chuẩn Đặt tên (Naming Conventions)

| Đối tượng | Quy chuẩn | Ví dụ Chuẩn (Good) | Ví dụ Sai (Bad) |
| :--- | :--- | :--- | :--- |
| **Tên Thư mục** | `snake_case` hoặc kebab-case ngắn | `models/`, `data_loader/`, `pre_trained/` | `Models/`, `dataLoader/`, `MyData/` |
| **Tên File Markdown** | VIẾT HOA TOÀN BỘ (`UPPER_SNAKE_CASE.md` hoặc `UPPERCASE.md`) | `README.md`, `REPORT.md`, `TEMPLATE.md` | `readme.md`, `report.md`, `template.md` |
| **Tên File Python** | `snake_case.py` | `datamodule.py`, `metrics.py`, `train.py` | `DataModule.py`, `my_Train.py` |
| **Tên File Cấu hình** | `snake_case.yaml` | `default.yaml`, `template.yaml` | `Default.YAML`, `my-config.yml` |
| **Tên Class** | `PascalCase` | `StandardCVDataset`, `CVDataModule`, `CustomCNN` | `standard_cv_dataset`, `cvDataModule` |
| **Tên Function / Method** | `snake_case` (bắt đầu bằng động từ) | `calculate_iou()`, `apply_filter()`, `build_model()` | `CalculateIoU()`, `iou()`, `DoProcess()` |
| **Tên Biến thông thường** | `snake_case` | `image_tensor`, `batch_size`, `learning_rate` | `ImageTensor`, `bs`, `lrVal` |
| **Tên Hằng số** | `UPPER_SNAKE_CASE` | `DEFAULT_SEED`, `IMAGE_SIZE`, `NUM_CLASSES` | `default_seed`, `ImageSize` |
| **Tên Biến Riêng tư (Private)**| `_leading_snake_case` | `_init_weights()`, `_cached_features` | `__init_weights()`, `privateFeatures` |

---

## 2. Quy chuẩn Cấu trúc Thư mục (Folder Structure Invariants)

AI không được tự ý tạo thêm thư mục gốc mới ngoài danh mục đã được phê duyệt:

```text
<project-root>/
├── configs/          # CHỈ chứa file .yaml cấu hình (Hydra)
├── data/             # CHỈ chứa dữ liệu thô, trung gian và processed (ignored by Git)
├── docs/             # CHỈ chứa tài liệu hướng dẫn và báo cáo kỹ thuật markdown (Tên file VIẾT HOA)
├── experiments/      # CHỈ chứa output sinh ra tự động: checkpoints, logs, predictions
├── notebooks/        # CHỈ chứa notebook tương tác .ipynb
├── requirements/     # CHỈ chứa danh sách dependencies theo môi trường .txt
├── scripts/          # CHỈ chứa shell scripts tự động hóa .sh
├── src/              # CHỈ chứa mã nguồn Python chính thức của package
└── tests/            # CHỈ chứa unit tests và fixtures phục vụ pytest
```

---

## 3. Danh sách Điều cấm Kỵ (Negative Constraints for AI)

1. **Cấm tự ý tạo file chứa nội dung giả định**: Không tạo các file có tên cụ thể khi chưa được chỉ định trong task.
2. **Cấm tự ý bịa đặt hoặc suy đoán thông tin không xác thực (Zero-Assumption & Anti-Hallucination)**: Tuyệt đối KHÔNG tự ý đưa vào tài liệu/mã nguồn bất kỳ thông tin nào không biết hoặc không có trong đề bài/tài liệu được giao (ví dụ: tên khoa, bộ môn, mã định danh, tác giả, tổ chức chưa được xác thực). Chỉ sử dụng đúng và đủ các thông tin được cung cấp rõ ràng.
3. **Bắt buộc viết hoa toàn bộ tên file Markdown (`.md`)**: Toàn bộ file Markdown trong toàn repository (bao gồm `README.md`, tài liệu `docs/`, báo cáo thí nghiệm, v.v.) BẮT BUỘC phải đặt tên VIẾT HOA TOÀN BỘ (`UPPER_SNAKE_CASE.md` hoặc `UPPERCASE.md`), ví dụ: `README.md`, `REPORT.md`, `TEMPLATE.md`, `TASK_DESCRIPTION.md`. Nghiêm cấm đặt tên file `.md` dạng chữ thường.
4. **Cấm hardcode đường dẫn tuyệt đối (Privacy & Portability Invariant)**: Tuyệt đối KHÔNG sử dụng đường dẫn tuyệt đối chứa username hệ thống (như `/home/user/...`, `file:///home/...`, `C:\...`); 100% liên kết tài liệu và mã nguồn phải dùng đường dẫn tương đối (ví dụ: `../../src/...`). Quy định này bảo mật tuyệt đối thông tin người dùng, đảm bảo tính di động của repository trên mọi hệ điều hành và hỗ trợ `Ctrl + Click` nhảy trực tiếp tới file đích trong VS Code.
5. **Cấm sử dụng `print` trong module `src/`**: Luôn sử dụng logger từ `src.common.logger`.
6. **Cấm commit file nhị phân lớn**: Không lưu trọng số model nặng, file zip hoặc video vào Git.
7. **Quy chuẩn Tách bạch Giữa Notebook và Báo cáo (Notebook vs. Report Separation)**:
   - Trong Jupyter Notebook (`notebooks/**/*.ipynb`): **TUYỆT ĐỐI KHÔNG chèn Sơ đồ Luồng quy trình (Workflow Diagram)** (dù bằng ảnh, SVG hay mã Mermaid) và KHÔNG đưa danh mục tài liệu tham khảo học thuật vào cuối notebook. Notebook bắt buộc mở đầu bằng Cell 0 (Mục tiêu), Cell 1 gồm Bảng Lộ trình 4 cột chuẩn (`Step`, `Description`, `What it does`, `Import path`) kèm Mục Liên kết Nhanh Tài Liệu & Báo Cáo kỹ thuật liên quan, sau đó đi thẳng vào các cell thực thi kỹ thuật.
   - Trong Báo cáo Chuyên sâu (`docs/**/*.md` và `docs/**/*.html`): **BẮT BUỘC lưu trữ và trực quan hóa toàn diện Sơ đồ Luồng Công việc (Visual Workflow Pipeline)** (gồm khối mã Mermaid + ảnh vector/infographic) và **Mục Tài Liệu Tham Khảo (References & Technical Documentation)** đầy đủ 4 nhóm chuẩn.
8. **Cấm thiếu Báo cáo HTML trực quan (Missing HTML Visual Report)**: Bắt buộc phải có file báo cáo HTML trực quan (`*.html`) đi kèm với file Markdown (`*.md`) trong `docs/` để hiển thị trực quan toàn bộ kết quả ảnh, biểu đồ và số liệu khi mở trên trình duyệt.
9. **Cấm thiếu Mục Tài Liệu Tham Khảo trong Báo cáo (Mandatory References in Reports)**: Mọi báo cáo kỹ thuật (`docs/**/*.md`, `docs/**/*.html`) bắt buộc phải có Mục Tài Liệu Tham Khảo (References & Technical Documentation) ở cuối tài liệu với đầy đủ 4 phân nhóm chuẩn (Official API Docs, Technical Standards, CV Conventions, Internal Architecture Links).
10. **Bất biến Khung Nhận thức Độc lập & Tái Sử dụng của Thư mục `.agents/` (Agents Framework Portability & Generic Invariant)**:
    - Thư mục `.agents/` là bộ quy chuẩn nhận thức khung (Core Cognitive Blueprint) dùng chung cho mọi dự án Computer Vision & MLOps.
    - Tuyệt đối KHÔNG đưa thông tin cụ thể của bất kỳ bài toán, bài thực hành riêng lẻ, tên tổ chức/trường học, tên tác giả, hay các tệp nghiệm thu cụ thể vào trong `.agents/`.
    - Mọi quy định, đường dẫn và sơ đồ phải ở dạng trừu tượng/khung mẫu tổng quát (meta-framework, abstract placeholders như `<project-root>`, `XX`, v.v.) để đảm bảo tính di động tuyệt đối: khi sao chép thư mục `.agents/` sang bất kỳ dự án mới nào đều hoạt động ngay lập tức mà không gây ô nhiễm (pollution) ngữ cảnh hay xung đột (zero conflict).

---

## 4. Quy chuẩn Conventional Commits

Mọi commit do AI đề xuất hoặc kỹ sư thực hiện phải tuân thủ nghiêm ngặt cú pháp:

```text
<type>(<scope>): <subject>

[optional body: mô tả nguyên nhân và giải pháp kỹ thuật]

[optional footer: liên kết issue hoặc breaking changes]
```

### Danh mục Types cho Computer Vision:
- `feat`: Thêm tính năng mới (thuật toán lọc, kiến trúc model, metric mới, datamodule).
- `fix`: Sửa lỗi (shape mismatch, zero-division, NaN loss, off-by-one index).
- `perf`: Tối ưu hóa hiệu năng (vectorization, DataLoader prefetching, GPU memory).
- `refactor`: Tái cấu trúc mã nguồn mà không làm thay đổi tính năng hay kết quả.
- `test`: Thêm hoặc cập nhật unit test, smoke test, benchmark fixtures.
- `docs`: Cập nhật tài liệu kỹ thuật, docstring, báo cáo thí nghiệm.
- `chore`: Thay đổi cấu hình build, pre-commit, package dependencies.

### Danh mục Scopes chuẩn:
`data`, `models`, `pipelines`, `metrics`, `visualizer`, `configs`, `tests`, `scripts`

### Quy tắc commit:
1. Viết subject ở thể mệnh lệnh hiện tại (`implement`, `add`, `fix`, không dùng `implemented`, `fixing`).
2. Subject không viết hoa chữ đầu, không kết thúc bằng dấu chấm, tối đa 72 ký tự.
3. Commit nguyên tử (Atomic Commit): Một commit chỉ giải quyết trọn vẹn một tác vụ độc lập.

---

## 5. Bất biến Kỹ thuật Ngành Thị giác Máy tính (CV/DL Domain Invariants)

Mọi dòng code xử lý hình ảnh và mô hình học sâu phải tuân thủ nghiêm ngặt 5 tiêu chuẩn kỹ thuật:

### 5.1. Quy chuẩn Tensor & Kích thước Chiều (Shape Conventions)
- **OpenCV / NumPy**: Luôn ở định dạng `(H, W, C)` hoặc `(H, W)` (grayscale), kiểu dữ liệu `uint8` [0, 255] hoặc `float32` [0.0, 1.0].
- **PyTorch**: Luôn ở định dạng `(B, C, H, W)` (batch) hoặc `(C, H, W)` (đơn mẫu), kiểu dữ liệu `float32`.
- **Docstrings**: Bắt buộc ghi rõ shape tensor đầu vào và đầu ra trong mọi hàm xử lý ảnh.

### 5.2. Nhận diện Không gian Màu (Color Space Awareness)
- **OpenCV**: Đọc và ghi mặc định ở hệ màu `BGR`.
- **PIL / Matplotlib / Torchvision**: Sử dụng mặc định hệ màu `RGB`.
- **Quy tắc**: Phải chuyển đổi tường minh qua `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` khi trao đổi dữ liệu giữa OpenCV và Torchvision / Matplotlib. Cấm để lệch màu ẩn.

### 5.3. Định dạng Tọa độ Hộp bao (Bounding Box Formats)
Không bao giờ ngầm định định dạng hộp bao. Phải ghi rõ 1 trong các chuẩn sau:
- `XYXY` / Pascal VOC: `[x1, y1, x2, y2]` (pixel tuyệt đối).
- `XYWH` / COCO: `[x_min, y_min, width, height]` (pixel tuyệt đối).
- `CXCYWH` / YOLO: `[cx, cy, w, h]` (chuẩn hóa về khoảng [0.0, 1.0]).

### 5.4. Chuẩn hóa Tính Tái lập Thí nghiệm (Reproducibility Mandate)
Mọi thí nghiệm phải sử dụng hàm seed tập trung để khóa ngẫu nhiên trên toàn bộ môi trường:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`

### 5.5. Quản lý Tài nguyên GPU & Tối ưu Băng thông (GPU Hygiene)
- Không hardcode `.cuda()` hoặc `'cuda:0'`, luôn dùng `device = torch.device(...)` nhận từ file cấu hình.
- DataLoader trên GPU phải bật `pin_memory=True` và cấu hình `num_workers > 0` phù hợp với số CPU cores.
- Sử dụng `torch.cuda.amp.autocast()` khi huấn luyện để tối ưu tốc độ và dung lượng VRAM.

---

## 6. Quy chuẩn Bắt buộc cho Jupyter Notebooks (Notebook Standard Blueprint)

Mọi Jupyter Notebook thực hành (`notebooks/labs/*.ipynb` hoặc `notebooks/exploratory/*.ipynb`) BẮT BUỘC phải tuân thủ kiến trúc phân tầng chuẩn:

```text
[ Cell 0: Header & Mục tiêu ] ──► [ Cell 1: Bảng Roadmap (4 cột) + Links Tài liệu liên quan ] ──► [ Các Cells Thực hành Kỹ thuật ]
```

### 6.1. Cell 1: Bảng Lộ trình Thực thi & Liên kết Tài liệu (Roadmap & Documentation References)
- Bắt buộc nằm ở Cell Markdown độc lập ngay sau Cell Tiêu đề (Cell 1).
- Gồm 2 phần chính:
  1. **Bảng Lộ trình Thực thi 4 cột chuẩn bắt buộc**:

| Step | Description | What it does | Import path |
| :---: | :--- | :--- | :--- |
| `1` | `[Tên Bước Kỹ Thuật]` | `[Mô tả chi tiết hành vi, giải thuật, chỉ số]` | `—` |
| `2` | `...` | `...` | `[`src/...py`](../../src/...py)` |

     - **Yêu cầu đối với 4 cột**:
       1. `Step`: Đánh số thứ tự bước thực thi (1, 2, 3...).
       2. `Description`: Tên ngắn gọn của bước kỹ thuật.
       3. `What it does`: Mô tả chính xác việc mà bước này thực hiện.
       4. `Import path`: **BẮT BUỘC là LINK BẤM ĐƯỢC TRỰC TIẾP** trỏ thẳng tới file mã nguồn phụ trách (`[src/...](../../src/...)`). Nếu là cấu hình môi trường / thư viện gốc, ghi `—`.

  2. **Mục Liên kết Nhanh Tài Liệu & Báo Cáo Kỹ Thuật (Related Documentation References)**:
     - Nằm ngay dưới bảng Roadmap sau đường phân cách `---`.
     - Cung cấp các liên kết Markdown trực tiếp bấm mở nhanh các tài liệu liên quan của module / thí nghiệm:
       + 📄 Báo Cáo Kỹ Thuật Chi Tiết: `[`docs/.../REPORT.md`](../../docs/.../REPORT.md)` *(chứa toàn bộ Sơ đồ luồng xử lý toàn diện và phân tích học thuật)*.
       + 🌐 Báo Cáo Trực Quan HTML: `[`docs/.../REPORT.html`](../../docs/.../REPORT.html)` *(mở trên trình duyệt xem toàn bộ visual gallery & artifacts)*.
       + 📋 Đặc Tả Yêu Cầu Kỹ Thuật: `[`docs/.../TASK_DESCRIPTION.md`](../../docs/.../TASK_DESCRIPTION.md)` *(nội dung bài toán và tiêu chí đánh giá)*.
       + 🏠 Tài Liệu Hướng Dẫn Dự Án: `[`README.md`](../../README.md)` *(kiến trúc mã nguồn và hướng dẫn chạy tự động)*.

### 6.2. Sơ đồ Luồng Công việc (Workflow Diagram) - DÀNH RIÊNG CHO BÁO CÁO (REPORTS ONLY)
- **Tuyệt đối KHÔNG đưa sơ đồ quy trình vào Jupyter Notebook**: Giúp notebook luôn tinh gọn, tập trung hoàn toàn vào việc chạy mã nguồn thực thi, tương tác kết quả và tránh lỗi hiển thị mã Mermaid thô.
- **Bắt buộc lưu trữ trong Báo cáo**: Sơ đồ luồng công việc toàn diện (Input Layer, Clean Arch `src/`, Core Processing Stages, Artifacts Output, QA Verification) BẮT BUỘC phải nằm trong:
  1. File Markdown báo cáo (`docs/**/REPORT.md`) (khối mã Mermaid + liên kết ảnh/vector).
  2. File HTML báo cáo (`docs/**/REPORT.html`) (giao diện đồ họa vector SVG / infographic tương tác).

---

## 7. Quy chuẩn Báo cáo HTML Trực quan (Interactive HTML Report Mandate)

Mỗi cột mốc thí nghiệm hoặc báo cáo hoàn thành, bên cạnh báo cáo Markdown (`*.md`), BẮT BUỘC phải sinh kèm một file báo cáo HTML trực quan độc lập (`*.html`) nằm cùng thư mục trong `docs/`:

### 7.1. Mục đích Kỹ thuật
- Giúp người dùng, giảng viên và kỹ sư có thể nhấp đúp hoặc mở trực tiếp trên bất kỳ trình duyệt web nào (Chrome, Firefox, Safari, Edge, VS Code Simple Browser) để quan sát trực quan ngay lập tức toàn bộ kết quả thí nghiệm, hình ảnh trước/sau (before/after), lưới không gian màu, bounding boxes và bảng số liệu định lượng mà không cần công cụ render markdown hay môi trường Jupyter Notebook.

### 7.2. Yêu cầu Cấu trúc File HTML
1. **Thiết kế & Giao diện**: Giao diện hiện đại (Modern Clean CSS), responsive trên mọi độ phân giải, chia khối (Card layouts), có phối màu chuyên nghiệp và trạng thái huy hiệu (Badges).
2. **Đường dẫn Ảnh tương đối**: Toàn bộ thẻ `<img>` phải sử dụng đường dẫn tương đối chính xác trỏ tới các artifacts trong `experiments/` và `data/`.
3. **Thành phần Bắt buộc**:
   - Header thông tin chính thống (Tên dự án/Module, Phiên bản môi trường thực thi).
   - Sơ đồ trực quan Workflow Pipeline toàn diện (SVG hoặc Mermaid layout).
   - Bảng Lộ trình thực thi 4 cột chuẩn (`Step`, `Description`, `What it does`, `Import path`), trong đó cột `Import path` là link `<a href="...">` bấm vào mở trực tiếp mã nguồn.
   - Phòng trưng bày ảnh thực nghiệm (Visual Gallery) phân mục theo từng yêu cầu với chú thích kích thước, định dạng, dung lượng.
   - Bảng tổng kết kết quả kiểm thử định lượng (Unit tests, Smoke tests, Linter checks).
   - Mục Tài liệu tham khảo (References & Technical Documentation) với các liên kết chính thức.

---

## 8. Quy chuẩn Mục Tài Liệu Tham Khảo (References Standard - Reports)

Mọi báo cáo thực nghiệm chuyên sâu (`docs/**/*.md`, `docs/**/*.html`) BẮT BUỘC phải có **Mục Tài Liệu Tham Khảo (References & Technical Documentation)** ở cuối tài liệu với 4 phân nhóm quy chuẩn (trong khi Notebook chỉ chứa các link liên kết nhanh tài liệu nội bộ ở Cell 1):

1. **Thư viện Cốt lõi & Tài liệu API Chính thức (Official Library Documentation)**:
   - Các API cụ thể được sử dụng trong bài toán (ví dụ: OpenCV, Pillow, PyTorch, NumPy, Matplotlib, Albumentations, v.v.).
   - Đường dẫn liên kết trực tiếp tới trang tài liệu chính thức của thư viện.
2. **Tiêu chuẩn Kỹ thuật & Không gian Màu (Technical Standards & Specifications)**:
   - Các tiêu chuẩn quốc tế, bài báo khoa học hoặc giải thuật nền tảng liên quan trực tiếp đến nghiệp vụ của bài toán.
3. **Quy ước Thị giác Máy tính & Định dạng Dữ liệu (CV Conventions & Formats)**:
   - Quy ước thứ tự kênh màu bộ nhớ (Memory Layout: BGR vs RGB), dtypes và tensor shapes.
   - Quy ước định dạng bounding box, mask phân đoạn hoặc keypoints tương ứng.
4. **Mã Nguồn Module Tái Sử Dụng Nội Bộ (Internal Project Architecture)**:
   - Danh sách link bấm được trực tiếp tới các module mã nguồn trong package `src/` phụ trách nghiệp vụ.

