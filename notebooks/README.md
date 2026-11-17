# Jupyter Notebooks Governance & Technical Architecture (`/notebooks`)

- **Topic**: Đề tài 11 — Nghiên cứu và Thực hiện Kỹ thuật Phát hiện Ảnh Giả Mạo Deepfake (Deepfake Detection & Defense)
- **Institution**: Trường Đại Học Giao Thông Vận Tải TP.HCM (UTH) — Khoa Công Nghệ Thông Tin
- **Target Audience**: Hội đồng chấm Đồ án / Khóa luận tốt nghiệp, Giảng viên hướng dẫn, Nghiên cứu sinh & Sinh viên
- **Purpose**: Thiết lập chuẩn mực phân tách tuyệt đối giữa mã nguồn huấn luyện (`src/`, `scripts/`) và không gian trình diễn, phân tích dữ liệu trực quan (`notebooks/coursework/`); đồng thời quản trị bộ notebook 4 phần hợp nhất (4-Part Unified Coursework Suite).
- **Builder Engine**: [`scripts/build_unified_coursework_suite.py`](../scripts/build_unified_coursework_suite.py)
- **Legacy Archive**: [`notebooks/coursework/legacy_v1/`](coursework/legacy_v1/README.md)
- **Standard Reference**: [`.agents/rules/NOTEBOOK_HEADER_CONVENTION.md`](../.agents/rules/NOTEBOOK_HEADER_CONVENTION.md), [`docs/PURPOSE.md`](../docs/PURPOSE.md)
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-17T09:35:00+07:00

---

## 1. Golden Rules for Coursework Notebooks

1. **Không Chứa Vòng Lặp Huấn Luyện Nặng (No Monolithic Training Loops)**:
   - Toàn bộ pipeline huấn luyện mô hình nặng chạy thông qua các kịch bản CLI trong `scripts/` (ví dụ: `scripts/train_m3_dual_engine.py`) hoặc mô-đun hoá trong `src/pipelines/` kèm hệ thống checkpointing và ghi log có cấu trúc (`src.common.logger`).
   - Notebook trong `notebooks/` chỉ phục vụ 4 mục tiêu tương tác:
     - Khám phá và phân tích đặc trưng dữ liệu (EDA, phân bố gamut màu da YCbCr, t-SNE, PCA).
     - Trực quan hoá cơ chế tiền xử lý và tăng cường ảnh (soft-mask feathering $\sigma=4.0$, CurationGate 3 tầng).
     - Nạp checkpoint mô hình thực nghiệm (`baseline_d0_best.pt`, `active_learning_m2_best.pt`, `active_learning_m3_best.pt`) để suy luận trực tiếp (live inference).
     - Giám định pháp y số (digital forensics) và hiển thị bản đồ kích hoạt không gian (spatial attention heatmaps).

2. **Giới Hạn Dung Lượng & Vệ Sinh Biểu Đồ (< 2.5 MB / Notebook, Suite < 10 MB)**:
   - Mọi đồ thị matplotlib đều được thiết lập chuẩn xuất bản (DPI 110–120), căn lề chặt chẽ (`bbox_inches='tight'`).
   - Lệnh `plt.close("all")` được gọi ngay lập tức sau mỗi lần kết xuất để loại bỏ rò rỉ bộ nhớ và phình to chuỗi base64.
   - Toàn bộ bộ 4 notebook tổng cộng chỉ chiếm **~6.7 MB** trên đĩa cứng, hoàn toàn nằm trong ngưỡng < 10.0 MB.

3. **Mã Nguồn Mở, Tự Động Hoá & Khả Năng Tái Lập Tuyệt Đối**:
   - Toàn bộ suite được biên dịch và chạy suy luận tự động bằng công cụ biên dịch `scripts/build_unified_coursework_suite.py`.
   - 100% dữ liệu suy luận và kiểm thử là dữ liệu thực từ các tập dữ liệu chuẩn (FFHQ, CelebA-HQ, Challenge Vault, Universal Benchmark 20.4k ảnh). Tuyệt đối không dùng dữ liệu giả lập (mock data) hay trọng số ngẫu nhiên uninitialized.
   - Tuyệt đối không chứa đường dẫn tuyệt đối cục bộ (`/home/...`, `file:///`) hay tên tài khoản cá nhân. Tất cả liên kết đều sử dụng đường dẫn tương đối (relative paths).

---

## 2. Cấu Trúc Thư Mục (Directory Structure)

```text
notebooks/
├── coursework/                                # BỘ NOTEBOOK CHÍNH THỨC 4 PHẦN (4-Part Unified Suite)
│   ├── 00_master_deepfake_defense_showcase.ipynb      # Phần 0: Master Showcase & Live Forensic Inspector (~1.7 MB)
│   ├── 01_diffusion_dataset_and_curation.ipynb        # Phần 1: Real Manifold & Dual-Engine Portfolio (~1.7 MB)
│   ├── 02_hard_samples_and_active_learning.ipynb      # Phần 2: Hard-Sample Mining & Active Learning (~1.7 MB)
│   ├── 03_champion_m3_and_universal_benchmark.ipynb   # Phần 3: Champion M3 & Universal Benchmark (~1.6 MB)
│   └── legacy_v1/                                     # Kho lưu trữ 10 notebook thử nghiệm lịch sử v1 (~49.4 MB)
│       ├── 00_master_deepfake_defense_coursework.ipynb
│       ├── 01_eda_faces_demographics.ipynb
│       ├── 02_sd_generation_and_curation.ipynb
│       ├── 03_hard_samples_and_benchmark_comparison.ipynb
│       ├── 04_active_learning_production_visualizer.ipynb
│       ├── 05_synthetic_face_6k_realimage_benchmark.ipynb
│       ├── 06_specialization_vs_generalization_synthesis.ipynb
│       ├── 07_universal_deepfake_benchmark.ipynb
│       ├── 08_champion_m3_dual_engine_quad_vault_benchmark.ipynb
│       ├── generator_realistic_vision_kaggle.ipynb
│       └── README.md                                  # Tài liệu ghi nhận provenance kho legacy
├── exploratory/                                       # Notebook nghiên cứu & thử nghiệm nhanh
└── README.md                                          # Tài liệu quản trị chuẩn (file hiện tại)
```

---

## 3. Chi Tiết Bộ Notebook 4 Phần Hợp Nhất (The 4-Part Unified Narrative Suite)

Hệ thống notebook mới được tái cấu trúc từ 10 notebook sơ khởi thành 4 chương chuyên đề liền mạch, phản ánh chính xác hành trình khoa học từ thu thập dữ liệu, phân tích mẫu khó, học chủ động đến đối chuẩn toàn diện:

| Phần | Tệp Notebook | Trọng Tâm Khoa Học & Đóng Góp | Kích Thước | Trạng Thái Kiểm Thử |
| :---: | :--- | :--- | :---: | :---: |
| **0** | [`00_master_deepfake_defense_showcase.ipynb`](coursework/00_master_deepfake_defense_showcase.ipynb) | **Báo Cáo Tổng Hợp & Giám Định Pháp Y Số Tương Tác**: Bức tranh toàn cảnh toàn bộ đề tài ($M_0 \to M_3$), Bảng điểm vàng (Golden Scorecard), Bộ công cụ Live Forensic Inspector suy luận thời gian thực với bản đồ nhiệt chú ý không gian DINOv2 (Spatial Attention Heatmap), và 3 trụ cột bảo vệ đồ án trước Hội đồng UTH. | ~1.7 MB | ✅ Đạt (0 lỗi) |
| **1** | [`01_diffusion_dataset_and_curation.ipynb`](coursework/01_diffusion_dataset_and_curation.ipynb) | **Đa Tạp Khuôn Mặt Thật & Động Cơ Khuếch Tán Kép**: Khảo sát 4.000 khuôn mặt thật (FFHQ & CelebA-HQ), phân tích phổ màu da YCbCr, bộ tạo sinh kép (Stable Diffusion 1.5 + RealisticVision v5.1), kỹ thuật làm mịn biên viền mềm (Soft-Mask Feathering với $\sigma = 4.0$), kiểm toán chất lượng 3 tầng CurationGate. | ~1.7 MB | ✅ Đạt (0 lỗi) |
| **2** | [`02_hard_samples_and_active_learning.ipynb`](coursework/02_hard_samples_and_active_learning.ipynb) | **Khai Thác Mẫu Khó & Vòng Lặp Học Chủ Động (Active Learning)**: Kiến trúc DINOv2 ViT-S/14 với chiến lược mở rã đông từng phần (8.06% trọng số có thể huấn luyện), 740 mẫu khó được khai thác qua miền tần số và không gian, Challenge Vault (213 mẫu), 4 chế độ lỗi sinh học, minh chứng thực nghiệm bước nhảy vọt độ nhạy từ $11.71\%$ ($M_0$) lên $91.89\%$ ($M_2$). | ~1.7 MB | ✅ Đạt (0 lỗi) |
| **3** | [`03_champion_m3_and_universal_benchmark.ipynb`](coursework/03_champion_m3_and_universal_benchmark.ipynb) | **Mô Hình Champion M3 & Đối Chuẩn Toàn Cầu Universal Benchmark**: Kiểm toán Zero Identity Leakage trên $D_3$, mô hình vô địch $M_3$ Dual-Engine Warm-Start, Quad-Vault Benchmark (đạt 93.13% trên RV5.1 hoàn toàn chưa thấy khi train), Universal Benchmark trên 20.400 ảnh thuộc 34 trình tạo sinh và 8 họ kiến trúc, phân tích ngưỡng tối ưu $\tau \in [0.05, 0.95]$. | ~1.6 MB | ✅ Đạt (0 lỗi) |

---

## 4. Kho Lưu Trữ Lịch Sử (Legacy v1 Archive)

Trong giai đoạn đầu của đề tài, các thử nghiệm được triển khai phân tán trên 10 file notebook riêng lẻ trong `notebooks/coursework/legacy_v1/`.

### Lý do lưu trữ và chuẩn hoá:
1. **Dung lượng lớn và lãng phí bộ nhớ (~49.4 MB)**: Các biểu đồ kích thước lớn và chuỗi base64 không nén khiến kho mã nguồn nặng nề và khó kiểm soát trên Git.
2. **Trùng lặp nội dung**: Quy trình EDA dữ liệu và trực quan hoá lặp lại nhiều lần giữa các notebook mà không tái sử dụng thư viện chung trong `src/`.
3. **Khó bảo trì và kiểm thử tự động**: Một số notebook phụ thuộc vào môi trường GPU ngoài (như Kaggle) hoặc thiếu tính nhất quán khi chạy trên hệ thống CI/CD chuẩn.

Tất cả 10 notebook này đã được chuyển vào [`notebooks/coursework/legacy_v1/`](coursework/legacy_v1/README.md) nhằm giữ trọn vẹn dữ liệu gốc và bằng chứng thực nghiệm (provenance), trong khi không gian chính thức được dành riêng cho bộ 4 notebook chuẩn hoá.

---

## 5. Kiến Trúc Ô Chuẩn Mực Trong Mỗi Notebook (Required Cell Architecture)

Để đảm bảo tính khoa học và thẩm mỹ cao nhất khi chấm điểm, mỗi notebook đều tuân thủ nghiêm ngặt cấu trúc 2 ô đầu tiên:

- **Cell 0 [Markdown]**: Banner viện/trường Đại học Giao Thông Vận Tải TP.HCM (UTH), thông tin Đề tài 11, thành viên nhóm thực hiện, số hiệu Phần và mục tiêu khoa học chi tiết.
- **Cell 1 [Markdown]**: Bảng lộ trình (Roadmap Navigation Table) 4 cột dẫn link tương đối đến toàn bộ các notebook trong bộ suite, tích hợp huy hiệu đánh dấu phần đang mở (`🎯 (Đang xem)`).
- **Cell 2+ [Code / Markdown]**: Các khối lệnh thực thi gọn gàng, nhập các hàm từ `src/` và hiển thị kết quả kiểm chứng trực quan kèm chú giải sư phạm.

---

## 6. Trình Tạo Sinh Và Chạy Tự Động (Builder Engine)

Toàn bộ 4 notebook được biên dịch và tạo mới hoàn toàn tự động bằng công cụ:
`scripts/build_unified_coursework_suite.py`.

### Điểm nổi bật của Builder Engine:
- **Tự Động Tạo Cấu Trúc AST**: Tạo cây JSON chuẩn Jupyter nbformat v4 với đầy đủ metadata và thẻ ngôn ngữ.
- **Thực Thi Trực Tiếp Trong Quá Trình Build (Headless Execution)**: Engine sử dụng `nbclient` / kernel trực tiếp để nạp trọng số mô hình thật từ `models/`, tính toán ma trận nhầm lẫn (confusion matrix), đường cong ROC, và xuất biểu đồ sắc nét.
- **Tối Ưu Hoá Bộ Nhớ & Kích Thước**: Tự động chèn lệnh `plt.close("all")`, định dạng kích thước ảnh hiển thị chuẩn nhằm giữ mọi notebook dưới ngưỡng 2.5 MB.

### Lệnh thực thi:

```bash
# Biên dịch và chạy từng notebook riêng biệt:
python scripts/build_unified_coursework_suite.py --notebook 0
python scripts/build_unified_coursework_suite.py --notebook 1
python scripts/build_unified_coursework_suite.py --notebook 2
python scripts/build_unified_coursework_suite.py --notebook 3

# Biên dịch toàn bộ 4 notebook một cách tuần tự:
python scripts/build_unified_coursework_suite.py --notebook all
```

---

## 7. Quy Trình Kiểm Thử & Đảm Bảo Chất Lượng (Quality Verification)

Hệ thống kiểm thử tự động được thiết lập để bảo vệ chất lượng của bộ notebook:

```bash
# 1. Kiểm tra tính toàn vẹn của cấu trúc suite, dung lượng, header và link tương đối:
pytest tests/test_notebook_suite_integrity.py -v

# 2. Kiểm tra chất lượng đầu ra đồ thị và bảng số liệu khoa học của từng notebook:
pytest tests/test_notebook_outputs.py -v

# 3. Chạy toàn bộ smoke tests phục vụ pre-commit check:
make test-smoke

# 4. Kiểm tra cú pháp và tính hợp lệ của mã nguồn:
make lint
```
