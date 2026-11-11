# PROJECT_ROADMAP.md — Lộ Trình Tiến Hóa Dự Án Toàn Diện (M0 -> M12)

- **Motivation/Background**: Theo dõi toàn bộ quá trình tiến hóa của Đề tài 11 từ đề xuất ban đầu đến hệ thống phòng thủ deepfake phổ quát hoàn chỉnh.
- **Purpose**: Cung cấp cột mốc lịch sử, tiến độ và tiêu chuẩn nghiệm thu cho tất cả 13 mốc phát triển ($M_0 \to M_{12}$).
- **Overview Pipeline**: Khởi tạo ($M_0 \to M_3$) -> Vòng lặp cơ sở ($M_4 \to M_7$) -> Quy mô sản xuất ($M_8 \to M_9$) -> Benchmark phổ quát ($M_{10} \to M_{11}$) -> Đóng gói giảng dạy ($M_{12}$).
- **Detailed Plan**: §1 Bảng lộ trình 13 mốc công việc ($M_0 \to M_{12}$); §2 Tiêu chuẩn kiểm chuẩn và chất lượng kỹ thuật.
- **References**: [../PURPOSE.md](../PURPOSE.md), [../OVERVIEW.md](../OVERVIEW.md), `TASK_PROGRESS.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00

---

## 1. Bảng Lộ Trình 13 Mốc Công Việc (`M0` $\rightarrow$ `M12`)

| Mốc | Tên Công Việc | Nội Dung Thực Hiện Kỹ Thuật | Sản Phẩm Đầu Ra | Trạng Thái |
| :--- | :--- | :--- | :--- | :---: |
| **`M0`** | Chốt đề bài & thiết kế hệ thống | Tiếp thu phản hồi hội đồng UTH: Chuyển dịch trọng tâm sang SD Inpainting $\to$ DINOv2 $\to$ Hard Mining $\to$ Active Learning. | Tài liệu thiết kế & Slide thuyết trình | `[x]` Đạt |
| **`M1`** | Chuẩn bị đa tạp ảnh thật (Real Manifold) | Tuyển chọn 10.000 ảnh từ FFHQ & CelebA-HQ, trích xuất 68 landmarks, kiểm định tính chính diện và phổ màu da. | Manifest ảnh thật 4.000 mẫu chuẩn (`manifest.jsonl`) | `[x]` Đạt |
| **`M2`** | Động cơ sinh ảnh Stable Diffusion v1.5 | Triển khai 3 phương thái sinh: `sd_inpaint`, `sd_img2img`, `sd_text2img`, áp dụng Soft-Mask Feathering ($\sigma=4.0$). | 2.745 ảnh sinh thô | `[x]` Đạt |
| **`M3`** | Lọc chất lượng tự động 3 tầng (CurationGate) | Áp dụng Face Detection ($\ge 0.85$), Laplacian Sharpness ($\ge 15.0$), CLIP-IQA ($\ge 0.60$) để triệt tiêu dị tật thô. | 2.200 ảnh sinh đạt chuẩn (`curated_manifest.jsonl`) | `[x]` Đạt |
| **`M4`** | Khai thác mẫu cực khó (Hard Sample Mining) | Cô lập các mẫu nằm sát ranh giới quyết định ($|P-0.5| \le 0.15$) và các mẫu vi phạm độ tự tin cao. | 740 mẫu khó khai thác, 213 mẫu Challenge Vault | `[x]` Đạt |
| **`M5`** | Phân vùng dữ liệu khóa danh tính (Identity-Disjoint) | Chia tập Train (2.813), Val (598), Test (589) đảm bảo cùng một người không bao giờ xuất hiện ở cả hai bên. | 100% Zero Leakage Asserted | `[x]` Đạt |
| **`M6`** | Huấn luyện mô hình cơ sở & tiến hóa $M_0 \to M_1$ | Huấn luyện DINOv2 trên $D_0$ (AUC: 0.9327), nạp 120 mẫu khó huấn luyện $M_1$ (AUC: 0.9445). | Checkpoints `baseline_d0_best.pt`, `active_learning_m1_best.pt` | `[x]` Đạt |
| **`M7`** | Báo cáo kiến trúc ban đầu & Slide 5 phút | Tổng hợp báo cáo kiến trúc hệ thống 3 bước và slide HTML tương tác tỷ lệ 16:9. | Bộ slide HTML và kịch bản bảo vệ | `[x]` Đạt |
| **`M8`** | Triệt tiêu Mock Data & Tích hợp DINOv2 Production | Xóa bỏ hoàn toàn mã giả/mock data, xác thực 15 checkpoint PyTorch thực tế, huấn luyện mô hình Champion $M_2^*$. | Checkpoint $M_2^*$ (Challenge Acc: 94.84%) | `[x]` Đạt |
| **`M9`** | Đa tạp khuếch tán kép (Dual-Engine Diffusion) | Tích hợp RealisticVision v5.1 làm động cơ thứ hai, đối đầu kết cấu vi thể da với SD 1.5. | Danh mục ảnh kép & bảng so sánh danh mục | `[x]` Đạt |
| **`M10`** | Kiểm định ngoại lai 6.000 ảnh (OOD Blind Transfer) | Kiểm thử mù detectors trên 6.000 ảnh RealisticVision v5.1: $M_0$ đạt 68.4%, $M_2^*$ nhảy vọt lên 89.1%. | `experiments/logs/legacy_6k_benchmark_results.json` | `[x]` Đạt |
| **`M11`** | Universal Deepfake Benchmark (20.400 ảnh) | Sát hạch trên 34 thuật toán sinh, 5 họ mô hình (GAN, Diffusion, Swap, Inpaint, Talking Head). | $M_2^*$ đạt 82.4% Accuracy, 0.897 AUC | `[x]` Đạt |
| **`M12`** | Chuẩn hóa bộ 8 Notebook & Master Visual Showcase | Xây dựng Notebook 00 Master Showcase (67 cells) và tái cấu trúc toàn bộ 8 notebooks chạy 100% không lỗi. | 8 Coursework Notebooks hoàn hảo không cell rỗng | `[x]` Đạt |

---

## 2. Tiêu Chuẩn Nghiệm Thu & Kiểm Chuẩn Chất Lượng Kỹ Thuật

1. **Kiểm thử tự động:** `make test-smoke` (12/12 passing trong < 6 giây) và `make test` (100% passing).
2. **Toàn vẹn số liệu:** 100% không mock data, không uninitialized random weights, 15 checkpoint PyTorch thực tế.
3. **Tiêu chuẩn Notebook:** 0 cell code rỗng, 0 cell thiếu output, 0 bare `plt.show()`, 100% hàm gọi từ `src.common.visualizer`.
4. **Tiêu chuẩn Markdown:** 100% file `.md` đặt tên UPPERCASE/UPPER_SNAKE_CASE, mở đầu bằng header 7 trường bắt buộc, không chứa đường dẫn tuyệt đối hệ thống cá nhân.
