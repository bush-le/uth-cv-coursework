# README.md — Danh Mục Toàn Diện Hệ Thống Tài Liệu Đề Tài 11 (Nhóm 4)

- **Motivation/Background**: Cung cấp bản đồ điều hướng toàn diện (Master Navigation Sitemap) cho toàn bộ hệ thống tài liệu, kế hoạch kỹ thuật, báo cáo thực nghiệm và bộ 8 notebook giảng dạy của Đề tài 11.
- **Purpose**: Định vị chính xác vị trí, vai trò và mối liên kết logic giữa tất cả các thành phần trong thư mục `docs/`, `notebooks/` và `.agents/`.
- **Overview Pipeline**: Mục tiêu nghiên cứu ([PURPOSE.md](PURPOSE.md)) -> Thiết kế hệ thống ([OVERVIEW.md](OVERVIEW.md)) -> 8 Coursework Notebooks ([notebooks/coursework/](../notebooks/coursework/)) -> Đặc tả giai đoạn (`docs/phases/`) -> Kế hoạch kỹ thuật (`docs/plans/`) -> Báo cáo & Slide bảo vệ (`docs/reports/`).
- **Detailed Plan**: §1 Tài liệu cốt lõi; §2 Bộ 8 Coursework Notebooks; §3 Đặc tả kỹ thuật các giai đoạn; §4 Kế hoạch kỹ thuật chi tiết; §5 Báo cáo thực nghiệm & kiểm toán; §6 Slide thuyết trình & bảo vệ đồ án.
- **References**: [PURPOSE.md](PURPOSE.md), [OVERVIEW.md](OVERVIEW.md), `reports/MASTER_PROJECT_CROSSCHECK_AND_FACTCHECK_REPORT.md`.
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-11T23:25:00+07:00

---

## 1. Tài Liệu Cốt Lõi (Đọc Đầu Tiên)

- **[PURPOSE.md](PURPOSE.md)**: Đặc tả yêu cầu chính thức của giảng viên UTH, mục tiêu của **Nhóm 4** và luồng kiến trúc 3 bước.
- **[OVERVIEW.md](OVERVIEW.md)**: Tổng quan kiến trúc hệ thống từ sinh ảnh khuếch tán kép đến Active Learning và DINOv2.
- **[progress/PROJECT_ROADMAP.md](progress/PROJECT_ROADMAP.md)**: Lộ trình tiến hóa toàn diện qua 12 mốc từ $M_0$ đến $M_{12}$.
- **[progress/TASK_PROGRESS.md](progress/TASK_PROGRESS.md)**: Bảng theo dõi tiến độ chi tiết từng nhiệm vụ kỹ thuật.

---

## 2. Danh Mục Bộ 8 Coursework Notebooks Chuẩn Mực (`notebooks/coursework/`)

Toàn bộ 8 notebooks đã được chuẩn hóa theo kiến trúc mô-đun hóa (Clean Architecture), 100% cell chứa code thực thi từ `src.common.visualizer` và đã được thực thi kiểm chứng (Verified Pre-rendered Outputs):

| Notebook | Tên Khoa Học | Mục Tiêu & Trực Quan Hóa Chính |
| :--- | :--- | :--- |
| **[00_master_deepfake_defense_coursework.ipynb](../notebooks/coursework/00_master_deepfake_defense_coursework.ipynb)** | **Master Deepfake Defense Showcase** | **Toàn cảnh đồ án từ A-Z (67 cells): Đa tạp ảnh thật, SD kép, 213 mẫu khó, tiến hóa DINOv2, OOD 6k, Universal 20.4k, Live Inspector** |
| **[01_eda_faces_demographics.ipynb](../notebooks/coursework/01_eda_faces_demographics.ipynb)** | Exploratory Data Analysis on Real Faces | Khảo sát 10.000 ảnh thật FFHQ/CelebA-HQ, phân bố tuổi/giới tính/kính mắt, góc xoay đầu 3D, phổ màu da |
| **[02_sd_generation_and_curation.ipynb](../notebooks/coursework/02_sd_generation_and_curation.ipynb)** | Dual-Engine Synthesis & Multi-Stage Curation | Phân tích quy trình sinh ảnh SD 1.5 & RealisticVision v5.1, làm mềm biên Gaussian $\sigma=4.0$, phễu lọc 3 tầng |
| **[03_hard_samples_and_benchmark_comparison.ipynb](../notebooks/coursework/03_hard_samples_and_benchmark_comparison.ipynb)** | Hard-Sample Mining & Boundary Ambiguity | Khai thác 213 mẫu cực khó, t-SNE Manifold Intrusion, phân tích dị tật sinh học mắt/tai/da/răng |
| **[04_active_learning_production_visualizer.ipynb](../notebooks/coursework/04_active_learning_production_visualizer.ipynb)** | DINOv2 Architecture & Active Learning Evolution | Tiến hóa $M_0 \to M_1 \to M_2 \to M_2^*$, sơ đồ khối DINOv2, động học hội tụ, ROC, ma trận nhầm lẫn |
| **[05_synthetic_face_6k_realimage_benchmark.ipynb](../notebooks/coursework/05_synthetic_face_6k_realimage_benchmark.ipynb)** | OOD Blind Transfer on 6k RealisticVision Faces | Thử thách khái quát hóa ngoại lai trên 6.000 ảnh siêu thực, bước nhảy độ chính xác $68.4\% \to 89.1\%$ |
| **[06_specialization_vs_generalization_synthesis.ipynb](../notebooks/coursework/06_specialization_vs_generalization_synthesis.ipynb)** | Specialization vs Generalization Analysis | Phân tích cân bằng Pareto giữa độ chuyên biệt hóa (In-Distribution) và độ tổng quát hóa (OOD) |
| **[07_universal_deepfake_benchmark.ipynb](../notebooks/coursework/07_universal_deepfake_benchmark.ipynb)** | Universal Multi-Family Defense Benchmark | Đánh giá toàn cầu trên 20.400 ảnh, 34 thuật toán sinh, 5 họ mô hình (GAN, Diffusion, Swap, Inpaint, Talking Head) |

---

## 3. Đặc Tả Kỹ Thuật Các Giai Đoạn (`docs/phases/`)

- **[phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md](phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md)**: Đặc tả quy trình sinh ảnh khuếch tán kép và lọc chất lượng đa tầng.
- **[phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md](phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md)**: Đặc tả toán học khai thác mẫu khó dựa trên khoảng cách ranh giới quyết định.
- **[phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md](phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md)**: Đặc tả phân vùng dữ liệu không rò rỉ danh tính và tái huấn luyện bộ dò.
- **[phases/PHASE_EXTERNAL_6K_BENCHMARK.md](phases/PHASE_EXTERNAL_6K_BENCHMARK.md)**: Đặc tả bài kiểm tra mù OOD trên 6.000 ảnh RealisticVision v5.1.
- **[phases/PHASE_NOTEBOOK_06_SPECIALIZATION_VS_GENERALIZATION.md](phases/PHASE_NOTEBOOK_06_SPECIALIZATION_VS_GENERALIZATION.md)**: Đặc tả phân tích tương quan giữa In-Distribution và Out-of-Distribution.

---

## 4. Kế Hoạch Triển Khai Kỹ Thuật (`docs/plans/`)

- **[plans/2026_11_09_BASELINE_AUDIT_AND_CORRECTION_PLAN.md](plans/2026_11_09_BASELINE_AUDIT_AND_CORRECTION_PLAN.md)**: Kế hoạch kiểm toán và chuẩn hóa mô hình cơ sở $M_0$.
- **[plans/2026_11_10_ACTIVE_LEARNING_D2_PLAN.md](plans/2026_11_10_ACTIVE_LEARNING_D2_PLAN.md)**: Kế hoạch triển khai vòng lặp Active Learning $D_2$ và tinh chỉnh $M_2^*$.
- **[plans/2026_11_10_SYSTEM_REMEDIATION_AND_SCALING_PLAN.md](plans/2026_11_10_SYSTEM_REMEDIATION_AND_SCALING_PLAN.md)**: Kế hoạch tối ưu hóa hiệu năng, triệt tiêu mock data và mở rộng quy mô.
- **[plans/2026_11_10_UNIVERSAL_BENCHMARK_PLAN.md](plans/2026_11_10_UNIVERSAL_BENCHMARK_PLAN.md)**: Kế hoạch tích hợp và đánh giá Universal Benchmark 20.400 ảnh.

---

## 5. Báo Cáo Nghiên Cứu & Kiểm Toán Thực Nghiệm (`docs/reports/`)

- **[reports/MASTER_PROJECT_CROSSCHECK_AND_FACTCHECK_REPORT.md](reports/MASTER_PROJECT_CROSSCHECK_AND_FACTCHECK_REPORT.md)**: **Báo cáo kiểm toán chéo và xác minh sự thật toàn diện nhất của dự án**.
- **[reports/MASTER_SYSTEM_AUDIT_AND_IMPROVEMENT_REPORT.md](reports/MASTER_SYSTEM_AUDIT_AND_IMPROVEMENT_REPORT.md)**: Báo cáo tổng kiểm toán hệ thống và các điểm cải tiến thuật toán.
- **[reports/ACTIVE_LEARNING_M2_REFINED_REPORT.md](reports/ACTIVE_LEARNING_M2_REFINED_REPORT.md)**: Báo cáo chi tiết kết quả huấn luyện mô hình Champion $M_2^*$.
- **[reports/ACTIVE_LEARNING_HARD_SAMPLE_AUDIT_REPORT.md](reports/ACTIVE_LEARNING_HARD_SAMPLE_AUDIT_REPORT.md)**: Báo cáo kiểm toán 213 mẫu cực khó và 5 dạng dị tật sinh học.
- **[reports/LEGACY_6K_OOD_BENCHMARK_REPORT.md](reports/LEGACY_6K_OOD_BENCHMARK_REPORT.md)**: Báo cáo thực nghiệm OOD trên 6.000 ảnh RealisticVision v5.1.
- **[reports/EDA_UNIVERSAL_BENCHMARK_REPORT.md](reports/EDA_UNIVERSAL_BENCHMARK_REPORT.md)**: Báo cáo phân tích thăm dò tập dữ liệu Universal 34 phương pháp.
- **[reports/BENCHMARK_VAULT_INTEGRITY_AUDIT_REPORT.md](reports/BENCHMARK_VAULT_INTEGRITY_AUDIT_REPORT.md)**: Báo cáo tính toàn vẹn và độ tin cậy của các kho dữ liệu kiểm thử.
- **[reports/CURATION_GATE_MATHEMATICAL_AUDIT_REPORT.md](reports/CURATION_GATE_MATHEMATICAL_AUDIT_REPORT.md)**: Cơ sở toán học của 3 tầng lọc Curation Gate.
- **[reports/DIFFUSION_ENGINE_AND_PROMPT_GRAMMAR_AUDIT_REPORT.md](reports/DIFFUSION_ENGINE_AND_PROMPT_GRAMMAR_AUDIT_REPORT.md)**: Phân tích ngữ pháp prompt và các tham số khuếch tán.
- **[reports/DETECTOR_ARCHITECTURE_AND_TRAINING_AUDIT.md](reports/DETECTOR_ARCHITECTURE_AND_TRAINING_AUDIT.md)**: Phân tích kiến trúc mạng nơ-ron và hàm mất mát.
- **[reports/VISUAL_ANATOMICAL_ARTIFACT_AUDIT_REPORT.md](reports/VISUAL_ANATOMICAL_ARTIFACT_AUDIT_REPORT.md)**: Báo cáo phân loại dị tật giải phẫu học trên khuôn mặt nhân tạo.

---

## 6. Slide Thuyết Trình & Tài Liệu Bảo Vệ Đồ Án

- **[reports/presentation_5min_define_and_design.html](reports/presentation_5min_define_and_design.html)**: Bộ 11 Slide HTML thuyết trình tương tác (chuẩn tỉ lệ $16:9$, độ phân giải $1920 \times 1080$).
- **[reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md](reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md)**: Kịch bản và lời thoại chi tiết cho từng slide thuyết trình.
- **[reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md](reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md)**: Tóm tắt cấu trúc và phân bổ thời gian cho bài bảo vệ.
- **[reports/PRESENTATION_DEFENSE_QA_CHEAT_SHEET.md](reports/PRESENTATION_DEFENSE_QA_CHEAT_SHEET.md)**: Bộ câu hỏi và câu trả lời phản biện nhanh trước hội đồng.
