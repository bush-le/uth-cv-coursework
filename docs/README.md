# README.md — Danh Mục Tài Liệu Của Dự Án (Nhóm 4)

- **Motivation/Background**: Giúp người đọc tìm nhanh tất cả tài liệu và slide thuyết trình của đề tài sau khi chỉnh lại theo yêu cầu của giảng viên.
- **Purpose**: Liệt kê rõ từng file tài liệu trong thư mục `docs/` và chức năng của từng file.
- **Overview Pipeline**: Đề bài ([PURPOSE.md](PURPOSE.md)) -> Tổng quan hệ thống ([OVERVIEW.md](OVERVIEW.md)) -> Chi tiết 3 bước thực hiện (`docs/phases/`) -> Báo cáo & Slide thuyết trình (`docs/reports/`).
- **Detailed Plan**: §1 Tài liệu chính; §2 Chi tiết 3 bước; §3 Lý thuyết & thực nghiệm; §4 Báo cáo & bộ 11 Slide.
- **References**: [PURPOSE.md](PURPOSE.md), [OVERVIEW.md](OVERVIEW.md), [SYSTEM_DEFINE_AND_DESIGN_REPORT.md](reports/SYSTEM_DEFINE_AND_DESIGN_REPORT.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-09T13:19:40+07:00

---

## 1. Tài Liệu Chính (Đọc Đầu Tiên)

- **[PURPOSE.md](PURPOSE.md)**: Giải thích đề bài, mục tiêu của **Nhóm 4** và luồng 3 bước theo đúng yêu cầu của giảng viên.
- **[OVERVIEW.md](OVERVIEW.md)**: Tóm tắt cách hệ thống hoạt động từ lúc sinh ảnh giả bằng Stable Diffusion đến lúc huấn luyện lại mô hình `dinoViT`.
- **[progress/PROJECT_ROADMAP.md](progress/PROJECT_ROADMAP.md)**: Lộ trình thực hiện từng chặng từ `M0` đến `M7`.

---

## 2. Chi Tiết 3 Bước Thực Hiện (`docs/phases/`)

- **[phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md](phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md)**: **Bước 1** — Cách dùng Stable Diffusion tạo ảnh khuôn mặt giả (3 chế độ) và lọc bỏ ảnh bị lỗi.
- **[phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md](phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md)**: **Bước 2** — Cách dùng mô hình gốc `dinoViT` để kiểm tra ảnh giả: ảnh nào lọt lưới (`Bypass`) thì gom lại làm **Mẫu khó (`Hard Sample`)**.
- **[phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md](phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md)**: **Bước 3** — Cách gộp dữ liệu (`Dữ liệu gốc + Ảnh sinh mới + Mẫu khó`), chia tập không trùng người và huấn luyện lại `dinoViT`.

---

## 3. Lý Thuyết & Cách Chấm Điểm (`docs/coursework/` & `docs/experiments/`)

- **[coursework/LITERATURE_REVIEW.md](coursework/LITERATURE_REVIEW.md)**: Giải thích ngắn gọn vì sao ảnh tạo bởi Stable Diffusion dễ đánh lừa các mô hình cũ và vì sao chọn `dinoViT`.
- **[coursework/TAXONOMY_AND_DATA_SPEC.md](coursework/TAXONOMY_AND_DATA_SPEC.md)**: Cách phân loại 3 nhóm dữ liệu (`baseline`, `synthetic`, `hard_sample`) và các thông tin lưu kèm mỗi bức ảnh.
- **[experiments/EXPERIMENT_PROTOCOL.md](experiments/EXPERIMENT_PROTOCOL.md)**: Cách so sánh 3 chế độ huấn luyện `D0` (chỉ dữ liệu gốc), `D1` (thêm ảnh sinh mới thường), `D2` (thêm cả mẫu khó) để chứng minh tác dụng.

---

## 4. Báo Cáo & Bộ Slide Thuyết Trình 5 Phút (`docs/reports/`)

- **[reports/presentation_5min_define_and_design.html](reports/presentation_5min_define_and_design.html)**: Bộ 11 Slide HTML thuyết trình trực quan (tự động vừa khung màn hình `1920x1080`).
- **[reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md](reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md)**: Lời thoại gợi ý dễ đọc cho từng slide từ `Slide 01` đến `Slide 11`.
- **[reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md](reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md)**: Tóm tắt bố cục và nội dung chính của từng slide.
- **[reports/PART_01_PROBLEM_DEFINITION.md](reports/PART_01_PROBLEM_DEFINITION.md)**: Báo cáo Phần 1 — Đặt vấn đề và phạm vi thực hiện.
- **[reports/PART_02_SYSTEM_ARCHITECTURE.md](reports/PART_02_SYSTEM_ARCHITECTURE.md)**: Báo cáo Phần 2 — Thiết kế hệ thống 3 bước.
- **[reports/SYSTEM_DEFINE_AND_DESIGN_REPORT.md](reports/SYSTEM_DEFINE_AND_DESIGN_REPORT.md)**: Báo cáo tổng hợp hoàn chỉnh.
- **[reports/PRESENTATION_DEFENSE_QA_CHEAT_SHEET.md](reports/PRESENTATION_DEFENSE_QA_CHEAT_SHEET.md)**: Bộ câu hỏi và câu trả lời ngắn gọn khi bảo vệ trước giảng viên.

## Kaggle image pilot

- [Guide](reports/KAGGLE_SD_IMAGE_DEMO.md) / [HTML](reports/KAGGLE_SD_IMAGE_DEMO.html)
- [Notebook](../notebooks/SD_IMAGE_DEMO_KAGGLE.ipynb)
