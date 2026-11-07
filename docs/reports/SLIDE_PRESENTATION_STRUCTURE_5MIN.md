# Cấu Trúc Bộ Slide Trình Chiếu 5 Phút: Define & Design System

Tài liệu này mô tả cấu trúc hiển thị trực quan của tệp [`docs/reports/presentation_5min_define_and_design.html`](./presentation_5min_define_and_design.html) theo đúng chuẩn khung hình cố định `1920×1080` (`viewport-base.css`) và hệ màu đơn sắc `Blue Professional` (`#fdfae7` + `#1e2bfa`).

## 1. Quy Chuẩn Trực Quan Hóa Toàn Cục
- **Khung hình trình chiếu cố định**: `1920px × 1080px` (tỷ lệ 16:9), tự động co giãn đồng nhất theo kích thước màn hình trình duyệt mà không làm lệch bố cục.
- **Lấp đầy 100% không gian mỗi slide**: Sử dụng hệ lưới CSS Grid đa hàng/đa cột co giãn toàn khung (`1776px × 820px` vùng nội dung), kết hợp sơ đồ vector `<svg>` nội tuyến trên tất cả 6 slide.
- **Bảng màu `Blue Professional` thuần khiết**:
  - Nền kem ấm (`#fdfae7`)
  - Điểm nhấn xanh Cobalt điện tử duy nhất (`#1e2bfa`)
  - Chữ chính đen đậm (`#111111`) và chữ phụ xám trung tính (`#6b6b6b`)
- **Tính trung thực học thuật**: Chỉ trình bày định hướng thiết kế, cấu trúc luồng dữ liệu và phương pháp đánh giá dự kiến; không đưa các con số giả định hoặc thông số chưa qua thực nghiệm.

---

## 2. Bản Đồ 6 Slide Trực Quan

| Slide | Nhãn Phần | Bố Cục Trực Quan (100% Canvas Fill) | Sơ Đồ Vector `<svg>` Trung Tâm |
| :--- | :--- | :--- | :--- |
| **Slide 01** | `COVER & ROADMAP` | Chia 2 cột toàn khung (`1.08fr : 0.92fr`) | Sơ đồ trục dọc 4 trụ cột từ `Define Problem` đến `Design Evaluation` |
| **Slide 02** | `01 / PROBLEM DEFINITION` | Lưới 2 tầng (`1.38fr : 0.82fr`) | Sơ đồ đối chiếu 2 luồng: *Khai thác dữ liệu tĩnh thụ động* vs. *Tổng hợp dữ liệu chủ động với Stable Diffusion* |
| **Slide 03** | `02 / OBJECTIVES & SCOPE` | Lưới 2 tầng (`1.32fr : 0.88fr`) | 3 sơ đồ luồng `Input → Stable Diffusion → Output` cho *Text-to-Image*, *Facial Inpainting*, *Image-to-Image* + Ma trận *In-Scope / Out-of-Scope* |
| **Slide 04** | `03 / DATASET DESIGN` | Chia 2 cột (`0.94fr : 1.06fr`) | Sơ đồ **Identity-Disjoint Partitioning** với rào chắn độc lập danh tính giữa tập Train/Val và tập Test + Lưới 6 trường Metadata |
| **Slide 05** | `04 / SYSTEM ARCHITECTURE` | Lưới 2 tầng Hero (`1.56fr : 0.64fr`) | Sơ đồ **Đường ống 6 giai đoạn dự kiến** (`Data Acquisition` $\to$ `Face Preprocessing` $\to$ `Stable Diffusion Generation` $\to$ `Quality Control` $\to$ `Labeling & Metadata` $\to$ `Dataset Export` $\implies$ `Downstream Deepfake Detection`) |
| **Slide 06** | `05 / EVALUATION & EXPECTED OUTCOMES` | Lưới 2 tầng (`1.26fr : 0.84fr`) | Sơ đồ **Thực nghiệm đối chứng 2 nhánh** (`Baseline` vs. `Proposed` $\to$ `Deepfake Detector` $\to$ `Held-Out Test Set` $\to$ `Metrics`) + 4 thẻ sản phẩm đầu ra |
