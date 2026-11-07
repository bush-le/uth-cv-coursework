# Cấu Trúc Bộ Slide 8 Trang Siêu Trực Quan (Define & System Design — Nhóm 4)

- **Tệp trình chiếu chính**: `docs/reports/presentation_5min_define_and_design.html`
- **Tài liệu thiết kế gốc**: `docs/reports/2026-11-07-presentation-slide-deck-design.md`
- **Nhóm thực hiện**: `Nhóm thực hiện: Nhóm 4`
- **Bảng màu & Khung hiển thị**: `Blue Professional` (`#fdfae7` kem ấm, `#1e2bfa` xanh lam coban duy nhất) trên sân khấu cố định `1920×1080` (`viewport-base.css`).

---

## Bảng Ánh Xạ Chi Tiết 8 Slide (`#slide-1` $\to$ `#slide-8`)

| Mã Slide | Tiêu đề | Thành phần Đồ họa Vector & Chân dung Minh họa (Hướng 1 + 2 + 3) |
| :--- | :--- | :--- |
| **Slide 01** (`#slide-1`) | **Cover & Core System Vision** | Cột trái: Ghi danh `Nhóm thực hiện: Nhóm 4` + Dải đồ họa 4 nút (*Khuôn mặt gốc $\to$ Stable Diffusion $\to$ Phễu lọc 3 tầng $\to$ Detector*). Cột phải: Lộ trình 4 trụ cột với trục dẫn xung nhịp động. |
| **Slide 02** (`#slide-2`) | **01 / Problem Definition** | Trái: Đối chiếu chân dung *Deepfake cổ điển (lộ viền ghép)* vs. *Stable Diffusion (khử nhiễu toàn cục từ Latent có tia quét)*. Phải: Đồ thị không gian đặc trưng 2D & dịch chuyển **Decision Boundary**. |
| **Slide 03** (`#slide-3`) | **02 / Generation Mechanisms** | Lưới 3 cột toàn màn hình minh họa: `01. Text-to-Image` (Prompt $\to$ Noise $z_T \to$ Chân dung mới), `02. Facial Inpainting` (Ảnh gốc + Mặt nạ nhị phân mắt/miệng $\to$ Tái tạo cục bộ), `03. Image-to-Image` (Ảnh gốc + 5 điểm mốc $\to$ Biến đổi diện mạo). |
| **Slide 04** (`#slide-4`) | **03 / Scope & Boundaries** | Huy hiệu `Nhóm thực hiện: Nhóm 4` ở đầu trang + Ma trận 2 cột: 4 cam kết **In-Scope** vs. 4 giới hạn loại trừ **Out-of-Scope**. |
| **Slide 05** (`#slide-5`) | **04 / Dataset Design** | Huy hiệu `Nhóm thực hiện: Nhóm 4` + Trái: 3 nhóm nguồn gốc (`Real Faces`, `Deepfake Faces`, `SD Synthetic Faces`) & Thẻ định danh 6 trường. Phải: Két khóa sinh trắc học **Zero Identity Leakage**. |
| **Slide 06** (`#slide-6`) | **05 / System Architecture** | Sơ đồ **Đường ống 6 giai đoạn dự kiến** (`Stage 01` $\to$ `Stage 06`) có mũi tên dòng chảy động (`@keyframes flowPulse`) dẫn xuống thanh tác vụ hạ nguồn `Deepfake Detection & Comparative Evaluation`. |
| **Slide 07** (`#slide-7`) | **06 / Automated Curation Gate** | Trái: Chi tiết `Stage 02` (Quét khung mặt có tia laser `@keyframes scanLine` + Căn chỉnh trục hình học + Tách mặt nạ). Phải: Chi tiết `Stage 04` (Phễu lọc 3 tầng tách nhánh *Mẫu suy biến bị loại* vs. *Mẫu tổng hợp đạt chuẩn*). |
| **Slide 08** (`#slide-8`) | **07 / Evaluation & Deliverables** | Trên: Sơ đồ thực nghiệm đối chứng 2 nhánh (`Baseline` vs. `Proposed`) hội tụ vào `Deepfake Detector` + Đồ thị minh họa kỳ vọng đường cong **ROC-AUC**. Dưới: 4 thẻ sản phẩm bàn giao. |
