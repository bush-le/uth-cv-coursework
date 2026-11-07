# Cấu Trúc Bộ Slide Hợp Nhất 11 Trang: Define & System Design (2D Image & SD-Video — Nhóm 4)

Tài liệu này mô tả kiến trúc trực quan của bộ slide hợp nhất 11 trang (`docs/reports/presentation_5min_define_and_design.html`) kết hợp nền tảng ảnh tĩnh 2D (`baseline/initial-baseline`) và nhánh mở rộng chiều thời gian trên video (`sd-video`) cho Đề tài 11 (**Nhóm thực hiện: Nhóm 4**).

---

## 1. Nguyên Tắc Thiết Kế Cốt Lõi

1. **Giao diện Trình chiếu Tinh gọn (`100% Clean UI`)**:
   - Cố định khung hình `1920×1080` (`16:9` chuẩn `viewport-base.css`), bảng màu đơn sắc `Blue Professional` (`--bg: #fdfae7`, `--primary: #1e2bfa`, `--text: #111111`, `--text-muted: #6b6b6b`).
   - Loại bỏ hoàn toàn ngăn kéo lời dẫn (`N`) và popup tài liệu tham khảo (`R`), chỉ giữ thanh điều hướng tối giản ở đáy khung hình (`SLIDE 01 / 11`, nút `‹ / ›`, `11` chấm điều hướng và thanh tiến độ).
2. **Không Số Liệu Giả Định (`Zero Unexecuted Metrics`)**:
   - Không hiển thị bất kỳ con số độ phân giải cứng, số lượng mẫu, ngưỡng lọc hay kết quả thực nghiệm chưa chạy trong văn bản hiển thị.
3. **Ghi Danh Nhóm Thực Hiện Rõ Ràng (`Nhóm thực hiện: Nhóm 4`)**:
   - Hiển thị nổi bật trên **Slide 01** (Trang bìa), **Slide 03** (Phạm vi 2 tầng), **Slide 04** (3 chế độ tạo sinh 2D), và **Slide 10** (Thiết kế bộ dữ liệu hợp nhất & Két khóa cụm liên thông).

---

## 2. Chi Tiết Bố Cục 11 Slides

- **Slide 01 (`#slide-1`)**: Trang bìa *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*, dải đồ họa liên hoàn **Chặng 1 (Ảnh tĩnh 2D) $\to$ Chặng 2 (Video `A←B`) $\to$ Cổng QA & Két khóa $\to$ Detector (`D0/D1/D2`)** và lộ trình 4 trụ cột (`Nhóm thực hiện: Nhóm 4`).
- **Slide 02 (`#slide-2`)**: `01 / PROBLEM DEFINITION` — Sơ đồ đối chiếu vết ghép cục bộ (`Face-Swap`) với khử nhiễu toàn cục (`Stable Diffusion`) và đồ thị dịch chuyển biên quyết định (`Decision Boundary`).
- **Slide 03 (`#slide-3`)**: `02 / UNIFIED SCOPE & BOUNDARIES` (`Nhóm thực hiện: Nhóm 4`) — Ma trận phạm vi hợp nhất 2 tầng (**Tầng 1: Ảnh tĩnh 2D**, **Tầng 2: Video ngắn `A←B`**, **Khóa cụm liên thông**, **Đối chứng `D0/D1/D2`**) đối chiếu với 4 giới hạn loại trừ (`X1`–`X4`).
- **Slide 04 (`#slide-4`)**: `03 / 2D GENERATION MECHANISMS` (`Nhóm thực hiện: Nhóm 4`) — Bộ sưu tập 3 cột chân dung minh họa cho `01. Text-to-Image (T2I)`, `02. Facial Inpainting`, và `03. Image-to-Image (Img2Img)`.
- **Slide 05 (`#slide-5`)**: `04 / 2D SYSTEM ARCHITECTURE` — Sơ đồ đường ống 6 giai đoạn tự động có hiệu ứng xung dữ liệu (`Stage 01` $\to$ `Stage 06`) bàn giao xuống `Downstream Detector`.
- **Slide 06 (`#slide-6`)**: `05 / 2D AUTOMATED CURATION GATE` — Sơ đồ tiền xử lý 5 điểm mốc & tách mặt nạ (`Stage 02`) kết hợp phễu sàng lọc chất lượng 3 tầng (`Stage 04`).
- **Slide 07 (`#slide-7`)**: `06 / VIDEO EXTENSION: DEFINE` — Phương trình thị giác `Ảnh danh tính A + Video chuyển động B (t1, t2, t3 có tay che ngang mặt ở t3) → Video hoán đổi A←B` kèm thanh timeline động (`@keyframes playhead`) và 3 yêu cầu kỹ thuật cốt lõi (`Danh tính`, `Diễn xuất`, `Liên tục thời gian & Che khuất`).
- **Slide 08 (`#slide-8`)**: `07 / VIDEO MODEL CANDIDATES` — Sơ đồ đối chiếu 4 mô hình SD Video trong 2 họ kiến trúc (**Họ 01: `Face-Adapter`, `REFace`** vs. **Họ 02: `VFace`, `VividFace`**) cùng ma trận thử nghiệm Pilot và nguyên tắc tiền xử lý đặc thù.
- **Slide 09 (`#slide-9`)**: `08 / VIDEO PIPELINE & 3-AXIS QA` — Đường ống sinh video tái lập 6 bước (khóa cấu hình trên tập Development trước khi chạy Holdout) và sơ đồ chấm điểm 3 trục (`01 Identity`, `02 Motion`, `03 Temporal & Occlusion`) nối vào cổng QA (`Pass / Quarantine`).
- **Slide 10 (`#slide-10`)**: `09 / UNIFIED DATASET & ZERO LEAKAGE VAULT` (`Nhóm thực hiện: Nhóm 4`) — Ba nhóm nguồn gốc dữ liệu (Ảnh 2D & Video), thẻ căn cước mẫu 6 trường truy vết và sơ đồ **Két khóa cụm liên thông (`ZERO ID & VIDEO LEAK`)** khóa đồng thời `Danh tính gốc A` lẫn `Video cha B`.
- **Slide 11 (`#slide-11`)**: `10 / DETECTOR EVALUATION & ROADMAP` — Sơ đồ thực nghiệm đối chứng 3 chế độ huấn luyện (`D0: Baseline gốc`, `D1: + Augmentation thường`, `D2: + SD Synthetic qua QA`) hội tụ vào `Deepfake Detector`, đồ thị đường cong `ROC-AUC` minh họa và 4 thẻ lộ trình thực thi (`M0 → M7`).
