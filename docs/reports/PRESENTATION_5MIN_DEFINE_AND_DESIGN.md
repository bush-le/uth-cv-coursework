# Kịch Bản Thuyết Trình Hợp Nhất 11 Slides: Define & System Design (2D Image & SD-Video — Đề Tài 11)

- **Đề tài**: *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*
- **Nhóm thực hiện**: **Nhóm 4**
- **Tệp trình chiếu tương tác**: `docs/reports/presentation_5min_define_and_design.html`
- **Giao diện trình chiếu**: `1920×1080` (`16:9` cố định), `Blue Professional` (`#fdfae7` / `#1e2bfa`), **100% Clean UI** (không hộp thoại ẩn, không số liệu chưa thực thi).

---

## Bảng Phân Bổ Cấu Trúc 11 Slides Theo 3 Chặng

| Chặng | Slide | Tiêu đề | Trọng tâm trực quan |
| :--- | :--- | :--- | :--- |
| **Chặng I: Tầm nhìn & Nền tảng 2D** | **Slide 01** | Cover & Unified 2-Track Vision | Dải đồ họa liên hoàn: Chặng 1 (Ảnh tĩnh 2D) $\to$ Chặng 2 (Video `A←B`) $\to$ Cổng QA $\to$ Detector (`D0/D1/D2`) & Lộ trình 4 trụ cột (`Nhóm thực hiện: Nhóm 4`). |
| | **Slide 02** | `01 / PROBLEM DEFINITION` | Đối chiếu vết ghép cục bộ (`Face-Swap`) với khử nhiễu toàn cục (`Stable Diffusion`) & Dịch chuyển biên quyết định (`Decision Boundary`). |
| | **Slide 03** | `02 / UNIFIED SCOPE & BOUNDARIES` | Ma trận phạm vi 2 tầng (**Tầng 1: Ảnh tĩnh 2D** + **Tầng 2: Mở rộng Video ngắn `A←B`**) & 4 giới hạn loại trừ (`Nhóm thực hiện: Nhóm 4`). |
| | **Slide 04** | `03 / 2D GENERATION MECHANISMS` | Bộ sưu tập 3 cột chân dung minh họa: `01. Text-to-Image`, `02. Facial Inpainting` (kèm mặt nạ nhị phân), `03. Image-to-Image` (kèm 5 điểm mốc) (`Nhóm thực hiện: Nhóm 4`). |
| | **Slide 05** | `04 / 2D SYSTEM ARCHITECTURE` | Đường ống 6 giai đoạn trên miền ảnh tĩnh (`Stage 01` $\to$ `Stage 06` $\implies$ `Downstream Detector`). |
| | **Slide 06** | `05 / 2D AUTOMATED CURATION GATE` | Tiền xử lý 5 điểm mốc (`Stage 02`) có tia quét động & Phễu sàng lọc chất lượng 3 tầng (`Stage 04`). |
| **Chặng II: Mở rộng Video `A←B`** | **Slide 07** | `06 / VIDEO EXTENSION: DEFINE` | Phương trình thị giác `Ảnh danh tính A + Video chuyển động B (t1, t2, t3 có tay che mặt ở t3) → Video hoán đổi A←B` & 3 yêu cầu cốt lõi. |
| | **Slide 08** | `07 / VIDEO MODEL CANDIDATES` | Đối chiếu 4 mô hình SD Video trong 2 họ kiến trúc: **Họ 01 (`Face-Adapter`, `REFace`)** vs. **Họ 02 (`VFace`, `VividFace`)** & Ma trận thử nghiệm Pilot. |
| | **Slide 09** | `08 / VIDEO PIPELINE & 3-AXIS QA` | Đường ống sinh video tái lập có khóa tập Development & Cổng kiểm định 3 trục (`01 Identity`, `02 Motion`, `03 Temporal & Occlusion`). |
| **Chặng III: Két Cụm & Đối chứng** | **Slide 10** | `09 / UNIFIED DATASET & ZERO LEAKAGE VAULT` | Ba nhóm nguồn gốc (Ảnh 2D & Video), thẻ căn cước mẫu và **Két khóa cụm liên thông** khóa đồng thời `Danh tính A` lẫn `Video cha B` (`Nhóm thực hiện: Nhóm 4`). |
| | **Slide 11** | `10 / DETECTOR EVALUATION & ROADMAP` | Thực nghiệm đối chứng 3 chế độ huấn luyện (`D0` gốc vs. `D1` Augmentation thường vs. `D2` SD qua QA), đường cong `ROC-AUC` minh họa & Lộ trình `M0 → M7`. |

---

## Lời Dẫn Thuyết Trình Gợi Ý Theo Từng Slide

### Slide 01 — Trang Bìa & Tầm Nhìn Hợp Nhất (`Nhóm thực hiện: Nhóm 4`)
> "Xin chào Thầy/Cô và các bạn. Hôm nay **Nhóm 4** xin trình bày phần **Define Problem** và **System Design** cho Đề tài 11: *Tạo sinh bộ dữ liệu khuôn mặt tổng hợp bằng Stable Diffusion phục vụ phát hiện Deepfake*. Điểm nhấn trong kiến trúc của nhóm là lộ trình hợp nhất 2 chặng: khởi đầu từ nền tảng ảnh tĩnh 2D và mở rộng sang chiều thời gian trên các đoạn video hoán đổi khuôn mặt ngắn `A←B`."

### Slide 02 — `01 / PROBLEM DEFINITION`: Dịch Chuyển Miền & Biên Quyết Định
> "Về định nghĩa bài toán: các bộ phát hiện truyền thống thường dựa vào vết cắt ghép cục bộ của Face-Swap cổ điển. Khi đối mặt với ảnh và video sinh từ không gian tiềm ẩn của Stable Diffusion vốn có kết cấu liền mạch, biên quyết định cũ rơi vào vùng mù. Do đó, nhóm chủ động tạo sinh dữ liệu có kiểm soát để mở rộng biên quyết định cho Detector."

### Slide 03 — `02 / UNIFIED SCOPE & BOUNDARIES`: Phạm Vi 2 Tầng & Giới Hạn Loại Trừ (`Nhóm thực hiện: Nhóm 4`)
> "**Nhóm thực hiện: Nhóm 4** xác lập ma trận phạm vi 2 tầng rõ ràng: Tầng 1 là nền tảng ảnh tĩnh 2D và Tầng 2 là mở rộng video ngắn `A←B` kèm khóa cụm liên thông và đối chứng 3 chế độ `D0, D1, D2`. Đồng thời, nhóm loại trừ 4 hạng mục ngoài phạm vi: không tổng hợp âm thanh/giọng nói, không pre-train lại nền tảng Stable Diffusion từ đầu, không mạo danh cá nhân thật ngoài bộ dữ liệu học thuật mở, và không tối ưu phần cứng thiết bị biên."

### Slide 04 — `03 / 2D GENERATION MECHANISMS`: Ba Chế Độ Tạo Sinh Khuôn Mặt 2D (`Nhóm thực hiện: Nhóm 4`)
> "Trên tầng nền tảng ảnh tĩnh 2D, **Nhóm 4** kết hợp 3 cơ chế tạo sinh bổ trợ: **Text-to-Image** sinh danh tính nhân tạo hoàn toàn mới từ đặc tả văn bản; **Facial Inpainting** dùng mặt nạ nhị phân sửa đổi cục bộ vùng mắt, mũi, miệng trên khuôn mặt thật; và **Image-to-Image** biến đổi thuộc tính diện mạo nhưng bảo toàn bố cục hình học 5 điểm mốc."

### Slide 05 — `04 / 2D SYSTEM ARCHITECTURE`: Đường Ống 6 Giai Đoạn Dự Kiến
> "Để vận hành tự động trên miền ảnh tĩnh, hệ thống được thiết kế theo đường ống 6 giai đoạn nối tiếp: từ thu thập dữ liệu nguồn (**Stage 01**), tiền xử lý khuôn mặt (**Stage 02**), tạo sinh đa chế độ bằng Stable Diffusion (**Stage 03**), sàng lọc chất lượng (**Stage 04**), gán nhãn siêu dữ liệu (**Stage 05**) đến xuất bản phân tập (**Stage 06**) cho bộ phát hiện hạ nguồn."

### Slide 06 — `05 / 2D AUTOMATED CURATION GATE`: Tiền Xử Lý & Phễu Lọc 3 Tầng
> "Hai chốt chặn kỹ thuật quan trọng nhất trên miền 2D là **Stage 02** — nơi khuôn mặt được phát hiện 5 điểm mốc, căn chỉnh hình học và tách mặt nạ tự động — cùng **Stage 04**, nơi mọi ảnh tổng hợp phải đi qua phễu lọc 3 tầng về độ sắc nét, cấu trúc giải phẫu mặt và độ khớp ngữ nghĩa trước khi được đưa vào tập huấn luyện."

### Slide 07 — `06 / VIDEO EXTENSION: DEFINE`: Hoán Đổi Khuôn Mặt Trên Video (`Ảnh A + Video B → Video A←B`)
> "Bước sang Chặng 2 mở rộng chiều thời gian trên video: bài toán đặt ra là kết hợp **Ảnh danh tính A** với **Video chuyển động B** qua chuỗi khung hình liên tiếp `t1, t2, t3` để tạo ra **Video hoán đổi A←B**. Hệ thống phải giải quyết đồng thời 3 yêu cầu: giữ đúng danh tính của Người A, bám sát góc đầu và biểu cảm của Video B, và duy trì liên tục thời gian mà không vẽ đè lên bàn tay hay vật cản che mặt ở `t3`."

### Slide 08 — `07 / VIDEO MODEL CANDIDATES`: Bốn Mô Hình SD Video Trong 2 Họ Kiến Trúc
> "Để giải quyết bài toán này một cách thực증, nhóm đối sánh 4 mô hình Stable Diffusion Video tiền huấn luyện chia làm 2 họ kiến trúc: **Họ 01** xử lý độc lập từng khung hình gồm **Face-Adapter** và **REFace**; **Họ 02** có thành phần liên kết thời gian gồm **VFace** (mở rộng trực tiếp từ REFace) và **VividFace** (điều kiện hóa theo điểm mốc và cụm khung hình). Nhóm triển khai ma trận thử nghiệm Pilot trên tập cặp đầu vào đại diện và giữ nguyên quy cách tiền xử lý đặc thù của từng mô hình."

### Slide 09 — `08 / VIDEO PIPELINE & 3-AXIS QA`: Đường Ống Sinh Video Tái Lập & Cổng Kiểm Định 3 Trục
> "Đường ống sinh video được thiết kế tái lập hoàn toàn với nguyên tắc khóa cấu hình trên tập Development trước khi chạy tập Holdout. Mọi video sinh ra được chấm điểm tự động trên 3 trục: **Identity** (độ tương đồng Cosine nhận dạng), **Motion** (độ khớp góc đầu và biểu cảm), và **Temporal** (căn chỉnh bù chuyển động kết hợp mặt nạ vùng hợp lệ và vật cản che khuất), từ đó sàng lọc qua cổng QA trước khi đóng gói."

### Slide 10 — `09 / UNIFIED DATASET & ZERO LEAKAGE VAULT`: Bộ Dữ Liệu Hợp Nhất & Két Khóa Cụm Liên Thông (`Nhóm thực hiện: Nhóm 4`)
> "Ở tầng quản trị dữ liệu hợp nhất, **Nhóm 4** tổ chức 3 nhóm nguồn gốc dữ liệu với thẻ căn cước 6 trường truy vết. Đặc biệt, để ngăn chặn rò rỉ dữ liệu ở cả ảnh 2D lẫn video, nhóm áp dụng **Két khóa cụm liên thông (Connected-Component Vault)**: khóa đồng thời **Danh tính gốc A** lẫn **Video cha B**, đảm bảo không có bất kỳ khuôn mặt nguồn hay bối cảnh video gốc nào xuất hiện chéo giữa tập Huấn luyện và tập Kiểm thử độc lập."

### Slide 11 — `10 / DETECTOR EVALUATION & ROADMAP`: Đối Chứng `D0 / D1 / D2` & Lộ Trình `M0 → M7`
> "Cuối cùng, để chứng minh giá trị thực tế của dữ liệu tổng hợp, nhóm thiết kế thực nghiệm đối chứng 3 chế độ huấn luyện cùng ngân sách trên cùng kiến trúc Detector: **D0** (chỉ dữ liệu gốc), **D1** (bổ sung Augmentation cổ điển làm đối chứng công bằng), và **D2** (bổ sung dữ liệu tổng hợp Stable Diffusion 2D & Video đã qua cổng QA), đo lường qua đường cong **ROC-AUC**, **AP** và **TPR tại FPR cố định** theo lộ trình từ mốc **M0** đến **M7**. Em xin cảm ơn Thầy/Cô và các bạn đã lắng nghe."
