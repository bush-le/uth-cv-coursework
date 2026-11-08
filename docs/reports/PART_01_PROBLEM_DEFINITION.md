# PART_01_PROBLEM_DEFINITION.md — Báo Cáo Phần 1: Đặt Vấn Đề & Phạm Vi Thực Hiện (Nhóm 4)

- **Motivation/Background**: Trình bày toàn diện bài toán nghiên cứu từ góc nhìn Phân loại Deepfake (What), Stable Diffusion là gì và tạo data dạng nào, bộ ba luận cứ Problem -> Business -> Technical, và luồng 3 bước thực tế theo định hướng của giảng viên.
- **Purpose**: Làm tài liệu nền tảng cho Phần 1 của báo cáo đề tài (tương ứng với `Slide 01` đến `Slide 04`).
- **Overview Pipeline**: Phân loại Deepfake (Cũ vs. Mới) -> Stable Diffusion & 3 dạng dữ liệu -> Bộ 3 luận cứ Problem - Business - Technical -> Luồng 3 bước kỹ thuật của Nhóm 4 -> Phạm vi thực hiện.
- **Detailed Plan**: §1 Phân loại Deepfake (What is Deepfake?); §2 Stable Diffusion là gì & hỗ trợ tạo data dạng nào; §3 Bộ ba luận cứ Problem -> Business -> Technical; §4 Luồng 3 bước kỹ thuật khép kín; §5 Phạm vi thực hiện & giới hạn loại trừ.
- **References**: [../PURPOSE.md](../PURPOSE.md), [PART_02_SYSTEM_ARCHITECTURE.md](PART_02_SYSTEM_ARCHITECTURE.md), [presentation_5min_define_and_design.html](presentation_5min_define_and_design.html).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T19:00:00+07:00

---

## 1. Phân Loại Deepfake (What Is Deepfake?)

Deepfake là thuật ngữ chỉ các nội dung hình ảnh, video hoặc âm thanh khuôn mặt con người bị làm giả một cách tinh vi bằng Trí tuệ Nhân tạo (AI). Trong xử lý ảnh khuôn mặt, Deepfake được phân loại thành 2 thế hệ chính:

1. **Thế hệ 1 — Deepfake cắt ghép truyền thống (Face-Swap / Blending)**:
   - *Cách làm*: Cắt vùng mặt của người nguồn rồi dán đè lên khuôn mặt của người đích (dùng Autoencoder hoặc thuật toán đồ họa).
   - *Đặc điểm nhận diện*: Thường để lại **đường viền cắt dán mờ quanh viền mặt**, răng cưa biên ghép nối hoặc lệch tông màu da và ánh sáng giữa vùng mặt với cổ.
   - *Khả năng bắt giữ*: Các mô hình phát hiện truyền thống (CNNs, Xception) chỉ cần tìm ra các vết ranh giới cắt ghép này là phát hiện được.

2. **Thế hệ 2 — Deepfake tạo sinh khuếch tán (Diffusion-Based Deepfakes / Stable Diffusion)**:
   - *Cách làm*: Sinh trực tiếp toàn bộ ảnh hoặc vẽ lại nội vùng từ không gian tiềm ẩn (Latent Space) qua quá trình khử nhiễu liên tục.
   - *Đặc điểm nhận diện*: **Không còn bất kỳ vết cắt ghép hay ranh giới nối nào**. Toàn bộ kết cấu da mặt, lỗ chân lông, mắt, miệng và ánh sáng xung quanh đều liền mạch và tự nhiên.
   - *Khả năng bắt giữ*: Các mô hình phát hiện cũ bị "mù" hoàn toàn vì không còn vết cắt ghép để nhận diện, dẫn đến đoán nhầm ảnh giả thành ảnh thật.

---

## 2. Stable Diffusion Là Như Thế Nào & Hỗ Trợ Tạo Dữ Liệu Dạng Nào?

### 2.1. Stable Diffusion Là Gì?
Stable Diffusion là mô hình tạo sinh khuếch tán trong không gian tiềm ẩn (Latent Diffusion Model - LDM). Thay vì tạo trực tiếp trên từng điểm ảnh gây tốn kém bộ nhớ, Stable Diffusion nén ảnh vào không gian tiềm ẩn nhỏ gọn (`Latent Space`), bắt đầu từ một lưới nhiễu ngẫu nhiên (`Noise z_T`), sau đó sử dụng mạng U-Net kết hợp cơ chế hướng dẫn (`Cross-Attention`) để khử nhiễu từng bước (`z_T → z_0`), và cuối cùng dùng bộ giải mã VAE để tái tạo lại bức ảnh khuôn mặt chân thực.

### 2.2. Stable Diffusion Hỗ Trợ Tạo 3 Dạng Dữ Liệu Khuôn Mặt Bổ Trợ
Trong đề tài này, **Nhóm 4** khai thác Stable Diffusion để tạo ra 3 dạng dữ liệu khuôn mặt giả khác nhau:

1. **Dạng 1: `Text-to-Image (T2I)` — Sinh danh tính nhân tạo mới hoàn toàn (`sd_text2img`)**:
   - Nhận câu mô tả chữ (Prompt: độ tuổi, giới tính, biểu cảm, góc chụp) và sinh ra một khuôn mặt người hoàn toàn mới.
   - *Tác dụng*: Mở rộng độ đa dạng chủng tộc, lứa tuổi mà không sợ trùng lặp với người thật ngoài đời.
2. **Dạng 2: `Facial Inpainting` — Sửa cục bộ các vùng nhạy cảm (`sd_inpainting`)**:
   - Dùng mặt nạ nhị phân khoanh vùng mắt, mũi hoặc miệng trên ảnh mặt người thật, sau đó dùng Stable Diffusion vẽ lại riêng vùng đó theo biểu cảm mới.
   - *Tác dụng*: Giữ nguyên 100% hình dáng đầu, tóc và nền thật, chỉ làm giả một phần khuôn mặt. Đây là dạng dữ liệu giả tinh vi nhất, đánh trúng điểm mù của các mô hình phát hiện.
3. **Dạng 3: `Image-to-Image (Img2Img)` — Biến đổi diện mạo giữ khung hình học (`sd_img2img`)**:
   - Đưa ảnh khuôn mặt thật vào không gian tiềm ẩn, cộng thêm nhiễu nhẹ và hướng dẫn Stable Diffusion thay đổi thuộc tính (trẻ hóa, già đi, đổi ánh sáng) nhưng giữ nguyên khung 5 điểm mốc (2 mắt, mũi, 2 khóe miệng).
   - *Tác dụng*: Tạo ra các cặp ảnh đối chứng (Gốc vs. Biến đổi), giúp mô hình học cách phân biệt sự thay đổi cấu trúc bề mặt da.

---

## 3. Bộ Ba Luận Cứ: Problem ➔ Business ➔ Technical

Để trả lời thuyết phục câu hỏi của giảng viên và hội đồng, đề tài được cấu trúc chặt chẽ theo 3 tầng luận cứ:

```text
[TẦNG 1: PROBLEM]   Sự dịch chuyển từ Deepfake cắt ghép sang Deepfake Stable Diffusion.
                    Ảnh giả quá tự nhiên khiến các mô hình phát hiện cũ bị "mù".
                             │
                             ▼
[TẦNG 2: BUSINESS]  Nguy cơ trực tiếp đe dọa cổng xác thực danh tính điện tử eKYC,
                    lừa đảo ngân hàng, giả mạo danh tính cá nhân và rủi ro an ninh.
                    Doanh nghiệp cần bộ lọc phát hiện chủ động, có thể bắt được ảnh SD.
                             │
                             ▼
[TẦNG 3: TECHNICAL] Chu trình 3 bước khép kín của Nhóm 4:
                    Sinh ảnh SD  ──►  Qua dinoViT tìm Mẫu khó  ──►  Huấn luyện lại chứng minh tác dụng.
```

1. **Tầng 1 — Problem (Vấn Đề Kỹ Thuật)**:
   - Các mô hình phát hiện Deepfake hiện có dựa trên giả định ảnh giả sẽ có vết ghép nối.
   - Khi Stable Diffusion xuất hiện, ranh giới phân lớp cũ bị lệch (Domain Shift), tạo ra một "vùng mù" lớn làm bỏ sót các ảnh giả chất lượng cao.
2. **Tầng 2 — Business Impact (Ý Nghĩa Nghiệp Vụ & Ứng Dụng Thực Tế)**:
   - **Xác thực điện tử (eKYC)**: Mở tài khoản ngân hàng từ xa, ví điện tử, chấm công sinh trắc học và cổng dịch vụ công đang phụ thuộc vào việc nhận diện khuôn mặt thật.
   - **Rủi ro tài chính & an ninh**: Kẻ xấu dùng Stable Diffusion tạo khuôn mặt giả tinh vi để vượt qua bài kiểm tra eKYC, lừa đảo chiếm đoạt tài khoản hoặc mạo danh người khác.
   - **Nhu cầu của doanh nghiệp**: Không thể chờ kẻ xấu tấn công mới tìm cách vá. Doanh nghiệp cần chủ động tạo ra dữ liệu ảnh giả tinh vi để "huấn luyện trước" cho hệ thống phòng thủ.
3. **Tầng 3 — Technical Solution (Giải Pháp Kỹ Thuật Khép Kín Của Nhóm 4)**:
   - Nhóm không chỉ dừng lại ở việc sinh dữ liệu, mà xây dựng một vòng lặp khép kín:
     - **Phase 1**: Tạo dữ liệu ảnh giả bằng Stable Diffusion và lọc sạch qua phễu 3 tầng.
     - **Phase 2**: Dùng mô hình chuẩn `dinoViT` làm thước đo; ảnh nào lừa được `dinoViT` (`Bypass`) thì thu thập làm **Mẫu khó (`Hard Sample`)**.
     - **Phase 3**: Huấn luyện lại `dinoViT` trên 3 nguồn dữ liệu; nếu mô hình bắt lại được các mẫu khó đó $\rightarrow$ **chứng minh được tác dụng thực tế của dữ liệu**.

---

## 4. Chu Trình 3 Bước Kỹ Thuật Khép Kín (Luồng Chi Tiết)

```text
[Bước 1: Sinh dữ liệu]
Ảnh mặt thật + Stable Diffusion  ───►  Ảnh giả mới (Synthetic DF Image)
                                                │
                                                ▼
[Bước 2: Thử thách mô hình gốc (Baseline dinoViT)]
Đưa ảnh giả mới qua mô hình dinoViT gốc:
  ├── Nếu dinoViT BẮT ĐƯỢC (Reject)  ───►  Ảnh giả thông thường (Synthetic Data)
  └── Nếu dinoViT BỊ LỪA (Bypass)   ───►  Lấy làm MẪU KHÓ (Hard Sample)
                                                │
                                                ▼
[Bước 3: Huấn luyện lại (Fine-Tune) & Chứng minh tác dụng]
Gộp (Dữ liệu gốc + Ảnh sinh mới + Mẫu khó) ───► Huấn luyện lại dinoViT
  └── Kết quả: Mô hình mới bắt được các Mẫu khó trước đó ───► CHỨNG MINH DỮ LIỆU CÓ TÁC DỤNG!
```

---

## 5. Phạm Vi Thực Hiện & Những Phần Không Làm

- **4 Phần Nhóm Tập Trung Thực Hiện (`In-Scope`)**:
  1. Tạo ảnh khuôn mặt giả 2D bằng 3 chế độ Stable Diffusion và lọc bỏ ảnh lỗi tự động.
  2. Chạy kiểm tra qua Baseline `dinoViT` để tách ảnh bị bắt (`Reject`) và ảnh vượt rào (`Bypass` $\rightarrow$ `Hard Sample`).
  3. Chia tập Học / Thi không trùng người (`source_real_id`) để triệt tiêu hoàn toàn rò rỉ dữ liệu.
  4. So sánh 3 chế độ `D0` (chỉ dữ liệu gốc), `D1` (+ ảnh mới thường), `D2` (+ cả mẫu khó) để chứng minh tác dụng.

- **4 Giới Hạn Loại Trừ (`Out-of-Scope`)**:
  1. Không tạo giọng nói hay âm thanh giả.
  2. Không huấn luyện lại mạng nền Stable Diffusion từ đầu (tận dụng trọng số mở sẵn có).
  3. Không mạo danh cá nhân thật ngoài đời (chỉ dùng tập dữ liệu mở phục vụ nghiên cứu).
  4. Không tối ưu nén mô hình cho thiết bị di động biên.
