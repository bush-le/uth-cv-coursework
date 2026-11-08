# LITERATURE_REVIEW.md — Tổng Quan Lý Thuyết Đơn Giản & Dễ Hiểu

- **Motivation/Background**: Giải thích ngắn gọn vì sao các mô hình phát hiện Deepfake cũ thường thất bại trước ảnh tạo bởi Stable Diffusion, và vì sao nhóm sử dụng `dinoViT` kết hợp với việc tìm mẫu khó (`Hard Sample`).
- **Purpose**: Làm cơ sở lý thuyết dễ hiểu cho báo cáo và phần trả lời câu hỏi của giảng viên.
- **Overview Pipeline**: Sự khác nhau giữa ghép mặt cũ và Stable Diffusion -> Tại sao chọn `dinoViT` làm mô hình gốc -> Tại sao học từ các mẫu bị bỏ lọt (`Hard Sample`) lại giúp mô hình giỏi hơn.
- **Detailed Plan**: §1 Ghép mặt kiểu cũ khác gì với Stable Diffusion?; §2 Tại sao dùng `dinoViT` làm mô hình chuẩn?; §3 Tại sao phải tìm Mẫu khó (`Hard Sample`) để học lại?
- **References**: [ ../PURPOSE.md ](../PURPOSE.md), [ TAXONOMY_AND_DATA_SPEC.md ](TAXONOMY_AND_DATA_SPEC.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:38:00+07:00

---

## 1. Ảnh Ghép Mặt Kiểu Cũ Khác Gì Với Ảnh Tạo Bởi Stable Diffusion?

- **Deepfake cắt ghép kiểu cũ (Face-Swap)**:
  - Cách làm: Cắt khuôn mặt của người này rồi dán đè lên khuôn mặt của người khác.
  - Dấu vết để lại: Rất dễ bị lộ đường viền cắt ghép quanh má và cằm, màu da bên trong mặt lệch với vùng cổ, hoặc ánh sáng hai bên không khớp nhau.
  - Các mô hình phát hiện cũ chủ yếu chỉ đi tìm những đường viền ghép nối này để kết luận ảnh giả.

- **Ảnh giả tạo bằng Stable Diffusion**:
  - Cách làm: Không cắt dán thủ công mà vẽ lại mượt mà toàn bộ bức ảnh (hoặc vùng mắt/miệng) từ không gian đặc trưng ẩn (Latent Space).
  - Kết quả: Da mặt, ánh sáng và tóc hòa trộn tự nhiên, **không còn đường viền cắt ghép rõ rệt**. Vì vậy, các mô hình phát hiện chỉ học dữ liệu cũ sẽ dễ bị đánh lừa và đoán nhầm ảnh giả này là ảnh thật.

---

## 2. Tại Sao Nhóm Chọn `dinoViT` Làm Mô Hình Phát Hiện Gốc (Baseline)?

- **`dinoViT` (DINO Vision Transformer)** là mô hình thị giác máy tính học theo cơ chế tự giám sát (Self-Supervised Learning):
  1. Thay vì chỉ nhìn vài điểm ảnh lẻ tẻ, `dinoViT` chia khuôn mặt thành lưới các ô vuông nhỏ (patch) và so sánh mối liên hệ giữa tất cả các vùng trên mặt (mắt, mũi, miệng, ánh sáng nền) cùng một lúc.
  2. Nhờ vậy, `dinoViT` tạo ra một vector đại diện tổng thể (`[CLS]` token) rất mạnh, thường được dùng làm mô hình chuẩn (Benchmark) trong các nghiên cứu phát hiện ảnh giả hiện nay.

---

## 3. Tại Sao Phải Tìm Mẫu Khó (`Hard Sample`) Để Huấn Luyện Lại?

Nếu chỉ lấy đại trà các ảnh sinh ra từ Stable Diffusion để dạy cho mô hình thì sẽ có rất nhiều ảnh "dễ" (bản thân mô hình gốc `dinoViT` đã nhận ra được ngay). Những ảnh dễ này không giúp mô hình khôn hơn nhiều.

Ngược lại, theo đúng hướng dẫn của giảng viên:
1. Ta đưa toàn bộ ảnh giả mới qua **Baseline `dinoViT`** để kiểm tra trước.
2. Những bức ảnh nào **bị bắt giữ (`Reject`)** nghĩa là mô hình đã biết rồi.
3. Những bức ảnh nào **vượt qua được `dinoViT` (`Bypass`)** — tức là ảnh giả nhưng lại làm mô hình gốc tưởng là ảnh thật — chính là **Mẫu khó (`Hard Sample`)** chứa đựng điểm mù của mô hình.
4. Khi ta gom riêng các **Mẫu khó** này lại, cho mô hình gặp chúng nhiều hơn và phạt nặng hơn nếu đoán sai trong lúc huấn luyện lại (`Fine-Tune`), ranh giới phân biệt thật/giả của `dinoViT` sẽ dịch chuyển để bao phủ luôn cả vùng điểm mù đó.
