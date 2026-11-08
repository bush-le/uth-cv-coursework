# PRESENTATION_DEFENSE_QA_CHEAT_SHEET.md — Hỏi Đáp Ngắn Gọn Khi Bảo Vệ Đề Tài (Nhóm 4)

- **Motivation/Background**: Chuẩn bị sẵn các câu trả lời cực kỳ ngắn gọn, dễ nhớ và đi thẳng vào trọng tâm khi giảng viên đặt câu hỏi trong buổi báo cáo.
- **Purpose**: Giúp các thành viên **Nhóm 4** tự tin trả lời mọi câu hỏi về Stable Diffusion, `dinoViT`, `Reject/Bypass`, `Hard Sample` và cách chứng minh tác dụng.
- **Overview Pipeline**: Câu hỏi về ý tưởng 3 bước -> Câu hỏi về cách tìm Mẫu khó qua `dinoViT` -> Câu hỏi về cách huấn luyện lại và chống trùng người.
- **Detailed Plan**: 6 câu hỏi thường gặp nhất và câu trả lời ngắn gọn dễ nói.
- **References**: [ PRESENTATION_5MIN_DEFINE_AND_DESIGN.md ](PRESENTATION_5MIN_DEFINE_AND_DESIGN.md), [ SYSTEM_DEFINE_AND_DESIGN_REPORT.md ](SYSTEM_DEFINE_AND_DESIGN_REPORT.md).
- **Created**: 2026-11-02T07:30:00+07:00
- **Last Updated**: 2026-11-08T18:40:00+07:00

---

## 1. Bộ 6 Câu Hỏi & Câu Trả Lời Ngắn Gọn Nhất

### Câu 1: Em hãy tóm tắt đề tài của nhóm trong 3 câu ngắn gọn nhất?
> **Trả lời**:
> Dạ thưa thầy, đề tài của **Nhóm 4** làm theo đúng 3 bước khép kín ạ:
> 1. **Bước 1**: Dùng ảnh mặt thật và **Stable Diffusion** để tạo ra ảnh khuôn mặt giả (`Synthetic DF img`) rồi lọc bỏ ảnh lỗi.
> 2. **Bước 2**: Đưa các ảnh giả đó qua mô hình chuẩn **Baseline `dinoViT`**; ảnh nào đánh lừa được `dinoViT` (**Bypass**) thì nhóm thu lại làm **Mẫu khó (`Hard Sample`)**.
> 3. **Bước 3**: Gộp chung **Dữ liệu gốc + Ảnh sinh mới + Mẫu khó** để huấn luyện lại (`Fine-Tune`) `dinoViT`; khi mô hình mới bắt lại được các mẫu khó đó thì chứng minh được dữ liệu nhóm tạo ra thực sự có tác dụng ạ.

---

### Câu 2: Thế nào là mẫu `Reject` và thế nào là mẫu `Bypass` (`Hard Sample`)?
> **Trả lời**:
> Dạ, tất cả ảnh đưa vào kiểm tra ở Bước 2 đều là **ảnh giả** do Stable Diffusion tạo ra:
> - Nếu mô hình gốc `dinoViT` chấm điểm nghi ngờ giả $\ge 0.5$, tức là nó **phát hiện được đây là ảnh giả** $\rightarrow$ gọi là **Reject** (bị bắt giữ), nhóm xếp vào ảnh tổng hợp thông thường.
> - Nếu `dinoViT` chấm điểm $< 0.5$, tức là nó **bị lừa và tưởng nhầm là ảnh thật** $\rightarrow$ gọi là **Bypass** (vượt rào). Những bức ảnh vượt rào này chính là điểm yếu của mô hình gốc nên được lấy làm **Mẫu khó (`Hard Sample`)** ạ.

---

### Câu 3: Loại ảnh giả nào của Stable Diffusion dễ đánh lừa `dinoViT` gốc nhất?
> **Trả lời**:
> Dạ, đó là những bức ảnh tạo bằng chế độ **Sửa vùng cục bộ (`Facial Inpainting`)** hoặc **Biến đổi nhẹ từ ảnh gốc (`Img2Img`)**. Vì phần lớn khuôn mặt, mái tóc và nền xung quanh vẫn là của ảnh thật, chỉ có vùng mắt hoặc khuôn miệng được vẽ lại mượt mà nên mô hình `dinoViT` gốc nhìn tổng thể rất dễ bị đánh lừa ạ.

---

### Câu 4: Khi huấn luyện lại (`Fine-Tune`) ở Bước 3, nhóm làm gì để `dinoViT` bắt được các Mẫu khó?
> **Trả lời**:
> Dạ, bên cạnh việc gộp đủ 3 nguồn (`Dữ liệu gốc + Ảnh sinh mới + Mẫu khó`), nhóm áp dụng 3 cách đơn giản:
> 1. **Mở khóa các tầng cuối của `dinoViT`** để mô hình học thêm chi tiết vi mô mới mà không quên kiến thức cũ.
> 2. **Nhân bản Mẫu khó (`Oversampling`)** để mô hình được gặp và học kỹ hơn các trường hợp từng bị lừa.
> 3. **Tăng điểm phạt (`Weighted Loss`)** cao gấp đôi nếu mô hình vẫn đoán sai các Mẫu khó ạ.

---

### Câu 5: Làm sao biết mô hình thực sự giỏi lên chứ không phải do nhớ mặt người ở tập học?
> **Trả lời**:
> Dạ, nhóm gắn mã số người thật gốc (`source_real_id`) cho mọi bức ảnh. Tất cả ảnh thật, ảnh giả thường và mẫu khó được tạo ra từ cùng một người đều bị khóa chung vào một tập. Tập Kiểm thử (`Held-Out Test`) hoàn toàn là những khuôn mặt người mới mà mô hình chưa từng nhìn thấy lúc học ạ.

---

### Câu 6: Nhóm dựa vào con số nào để chứng minh dữ liệu có tác dụng?
> **Trả lời**:
> Dạ, nhóm so sánh 3 chế độ huấn luyện `D0` (chỉ dữ liệu gốc), `D1` (thêm ảnh sinh mới thường) và `D2` (thêm cả Mẫu khó). Khi chuyển từ `D0` lên `D2`:
> - Tỷ lệ bắt được mẫu khó (**Hard-Sample Recall**) tăng vọt,
> - Tỷ lệ bị lừa (**Bypass Rate**) giảm mạnh,
> - Và điểm **ROC-AUC** trên tập kiểm thử đạt mức cao nhất. Đó là bằng chứng trực tiếp chứng minh bộ dữ liệu có tác dụng ạ.
