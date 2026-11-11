# Hard Samples Vault (`data/hard_samples/`)

## 1. Tổng Quan Kho Mẫu Khó Đối Kháng (Overview)
Thư mục này là **Kho Lưu Trữ Thống Nhất (Centralized Vault)** chứa toàn bộ các mẫu đối kháng cực khó (Hard Adversarial Samples) được khai thác tự động qua 2 vòng Closed-Loop Active Learning của đề tài.

- **Tổng số mẫu khó được phát hiện**: **740 mẫu**
- **Tập tin chỉ mục thống nhất**: [`hard_samples_manifest.jsonl`](hard_samples_manifest.jsonl)
- **Kho ảnh thực tế**: [`images/`](images/) (được phân loại theo phương thái sinh ảnh: `sd_inpaint`, `sd_img2img`, `sd_text2img`)
- **Báo cáo kiểm toán nhân khẩu học & độ khó**: [`audit_reports/hardness_report.json`](audit_reports/hardness_report.json)

---

## 2. Nguồn Gốc & Cơ Chế Phân Bổ (Provenance & Allocation)

Toàn bộ 740 mẫu khó bắt nguồn từ quá trình dò quét thực tế trên các checkpoint thần kinh:
1. **Vòng 1 (`data/active_learning/round_1/`)**: 553 mẫu khó đánh lừa mô hình tiền nghiệm $M_0$.
2. **Vòng 2 (`data/active_learning/round_2/`)**: 187 mẫu khó đánh lừa mô hình $M_1$.

### Cơ Chế Phân Bổ (85% Huấn Luyện / 15% Kiểm Định Độc Lập):
- **85% Bộ Đệm Tái Huấn Luyện (629 mẫu)**: Được sáp nhập vào tập huấn luyện của đợt Vòng 2 ([`data/benchmarks/D2_hard_boosted/train.jsonl`](../benchmarks/D2_hard_boosted/train.jsonl)) nhằm gia tăng độ dẻo dai đối kháng cho mô hình $M_2^*$.
- **15% Kho Kiểm Thử Thách Thức (111 mẫu)**: Được cách ly nghiêm ngặt và hợp nhất vào kho kiểm định mù đối kháng tích lũy ([`data/active_learning/cumulative_challenge_test.jsonl`](../active_learning/cumulative_challenge_test.jsonl), tổng cộng 213 mẫu) để kiểm tra độc lập $M_0, M_1, M_2^*$.

---

## 3. Phân Rã Theo Phương Thái Sinh Ảnh (Manipulation Modalities)

| Phương Thái | Số Lượng Mẫu Khó | Tỷ Lệ | Nguyên Nhân Vượt Mặt Mô Hình |
| :--- | :---: | :---: | :--- |
| **`sd_inpaint` (Chỉnh sửa cục bộ)** | 578 | 78.1% | Ranh giới ghép quanh mắt/môi siêu mỏng, 85-90% bối cảnh là ảnh thật nguyên bản FFHQ. |
| **`sd_img2img` (Biến đổi diện mạo)** | 136 | 18.4% | Giữ nguyên hình học giải phẫu gốc, chỉ can thiệp tinh vi vào bề mặt da và ánh sáng. |
| **`sd_text2img` (Tạo ảnh thuần prompt)** | 26 | 3.5% | Các ca khuôn mặt cân đối hoàn hảo, triệt tiêu tối đa các tạo tác méo mó thông thường. |
