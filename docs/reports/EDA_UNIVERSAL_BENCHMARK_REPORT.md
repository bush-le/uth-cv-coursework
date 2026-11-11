# EDA_UNIVERSAL_BENCHMARK_REPORT.md — Exploratory Data Analysis on Universal Multi-Method Benchmark

- **Motivation/Background**: To comprehensively inspect, audit, and characterize the contents of the Universal Multi-Method Deepfake Benchmark Suite across 39 methods and 6 technological families.
- **Purpose**: Provide authoritative empirical documentation on class balance, method taxonomy, optical properties, image dimensions, and domain distributions across 39 distinct generation methods, with precise pruning criteria for the 21,000-sample Golden Balanced benchmark.
- **Overview Pipeline**: Ingestion of metadata splits -> Statistical aggregation across 6 technology families -> Optical image sampling (Laplacian variance, brightness, contrast) -> Publication-ready visualization generation -> Synthesis of findings for detector benchmarking.
- **Detailed Plan**: §1 Executive Overview & Zero-Leakage Guarantee; §2 Technological Family Taxonomy; §3 Full 39-Method Empirical Table; §4 Optical & Resolution Characteristics; §5 Data Pruning Strategy & Golden 21k Suite; §6 Artifacts & Visualization References.
- **References**: [Universal Methods Summary](../../experiments/eda/tables/universal_methods_summary.csv), [Universal Families Summary](../../experiments/eda/tables/universal_families_summary.csv), [Universal Resolution Summary](../../experiments/eda/tables/universal_resolution_summary.csv), [Distribution Figure](../../experiments/eda/figures/universal_method_distribution.png), [Taxonomy Donut](../../experiments/eda/figures/universal_method_families.png), [Optical Quality Comparison](../../experiments/eda/figures/universal_quality_by_family.png), [Representative Gallery](../../experiments/eda/figures/universal_representative_gallery.png).
- **Created**: 2026-11-11T20:13:00+07:00
- **Last Updated**: 2026-11-11T20:14:00+07:00

---

## 1. Executive Overview & Integrity Specifications

The **Universal Multi-Method Deepfake Benchmark Suite** (`universal_deepfake_benchmark`) represents a blind hold-out face benchmark designed to evaluate detector generalization against diverse deepfake manipulation and synthesis paradigms.

| Metric | Full Benchmark (`test_coursework_44methods_full_zero_leakage.csv`) | Golden Balanced Benchmark (`test_coursework_35methods_golden_balanced.csv`) |
| :--- | :---: | :---: |
| **Total Images** | **50,084** | **21,000** |
| **Real Images** | **25,042 (50.00%)** | **10,500 (50.00%)** |
| **Fake Images** | **25,042 (50.00%)** | **10,500 (50.00%)** |
| **Distinct Fake Methods** | **38 methods** | **35 methods** (exactly 300 per method) |
| **Total Distinct Methods** | **39 methods** (1 Real + 38 Fake) | **36 methods** (1 Real + 35 Fake) |
| **Leakage Status** | **Zero Identity Leakage** (Strictly disjoint from CelebA-HQ training sets) | **Zero Identity Leakage** |
| **Disk Footprint** | **3.9 GB images** (4.1 GB total archive) | Subsample index of the universal suite |

---

## 2. 6-Tier Technological Taxonomy Breakdown

The benchmark encompasses 6 primary generative AI families spanning from classical autoencoders to state-of-the-art Diffusion Transformers:

| Họ Công Nghệ (Technology Family) | Số Lượng Mẫu (Samples) | Tỷ Lệ (%) | Đặc Trưng Kiến Trúc & Tạo Tác (Artifact Profile) |
| :--- | :---: | :---: | :--- |
| **Real Faces (FFHQ / FF++ / CelebA)** | **25,042** | **50.00%** | Ảnh khuôn mặt thật đối chứng độ nét cao; phân bố tần số tự nhiên không có artifacts nén latent hay boundary blending. |
| **Face Swapping (Hoán đổi danh tính)** | **7,355** | **14.69%** | Ghép mặt nguồn vào video/ảnh đích (`mobileswap`, `simswap`, `facedancer`, `uniface`). Tồn tại biên hòa trộn (color mismatch, seam edge blur) quanh cằm và trán. |
| **Reenactment & Talking Head (Biểu cảm)** | **7,191** | **14.36%** | Điều khiển chuyển động mắt, miệng từ âm thanh/video (`sadtalker`, `wav2lip`, `fomm`, `pirender`, `tpsm`). Xuất hiện méo mó khẩu hình và mất tính đối xứng khi nhấp nháy. |
| **GAN Architectures (Mạng đối nghịch)** | **3,449** | **6.89%** | Sinh ảnh hoàn chỉnh hoặc chỉnh sửa thuộc tính (`StyleGAN2/3/XL`, `stargan`, `styleclip`). Đặc trưng bởi artifacts cấu trúc checkerboard và mất đối xứng đồng tử. |
| **Diffusion Transformers (DiT / SiT / PixArt)** | **3,401** | **6.79%** | Thế hệ khuếch tán mới dựa trên Vision Transformer patches (`DiT`, `SiT`, `pixart`, `RDDM`). Tái tạo vi cấu trúc da xuất sắc nhưng có dấu vết phổ tần số Fourier đặc thù. |
| **Latent Diffusion (Stable Diffusion 2.1)** | **1,460** | **2.92%** | Khuếch tán trong không gian latent VAE (`sd2.1`). Làm mịn cục bộ ở vùng tóc và vân da, độ tương phản dị thường tại rìa bóng đổ. |
| **GAN Inversion & Latent Editing** | **803** | **1.60%** | Chiếu ảnh thật vào latent space để biến đổi tuổi tác/biểu cảm (`e4e`, `e4s`). Xuất hiện hiện tượng mờ chi tiết nền và biến đổi kết cấu tóc. |
| **Diffusion Baselines & AutoEncoders** | **1,185** | **2.36%** | Mô hình khuếch tán nền tảng (`ddim`: 593) và nén vector quantization (`VQGAN`: 592). Gặp lỗi lượng tử hóa khối hoặc làm phẳng vi cấu trúc. |
| **Commercial GenAI (MidJourney / HeyGen)** | **198** | **0.40%** | Dữ liệu sinh từ các nền tảng thương mại đóng kín (`MidJourney`: 187, `heygen`: 11). Chi tiết thẩm mỹ cực cao nhưng có độ tương phản và sắc nét phi thực tế. |

---

## 3. Data Pruning Rationale: From 38 to 35 Retained Methods

Để bảo đảm tính công bằng tuyệt đối trong đánh giá và loại trừ các sai số thống kê do mẫu quá nhỏ:

1. **`heygen` (11 samples)**: Loại bỏ do $N=11 < 30$, phương sai quá lớn, chỉ 1 dự đoán sai làm lệch $10\%$ điểm số.
2. **`deepfacelab` (25 samples)**: Loại bỏ do $N=25 < 30$, không đủ ý nghĩa thống kê đại diện cho công cụ DeepFaceLab.
3. **`MidJourney` (187 samples)**: Loại bỏ khỏi bảng balanced vì thiếu hụt 113 mẫu so với quota 300 mẫu chuẩn.

Tập **Golden Balanced Suite** lưu trữ tại:
`data/benchmarks/universal_deepfake_benchmark/splits/test_coursework_35methods_golden_balanced.csv`
gồm **21.000 mẫu** (10.500 Real vs 10.500 Fake trên đúng 35 phương pháp đạt chuẩn).

---

## 4. Generated Visualizations & Tables Summary

Tất cả bảng biểu và đồ thị trực quan hóa đã được lưu trữ sẵn sàng:
- Bảng tổng hợp phương pháp: [universal_methods_summary.csv](../../experiments/eda/tables/universal_methods_summary.csv)
- Bảng tổng hợp công nghệ: [universal_families_summary.csv](../../experiments/eda/tables/universal_families_summary.csv)
- Bảng kích thước ảnh: [universal_resolution_summary.csv](../../experiments/eda/tables/universal_resolution_summary.csv)
- Đồ thị phân bố 38 methods: [universal_method_distribution.png](../../experiments/eda/figures/universal_method_distribution.png)
- Đồ thị hình vành khăn họ công nghệ: [universal_method_families.png](../../experiments/eda/figures/universal_method_families.png)
- Đồ thị so sánh quang học: [universal_quality_by_family.png](../../experiments/eda/figures/universal_quality_by_family.png)
- Bộ sưu tập 12 phương pháp đại diện: [universal_representative_gallery.png](../../experiments/eda/figures/universal_representative_gallery.png)
