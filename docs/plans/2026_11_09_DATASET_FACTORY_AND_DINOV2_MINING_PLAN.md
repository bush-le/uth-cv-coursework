# 2026_11_09_DATASET_FACTORY_AND_DINOV2_MINING_PLAN.md — Kế Hoạch Xây Dựng Nhà Máy Dữ Liệu Stable Diffusion & Khai Thác Mẫu Khó Bằng Meta DINOv2

- **Motivation/Background**: Theo định hướng thống nhất ở Hướng A, dự án tập trung toàn lực vào mục tiêu cốt lõi của Đề tài 11: Tạo lập bộ dữ liệu khuôn mặt Deepfake chân thực bằng Stable Diffusion v1.5 từ 4.000 ảnh gốc FFHQ + CelebA-HQ, áp dụng Curation Gate lọc chất lượng, sử dụng không gian biểu diễn đặc trưng của Meta DINOv2 để bóc tách Mẫu Khó (`Hard Samples` có khả năng đánh lừa thị giác máy tính), và đóng gói bộ dữ liệu hoàn chỉnh chuẩn khoa học. Phần huấn luyện lại detector (Hướng B) sẽ thực hiện sau khi có bộ dữ liệu hoàn chỉnh.
- **Purpose**: Là bản kế hoạch kỹ thuật chi tiết từng bước (task-by-task implementation plan) để triển khai, kiểm thử và bàn giao sản phẩm Dataset hoàn chỉnh cho Nhóm 4.
- **Overview Pipeline**: Task 1 (Cải tổ pipeline đào Mẫu Khó hỗ trợ DINOv2 Feature Similarity thuần) -> Task 2 (Tối ưu hóa generator sinh hàng loạt & hỗ trợ resume) -> Task 3 (Sinh tập dữ liệu SD đa chế độ & Lọc Curation Gate) -> Task 4 (Bóc tách Mẫu Khó thật sự bằng DINOv2) -> Task 5 (Đóng gói Dataset 5 tầng, Cập nhật Visual Gallery & 3 Notebooks) -> Task 6 (Kiểm thử toàn diện & Git Commit).
- **Detailed Plan**: §1 Thông tin chung & Ràng buộc toàn cục; §2 Tiêu điểm thẩm định (Review Focus); §3 Danh sách Task chi tiết từ 1 đến 6.
- **References**: [../PURPOSE.md](../PURPOSE.md), [../OVERVIEW.md](../OVERVIEW.md), [../phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md](../phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md), [../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md](../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md), [2026_11_09_BASELINE_AUDIT_AND_CORRECTION_PLAN.md](2026_11_09_BASELINE_AUDIT_AND_CORRECTION_PLAN.md).
- **Created**: 2026-11-09T13:30:00+07:00
- **Last Updated**: 2026-11-09T13:30:00+07:00

---

# Stable Diffusion Dataset Factory & DINOv2 Feature-Space Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng nhà máy sinh dữ liệu khuôn mặt Deepfake bằng Stable Diffusion v1.5 (cân bằng 3 chế độ `text2img`, `inpainting`, `img2img`), áp dụng phễu lọc chất lượng, dùng không gian vector đặc trưng của Meta DINOv2 để bóc tách Mẫu Khó (ảnh giả có độ tương đồng quá cao với mặt thật), và đóng gói bộ dữ liệu chuẩn 5 tầng bàn giao cho môn học.

**Architecture:** Sử dụng pipeline Hugging Face `diffusers` kết hợp tối ưu bộ nhớ `attention_slicing` để tạo ảnh; `ImageCurationGate` lọc sắc nét Laplacian; `MetaDinoV2Wrapper` trích xuất vector embedding 384 chiều đo khoảng cách Cosine giữa ảnh SD và ảnh thật gốc (`source_real_id`) để xác định Mẫu Khó mà không cần phụ thuộc vào mô hình phân lớp sơ bộ; `scripts/organize_topic_dataset.py` đóng gói bộ dữ liệu kèm nhãn nhân khẩu học.

**Tech Stack:** PyTorch 2.4+, Torchvision, Hugging Face Diffusers, Transformers, Meta DINOv2 (`facebook/dinov2-small`), Pillow, NumPy, PyTest.

**Spec:** [docs/PURPOSE.md](../PURPOSE.md), [docs/phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md](../phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md).

## Global Constraints

- **Zero-Mock Data**: Mọi ảnh giả phải được tạo ra bởi Stable Diffusion thật (`runwayml/stable-diffusion-v1-5`), không dùng mock noise toán học.
- **Không Rò Rỉ Danh Tính (`identity_disjoint=True`)**: Mọi ảnh giả sinh ra từ người thật nào bắt buộc kế thừa `source_real_id` của người đó.
- **Bảo Tồn Nhãn Nhân Khẩu Học**: Mọi ảnh giả sinh ra phải lưu giữ thông tin nhân khẩu học (độ tuổi, giới tính, kính) từ ảnh gốc để phục vụ báo cáo kiểm toán.
- **Kích Thước Chuẩn**: Ảnh lưu trữ và xuất bản có độ phân giải đồng bộ $256 \times 256$ RGB trong khoảng $[0.0, 1.0]$.
- **Không Đường Dẫn Tuyệt Đối**: Toàn bộ liên kết và đường dẫn trong code/tài liệu là tương đối (`data/raw/`, `data/synthetic/`).

## Review Focus

1. **Khả năng Resume khi sinh hàng loạt**: Nếu quá trình sinh ảnh bị ngắt quãng giữa chừng, script phải tự động bỏ qua các ảnh đã có trên đĩa, tiếp tục sinh từ ảnh tiếp theo mà không làm hỏng manifest.
2. **Cân đối tỷ lệ 3 chế độ**: Phải phân bổ đều 33% `img2img`, 33% `inpaint`, 33% `text2img`.
3. **Mặt nạ Inpainting mượt mà**: Phải làm mờ viền mặt nạ (Gaussian blur) để vùng inpainting hòa trộn tự nhiên với da mặt gốc.
4. **Ngưỡng đo DINOv2 khách quan**: Khoảng cách Cosine giữa vector ảnh giả và ảnh thật gốc phải được chuẩn hóa trong $[0, 1]$; ảnh có độ tương đồng Cosine $\ge \tau_{\text{sim}}$ (mặc định $0.85$) được ghi nhận là Mẫu Khó.
5. **Tính toàn vẹn của tệp ảnh**: Không lưu manifest cho các ảnh bị lỗi ghi đĩa hoặc kích thước 0 byte.

---

### Task 1: Nâng Cấp `mine_hard_samples.py` Hỗ Trợ DINOv2 Feature Similarity

**Files:**
- Modify: `src/pipelines/mine_hard_samples.py:70-195`
- Test: `tests/test_pipelines.py`

**Interfaces:**
- Produces: `mine_hard_samples_by_feature_similarity(extractor, synthetic_metadata, real_manifest, threshold=0.85) -> HardSampleMiningReport`

- [ ] **Step 1: Viết test cho phương pháp khai thác mẫu khó bằng DINOv2 Feature Cosine Similarity**

```python
# tests/test_pipelines.py
def test_mine_hard_samples_by_feature_similarity():
    # Verify that synthetic fakes with high cosine similarity to their real counterpart are flagged as hard samples
    pass
```

- [ ] **Step 2: Chạy test xác nhận fail (RED)**  
Run: `pytest tests/test_pipelines.py -k "feature_similarity" -v`  
Expected: FAIL

- [ ] **Step 3: Triển khai hàm `mine_hard_samples_by_feature_similarity` trong `src/pipelines/mine_hard_samples.py`**
Sử dụng `MetaDinoV2Wrapper` trích xuất embedding chuẩn hóa $\vec{v}_{\text{syn}}$ và $\vec{v}_{\text{real}}$. Tính tích vô hướng $\cos(\vec{v}_{\text{syn}}, \vec{v}_{\text{real}}) = \vec{v}_{\text{syn}} \cdot \vec{v}_{\text{real}}$. Nếu $\cos \ge \text{threshold}$ (mặc định $0.85$), ảnh giả có biểu diễn không gian tiềm ẩn quá giống ảnh thật $\to$ gán `is_hard_sample = True`.

- [ ] **Step 4: Chạy test xác nhận pass (GREEN)**  
Run: `pytest tests/test_pipelines.py -k "feature_similarity" -v`  
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/pipelines/mine_hard_samples.py tests/test_pipelines.py
git commit -m "feat: add DINOv2 feature similarity cosine mining for objective hard sample discovery"
```

---

### Task 2: Tối Ưu Hóa Generator Hàng Loạt & Hỗ Trợ Resume

**Files:**
- Modify: `src/pipelines/generate_deepfake_images.py:100-240`
- Test: `tests/test_pipelines.py`

**Interfaces:**
- CLI Flags: `--num-samples N`, `--resume`, `--batch-size 1`, `--modes all|img2img|inpaint|text2img`

- [ ] **Step 1: Viết test cho cờ resume và phân bổ 3 chế độ cân bằng**

```python
# tests/test_pipelines.py
def test_generate_pipeline_balanced_mode_distribution():
    # Ensure alternating (i % 3) generates equal distribution of img2img, inpaint, text2img
    pass
```

- [ ] **Step 2: Chạy test xác nhận fail (RED)**  
Run: `pytest tests/test_pipelines.py -k "balanced_mode" -v`  
Expected: FAIL

- [ ] **Step 3: Triển khai hỗ trợ `--resume` và cân đối phân bổ trong `src/pipelines/generate_deepfake_images.py`**
Nếu tệp ảnh đích đã tồn tại và hợp lệ trên đĩa, tự động nạp lại bản ghi vào manifest và bỏ qua bước tính toán khuếch tán để tiết kiệm thời gian.

- [ ] **Step 4: Chạy test xác nhận pass (GREEN)**  
Run: `pytest tests/test_pipelines.py -k "balanced_mode" -v`  
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/pipelines/generate_deepfake_images.py tests/test_pipelines.py
git commit -m "feat: enhance deepfake generator with resume support and strictly balanced mode allocation"
```

---

### Task 3: Chạy Sinh Bộ Dữ Liệu Deepfake Chân Thực Quy Mô & Curation Gate

**Files:**
- Execute: `src/pipelines/generate_deepfake_images.py`
- Execute: `src/pipelines/curate_image_dataset.py`
- Output: `data/synthetic/img2img/`, `data/synthetic/inpainting/`, `data/synthetic/text2image/`, `data/synthetic/fake_manifest.jsonl`, `data/synthetic/curated_manifest.jsonl`

- [ ] **Step 1: Chạy sinh 30 bức ảnh khuôn mặt giả bằng Stable Diffusion v1.5**
Run: `python3 -m src.pipelines.generate_deepfake_images --model runwayml/stable-diffusion-v1-5 --max-samples 30 --output-dir data/synthetic --output-manifest data/synthetic/fake_manifest.jsonl`  
Expected: 30 ảnh phân bổ đều 10 `img2img`, 10 `inpaint`, 10 `text2img` lưu tại `data/synthetic/`.

- [ ] **Step 2: Chạy phễu lọc Curation Gate để sàng lọc chất lượng**
Run: `python3 -m src.pipelines.curate_image_dataset --input-manifest data/synthetic/fake_manifest.jsonl --output-manifest data/synthetic/curated_manifest.jsonl --min-sharpness 40.0`  
Expected: `curated_manifest.jsonl` ghi nhận các ảnh đạt độ sắc nét và cấu trúc giải phẫu.

- [ ] **Step 3: Xác minh tỷ lệ đạt chuẩn của Curation Gate**
Đảm bảo tỷ lệ giữ lại $\ge 80\%$.

- [ ] **Step 4: Commit**
```bash
git add data/synthetic/.gitkeep
git commit -m "feat: generate 30 high-fidelity multimodal synthetic facial images and curate via Curation Gate"
```

---

### Task 4: Khai Thác Mẫu Khó Khách Quan Bằng DINOv2 Feature Cosine Similarity

**Files:**
- Execute: `src/pipelines/mine_hard_samples.py`
- Output: `data/hard_samples/images/*.png`, `data/hard_samples/hard_samples_manifest.jsonl`, `data/hard_samples/audit_reports/hard_sample_demographics.json`

- [ ] **Step 1: Chạy bóc tách mẫu khó bằng độ tương đồng đặc trưng DINOv2**
Run: `python3 -m src.pipelines.mine_hard_samples --method feature_similarity --threshold 0.85 --input-manifest data/synthetic/curated_manifest.jsonl --raw-manifest data/raw/manifest.jsonl --hard-manifest data/hard_samples/hard_samples_manifest.jsonl`  
Expected: Tách ra khoảng 15-30% ảnh giả có đặc trưng gần với ảnh thật nhất làm Mẫu Khó (`Hard Samples`).

- [ ] **Step 2: Xác minh tệp ảnh và báo cáo kiểm toán**
Kiểm tra `data/hard_samples/images/` chứa đúng các ảnh mẫu khó và báo cáo `hard_sample_demographics.json` thống kê đầy đủ nhóm tuổi/giới tính.

- [ ] **Step 3: Commit**
```bash
git add data/hard_samples/.gitkeep
git commit -m "feat: extract legitimate hard samples via DINOv2 latent feature similarity audit"
```

---

### Task 4.5: Đóng Gói Bộ Dữ Liệu 5 Tầng & Đồng Bộ Benchmark Splits

**Files:**
- Modify: `scripts/organize_topic_dataset.py`
- Output: `data/benchmarks/D0_baseline/`, `data/benchmarks/D1_augmented/`, `data/benchmarks/D2_hard_boosted/`

- [ ] **Step 1: Chạy script tổ chức và đóng gói dataset**
Run: `python3 -m scripts.organize_topic_dataset`  
Expected: Cấu trúc 5 tầng sạch sẽ, các file `train.jsonl`, `val.jsonl`, `test.jsonl` trong `data/benchmarks/` đều có đủ nhãn 0 và 1 với `identity_disjoint=True`.

- [ ] **Step 2: Commit**
```bash
git add scripts/organize_topic_dataset.py
git commit -m "feat: organize standardized 5-tier dataset and export stratified benchmark splits"
```

---

### Task 5: Cập Nhật Visual Comparison Gallery & 3 Coursework Notebooks

**Files:**
- Execute: `scripts/build_visual_gallery.py`
- Execute: `scripts/create_coursework_notebooks.py`
- Output: `experiments/predictions/real_vs_synthetic_gallery.png`, `notebooks/coursework/01_eda_faces_demographics.ipynb`, `notebooks/coursework/02_sd_generation_and_curation.ipynb`, `notebooks/coursework/03_hard_samples_and_benchmark_comparison.ipynb`

- [ ] **Step 1: Tạo lưới ảnh đối chứng trực quan độ phân giải cao**
Run: `python3 -m scripts.build_visual_gallery`  
Expected: Tạo thành công ảnh gallery sắc nét đối chứng 3 hàng: Real vs. Synthetic SD vs. DINOv2 Hard Samples.

- [ ] **Step 2: Cập nhật 3 Coursework Notebooks**
Run: `python3 scripts/create_coursework_notebooks.py`  
Expected: 3 Notebooks được cập nhật với các đường dẫn dữ liệu mới và mã hiển thị gallery trực quan.

- [ ] **Step 3: Commit**
```bash
git add experiments/predictions/real_vs_synthetic_gallery.png notebooks/coursework/*.ipynb
git commit -m "feat: update visual gallery and coursework inspection notebooks with real generated dataset"
```

---

### Task 6: Kiểm Định Toàn Diện Test Suite & Đẩy Lên Git Remote

**Files:**
- Execute: `make test-smoke`, `pytest tests/ -v`, `python3 -m py_compile`

- [ ] **Step 1: Chạy smoke test nhanh**
Run: `make test-smoke`  
Expected: 10/10 smoke tests PASS trong < 5 giây.

- [ ] **Step 2: Chạy toàn bộ test suite**
Run: `pytest tests/ -v`  
Expected: 100% tests PASS.

- [ ] **Step 3: Kiểm tra biên dịch cú pháp Python**
Run: `python3 -m py_compile $(find src tests scripts -name "*.py")`  
Expected: Exit code 0.

- [ ] **Step 4: Commit & Đẩy lên remote**
```bash
git push origin baseline/initial-baseline
```
