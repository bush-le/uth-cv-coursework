# 2026_11_09_REMOVE_ALL_MOCKS_REAL_SD_DINOVIT.md — Kế Hoạch Loại Bỏ Toàn Bộ Mock & Triển Khai Mô Hình Thật 100%

- **Motivation/Background**: Thay thế hoàn toàn cơ chế tạo ảnh giả lập toán học (`mock_sd` hình oval + noise) và mô hình `dinoViT` trọng số ngẫu nhiên bằng hệ thống AI thật (Stable Diffusion v1.5 / Realistic Vision từ Hugging Face và DINOv2 từ Meta AI) để tạo ra bộ dữ liệu khuôn mặt giả chân thực (photorealistic), chất lượng cao, phục vụ chuẩn mực nghiên cứu khoa học.
- **Purpose**: Là bản kế hoạch kỹ thuật chi tiết từng bước (task-by-task implementation plan) để loại bỏ 100% mock, nạp trọng số nơ-ron thật, và chạy trọn vẹn chu trình 3 bước (Sinh dữ liệu -> Lọc mẫu khó -> Huấn luyện đối chứng D0/D1/D2).
- **Overview Pipeline**: Task 1 (Cài đặt & kiểm tra runtime Diffusers/Transformers/CUDA) -> Task 2 (Refactor StableDiffusionEngine sang pipeline Diffusers thật) -> Task 3 (Nạp trọng số Pretrained Meta DINOv2) -> Task 4 (Sinh tập ảnh giả chân thực Phase 1) -> Task 5 (Chấm điểm & Khai thác Mẫu khó Phase 2) -> Task 6 (Huấn luyện lại D0 vs D1 vs D2 Phase 3) -> Task 7 (Kiểm định toàn diện & Xuất Gallery so sánh trực quan).
- **Detailed Plan**: §1 Thông tin chung & Ràng buộc toàn cục; §2 Tiêu điểm thẩm định (Review Focus); §3 Danh sách Task chi tiết từ 1 đến 7.
- **References**: [../PURPOSE.md](../PURPOSE.md), [../phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md](../phases/PHASE_01_DIFFUSION_SYNTHESIS_SPEC.md), [../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md](../phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md), [../phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md](../phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md).
- **Created**: 2026-11-09T11:40:00+07:00
- **Last Updated**: 2026-11-09T11:45:00+07:00

---

# Remove All Mocks & Deploy Real Stable Diffusion + Meta DINOv2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chuyển đổi toàn bộ hệ thống từ mô phỏng toán học (`mock_sd` oval noise) sang bộ sinh ảnh Deepfake chân thực 100% bằng mạng Stable Diffusion thật (`runwayml/stable-diffusion-v1-5` / `Realistic_Vision`) kết hợp mô hình trích xuất đặc trưng `Meta DINOv2` chính thức, tạo ra bộ dữ liệu khuôn mặt giả chất lượng cao đạt chuẩn công bố khoa học.

**Architecture:** Sử dụng kiến trúc Pipeline Hugging Face `diffusers` (`StableDiffusionPipeline`, `StableDiffusionInpaintPipeline`, `StableDiffusionImg2ImgPipeline`) với kỹ thuật tối ưu hóa bộ nhớ `enable_attention_slicing()` và `float16` trên GPU RTX 3050. Thay thế backbone Transformer ngẫu nhiên bằng `facebook/dinov2-small` pre-trained của Meta để chấm điểm phát hiện và khai thác Mẫu khó (Hard Samples).

**Tech Stack:** PyTorch, Torchvision, Diffusers (v0.42+), Transformers (v5.22+), Accelerate, Safetensors, Pillow, NumPy, PyTest.

**Spec:** [../PURPOSE.md](../PURPOSE.md)

## Global Constraints

- **Zero Mock trong môi trường sản xuất**: Loại bỏ hoàn toàn cờ mặc định `mock_sd`; mọi lệnh chạy pipeline bắt buộc sử dụng checkpoint thật. Mock chỉ được giữ làm chế độ kiểm thử nhanh `--mock` cho CI unit-test khi không có kết nối mạng.
- **Tuân thủ hợp đồng Tensor**: Tensor ảnh luôn là `(3, H, W)` float32 trong $[0.0, 1.0]$ RGB. Kích thước chuẩn đưa vào detector là $256 \times 256$.
- **Giữ sạch VRAM GPU**: Dùng `torch.cuda.amp.autocast()` và `pipe.enable_attention_slicing()` để chạy an toàn trong ngưỡng 4GB VRAM của card RTX 3050 Laptop, không để xảy ra Out-Of-Memory (OOM).
- **Không rò rỉ danh tính (Identity-Disjoint)**: Ảnh giả sinh ra từ người thật nào bắt buộc mang cùng `source_real_id` của người đó để tập Train và Test độc lập tuyệt đối.
- **Không đường dẫn tuyệt đối**: Mọi đường dẫn file trong code và tài liệu đều là tương đối (`data/raw/`, `configs/`).

## Review Focus

1. **VRAM Exhaustion (OOM) khi nạp Stable Diffusion**: Pipeline phải tự động bật `attention_slicing` và giải phóng cache `torch.cuda.empty_cache()` sau mỗi batch.
2. **Khuôn mặt dị dạng khi Inpainting**: Vùng mặt nạ inpainting mắt/miệng phải được xử lý làm mờ viền (Gaussian blur mask) để vết ghép tự nhiên, không bị vệt cắt cứng.
3. **Mất nhãn nhân khẩu học khi sinh ảnh**: Ảnh giả sinh ra từ người già/trẻ em/kính râm phải kế thừa chính xác metadata nhân khẩu học từ ảnh gốc.
4. **Download timeout từ Hugging Face Hub**: Tự động hỗ trợ cache cục bộ tại `~/.cache/huggingface/hub` và retry nếu kết nối mạng gián đoạn.
5. **Độ phân giải không đồng bộ**: Ảnh sinh ra $512 \times 512$ phải được resize tự động về $256 \times 256$ trước khi lưu manifest và nạp vào dinoViT.

---

### Task 1: Thiết Lập Runtime AI Thật & Tối Ưu Hóa Bộ Nhớ GPU

**Files:**
- Modify: `src/common/device.py` (hoặc tạo mới nếu chưa có)
- Modify: `configs/config.yaml:12-25`
- Test: `tests/test_runtime_device.py`

**Interfaces:**
- Produces: `get_optimal_device() -> torch.device`, `get_torch_dtype() -> torch.dtype`, `optimize_pipeline_memory(pipe: Any) -> None`

- [ ] **Step 1: Viết test kiểm tra thiết bị tính toán và bộ nhớ**

```python
# tests/test_runtime_device.py
import torch
from src.common.device import get_optimal_device, get_torch_dtype

def test_device_detection():
    device = get_optimal_device()
    assert isinstance(device, torch.device)
    dtype = get_torch_dtype(device)
    assert dtype in (torch.float16, torch.float32)
```

- [ ] **Step 2: Chạy test để xác nhận fail**
Run: `pytest tests/test_runtime_device.py -v`  
Expected: FAIL (ModuleNotFoundError: No module named 'src.common.device')

- [ ] **Step 3: Cài đặt `src/common/device.py` hỗ trợ quản lý bộ nhớ GPU/CPU linh hoạt**
Viết hàm phát hiện GPU, tự động chuyển `float16` khi có CUDA, bật `attention_slicing` cho mô hình khuếch tán.

- [ ] **Step 4: Chạy test để xác nhận pass**
Run: `pytest tests/test_runtime_device.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/common/device.py tests/test_runtime_device.py configs/config.yaml
git commit -m "feat: add optimal device and memory management for diffusion runtime"
```

---

### Task 2: Cải Tổ StableDiffusionEngine Sang Mạng Nơ-ron Thật 100%

**Files:**
- Modify: `src/models/diffusion/stable_diffusion_engine.py`
- Test: `tests/test_stable_diffusion.py`

**Interfaces:**
- Consumes: `diffusers.StableDiffusionPipeline`, `diffusers.StableDiffusionInpaintPipeline`, `diffusers.StableDiffusionImg2ImgPipeline`
- Produces: `StableDiffusionEngine.generate_text2img(prompt, ...) -> torch.Tensor`, `generate_inpaint(source_image, mask, prompt, ...) -> torch.Tensor`, `generate_img2img(source_image, prompt, strength, ...) -> torch.Tensor`

- [ ] **Step 1: Viết test cho engine Diffusers thật (hỗ trợ cả mode mock khi test nhanh)**

```python
# tests/test_stable_diffusion.py
def test_real_engine_instantiation():
    engine = StableDiffusionEngine(model_name="runwayml/stable-diffusion-v1-5", use_mock_fallback=False)
    assert engine.model_name == "runwayml/stable-diffusion-v1-5"
```

- [ ] **Step 2: Chạy test xác nhận fail**
Run: `pytest tests/test_stable_diffusion.py::test_real_engine_instantiation -v`  
Expected: FAIL (argument mismatch or logic error)

- [ ] **Step 3: Cải tổ `src/models/diffusion/stable_diffusion_engine.py`**
Tích hợp trực tiếp 3 pipeline của `diffusers`:
1. `AutoPipelineForText2Image` (hoặc `StableDiffusionPipeline`)
2. `AutoPipelineForInpainting` (hoặc `StableDiffusionInpaintPipeline`)
3. `AutoPipelineForImage2Image` (hoặc `StableDiffusionImg2ImgPipeline`)
Tự động kích hoạt negative prompt chuẩn: `"deformed, blurry, bad anatomy, bad eyes, disfigured, poorly drawn face, mutated, extra limbs"`.

- [ ] **Step 4: Chạy test xác nhận pass**
Run: `pytest tests/test_stable_diffusion.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/models/diffusion/stable_diffusion_engine.py tests/test_stable_diffusion.py
git commit -m "feat: implement real diffusers pipelines for SD text2img, inpaint, and img2img"
```

---

### Task 3: Tích Hợp Trọng Số Meta DINOv2 Pretrained Cho Detector Baseline

**Files:**
- Modify: `src/models/detector/image_detector.py`
- Modify: `configs/detection/detector_baseline.yaml`
- Test: `tests/test_image_detector.py`

**Interfaces:**
- Consumes: `transformers.AutoModel.from_pretrained("facebook/dinov2-small")`
- Produces: `DeepfakeImageDetector(backbone="dinov2_small") -> logits, predict_proba`

- [ ] **Step 1: Viết test nạp backbone DINOv2 từ Meta**

```python
# tests/test_image_detector.py
def test_dinov2_pretrained_backbone():
    detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
    x = torch.rand(2, 3, 256, 256)
    proba = detector.predict_proba(x)
    assert proba.shape == (2, 1)
    assert 0.0 <= proba.min() and proba.max() <= 1.0
```

- [ ] **Step 2: Chạy test xác nhận fail**
Run: `pytest tests/test_image_detector.py::test_dinov2_pretrained_backbone -v`  
Expected: FAIL

- [ ] **Step 3: Triển khai nạp Meta DINOv2 (`facebook/dinov2-small`) trong `src/models/detector/image_detector.py`**
Sử dụng `transformers.AutoModel.from_pretrained('facebook/dinov2-small')` làm extractor 384 chiều, gắn đầu phân loại MLP `(384 -> 128 -> 1)` với Dropout 0.3.

- [ ] **Step 4: Chạy test xác nhận pass**
Run: `pytest tests/test_image_detector.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/models/detector/image_detector.py configs/detection/detector_baseline.yaml tests/test_image_detector.py
git commit -m "feat: integrate official Meta DINOv2 pretrained backbone for deepfake detection"
```

---

### Task 4: Chạy Sinh Ảnh Giả Chân Thực Phase 1 Bằng Mạng Thật

**Files:**
- Modify: `src/pipelines/generate_deepfake_images.py`
- Modify: `scripts/run_pipeline.sh:34`
- Test: `tests/test_pipelines.py`

**Interfaces:**
- Consumes: `data/raw/manifest.jsonl` (ảnh thật đầu vào FFHQ/CelebA-HQ)
- Produces: `data/processed/synthetic_faces/*.png` (ảnh giả chân thực), `data/processed/fake_manifest.jsonl`

- [ ] **Step 1: Đổi mặc định `--model` sang checkpoint thật và hỗ trợ phân luồng batch**
Mặc định model là `"runwayml/stable-diffusion-v1-5"`, độ phân giải $256 \times 256$, số bước suy luận `num_inference_steps=25-30`.

- [ ] **Step 2: Chạy kiểm thử sinh thử nghiệm 5 ảnh thật**
Run: `python3 -m src.pipelines.generate_deepfake_images --model runwayml/stable-diffusion-v1-5 --max-samples 5`  
Expected: 5 ảnh khuôn mặt giả chân thực được lưu tại `data/processed/synthetic_faces/`.

- [ ] **Step 3: Kiểm tra trực quan ảnh sinh ra bằng công cụ đo độ nét**
Đảm bảo độ sắc nét Laplacian $\ge 40.0$, không bị biến dạng hình học.

- [ ] **Step 4: Chạy bộ lọc Curation Gate loại bỏ ảnh mờ**
Run: `bash scripts/run_pipeline.sh curate`  
Expected: `curated_manifest.jsonl` được tạo với tỷ lệ đạt chuẩn $> 80\%$.

- [ ] **Step 5: Commit**
```bash
git add src/pipelines/generate_deepfake_images.py scripts/run_pipeline.sh
git commit -m "feat: deploy real photorealistic image synthesis in Phase 1 generation pipeline"
```

---

### Task 5: Chạy Baseline Meta DINOv2 & Khai Thác Mẫu Khó Phase 2

**Files:**
- Modify: `src/pipelines/mine_hard_samples.py`
- Test: `tests/test_pipelines.py`

**Interfaces:**
- Consumes: `data/processed/curated_manifest.jsonl`, `DeepfakeImageDetector(backbone="dinov2_small", pretrained=True)`
- Produces: `data/processed/hard_samples_manifest.jsonl` (những ảnh giả vượt qua được DINOv2)

- [ ] **Step 1: Viết test cho quy trình khai thác mẫu khó với DINOv2**

```python
def test_hard_sample_mining_flow(tmp_path):
    # Verify bypass identification logic with pretrained extractor
    pass
```

- [ ] **Step 2: Thực thi khai thác mẫu khó trên tập ảnh thật vừa sinh**
Run: `bash scripts/run_pipeline.sh mine --backbone dinov2_small --threshold 0.50`  
Expected: Tách thành công nhóm `Reject` ($P(\text{fake}) \ge 0.5$) và nhóm `Bypass` ($P(\text{fake}) < 0.5 \rightarrow$ Hard Samples).

- [ ] **Step 3: Xác minh file `data/processed/hard_samples_manifest.jsonl`**
Kiểm tra từng mẫu khó có đầy đủ cờ `is_hard_sample=True`, `baseline_score`, và giữ nguyên `demographics`.

- [ ] **Step 4: Commit**
```bash
git add src/pipelines/mine_hard_samples.py
git commit -m "feat: execute hard sample mining with real pretrained DINOv2 baseline"
```

---

### Task 6: Tinh Chỉnh (Fine-Tune) Đối Chứng D0 vs. D1 vs. D2 Phase 3

**Files:**
- Modify: `src/pipelines/train.py`
- Modify: `src/pipelines/evaluate.py`
- Test: `tests/test_pipelines.py`

**Interfaces:**
- Consumes: `data/raw/manifest.jsonl` ($D_0$), `data/processed/curated_manifest.jsonl` ($D_1$), `data/processed/hard_samples_manifest.jsonl` ($D_2$)
- Produces: `experiments/checkpoints/best_model.pt`, `experiments/logs/benchmark_comparison.json`

- [ ] **Step 1: Cấu hình kịch bản huấn luyện đối chứng 3 giai đoạn**
  - $D_0$: Chỉ Real + Baseline Fake ban đầu.
  - $D_1$: $D_0$ + Toàn bộ Synthetic SD.
  - $D_2$: $D_0$ + Synthetic SD + Mẫu khó (nhân đôi trọng số loss `hard_sample_loss_weight=2.0`).
- [ ] **Step 2: Thực hiện huấn luyện tinh chỉnh**
Run: `bash scripts/run_pipeline.sh train configs/config.yaml`
- [ ] **Step 3: Chạy đánh giá kiểm chứng tác dụng**
Run: `bash scripts/run_pipeline.sh eval configs/config.yaml`  
Expected: Chứng minh $D_2$ có `ROC-AUC` cao hơn $D_0$, `hard_sample_recall_gain` $> 0$, và `bypass_rate_reduction` $> 0$.
- [ ] **Step 4: Commit**
```bash
git add src/pipelines/train.py src/pipelines/evaluate.py
git commit -m "feat: complete Phase 3 closed-loop benchmark comparing D0, D1, and D2"
```

---

### Task 7: Tự Động Hóa Gallery Trực Quan & Kiểm Định Toàn Diện

**Files:**
- Create: `src/common/visualizer.py` (cập nhật hàm vẽ gallery so sánh trước/sau)
- Test: `tests/test_full_suite.py`

**Interfaces:**
- Produces: `experiments/predictions/real_vs_synthetic_gallery.png`, Báo cáo kết quả số liệu.

- [ ] **Step 1: Tạo script kết xuất hình ảnh so sánh trực quan**
Vẽ lưới ảnh: [Ảnh thật gốc] $\leftrightarrow$ [Ảnh giả SD sinh ra] $\leftrightarrow$ [Mẫu khó vượt rào DINOv2].
- [ ] **Step 2: Chạy toàn bộ 46+ unit tests**
Run: `pytest tests/ -v`  
Expected: 100% tests pass.
- [ ] **Step 3: Kiểm tra biên dịch lint & format**
Run: `python3 -m py_compile src/**/*.py tests/*.py scripts/*.py`  
Expected: Exit code 0.
- [ ] **Step 4: Commit & Đẩy lên remote**
```bash
git add .
git commit -m "chore: complete real AI model migration, eliminate all mocks, and verify full pipeline"
git push origin baseline/initial-baseline
```
