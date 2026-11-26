# Thiết Kế Giao Diện Trực Quan Streamlit Hybrid — Topic 11 (UTH Computer Vision)

- **Ngày thiết kế**: 2026-11-26
- **Phạm vi**: Dự án nâng cấp giao diện trình diễn & thẩm định dữ liệu End-to-End cho đề tài *Stable Diffusion-Based Synthetic Facial Dataset Generation for Deepfake Detection*
- **Trạng thái**: Ready for User Approval (Đã qua rà soát kỹ thuật đối chiếu codebase & dữ liệu)
- **Tài liệu liên quan**: [PURPOSE.md](../../PURPOSE.md), [AI_REFERENCE.md](../../../.agents/AI_REFERENCE.md), [coursework_metrics.py](../../../src/analysis/coursework_metrics.py), [run_end_to_end_showcase.py](../../../src/pipelines/run_end_to_end_showcase.py)

---

## 1. Mục Đích & Bối Cảnh (Intent & Why)

Hiện tại phần trình bày tĩnh của dự án (4 coursework notebooks `00`–`03` và [presentation_final_acceptance.html](../../reports/presentation_final_acceptance.html)) đã đồng bộ số liệu sống từ [coursework_metrics.py](../../../src/analysis/coursework_metrics.py). Tuy nhiên phần giao diện tương tác ở `notebooks/coursework/end_to_end_pipeline_interface.ipynb` vẫn hoạt động bằng cách sửa biến Python trong cell rồi chạy lại tuần tự, chưa đủ trực quan cho buổi bảo vệ đồ án trực tiếp (5–10 phút hỏi đáp & demo live).

Giao diện mới phải thể hiện **đúng bản chất 4 tầng của dự án**:
1. **Minh bạch cả 3 chế độ sinh dữ liệu (`Text2Img`, `Img2Img`, `Inpainting`)**: Khác biệt rõ đầu vào của từng chế độ (`Prompt + Latent Noise` vs `Ảnh FFHQ gốc + Strength` vs `Ảnh FFHQ gốc + Mask vùng sửa`) đến đầu ra và so sánh trực diện Trước/Sau.
2. **Đặt Tầng Đánh Giá Dữ Liệu (`Curation Gate`) làm trọng tâm (Dataset-First)**: Người xem nhìn thấy bằng mắt thường vì sao một ảnh sinh ra **✓ VƯỢT QUA để được chọn** hay **✕ BỊ LOẠI** (6 tiêu chí đo lường, lớp phủ 5 điểm YuNet, vùng da má FFT, độ lệch màu `ΔE`, viền ghép `Seam Diff`), và có thể kéo thanh ngưỡng lọc để thử nghiệm tác động lên tập mẫu.
3. **Chứng minh tính hữu dụng (`Hard-Sample Mining` → `M0 → M2* → M3`)**: Các mẫu vượt qua Curation Gate đi tiếp qua mô hình gốc `M0` (tách ra các mẫu đánh lừa được `M0` làm Hard Samples), rồi đối chiếu với `M2*` và `M3 Champion` kèm bản đồ nhiệt giải thích vùng mặt nào giúp `M3` bắt được ca khó.

---

## 2. Mục Tiêu, Ngoài Phạm Vi & Điều Cấm Kỵ

### 2.1. Mục tiêu (Goals)
- Xây dựng Web App **Streamlit Hybrid** bố cục **1 màn hình xuyên suốt (`unified-1screen`)**:
  - Duyệt **tức thì (0s)** kho dữ liệu thật qua cả 3 chế độ SD 1.5 + ngoại miền OOD Realistic Vision v5.1.
  - Cho phép **Upload ảnh bất kỳ** (hoặc chọn ảnh FFHQ) để chạy live `ImageCurationGate` + `M0 / M2* / M3` trong ~1–2 giây.
  - Cung cấp nút **"Sinh thử 1 ảnh Live (SD 1.5)"** (chọn chế độ `Text2Img` / `Img2Img` / `Inpaint` + vùng mask `lips` / `periocular` / `full`) chạy an toàn trên GPU 4GB VRAM.
  - Hiển thị **Kịch bản Vàng 1-Click (Golden Presets)** cho buổi báo cáo kèm dải **Scorecard Proof of Utility** tổng cục đồng bộ từ `collect_live_coursework_metrics()`.
- Tuân thủ 100% bất biến kiến trúc trong `AGENTS.md`: toàn bộ logic CV/DL nằm trong `src/`, app Streamlit chỉ là tầng trình bày mỏng trong `scripts/`.

### 2.2. Ngoài phạm vi (Non-Goals)
- Không huấn luyện lại mô hình `M0..M3` hay sinh batch hàng trăm/ngàn ảnh từ UI (việc đó thuộc `scripts/run_pipeline.sh`).
- Không vẽ mask tự do bằng canvas chuột phức tạp trên ảnh (sử dụng hàm sinh mask giải phẫu tự động `generate_dynamic_face_mask()` theo vùng `lips`, `periocular`, `full` hoặc `nose`).
- Không triển khai lên server công cộng đa người dùng (chạy local phục vụ trình diễn và kiểm thử).
- Không thay thế hay xóa 4 coursework notebooks `00`–`03` và 2 báo cáo HTML tĩnh.

### 2.3. Điều cấm kỵ (Anti-Goals)
- **Cấm dùng mock weights hoặc số liệu bịa đặt** trong bất kỳ màn hình inference hay scorecard nào; mọi dự đoán `M0 / M2* / M3` phải nạp checkpoint thật (`model_state_dict` trong `experiments/checkpoints/`).
- **Cấm để treo UI hoặc tràn VRAM 4GB** khi kéo thanh ngưỡng `τ` / ngưỡng lọc hoặc bấm sinh ảnh SD 1.5 live.
- **Cấm `print()` trong `src/`** (dùng `src.common.logger.get_logger`) và **cấm để lộ đường dẫn tuyệt đối cá nhân** trong Markdown/UI labels.

---

## 3. Ràng Buộc Kỹ Thuật & Môi Trường (Constraints)

- **Phần cứng thực tế**: `NVIDIA GeForce RTX 3050 Laptop GPU` (**4096 MiB VRAM**, ~3.78 GiB khả dụng). Vì giới hạn 4GB VRAM:
  - Không được nạp đồng thời Stable Diffusion (`~2.5–3.2GB FP16`) + CLIP ViT-B/32 + 3 model DINOv2 (`~100MB/model`) trong cùng một tiến trình GPU.
  - Tách bước sinh 1 ảnh Stable Diffusion thành worker subprocess ngắn hạn, đồng thời **giải phóng GPU cache của Streamlit** trước khi gọi worker và nạp lại detector cache sau khi worker kết thúc.
- **Môi trường Offline Cache**: Hugging Face cache cục bộ hiện có sẵn `facebook/dinov2-small`, `runwayml/stable-diffusion-v1-5`, và `stable-diffusion-v1-5/stable-diffusion-inpainting` (còn `configs/dataset_factory.yaml` trỏ `stable-diffusion-v1-5/stable-diffusion-v1-5`). Worker và `ClipScorer` cần hỗ trợ fallback an toàn giữa hai repo ID và fallback `local_files_only=True` hoặc CPU cho CLIP nếu mất mạng.
- **Hợp đồng Tensor & Hệ màu (`AGENTS.md`)**:
  - OpenCV/NumPy dùng `(H, W, 3)` `uint8` `[0, 255]` hệ `BGR`; PIL/Streamlit/PyTorch dùng `RGB`. Phải chuyển đổi tường minh qua `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` khi giao tiếp với OpenCV/YuNet.
  - Inference DINOv2 bắt buộc resize ảnh đầu vào về `(512, 512)` bằng `Image.Resampling.LANCZOS`, chuẩn hóa tensor `(B, 3, 512, 512)` `float32` `[0.0, 1.0]` (để `MetaDinoV2Wrapper` nội suy tiếp về bội số 14 = `504×504`), chạy trong `model.eval()` + `torch.no_grad()` + `torch.amp.autocast("cuda", enabled=...)`.

---

## 4. Các Phương Án Đã Cân Nhắc & Quyết Định (Approaches)

### 4.1. Lựa chọn Kiến trúc Thực thi
1. **Phương án A — Streamlit Hybrid (`@st.cache_resource` + Worker tách riêng) [ĐÃ CHỌN]**:
   - *Cơ chế*: Kho mẫu 3 chế độ (Text2Img / Img2Img / Inpaint + RV5.1 OOD) đọc trực tiếp từ kết quả và manifest thật (~0s); khi người dùng upload ảnh hoặc đổi ảnh FFHQ mới thì chạy `ImageCurationGate` + `M0/M2*/M3` trực tiếp trên model đã cache (~1–2s); chỉ khi bấm nút *"Sinh thử 1 ảnh Live (SD 1.5)"* mới giải phóng GPU cache và gọi subprocess `live_sd_worker`.
   - *Đánh đổi*: Cần thêm dependency `streamlit` vào `requirements/ui.txt` (không động tới `requirements/notebook.txt`). Đổi lại giao diện mượt tuyệt đối khi kéo thanh trượt `τ`/Curation, không bao giờ đứng hình hay OOM 4GB VRAM lúc bảo vệ.
2. **Phương án B — Streamlit Full-Live GPU (Sinh SD mỗi lần tương tác)**:
   - *Lý do loại*: Mỗi lần đổi ca mẫu phải đợi 15–35 giây, xung đột 4GB VRAM giữa SD 1.5 và CLIP/DINOv2, rủi ro cao khi thuyết trình.
3. **Phương án C — Giữ Notebook + Mở rộng HTML Tĩnh**:
   - *Lý do loại*: Không hỗ trợ upload ảnh bất kỳ từ máy hội đồng để kiểm thử live và không kéo ngưỡng lọc dữ liệu tương tác với tính toán Python thật mà không mở code cell.

### 4.2. Lựa chọn Bố cục Giao diện
1. **Gộp 1 màn hình xuyên suốt (`unified-1screen`) [ĐÃ CHỌN]** vs **Tách 2 Tab (`staged-2tabs`)**: Chọn 1 màn hình với luồng đọc từ trên xuống: *Thanh Presets & Scorecard → Chế độ & Lưới Đánh Giá Data → Bảng X-Ray 4 Cổng (Input → Output → Curation Gate → M0/M2*/M3 & Patch Map)*.

---

## 5. Thiết Kế Kiến Trúc & Hợp Đồng Mô-đun

```text
scripts/app_streamlit.py (Tầng UI mỏng - zero ML/CV domain logic)
   │
   ├──► src/analysis/coursework_metrics.py   (Scorecard toàn cục 4 vault + 20.4K)
   ├──► src/analysis/ui_showcase.py          (Core UI Backend: Presets, Inspector, Overlays, DINOv2 Map)
   │        ├── src/pipelines/curate_image_dataset.py (ImageCurationGate)
   │        ├── src/common/metrics.py                 (YuNet, FFT da má, CLIP, ΔE, Seam)
   │        └── src/models/detector/image_detector.py (M0, M2*, M3 checkpoints)
   │
   └──► [Subprocess on button click] python -m src.pipelines.live_sd_worker
            └── src/models/diffusion/stable_diffusion_engine.py (SD 1.5 + generate_dynamic_face_mask)
```

### 5.1. Backend Trực Quan Hóa & Kiểm Định — `src/analysis/ui_showcase.py` (Mới)
Cung cấp các hàm thuần túy, tái sử dụng được trong unit tests và UI:

1. **Quản lý Kho Mẫu Held-Out & Kịch Bản Vàng (`build_ui_showcase_catalog`, `load_curated_showcase_catalog`, `get_golden_presets`)**:
   - `[Design Call]` **Chuẩn hóa nguồn dữ liệu Held-Out 100% & Đồng bộ Checkpoint Mới Nhất**:
     - Không lấy trực tiếp `experiments/demo_runs/sd15_pregen_200/scored_manifest.jsonl` cũ (do sinh trước đợt retrain `19:34–21:10` và chứa 65% ảnh `d0_train`).
     - Kế hoạch triển khai sẽ chạy đồng bộ lại `experiments/demo_runs/sd15_pregen_200/` theo checkpoint hiện hành VÀ dựng thêm 1 chỉ mục tĩnh chuyên cho UI tại `experiments/demo_runs/ui_showcase_catalog.json` chứa tập mẫu đại diện cân bằng (gồm cả điểm `M0/M2*/M3` mới nhất và `cheek_fft_ratio` tách riêng):
       - **3 Tab SD 1.5 (`sd_text2img`, `sd_img2img`, `sd_inpaint`)**: toàn bộ ca **✓ VƯỢT QUA** & **Mẫu khó** rút 100% từ tập **held-out test** [master_frozen_test.jsonl](../../../data/benchmarks/master_frozen_test.jsonl) (`159` text2img, `222` img2img, `207` inpaint — đều có `source_real_id` trỏ FFHQ gốc) + `data/active_learning/cumulative_challenge_test.jsonl`; toàn bộ ca **✕ BỊ LOẠI** rút từ 40 ảnh rớt của `sd15_pregen_200/curation_report.jsonl` (`21` text2img, `7` img2img, `12` inpaint — chưa từng dùng để train bất kỳ mô hình nào).
       - **Tab 4 (`ood_and_real`)**: rút ca **OOD RV5.1** từ tập độc lập danh tính [unseen_rv51_test.jsonl](../../../data/benchmarks/D3_clean/unseen_rv51_test.jsonl) (`304` `rv51_img2img`, `297` `rv51_inpaint`) + **Ảnh thật FFHQ đối chứng (`label=0`)** từ `master_frozen_test.jsonl` (để chứng minh True Negative / không báo động nhầm). Lưu ý rõ: `synthetic_face_dataset_6k/` không lưu ảnh nguồn Kaggle CelebA-HQ cục bộ nên card OOD RV5.1 chỉ hiển thị đơn ảnh kèm lớp phủ YuNet/FFT/Heatmap (ẩn thanh kéo rèm Input↔Output).
   - `get_golden_presets()` trả 4 kịch bản đại diện tức thì (`Text2Img`, `Img2Img`, `Inpaint`, `OOD RV5.1 & Real FFHQ`) bảo đảm toàn bộ ca ✓ Đạt đều thuộc tập held-out 100%.

2. **Tải & Cache Detector Thật (`load_detector_bundle`, `unload_detector_bundle`)**:
   - Nạp 3 checkpoint chính thức:
     - `M0 (Baseline)`: `experiments/checkpoints/baseline_d0_best.pt`
     - `M2* (Active R2)`: `experiments/checkpoints/active_learning_m2_best.pt`
     - `M3 (Champion)`: `experiments/checkpoints/active_learning_m3_best.pt`
   - Đọc backbone từ `configs/config.yaml` (`dinov2_small`), `strict=True`, chuyển `model.eval()`. `[Design Call]` Cho phép chọn thiết bị `"auto"` (GPU nếu còn trống >= 900 MiB VRAM, ngược lại fallback CPU cho 3 model nhẹ để nhường toàn bộ GPU cho SD worker nếu người dùng hay sinh ảnh). `unload_detector_bundle()` thực hiện `del` + `gc.collect()` + `torch.cuda.empty_cache()` trước khi chạy worker SD.

3. **Kiểm Định Live Một Ảnh (`inspect_pipeline_sample`)**:
   - Đầu vào: `target_image` `(H, W, 3)` RGB hoặc `Path`, tùy chọn `source_image`, `mask` `(1, H, W)` hoặc `mask_region`, `mode` (`sd_text2img`, `sd_img2img`, `sd_inpaint`, `rv51_ood`, `real_ffhq`, `custom_upload`), `prompt`, các ngưỡng từ thanh trượt (`min_sharpness`, `min_cheek_fft`, `min_clip`, `max_delta_e`, `max_seam`, `tau`), và cờ `run_detector_if_rejected: bool = False`.
   - Quy trình & đo thời gian (`latencies_ms`):
     - Resize bản chuẩn làm việc về `(512, 512)` RGB.
     - Chạy `detect_face_yunet()` + `extract_face_biometrics_rgb()` + `audit_image_extended()` (và `ClipScorer` cấu hình offline-safe trên CPU để không tranh VRAM GPU; nếu môi trường không có weight CLIP thì `clip_score = None` và cảnh báo nhẹ thay vì crash).
     - `[Design Call]` **Trích tách `cheek_fft_ratio` thực & đồng bộ `Min CLIP` theo chế độ**:
       - Gọi riêng `compute_cheek_skin_fft()` khi kiểm định (và lưu sẵn vào `ui_showcase_catalog.json`) để tách `cheek_fft_ratio` khỏi `fft_high_freq_ratio` (whole-face), giúp thanh trượt `Min Cheek FFT`, 2 ô ROI vùng má vẽ trên mặt, và tiêu chí Đạt/Rớt khớp nhau 100% về toán học mà không thay đổi contract của `ExtendedImageAuditReport`.
       - Thanh `Min CLIP` khởi tạo mặc định đúng theo chế độ đang chọn qua `resolve_effective_min_clip(0.20, mode)` (`0.20` cho `text2img`, `0.12` cho `img2img`/`inpaint`), và khi kéo thanh trượt sẽ áp trực tiếp ngưỡng đó lên chế độ đang xem (không bị kẹp chết ở `0.12`).
     - Áp lại bộ ngưỡng tương tác từ người dùng lên các chỉ số đo được để quyết định `gate_passed: bool` và danh sách `rejection_reasons` (bằng cả lý do kỹ thuật gốc lẫn mô tả tiếng Việt dễ hiểu lúc thuyết trình).
     - Chạy suy luận `M0`, `M2*`, `M3` nếu `gate_passed` (hoặc nếu `mode == "real_ffhq"` / người dùng bật tùy chọn `"Chạy thử Detector kể cả khi ảnh bị loại"` cho ảnh tự upload `[Design Call]`).
     - Trả về `InspectionResult` (dataclass) gồm: `gate_passed`, bảng 6 chỉ số kèm ngưỡng (bao gồm cả `cheek_fft_ratio` và `fft_high_freq_ratio`), `detector_scores` `{M0, M2*, M3}`, `is_hard_sample` (`label == 1 and M0 < tau`), `overlays` (ảnh đã vẽ lớp phủ), và `heatmaps` (`M0` vs `M3`).

4. **Lớp Phủ Thị Giác & Bản Đồ Giải Thích (`render_face_audit_overlay`, `compute_dinov2_patch_heatmap`, `render_latent_noise_preview`, `build_before_after_wipe_html`)**:
   - `render_face_audit_overlay()`: Chuyển tường minh RGB→BGR cho vẽ OpenCV (hoặc vẽ bằng PIL/NumPy RGB thuần) để phủ bounding box xanh/đỏ của YuNet, 5 điểm landmark (2 mắt, mũi, 2 khóe miệng), 2 ô ROI vùng da má đo FFT, và đường viền mask inpaint lên bản sao `(512, 512, 3)` RGB.
   - `compute_dinov2_patch_heatmap()`: Trích đặc trưng không gian ở tầng chuẩn hóa cuối `outputs.last_hidden_state` của `MetaDinoV2Wrapper.backbone` (với kích thước đầu vào `(1, 3, 504, 504)` tức lưới `36 × 36` patch `14×14`): tính độ tương đồng cosine giữa từng patch token `[1:]` với `[CLS]` token `[:, 0]` rồi kết hợp trọng số đầu phân loại `classifier` theo grad-free directional projection (hoặc `register_forward_hook` / gradient attribution `|∂logit/∂patch|` trong 1 bước backward nhẹ nếu người dùng bật giải thích `M0` vs `M3`) `[Design Call]`. Cách trích `last_hidden_state` không phụ thuộc `output_attentions=True` nên an toàn 100% với PyTorch SDPA mặc định.
   - `[Design Call]` Dùng **Grad×Patch Sensitivity** (1 forward+backward nhỏ trên `MetaDinoV2Wrapper` hoặc patch-logit norm trên lưới `36×36` nội suy bilinear lên `512×512` colormap JET phủ alpha 0.45) cho ảnh đang chọn giúp bản đồ nhiệt phản ánh đúng sự khác biệt quyết định của **`M0` vs `M3`** (khác biệt trọng số Block 11 & classification head), thay vì attention tự giám sát giống hệt nhau giữa các model.
   - `render_latent_noise_preview()`: Tạo ảnh trực quan `(256, 256, 3)` biểu diễn nhiễu Gaussian khởi tạo từ `seed` (`torch.Generator().manual_seed(seed)`) cho đầu vào `Text2Img`.
   - `build_before_after_wipe_html()`: Tạo đoạn HTML/CSS/JS tự chứa (base64 JPEG inline) thanh kéo rèm Trước/Sau mượt để nhúng qua `streamlit.components.v1.html()` mà không cần cài thêm thư viện third-party hay gây re-run Python khi kéo chuột.

### 5.2. Worker Sinh Ảnh Live Cô Lập VRAM — `src/pipelines/live_sd_worker.py` (Mới)
- Cho phép gọi qua CLI:
  `python -m src.pipelines.live_sd_worker --mode {text2img|img2img|inpaint} --prompt "..." [--source-image ...] [--mask-region lips|periocular|full] [--steps 20] [--guidance 6.2] [--strength 0.30] [--seed 42] --output-dir ...`
- Thiết kế bảo vệ 4GB VRAM (`RTX 3050`):
  - Khởi tạo `StableDiffusionEngine` với `cpu_offload=True` và kiểm tra cập nhật nhỏ trong `StableDiffusionEngine._switch_mode()` để truyền `local_files_only=self.local_files_only` + fallback thử cả `runwayml/stable-diffusion-v1-5` và `stable-diffusion-v1-5/stable-diffusion-v1-5` theo cache Hugging Face cục bộ.
  - Bật `enable_attention_slicing("auto")` và `enable_vae_slicing()` (đã có sẵn trong `optimize_pipeline_memory()`).
  - Nếu `--mode inpaint`: phát hiện mặt bằng `detect_face_yunet()` trên ảnh nguồn `(512, 512)` RGB → gọi `generate_dynamic_face_mask(..., target_region=args.mask_region)` để tạo tensor mask `(1, 512, 512)`.
  - Lưu đồng bộ 3–4 file kết quả vào `--output-dir`: `input.png` (nếu `img2img`/`inpaint`), `mask.png` (nếu `inpaint`), `output.png`, và `metadata.json` (`mode`, `model_id`, `prompt`, `seed`, `steps`, `guidance_scale`, `strength`, `mask_region`, `latency_ms`).
  - Kết thúc process để giải phóng 100% CUDA memory về hệ điều hành trước khi quay lại Streamlit.

### 5.3. Tầng Giao Diện Streamlit — `scripts/app_streamlit.py` (Mới)
Cấu trúc **1 màn hình cuộn dọc xuyên suốt** bám sát bản phác thảo v3 đã chốt:

1. **Thanh Điều Khiển Trên Cùng (Header + Presenter Bar)**:
   - Tiêu đề Đề tài 11 + huy hiệu trạng thái tài nguyên (`GPU/CPU`, VRAM khả dụng, độ trễ xử lý).
   - **4 nút Kịch bản Vàng 1-Click**: `[Ca 1: Text2Img (Đạt vs Cụt trán)]` · `[Ca 2: Img2Img (Đạt vs Da sáp FFT)]` · `[Ca 3: Inpaint (Kính/Môi vs 2 mặt)]` · `[Ca 4: Ngoại miền OOD RV5.1]`.
   - Expander **"Scorecard Toàn Cục M0 → M1 → M2* → M3 (Proof of Utility)"** gọi `collect_live_coursework_metrics()` để đối chiếu ngay kết quả vĩ mô (Master Test, Challenge Hard Vault, Unseen RV5.1, Universal 20.4K) ngay trong buổi bảo vệ.

2. **Khối 1 — Chọn Chế Độ, Nguồn Ảnh & Thẩm Định Lô Dữ Liệu (`Data Curation Gate`)**:
   - `[Design Call]` **4 nút chuyển chế độ rạch ròi** (`1. Text2Img` · `2. Img2Img` · `3. Inpaint` · `4. OOD RV5.1 & Real FFHQ`) + nút mở `Upload Ảnh Lạ / Sinh Live SD 1.5`:
     - 3 Tab đầu giữ gọn cho `SD 1.5` (luôn có đủ đầu vào → đầu ra & phễu Đạt/Loại riêng của chế độ đó).
     - Tab 4 tách riêng nhóm `OOD RV5.1 (Unseen)` và `Ảnh Thật FFHQ Đối Chứng (Held-out Real, label=0)` — không trộn ảnh Real hay mẫu RV5.1 ngoại miền vào công thức tỷ lệ Đạt/Loại của `SD 1.5`.
   - Thanh slider điều chỉnh trực tiếp ngưỡng lọc (`Min Sharpness`, `Min Cheek FFT`, `Min CLIP` mặc định `0.20` ở Text2Img và `0.12` ở Img2Img/Inpaint, `Ngưỡng Detector τ`).
   - Dải phễu tóm tắt + Lưới thẻ ảnh thật (lọc `Tất cả` / `✓ Vượt qua` / `✕ Bị loại` với các Tab 1–3, hoặc lọc `RV5.1 OOD` / `FFHQ Real` ở Tab 4) tự đổi viền xanh/đỏ và gắn nhãn lỗi ngay khi kéo ngưỡng.
   - Khu **Upload Ảnh Lạ & Sinh Live (Expander gọn)**:
     - Upload file `.png/.jpg/.webp` (tùy chọn upload thêm ảnh gốc nếu muốn kiểm tra cặp `Img2Img`/`Inpaint`).
     - Hoặc chọn ảnh gốc FFHQ + Chế độ (`Text2Img` / `Img2Img` / `Inpaint`) + Vùng Mask (`lips` / `periocular` / `full`) + Prompt mẫu/tự sửa + Seed → bấm **"Sinh thử 1 ảnh Live (SD 1.5)"** (hiển thị thanh tiến trình rõ ràng khi gọi `live_sd_worker`, xong tự động đẩy ngay ảnh mới sinh xuống Khối X-Ray bên dưới!).

3. **Khối 2 — Bảng Soi X-Ray 4 Tầng Cho Ảnh Đang Chọn**:
   - **Cột A (Tầng 1 — Đầu vào → Đầu ra theo đúng chế độ)**:
     - *Text2Img*: Khối **Latent Noise (sinh từ `seed`) + Prompt + Negative Prompt + Steps/CFG** → Ảnh đầu ra + overlay.
     - *Img2Img*: Ảnh FFHQ gốc + Strength → Ảnh đầu ra + **Thanh kéo rèm Trước/Sau (`build_before_after_wipe_html`)**.
     - *Inpaint*: Ảnh FFHQ gốc có **lớp phủ Mask vùng sửa** → Ảnh đầu ra + **Thanh kéo rèm Trước/Sau** + ảnh viền mask.
     - *OOD RV5.1 / FFHQ Real*: Hiển thị đơn ảnh đầu vào kiểm định kèm ghi chú nguồn gốc (`Unseen RV5.1 — nguồn Kaggle CelebA-HQ không lưu cục bộ` hoặc `Held-out FFHQ Real ground-truth = 0`) và lớp phủ YuNet/FFT.
     - Có nút tick bật/tắt lớp phủ (`YuNet bbox + 5 landmarks + 2 ô FFT da má + Mask`).
   - **Cột B (Tầng 2 — Phiếu Thẩm Định Data Gate: "Có vượt qua để được chọn?")**:
     - Banner nổi bật **`✓ ĐẠT CHUẨN — ĐƯỢC CHỌN`** (xanh) hoặc **`✕ BỊ LOẠI TẠI CỔNG DATA`** (đỏ kèm danh sách lý do chi tiết).
     - 6 thanh tiến trình kèm vạch ngưỡng (`Laplacian Sharpness`, `Cheek Skin FFT` theo `cheek_fft_ratio`, `CLIP ViT-B/32`, `YuNet Geometry`, `CIELab ΔE`, `Boundary Seam / Highlight Blowout`).
   - **Cột C (Tầng 3 & 4 — Bẫy Mẫu Khó M0 & Đấu Trường M0 → M2* → M3)**:
     - Nếu rớt Tầng 2 (và không bật ép chấm): khóa hiển thị rõ ảnh bị loại ngay từ khâu làm sạch dữ liệu.
     - Nếu là `FFHQ Real (label=0)`: hiển thị rõ kiểm định **False-Positive Check** (`P(fake) < τ` → `✓ True Negative`, không báo động nhầm).
     - Nếu là ảnh giả vượt qua gate: gắn nhãn trạng thái tại `M0` (`BỎ LỌT < τ → HARD SAMPLE` hay `BẮT NGAY`), hiển thị 3 thanh `P(fake)` của `M0 Baseline`, `M2* Active R2`, `M3 Champion` tại `τ` hiện tại, kèm cặp **Bản đồ nhiệt `M0` vs `M3`** đặt cạnh nhau để giảng viên thấy vùng mắt/miệng mà `M3` nhận ra dấu vết giả.

### 5.4. Dependencies, Tự động hóa & Kiểm thử
- Tạo `requirements/ui.txt` chứa `streamlit>=1.38.0,<2.0.0` (giữ `requirements/notebook.txt` tách biệt).
- Thêm target `make ui` trong `Makefile`: chạy `$(PYTHON) -m streamlit run scripts/app_streamlit.py`.
- Tạo `tests/test_ui_showcase.py`:
  - Kiểm tra `load_curated_showcase_catalog()` và `get_golden_presets()` đọc đủ 3 chế độ (Pass & Fail) + RV5.1 OOD từ file thật.
  - Kiểm tra `inspect_pipeline_sample()` trên ảnh thật và 3 checkpoint thật (`M0`, `M2*`, `M3`), xác nhận shape overlay `(512, 512, 3)` `uint8`, có heatmap hợp lệ của `M0` và `M3`, và cơ chế chặn khi ảnh không đạt gate.
  - Kiểm tra `live_sd_worker.py` parse CLI và chạy được chế độ `mock_sd` trong unit test (riêng production inference của app luôn dùng model thật, không dùng mock).

---

## 6. Phần Để Lại Cho Người Triển Khai (Left to Builder)

- Tinh chỉnh CSS chi tiết trong `scripts/app_streamlit.py` (khoảng cách thẻ, màu dark/light đồng bộ theme Streamlit).
- Quyết định kích thước batch mẫu mặc định tải lên lưới gallery (ví dụ 18–24 ảnh đại diện mỗi chế độ có nút tải thêm để giữ tốc độ render lưới < 0.3s).
- Cài đặt gói `streamlit` vào `.venv` từ `requirements/ui.txt` và chạy smoke verification trực tiếp của Streamlit server.
