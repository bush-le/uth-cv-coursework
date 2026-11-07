# Protocol — Pretrained SD video benchmark

- **Motivation/Background**: So sánh output cần giữ inputs và cách đo nhất quán, không suy diễn tốc độ từ tên model.
- **Purpose**: Khóa registry SD-only, controls và metrics hai phase trước thực nghiệm.
- **Overview Pipeline**: Feasibility -> Pilot -> Development -> Holdout -> Selection -> Batch -> Conditional detector.
- **Detailed Plan**: §1 Registry; §2 Stages; §3 Generator metrics; §4 Utility metrics; §5 Fine-tune; §6 Artifacts.
- **References**: [DEFINE](../PURPOSE.md), [DESIGN](../phases/SD_VIDEO_SYSTEM_DESIGN.md), [ROADMAP](../progress/SD_VIDEO_ROADMAP.md).
- **Created**: 2026-11-07T14:12:25+07:00
- **Last Updated**: 2026-11-07T14:21:07+07:00

---

[STATUS: ACTIVE]

## 1. Registry và điều kiện vào benchmark

- **Face-Adapter:** [code](https://github.com/FaceAdapter/Face-Adapter), [weights](https://huggingface.co/FaceAdapter/FaceAdapter). Face swapping trên ảnh; video wrapper phải xử lý frames và dựng clip. Pin base SD checkpoint từ code/config đã chọn, không suy đoán phiên bản.
- **REFace:** [code](https://github.com/Sanoojan/REFace), [weights](https://huggingface.co/Sanoojan/REFace). SD 1.4 initialization, baseline frame.
- **VFace:** [code](https://github.com/Sanoojan/VFace), [backbone weights](https://huggingface.co/Sanoojan/REFace). Video extension trên REFace với flow/temporal processing; chia sẻ backbone nên không phải generator nền độc lập với REFace.
- **VividFace:** [code](https://github.com/deepcs233/VividFace), [weights](https://huggingface.co/deepcs233/VividFace). SD 1.5 + video components; custom input cần bbox và 5 landmarks/frame theo README.

Repo công bố code/liên kết weights, nhưng task DEFINE/DESIGN chưa tải toàn bộ checkpoints, kiểm tra licenses hoặc chạy GPU. Pin actual revisions/hashes ở implementation. Chỉ registry trên; không thêm diffusion khác nền hoặc GAN/LivePortrait. Phụ trợ recognition/detection/flow không bị giới hạn là SD.

## 2. Experimental stages và fair comparison

1. Smoke upstream: sample nhỏ nhất hợp lệ, xác minh CUDA/weights/dependency.
2. Pilot: **3 cặp khác người × 4 backend = tối đa 12 clip**, mỗi output 5s/10 FPS/50 frames; 600 frame cuối, không nhất thiết 600 forward passes.
3. Development đề xuất 10–20 cặp đa dạng theo budget; chạy lặp warm runs trên subset, tách cold download và load. Quy mô chính thức chốt sau pilot.
4. Khóa QA/config trên development rồi đánh giá holdout tách người/video gốc. Không dùng pilot để tuyên bố generalization.
5. Batch dataset: mục tiêu đề xuất 100 clip QA-pass, điều chỉnh theo thực đo. Đây không phải yêu cầu giảng viên đã xác nhận.

Giữ source/target/time segment/canonical frames và evaluator. **Upstream-default hợp lệ** trước; **tuned-budget** là thí nghiệm riêng. Không ép cùng steps/scheduler khi các model không tương đương. Ghi native size, actual processed frames, chunk overlap, context padding và output resampling. Báo backend lỗi, missing frames và coverage, không chỉ chọn các clip đẹp.

So sánh hệ thống không phải ablation một biến: khác training data/backbone/conditioning. Giữ chung phần có thể giữ và công khai phần khác. Nếu thử cải thiện riêng một backend, thay một yếu tố mỗi ablation.

## 3. Phase 1 — Chất lượng generator và dataset

### 3.1 Identity và thuộc tính đích

- Cosine embedding output–source, thêm output–target để phát hiện chưa swap; ID retrieval khi có gallery. Report clip mean và lower-tail/worst valid frames.
- Encoder đánh giá ưu tiên độc lập encoder conditioning; không dùng một cutoff cosine phổ quát. Hiệu chỉnh genuine/impostor trên development, ghi threshold/method.
- Pose error yaw/pitch/roll theo độ so target; expression/landmark error chuẩn hóa và hạn chế nhiễu từ hình dạng identity. Landmark distance thô không phải expression thuần.
- Báo validity/confidence và tỷ lệ frame tính được; không bỏ no-face frames khỏi denominator mà không ghi.

### 3.2 Temporal và chất lượng trực quan

- Motion-compensated warping error trên visible valid face regions, mask occlusion và reference flow thống nhất.
- Raw consecutive-frame difference không phải warping error. Kèm motion fidelity để frozen clip không pass chỉ vì temporal error thấp.
- QA rubric: sai ID, anatomy lỗi, seam, flicker, vật che bị ghi đè, background drift, chunk discontinuity; clip pass/reject và severity từng lỗi.
- Pilot có hai reviewer độc lập nếu nhóm bố trí được; ghi agreement và cách phân xử. Randomize thứ tự backend, tách đánh giá ID và preference đẹp/xấu.

### 3.3 Dataset và cost

- Completed/attempted, QA-pass/completed, QA-pass/attempted; unique identities/pairs/parent clips, phân bố độ khó, rejection reasons và metric coverage.
- Model load/download, prepare, generation, postprocess và end-to-end seconds/clip; seconds/output frame cùng actual processed frame count.
- GPU allocated/reserved peak và observed process VRAM nếu có, RAM, OOM/failures; ghi precision/offload/multi-GPU/cold-warm cache.
- Không đo VRAM ở parent không nạp model rồi công bố thành peak inference.

### 3.4 Metric không dùng sai và tiêu chí chọn

Không lấy PSNR/SSIM toàn mặt so người đích làm mục tiêu chính vì đổi danh tính. FID/FVD optional khi đủ sample, extractor thật, preprocessing/cỡ mẫu/uncertainty rõ; không dùng pooling random hoặc variance shortcut rồi gọi FVD chuẩn. CLIP alignment không thay identity metric. Face confidence không chứng minh anatomy đúng.

Success là benchmark tái lập và selection có evidence. Development dùng để khóa thresholds/tolerance, Pareto chất lượng–chi phí và ngân sách; holdout chỉ kiểm định. Không viết sẵn số điểm thắng/thời gian model. Model không khả thi được ghi bằng log thực, không loại âm thầm.

## 4. Phase 2 — Dataset utility cho detector, có điều kiện

Chỉ mở sau dataset hợp lệ, baseline real/fake và independent test có quyền sử dụng. Không gọi generator tốt vì detector đoán fake dễ hoặc khó.

- D0: detector học baseline real/fake.
- D1: cùng baseline + augmentation truyền thống/oversampling đối chứng budget.
- D2: cùng baseline + SD fake sinh từ train identities/clips.

Đề xuất ResNet-18 ImageNet pretrained, fine-tune binary head/backbone cùng initialization/optimizer/seed list. Đây là detector phụ trợ, không phải thêm generator ngoài SD. Mean frame scores -> clip score là baseline cố định, không gọi temporal detector.

Giữ val/test, real/fake sampling và optimizer-step budget; báo matched-budget và lợi ích thực dụng nếu data size khác. Threshold chọn val, không tune test. Clip-level ROC-AUC/AP/TPR@FPR chính; F1/accuracy/precision/recall bổ sung. Single-class AUC undefined; equal scores trên tập đủ hai lớp phải AUC=0.5, dùng metric implementation chuẩn có tests.

Holdout generator khác checkpoint/family/method mạnh hơn chỉ đổi seed. Báo in-domain và external/cross-generator riêng. Nếu baseline detector data có fake ngoài SD, đó là dữ liệu đối chứng được khai báo, không mở registry generator. Bootstrap theo identity/group hoặc clip độc lập, không coi frame cùng clip độc lập; nhiều seeds và CI của chênh lệch D2–D0/D1 nếu đủ sample.

**Utility success:** cải thiện primary metric vượt mức hữu ích được khóa trước test, uncertainty đủ thuyết phục và không giảm baseline performance quá tolerance đã đặt. Nếu chưa đủ evidence, kết luận chưa chứng minh; không bắt buộc kết quả dương.

## 5. Fine-tuning generator không phải baseline

Chỉ mở sau error analysis cụ thể, upstream support và train data/license/compute phù hợp. Freeze pretrained baseline, train-only identities, tune development và so cùng holdout/QA/cost. Không dùng data fine-tune làm test. Nếu không tốt hơn, giữ pretrained. Config tuning không phải cập nhật weights.

## 6. Artifacts tối thiểu

Input manifest/hash và permissions; code/model/env revisions; resolved configs; job logs/status; videos/hashes; frame coverage; metric implementation/version; per-clip scores, reviewer rubric, summary với counts và CI khi hợp lệ. Chưa có kết quả đo trong tài liệu này.

---

## References & Technical Documentation

### Official API Docs

[Diffusers](https://huggingface.co/docs/diffusers/index), [scikit-learn metrics](https://scikit-learn.org/stable/modules/model_evaluation.html), [Torchvision](https://docs.pytorch.org/vision/stable/index.html).

### Technical Standards

[FFmpeg](https://ffmpeg.org/ffmpeg.html), [JSON Lines](https://jsonlines.org/).

### CV Conventions

[Face-Adapter](https://github.com/FaceAdapter/Face-Adapter), [REFace](https://github.com/Sanoojan/REFace), [VFace](https://github.com/Sanoojan/VFace), [VividFace](https://github.com/deepcs233/VividFace).

### Internal Architecture Links

[DEFINE](../PURPOSE.md), [DESIGN](../phases/SD_VIDEO_SYSTEM_DESIGN.md), [ROADMAP](../progress/SD_VIDEO_ROADMAP.md), [REPORT](../reports/SD_VIDEO_DEFINE_AND_DESIGN.md).
