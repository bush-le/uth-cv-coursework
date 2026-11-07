# DEFINE — Stable Diffusion Video Deepfake Dataset

- **Motivation/Background**: Nhóm cần khóa phạm vi video SD-only trước benchmark pretrained và sinh dữ liệu.
- **Purpose**: Định nghĩa bài toán, users, input/output, features, constraints và success criteria.
- **Overview Pipeline**: Data integrity -> Pretrained comparison -> Selection -> Dataset -> Conditional detector study.
- **Detailed Plan**: §1 Project name; §2 Problem; §3 Users; §4 Outputs; §5 Features; §6 Constraints; §7 Evaluation.
- **References**: [Lifecycle](../.agents/AI_FIRST_LIFECYCLE.md), [Design](phases/SD_VIDEO_SYSTEM_DESIGN.md), [Protocol](experiments/SD_VIDEO_BASELINE_PROTOCOL.md).
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-07T14:21:07+07:00

---

[STATUS: ACTIVE]

## 1. Project name

**Xây dựng và đánh giá pipeline sinh dữ liệu video deepfake hoán đổi khuôn mặt bằng các phương pháp dựa trên Stable Diffusion.**

English: Building and Evaluating a Stable Diffusion-Based Video Face-Swapping Dataset Pipeline.

Nhánh sd-video bắt đầu từ main skeleton b6ba329. Đây là đặc tả theo trao đổi người dùng, không phải xác nhận rubric hoặc phê duyệt giảng viên. Source vẫn là scaffold; chưa tích hợp backend video hoặc có benchmark thực.

## 2. Problem statement

Cho ảnh nguồn S cung cấp danh tính và video đích V cung cấp tư thế, biểu cảm, chuyển động và bối cảnh. Sinh V' có danh tính S và giữ các thuộc tính đích, rồi lưu provenance và chất lượng thành dataset tái lập được.

**RQ1:** Trên cùng cặp nguồn–đích, các hệ thống pretrained dựa trên SD khác nhau thế nào về chuyển danh tính, bảo toàn thuộc tính, temporal consistency, failure rate và chi phí trên Kaggle?

**RQ2, giai đoạn sau:** Dataset được chọn có giúp detector phát hiện tốt hơn trên clip độc lập so với baseline và đối chứng cùng ngân sách train không?

Hoàn thành RQ1 trước RQ2. Không khẳng định sẵn model nhanh nhất hoặc dataset tốt hơn. Frame đẹp chưa chứng minh video ổn định hay dataset hữu ích.

## 3. Target users

Nhóm sinh viên chạy pipeline; người đánh giá kiểm tra demo/bằng chứng; nghiên cứu detector về sau dùng dataset trong giới hạn giấy phép. Không suy diễn thành ứng dụng real-time hoặc thương mại.

## 4. Intended output

- Một mẫu là một clip; frame và biến thể seed cùng clip không phải quan sát độc lập.
- Ảnh JPG/PNG RGB, một mặt rõ; video MP4 H.264 được khuyến nghị. Raw input bất biến.
- Pilot **5 giây, 10 FPS cố định, 50 frame**, một người, không chuyển cảnh. Ưu tiên video gốc 720p, mặt đủ rõ; upscale không phục hồi chi tiết thật đã mất.
- **512×512 là crop thử khi backend hỗ trợ**, không ép toàn video hoặc mọi backend cùng kích thước nội bộ.
- Output gồm generated video, liên kết originals, source/target identities, manifest, resolved config, revisions/hashes, seed, logs, metrics và QA.
- Backend yêu cầu chunk/frame count khác phải khai báo padding/context/cropping, actual processed frames và xử lý biên; kiểm tra canonical output. Không lấp frame lỗi bằng originals rồi gọi successful swap.
- Audio ngoài benchmark hình ảnh ban đầu; nếu mux lại phải kiểm tra đồng bộ và ghi metadata.

## 5. Core features

1. Data validation, manifests, pairing, grouped split và cache preprocessing theo backend.
2. Bốn adapter pretrained độc lập: Face-Adapter, REFace, VFace, VividFace.
3. Benchmark cùng input, lưu outputs và failures, tách cold/warm cost.
4. Development selection rồi holdout, khóa config, batch generation có resume và QA.
5. Dataset versioned; detector utility phase sau với gate riêng.

## 6. Technical constraints

Kaggle CUDA; ghi GPU/RAM/VRAM/quota theo phiên thực. Hai GPU không tự gộp VRAM. Pin môi trường riêng theo upstream khi xung đột dependency.

Chỉ generator dựa trên SD: không thêm Wan/GAN/LivePortrait vào registry. Detection/landmarks/recognition/flow là phụ trợ được phép. Public weights không đồng nghĩa unrestricted license: kiểm tra code, weights, auxiliaries, input và quyền output trước chia sẻ.

Pipeline script-only trong [src/pipelines](../src/pipelines/), notebook chỉ gọi code và xem output. Không train SD từ đầu. Fine-tune generator optional sau baseline, cần dữ liệu/training support/license/failure hypothesis; không mặc định mọi repo hỗ trợ.

MVP không nhiều người, video dài hoặc real-time. Không coi frame-by-frame là temporal model. Không loại mẫu khó để tăng score mà không báo coverage.

## 7. Success criteria

**Engineering:** job truy vết được, output đúng order/FPS/duration; errors rõ; production từ chối mock/ảnh đen fallback; tái chạy một sample từ config lưu sẵn.

**RQ1:** benchmark trên cùng pairs với ID/pose/temporal, QA, cost và failure/coverage; backend không khả thi phải báo, không loại âm thầm. Không yêu cầu cả bốn đều chạy được trên phần cứng chưa kiểm chứng.

**Selection:** hiệu chỉnh QA/config trên development và khóa trước holdout. Dùng Pareto chất lượng–chi phí, đặt ngưỡng vận hành sau pilot; không tự gán điểm ID/latency chưa đo.

**RQ2:** đối chứng trên test độc lập có evidence; kết quả không cải thiện vẫn hợp lệ. Không bắt buộc đạt AUC đặt tùy ý.

---

## References & Technical Documentation

### Official API Docs

[Diffusers](https://huggingface.co/docs/diffusers/index), [Torchvision](https://docs.pytorch.org/vision/stable/index.html).

### Technical Standards

[JSON Lines](https://jsonlines.org/), [FFmpeg](https://ffmpeg.org/ffmpeg.html).

### CV Conventions

[Face-Adapter](https://github.com/FaceAdapter/Face-Adapter), [REFace](https://github.com/Sanoojan/REFace), [VFace](https://github.com/Sanoojan/VFace), [VividFace](https://github.com/deepcs233/VividFace).

### Internal Architecture Links

[DESIGN](phases/SD_VIDEO_SYSTEM_DESIGN.md), [PROTOCOL](experiments/SD_VIDEO_BASELINE_PROTOCOL.md), [ROADMAP](progress/SD_VIDEO_ROADMAP.md), [REPORT](reports/SD_VIDEO_DEFINE_AND_DESIGN.md).
