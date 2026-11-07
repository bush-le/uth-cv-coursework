# Roadmap — SD video milestones

- **Motivation/Background**: Mỗi milestone phải có deliverable và bằng chứng hoàn thành, không đánh complete từ tên file.
- **Purpose**: Theo dõi DEFINE/DESIGN, baseline benchmark, dataset và detector study có điều kiện.
- **Overview Pipeline**: M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> Conditional M6 -> M7.
- **Detailed Plan**: §1 Dependency/status; §2 Milestones/DoD; §3 Next tasks.
- **References**: [DEFINE](../PURPOSE.md), [DESIGN](../phases/SD_VIDEO_SYSTEM_DESIGN.md), [PROTOCOL](../experiments/SD_VIDEO_BASELINE_PROTOCOL.md).
- **Created**: 2026-11-07T14:12:25+07:00
- **Last Updated**: 2026-11-07T14:21:07+07:00

---

[STATUS: ACTIVE]

## 1. Dependency và trạng thái

M0 DEFINE/DESIGN -> M1 input integrity -> M2 backend feasibility -> M3 pretrained comparison -> M4 selection/holdout -> M5 dataset -> M6 optional detector -> M7 release.

Chưa có video backend triển khai trên nhánh này. Không gán deadline, tên thành viên, quota GPU hoặc metric targets chưa được cung cấp. Estimate sau pilot.

## 2. Milestones và Definition of Done

### M0 — DEFINE & DESIGN | Hoàn thành tài liệu, chưa thực nghiệm

Deliverables: [PURPOSE](../PURPOSE.md), [DESIGN](../phases/SD_VIDEO_SYSTEM_DESIGN.md), [PROTOCOL](../experiments/SD_VIDEO_BASELINE_PROTOCOL.md), [REPORT](../reports/SD_VIDEO_DEFINE_AND_DESIGN.md). DoD: scope SD-only/bốn ứng viên, data contracts, success criteria và gates nhất quán; format/links được kiểm tra. Baseline code/tests generic không chứng minh video inference.

### M1 — Input integrity | Chưa bắt đầu

Chọn input có quyền dùng; probe/normalize; schema/pairing; group split và cache keys. DoD: 3 cặp khác người, raw hashes, canonical 5s/10 FPS/50 frames, manifest và negative tests cho lỗi input/leakage.

### M2 — Backend feasibility | Chưa bắt đầu

Pin repo/checkpoint, env riêng, sample upstream rồi sample thực nhóm. DoD: mỗi ứng viên có output hợp lệ hoặc incompatibility/OOM log và cấu hình đã thử; licenses/preprocess contracts kiểm tra trước chia sẻ. Không hứa cả bốn chạy Kaggle từ README.

### M3 — Baseline comparison | Chưa bắt đầu

Pilot tối đa 12 videos, cùng pairs, upstream-default hợp lệ. DoD: gallery cạnh nhau, per-clip metrics/configs, cold/warm timing, coverage/failures và preliminary trade-off. Không suy rộng generalization từ ba pairs.

### M4 — Configuration selection & holdout | Chưa bắt đầu

Mở development theo budget, tuning cấu hình được hỗ trợ, khóa QA/operating tolerance rồi holdout. DoD: backend/config có evidence, runtime/VRAM thực đo, valid coverage và giới hạn. Fine-tune là nhánh optional có proposal riêng nếu cần.

### M5 — Batch dataset & QA | Chưa bắt đầu

Job state/resume, batch generation và export. DoD: dataset card/manifest hashes, mục tiêu đề xuất 100 clip QA-pass hoặc quy mô điều chỉnh có lý do; rejected/failure logs đầy đủ. Không đếm seed/frame như identities độc lập.

### M6 — Detector utility | Có điều kiện, chưa bắt đầu

Gate: M5, baseline real/fake, grouped split và independent test đủ dùng. DoD: D0/D1/D2 matched-budget, train-save-resume, clip metrics và uncertainty; thresholds chọn val, test không dùng tuning; công bố cả kết quả không cải thiện. Không yêu cầu fine-tune generator.

### M7 — Report/release | Chưa bắt đầu

DoD: versioned code/config/data references, env, logs, gallery và MD+HTML; một sample chạy lại được. Phân biệt generator quality với utility. Nếu chỉ hoàn thành RQ1, ghi RQ2 chưa đánh giá.

## 3. Tác vụ tiếp theo

1. Implement manifest/probe/normalize contracts và tests, chưa tải model.
2. Smoke một backend trên GPU rồi thêm dần; không chờ bốn backend tích hợp mới test end-to-end.
3. Cố định pairs và Kaggle client gọi script, outputs ngoài Git.

Phân task 30 phút–4 giờ theo workflow; không cam kết model runtime trước đo. Checkpoints generator là input immutable. Chỉ detector training phase sau cần optimizer/scaler/scheduler/RNG state nếu dùng, cùng test resume.

---

## References & Technical Documentation

### Official API Docs

[Diffusers](https://huggingface.co/docs/diffusers/index), [PyTorch](https://docs.pytorch.org/docs/stable/index.html).

### Technical Standards

[JSON Lines](https://jsonlines.org/), [FFmpeg](https://ffmpeg.org/ffmpeg.html).

### CV Conventions

[Face-Adapter](https://github.com/FaceAdapter/Face-Adapter), [REFace](https://github.com/Sanoojan/REFace), [VFace](https://github.com/Sanoojan/VFace), [VividFace](https://github.com/deepcs233/VividFace).

### Internal Architecture Links

[DEFINE](../PURPOSE.md), [DESIGN](../phases/SD_VIDEO_SYSTEM_DESIGN.md), [PROTOCOL](../experiments/SD_VIDEO_BASELINE_PROTOCOL.md), [REPORT](../reports/SD_VIDEO_DEFINE_AND_DESIGN.md).
