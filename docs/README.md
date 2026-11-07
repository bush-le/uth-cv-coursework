# Documentation index ? SD video

- **Motivation/Background**: C?n ??c t? SD-video nh?t qu?n tr??c benchmark pretrained.
- **Purpose**: T?ng h?p ph?m vi, t?i li?u, thi?t k? v? tr?ng th?i b?ng ch?ng.
- **Overview Pipeline**: DEFINE -> DESIGN -> Pretrained benchmark -> Selection -> Dataset -> Conditional detector.
- **Detailed Plan**: C?c m?c b?n d??i tr?nh b?y scope, ki?n tr?c, protocol, milestones v? gi?i h?n.
- **References**: [Lifecycle](../.agents/AI_FIRST_LIFECYCLE.md), [DEFINE](PURPOSE.md), [DESIGN](phases/SD_VIDEO_SYSTEM_DESIGN.md).
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-07T14:21:07+07:00

---

[STATUS: ACTIVE]

## 1. ??c theo th? t?

1. [PURPOSE / DEFINE](PURPOSE.md): t?n, problem, users, outputs, features, constraints v? criteria.
2. [SYSTEM DESIGN](phases/SD_VIDEO_SYSTEM_DESIGN.md): architecture, modules, dataflow, interfaces, milestones, risks.
3. [BASELINE PROTOCOL](experiments/SD_VIDEO_BASELINE_PROTOCOL.md): b?n repo SD, controls, quality/cost v? detector utility.
4. [ROADMAP](progress/SD_VIDEO_ROADMAP.md): milestones c? DoD v? tr?ng th?i th?t.
5. [REPORT](reports/SD_VIDEO_DEFINE_AND_DESIGN.md) / [HTML](reports/SD_VIDEO_DEFINE_AND_DESIGN.html): b?n t?ng h?p review.

## 2. Quy t?c

[.agents](../.agents/README.md) gi? governance generic; [docs](./) gi? knowledge d? ?n. Markdown UPPER_SNAKE_CASE, header 7 tr??ng, relative links v? status. Kh?ng c? s? ?o b?a ??t. Notebook ch? client g?i module/script v? xem k?t qu?.

## 3. Hi?n tr?ng

??c t? ch?a ph?i implementation. Ch?a c? output video, benchmark GPU, dataset ho?c fine-tuning tr?n nh?nh n?y. M?i metric target s? ???c kh?a t? development tr??c holdout, kh?ng d?ng k?t qu? gi? c?a h??ng c?.

---

## References & Technical Documentation

### Official API Docs

[Diffusers](https://huggingface.co/docs/diffusers/index), [PyTorch](https://docs.pytorch.org/docs/stable/index.html).

### Technical Standards

[JSON Lines](https://jsonlines.org/), [FFmpeg](https://ffmpeg.org/ffmpeg.html).

### CV Conventions

[Face-Adapter](https://github.com/FaceAdapter/Face-Adapter), [REFace](https://github.com/Sanoojan/REFace), [VFace](https://github.com/Sanoojan/VFace), [VividFace](https://github.com/deepcs233/VividFace).

### Internal Architecture Links

[DEFINE](PURPOSE.md), [DESIGN](phases/SD_VIDEO_SYSTEM_DESIGN.md), [PROTOCOL](experiments/SD_VIDEO_BASELINE_PROTOCOL.md), [ROADMAP](progress/SD_VIDEO_ROADMAP.md).
