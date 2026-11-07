# SD video ? Project overview

- **Motivation/Background**: C?n ??c t? SD-video nh?t qu?n tr??c benchmark pretrained.
- **Purpose**: T?ng h?p ph?m vi, t?i li?u, thi?t k? v? tr?ng th?i b?ng ch?ng.
- **Overview Pipeline**: DEFINE -> DESIGN -> Pretrained benchmark -> Selection -> Dataset -> Conditional detector.
- **Detailed Plan**: C?c m?c b?n d??i tr?nh b?y scope, ki?n tr?c, protocol, milestones v? gi?i h?n.
- **References**: [Lifecycle](../.agents/AI_FIRST_LIFECYCLE.md), [DEFINE](PURPOSE.md), [DESIGN](phases/SD_VIDEO_SYSTEM_DESIGN.md).
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-07T14:54:10+07:00

---

[STATUS: ACTIVE]

## 1. H??ng d? ?n

Benchmark pretrained SD video face swapping tr??c, ch?n c?u h?nh r?i sinh dataset. Fine-tune generator optional; detector utility phase sau. Kh?ng k? th?a image baseline code ho?c k?t qu? m? ph?ng.

## 2. T?i li?u chu?n

- [DEFINE](PURPOSE.md).
- [DESIGN](phases/SD_VIDEO_SYSTEM_DESIGN.md).
- [Protocol v? metrics hai phase](experiments/SD_VIDEO_BASELINE_PROTOCOL.md).
- [Milestones v? DoD](progress/SD_VIDEO_ROADMAP.md).
- [Report Markdown](reports/SD_VIDEO_DEFINE_AND_DESIGN.md) / [HTML tr?c quan](reports/SD_VIDEO_DEFINE_AND_DESIGN_SLIDES.html).

## 3. Tr?ng th?i

M0 t?i li?u ?? c?; M1?M5 implementation/benchmark/dataset ch?a ch?y. M6 detector c? ?i?u ki?n; M7 report th?c nghi?m ch?a c?. Tests generic c?a skeleton kh?ng ch?ng minh inference video ch?y ???c. Ch?a c? model chi?n th?ng ho?c t?c ?? ?? x?c minh.

## 4. Deliverables checklist

- [x] DEFINE, DESIGN, protocol v? roadmap SD-only.
- [ ] Data integrity, normalization v? grouped split.
- [ ] Feasibility b?n backend, c? log c? failures.
- [ ] Pilot comparison, QA v? holdout selection.
- [ ] Dataset versioned, provenance v? batch resume.
- [ ] Detector utility n?u v??t gate d? li?u/budget.
- [ ] B?o c?o th?c nghi?m v? demo t?i l?p.

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
