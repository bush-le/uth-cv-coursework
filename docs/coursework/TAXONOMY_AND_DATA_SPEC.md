# Facial Occlusion Taxonomy & Triplet Data Contract Specification

- **Motivation/Background**: Establishes exact semantic class mappings, triplet data structures, tensor shape contracts, image formatting rules, and dataset directory hierarchies for synthetic diffusion generation and facial de-occlusion pipelines.
- **Purpose**: Prevent data corruption, label drift, silent RGB/BGR color inversion, and lossy compression artifacts across triplet generation and facial restoration models.
- **Overview Pipeline**: Semantic taxonomy -> Triplet data structure & formats -> PyTorch tensor shape contracts -> Directory hierarchy -> Data quality acceptance gates.
- **Detailed Plan**: §1 Semantic Occlusion Taxonomy; §2 Data Formats & Triplet Contract; §3 PyTorch & OpenCV Tensor Shape Contracts; §4 Directory Structure & Split Ratios; §5 Data Quality Acceptance Criteria.
- **References**: `docs/PURPOSE.md`, `docs/coursework/LITERATURE_REVIEW.md`, `configs/dataset/occlusion_synthetic.yaml`.
- **Created**: 2026-11-04T08:16:00+07:00
- **Last Updated**: 2026-11-05T18:30:00+07:00

---

## 1. Semantic Occlusion Taxonomy

In facial de-occlusion, occlusion categories dictate reconstruction difficulty (e.g., inpainting behind thin transparent eyeglasses vs large opaque surgical masks). Occlusions are categorized into 5 foreground classes:

### 1.1 Semantic Occlusion Categories

| Class ID | Semantic Label | Description / Typical Elements | De-occlusion Difficulty |
| :---: | :--- | :--- | :--- |
| **0** | `background_face` | Clean face skin, neck, eyes, lips, and background. | Ground-truth reference |
| **1** | `mask_protective` | Surgical masks, N95 masks, respirators, face shields. | Extreme (large missing area: nose, lips, chin) |
| **2** | `eyeglasses` | Sunglasses, reading glasses, tinted visors, goggles. | High (eye alignment, pupil recovery, reflections) |
| **3** | `hand_self` | Fingers, palms, fists resting on or covering facial skin. | High (skin-on-skin boundary, skin tone blending) |
| **4** | `held_object` | Microphones, mobile phones, cups, pens, cutlery. | Moderate (sharp geometric edges, localized hole) |
| **5** | `apparel_organic` | Wool scarves, high collars, loose hair strands across eyes. | High (complex textures, hairline recovery) |

---

## 2. Data Formats & Triplet Contract

Each dataset sample consists of a synchronized **Triplet $(I_{\text{clean}}, I_{\text{occ}}, M)$**:

1. **Clean Ground Truth Face ($I_{\text{clean}}$)**:
   - Format: `.png` or lossless `.jpg` ($\text{quality} \ge 95$).
   - Resolution: $512 \times 512$ pixels, 3 channels (RGB).
   - Role: Supervision target for reconstruction and identity preservation losses.
2. **Synthesized Occluded Face ($I_{\text{occ}}$)**:
   - Format: `.png` or lossless `.jpg` ($\text{quality} \ge 95$).
   - Resolution: $512 \times 512$ pixels, 3 channels (RGB).
   - Role: Model input to be restored.
3. **Occlusion Mask ($M$)**:
   - Format: **`.png` strictly (ZERO JPEG ALLOWED)**.
   - Resolution: $512 \times 512$ pixels, 8-bit single-channel grayscale (`uint8`).
   - Pixel Values: $\{0, 1\}$ for binary de-occlusion or $\{0, 1, 2, 3, 4, 5\}$ for multi-class conditioning.
   - Role: Conditional input mask or auxiliary segmentation supervision.
4. **Restored Prediction ($\hat{I}_{\text{clean}}$)**:
   - Normalized float32 during inference; saved as $512 \times 512 \times 3$ PNG for evaluation.

---

## 3. PyTorch & OpenCV Tensor Shape Contracts

All de-occlusion data loaders and model forward passes must enforce these tensor shape signatures:

| Stage | Variable Name | Shape | Dtype | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Disk Read (CV2)** | `img_clean_bgr` | `(512, 512, 3)` | `uint8` | $[0, 255]$ | Unoccluded reference |
| **Disk Read (CV2)** | `img_occ_bgr` | `(512, 512, 3)` | `uint8` | $[0, 255]$ | Occluded input |
| **Disk Mask Read** | `mask_raw` | `(512, 512)` | `uint8` | $\{0, \dots, 5\}$ | Ground-truth mask |
| **Model Target** | `clean_images` | `(B, 3, 512, 512)` | `float32` | $[0.0, 1.0]$ or $[-1.0, 1.0]$ | Normalized RGB target |
| **Model Input** | `occ_images` | `(B, 3, 512, 512)` | `float32` | $[0.0, 1.0]$ or $[-1.0, 1.0]$ | Occluded RGB input |
| **Condition Mask** | `masks` | `(B, 1, 512, 512)` | `float32` | $\{0.0, 1.0\}$ | Binary mask condition |
| **Mask-Guided Input** | `model_input` | `(B, 4, 512, 512)` | `float32` | Standardized | `torch.cat([occ, mask], dim=1)` |
| **Restored Output** | `pred_restored` | `(B, 3, 512, 512)` | `float32` | $[0.0, 1.0]$ or $[-1.0, 1.0]$ | Restored face $\hat{I}_{\text{clean}}$ |
| **Identity Features** | `id_embeds` | `(B, 512)` | `float32` | $L_2\text{-normalized}$ | ArcFace deep embedding |

---

## 4. Directory Structure & Split Ratios

```text
data/
├── raw/
│   ├── celeba_hq_clean/        # 10,000 clean unoccluded source faces
│   └── realocc_benchmark/      # Real-world benchmark test images & masks
├── synthetic_diffusion/
│   ├── clean/                  # Paired unoccluded ground-truth faces (I_clean)
│   │   ├── train/              # 8,000 clean images
│   │   └── val/                # 1,000 clean images
│   ├── occluded/               # Synthesized occluded faces (I_occ)
│   │   ├── train/              # 8,000 occluded images
│   │   └── val/                # 1,000 occluded images
│   └── masks/                  # Pixel-accurate occlusion masks (M)
│       ├── train/              # 8,000 integer masks (.png)
│       └── val/                # 1,000 integer masks (.png)
└── real_eval/
    ├── images/
    │   └── test/               # RealOcc / FaceOcc benchmark occluded test images
    └── masks/
        └── test/               # Benchmark occlusion ground-truth masks
```

---

## 5. Data Quality Acceptance Criteria

Every synthesized triplet $(I_{\text{clean}}, I_{\text{occ}}, M)$ must pass 4 programmatic quality gates before inclusion:

1. **Face Retention Gate**: Face detector (RetinaFace / YuNet) detects a face in $I_{\text{occ}}$ with confidence $\ge 0.85$.
2. **Prompt-Image Consistency Gate**: CLIP Score between $I_{\text{occ}}$ and the category prompt $\ge 0.28$.
3. **Mask Coverage Ratio Gate**: Occlusion area ratio $\frac{\text{Area}(M > 0)}{512 \times 512} \in [0.05, 0.65]$.
4. **Identity Invariance Outside Mask Gate**: ArcFace cosine similarity between $I_{\text{clean}}$ and $I_{\text{occ}}$ outside the masked region must be $\ge 0.88$, ensuring no identity drift occurs outside the synthetic occlusion.
