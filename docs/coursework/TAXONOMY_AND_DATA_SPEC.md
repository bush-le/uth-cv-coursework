# Deepfake Manipulation Taxonomy & Hard-Sample Data Contract Specification

- **Motivation/Background**: Establishes exact manipulation categories, JSONL manifest schemas, hard-sample provenance fields, tensor shape contracts, and identity-disjoint split rules for the Stable Diffusion + `dinoViT` coursework project.
- **Purpose**: Prevent data leakage, silent RGB/BGR color inversion, and schema drift across Phase 1 (SD Synthesis), Phase 2 (`dinoViT` Hard-Sample Mining), and Phase 3 (Fine-Tuning).
- **Overview Pipeline**: Manipulation taxonomy -> `ImageMetadata` & `ImageSample` contract -> PyTorch tensor shapes -> Identity-disjoint directory & split hierarchy.
- **Detailed Plan**: §1 Diffusion Manipulation Taxonomy; §2 JSONL Manifest & Hard-Sample Schema (`ImageMetadata`); §3 Tensor Shape & Color Space Invariants; §4 Identity-Disjoint Partitioning & Hard-Sample Oversampling.
- **References**: `docs/PURPOSE.md`, `docs/coursework/LITERATURE_REVIEW.md`, `configs/dataset/deepfake_image.yaml`, `src/data/image_schema.py`.
- **Created**: 2026-11-04T08:16:00+07:00
- **Last Updated**: 2026-11-08T17:48:00+07:00

---

## 1. Diffusion Manipulation Taxonomy

| Label ID | Manipulation Type | Provenance Bucket (`data_source`) | Description & Forensic Characteristics |
| :---: | :--- | :--- | :--- |
| **0** | `real` | `baseline` | Unmodified real human facial portrait (`CelebA / FFHQ`). |
| **1** | `sd_text2img` | `synthetic` / `hard_sample` | Whole-face synthesis from text prompt $z_T \to z_0$; global diffusion textures. |
| **1** | `sd_inpaint` | `synthetic` / `hard_sample` | Localized facial region editing (eyes, mouth, expression) via spatial binary mask $M$. |
| **1** | `sd_img2img` | `synthetic` / `hard_sample` | Partial latent noise injection ($S \in [0.4, 0.75]$) and denoising; subtle identity/texture morphing. |

---

## 2. JSONL Manifest & Hard-Sample Schema (`ImageMetadata`)

Defined in `src/data/image_schema.py`:

```json
{
  "image_id": "fake_real_001_0",
  "label": 1,
  "manipulation_type": "sd_img2img",
  "prompt": "subtle facial manipulation, realistic skin texture",
  "source_real_id": "real_001",
  "file_path": "data/processed/synthetic_faces/fake_real_001_0.png",
  "bbox": [32, 32, 224, 224],
  "is_hard_sample": true,
  "baseline_score": 0.241,
  "data_source": "hard_sample"
}
```

- `is_hard_sample` (`bool`): Set to `True` in Phase 2 (`mine_hard_samples.py`) when a synthetic fake sample ($y=1$) achieves $P_{\text{baseline}}(\text{fake}) < \tau_{\text{bypass}}$ ($\tau_{\text{bypass}} = 0.50$), meaning it bypassed the Baseline `dinoViT` detector.
- `baseline_score` (`Optional[float]`): Exact probability $P(\text{fake}) \in [0.0, 1.0]$ output by the frozen Baseline `dinoViT`.
- `data_source` (`str`): Tracks whether the sample belongs to `"baseline"`, `"synthetic"` (rejected/caught by baseline), or `"hard_sample"` (bypassed baseline).

---

## 3. Tensor Shape & Color Space Invariants

| Component | Tensor Shape | Color Space | Value Range | Dtype |
| :--- | :--- | :--- | :--- | :--- |
| **`ImageSample.image`** | `(3, H, W)` | RGB | `[0.0, 1.0]` | `torch.float32` |
| **DataLoader Batch (`images`)** | `(B, 3, H, W)` | RGB | `[0.0, 1.0]` or ImageNet norm | `torch.float32` |
| **DataLoader Batch (`labels`)** | `(B,)` | Binary `{0, 1}` | `0` (Real), `1` (Fake) | `torch.int64` |
| **DataLoader Batch (`is_hard_sample`)** | `(B,)` | Boolean | `{False, True}` | `torch.bool` |
| **Inpainting Mask (`mask`)** | `(1, H, W)` | Binary | `{0.0, 1.0}` | `torch.float32` |
| **Detector Logits** | `(B, 1)` | Log-odds | `(-inf, +inf)` | `torch.float32` |

---

## 4. Identity-Disjoint Partitioning & Hard-Sample Oversampling

1. **Identity-Disjoint Invariant**:
   $$\forall x \text{ with } \text{source\_real\_id} = s \implies \text{Split}(x) = \text{Split}(s)$$
   Implemented in `DeepfakeImageDataModule._partition_splits()`: all synthetic and hard-sample variants derived from the same real person `source_real_id` are strictly grouped into the same split (`70% Train / 15% Val / 15% Test`).
2. **Hard-Sample Oversampling**:
   In the training split only, samples with `is_hard_sample=True` are oversampled by `hard_sample_oversample_factor` (default $2\times$) to accelerate decision-boundary adaptation during `dinoViT` fine-tuning.
