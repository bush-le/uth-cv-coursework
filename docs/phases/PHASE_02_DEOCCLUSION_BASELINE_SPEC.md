# Phase 2: Facial De-occlusion Baseline Specification

- **Motivation/Background**: Formulates the architectural standards, composite reconstruction loss formulations, identity preservation constraints, and training protocol for the downstream facial de-occlusion network.
- **Purpose**: Serve as the technical contract for implementing `src/models/builder.py`, `src/models/deep/deocclusion.py`, `src/common/metrics.py`, and `src/pipelines/train.py`.
- **Overview Pipeline**: Paired triplet ingestion $(I_{\text{clean}}, I_{\text{occ}}, M)$ -> Mask-guided deep restoration network -> Composite loss computation (Hole L1 + Valid L1 + Perceptual LPIPS + ArcFace ID) -> Backpropagation -> Image restoration metric logging (PSNR / SSIM / LPIPS / ArcFace ID Sim).
- **Detailed Plan**: §1 Model Architectural Standards; §2 Forward Pass & Tensor Shapes; §3 Composite De-occlusion Loss Formulation; §4 Evaluation Metrics (Fidelity, Realism & Identity); §5 Hyperparameters & Optimization Schedule.
- **References**: `docs/PURPOSE.md`, `docs/coursework/TAXONOMY_AND_DATA_SPEC.md`, Yu et al. (Gated Convolution, ICCV 2019), Deng et al. (ArcFace, CVPR 2019), Zhang et al. (LPIPS, CVPR 2018).
- **Created**: 2026-11-05T18:30:00+07:00
- **Last Updated**: 2026-11-05T18:30:00+07:00

---

## 1. Model Architectural Standards

To ensure a balanced baseline that is computationally tractable on a single GPU yet capable of high-fidelity facial completion, we establish two architectural tiers:

### 1.1 Primary Modern Baseline: Mask-Guided GatedConv / U-Net
- **Architecture**: Deep U-Net with Gated Convolutions (or Partial Convolutions) and Dilated Residual Bottlenecks.
- **Key Mechanism**: Dynamic gating channels allow the network to distinguish between valid face pixels and missing/occluded regions, preventing color bleeding and boundary blurring.
- **Resolution**: $512 \times 512$ with multiscale skip connections preserving global facial symmetry and local skin pores.

### 1.2 Advanced Generative Benchmark: Conditional Diffusion / LaMa
- **Architecture**: Large Mask Inpainting (LaMa with Fast Fourier Convolutions) or Conditional Latent Diffusion Restorer.
- **Purpose**: Serves as the top-tier comparative benchmark for full receptive-field facial completion.

---

## 2. Forward Pass & Tensor Shapes

All facial de-occlusion models must adhere strictly to these contract signatures:

```python
def forward(
    self,
    occ_images: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Restores unoccluded facial portrait from occluded input and mask.

    Args:
        occ_images: Standardized RGB image tensor of shape (B, 3, H, W).
                    Expected dtype float32, normalized to [0.0, 1.0].
        masks: Single-channel binary occlusion mask of shape (B, 1, H, W).
               Expected dtype float32, where 1.0 indicates occluded hole to restore
               and 0.0 indicates unoccluded valid context.

    Returns:
        pred_restored: Restored RGB facial portrait of shape (B, 3, H, W).
                       Expected dtype float32 in range [0.0, 1.0].
    """
```

### 2.1 Composite Output Composition
To preserve pristine ground-truth information outside the occlusion mask, inference produces the composite face:
$$\hat{I}_{\text{comp}} = (1 - M) \odot I_{\text{occ}} + M \odot \hat{I}_{\text{pred}}$$

---

## 3. Composite De-occlusion Loss Formulation

Facial restoration requires balancing pixel fidelity, sharp high-frequency skin textures, and biometric identity preservation:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{hole}} \mathcal{L}_{\text{hole}} + \lambda_{\text{valid}} \mathcal{L}_{\text{valid}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{style}} \mathcal{L}_{\text{style}} + \lambda_{\text{id}} \mathcal{L}_{\text{id}}$$

### 3.1 Mask-Weighted $L_1$ Reconstruction Loss
- **Occluded Hole Loss**:
  $$\mathcal{L}_{\text{hole}} = \frac{1}{\|M\|_1 + \epsilon} \|M \odot (\hat{I}_{\text{pred}} - I_{\text{clean}})\|_1, \quad (\lambda_{\text{hole}} = 6.0)$$
- **Valid Context Loss**:
  $$\mathcal{L}_{\text{valid}} = \frac{1}{\|1 - M\|_1 + \epsilon} \|(1 - M) \odot (\hat{I}_{\text{pred}} - I_{\text{clean}})\|_1, \quad (\lambda_{\text{valid}} = 1.0)$$

### 3.2 Perceptual Loss ($\mathcal{L}_{\text{perc}}$)
Computed via pretrained VGG-19 feature activations across layers `relu1_2`, `relu2_2`, `relu3_4`, `relu4_4`:
$$\mathcal{L}_{\text{perc}} = \sum_{l} \frac{1}{N_l} \|\phi_l(\hat{I}_{\text{comp}}) - \phi_l(I_{\text{clean}})\|_1, \quad (\lambda_{\text{perc}} = 0.1)$$

### 3.3 Style Loss ($\mathcal{L}_{\text{style}}$)
Gram matrix matching on VGG feature maps penalizes blurry or unnatural skin textures:
$$\mathcal{L}_{\text{style}} = \sum_l \frac{1}{C_l^2} \|\mathcal{G}(\phi_l(\hat{I}_{\text{comp}})) - \mathcal{G}(\phi_l(I_{\text{clean}}))\|_1, \quad (\lambda_{\text{style}} = 120.0)$$

### 3.4 Identity Preservation Loss ($\mathcal{L}_{\text{id}}$)
Deep facial embedding cosine distance via ArcFace ensures the person's identity is strictly preserved:
$$\mathcal{L}_{\text{id}} = 1 - \frac{\mathcal{F}_{\text{ArcFace}}(\hat{I}_{\text{comp}}) \cdot \mathcal{F}_{\text{ArcFace}}(I_{\text{clean}})}{\|\mathcal{F}_{\text{ArcFace}}(\hat{I}_{\text{comp}})\|_2 \|\mathcal{F}_{\text{ArcFace}}(I_{\text{clean}})\|_2}, \quad (\lambda_{\text{id}} = 0.5)$$

---

## 4. Evaluation Metrics (Fidelity, Realism & Identity)

Validation and testing evaluate four complementary dimensions:

### 4.1 Peak Signal-to-Noise Ratio (PSNR)
$$\text{MSE} = \frac{1}{3HW} \|\hat{I}_{\text{comp}} - I_{\text{clean}}\|_2^2, \quad \text{PSNR} = 10 \log_{10}\left(\frac{1.0}{\text{MSE}}\right)$$
Target: $\ge 28.0\text{ dB}$ (Full image), $\ge 24.5\text{ dB}$ (Hole region $\text{PSNR}_{\text{hole}}$).

### 4.2 Structural Similarity Index (SSIM)
Evaluates luminance, contrast, and structural preservation with a Gaussian window:
$$\text{SSIM}(\hat{I}_{\text{comp}}, I_{\text{clean}}) \ge 0.88$$

### 4.3 Learned Perceptual Image Patch Similarity (LPIPS)
Measures human perceptual distance in deep feature space:
$$\text{LPIPS}(\hat{I}_{\text{comp}}, I_{\text{clean}}) \le 0.090$$

### 4.4 ArcFace Identity Cosine Similarity
$$\text{ID\_Sim} = \cos\left(\mathcal{F}_{\text{ArcFace}}(\hat{I}_{\text{comp}}), \mathcal{F}_{\text{ArcFace}}(I_{\text{clean}})\right) \ge 0.82$$

---

## 5. Hyperparameters & Optimization Schedule

- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 10^{-4}$).
- **Base Learning Rate**: $\eta = 2 \times 10^{-4}$ with Cosine Annealing scheduler (minimum lr $10^{-6}$, 5 warmup epochs).
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast(dtype=torch.float16)`).
- **Epochs**: 50 epochs on curated synthetic triplets ($8,000$ training triplets).
- **Checkpointing**: Full-state checkpointing saving `best_model.pt` (highest validation PSNR / lowest LPIPS composite score) and `last_model.pt`.
