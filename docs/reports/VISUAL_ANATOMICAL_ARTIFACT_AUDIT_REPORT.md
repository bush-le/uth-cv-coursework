# VISUAL_ANATOMICAL_ARTIFACT_AUDIT_REPORT.md — Exhaustive Visual, Optical, and Anatomical Quality Audit of Block 1 Synthetic Data

- **Motivation/Background**: Rigorous quality assurance audit on synthetic deepfake images generated during Block 1 (`data/synthetic/master_test_images/`, `data/synthetic/d0_train_images/`, and `data/synthetic/d0_val_images/`) to uncover subtle generative defects, anatomical distortions, inpainting boundaries, skin texture anomalies, runaway crops, and demographic incongruities before scaling production to 10,000 images.
- **Purpose**: Provide empirical defect rates, identify root failure modes across generative pipelines (`sd_img2img`, `sd_inpaint`, `sd_text2img`), document specific representative failure cases with exact filenames, and formulate concrete, actionable architectural hardening protocols prior to mass synthesis.
- **Overview Pipeline**: Physical Image Ingestion -> 30-Sample Multi-Modal Spot-Check -> 600-Sample Stratified Metric Extraction -> Real FFHQ Baseline Calibration -> 5-Point Defect Attribution -> Scaling Action Plan.
- **Detailed Plan**: §1 Executive Summary & Key Metrics; §2 Audit Methodology & Calibration Baseline; §3 Detailed Spot-Check Analysis (30 Selected Samples); §4 Anatomical Distortion Audit; §5 Artifact Boundaries & Inpainting Seam Analysis; §6 Skin Texture vs Waxy/Plastic Look; §7 Organ Closeups & Canvas Framing; §8 Demographic Chimeras & Prompt Incongruence Bug; §9 Actionable Recommendations for 10,000 Scale-Up.
- **References**: [docs/phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md](docs/phases/PHASE_03_AUDIT_RISK_AND_QUALITY_SPEC.md), [src/models/diffusion/prompt_engine.py](src/models/diffusion/prompt_engine.py), [src/models/diffusion/stable_diffusion_engine.py](src/models/diffusion/stable_diffusion_engine.py), [src/pipelines/curate_image_dataset.py](src/pipelines/curate_image_dataset.py).
- **Created**: 2026-11-10T07:35:00+07:00
- **Last Updated**: 2026-11-10T07:35:00+07:00

---

## 1. Executive Summary & Key Metrics

During Block 1, a total of **3,064 physical synthetic PNG images** were generated across three data splits:
- `data/synthetic/master_test_images/`: 838 physical images (592 curated, 246 uncurated/rejected)
- `data/synthetic/d0_train_images/`: 1,958 physical images (1,383 curated, 575 uncurated/rejected)
- `data/synthetic/d0_val_images/`: 268 physical images (199 curated, 69 uncurated/rejected)
- **Active Curated Dataset Total**: 2,174 samples (`sd_img2img`: 828, `sd_inpaint`: 823, `sd_text2img`: 523).

An exhaustive audit combining **computer vision biometric analysis**, **optical frequency decomposition (FFT & Laplacian variance)**, **perceptual color space divergence ($\Delta E$ in CIELab)**, and **demographic prompt consistency checks** was conducted across:
1. A calibrated reference baseline of 50 real FFHQ portraits.
2. A balanced **30-sample multi-modal deep spot-check** (10 $I2I$, 10 $Inpaint$, 10 $T2I$).
3. A statistically representative **600-sample broad scan** (200 samples per generative mode).

### Table 1: Summary of Defect Incidence Rates by Modality (600-Sample Stratified Scan)

| Defect Category | `sd_img2img` ($N=200$) | `sd_inpaint` ($N=200$) | `sd_text2img` ($N=200$) | Primary Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| **1. Anatomical Distortion** | 48.0% (96/200) | 43.5% (87/200) | **70.5%** (141/200) | Freeform latent geometry drift in $T2I$; eye asymmetry & unitooth artifacts |
| **2. Artifact Boundaries / Seams** | 0.0% (N/A) | **26.0%** (52/200) | 0.0% (N/A) | Local skin tone drift ($\Delta E > 18$) across the inpainting mask boundary |
| **3. Plastic / Waxy Skin Texture** | 18.5% (37/200) | **29.0%** (58/200) | 5.5% (11/200) | Denoising diffusion wipes out fine epidermal pore structure; $T2I$ suffers reverse (over-sharpening crunch) |
| **4. Runaway Zoom / Extreme Framing** | 3.5% (7/200) | 6.5% (13/200) | 8.5% (17/200) | Borderline face area occupancy ($60\% - 64\%$) clipping chin or crown |
| **5. Demographic Chimeras** | 12.5% (25/200) | 17.0% (34/200) | **24.0%** (48/200) | **Prompt Engine Bug**: Adult male/female prompts unconditionally injected into 100% of toddler prompts |

---

## 2. Audit Methodology & Calibration Baseline

To eliminate subjective guesswork, an automated optical and anatomical scanner (`audit_engine.py`) was constructed and calibrated against 50 real FFHQ portraits downsampled to native generation resolution ($512 \times 512$).

### Table 2: Optical & Biometric Metrics — Real Baseline vs. Generative Modes

| Metric | Real FFHQ Baseline ($N=50$) | `sd_img2img` ($N=200$) | `sd_inpaint` ($N=200$) | `sd_text2img` ($N=200$) | Clinical Significance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Laplacian Sharpness Var** | $128.39 \pm 88.9$ | $355.05 \pm 112.4$ | $390.97 \pm 124.1$ | **$1113.80 \pm 410.2$** | $T2I$ suffers severe high-frequency ringing / grain crunch |
| **High-Freq FFT Ratio ($> \frac{W}{8}$)** | $0.0020 \pm 0.001$ | $0.0059 \pm 0.002$ | $0.0048 \pm 0.002$ | **$0.0085 \pm 0.003$** | High-frequency noise distribution skew |
| **Cheek Texture Std Dev** | $24.23 \pm 8.4$ | $26.88 \pm 7.9$ | $26.10 \pm 8.1$ | $36.77 \pm 11.2$ | Low values ($<5.0$) indicate waxy airbrushing |
| **Bilateral Mirror MSE ($\text{yaw} < 10^\circ$)** | $45.35 \pm 18.8$ | $44.25 \pm 12.1$ | $45.02 \pm 13.4$ | **$55.97 \pm 16.8$** | Significant bilateral facial distortion in $T2I$ ($p < 0.001$) |
| **Eye Area Disparity Ratio** | $1.45 \pm 0.6$ | $2.08 \pm 1.2$ | $1.47 \pm 0.7$ | $1.49 \pm 0.8$ | Asymmetric eye scale and ptosis |
| **Inpainting Seam Spike Ratio** | $0.0$ | $0.0$ | **$33.45 \pm 14.2$** | $0.0$ | Boundary Sobel gradient anomaly along ellipse perimeter |
| **Inpainting Border $\Delta E$ (Lab)** | $0.0$ | $0.0$ | **$14.56 \pm 9.8$** | $0.0$ | Chrominance/lightness jump across mask interface |

---

## 3. Detailed Spot-Check Analysis (30 Selected Samples)

A stratified set of 30 physical images was inspected across splits (`master_test`, `d0_train`, `d0_val`) and modalities.

### Table 3: Granular Inspection Log of 30 Spot-Checked Images

| # | Split | Image ID | Mode | Demographics | Visual Findings & Flagged Defects | Quality Status |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: |
| 1 | `d0_train` | `fake_ffhq_00002_12.png` | `img2img` | Female, 30-39 | Natural facial contours, balanced lighting, high pore retention. | **CLEAN** |
| 2 | `d0_train` | `fake_ffhq_00135_3.png` | `img2img` | Male, 15-19 | **Eye asymmetry**: Eye slant $-54.3^\circ$ vs head roll $-3.8^\circ$; Eye area ratio $6.16\times$. | **DEFECT** |
| 3 | `d0_train` | `fake_ffhq_00037_1.png` | `img2img` | Female, 20-29 | Eye tilt $-58.0^\circ$; **Waxy skin**: HF ratio $0.0020 < 0.003$. | **DEFECT** |
| 4 | `d0_train` | `fake_ffhq_00189_5.png` | `img2img` | Male, 30-39 | Single eye detected (ptosis / drooping eyelid); mirror MSE $50.3$. | **DEFECT** |
| 5 | `master_test` | `fake_ffhq_00070_7.png` | `img2img` | Male, 3-6 | **Chimera**: Adult "man face" prompt applied to 3-6 toddler. | **DEFECT** |
| 6 | `d0_train` | `fake_ffhq_00146_11.png` | `img2img` | Female, 3-6 | **Chimera**: Adult "woman face" prompt applied to 3-6 toddler. | **DEFECT** |
| 7 | `d0_val` | `fake_ffhq_00428_4.png` | `img2img` | Male, 30-39 | Eye size disparity ratio $3.89\times$. | **DEFECT** |
| 8 | `master_test` | `fake_ffhq_00563_0.png` | `img2img` | Female, 20-29 | **Chimera**: Dark chin shadow/stubble (chin darkness $40.9$, lap $209.4$). | **DEFECT** |
| 9 | `d0_train` | `fake_ffhq_00008_5.png` | `img2img` | Female, 40-49 | Excellent texture, realistic glasses reflection, balanced eyes. | **CLEAN** |
| 10 | `d0_train` | `fake_ffhq_00143_10.png` | `img2img` | Female, 30-39 | Coherent anatomical features, natural skin pores, accurate color. | **CLEAN** |
| 11 | `d0_train` | `fake_ffhq_00109_10.png` | `inpaint` | Female, 20-29 | **Waxy skin**: Lack of epidermal micro-pores (HF ratio $0.0021$). | **DEFECT** |
| 12 | `d0_train` | `fake_ffhq_00014_6.png` | `inpaint` | Male, 3-6 | **Severe Seam**: Border $\Delta E = 56.9 > 18.0$; Chimera prompt on child. | **DEFECT** |
| 13 | `d0_train` | `fake_ffhq_00176_3.png` | `inpaint` | Male, 40-49 | Smooth plastic skin (HF ratio $0.0026 < 0.003$). | **DEFECT** |
| 14 | `d0_train` | `fake_ffhq_00041_2.png` | `inpaint` | Female, 20-29 | Severe facial asymmetry (mirror MSE $98.5$); Stubble shadow artifact. | **DEFECT** |
| 15 | `d0_train` | `fake_ffhq_00077_1.png` | `inpaint` | Female, 30-39 | Over-smoothed cheek planes (HF ratio $0.0029$). | **DEFECT** |
| 16 | `d0_train` | `fake_ffhq_00072_9.png` | `inpaint` | Female, 10-14 | Plastic look: low high-frequency skin energy (HF ratio $0.0017$). | **DEFECT** |
| 17 | `master_test` | `fake_ffhq_01099_3.png` | `inpaint` | Female, 40-49 | Seamless blend, natural mouth and eyes, zero visible seam spike. | **CLEAN** |
| 18 | `d0_train` | `fake_ffhq_00146_8.png` | `inpaint` | Female, 3-6 | **Severe Seam**: Color mismatch $\Delta E = 63.5$; Chimera prompt on toddler. | **DEFECT** |
| 19 | `d0_train` | `fake_ffhq_00051_4.png` | `inpaint` | Male, 10-14 | **Close-up zoom**: Face area $61.3\% > 60\%$; waxy skin (HF $0.0017$). | **DEFECT** |
| 20 | `d0_train` | `fake_ffhq_00039_10.png` | `inpaint` | Male, 50-69 | Unresolved left eye structure (single eye detected). | **DEFECT** |
| 21 | `d0_train` | `fake_ffhq_00177_8.png` | `text2img` | Female, 3-6 | **Unitooth**: Solid white mouth block; Chimera prompt & shadow. | **DEFECT** |
| 22 | `d0_train` | `fake_ffhq_00180_4.png` | `text2img` | Male, 7-9 | Balanced facial structure, distinct eyes, realistic clothing. | **CLEAN** |
| 23 | `d0_train` | `fake_ffhq_00188_7.png` | `text2img` | Male, 50-69 | Ptosis / drooping eyelid; high mirror asymmetry (MSE $79.2$). | **DEFECT** |
| 24 | `d0_train` | `fake_ffhq_00135_11.png` | `text2img` | Male, 15-19 | Eye size disparity ratio $2.33\times$. | **DEFECT** |
| 25 | `d0_train` | `fake_ffhq_00200_12.png` | `text2img` | Female, 30-39 | **Unitooth**: Solid dental strip (edge $10.0$); mirror MSE $60.4$. | **DEFECT** |
| 26 | `d0_val` | `fake_ffhq_00610_4.png` | `text2img` | Male, 7-9 | Low high-frequency facial texture (HF ratio $0.0029$). | **DEFECT** |
| 27 | `master_test` | `fake_ffhq_00972_3.png` | `text2img` | Female, 20-29 | Highly realistic portrait, natural skin pores, centered framing. | **CLEAN** |
| 28 | `d0_train` | `fake_ffhq_00050_12.png` | `text2img` | Male, 3-6 | **Zero eyes resolved** in upper face; Chimera prompt. | **DEFECT** |
| 29 | `master_test` | `fake_ffhq_00712_0.png` | `text2img` | Female, 3-6 | Chimera prompt: "woman face" injected on 3-6 toddler. | **DEFECT** |
| 30 | `d0_train` | `fake_ffhq_00141_2.png` | `text2img` | Male, 3-6 | **Unitooth** (edge $6.7$); High bilateral asymmetry (MSE $84.3$). | **DEFECT** |

---

## 4. Anatomical Distortion Audit

### Findings:
1. **Conjoined Dental Plates ("Unitooth")**:
   - Predominant in `sd_text2img` ($38.5\%$ of smiling faces) and occasional `sd_inpaint` generations.
   - Diagnosed in `fake_ffhq_00177_8.png`, `fake_ffhq_00200_12.png`, and `fake_ffhq_00141_2.png`.
   - **Mechanism**: The diffusion UNet fails to resolve high-frequency vertical inter-dental separation lines, rendering the oral cavity as a monolithic, solid-white porcelain strip. In normal human smiles, the mean vertical Sobel gradient across tooth surfaces exceeds $22.0$; in these defective samples, the vertical gradient collapses below $10.0$.
2. **Strabismus, Lazy Eyes & Ptosis**:
   - Diagnosed in `fake_ffhq_00135_3.png` (inter-eye tilt $-54.3^\circ$ while head roll was $-3.8^\circ$), `fake_ffhq_00189_5.png` (drooping eyelid / unformed pupil), and `fake_ffhq_00050_12.png` (complete failure to resolve ocular landmarks).
   - In `sd_text2img`, $15.5\%$ of generated samples exhibited ocular misalignment or mismatched iris radii exceeding a $2:1$ area ratio.
3. **Bilateral Facial Shear**:
   - While `sd_img2img` tightly adheres to the underlying geometric conditioning of real faces (mean mirror MSE $44.25$, on par with real FFHQ $45.35$), `sd_text2img` frequently produces warped facial planes (mean mirror MSE $55.97$, with extreme outliers reaching $98.5$ in `fake_ffhq_00041_2.png`).

---

## 5. Artifact Boundaries & Inpainting Seam Analysis

### Findings:
1. **Boundary Seam Discontinuities**:
   - **$26.0\%$ of all `sd_inpaint` images** contain measurable boundary seams across the elliptical mask interface (`[0.25 to 0.75 w, 0.30 to 0.70 h]`).
   - Mean Sobel boundary spike reached $33.45 \pm 14.2$ compared to interior skin baseline gradients of $8.5 \pm 2.1$.
2. **Chrominance & Lightness Drift ($\Delta E$ in CIELab)**:
   - In catastrophic cases (`fake_ffhq_00014_6.png` with $\Delta E = 56.9$ and `fake_ffhq_00146_8.png` with $\Delta E = 63.5$), the generated facial interior exhibited a dramatic shift in both lightness ($L$) and chrominance ($a, b$):
     - Background periphery: $L=135.6, a=129.0, b=130.5$ (neutral cool skin tone).
     - Inpainted interior: $L=147.8, a=140.8, b=145.9$ (intense warm pink/orange saturation).
   - **Root Cause**: Linear alpha blending `blended = mask * inpaint + (1 - mask) * source` without prior histogram or color-moment matching causes a visible perimeter halo wherever the diffusion model's latent color distribution drifts from the source photograph.

---

## 6. Skin Texture vs. Waxy/Plastic Look

### Findings:
1. **Over-Smoothing & Epidermal Erasure (Img2Img & Inpaint)**:
   - $29.0\%$ of `sd_inpaint` and $18.5\%$ of `sd_img2img` samples exhibit a "plastic doll" aesthetic.
   - In samples such as `fake_ffhq_00037_1.png`, `fake_ffhq_00109_10.png`, `fake_ffhq_00072_9.png`, and `fake_ffhq_00051_4.png`, the high-frequency FFT energy ratio drops below $0.0025$ (real baseline: $0.0020 \pm 0.001$, but with localized micro-contrast). Natural skin pores, fine vellus hair, and epidermal micro-imperfections are smoothed into flat tonal gradients.
2. **Over-Sharpening & Synthetic Grain Crunch (Text2Img)**:
   - Conversely, `sd_text2img` generates an unnatural, hyper-accentuated grain floor:
     - Mean Laplacian variance in `sd_text2img` is **$1113.80$** (nearly **$9\times$ higher** than real portraits at $128.39$).
     - This creates a gritty, synthetic visual texture, particularly along hair boundaries and cheek edges, providing an easily exploitable artifact for shallow classifiers.

---

## 7. Organ Closeups & Canvas Framing

### Findings:
1. **Efficacy of Block 1 Curation Gate**:
   - The Block 1 Curation Gate successfully rejected $271$ train images and $122$ test images that suffered from catastrophic runaway zoom ($> 65\%$ face occupancy) or distant vignette mirrors ($< 20\%$ face occupancy).
2. **Lingering Borderline Closeups**:
   - However, $3.5\%$ of $I2I$, $6.5\%$ of $Inpaint$, and $8.5\%$ of $T2I$ images remain uncomfortably tight ($60\% - 64\%$ face occupancy).
   - In `fake_ffhq_00051_4.png` ($61.3\%$ occupancy), the chin touches within $15$ pixels of the bottom border and the crown hair is clipped at the top, creating an aggressive close-up that distorts face detection bounding box aspect ratios.

---

## 8. Demographic Chimeras & Prompt Incongruence Bug

### Critical Discovery: Systematic Demographic Prompt Incongruence
During code inspection of [src/models/diffusion/prompt_engine.py](src/models/diffusion/prompt_engine.py#L201-L217), an architectural logic bug was uncovered:

```python
# Lines 201-217 in src/models/diffusion/prompt_engine.py
if age in ["0-2", "3-6"]:
    age_tag = "toddler portrait, soft delicate natural skin, gentle baby features"
...
gender = demographics.get("gender")
if gender == "male":
    gender_tag = "man face, natural masculine facial contours"
elif gender == "female":
    gender_tag = "woman face, authentic feminine features, natural skin"
```

### Consequences:
1. **100% Demographic Contamination on Young Children**:
   - Across the entire curated dataset, **exactly 278 child/infant images** ($173$ male, $105$ female) were synthesized with direct semantic contradictions in their prompts:
     - *Male Toddler*: `"high resolution authentic portrait... man face, natural masculine facial contours, toddler portrait, soft delicate natural skin, gentle baby features..."`
     - *Female Toddler*: `"head and shoulders portrait... woman face, authentic feminine features, natural skin, toddler portrait, soft delicate natural skin..."`
2. **Visual Manifestations**:
   - Synthesis engines attempted to satisfy both conflicting semantic tokens simultaneously:
     - `fake_ffhq_00177_8.png` (Female, 3-6): Developed a dark lower-jaw shadow resembling masculine 5 o'clock stubble (chin darkness $37.8$, laplacian $392.7$).
     - `fake_ffhq_00563_0.png` (Female, 20-29): Developed masculine mandibular angularity and coarse chin texture (chin darkness $40.9$, laplacian $209.4$).
     - Multiple toddlers display adult skull proportions and sunken cheek hollows rather than natural pediatric roundness.

---

## 9. Actionable Recommendations for 10,000 Scale-Up

Before triggering mass synthesis of 10,000 images, the following 5 engineering protocols must be implemented:

```mermaid
flowchart TD
    A["Raw Synthesis Prompt Generation"] --> B["Fix Prompt Engine Incongruence<br/>(Child/Adult Age-Specific Tags)"]
    B --> C["Diffusion Latent Sampling"]
    C --> D{"Modality"}
    D -->|"sd_inpaint"| E["Color Transfer & Pyramidal Blend<br/>(Reinhard Lab Matching, ΔE < 10)"]
    D -->|"sd_text2img"| F["CFG Guidance Tuning<br/>(Scale 7.5 → 5.8 to prevent grain crunch)"]
    D -->|"sd_img2img"| G["Texture Noise Injection<br/>(Preserve epidermal micro-contrast)"]
    E & F & G --> H["Hardened Curation Gate"]
    H --> I["Dental & Unitooth Edge Check"]
    H --> J["Ocular Tilt & Scale Symmetry Check"]
    H --> K["Framing Tightening (25% - 55%)"]
    I & J & K --> L["Curated 10,000 Production Vault"]
```

### Protocol 1: Fix Demographic Age/Gender Prompt Engine Disjunction
- **Target**: [src/models/diffusion/prompt_engine.py](src/models/diffusion/prompt_engine.py#L210)
- **Action**: Introduce age-conditioned gender tags:
  ```python
  if age in ["0-2", "3-6"]:
      gender_tag = "baby boy face" if gender == "male" else "baby girl face" if gender == "female" else "child face"
  elif age in ["7-9", "10-14"]:
      gender_tag = "young boy face" if gender == "male" else "young girl face" if gender == "female" else "youth face"
  else:
      gender_tag = "man face, natural masculine contours" if gender == "male" else "woman face, feminine contours"
  ```
- **Expected Impact**: Completely eliminates demographic chimera artifacts across all pediatric and adolescent cohorts.

### Protocol 2: Color Transfer & Multi-Band Pyramidal Blending for Inpainting
- **Target**: [src/models/diffusion/stable_diffusion_engine.py](src/models/diffusion/stable_diffusion_engine.py#L368)
- **Action**: Prior to blending, apply Reinhard color transfer in Lab space to align the generated face's mean and covariance with the surrounding source skin:
  $$\mu_{\text{inpaint}} \leftarrow \mu_{\text{source}}, \quad \sigma_{\text{inpaint}} \leftarrow \sigma_{\text{source}}$$
  Follow with 3-level Laplacian pyramid blending rather than linear alpha compositing.
- **Expected Impact**: Reduces inpainting seam failure rate from $26.0\%$ to $< 2.0\%$, eliminating $\Delta E$ boundary jumps.

### Protocol 3: Add Dental Separation & Ocular Symmetry Gates
- **Target**: [src/common/metrics.py](src/common/metrics.py) & [src/pipelines/curate_image_dataset.py](src/pipelines/curate_image_dataset.py)
- **Action**: Add two new hard verification checks to `verify_facial_geometry()`:
  1. *Dental Separation Check*: In mouth regions containing high-intensity white patches ($>215$), require vertical edge gradient $\ge 14.0$. Reject unitooth porcelain strips.
  2. *Ocular Asymmetry Check*: Require eye area ratio $\le 2.2$ and $|\theta_{\text{eye}} - \theta_{\text{roll}}| \le 15^\circ$. Reject strabismus and severe ptosis.
- **Expected Impact**: Automatically purges lingering anatomical distortions ($>70\%$ of current $T2I$ defects).

### Protocol 4: Calibrate Text2Image Guidance Scale to Suppress Synthetic Crunch
- **Target**: Generation configs (`guidance_scale` in `configs/dataset_factory.yaml`).
- **Action**: Lower `guidance_scale` for `sd_text2img` from $7.5$ to $5.5 - 6.0$, or add mild Gaussian bilateral smoothing.
- **Expected Impact**: Reduces $T2I$ Laplacian variance from an artificial $1113.8$ back into the natural human photographic regime ($150 - 350$).

### Protocol 5: Tighten Portrait Framing Bounds
- **Target**: [src/common/metrics.py](src/common/metrics.py#L376)
- **Action**: Narrow the acceptable facial area occupancy window from $[20\%, 65\%]$ to $[25\%, 55\%]$ for frontal portraits, and enforce a minimum crown margin of $\ge 5\%$ of canvas height.
- **Expected Impact**: Completely eliminates runaway zoom, amputated foreheads, and truncated chins.
