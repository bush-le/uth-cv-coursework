# Controlled Experiment Protocol: Proof of Synthetic Data Utility via dinoViT Hard-Sample Mining

- **Motivation/Background**: Formulates the scientific evaluation protocol, comparative ablation matrix, and statistical proof-of-utility criteria per the instructor's feedback: `Phase 1 SD Data -> Baseline dinoViT (Reject/Bypass -> Hard Sample) -> Fine-Tune on (Baseline + Syn + Hard Sample)`.
- **Purpose**: Provide an unambiguous, reproducible experimental harness that proves our Stable Diffusion synthetic dataset and hard-sample mining pipeline materially improve deepfake detection.
- **Overview Pipeline**: Baseline `dinoViT` evaluation -> Bypass / Hard-Sample extraction -> Progressive fine-tuning regimes -> Comparative evaluation (`compare_baseline_vs_finetuned`).
- **Detailed Plan**: §1 Central Scientific Hypothesis; §2 3-Stage Experimental Protocol; §3 Controlled Ablation Matrix; §4 Evaluation Metrics & Proof-of-Utility Criteria.
- **References**: `docs/PURPOSE.md`, `docs/coursework/LITERATURE_REVIEW.md`, `src/pipelines/mine_hard_samples.py`, `src/pipelines/evaluate_image_benchmark.py`.
- **Created**: 2026-11-04T08:22:00+07:00
- **Last Updated**: 2026-11-08T17:48:00+07:00

---

## 1. Central Scientific Hypothesis

> **Hypothesis**: While a baseline `dinoViT` detector rejects easy synthetic manipulations, a significant fraction of photorealistic Stable Diffusion facial manipulations (`sd_inpaint`, `sd_img2img`, `sd_text2img`) **bypass** the baseline decision boundary ($P(\text{fake}) < 0.50$). Mining these bypassed generations as **Hard Samples** and fine-tuning `dinoViT` on the combined **`Data Baseline + Synthetic Data + Hard Samples`** pool with hard-sample loss weighting ($\lambda_{\text{hard}} = 2.0$) enables the detector to successfully catch the bypassed fakes, substantially reducing the Bypass Rate and boosting overall ROC-AUC and Recall on unseen test splits.

---

## 2. 3-Stage Experimental Protocol

1. **Stage 1 — Synthetic DF Generation & Curation (`Phase 1`)**:
   - Generate synthetic facial manipulations from clean real faces using `StableDiffusionEngine`.
   - Filter through `ImageCurationGate` (Sharpness $\ge 50.0$, CLIP $\ge 0.28$, Face Confidence $\ge 0.85$) so that only realistic, artifact-free images enter the benchmark (preventing trivial anomalies from inflating detector accuracy).
2. **Stage 2 — Baseline `dinoViT` Evaluation & Hard-Sample Mining (`Phase 2`)**:
   - Evaluate the curated synthetic DF dataset using the **Baseline `dinoViT`** detector (`mine_hard_samples.py`).
   - Partition every synthetic DF image $x_i$ ($y_i = 1$) based on baseline prediction $p_i = P_{\text{base}}(\text{fake} \mid x_i)$:
     - $p_i \ge 0.50 \implies$ **Rejected (Caught)** $\rightarrow$ stored as `Regular Synthetic Data`.
     - $p_i < 0.50 \implies$ **Bypassed (Fooled Baseline)** $\rightarrow$ stored as **`Hard Sample`** (`is_hard_sample = True`).
3. **Stage 3 — Progressive Fine-Tuning & Comparative Audit (`Phase 3`)**:
   - Fine-tune `dinoViT` (`unfreeze_last_n_blocks=2`, `lr=2e-4`, `hard_sample_loss_weight=2.0`) and compare against the frozen/baseline `dinoViT` on an identity-disjoint held-out test set containing Real faces, Regular Synthetic Fakes, and Held-Out Hard Samples.

---

## 3. Controlled Ablation Matrix

| Exp ID | Model Configuration | Training Data Pool | Overall ROC-AUC | Bypass Rate ($\downarrow$) | Hard-Sample Recall ($\uparrow$) | Key Takeaway |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **EXP-1 (Baseline)** | `dinoViT` (Baseline) | `Data Baseline` only | Baseline reference | High (Vulnerable) | $0.0\%$ (Bypassed by definition) | Proves SD synthetic fakes can bypass baseline `dinoViT` |
| **EXP-2 (+Syn Data)** | `dinoViT` (Fine-tuned) | `Data Baseline + Syn Data` | Moderate Gain | Reduced | Moderate Recovery | Shows general utility of Phase 1 SD synthetic data |
| **EXP-3 (+Hard Samples)** | **`dinoViT` (Proposed Closed-Loop)** | **`Data Baseline + Syn Data + Hard Samples`** | **Highest ($\ge 92\%$)** | **Lowest ($< 10\%$)** | **High Recovery ($\ge 85\%$)** | **Proves full closed-loop utility (`chứng minh được tác dụng`)** |

---

## 4. Evaluation Metrics & Proof-of-Utility Criteria

Implemented in `src/common/metrics.py` and `src/pipelines/evaluate_image_benchmark.py`:
1. **ROC-AUC (`roc_auc`)**: Area under the Receiver Operating Characteristic curve via vectorized Wilcoxon-Mann-Whitney rank-sum test.
2. **Bypass Rate (`bypass_rate`)**:
   $$\text{Bypass Rate} = \frac{1}{|S_{\text{fake}}|} \sum_{x \in S_{\text{fake}}} \mathbb{I}\left[P(\text{fake} \mid x) < \tau\right]$$
3. **Hard-Sample Recall (`hard_sample_recall`)**:
   $$\text{Hard-Sample Recall} = \frac{\sum_{i} \mathbb{I}[\hat{y}_i = 1 \land y_i = 1 \land \text{hard}_i = \text{True}]}{\sum_{i} \mathbb{I}[y_i = 1 \land \text{hard}_i = \text{True}]}$$
4. **Proof of Utility (`compare_baseline_vs_finetuned`)**: Positive `roc_auc_gain > 0`, positive `hard_sample_recall_gain > 0`, and positive `bypass_rate_reduction > 0`.
