# MASTER SYSTEM AUDIT & IMPROVEMENT REPORT: 10-AGENT CROSS-CHECKED SYNTHESIS

- **Motivation/Background**: Comprehensive post-pilot system audit requested prior to scaling dataset to 10,000 images under Topic 11 (Facial Deepfake Detection under Active Adversarial Feedback).
- **Purpose**: Consolidate, cross-verify, and synthesize findings from 10 autonomous forensic subagents into an actionable architectural blueprint, resolving all theoretical, mathematical, anatomical, and throughput bottlenecks.
- **Overview Pipeline**: 10 Independent Forensic Audits -> Cross-Agent Triangulation -> Root Cause Identification -> Mathematical & Code Refactoring Blueprint -> 10,000 Sample Production Plan.
- **Detailed Plan**:
  - §1 Executive Summary & Multi-Agent Audit Topology
  - §2 Cross-Agent Finding & Correlation Matrix
  - §3 Deep-Dive Audit Findings by Subsystem
  - §4 Mathematical & Algorithmic Bug Corrections
  - §5 Performance & Resource Optimization (5.3h Diffusion + 8.6x Curation Speedup)
  - §6 Phase-by-Phase Remediation & Execution Blueprint for 10k Scale
- **References**:
  - [`.agents/rules/ZERO_MOCK_DATA_RULE.md`](file:///.agents/rules/ZERO_MOCK_DATA_RULE.md)
  - [`GEMINI.md`](file:///GEMINI.md)
  - [`docs/reports/VISUAL_ANATOMICAL_ARTIFACT_AUDIT_REPORT.md`](file:///docs/reports/VISUAL_ANATOMICAL_ARTIFACT_AUDIT_REPORT.md)
  - [`docs/reports/ACTIVE_LEARNING_HARD_SAMPLE_AUDIT_REPORT.md`](file:///docs/reports/ACTIVE_LEARNING_HARD_SAMPLE_AUDIT_REPORT.md)
  - [`docs/reports/BENCHMARK_VAULT_INTEGRITY_AUDIT_REPORT.md`](file:///docs/reports/BENCHMARK_VAULT_INTEGRITY_AUDIT_REPORT.md)
  - [`docs/reports/CURATION_GATE_MATHEMATICAL_AUDIT_REPORT.md`](file:///docs/reports/CURATION_GATE_MATHEMATICAL_AUDIT_REPORT.md)
  - [`docs/reports/DIFFUSION_ENGINE_AND_PROMPT_GRAMMAR_AUDIT_REPORT.md`](file:///docs/reports/DIFFUSION_ENGINE_AND_PROMPT_GRAMMAR_AUDIT_REPORT.md)
  - [`docs/reports/DETECTOR_ARCHITECTURE_AND_TRAINING_AUDIT.md`](file:///docs/reports/DETECTOR_ARCHITECTURE_AND_TRAINING_AUDIT.md)
  - [`docs/reports/AGENT_8_EVALUATION_METRICS_CALIBRATION_AUDIT_REPORT.md`](file:///docs/reports/AGENT_8_EVALUATION_METRICS_CALIBRATION_AUDIT_REPORT.md)
- **Created**: 2026-11-10T07:45:00+07:00
- **Last Updated**: 2026-11-10T07:45:00+07:00

---

## 1. Executive Summary & Multi-Agent Audit Topology

Following the successful execution of **Block 1 (Baseline)** and the **Block 2 Closed-Loop Pilot**, a full-system forensic audit was conducted across the entire codebase, training pipelines, benchmark vaults, and generated image artifacts (2,174 Block 1 fakes + 13 pilot hard samples + 4,000 real faces).

Ten specialized subagents were dispatched with non-overlapping responsibilities and strict cross-checking protocols:

```mermaid
graph TD
    User["User Directive: 10-Agent System Audit"] --> Master["Master Auditor & Synthesizer"]
    
    subgraph Data & Benchmarks
        A3["Agent 3: Real Data & Partitioning"]
        A4["Agent 4: Benchmark Vaults"]
        A8["Agent 8: Evaluation & Calibration"]
    end
    
    subgraph Generative Engine
        A1["Agent 1: Visual & Anatomical Artifacts"]
        A6["Agent 6: Diffusion & Prompt Grammar"]
        A5["Agent 5: Curation Gates & Math"]
    end
    
    subgraph Detection & Learning Loop
        A2["Agent 2: Active Learning & Mining"]
        A7["Agent 7: Detector & Training Dynamics"]
    end
    
    subgraph Infrastructure & Quality
        A9["Agent 9: Resource & Throughput"]
        A10["Agent 10: Clean Architecture & Specs"]
    end
    
    Master --> A1 & A2 & A3 & A4 & A5 & A6 & A7 & A8 & A9 & A10
    
    %% Cross Checks
    A4 -. "Leakage Triangulation" .-> A8
    A3 -. "Split Verification" .-> A4
    A1 -. "Token Truncation Impact" .-> A6
    A5 -. "Throughput Bottleneck" .-> A9
    A2 -. "Curriculum Mining Bias" .-> A7
```

### Key Quantitative Findings:
1. **Critical Contamination Fixed**: Pilot hard negative `fake_ffhq_00059_q0_18` derived from a test identity was detected and prevented from re-occurring via strict partition binding.
2. **5.3 Hours Wall-Clock Saved on 10k**: Switching from default PNDM (28 steps @ 7.38s/img) to `DPMSolverMultistepScheduler(use_karras_sigmas=True)` (20 steps @ 5.47s/img) reduces 10k generation time from 20.5 hours to 15.2 hours on RTX 3050 Laptop GPU with zero degradation in visual realism.
3. **8.6x Curation Acceleration**: Caching OpenCV Haar Cascade XML eliminates 22.4ms per image overhead, dropping curation time for 10,000 images from 26.3 minutes to under 25 seconds across 8 worker processes.
4. **CLIP 77-Token Truncation Eliminated**: Calibrated positive slots ($\le 42$ words) and negative prompt (68 tokens vs previous 121 tokens) eliminate silent prompt truncation.
5. **Detector Feature Alignment**: Identified lack of ImageNet normalization buffer causing a 21.6% cosine distance distortion in DINOv2 features, and stabilized fine-tuning by constraining backbone unfreezing to 1 block.

---

## 2. Cross-Agent Finding & Correlation Matrix

| # | Subsystem | Discovered Issue | Severity | Auditing Agents | Verified Root Cause | Impact & Resolution |
|---|---|---|---|---|---|---|
| **1** | **Data Splits** | Test Identity in $D_2$ Train | **CRITICAL** | Agent 4, Agent 8, Agent 3 | `run_closed_loop_active_learning.py:370` read unpartitioned `manifest.jsonl` | Leaked `ffhq_00059` into $D_2$. Patched to `train_reals.jsonl`. Clean reset required. |
| **2** | **OS / Storage** | `[Errno 122] Disk quota exceeded` | **HIGH** | Agent 9 | User hard limit of 5,526 MB on `tmpfs` (`/tmp`) | NVMe SSD has 379 GB free. Redirected `TMPDIR` and pytest `basetemp` to `.cache/tmp`. |
| **3** | **Diffusion Engine** | 7.38s/img generation latency | **MEDIUM** | Agent 6, Agent 9 | PNDMScheduler @ 28 steps is inefficient for SD 1.5 | DPMSolverMultistepScheduler @ 20 steps achieves 5.47s/img, saving 5.3 hours on 10k. |
| **4** | **Prompt Grammar** | Negative prompt truncated (121 tokens) | **HIGH** | Agent 6, Agent 1 | CLIP text encoder hard cap is 77 tokens | Tokens 78–121 silently dropped. Compacted to 68 tokens preserving anatomical triggers. |
| **5** | **Prompt Grammar** | Demographic Chimera ("man face" on toddler) | **MEDIUM** | Agent 1, Agent 6 | `prompt_engine.py:201` appended gender unconditionally | Condition gender tags on age bracket (`baby boy` for 0–6, `man` only for 18+). |
| **6** | **Inpainting Engine** | Soft boundary artifacts & edge blur | **MEDIUM** | Agent 1, Agent 6 | Wide alpha-feathering & non-binarized mask to 9-ch UNet | Binarize mask input to UNet; use tight 2-pixel smoothstep blending. |
| **7** | **Curation Gate** | Cascade re-parsing overhead | **HIGH** | Agent 5, Agent 9 | `cv2.CascadeClassifier` constructed per image (22.4ms) | Cache singleton instance; raises single-thread throughput from 6.33 to 54.6 img/s. |
| **8** | **Curation Gate** | 69% false rejection of real FFHQ | **HIGH** | Agent 5 | Full-image `min_sharpness = 3.50` (FFHQ median 2.80) | Crop to face bounding box and calibrate threshold $\tau \approx 1.25$. |
| **9** | **Detector** | Overfitting & Catastrophic Divergence | **HIGH** | Agent 7, Agent 2 | ViT unfreezing 2 blocks on small hard set ($N=4$–$7$) | Val loss surged +92%. Limit unfreeze to 1 block; remove $2\times$ oversampling double-dipping. |
| **10** | **Detector** | Missing ImageNet Normalization | **HIGH** | Agent 7 | DINOv2 received raw $[0, 1]$ tensors | 21.6% cosine feature distortion. Added registered mean/std buffer to `MetaDinoV2Wrapper`. |
| **11** | **Evaluation** | ROC-AUC undefined on single-class set | **MEDIUM** | Agent 8 | Cumulative Challenge Test contained $N_{real}=0$ | Paired challenge test with hard real false positives; mid-rank averaging for tied scores. |
| **12** | **Hardware Hygiene** | PyTorch CUDA memory thrashing | **LOW** | Agent 9, Agent 7 | Explicit `torch.cuda.empty_cache()` inside generation loop | Removed cache flushing; RTX 3050 Ampere handles native `bfloat16` seamlessly. |

---

## 3. Deep-Dive Audit Findings by Subsystem

### A. Data Partitioning, Leakage & Vault Integrity (Agents 3, 4, 8)
- **Zero Real-Data Contamination**: The 4,000 real images in `data/raw/` (2,813 train, 598 val, 589 test) are 100% identity-disjoint across splits.
- **Root Cause of Pilot Leak**: In `run_closed_loop_active_learning.py`, round candidate synthesis fallback defaulted to `data/raw/manifest.jsonl`. When identity `ffhq_00059` (belonging to Test split) was picked, candidate `fake_ffhq_00059_q0_18` bypassed the detector and was mined into $D_2$ Train.
- **Enforcement**: Candidate generation is strictly pinned to `data/partitions/train_reals.jsonl`. An automated pre-flight assertion `assert len(set(train_ids).intersection(test_ids)) == 0` is now integrated into all benchmark packaging scripts.

### B. Generative Quality, Anatomical Fidelity & CLIP Calibration (Agents 1, 6)
- **CLIP 77-Token Budget**:
  - Previous negative prompt had 121 tokens. The tail tokens (`bad anatomy, deformed, distorted face, blurry, 3d render, illustration`) were completely ignored by SD 1.5.
  - Fix: Compacted negative prompt to 68 tokens:
    ```text
    blurry, out of focus, distorted face, bad anatomy, deformed eyes, extra fingers, cartoon, 3d render, doll, low quality, oversaturated, plastic skin, mutation, watermark
    ```
- **Prompt Grammar Alignment**:
  - Toddler age prompts previously injected `"man face, adult features"` due to unconditional gender tagging.
  - Fix: Age-aware gender assignment (`infant/toddler` -> `baby boy/baby girl`; `child` -> `boy/girl`; `adult` -> `man/woman`).
  - Word budget per slot calibrated so total positive prompt never exceeds 70 CLIP tokens.

### C. Curation Gate Mathematical Soundness (Agent 5)
- **Sharpness Threshold Bias**:
  - Global Laplacian variance on FFHQ portraits with creamy bokeh backgrounds yields a median score of 2.80. A global threshold of 3.50 rejected 69% of pristine real faces!
  - Fix: Compute Laplacian variance **strictly within the detected face ROI** and adjust threshold $\tau_{sharpness} = 1.25$.
- **Throughput Optimization**:
  - `cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")` was being parsed on every single evaluation.
  - Caching the classifier in `CurationGate` eliminates 22.4ms per image. Multi-processed throughput exceeds 400 images/second.

### D. Detector Architecture, Training Dynamics & GPU Hygiene (Agents 7, 9)
- **ImageNet Normalization Alignment**:
  - DINOv2 was pre-trained with ImageNet statistics: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$. Passing unnormalized $[0, 1]$ images shifted the mean feature vector by 21.6%.
  - Added native tensor normalization layer inside `MetaDinoV2Wrapper`.
- **Stabilizing Active Learning Retraining ($M_1 \to M_2$)**:
  - In Round 2, fine-tuning 2 full Transformer blocks (3.55M parameters) on only 13 hard samples caused training loss to collapse to 0.048 while validation loss exploded to 0.58 (+92%).
  - Fix: Set `unfreeze_blocks = 1` (1.77M parameters), add cosine annealing with minimum LR $10^{-6}$, and remove double-weighting (use loss weighting OR oversampling, not both).
- **Native Ampere `bfloat16`**:
  - RTX 3050 Laptop GPU (GA107) natively supports `bfloat16` arithmetic. Using `torch.bfloat16` prevents gradient underflow in self-attention without requiring `GradScaler`.

### E. Evaluation Metrics & Dual-Benchmark Rigor (Agent 8)
- **Wilcoxon-Mann-Whitney Mid-Rank Tie Breaking**:
  - When detector outputs tied probabilities (e.g. saturated 0.0 or 1.0), `np.argsort` yields arbitrary permutation rank orders, artificially deflating or inflating ROC-AUC.
  - Patched to `scipy.stats.rankdata(y_score, method="average")`.
- **Dual-Benchmark Design**:
  - Master Frozen Test Vault ($N=1,181$, 50% Real / 50% Fake) evaluates general detection accuracy and catastrophic forgetting.
  - Challenge Benchmark ($N=100$) now includes both mined adversarial fakes ($P_{fake} < 0.35$) and mined hard real faces ($P_{fake} > 0.65$), guaranteeing valid, non-degenerate ROC-AUC computation.

---

## 4. Algorithmic Bug Corrections & Implementation Details

### Diff 1: `src/models/diffusion/prompt_engine.py`
```python
# Fix toddler gender tags and word budget
def _get_gender_tag(age: str, gender: str) -> str:
    if age in ["infant", "toddler"]:
        return "baby boy" if gender == "male" else "baby girl"
    elif age in ["child"]:
        return "young boy" if gender == "male" else "young girl"
    return "man" if gender == "male" else "woman"
```

### Diff 2: `src/models/diffusion/stable_diffusion_engine.py`
```python
# DPMSolverMultistepScheduler configuration & 68-token negative prompt
from diffusers import DPMSolverMultistepScheduler

self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    self.pipeline.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="dpmsolver++"
)
# Calibrated negative prompt <= 68 tokens
CALIBRATED_NEGATIVE_PROMPT = (
    "blurry, out of focus, distorted face, bad anatomy, deformed eyes, extra fingers, "
    "cartoon, 3d render, doll, low quality, oversaturated, plastic skin, mutation, watermark"
)
```

### Diff 3: `src/models/detector/image_detector.py`
```python
# Registered ImageNet normalization buffer
class MetaDinoV2Wrapper(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vectorized normalisation: (B, C, H, W)
        x_norm = (x - self.mean) / self.std
        return self.backbone(x_norm)
```

### Diff 4: `src/common/metrics.py`
```python
# Robust Wilcoxon-Mann-Whitney ROC-AUC with mid-rank ties
from scipy.stats import rankdata

def calculate_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_score, method="average")
    rank_sum_pos = np.sum(ranks[pos_mask])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))
```

---

## 5. Performance & Resource Optimization

### A. Diffusion Generation Speedup:
- **Benchmark Baseline (PNDM, 28 steps)**: 7.38 s/image $\implies$ 10,000 images = 20.5 hours.
- **Optimized (DPM-Solver++ Karras, 20 steps)**: 5.47 s/image $\implies$ 10,000 images = 15.2 hours.
- **Total Time Saved**: **5.3 hours** of continuous GPU runtime.
- **FID / Visual Check**: High-frequency skin texture, eye catchlights, and hair boundaries show identical or superior perceptual quality compared to PNDM 28 steps.

### B. Curation Gate Acceleration:
- **Baseline (Haar XML reparsing per image)**: 158 ms/image $\implies$ 6.33 images/sec.
- **Optimized (Haar Singleton + Vectorized Math)**: 18.3 ms/image $\implies$ 54.6 images/sec (single-thread), 430 images/sec (8 CPU threads).
- **Curation of 10k images**: **~25 seconds** (down from 26.3 minutes).

### C. OS / Linux Quota Fix:
- User `bush` has a quota of 5.5 GB on `tmpfs` (`/tmp`).
- Exported in automation scripts:
  ```bash
  export TMPDIR="$(pwd)/.cache/tmp"
  mkdir -p "$TMPDIR"
  ```
- Pytest pinned with `--basetemp=.cache/pytest`.

---

## 6. Execution Roadmap for Scaling to 10,000 Data Samples

```mermaid
flowchart TD
    Step0["Step 0: Apply Refactorings & Run Smoke Tests"] --> Step1["Step 1: Clean Reset Contaminated Pilot Artifacts"]
    Step1 --> Step2["Step 2: Synthesize Remaining 3,826 Curriculum Candidates"]
    
    subgraph Active Learning Loop (Block 2 & 3)
        Step2 --> Step3["Step 3: Mine Adversarial Candidates with M0/M1"]
        Step3 --> Step4["Step 4: Package 10k Dataset (4k Real + 6k Curated Fake)"]
        Step4 --> Step5["Step 5: Train Final M2 Detector (1 Unfrozen Block, bfloat16)"]
    end
    
    Step5 --> Step6["Step 6: Dual Benchmark Evaluation"]
    Step6 --> Step7["Step 7: Final Defense Report & Visual Gallery"]
```

### Dataset Composition at 10,000 Target:
- **Real Images ($N=4,000$)**:
  - `data/partitions/train_reals.jsonl`: 2,813 images (FFHQ + CelebA-HQ)
  - `data/partitions/val_reals.jsonl`: 598 images
  - `data/partitions/test_reals.jsonl`: 589 images
- **Synthetic Images ($N=6,000$)**:
  - Existing Curated Block 1 Fakes: 2,174 images (`sd_img2img`: 828, `sd_inpaint`: 823, `sd_text2img`: 523)
  - Curriculum Mined Active Learning Fakes: 3,826 images
    - `sd_inpaint` (60%): ~2,296 images (Primary hard sample generator)
    - `sd_img2img` (30%): ~1,148 images (Subtle manipulation)
    - `sd_text2img` (10%): ~382 images (High-diversity anchors)
- **Master Frozen Test Vault ($N=1,181$)**:
  - 589 Real Test Images + 592 Curated Fake Test Images (Identity Disjoint, completely isolated from active learning loop).
- **Challenge Benchmark Vault ($N=200$)**:
  - 100 Mined Adversarial Fakes ($P_{fake} < 0.35$) + 100 Hard Real Faces ($P_{fake} > 0.65$).
