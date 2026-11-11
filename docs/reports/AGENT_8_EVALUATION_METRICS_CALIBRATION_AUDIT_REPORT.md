# AGENT_8_EVALUATION_METRICS_CALIBRATION_AUDIT_REPORT.md — Evaluation Metrics, Calibration & Dual-Benchmark Audit Report

- **Motivation/Background**: Audits the quantitative evaluation pipelines, metrics implementations, temperature calibration dynamics, and dual-benchmark data integrity across active learning cycles to ensure rigorous, unpolluted evaluation.
- **Purpose**: Provide a comprehensive mathematical, architectural, and data-integrity audit of `src/pipelines/evaluate_image_benchmark.py`, `src/models/evaluators/calibrated_mining.py`, `src/common/metrics.py`, and `experiments/logs/`, with actionable remediation for the 10,000-sample scale-up.
- **Overview Pipeline**: Audit Ingestion -> Dual-Benchmark Leakage Audit -> Calibration & Overconfidence Analysis -> Metric Formulation & Edge-Case Audit -> Scale-Up 10k Architectural Recommendations.
- **Detailed Plan**: §1 Executive Audit Verdict; §2 Dual-Benchmark Protocol & Data Contamination Audit; §3 Calibration & Temperature Scaling Mechanics; §4 Bypass Rate & Decision Threshold Dynamics; §5 Metric Implementation Correctness & Edge-Case Analysis; §6 10,000 Scale-Up Recommendations.
- **References**: [GEMINI.md](GEMINI.md), [evaluate_image_benchmark.py](src/pipelines/evaluate_image_benchmark.py), [calibrated_mining.py](src/models/evaluators/calibrated_mining.py), [metrics.py](src/common/metrics.py), [closed_loop_metrics.json](experiments/logs/closed_loop_metrics.json).
- **Created**: 2026-11-10T07:35:00+07:00
- **Last Updated**: 2026-11-10T07:35:00+07:00

---

## 1. 🛡️ Executive Audit Verdict

An exhaustive audit of the evaluation and calibrated mining ecosystem was conducted across the codebase and experimental logs.

| Audit Pillar | Current State | Risk Level | Primary Root Cause |
|---|---|---|---|
| **1. Dual-Benchmark Protocol** | Master Vault is frozen & balanced; Dynamic Challenge Vault has 0 reals and suffered test-identity leakage. | **CRITICAL** | Silent fallback to unpartitioned `data/raw/manifest.jsonl` contaminated training pool with test identity `ffhq_00059`. Single-class challenge vault degenerates metrics. |
| **2. Temperature Calibration** | Fitted $T=3.16$ (R1) and $T=2.17$ (R2) via validation NLL minimization. | **MODERATE** | ViT detector is severely overconfident. $T$ softens logits, but has mathematical zero-effect on FN selection at $\tau=0.5$; alters only the boundary uncertainty window. |
| **3. Threshold & Bypass Rate** | Fixed threshold $\tau=0.50$; Bypass Rate is mathematically $1 - \text{Recall}_{\text{fake}} = \text{FNR}$. | **MODERATE** | $\tau=0.50$ is sub-optimal under imbalanced priors ($3:1$ real:fake). Calibrating threshold via Youden's $J$ or EER on validation significantly reduces bypass rate. |
| **4. Metric Calculations** | Wilcoxon-Mann-Whitney ROC-AUC lacks mid-rank tie handling; fallback defaults return $1.0$ on empty/single-class sets. | **HIGH** | Tied score ranking produces arbitrary $1.0$ or $0.0$; empty hard mask returns dummy $1.0$ hard recall; single-class ROC-AUC returns $1.0$ instead of `NaN`. |

---

## 2. 🔬 Dual-Benchmark Protocol & Data Contamination Audit

The evaluation strategy specifies a **Dual-Benchmark Protocol**:
1. **Master Frozen Test Vault** ($\mathcal{D}_{\text{test}}^{\text{frozen}}$): Anchor benchmark consisting of $N = 1,181$ unseen samples ($589$ real, $592$ fake), balanced $1:1$, with frozen identity partitions.
2. **Cumulative Challenge Test Vault** ($\mathcal{D}_{\text{test}}^{\text{challenge}}$): Dynamically accumulated benchmark collecting $15\%$ holdout slices of mined hard samples ($H_r^{\text{test}}$) across rounds.

### 2.1 Audit Finding 1.1: Identity Contamination in Candidate Synthesis (Critical)
- **Mechanism**: In `src/pipelines/run_closed_loop_active_learning.py` lines 370–371:
  ```python
  train_reals_path = Path("data/partitions/train_reals.jsonl")
  raw_manifest_path = train_reals_path if train_reals_path.exists() else Path("data/raw/manifest.jsonl")
  ```
  During the execution of active learning rounds, candidate generation drew sequentially from the unpartitioned `data/raw/manifest.jsonl` (which begins at `ffhq_00000`, `ffhq_00001`, ..., `ffhq_00059`).
- **Empirical Evidence**:
  - Out of 60 generated candidates per round:
    - **42** originated from `train_reals.jsonl`
    - **11** originated from `val_reals.jsonl` (`ffhq_00023`, `ffhq_00027`, `ffhq_00036`, `ffhq_00054`)
    - **7** originated from `test_reals.jsonl` (`ffhq_00059`)
  - In Round 2, `fake_ffhq_00059_q0_18` (derived from test identity `ffhq_00059`) was mined as a hard False Negative ($p = 0.3057$).
  - `split_hard_samples` placed `fake_ffhq_00059_q0_18` into `h_train`, appending it to `data/benchmarks/D2_hard_boosted/train.jsonl`.
  - Detector M2 was retrained on this sample and evaluated on `data/benchmarks/D0_baseline/test.jsonl`, which contains the exact same real subject `ffhq_00059` and 4 test fakes (`fake_ffhq_00059_0` through `3`).
- **Impact**: Violates strict identity-disjointness between training and test sets.

### 2.2 Audit Finding 1.2: Single-Class Degeneracy in Cumulative Challenge Vault (Critical)
- **Mechanism**: The cumulative challenge vault (`data/active_learning/cumulative_challenge_test.jsonl`) contains strictly synthetic deepfakes ($N_{\text{fake}} = 2$, $N_{\text{real}} = 0$).
- **Impact**:
  - Evaluating ROC-AUC requires ranking positive against negative instances:
    $$\text{AUC} = \frac{1}{N_{\text{pos}} N_{\text{neg}}} \sum_{i \in \text{pos}} \sum_{j \in \text{neg}} \left[ \mathbb{I}(s_i > s_j) + \frac{1}{2} \mathbb{I}(s_i = s_j) \right]$$
  - When $N_{\text{neg}} = 0$, the denominator is $0$, rendering ROC-AUC mathematically undefined.
  - Line 126 of `src/common/metrics.py` implements: `if n_pos == 0 or n_neg == 0: return 1.0`.
  - As a result, `closed_loop_metrics.json` recorded **ROC-AUC = 1.000** for Round 1 and Round 2 on the challenge vault, even when the model had **Accuracy = 0.0%** and **Bypass Rate = 100.0%**.
  - False Positive Rate (FPR) and specificity cannot be computed on a pure-positive vault.

---

## 3. 🌡️ Calibration & Temperature Scaling Mechanics

### 3.1 Mathematical Formulation of Temperature Scaling
For detector logit $z(x) \in \mathbb{R}$, temperature scaling maps $z(x) \mapsto z(x)/T$ for scalar $T > 0$:
$$\hat{p}_T(y=1|x) = \sigma\left(\frac{z(x)}{T}\right) = \frac{1}{1 + e^{-z(x)/T}}$$
Optimal $T^*$ minimizes the Negative Log-Likelihood (NLL) on validation logits:
$$T^* = \arg\min_{T > 0} -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \left[ y_i \ln \sigma(z_i / T) + (1 - y_i) \ln (1 - \sigma(z_i / T)) \right]$$

### 3.2 Audit of Fitted Temperatures ($T = 3.1617 \to T = 2.1681$)
1. **Severe Raw Logit Overconfidence**:
   - $T^* > 1.0$ proves that raw detector logits are overconfident.
   - For $T = 3.1617$, a raw logit $z = 4.0$ ($p_{\text{raw}} = 0.9820$) is softened to $z/T = 1.2651$ ($p_{\text{cal}} = 0.7799$).
   - For negative predictions, $z = -4.0$ ($p_{\text{raw}} = 0.0180$) softens to $p_{\text{cal}} = 0.2201$.
   - This occurs because standard cross-entropy loss with AdamW pushes unregularized classifier weights towards high norms, saturating logits even when true predictive confidence is lower.
2. **Trajectory Across Rounds**:
   - From Round 1 ($T = 3.1617$) to Round 2 ($T = 2.1681$), the required scaling factor decreased by $31.4\%$.
   - Incorporating mined hard samples with oversampling factor 2 and loss weight 2.0 forced the detector to penalize misclassified edge-cases, acting as an implicit logit regularizer and moderating overconfidence.

### 3.3 Theoretical Invariance: Why Calibration Does NOT Affect False Negative Mining at $\tau = 0.5$
A critical mathematical property of scalar temperature scaling is strict monotonicity:
$$\frac{\partial}{\partial z} \left[ \sigma\left(\frac{z}{T}\right) \right] = \frac{1}{T} \sigma\left(\frac{z}{T}\right) \left(1 - \sigma\left(\frac{z}{T}\right)\right) > 0 \quad \forall T > 0$$
Furthermore, for any $T > 0$:
$$\sigma\left(\frac{z}{T}\right) < 0.5 \iff \frac{z}{T} < 0 \iff z < 0 \iff \sigma(z) < 0.5$$
**Theorem**: The set of False Negative samples $\{x \in \mathcal{D}_{\text{fake}} \mid \hat{p}(x) < 0.5\}$ is **identically invariant** under temperature scaling for any $T > 0$.
Temperature scaling changes the numeric score, but does **not** change which samples cross the $0.5$ boundary.

### 3.4 Where Calibration Changes Behavior: The Boundary Uncertainty Window
Where temperature scaling has substantial impact is the second criterion:
$$\text{Is\_Uncertain} = |p_{\text{cal}} - \tau| \le \delta \iff 0.5 - \delta \le \sigma\left(\frac{z}{T}\right) \le 0.5 + \delta$$
At $\delta = 0.10$:
- In terms of calibrated probabilities: $p_{\text{cal}} \in [0.40, 0.60]$.
- In terms of raw logits: $z \in [-T \ln(1.5), T \ln(1.5)] \approx [-0.4055 \cdot T, 0.4055 \cdot T]$.
- In terms of uncalibrated probabilities:
  - If $T = 1.0$: $p_{\text{raw}} \in [0.400, 0.600]$ (window width = $0.200$)
  - If $T = 2.1681$: $p_{\text{raw}} \in [0.293, 0.707]$ (window width = $0.414$)
  - If $T = 3.1617$: $p_{\text{raw}} \in [0.217, 0.783]$ (window width = $0.566$)

```
Window Width Expansion under Temperature Scaling (delta = 0.10):
T = 1.00:  [--------- 0.40 ======== 0.60 ---------] (Width: 0.200)
T = 2.17:  [----- 0.293 ================= 0.707 -----] (Width: 0.414)
T = 3.16:  [--- 0.217 ======================= 0.783 ---] (Width: 0.566)
```
Consequently, an unstable temperature $T$ wildly modulates the volume of mined "uncertain" candidates.

---

## 4. 📐 Bypass Rate & Decision Threshold Dynamics

### 4.1 Equivalence to False Negative Rate (FNR)
In `src/common/metrics.py`:
$$\text{Bypass Rate}(\tau) = \frac{1}{N_{\text{fake}}} \sum_{i=1}^{N_{\text{fake}}} \mathbb{I}(p_i < \tau) = \frac{\text{FN}}{\text{FN} + \text{TP}} = 1 - \text{Recall}_{\text{fake}}(\tau) = \text{FNR}(\tau)$$
Experimental metrics confirm exact duality:
- Baseline M0: $\text{Recall} = 0.67568$, $\text{Bypass Rate} = 0.32432$ ($\text{Sum} = 1.00000$)
- Round 1 M1: $\text{Recall} = 0.68919$, $\text{Bypass Rate} = 0.31081$ ($\text{Sum} = 1.00000$)
- Round 2 M2: $\text{Recall} = 0.83277$, $\text{Bypass Rate} = 0.16723$ ($\text{Sum} = 1.00000$)

### 4.2 Sub-optimality of Fixed Threshold $\tau = 0.50$
The default threshold $\tau = 0.50$ assumes symmetric misclassification costs and balanced class distributions.
However, in deepfake detection:
1. **Asymmetric Risk**: Allowing a synthetic face to bypass detection ($\text{Cost}_{\text{FN}}$) is far more critical than an analyst reviewing a false positive flag ($\text{Cost}_{\text{FP}}$).
2. **Prior Class Imbalance**: Training D0 has $2,813$ real vs $1,383$ fake ($67\%$ real). The validation set has $598$ real vs $199$ fake ($75\%$ real).
Under Bayes Decision Theory, the minimum-risk threshold is:
$$\tau^* = \frac{\text{Cost}_{\text{FP}} \cdot P(Y=0)}{\text{Cost}_{\text{FP}} \cdot P(Y=0) + \text{Cost}_{\text{FN}} \cdot P(Y=1)}$$
At $\tau = 0.50$, the baseline detector operates in a high-precision ($97.56\%$), low-recall ($67.57\%$) regime, yielding an artificially elevated bypass rate ($32.43\%$).

### 4.3 Validation-Calibrated Operating Thresholds
Two rigorous alternatives should be adopted:
1. **Youden's $J$ Statistic Maximization (Operational Threshold)**:
   $$J(\tau) = \text{TPR}(\tau) - \text{FPR}(\tau) = \text{Recall}_{\text{fake}}(\tau) - \text{FPR}_{\text{real}}(\tau)$$
   Selects $\tau_J = \arg\max_\tau J_{\text{val}}(\tau)$ strictly on the validation set, then applies it frozen to test evaluation.
2. **Equal Error Rate (EER) Operating Point (Benchmark Standard)**:
   $$\tau_{\text{EER}} \text{ such that } |\text{FPR}(\tau_{\text{EER}}) - \text{FNR}(\tau_{\text{EER}})| \to 0$$
   Standard biometric benchmark metric (ISO/IEC 19795) providing threshold-independent performance verification.

---

## 5. 🧮 Metric Implementation Correctness & Edge-Case Analysis

### 5.1 Flaw 5.1: Ranking Ties in Wilcoxon-Mann-Whitney ROC-AUC (High Severity)
- **File**: `src/common/metrics.py` lines 128–130:
  ```python
  order = np.argsort(y_s)
  ranks = np.empty_like(order, dtype=np.float64)
  ranks[order] = np.arange(len(y_s), dtype=np.float64) + 1.0
  ```
- **Bug**: `np.argsort` assigns arbitrary strictly increasing integers $1, 2, \dots, N$ to tied score values based solely on original array indices.
- **Proof of Failure**:
  - `y_true = [0, 0, 1, 1]`, `y_score = [0.5, 0.5, 0.5, 0.5]` $\implies$ returns **AUC = 1.000**.
  - `y_true = [1, 1, 0, 0]`, `y_score = [0.5, 0.5, 0.5, 0.5]` $\implies$ returns **AUC = 0.000**.
- **Fix**: Mid-rank averaging using `scipy.stats.rankdata(y_s, method="average")` or grouping tied scores:
  ```python
  from scipy.stats import rankdata
  ranks = rankdata(y_s, method="average")
  ```

### 5.2 Flaw 5.2: Deceptive Single-Class Return Values (Medium Severity)
- **File**: `src/common/metrics.py` lines 125–126:
  ```python
  if n_pos == 0 or n_neg == 0:
      return 1.0
  ```
- **Bug**: Returning $1.0$ for single-class datasets corrupts evaluation logs (e.g., Round 1 and Round 2 challenge vault metrics recorded ROC-AUC = 1.000).
- **Fix**: Return `float("nan")`.

### 5.3 Flaw 5.3: Dummy Hard Sample Recall Return Value (Medium Severity)
- **File**: `src/common/metrics.py` lines 185–186:
  ```python
  target_hard_mask = (targs == 1) & hard_mask
  if not np.any(target_hard_mask):
      return 1.0
  ```
- **Bug**: When evaluating datasets lacking hard sample annotations (such as `D0_baseline/test.jsonl`), this function returns $1.0$, causing experimental summaries to falsely claim 100% hard sample recall.
- **Fix**: Return `float("nan")` when `target_hard_mask.sum() == 0`.

### 5.4 Flaw 5.4: Redundant Re-implementation of Metrics
- In `src/pipelines/evaluate_image_benchmark.py` lines 92–102, precision, recall, and F1 are reimplemented locally using scalar division checks, bypassing `src/common/metrics.py`. These implementations should be unified to guarantee DRY compliance.

---

## 6. 🚀 Mathematical & Engineering Blueprint for the 10,000 Scale-Up

When scaling candidate generation and mining from $N=60$ to $N=10,000$ per round:

### 6.1 Computational Complexity Bottlenecks & Vectorization
1. **Embedding Deduplication Loop**:
   - Current logic in `CalibratedHardMiner.mine_hard_samples` performs an $O(N \cdot M)$ Python sequential loop:
     ```python
     for seen_emb in selected_embs:
         if 1.0 - np.dot(curr_emb, seen_emb) < min_cosine_dist: ...
     ```
   - For $N=10,000$ and $M=2,000$ hard samples, this requires $20,000,000$ Python iterations.
   - **Remediation**: Implement batch matrix-vector multiplication with GPU acceleration or use FAISS `IndexFlatIP`:
     $$\mathbf{S} = \mathbf{E}_{\text{mined}} \mathbf{e}_{\text{cand}}^T \in \mathbb{R}^{M \times 1}$$
     $$\max(\mathbf{S}) > 1 - d_{\text{min}} \implies \text{reject}$$
2. **Probability Scaling Loop**:
   - Current `apply_temperature_to_probabilities` uses a Python loop with `math.log` and clamping.
   - **Remediation**: Vectorize directly over PyTorch tensors:
     ```python
     cal_probs = torch.sigmoid(logits / self.temperature).cpu().numpy()
     ```

### 6.2 Guaranteed Identity Isolation Protocol
- Remove fallback logic in `run_closed_loop_active_learning.py` lines 370–371.
- Introduce an explicit pre-flight contract check before synthesis begins:
  ```python
  train_reals = load_partition("data/partitions/train_reals.jsonl")
  train_ids = {r.image_id for r in train_reals}
  assert all(sample.source_real_id in train_ids for sample in candidate_tasks), \
      "CRITICAL: Identity leakage detected! Candidate drawn from outside training partition."
  ```

### 6.3 Balanced Challenge Vault Architecture
- To prevent single-class degeneracy, the Cumulative Challenge Vault must maintain a balanced distribution by:
  1. Pairing the $15\%$ mined hard fake slice with an equal holdout slice of **Hard Reals** (mined False Positives: real faces with highest $P(\text{fake})$).
  2. Or evaluating challenge fakes against a frozen pool of unseen test reals ($N_{\text{real}} = 589$).
  This guarantees that ROC-AUC, EER, and FPR remain well-defined and statistically valid.

---

## 7. 📋 Actionable Verification Checklist

- [x] Mathematics of Temperature Scaling verified ($T > 1 \implies$ overconfidence softening).
- [x] Mathematical proof of threshold-invariance for False Negatives at $\tau = 0.5$ established.
- [x] Identity leakage in Round 1 and Round 2 identified and traced to `data/raw/manifest.jsonl`.
- [x] Single-class ROC-AUC degeneracy identified in Cumulative Challenge Vault.
- [x] Ranking tie flaw in Wilcoxon-Mann-Whitney ROC-AUC identified and proven empirically.
- [x] Hard sample recall dummy fallback of $1.0$ documented.
- [x] 10,000 scale-up architectural recommendations delivered.
