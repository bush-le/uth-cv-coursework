# AUDIT_PREFLIGHT_SUBAGENT7_METRICS.md — Evaluation Metrics, Calibration & Reporting Deep Audit

- **Motivation/Background**: Rigorous pre-flight mathematical, numerical, and structural audit of evaluation metrics, probability calibration mechanisms, and visual reporting diagnostics prior to final 10,000-sample production execution.
- **Purpose**: Provide a deep mathematical verification of Wilcoxon-Mann-Whitney ROC-AUC tie handling, Bypass Rate / Hard-Sample Recall behavior, Expected Calibration Error (ECE) formulation, and reliability diagrams across `src/common/metrics.py`, `src/pipelines/evaluate_image_benchmark.py`, `src/common/visualizer.py`, and `tests/test_metrics.py`.
- **Overview Pipeline**: Ingestion of Evaluation Codebase $\to$ Mathematical & Numerical Verification of Ranking & Calibration $\to$ Edge-Case & Boundary Stress-Testing $\to$ Identification of Latent Vulnerabilities & Gaps $\to$ Production-Grade Code & Test Suite Recommendations.
- **Detailed Plan**: §1 Executive Audit Scorecard; §2 Deep Mathematical Audit of Wilcoxon-Mann-Whitney ROC-AUC; §3 Bypass Rate & Hard-Sample Recall Boundary Dynamics; §4 Expected Calibration Error (ECE) & Reliability Diagram Formulation; §5 Pipeline Evaluation (`evaluate_image_benchmark.py`) DRY & Numerical Hygiene; §6 Production Drop-in Implementations & Test Expansions.
- **References**: [GEMINI.md](GEMINI.md), [src/common/metrics.py](src/common/metrics.py), [src/pipelines/evaluate_image_benchmark.py](src/pipelines/evaluate_image_benchmark.py), [src/common/visualizer.py](src/common/visualizer.py), [src/models/detector/calibration.py](src/models/detector/calibration.py), [tests/test_metrics.py](tests/test_metrics.py), [tests/test_calibration.py](tests/test_calibration.py).
- **Created**: 2026-11-10T08:35:00+07:00
- **Last Updated**: 2026-11-10T08:35:00+07:00

---

## 1. 🛡️ Executive Audit Scorecard

An exhaustive numerical and mathematical audit of the evaluation, calibration, and visualization modules was conducted. Below is the executive synthesis of findings:

| Audit Domain | Target Component | Status | Risk Level | Primary Audit Finding |
|---|---|---|---|---|
| **1. ROC-AUC Tie-Breaking** | `compute_roc_auc` in `src/common/metrics.py` | **VERIFIED CORRECT** | **LOW** | Vectorized mid-rank tie breaking (`np.unique` + `cumsum`) mathematically matches exact pairwise Mann-Whitney $U$. All-tie case yields exactly $0.5000$. Single-class and length $\le 1$ return `np.nan`. Latent risk: Mismatched array lengths raise `IndexError` instead of `ValueError`; NaNs in scores silently return $0.0$. |
| **2. Bypass Rate** | `compute_bypass_rate` in `src/common/metrics.py` | **CORRECT WITH EDGE RISK** | **LOW / MEDIUM** | Decision inequality `probs < threshold` is mathematically sound and strictly consistent with detector prediction convention ($\ge \tau \implies$ fake). Latent risk: Empty input `len(probs) == 0` returns `0.0`, creating false invulnerability claims on sets with 0 fakes. |
| **3. Hard Sample Recall** | `compute_hard_sample_recall` in `src/common/metrics.py` | **CRITICAL DEFECT** | **HIGH** | Line 194 returns `1.0` when no hard samples exist (`not np.any(target_hard_mask)`). This corrupts comparative gain tables (e.g. `ft_gain = 0.85 - 1.0 = -0.15`), falsely penalizing fine-tuned detectors. Must return `np.nan`. |
| **4. ECE & Calibration** | Missing in `src/common/metrics.py` | **MISSING MODULE** | **HIGH** | `expected_calibration_error` and `compute_calibration_curve` are completely absent from `src/common/metrics.py`. Test `test_temperature_scaling_calibration_improves_ece` only checks $T > 0$ without computing ECE. Equal-width vs equal-frequency binning and empty bin zero-weighting are unaddressed. |
| **5. Reliability Diagram** | Missing in `src/common/visualizer.py` | **MISSING MODULE** | **MEDIUM** | No visual calibration diagnostic (`plot_reliability_diagram`) exists in `src/common/visualizer.py`. Needed for scientific coursework deliverables and temperature scaling verification. |
| **6. DRY Pipeline Hygiene** | `evaluate_image_benchmark.py` | **REDUNDANT CODE** | **LOW** | Lines 92–102 reimplement precision, recall, and F1 with local scalar loops instead of calling modular functions in `src.common.metrics`. |

---

## 2. 🧮 Pillar 1: Wilcoxon-Mann-Whitney ROC-AUC Deep Mathematical Audit

### 2.1 Theoretical Foundation & Exact Pairwise Equivalence

For binary ground-truth labels $y_i \in \{0, 1\}$ and predicted continuous scores $s_i \in \mathbb{R}$, let $\mathcal{P} = \{i \mid y_i = 1\}$ with cardinality $N_{\text{pos}} = |\mathcal{P}|$, and $\mathcal{N} = \{j \mid y_j = 0\}$ with cardinality $N_{\text{neg}} = |\mathcal{N}|$.

The non-parametric Wilcoxon-Mann-Whitney (WMW) statistic with mid-rank tie breaking is defined as:
$$U = \sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} S(s_i, s_j), \quad \text{where } S(s_i, s_j) = \begin{cases} 1.0 & s_i > s_j \\ 0.5 & s_i = s_j \\ 0.0 & s_i < s_j \end{cases}$$

The Area Under the Receiver Operating Characteristic curve (ROC-AUC) is identically equal to the normalized Mann-Whitney $U$ statistic:
$$\text{ROC-AUC} = \frac{U}{N_{\text{pos}} \cdot N_{\text{neg}}}$$

By the classical rank-sum theorem, sorting all $N = N_{\text{pos}} + N_{\text{neg}}$ scores in ascending order and assigning 1-based ranks $r_k \in \{1, 2, \dots, N\}$ (with tied observations assigned the arithmetic mean of the rank positions they occupy), the sum of ranks for the positive class $R_{\text{pos}} = \sum_{i \in \mathcal{P}} r_i$ satisfies:
$$U = R_{\text{pos}} - \frac{N_{\text{pos}}(N_{\text{pos}} + 1)}{2}$$

Substituting into the normalized formulation yields:
$$\text{ROC-AUC} = \frac{R_{\text{pos}} - \frac{N_{\text{pos}}(N_{\text{pos}} + 1)}{2}}{N_{\text{pos}} \cdot N_{\text{neg}}}$$

### 2.2 Vectorized Implementation Analysis

In `src/common/metrics.py` (lines 128–140):
```python
order = np.argsort(y_s)
sorted_scores = y_s[order]
_, inverse_indices, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
ends = np.cumsum(counts)
starts = ends - counts + 1
avg_ranks = (starts + ends) / 2.0

ranks = np.empty_like(y_s, dtype=np.float64)
ranks[order] = avg_ranks[inverse_indices]

rank_sum_pos = float(np.sum(ranks[pos_mask]))
auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
return float(np.clip(auc, 0.0, 1.0))
```

#### Step-by-Step Mathematical Invariance Proof:
1. **Contiguous Block Ranking**: Because `sorted_scores = y_s[order]` is sorted, identical values form contiguous 1D sub-arrays.
2. `np.unique(sorted_scores, return_counts=True)` partitions the $N$ sorted elements into $K$ distinct value groups with cardinalities $c_1, c_2, \dots, c_K$, such that $\sum_{k=1}^K c_k = N$.
3. **1-Based Rank Spans**:
   - For group $k$, the 1-based start rank is $s_k = 1 + \sum_{j=1}^{k-1} c_j = \text{ends}_k - c_k + 1$.
   - The 1-based end rank is $e_k = \sum_{j=1}^k c_j = \text{ends}_k$.
4. **Mid-Rank Arithmetic Mean**:
   The sum of ranks spanning the arithmetic progression from $s_k$ to $e_k$ is:
   $$\sum_{r=s_k}^{e_k} r = \frac{c_k (s_k + e_k)}{2} \implies \bar{r}_k = \frac{1}{c_k} \sum_{r=s_k}^{e_k} r = \frac{s_k + e_k}{2}$$
   The formula `avg_ranks = (starts + ends) / 2.0` is therefore **exact with zero truncation error**.
5. **Permutation Unscrambling**:
   `inverse_indices` maps each element in `sorted_scores` to its group index $k \in \{0, \dots, K-1\}$.
   Setting `ranks[order] = avg_ranks[inverse_indices]` assigns element $i$ in the original array `y_s` its true mid-rank because `sorted_scores[m] = y_s[order[m]]`.

### 2.3 Comprehensive Corner-Case Verification

| Corner Case Test Scenario | Input Configuration | Expected Theoretical Value | Code Output | Verdict |
|---|---|---|---|---|
| **All Scores Identical (Tied)** | $y = [0, 1], s = [0.5, 0.5]$ or $[1.0, 1.0, 1.0, 1.0]$ | $0.500000$ (Chance discrimination) | `0.5` | **PASSED** |
| **Imbalanced All-Tie** | $N_{\text{pos}}=10, N_{\text{neg}}=90$, all $s=0.5$ | $0.500000$ (Independent of prior) | `0.5` | **PASSED** |
| **Single Class: All Positive** | $y = [1, 1, 1], s = [0.2, 0.5, 0.9]$ | `np.nan` (Denominator $N_{\text{pos}} N_{\text{neg}} = 0$) | `nan` | **PASSED** |
| **Single Class: All Negative** | $y = [0, 0, 0], s = [0.1, 0.4, 0.8]$ | `np.nan` (Denominator $N_{\text{pos}} N_{\text{neg}} = 0$) | `nan` | **PASSED** |
| **Empty Input Arrays** | $y = [], s = []$ | `np.nan` ($N_{\text{pos}}=0, N_{\text{neg}}=0$) | `nan` | **PASSED** |
| **Length 1 Array** | $y = [1], s = [0.85]$ | `np.nan` (Single class degenerate) | `nan` | **PASSED** |
| **Fully Inverted Ranking** | $y = [1, 1, 0, 0], s = [0.1, 0.2, 0.8, 0.9]$ | $0.000000$ | `0.0` | **PASSED** |
| **Perfect Discrimination** | $y = [0, 0, 1, 1], s = [0.1, 0.2, 0.8, 0.9]$ | $1.000000$ | `1.0` | **PASSED** |

#### Algebraic Proof of All-Tie Resolution ($s_i = c \ \forall i$):
When all $N$ scores are identical, `np.unique` returns $K=1$, $c_1 = N$.
$$\text{ends} = [N], \quad \text{starts} = [1], \quad \bar{r} = \frac{N + 1}{2}$$
Every sample receives identical rank $\frac{N + 1}{2}$.
$$R_{\text{pos}} = N_{\text{pos}} \cdot \frac{N + 1}{2} = N_{\text{pos}} \cdot \frac{N_{\text{pos}} + N_{\text{neg}} + 1}{2}$$
$$U = N_{\text{pos}} \left[ \frac{N_{\text{pos}} + N_{\text{neg}} + 1}{2} - \frac{N_{\text{pos}} + 1}{2} \right] = \frac{N_{\text{pos}} \cdot N_{\text{neg}}}{2}$$
$$\text{ROC-AUC} = \frac{U}{N_{\text{pos}} \cdot N_{\text{neg}}} = \frac{1}{2} \equiv 0.500000 \quad \blacksquare$$

### 2.4 Latent Deficiencies & Numerical Vulnerabilities in `compute_roc_auc`

1. **Mismatched Input Shapes**:
   If `len(y_true) != len(y_score)` (e.g. `y_true` has length 2 and `y_score` has length 1), `pos_mask` has size 2 while `ranks` has size 1. Evaluating `ranks[pos_mask]` triggers:
   `IndexError: boolean index did not match indexed array along axis 0; size of axis is 1 but size of corresponding boolean axis is 2`.
   *Remediation*: Explicit shape assertion at entry:
   ```python
   if len(y_t) != len(y_s):
       raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_score ({len(y_s)})")
   ```
2. **Silent Handling of NaN / Inf in `y_score`**:
   If `y_score` contains `np.nan` (e.g. from exploding logits or NaN activations), `np.argsort` pushes `np.nan` to the end. The function silently returns corrupted numbers ($0.0$ or $0.5$) rather than flagging the numerical breakdown.
   *Remediation*: Check `if np.isnan(y_s).any() or np.isinf(y_s).any(): return float("nan")`.
3. **Non-Binary Ground-Truth Labels**:
   `pos_mask = y_t == 1` and `neg_mask = y_t == 0` silently drops any labels $\notin \{0, 1\}$.
   *Remediation*: Add validation: `if not np.isin(y_t, [0, 1]).all(): raise ValueError("y_true must contain only binary labels {0, 1}")`.

---

## 3. 📉 Pillar 2: Bypass Rate & Hard-Sample Recall Boundary Dynamics

### 3.1 `compute_bypass_rate` Deep Audit

In `src/common/metrics.py` lines 145–171:
```python
def compute_bypass_rate(
    fake_probs: np.ndarray | torch.Tensor | list[float],
    threshold: float = 0.5,
) -> float:
    ...
    if len(probs) == 0:
        return 0.0

    bypassed = probs < threshold
    return float(np.mean(bypassed))
```

#### Mathematical Verification of Decision Boundary:
In Topic 11 (Facial Deepfake Detection):
- Class $0 \equiv \text{Real}$, Class $1 \equiv \text{Fake}$.
- The detector classifies an image as Fake if and only if $\hat{P}(\text{fake}) \ge \tau$.
- A synthetic deepfake image ($y = 1$) successfully fools (bypasses) the detector if the detector predicts Real ($\hat{y} = 0$).
- Therefore, the bypass condition is:
  $$\text{Bypass}(x_i) \iff \hat{P}(\text{fake} \mid x_i) < \tau$$
  $$\text{Bypass Rate}(\tau) = \frac{1}{N_{\text{fake}}} \sum_{i=1}^{N_{\text{fake}}} \mathbb{I}(\hat{P}_i < \tau) = \frac{\text{FN}_{\text{fake}}}{N_{\text{fake}}} = 1 - \text{Recall}_{\text{fake}}(\tau) = \text{FNR}(\tau)$$
- The comparison `probs < threshold` is **strictly correct**.

#### Boundary Point Consistency ($\hat{P} = \tau$):
- In `src/pipelines/evaluate_image_benchmark.py` line 66:
  `preds = (probs >= threshold).long()` $\implies$ $\hat{P} = \tau$ is classified as 1 (Fake / Caught).
- In `compute_bypass_rate`:
  `bypassed = probs < threshold` $\implies$ $\hat{P} = \tau$ evaluates to `False` (Not bypassed / Caught).
- The boundary behavior is **100% mathematically and logically coherent**.

#### Latent Defect: Empty Array Fallback:
In line 166: `if len(probs) == 0: return 0.0`.
- If an evaluation partition contains zero synthetic fake samples ($N_{\text{fake}} = 0$), the function reports `bypass_rate: 0.0`.
- In benchmark reporting, a 0.0% Bypass Rate indicates an impenetrable detector (0 fakes slipped through), when in truth no fakes were tested.
- *Remediation*: Return `float("nan")` when `len(probs) == 0`.

---

### 3.2 `compute_hard_sample_recall` Deep Audit

In `src/common/metrics.py` lines 173–199:
```python
def compute_hard_sample_recall(
    predictions: np.ndarray | list[int],
    targets: np.ndarray | list[int],
    is_hard_sample: np.ndarray | list[bool],
) -> float:
    preds = np.asarray(predictions).ravel()
    targs = np.asarray(targets).ravel()
    hard_mask = np.asarray(is_hard_sample, dtype=bool).ravel()

    target_hard_mask = (targs == 1) & hard_mask
    if not np.any(target_hard_mask):
        return 1.0  # <--- CRITICAL AUDIT DEFECT

    caught = (preds[target_hard_mask] == 1).sum()
    total_hard = target_hard_mask.sum()
    return float(caught / total_hard)
```

#### Detailed Proof of Why Returning `1.0` is a Catastrophic Defect:
1. **The Phantom 100% Recall Artifact**:
   When evaluating a dataset that lacks hard-sample annotations (e.g. `D0_baseline/test.jsonl` where `is_hard_sample` flags are all `False`):
   `np.any(target_hard_mask)` evaluates to `False`.
   The baseline detector $M_0$ is awarded `hard_sample_recall = 1.000` (100% recall on hard samples), despite having never encountered or been tested on a single hard sample!
2. **Distortion of Comparative Improvement Deltas**:
   In `src/pipelines/evaluate_image_benchmark.py` lines 149–151:
   ```python
   "hard_sample_recall_gain": (
       ft_metrics["hard_sample_recall"] - base_metrics["hard_sample_recall"]
   )
   ```
   If baseline $M_0$ reports dummy `1.0` (due to 0 hard samples), and fine-tuned model $M_2$ is evaluated on a real challenge set containing hard samples and catches $85\%$ of them (`hard_sample_recall = 0.85`):
   $$\text{Gain} = 0.85 - 1.00 = -0.15 \quad (-15\% \text{ regression!})$$
   The reporting table prints a false negative delta, declaring that active learning and hard-sample mining *degraded* hard-sample recall by 15%!
3. **Statistical Standard**:
   Recall on an empty subset is mathematically undefined ($\frac{0}{0}$). Scikit-learn and standard statistical packages either return `np.nan` or expose a `zero_division` argument (defaulting to `np.nan` or `0.0`).
4. *Remediation*:
   ```python
   if not np.any(target_hard_mask):
       return float("nan")
   ```

---

## 4. 🌡️ Pillar 3: Expected Calibration Error (ECE) & Reliability Diagrams Audit

### 4.1 Missing Implementation Status

Neither `src/common/metrics.py` nor `src/common/visualizer.py` implements:
- `compute_calibration_curve`
- `expected_calibration_error` (ECE)
- `plot_reliability_diagram`

Furthermore, in `tests/test_calibration.py` line 8:
```python
def test_temperature_scaling_calibration_improves_ece():
    """Verify temperature scaler learns positive temperature and maps logits to calibrated probabilities."""
    scaler = TemperatureScaler()
    logits = torch.tensor([-5.0, -3.0, 3.0, 6.0])
    labels = torch.tensor([0, 0, 1, 1])
    scaler.fit(logits, labels)
    assert scaler.temperature.item() > 0.0
```
This test is a **pseudo-test**: despite its descriptive name, it does **not** compute ECE, does **not** compute calibration curves, and only asserts that $T > 0$.

### 4.2 Mathematical Formulation of Expected Calibration Error (ECE)

For binary classification with predicted probabilities $p_i = \hat{P}(Y=1 \mid X_i) \in [0, 1]$ and binary ground-truth labels $y_i \in \{0, 1\}$:
Let the probability range $[0, 1]$ be partitioned into $M$ bins $B_1, B_2, \dots, B_M$.
For each bin $B_m$:
- The bin sample count is $N_m = |B_m| = \sum_{i=1}^N \mathbb{I}(p_i \in B_m)$.
- The bin average predicted probability (confidence) is:
  $$\text{conf}(B_m) = \frac{1}{N_m} \sum_{i \in B_m} p_i$$
- The bin empirical accuracy (true positive proportion) is:
  $$\text{acc}(B_m) = \frac{1}{N_m} \sum_{i \in B_m} y_i$$
- The Expected Calibration Error (ECE) is the sample-weighted average difference:
  $$\text{ECE} = \sum_{m=1}^M \frac{N_m}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- The Maximum Calibration Error (MCE) is the worst-case bin divergence:
  $$\text{MCE} = \max_{m: N_m > 0} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### 4.3 Binning Strategy Comparison: Equal-Width vs Equal-Frequency

| Dimension | Equal-Width Binning (`strategy='uniform'`) | Equal-Frequency Binning (`strategy='quantile'`) |
|---|---|---|
| **Bin Partitioning** | Uniform intervals: $B_m = (\frac{m-1}{M}, \frac{m}{M}]$ for $m=1, \dots, M$. | Empirical quantiles: $B_m = (q_{m-1}, q_m]$ where $q_k = \text{Percentile}(p, \frac{100k}{M})$. |
| **Bin Widths** | Fixed width $\Delta = \frac{1}{M}$ (e.g. $0.10$ for $M=10$). | Adaptive widths $\Delta_m = q_m - q_{m-1}$. |
| **Sample Distribution** | Highly non-uniform: Deep networks produce heavy mass near $0$ and $1$. Intermediate bins often have $N_m = 0$. | Uniform sample allocation: Each bin contains exactly $N_m \approx \frac{N}{M}$ samples. |
| **Statistical Variance** | High variance in sparsely populated intermediate bins ($\text{Var}(\text{acc}) = \frac{p(1-p)}{N_m}$). | Constant statistical variance across all bins. |
| **Reliability Diagram Standard** | **Universal Academic Standard** (Guo et al. ICML 2017, Niculescu-Mizil & Caruana ICML 2005). Maps directly to diagonal $y=x$. | Non-standard visual presentation (requires variable-width bar widths). |
| **Recommendation** | **Primary default** for both ECE calculation and reliability diagrams. | Supported as optional parameter `strategy="quantile"` for robustness audits. |

### 4.4 Rigorous Handling of Empty Bins ($N_m = 0$)

In equal-width binning on overconfident networks, intermediate bins frequently contain zero samples ($N_m = 0$).
1. **Mathematical Weighting**:
   $$\lim_{N_m \to 0} \frac{N_m}{N} |\text{acc}(B_m) - \text{conf}(B_m)| = 0 \cdot |\text{undefined}| \equiv 0$$
   An empty bin represents zero empirical evidence; its contribution to the expected error across the dataset is strictly **$0.0$**.
2. **Numerical Implementation Hazard**:
   If an algorithm naively computes `acc = np.mean(bin_targets)` on an empty slice, NumPy yields `np.nan`.
   Multiplying `0.0 * np.nan` evaluates to `np.nan` in IEEE 754 floating-point arithmetic!
   A single empty bin would thus cause the entire dataset ECE to degenerate into `np.nan`.
3. **Robust Vectorized Handling**:
   - For `compute_calibration_curve`: Return `prob_true[m] = np.nan`, `prob_pred[m] = np.nan`, and `bin_counts[m] = 0`.
   - For `expected_calibration_error`: Filter strictly over non-empty bins (`valid = bin_counts > 0`):
     ```python
     valid = bin_counts > 0
     if not np.any(valid):
         return float("nan")
     total_samples = np.sum(bin_counts[valid])
     abs_errors = np.abs(prob_true[valid] - prob_pred[valid])
     ece = float(np.sum((bin_counts[valid] / total_samples) * abs_errors))
     ```
   - For `plot_reliability_diagram`: Empty bins are rendered with 0 height and transparent/dashed outlines, preventing misleading visual artifacts.

---

## 5. 🔍 Pillar 4: Benchmark Pipeline (`evaluate_image_benchmark.py`) Audit

### 5.1 Code Duplication & DRY Violation
In `src/pipelines/evaluate_image_benchmark.py` lines 89–102:
```python
acc = float(np.mean(y_pred == y_true)) if len(y_true) > 0 else 0.0
auc = compute_roc_auc_vectorized(y_true, y_prob)

tp = np.sum((y_pred == 1) & (y_true == 1))
fp = np.sum((y_pred == 1) & (y_true == 0))
fn = np.sum((y_pred == 0) & (y_true == 1))

precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
```
- **Deficiency**: `compute_accuracy`, `compute_precision`, and `compute_recall` are already rigorously defined in `src/common/metrics.py`.
- Re-implementing them inline violates the Single Source of Truth invariant and creates maintenance drift.
- `f1_score` computation is missing from `src/common/metrics.py` and should be formalized there.

### 5.2 Tensor Shape & Numerical Hygiene Verification
- `model(images)` outputs shape `(B, 1)`.
- `labels` is formatted as `labels.to(dev).float().view(-1, 1)` (shape `(B, 1)`).
- `nn.BCEWithLogitsLoss()` computes elementwise binary cross-entropy and reduces via `mean`.
- Total loss accumulation:
  `total_loss += float(loss.item()) * images.size(0)`
  `avg_loss = total_loss / max(1, total_samples)`
  This weighted accumulation correctly handles variable batch sizes at dataset boundaries.

---

## 6. 🚀 Production Drop-In Code Recommendations

### 6.1 Proposed Enhancements to `src/common/metrics.py`

Below are the exact production-ready functions to be added/updated in `src/common/metrics.py`:

```python
def compute_f1_score(
    predictions: Union[np.ndarray, list[int]],
    targets: Union[np.ndarray, list[int]],
    positive_label: int = 1,
) -> float:
    """Compute binary classification F1-score (harmonic mean of precision and recall)."""
    p = compute_precision(predictions, targets, positive_label=positive_label)
    r = compute_recall(predictions, targets, positive_label=positive_label)
    if p + r == 0.0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def compute_calibration_curve(
    y_true: Union[np.ndarray, list[int]],
    y_prob: Union[np.ndarray, list[float]],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve (reliability diagram points) for binary probabilities.

    Args:
        y_true: Ground truth binary labels (0 or 1) of shape (N,).
        y_prob: Predicted fake probabilities in [0.0, 1.0] of shape (N,).
        n_bins: Number of confidence bins (default 10).
        strategy: 'uniform' (equal-width in [0, 1]) or 'quantile' (equal-frequency).

    Returns:
        Tuple of (prob_true, prob_pred, bin_counts):
            prob_true: Empirical fraction of positives in each bin (NaN for empty bins).
            prob_pred: Mean predicted probability in each bin (NaN for empty bins).
            bin_counts: Number of samples in each bin.
    """
    y_t = np.asarray(y_true, dtype=np.int64).ravel()
    y_p = np.asarray(y_prob, dtype=np.float64).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_prob ({len(y_p)})")
    if len(y_t) == 0:
        return np.array([]), np.array([]), np.array([], dtype=np.int64)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.percentile(y_p, quantiles * 100.0)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)
        n_bins = len(bin_edges) - 1
    else:
        raise ValueError(f"Unknown strategy: '{strategy}', expected 'uniform' or 'quantile'")

    prob_true = np.full(n_bins, np.nan, dtype=np.float64)
    prob_pred = np.full(n_bins, np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include upper boundary in the final bin
        if i == n_bins - 1:
            mask = (y_p >= low) & (y_p <= high)
        else:
            mask = (y_p >= low) & (y_p < high)

        count = int(np.sum(mask))
        bin_counts[i] = count
        if count > 0:
            prob_true[i] = float(np.mean(y_t[mask]))
            prob_pred[i] = float(np.mean(y_p[mask]))

    return prob_true, prob_pred, bin_counts


def expected_calibration_error(
    y_true: Union[np.ndarray, list[int]],
    y_prob: Union[np.ndarray, list[float]],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Compute Expected Calibration Error (ECE) via sample-weighted calibration divergence.

    Args:
        y_true: Ground truth binary labels (0 or 1) of shape (N,).
        y_prob: Predicted fake probabilities in [0.0, 1.0] of shape (N,).
        n_bins: Number of confidence bins (default 10).
        strategy: 'uniform' or 'quantile'.

    Returns:
        Scalar ECE in [0.0, 1.0], or NaN if input is empty.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_prob).ravel()
    if len(y_t) == 0:
        return float("nan")

    p_true, p_pred, counts = compute_calibration_curve(y_t, y_p, n_bins=n_bins, strategy=strategy)
    valid = counts > 0
    if not np.any(valid):
        return float("nan")

    total = np.sum(counts[valid])
    abs_errors = np.abs(p_true[valid] - p_pred[valid])
    ece = float(np.sum((counts[valid] / total) * abs_errors))
    return float(np.clip(ece, 0.0, 1.0))
```

### 6.2 Proposed Remediation to Existing Functions in `src/common/metrics.py`

#### 1. Fix `compute_hard_sample_recall`:
Replace line 194:
```python
# PREVIOUS BUGGY CODE:
if not np.any(target_hard_mask):
    return 1.0

# REMEDIATED CODE:
if not np.any(target_hard_mask):
    return float("nan")
```

#### 2. Input Validation in `compute_roc_auc`:
```python
# Add shape assertion before computing order:
if len(y_t) != len(y_s):
    raise ValueError(f"Shape mismatch: y_true ({len(y_t)}) vs y_score ({len(y_s)})")
if np.isnan(y_s).any() or np.isinf(y_s).any():
    return float("nan")
```

#### 3. Empty Input Validation in `compute_bypass_rate`:
```python
if len(probs) == 0:
    return float("nan")
```

### 6.3 Proposed Implementation for `src/common/visualizer.py`

```python
def plot_reliability_diagram(
    y_true: Union[np.ndarray, list[int]],
    y_prob: Union[np.ndarray, list[float]],
    output_path: Union[str, Path] = "experiments/eval/reliability_diagram.png",
    n_bins: int = 10,
    title: str = "Model Reliability Diagram & Calibration Curve",
) -> None:
    """Generate and save publication-grade Reliability Diagram with confidence histogram.

    Args:
        y_true: Ground truth binary labels of shape (N,).
        y_prob: Predicted fake probabilities of shape (N,).
        output_path: Output file path for PNG artifact.
        n_bins: Number of calibration bins.
        title: Plot super-title.
    """
    import matplotlib.pyplot as plt
    from src.common.metrics import compute_calibration_curve, expected_calibration_error

    p_true, p_pred, counts = compute_calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = 1.0 / n_bins

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(6.5, 7.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Diagonal perfect calibration anchor
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="#7f8c8d", label="Perfect Calibration")

    # Bar chart for empirical accuracy vs confidence
    valid = counts > 0
    ax_cal.bar(
        bin_centers[valid],
        p_true[valid],
        width=bin_width * 0.85,
        color="#3498db",
        alpha=0.7,
        edgecolor="#2980b9",
        label=f"Detector (ECE = {ece:.3f})",
    )

    ax_cal.set_ylabel("Empirical True Fraction (Accuracy)", fontsize=11)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.grid(True, linestyle=":", alpha=0.6)
    ax_cal.legend(loc="upper left", frameon=True)
    ax_cal.set_title(title, fontsize=12, fontweight="bold")

    # Bottom histogram of sample density
    total_samples = max(1, int(np.sum(counts)))
    ax_hist.bar(
        bin_centers,
        counts / total_samples,
        width=bin_width * 0.85,
        color="#95a5a6",
        edgecolor="#7f8c8d",
    )
    ax_hist.set_xlabel("Predicted Probability P(Fake)", fontsize=11)
    ax_hist.set_ylabel("Sample %", fontsize=11)
    ax_hist.set_ylim(0, max(0.1, float(np.max(counts / total_samples)) * 1.15))
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

---

## 7. 🧪 Pillar 5: Comprehensive Unit Test Suite Expansion

To guarantee 100% test coverage over all identified edge cases, the following tests should be appended to `tests/test_metrics.py` and `tests/test_calibration.py`:

```python
def test_roc_auc_corner_cases_exhaustive():
    """Verify compute_roc_auc on all pathological boundary conditions."""
    import math

    # 1. Empty inputs
    assert math.isnan(compute_roc_auc([], []))

    # 2. Length 1 input
    assert math.isnan(compute_roc_auc([1], [0.9]))
    assert math.isnan(compute_roc_auc([0], [0.1]))

    # 3. All negative
    assert math.isnan(compute_roc_auc([0, 0, 0], [0.1, 0.4, 0.8]))

    # 4. Dense ties
    assert compute_roc_auc([0, 0, 1, 1], [0.8, 0.8, 0.8, 0.8]) == pytest.approx(0.5)

    # 5. Mismatched shapes raise ValueError
    with pytest.raises(ValueError):
        compute_roc_auc([0, 1], [0.5])


def test_hard_sample_recall_empty_mask_returns_nan():
    """Verify hard sample recall returns NaN when no hard samples exist in batch."""
    import math

    preds = np.array([0, 1, 0])
    targets = np.array([0, 1, 0])
    is_hard = np.array([False, False, False])

    recall = compute_hard_sample_recall(preds, targets, is_hard)
    assert math.isnan(recall), f"Expected NaN on empty hard sample set, got {recall}"


def test_expected_calibration_error_properties():
    """Verify ECE calculation on perfectly calibrated and miscalibrated models."""
    import math

    # 1. Perfectly calibrated
    y_true = np.array([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
    y_prob = np.array([0.2] * 100 + [0.8] * 100)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-5)

    # 2. Empty inputs
    assert math.isnan(expected_calibration_error([], []))

    # 3. Empty bins do not trigger NaN
    y_true_sparse = np.array([0, 1])
    y_prob_sparse = np.array([0.05, 0.95])
    ece_sparse = expected_calibration_error(y_true_sparse, y_prob_sparse, n_bins=10)
    assert not math.isnan(ece_sparse)
    assert ece_sparse == pytest.approx(0.05, abs=1e-5)
```

---

## 8. 📋 Pre-Flight Remediation Action Plan

Before commencing the 10,000-sample production training and evaluation run:

1. [ ] **Update `src/common/metrics.py`**:
   - Add input validation in `compute_roc_auc` (shape mismatch check, NaN check).
   - Change `compute_hard_sample_recall` fallback from `1.0` to `float("nan")`.
   - Change `compute_bypass_rate` fallback on empty array to `float("nan")`.
   - Add `compute_f1_score`, `compute_calibration_curve`, and `expected_calibration_error`.
2. [ ] **Update `src/common/visualizer.py`**:
   - Add `plot_reliability_diagram` with dual-panel calibration curve and sample density histogram.
3. [ ] **Update `src/pipelines/evaluate_image_benchmark.py`**:
   - Refactor metric calculation block to call `compute_accuracy`, `compute_precision`, `compute_recall`, `compute_f1_score`, and `expected_calibration_error` from `src.common.metrics`.
4. [ ] **Expand Test Suite**:
   - Update `tests/test_metrics.py` and `tests/test_calibration.py` with the expanded assertions.
   - Run `pytest tests/test_metrics.py tests/test_calibration.py -v` to achieve 100% pass status.
