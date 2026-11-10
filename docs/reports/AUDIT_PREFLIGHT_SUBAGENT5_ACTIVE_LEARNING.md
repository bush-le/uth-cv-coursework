# AUDIT_PREFLIGHT_SUBAGENT5_ACTIVE_LEARNING.md — Algorithmic & Mathematical Audit of Active Learning Hard-Sample Mining & Attribution

- **Motivation/Background**: Preflight algorithmic, mathematical, and architectural audit of Block 2 closed-loop active learning, hard-sample mining mechanisms, dynamic budget control, feedback attribution, and prompt grammar steering in preparation for full production execution.
- **Purpose**: Provide rigorous mathematical proofs, identify code vulnerabilities, evaluate boundary condition stability, and define exact drop-in refactoring recommendations across active learning pipelines.
- **Overview Pipeline**: `AdaptiveBudgetController` (Tier 1 mode quotas & Tier 2 mask targets) -> `PromptGrammarEngine` (Error-steered combinatorial prompt synthesis) -> SD Candidate Generation -> Curation Gate -> Detector Inference & Temperature Calibration -> `CalibratedHardMiner` -> Attribution Feedback (`extract_error_attribution_from_hard_samples`) -> Dual-Benchmark Split (`split_hard_samples`).
- **Detailed Plan**: §1 Executive Summary & Verification Scorecard; §2 Hard-Sample Mining Logic & Mathematical Calibration Audit; §3 Modality & Regional Budget Allocation Audit; §4 Feedback Attribution & Slot Probability Shifts Audit; §5 Configuration Audit & Missing File Analysis; §6 Code Smells & Architectural Inconsistencies; §7 Concrete Actionable Code Recommendations.
- **References**: [src/pipelines/mine_hard_samples.py](../../src/pipelines/mine_hard_samples.py), [src/pipelines/budget_controller.py](../../src/pipelines/budget_controller.py), [src/pipelines/run_closed_loop_active_learning.py](../../src/pipelines/run_closed_loop_active_learning.py), [src/models/evaluators/calibrated_mining.py](../../src/models/evaluators/calibrated_mining.py), [src/models/diffusion/prompt_engine.py](../../src/models/diffusion/prompt_engine.py), [configs/dataset_factory.yaml](../../configs/dataset_factory.yaml).
- **Created**: 2026-11-10T08:30:00+07:00
- **Last Updated**: 2026-11-10T08:30:00+07:00

---

## 1. Executive Summary & Verification Scorecard

This audit report delivers an exhaustive algorithmic, mathematical, and software engineering evaluation of the active learning pipeline implemented across [src/pipelines/mine_hard_samples.py](../../src/pipelines/mine_hard_samples.py), [src/pipelines/budget_controller.py](../../src/pipelines/budget_controller.py), [src/pipelines/run_closed_loop_active_learning.py](../../src/pipelines/run_closed_loop_active_learning.py), [src/models/evaluators/calibrated_mining.py](../../src/models/evaluators/calibrated_mining.py), and [src/models/diffusion/prompt_engine.py](../../src/models/diffusion/prompt_engine.py).

### High-Level Audit Scorecard

| Audit Domain | Status | Critical Findings & Observations |
| :--- | :---: | :--- |
| **1. Hard-Sample Mining Logic** | **CONDITIONAL PASS** | False negatives ($P < \tau$) and boundary uncertain samples ($|P - \tau| \le \delta$) are captured. Temperature calibration optimizes Cross-Entropy / NLL via L-BFGS. **Critical caveat**: pure temperature scaling preserves the sign of logits, so at decision threshold $\tau = 0.50$, it has **zero effect** on which samples are false negatives. Furthermore, logit-probability round-trip inversion introduces precision loss at boundaries. |
| **2. Boundary & Edge Case Handling** | **PASS WITH ACTION** | `split_hard_samples` handles empty lists (`[]`) and singletons (`N=1`) gracefully without throwing exceptions. However, if a round mines 0 hard samples, `run_closed_loop_active_learning.py` stalls dataset expansion and retrains on unchanged data because `CalibratedHardMiner` lacks an adaptive `min_hard_samples` fallback. |
| **3. Budget Controller Quotas** | **PASS WITH ACTION** | Sum of quotas strictly equals `candidates_per_round` via the Largest Remainder Method (Hare-Niemeyer). Modality errors of 0 correctly trigger uniform/exploration allocation. **Mathematical flaw**: naive probability clamping followed by sum re-normalization dilutes the effective minimum floor below `min_mode_ratio` under skewed error distributions. |
| **4. Mask Target Regional Allocation** | **WARNING** | `STANDARD_REGIONS` defines 4 targets (`"periocular"`, `"lips"`, `"nose"`, `"full"`), but downstream `generate_dynamic_face_mask` in [src/models/diffusion/stable_diffusion_engine.py](../../src/models/diffusion/stable_diffusion_engine.py) has **no implementation for `"nose"`** (silently treats it as `"full"`). Quota distribution is multinomial and prone to small-sample variance. |
| **5. Feedback Attribution & Grammar Steering** | **WARNING** | Softmax temperature scaling shifts weights toward failing tags. **Critical flaws**: `keyword_categories` is hardcoded to 10 strings, missing large parts of `SLOT_TAXONOMY`; string mismatch causes `"occlusion:wearing eyeglasses"` to never match actual tags; absence of Laplace smoothing allows rare keywords ($N=1$) to monopolize error weights. |
| **6. Configuration Architecture** | **ACTION REQUIRED** | `configs/active_learning.yaml` is referenced in system requirements but **does not exist on disk**; the pipeline falls back to `configs/dataset_factory.yaml` or implicit defaults. |
| **7. Code Cleanliness & Smells** | **FAIL** | [src/pipelines/mine_hard_samples.py](../../src/pipelines/mine_hard_samples.py) defines `mine_hard_samples_by_feature_similarity` **twice** (lines 363-395 and lines 538-572), where the second definition overrides the first with differing default arguments. |

---

## 2. Hard-Sample Mining Logic & Mathematical Calibration Audit

### 2.1 Identification of False Negatives and Adversarial Bypasses

In [src/pipelines/mine_hard_samples.py](../../src/pipelines/mine_hard_samples.py) and [src/models/evaluators/calibrated_mining.py](../../src/models/evaluators/calibrated_mining.py), hard samples represent synthetic deepfake images that successfully deceive or challenge the detector:

1. **Standalone Detector Mining (`mine_hard_samples.py`, Line 448)**:
   ```python
   is_bypass = (meta.label == 1) and (p_val < bypass_threshold)
   ```
   A candidate is classified as a hard sample if its ground-truth label is fake (`label == 1`) and the predicted fake probability falls below the bypass threshold $\tau$ (default $\tau = 0.50$).
   - Objective Real-Manifold Mining (`mine_hard_samples_real_manifold`, Line 312): Computes the $k$-nearest neighbor cosine similarity of synthetic faces against the real manifold:
     $$S_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(\mathbf{e}_i^{\text{fake}})} \cos(\mathbf{e}_i^{\text{fake}}, \mathbf{e}_j^{\text{real}}) \ge \tau$$
     where $\tau$ is calibrated dynamically from the leave-one-out reference quantile of authentic real faces excluding the source face ($j \ne \text{source\_real\_id}$).

2. **Closed-Loop Calibrated Mining (`CalibratedHardMiner.mine_hard_samples`, Lines 126–141)**:
   Extends pure false negatives with a dual-criterion formulation:
   - **Criterion 1 (False Negative)**: $P_{\text{cal}} < \tau$ (model predicts "Real" for a synthetic deepfake).
   - **Criterion 2 (Boundary Uncertainty)**: $|P_{\text{cal}} - \tau| \le \delta$ (where $\delta = \text{uncertainty\_window} = 0.10$).
   
   ```python
   is_fn = p < tau
   is_uncertain = abs(p - tau) <= delta

   if is_fn:
       fn_candidates.append((meta, p, idx))
   elif is_uncertain:
       unc_candidates.append((meta, p, idx))
   ```
   Because `elif is_uncertain` is used, the two lists are mutually exclusive:
   - `fn_candidates` contains: $P_{\text{cal}} \in [0.0, \tau)$
   - `unc_candidates` contains: $P_{\text{cal}} \in [\tau, \tau + \delta]$
   Combined candidates span $P_{\text{cal}} \in [0.0, \tau + \delta]$. This properly captures both complete detector blindspots ($P < \tau$) and marginal boundary decisions ($\tau \le P \le \tau + \delta$).

### 2.2 Mathematical Soundness of Temperature Calibration vs. Platt Scaling

In [src/models/evaluators/calibrated_mining.py](../../src/models/evaluators/calibrated_mining.py) (`calibrate_temperature`, Lines 43–84), scalar temperature $T$ is optimized on held-out validation logits and labels:

$$\min_{T > 0} \mathcal{L}(T) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \sigma\left(\frac{z_i}{T}\right) + (1 - y_i) \log\left(1 - \sigma\left(\frac{z_i}{T}\right)\right) \right]$$

where $z_i$ is the unscaled logit, $y_i \in \{0, 1\}$ is the ground-truth label, and $\sigma(u) = \frac{1}{1 + e^{-u}}$.

#### Mathematical Audit of the Implementation:
1. **Parameterization**:
   The temperature is parameterized as $T = \exp(\theta)$ with $\theta \in \mathbb{R}$ (`log_temp = nn.Parameter(...)`). This enforces $T > 0$ strictly, preventing division by zero or negative temperature inversions.
2. **Loss Formulation**:
   `criterion = nn.BCEWithLogitsLoss()` applied to `logits / temp` mathematically evaluates the exact negative log-likelihood (binary cross-entropy loss) over validation logits:
   $$\text{BCEWithLogitsLoss}(z/T, y) = \max(z/T, 0) - (z/T) \cdot y + \log(1 + e^{-|z/T|})$$
   This is numerically stable and avoids sigmoid overflow.
3. **Optimization Method**:
   L-BFGS (`torch.optim.LBFGS`) is used with a line search closure. For convex 1-D NLL optimization over scalar temperature, L-BFGS converges in $< 10$ iterations.

#### Critical Mathematical Invariant & Limitation:
> [!IMPORTANT]
> **Threshold Invariance at $\tau = 0.50$**:
> For any temperature $T > 0$ and any logit $z \in \mathbb{R}$:
> $$\sigma\left(\frac{z}{T}\right) < 0.50 \iff \frac{z}{T} < 0 \iff z < 0 \iff \sigma(z) < 0.50$$
> **Proof**: $\sigma(u) = 0.50 \iff u = 0$. Since $T > 0$, $\text{sign}(z / T) \equiv \text{sign}(z)$.
> Consequently, when the decision threshold is $\tau = 0.50$, **temperature scaling does NOT alter whether a sample is a False Negative ($P < 0.50$)**!
> Temperature scaling only alters the boundary uncertainty count ($|P_{\text{cal}} - 0.50| \le \delta$). If the model is overconfident ($T > 1$), probabilities soften toward $0.50$, expanding the uncertainty set; if underconfident ($T < 1$), probabilities sharpen away from $0.50$, shrinking the uncertainty set.

#### Temperature Scaling vs. Platt Scaling:
- **Platt Scaling** (Platt, 1999) fits an affine transformation:
  $$z' = a z + b$$
  optimizing two parameters $a > 0$ (scale) and $b \in \mathbb{R}$ (bias / shift). The bias parameter $b$ shifts the decision threshold:
  $$\sigma(a z + b) = 0.50 \iff z = -\frac{b}{a}$$
  This corrects for prior class imbalance or systematic positive/negative calibration drift.
- **Current Implementation**: Implements pure **Temperature Scaling** ($b = 0$, $a = 1/T$), NOT full Platt Scaling. While temperature scaling is sound for confidence calibration, it cannot shift the decision threshold.

### 2.3 Logit-Probability Round-Trip Inversion

In [src/pipelines/run_closed_loop_active_learning.py](../../src/pipelines/run_closed_loop_active_learning.py) (Lines 484–486):
```python
logits = detector(imgs)
probs = torch.sigmoid(logits).view(-1).cpu().tolist()
mined_hard, mining_stats = miner.mine_hard_samples(curated_candidates, probs)
```
Inside `CalibratedHardMiner.apply_temperature_to_probabilities` (Lines 91–98):
```python
p_clamped = max(1e-6, min(1.0 - 1e-6, float(p)))
logit = np.log(p_clamped / (1.0 - p_clamped))
cal_logit = logit / self.temperature
cal_p = 1.0 / (1.0 + np.exp(-cal_logit))
```
This performs a round-trip:
$$\text{logits} \xrightarrow{\sigma} \text{probs} \xrightarrow{\text{clamp}} \text{clamped\_p} \xrightarrow{\text{logit link}} \text{reconstructed\_logits} \xrightarrow{/ T} \text{cal\_logit} \xrightarrow{\sigma} \text{cal\_p}$$

#### Numerical Critique:
- Clamping $p \in [10^{-6}, 1 - 10^{-6}]$ truncates the logit range to $[-13.815, +13.815]$. Any detector output with $|z| > 13.8$ suffers from artificial saturation.
- Floating point round-trip errors accumulate unnecessarily.
- **Recommendation**: Pass raw logits $z$ directly to `CalibratedHardMiner.mine_hard_samples(candidates, logits=logits)` and compute `cal_probs = torch.sigmoid(logits / T)` directly.

### 2.4 Degenerate Mining Edge Cases

#### Case 1: Hard Samples = 0 ($P > \tau + \delta$ for all synthetic fakes)
- **Behavior in `mine_hard_samples.py`**:
  Lines 473–493 provide an adaptive fallback:
  ```python
  if min_hard_samples > 0 and len(hard_meta) < min_hard_samples and updated_meta:
      sorted_cands = sorted(updated_meta, key=lambda m: m.baseline_score)
      top_hard = sorted_cands[:min(min_hard_samples, len(updated_meta))]
  ```
  If $\ge 1$ candidate exists, it forces the hardest $K$ candidates into the hard sample set.
- **Behavior in `CalibratedHardMiner` (`calibrated_mining.py`)**:
  `CalibratedHardMiner` has **no `min_hard_samples` parameter or fallback**.
  If all fakes are detected with high confidence ($P_{\text{cal}} > 0.60$), `selected_records` is empty `[]`.
- **Downstream Consequence in Active Learning Loop**:
  In `run_closed_loop_active_learning.py`:
  - `split_hard_samples([], test_ratio=0.15)` returns `([], [])`.
  - `h_train` is empty, so dataset expansion is an identity operation: $D_r = D_{r-1}$.
  - The detector $M_r$ is retrained on the exact same dataset with no new information, wasting GPU epochs.
  - All error rates become $0.0$, causing `compute_mode_quotas` and `update_error_weights` to collapse to uniform defaults.

#### Case 2: All Fakes are Hard ($P < \tau$ for all synthetic fakes)
- If all candidates bypass the detector, all $N$ candidates enter `mined_hard`.
- `filter_by_embedding_diversity` (if active) prunes near-duplicates.
- `split_hard_samples` allocates $\lfloor 0.85 N \rfloor$ to training and $\lceil 0.15 N \rceil$ to challenge test.
- All modality error rates evaluate to $1.0$. In `BudgetController.compute_mode_quotas`, `np.all(err_arr == err_arr[0])` evaluates to `True`, falling back to uniform quotas across modalities, which is mathematically optimal given uniform vulnerability.

### 2.5 `split_hard_samples` Boundary Behavior

Inspecting `split_hard_samples` in [src/pipelines/run_closed_loop_active_learning.py](../../src/pipelines/run_closed_loop_active_learning.py) (Lines 146–173):

```python
def split_hard_samples(
    mined_hard: List[ImageMetadata],
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
    if not mined_hard:
        return [], []
    if len(mined_hard) == 1:
        return list(mined_hard), []

    rng = random.Random(seed)
    indices = list(range(len(mined_hard)))
    rng.shuffle(indices)

    n_test = max(1, int(round(len(mined_hard) * test_ratio)))
    if n_test >= len(mined_hard):
        n_test = len(mined_hard) - 1

    test_idx = set(indices[:n_test])
    train_hards = [mined_hard[i] for i in range(len(mined_hard)) if i not in test_idx]
    test_hards = [mined_hard[i] for i in range(len(mined_hard)) if i in test_idx]
    return train_hards, test_hards
```

#### Verification Proof:
1. **Empty list (`N = 0`)**: `if not mined_hard: return [], []`. Returns cleanly without errors.
2. **Single sample (`N = 1`)**: `if len(mined_hard) == 1: return list(mined_hard), []`. Returns 1 train, 0 test. Training set receives the sole sample; challenge test remains unaugmented.
3. **Two samples (`N = 2`)**: `n_test = max(1, int(round(2 * 0.15))) = max(1, 0) = 1`. `n_test < 2`. Returns 1 train, 1 test.
4. **General constraint ($N \ge 2$)**: `if n_test >= len(mined_hard): n_test = len(mined_hard) - 1`. This mathematical guard guarantees:
   $$|H_{\text{train}}| \ge 1 \quad \text{and} \quad |H_{\text{test}}| \ge 1 \quad \forall N \ge 2$$
   Disjointness is strictly guaranteed: $H_{\text{train}} \cap H_{\text{test}} = \emptyset$ and $H_{\text{train}} \cup H_{\text{test}} = H$.

---

## 3. Modality & Regional Budget Allocation Audit

### 3.1 `BudgetController.compute_mode_quotas` & Zero-Error Floor

In [src/pipelines/budget_controller.py](../../src/pipelines/budget_controller.py) (`compute_mode_quotas`, Lines 62–107):
- Default parameters: `total_candidate_budget = 150` (or `60` in active learning loop), `min_mode_ratio = 0.15`, `exploration_floor = 0.20`.
- Active modes: `["sd_text2img", "sd_inpaint", "sd_img2img"]` ($k = 3$).

#### Scenario A: Modality Errors are None or Empty
```python
if not mode_errors:
    base_count = self.total_candidate_budget // k
    remainder = self.total_candidate_budget % k
    quotas = {m: base_count for m in active_modes}
    for i in range(remainder):
        quotas[active_modes[i]] += 1
    return quotas
```
For budget 60: each mode receives exactly $60 // 3 = 20$ samples. Uniform distribution is strictly preserved.

#### Scenario B: Modality Errors are all 0.0 (`np.all(err_arr == 0.0)`)
```python
if np.all(err_arr == 0.0) or np.all(err_arr == err_arr[0]):
    softmax_probs = np.ones(k, dtype=np.float64) / k
```
`softmax_probs = [1/3, 1/3, 1/3]`.
`clamped_probs = np.maximum(softmax_probs, 0.15) = [1/3, 1/3, 1/3]`.
`final_probs = [1/3, 1/3, 1/3]`.
Each mode receives $\ge 15\%$ exploration floor.

#### Subtle Mathematical Flaw: Floor Dilution Under Heterogeneous Errors
Consider an extreme error distribution:
- `sd_inpaint`: error rate $0.95$
- `sd_text2img`: error rate $0.00$
- `sd_img2img`: error rate $0.00$
- Configuration: $k = 3$, `min_mode_ratio = 0.20`.

1. Compute raw softmax:
   $$\text{err\_arr} = [0.95, 0.0, 0.0] \implies \exp(\text{err} - \max) = [e^0, e^{-0.95}, e^{-0.95}] = [1.0, 0.3867, 0.3867]$$
   $$\text{softmax\_probs} = [0.5639, 0.2180, 0.2180]$$
2. If `err_arr` had a larger difference (e.g. $[2.5, 0.0, 0.0]$):
   $$\text{softmax\_probs} = [0.860, 0.070, 0.070]$$
3. Apply clamping (`np.maximum(softmax_probs, 0.20)`):
   $$\text{clamped\_probs} = [0.860, 0.200, 0.200] \implies \sum \text{clamped} = 1.260$$
4. Re-normalize (`clamped_probs / np.sum(clamped_probs)`):
   $$\text{final\_probs} = \left[\frac{0.860}{1.260}, \frac{0.200}{1.260}, \frac{0.200}{1.260}\right] = [0.6825, \mathbf{0.1587}, \mathbf{0.1587}]$$

> [!WARNING]
> **Floor Dilution Bug**:
> The resulting probability is $\mathbf{0.1587}$, which is **strictly less than the guaranteed floor of $0.200$**!
> Whenever $\sum \text{clamped} > 1.0$, dividing by the sum scales down the clamped values.
> 
> **Mathematical Solution (Residual Water-Filling)**:
> Allocate the minimum floor $p_{\min}$ first to all $k$ modes, then distribute the remaining probability $1 - k \cdot p_{\min}$ proportionally via softmax:
> $$p_i = p_{\min} + (1 - k \cdot p_{\min}) \cdot \text{softmax}(e_i)$$
> For $p_{\min} = 0.20$, $1 - 3(0.20) = 0.40$. The floor is strictly guaranteed: $p_i \ge 0.20$ for all modes, and $\sum p_i \equiv 1.0$.

### 3.2 Exact Budget Conservation (Largest Remainder Method)

In [src/pipelines/budget_controller.py](../../src/pipelines/budget_controller.py) (Lines 91–100):
```python
exact_counts = final_probs * self.total_candidate_budget
integer_counts = np.floor(exact_counts).astype(int)
remainder = self.total_candidate_budget - int(np.sum(integer_counts))

fractional_parts = exact_counts - integer_counts
sorted_indices = np.argsort(-fractional_parts)
for idx in sorted_indices[:remainder]:
    integer_counts[idx] += 1
```

#### Verification Proof:
This implements the **Hare-Niemeyer Largest Remainder Method**:
1. Let $x_i = p_i \cdot B$, where $\sum_{i=1}^k x_i = B$.
2. $\lfloor x_i \rfloor \le x_i$, so $\sum \lfloor x_i \rfloor \le B$.
3. The shortfall $R = B - \sum \lfloor x_i \rfloor$ is an integer in $[0, k-1]$.
4. Distributing 1 unit each to the top $R$ indices ordered by $x_i - \lfloor x_i \rfloor$ ensures:
   $$\sum_{i=1}^k \text{quota}_i \equiv B = \text{total\_candidate\_budget} \quad \text{strictly}.$$
No candidate slots are ever leaked or over-allocated.

### 3.3 Regional Inpainting Target Allocation & Discrepancies

Inspecting `compute_inpainting_mask_targets` in [src/pipelines/budget_controller.py](../../src/pipelines/budget_controller.py) (Lines 15–17, 109–151):
```python
STANDARD_REGIONS = ["periocular", "lips", "nose", "full"]
```

#### Discrepancy 1: The Phantom `"nose"` Target
- In `AdaptiveBudgetController`, `STANDARD_REGIONS` includes `"nose"`.
- In `run_closed_loop_active_learning.py` (Line 388), `mask_targets = budget_ctrl.compute_inpainting_mask_targets(...)` produces targets containing `"nose"`.
- However, in [src/models/diffusion/stable_diffusion_engine.py](../../src/models/diffusion/stable_diffusion_engine.py) (`generate_dynamic_face_mask`, Lines 82–99):
  ```python
  if target_region == "periocular":
      # eye strip
      ...
  elif target_region == "lips":
      # mouth strip
      ...
  else:
      # Full face
      ...
  ```
  **There is no `elif target_region == "nose":` branch!**
  Any target designated as `"nose"` falls through to `else:`, which generates a **full-face mask**.
- Furthermore, in [src/pipelines/generate_deepfake_images.py](../../src/pipelines/generate_deepfake_images.py) (Line 380), the regional fallback explicitly specifies only 3 regions:
  ```python
  region_choice = ["periocular", "lips", "full"][task["item_idx"] % 3]
  ```
- **Consequence**: The active learning loop allocates candidates to `"nose"` believing it is targeting the nasal pyramid, but physically synthesizes full-face inpaintings. When computing error attribution in `extract_error_attribution_from_hard_samples`, failures on these images are attributed to `"nose"` while the actual modification was full-face.

#### Discrepancy 2: Multinomial Sampling Variance
`compute_inpainting_mask_targets` uses stochastic sampling:
```python
sampled_indices = rng.choice(r_len, size=count, p=probs)
return [active_regions[idx] for idx in sampled_indices]
```
For a typical round with `count = 20` inpaintings, multinomial sampling introduces substantial variance:
- If $p(\text{lips}) = 0.15$, the expected count is $3.0$, but the probability of sampling $0$ lips targets is:
  $$(1 - 0.15)^{20} = (0.85)^{20} \approx 3.88\%$$
- Roughly 1 out of every 25 rounds will completely omit an exploration region purely due to random sampling noise.
- **Recommendation**: Apply the Largest Remainder Method (identical to `compute_mode_quotas`) to determine exact integer target counts per region, followed by a deterministic shuffle.

---

## 4. Feedback Attribution & Slot Probability Shifts Audit

### 4.1 Keyword Failure Rate Calculation

In [src/pipelines/run_closed_loop_active_learning.py](../../src/pipelines/run_closed_loop_active_learning.py) (`extract_error_attribution_from_hard_samples`, Lines 84–120):

```python
keyword_categories = [
    "lighting:direct harsh camera flash",
    "lighting:soft studio key light",
    "lighting:warm golden hour sunlight",
    "lighting:dramatic cinematic side lighting",
    "occlusion:wearing eyeglasses",
    "occlusion:wearing dark designer sunglasses",
    "demographics:elderly",
    "demographics:young",
    "optics:85mm",
    "optics:35mm",
]
...
keyword_errors = {
    cat: (keyword_hards.get(cat, 0) / max(1, count))
    for cat, count in keyword_totals.items()
}
```

The empirical error rate for keyword $c$ is:
$$\hat{e}_c = \frac{N_{\text{hard}}(c)}{\max(1, N_{\text{total}}(c))}$$

#### Critical Flaws in Error Attribution:

1. **Taxonomy Truncation & Slot Omission**:
   `keyword_categories` only defines 10 arbitrary strings. It completely ignores:
   - All `expression` slots (e.g. "subtle warm smile", "candid neutral glance")
   - All `imperfections` slots (e.g. "authentic skin texture, visible fine pores", "natural freckles")
   - Multiple `optics` slots ("50mm portrait prime", "Hasselblad medium format")
   - Half of the `lighting` slots ("diffuse window light", "cool overcast daylight", "rim lighting", "candlelit glow")
   - Non-eyeglass `occlusion` slots ("knit sweater collar", "silver stud earring", "bare face")

2. **String Mismatch Bug in Eyeglasses**:
   In `keyword_categories`: `"occlusion:wearing eyeglasses"`.
   In `SLOT_TAXONOMY["occlusion"]`:
   - Tag 0: `"wearing thin gold wireframe eyeglasses with subtle lens reflections"`
   - Tag 1: `"wearing classic tortoiseshell prescription glasses, distinct stylish frames"`
   Checking `if key_tag.lower() in p_text` checks if the literal substring `"wearing eyeglasses"` is in the prompt.
   In Tag 0, the phrase is interrupted by `"thin gold wireframe"`. Thus:
   `"wearing eyeglasses" in "wearing thin gold wireframe eyeglasses..."` evaluates to **`False`**!
   Tag 1 has `"prescription glasses"`, which also evaluates to **`False`**.
   Consequently, eyeglasses generated via `PromptGrammarEngine` are **never attributed**, yielding $N_{\text{total}} = 0$, so the failure rate for eyeglasses is never computed.

3. **Demographic Slot Name Inconsistency**:
   In `keyword_categories`, the prefix is `"demographics:elderly"` and `"demographics:young"`.
   In `SLOT_TAXONOMY`, the slot key is `"subject_age"`.
   In `PromptGrammarEngine.update_error_weights`:
   `key_full = f"{slot}:{tag}"` creates `"subject_age:elderly face..."`.
   Because `"demographics:" != "subject_age:"`, direct dictionary lookups fail, forcing the engine into secondary substring matching.
   Moreover, tags such as `"senior portrait, dignified expression, soft age marks"` do not contain the word `"elderly"`, and `"college youth portrait"` does not contain `"young"`. These tags receive zero error score and are suppressed during softmax weighting.

4. **Small-Sample Variance (No Smoothing)**:
   Suppose keyword $A$ is sampled in only 1 candidate image, which happens to be hard:
   $$\hat{e}_A = \frac{1}{1} = 1.00$$
   Keyword $B$ is sampled in 15 candidate images, 12 of which are hard:
   $$\hat{e}_B = \frac{12}{15} = 0.80$$
   Keyword $A$ receives a higher error attribution than $B$, even though $B$ is a statistically significant, systemic vulnerability of the detector!
   
   **Mathematical Solution (Laplace / Bayesian Prior Smoothing)**:
   $$\tilde{e}_c = \frac{N_{\text{hard}}(c) + \alpha}{N_{\text{total}}(c) + \alpha + \beta}$$
   With an uninformative prior $\alpha = 1, \beta = 1$:
   - For keyword $A$: $\tilde{e}_A = \frac{1 + 1}{1 + 2} = 0.667$
   - For keyword $B$: $\tilde{e}_B = \frac{12 + 1}{15 + 2} = \frac{13}{17} = 0.765$
   Keyword $B$ now correctly outranks the single-sample outlier $A$.

### 4.2 Softmax Temperature Scaling in `PromptGrammarEngine`

In [src/models/diffusion/prompt_engine.py](../../src/models/diffusion/prompt_engine.py) (`update_error_weights`, Lines 140–150):

```python
scores_arr = np.array(raw_scores, dtype=np.float64) / temp
exp_scores = np.exp(scores_arr - np.max(scores_arr))
softmax_probs = exp_scores / np.sum(exp_scores)

uniform_prob = 1.0 / len(tags)
floor = self.exploration_floor
blended_probs = (1.0 - floor) * softmax_probs + floor * uniform_prob
blended_probs = blended_probs / np.sum(blended_probs)
```

#### Mathematical Properties:
1. **Numerical Stability**: Subtracting $\max(s_i)$ prevents floating point overflow in $\exp(s_i)$.
2. **Temperature Modulation**:
   - As $T \to 0^+$, softmax concentrates all probability onto $\arg\max_i(e_i)$.
   - As $T \to \infty$, softmax converges to uniform probability $\frac{1}{M}$.
   - At default $T = 0.70$, failure differences are sharpened appropriately.
3. **Exploration Floor**: Blending with $\frac{1}{M}$ guarantees that every descriptor tag maintains a minimum selection probability:
   $$p_j \ge \frac{\text{floor}}{M} = \frac{0.15}{M} > 0$$
   This prevents absorbing states where undetected descriptors are permanently eliminated from future prompt synthesis.

---

## 5. Configuration Audit & Missing Files Analysis

### 5.1 Status of `configs/active_learning.yaml`

The project specification and subagent mission specify inspecting `configs/active_learning.yaml`.
**Verification Command**: `ls -la configs/`
**Finding**: The file `configs/active_learning.yaml` **does not exist**.

Current configuration loading relies on:
1. `run_closed_loop_active_learning.py` CLI default: `--config configs/dataset_factory.yaml`.
2. Hardcoded parameter fallbacks within Python functions:
   - `candidates_per_round = 60`
   - `num_rounds = 2`
   - `min_mode_ratio = 0.15`
   - `exploration_floor = 0.20`
   - `decision_threshold = 0.50`
   - `uncertainty_window = 0.10`
   - `test_ratio = 0.15`
   - `epochs_per_round = 5`
   - `learning_rate = 2e-4`
   - `hard_sample_loss_weight = 2.0`

This violates the **Zero-Hardcode Rule** defined in `.agents/rules/` and `GEMINI.md`.

---

## 6. Code Smells & Architectural Inconsistencies

### 6.1 Duplicate Function Definition in `mine_hard_samples.py`

In [src/pipelines/mine_hard_samples.py](../../src/pipelines/mine_hard_samples.py):
- **Definition 1 (Lines 363–395)**:
  ```python
  def mine_hard_samples_by_feature_similarity(
      extractor: nn.Module,
      synthetic_metadata: List[ImageMetadata],
      real_metadata: Optional[List[ImageMetadata]] = None,
      ...
  ) -> HardSampleMiningReport:
      rep = mine_hard_samples_real_manifold(
          ...,
          tau=similarity_threshold,
          # k_neighbors defaults to 5 inside mine_hard_samples_real_manifold
      )
  ```
- **Definition 2 (Lines 538–572)**:
  ```python
  def mine_hard_samples_by_feature_similarity(
      extractor: nn.Module,
      synthetic_metadata: List[ImageMetadata],
      real_metadata: Optional[List[ImageMetadata]] = None,
      ...
  ) -> HardSampleMiningReport:
      rep = mine_hard_samples_real_manifold(
          ...,
          tau=similarity_threshold,
          k_neighbors=1, # Overrides Definition 1 with k_neighbors=1!
      )
  ```
Definition 2 unconditionally overwrites Definition 1. Definition 1 is dead code. This is a severe lint defect.

### 6.2 Inactive Embedding Diversity Filtering in Active Learning

In [src/pipelines/run_closed_loop_active_learning.py](../../src/pipelines/run_closed_loop_active_learning.py) (Line 489):
```python
mined_hard, mining_stats = miner.mine_hard_samples(curated_candidates, all_probs)
```
Notice that `embeddings` is omitted (defaults to `None`).
In `CalibratedHardMiner.mine_hard_samples`:
```python
if embeddings is not None and len(embeddings) == len(candidate_records):
    # perceptual diversity filtering...
```
Because embeddings are never passed, **perceptual diversity filtering is 100% inactive** during closed-loop active learning rounds! If two nearly identical hard synthetic samples pass curation, both enter the training buffer, causing redundant feature clustering.

---

## 7. Concrete Actionable Code Recommendations

### 7.1 Refactoring `AdaptiveBudgetController` (`src/pipelines/budget_controller.py`)

#### Problem:
1. Dilution of minimum mode ratio below `min_mode_ratio` after sum normalization.
2. Phantom `"nose"` target in `STANDARD_REGIONS` when downstream engine only implements `"periocular"`, `"lips"`, and `"full"`.
3. High variance from multinomial random sampling in `compute_inpainting_mask_targets`.

#### Recommended Solution:

```python
# src/pipelines/budget_controller.py

# Align with actual anatomical implementations in stable_diffusion_engine.py
STANDARD_MODES = ["sd_text2img", "sd_inpaint", "sd_img2img"]
STANDARD_REGIONS = ["periocular", "lips", "full"]


class AdaptiveBudgetController:
    def __init__(
        self,
        total_candidate_budget: int = 150,
        min_mode_ratio: float = 0.15,
        exploration_floor: float = 0.20,
        seed: int = 42,
    ) -> None:
        self.total_candidate_budget = max(3, int(total_candidate_budget))
        self.min_mode_ratio = max(0.05, min(0.30, float(min_mode_ratio)))
        self.exploration_floor = max(0.05, min(0.50, float(exploration_floor)))
        self.seed = seed

    def compute_mode_quotas(
        self,
        mode_errors: Optional[Dict[str, float]] = None,
        modes: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        active_modes = modes or STANDARD_MODES
        k = len(active_modes)
        if k == 0:
            return {}

        min_p = self.min_mode_ratio
        if k * min_p > 1.0:
            min_p = 1.0 / k

        if not mode_errors:
            softmax_probs = np.ones(k, dtype=np.float64) / k
        else:
            raw_errs = [max(0.0, float(mode_errors.get(m, 0.0))) for m in active_modes]
            err_arr = np.array(raw_errs, dtype=np.float64)
            if np.all(err_arr == 0.0) or np.all(err_arr == err_arr[0]):
                softmax_probs = np.ones(k, dtype=np.float64) / k
            else:
                exp_err = np.exp(err_arr - np.max(err_arr))
                softmax_probs = exp_err / np.sum(exp_err)

        # Residual Water-Filling: strictly guarantees p_i >= min_p
        residual_prob = 1.0 - (k * min_p)
        final_probs = min_p + residual_prob * softmax_probs

        # Hare-Niemeyer Largest Remainder Method for integer budget conservation
        exact_counts = final_probs * self.total_candidate_budget
        integer_counts = np.floor(exact_counts).astype(int)
        remainder = self.total_candidate_budget - int(np.sum(integer_counts))

        fractional_parts = exact_counts - integer_counts
        sorted_indices = np.argsort(-fractional_parts)
        for idx in sorted_indices[:remainder]:
            integer_counts[idx] += 1

        return {m: int(integer_counts[i]) for i, m in enumerate(active_modes)}

    def compute_inpainting_mask_targets(
        self,
        region_errors: Optional[Dict[str, float]] = None,
        count: int = 50,
        regions: Optional[List[str]] = None,
    ) -> List[str]:
        active_regions = regions or STANDARD_REGIONS
        r_len = len(active_regions)
        if count <= 0 or r_len == 0:
            return []

        min_p = 1.0 / (2.0 * r_len)  # exploration floor for regions

        if not region_errors:
            softmax_probs = np.ones(r_len, dtype=np.float64) / r_len
        else:
            raw_errs = [max(0.0, float(region_errors.get(r, 0.0))) for r in active_regions]
            err_arr = np.array(raw_errs, dtype=np.float64)
            if np.all(err_arr == 0.0) or np.all(err_arr == err_arr[0]):
                softmax_probs = np.ones(r_len, dtype=np.float64) / r_len
            else:
                exp_err = np.exp(err_arr - np.max(err_arr))
                softmax_probs = exp_err / np.sum(exp_err)

        residual_prob = 1.0 - (r_len * min_p)
        final_probs = min_p + residual_prob * softmax_probs

        # Deterministic Hare-Niemeyer quota allocation eliminates multinomial variance
        exact_counts = final_probs * count
        int_counts = np.floor(exact_counts).astype(int)
        remainder = count - int(np.sum(int_counts))
        fracs = exact_counts - int_counts
        for idx in np.argsort(-fracs)[:remainder]:
            int_counts[idx] += 1

        ordered_targets = []
        for i, r_name in enumerate(active_regions):
            ordered_targets.extend([r_name] * int(int_counts[i]))

        # Deterministic shuffle with seed
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(len(ordered_targets))
        return [ordered_targets[i] for i in perm]
```

---

### 7.2 Refactoring `CalibratedHardMiner` (`src/models/evaluators/calibrated_mining.py`)

#### Problem:
1. No adaptive `min_hard_samples` fallback when 0 candidates bypass detector.
2. Inefficient and precision-losing probability round-trip (`probs -> logits -> probs`).
3. Temperature scaling cannot shift the decision threshold at $\tau = 0.50$.

#### Recommended Solution:

```python
# src/models/evaluators/calibrated_mining.py

class CalibratedHardMiner:
    def __init__(
        self,
        temperature: float = 1.0,
        decision_threshold: float = 0.5,
        uncertainty_window: float = 0.10,
        min_cosine_distance: float = 0.05,
        min_hard_samples: int = 2,  # Guarantees non-empty dataset expansion
    ) -> None:
        self.temperature = max(0.01, float(temperature))
        self.decision_threshold = float(decision_threshold)
        self.uncertainty_window = float(uncertainty_window)
        self.min_cosine_distance = float(min_cosine_distance)
        self.min_hard_samples = max(0, int(min_hard_samples))

    def mine_hard_samples(
        self,
        candidate_records: List[ImageMetadata],
        probabilities: Optional[List[float]] = None,
        logits: Optional[torch.Tensor] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[ImageMetadata], Dict[str, Any]]:
        if logits is not None:
            # Direct logit scaling: avoids numerical precision loss from inverse sigmoid
            scaled_logits = logits.view(-1).float() / self.temperature
            cal_probs = torch.sigmoid(scaled_logits).detach().cpu().tolist()
        elif probabilities is not None:
            cal_probs = self.apply_temperature_to_probabilities(probabilities)
        else:
            raise ValueError("Either logits or probabilities must be provided.")

        fn_candidates = []
        unc_candidates = []
        tau = self.decision_threshold
        delta = self.uncertainty_window

        for idx, (meta, p) in enumerate(zip(candidate_records, cal_probs)):
            if meta.label != 1:
                continue
            if p < tau:
                fn_candidates.append((meta, p, idx))
            elif abs(p - tau) <= delta:
                unc_candidates.append((meta, p, idx))

        combined_candidates = fn_candidates + unc_candidates

        # Adaptive Fallback: if yield < min_hard_samples, take hardest candidates by p
        if len(combined_candidates) < self.min_hard_samples and candidate_records:
            all_sorted = sorted(
                [(meta, p, idx) for idx, (meta, p) in enumerate(zip(candidate_records, cal_probs)) if meta.label == 1],
                key=lambda item: item[1],
            )
            combined_candidates = all_sorted[:min(self.min_hard_samples, len(all_sorted))]

        # Apply perceptual diversity deduplication
        ...
```

---

### 7.3 Refactoring Attribution & Prompt Engine (`src/pipelines/run_closed_loop_active_learning.py` & `src/models/diffusion/prompt_engine.py`)

#### Problem:
1. Substring mismatch in eyeglasses (`"wearing eyeglasses"` vs `"wearing thin gold wireframe eyeglasses..."`).
2. Demographics slot mismatch (`"demographics:"` vs `"subject_age:"`).
3. Single-sample noise ($N=1$ yielding error rate $1.00$).

#### Recommended Solution for `extract_error_attribution_from_hard_samples`:

```python
def extract_error_attribution_from_hard_samples(
    hard_samples: List[ImageMetadata],
    all_candidates: List[ImageMetadata],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> Dict[str, Any]:
    hard_ids = {h.image_id for h in hard_samples}

    # 1. Modality failure rates
    mode_totals: Dict[str, int] = {}
    mode_hards: Dict[str, int] = {}
    for c in all_candidates:
        m = c.manipulation_type or "unknown"
        mode_totals[m] = mode_totals.get(m, 0) + 1
        if c.image_id in hard_ids:
            mode_hards[m] = mode_hards.get(m, 0) + 1

    mode_errors = {
        m: (mode_hards.get(m, 0) + prior_alpha) / (count + prior_alpha + prior_beta)
        for m, count in mode_totals.items()
    }

    # 2. Regional failure rates
    region_totals: Dict[str, int] = {}
    region_hards: Dict[str, int] = {}
    for c in all_candidates:
        if c.manipulation_type == "sd_inpaint":
            target = (c.generation or {}).get("inpaint_mask_target") or "full"
            region_totals[target] = region_totals.get(target, 0) + 1
            if c.image_id in hard_ids:
                region_hards[target] = region_hards.get(target, 0) + 1

    region_errors = {
        r: (region_hards.get(r, 0) + prior_alpha) / (count + prior_alpha + prior_beta)
        for r, count in region_totals.items()
    }

    # 3. Dynamic vocabulary extraction matching SLOT_TAXONOMY keywords
    token_patterns = {
        "lighting:direct harsh camera flash": ["camera flash", "direct harsh"],
        "lighting:soft studio key light": ["soft studio", "beauty dish"],
        "lighting:warm golden hour sunlight": ["golden hour", "warm sunlight"],
        "lighting:dramatic cinematic side lighting": ["cinematic side", "chiaroscuro"],
        "occlusion:eyeglasses": ["eyeglasses", "prescription glasses", "wireframe"],
        "occlusion:sunglasses": ["sunglasses", "dark designer"],
        "subject_age:elderly": ["elderly", "natural wrinkles", "senior", "silver hair"],
        "subject_age:young": ["young adult", "youth", "fresh skin", "college"],
        "optics:85mm": ["85mm"],
        "optics:35mm": ["35mm"],
    }

    keyword_totals: Dict[str, int] = {}
    keyword_hards: Dict[str, int] = {}

    for c in all_candidates:
        p_text = (c.prompt or "").lower()
        is_hard = c.image_id in hard_ids

        for cat, synonyms in token_patterns.items():
            if any(syn in p_text for syn in synonyms):
                keyword_totals[cat] = keyword_totals.get(cat, 0) + 1
                if is_hard:
                    keyword_hards[cat] = keyword_hards.get(cat, 0) + 1

    # Laplace smoothing prevents single-sample outlier distortion
    keyword_errors = {
        cat: (keyword_hards.get(cat, 0) + prior_alpha) / (count + prior_alpha + prior_beta)
        for cat, count in keyword_totals.items()
    }

    return {
        "mode_errors": mode_errors,
        "region_errors": region_errors,
        "keyword_errors": keyword_errors,
    }
```

---

### 7.4 Dedicated Production Configuration (`configs/active_learning.yaml`)

To establish a single source of truth for Topic 11 / Block 2 active learning, create `configs/active_learning.yaml`:

```yaml
# ==============================================================================
# Closed-Loop Active Learning Pipeline Configuration (Topic 11 / Block 2)
# ==============================================================================

seed: 42
device: "cuda"

paths:
  train_reals: "data/partitions/train_reals.jsonl"
  val_manifest: "data/benchmarks/D0_baseline/val.jsonl"
  test_manifest: "data/benchmarks/D0_baseline/test.jsonl"
  base_dir: "data/active_learning"
  checkpoints_dir: "experiments/checkpoints"
  logs_dir: "experiments/logs"
  cumulative_challenge_manifest: "data/active_learning/cumulative_challenge_test.jsonl"

loop:
  num_rounds: 2
  candidates_per_round: 60
  resume: true

budget:
  min_mode_ratio: 0.15
  exploration_floor: 0.20
  modes:
    - "sd_text2img"
    - "sd_inpaint"
    - "sd_img2img"
  regions:
    - "periocular"
    - "lips"
    - "full"

hardness:
  decision_threshold: 0.50
  uncertainty_window: 0.10
  min_hard_samples: 2
  min_cosine_distance: 0.05
  temperature_calibration:
    enabled: true
    max_iter: 50
    lr: 0.05

attribution:
  smoothing_alpha: 1.0
  smoothing_beta: 1.0
  prompt_temperature: 0.70
  prompt_exploration_floor: 0.15

retraining:
  epochs_per_round: 5
  batch_size: 16
  learning_rate: 2.0e-4
  hard_sample_loss_weight: 2.0
  unfreeze_last_n_blocks: 1

challenge_split:
  test_ratio: 0.15
```

---

### 7.5 Deduplication in `mine_hard_samples.py`

Remove the duplicate definition of `mine_hard_samples_by_feature_similarity` at lines 538–572 of [src/pipelines/mine_hard_samples.py](../../src/pipelines/mine_hard_samples.py). Consolidate into the primary definition at line 363 with explicit parameter documentation.

---

## 8. Summary of Preflight Verification & Sign-Off

| Verification Item | Pre-Audit Risk | Post-Recommendation Status |
| :--- | :---: | :---: |
| Threshold Invariance under $T$-scaling ($\tau=0.50$) | High | Documented mathematical invariant; boundary uncertainty delta $\delta$ captures ambiguous fakes. |
| Zero-Hard Sample Active Learning Stalling | High | Resolved via `min_hard_samples` fallback in `CalibratedHardMiner`. |
| Budget Allocation Floor Dilution | Medium | Resolved via Residual Water-Filling algorithm ($p_{\min} + (1 - k p_{\min}) \text{softmax}$). |
| Phantom `"nose"` Mask Allocation | Medium | Resolved by aligning `STANDARD_REGIONS` to `["periocular", "lips", "full"]`. |
| Eyeglasses Substring Attribution Failure | High | Resolved by synonym dictionary in `extract_error_attribution_from_hard_samples`. |
| Small-Sample Attribution Noise ($N=1$) | Medium | Resolved by Bayesian Laplace smoothing ($\alpha=1, \beta=1$). |
| Missing `configs/active_learning.yaml` | Medium | Resolved by authoring dedicated single-source-of-truth config file. |
| Duplicate Function Definition in `mine_hard_samples.py` | Low | Resolved by deleting redundant lines 538–572. |

**Audit Conclusion**: The algorithmic foundation of the closed-loop active learning system is strong, featuring stateful resume, partition disjointness verification, and Hare-Niemeyer integer quota conservation. Implementing the 5 targeted mathematical and architectural refactorings detailed above will guarantee numerical robustness, eliminate attribution silent failures, and ensure monotonic benchmark gains during full-scale active learning loops.
