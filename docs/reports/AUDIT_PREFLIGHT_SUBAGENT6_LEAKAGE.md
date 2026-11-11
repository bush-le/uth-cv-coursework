# AUDIT PREFLIGHT SUBAGENT 6: BENCHMARK VAULTS, DATA SPLITTING & LEAKAGE AUDIT

- **Motivation/Background**: Pre-flight operational integrity audit for Topic 11 (Clean 10,000 Dataset Scaling & Active Learning Pipeline) across partition files, baseline initialization, active learning loops, and frozen evaluation vaults.
- **Purpose**: Deliver mathematical proofs of zero identity leakage, audit sealing of the Master Frozen Test Vault, evaluate the Cumulative Challenge Test Set architecture, detect hidden leakage vectors, and prescribe production-grade code remediations.
- **Overview Pipeline**: Dataset Partitioning -> Mathematical Disjointness Verification -> Seed Traceability -> Master Vault Sealing -> Cumulative Challenge Test Slicing -> Concrete Code Hardening -> Final Verification Protocol.
- **Detailed Plan**:
  - §1 Executive Summary & Forensic Audit Scorecard
  - §2 Mathematical Identity Disjointness Proof & Verification
  - §3 Seed Provenance Traceability (Block 1 & Active Learning)
  - §4 Master Frozen Test Vault Sealing & Tamper-Proofing
  - §5 Cumulative Challenge Test Set Architecture & Class Balance
  - §6 Vulnerabilities, Discrepancies & Hidden Failure Modes
  - §7 Production Code Remediations & Concrete Patches
  - §8 Pre-Execution Verification Protocol & Contract Sign-Off
- **References**:
  - [`GEMINI.md`](GEMINI.md)
  - [`.agents/rules/ZERO_MOCK_DATA_RULE.md`](.agents/rules/ZERO_MOCK_DATA_RULE.md)
  - [`src/data/splits.py`](src/data/splits.py)
  - [`src/data/provenance.py`](src/data/provenance.py)
  - [`src/data/image_datamodule.py`](src/data/image_datamodule.py)
  - [`src/pipelines/run_block1_baseline.py`](src/pipelines/run_block1_baseline.py)
  - [`src/pipelines/run_closed_loop_active_learning.py`](src/pipelines/run_closed_loop_active_learning.py)
  - [`scripts/run_clean_10k_production_pipeline.py`](scripts/run_clean_10k_production_pipeline.py)
  - [`data/partitions/`](data/partitions/)
- **Created**: 2026-11-10T08:35:00+07:00
- **Last Updated**: 2026-11-10T08:35:00+07:00

---

## 1. Executive Summary & Forensic Audit Scorecard

A forensic audit of dataset partitioning, seed provenance, benchmark vault packaging, and evaluation datasets was conducted across the Topic 11 codebase. The audit inspected `src/data/splits.py`, `src/data/provenance.py`, `src/data/image_datamodule.py`, `src/pipelines/run_block1_baseline.py`, `src/pipelines/run_closed_loop_active_learning.py`, `scripts/run_clean_10k_production_pipeline.py`, and the manifest files in `data/partitions/` and `data/benchmarks/`.

### Forensic Audit Scorecard

| Audit Dimension | Target Invariant | Observed System State | Status | Primary Risk / Impact |
| :--- | :--- | :--- | :---: | :--- |
| **Partition Identity Disjointness** | $S_{\text{train}} \cap S_{\text{val}} \cap S_{\text{test}} = \emptyset$ across all real identities and multi-image subjects | 4,000 images, 3,169 unique identities. Exactly 443 multi-image identities (CelebA-HQ). Overlap = **0 identities, 0 images**. | **PASS (100%)** | None on static partition files. Theoretical & empirical proof holds. |
| **Pre-Generation Disjointness Gate** | Automated verification before any synthetic image is generated | `verify_partition_disjointness` is called in Active Learning, but **never** in Block 1. | **FAIL** | Block 1 runs without verifying partition integrity prior to expensive 2,200-sample synthesis. |
| **Disjointness Function Implementation** | Extraction of true biometrics across FFHQ and CelebA-HQ | Looks for non-existent `"identity"` key; defaults to `image_id`. Fails to group CelebA-HQ multi-image identities. | **DEFECT** | False negative in leakage detection: multi-image subject leaks would bypass the assertion undetected. |
| **Seed Manifest Isolation (Block 1)** | Train fakes $\leftarrow$ `train_reals`, Test fakes $\leftarrow$ `test_reals` | Verified: `master_test_fakes` from `test_reals.jsonl`, `d0_train_fakes` from `train_reals.jsonl`. | **PASS (100%)** | Clean isolation in generation pipelines. |
| **Seed Manifest Isolation (Active Learning)** | Candidates synthesized strictly from `train_reals.jsonl` | Pinned to `train_reals.jsonl`, but has unsafe fallback to `data/raw/manifest.jsonl` if missing. | **PASS w/ WARN** | Missing partition file triggers silent regression to full raw dataset (which contains test faces). |
| **Master Frozen Test Vault Integrity** | Permanent, write-once benchmark (589 Real + 589/600 Fake = 1,178/1,189) | Currently corrupted with **599 lines** (589 Real + 10 dry-run mock fakes with `file_path: null`). | **CRITICAL DEFECT** | Running dry-run clobbers the production benchmark vault. Blind `open(path, "w")` without write-once sealing. |
| **Cumulative Challenge Test Set** | 15% slices of mined hard fakes paired 1:1 with authentic test reals | Pairwise 1:1 matching implemented. Real negatives enable valid ROC-AUC. Append mode risks duplication. | **PASS w/ WARN** | Slicing by sample index rather than identity risks cross-round identity contamination. Rerun appends duplicate records. |

---

## 2. Mathematical Identity Disjointness Proof & Verification

### 2.1 Formal Mathematical Proof

Let $\mathcal{D} = \{ m_1, m_2, \dots, m_N \}$ denote the universe of real image metadata records ($N = 4,000$).

#### Definition 1: Identity Equivalence Relation
Define the identity extraction mapping $g: \mathcal{D} \to \mathcal{K}$ implemented by `get_group_key(m)` in [`src/data/splits.py`](src/data/splits.py):
$$g(m) = \begin{cases} \text{str}(m.\text{source\_real\_id}) & \text{if } m.\text{source\_real\_id} \neq \text{None} \\ \text{str}(m.\text{demographics}[\text{"identity\_id"}]) & \text{else if } m.\text{demographics} \text{ has } \text{"identity\_id"} \\ \text{str}(m.\text{image\_id}) & \text{otherwise} \end{cases}$$
where $\mathcal{K}$ is the set of unique biometric identities ($|\mathcal{K}| = 3,169$).

Define the binary relation $\sim$ on $\mathcal{D}$ such that:
$$m_a \sim m_b \iff g(m_a) = g(m_b)$$
- **Reflexivity**: $g(m_a) = g(m_a) \implies m_a \sim m_a$.
- **Symmetry**: $g(m_a) = g(m_b) \implies g(m_b) = g(m_a) \implies m_b \sim m_a$.
- **Transitivity**: $g(m_a) = g(m_b) \land g(m_b) = g(m_c) \implies g(m_a) = g(m_c) \implies m_a \sim m_c$.

Thus, $\sim$ is a formal equivalence relation partitioning $\mathcal{D}$ into mutually disjoint equivalence classes:
$$\mathcal{D} = \bigsqcup_{k \in \mathcal{K}} \mathcal{G}_k, \quad \text{where } \mathcal{G}_k = \{ m \in \mathcal{D} \mid g(m) = k \}$$
By the Fundamental Theorem of Equivalence Relations:
$$\mathcal{G}_{k_1} \cap \mathcal{G}_{k_2} = \emptyset \quad \forall k_1, k_2 \in \mathcal{K}, \; k_1 \neq k_2$$

#### Definition 2: Partition Splitting Function
In `assign_identity_splits(records, ratios, seed)`, the set of unique keys $\mathcal{K}$ is deterministically partitioned into three subsets:
$$\mathcal{K} = \mathcal{K}_{\text{train}} \sqcup \mathcal{K}_{\text{val}} \sqcup \mathcal{K}_{\text{test}}$$
where:
$$\mathcal{K}_{\text{train}} \cap \mathcal{K}_{\text{val}} = \emptyset, \quad \mathcal{K}_{\text{train}} \cap \mathcal{K}_{\text{test}} = \emptyset, \quad \mathcal{K}_{\text{val}} \cap \mathcal{K}_{\text{test}} = \emptyset$$

The partition manifests are constructed by inverse projection:
$$S_{\text{train}} = \bigcup_{k \in \mathcal{K}_{\text{train}}} \mathcal{G}_k, \quad S_{\text{val}} = \bigcup_{k \in \mathcal{K}_{\text{val}}} \mathcal{G}_k, \quad S_{\text{test}} = \bigcup_{k \in \mathcal{K}_{\text{test}}} \mathcal{G}_k$$

#### Theorem: Zero Identity Leakage
For any two splits $S_i, S_j \in \{ S_{\text{train}}, S_{\text{val}}, S_{\text{test}} \}$ with $i \neq j$:
$$S_i \cap S_j = \left( \bigcup_{k \in \mathcal{K}_i} \mathcal{G}_k \right) \cap \left( \bigcup_{l \in \mathcal{K}_j} \mathcal{G}_l \right) = \bigcup_{k \in \mathcal{K}_i, l \in \mathcal{K}_j} (\mathcal{G}_k \cap \mathcal{G}_l)$$
Since $\mathcal{K}_i \cap \mathcal{K}_j = \emptyset$, $k \neq l$ for every pair. Since $\mathcal{G}_k \cap \mathcal{G}_l = \emptyset$ for all $k \neq l$:
$$S_i \cap S_j = \emptyset \quad \blacksquare$$
**Conclusion**: No frame, variation, or synthetic derivative of any identity in $S_i$ can exist in $S_j$.

---

### 2.2 Empirical Audit of `data/partitions/`

The partition files currently stored in `data/partitions/` were subjected to exhaustive forensic cross-matching:

```
data/partitions/
├── train_reals.jsonl   (2,813 records, 2,218 unique identities)
├── val_reals.jsonl     (  598 records,   475 unique identities)
└── test_reals.jsonl    (  589 records,   476 unique identities)
Total:                  (4,000 records, 3,169 unique identities)
```

#### Forensic Overlap Matrix

| Partition A | Partition B | Shared `image_id` | Shared `source_real_id` | Shared Demographics `identity_id` | Leak Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`train_reals`** | **`val_reals`** | **0** | **0** | **0** | **PASS** |
| **`train_reals`** | **`test_reals`** | **0** | **0** | **0** | **PASS** |
| **`val_reals`** | **`test_reals`** | **0** | **0** | **0** | **PASS** |

#### Multi-Image Identity Clustering Verification
In CelebA-HQ, celebrities have multiple photographs. An audit of multi-frame clusters revealed:
- **Total identity clusters with $>1$ image**: Exactly **443 identities** spanning 1,274 images.
- **Max cluster size**: 8 images for a single identity.
- **Clusters spanning multiple partitions**: **EXACTLY 0**.
- **Result**: 100% of multi-image celebrity frames are strictly grouped into their assigned partition.

---

## 3. Seed Provenance Traceability (Block 1 & Active Learning)

### 3.1 Block 1 Seed Traceability

In [`src/pipelines/run_block1_baseline.py`](src/pipelines/run_block1_baseline.py):

```
                        [data/raw/manifest.jsonl] (N=4,000)
                                     │
                        create_identity_partitions()
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   [train_reals.jsonl]       [val_reals.jsonl]         [test_reals.jsonl]
      (2,813 reals)             (598 reals)               (589 reals)
           │                         │                         │
           ▼                         ▼                         ▼
   synthesize quota          synthesize quota          synthesize quota
   (target: 1,411)            (target: 200)             (target: 589)
           │                         │                         │
           ▼                         ▼                         ▼
   [d0_train_fakes]          [d0_val_fakes]          [master_test_fakes]
```

1. **Master Test Fakes (`master_test_fakes_curated.jsonl`)**:
   - Explicit input: `input_manifest = partitions_dir / "test_reals.jsonl"` (Line 330).
   - Provenance: Every fake record receives `source_real_id` corresponding to a held-out test identity (e.g. `fake_ffhq_00016_0` $\leftarrow$ `ffhq_00016`).
   - Leakage to Training: **0%**. Test fakes are packaged strictly into `data/benchmarks/master_frozen_test.jsonl` and `data/benchmarks/D0_baseline/test.jsonl`.
2. **D0 Train Fakes (`d0_train_fakes_curated.jsonl`)**:
   - Explicit input: `input_manifest = partitions_dir / "train_reals.jsonl"` (Line 349).
   - Provenance: Every fake record originates from a training partition identity.
   - Leakage to Test: **0%**.
3. **D0 Val Fakes (`d0_val_fakes_curated.jsonl`)**:
   - Explicit input: `input_manifest = partitions_dir / "val_reals.jsonl"` (Line 368).
   - Provenance: Every fake record originates from a validation partition identity.

### 3.2 Active Learning Candidate Generation Seed Traceability

In [`src/pipelines/run_closed_loop_active_learning.py`](src/pipelines/run_closed_loop_active_learning.py):

```python
# Lines 394-406
train_reals_path = Path("data/partitions/train_reals.jsonl")
test_reals_path = Path("data/partitions/test_reals.jsonl")
raw_manifest_path = train_reals_path if train_reals_path.exists() else Path("data/raw/manifest.jsonl")
if train_reals_path.exists() and test_reals_path.exists():
    verify_partition_disjointness(train_reals_path, test_reals_path)

run_image_generation(
    input_manifest=raw_manifest_path,
    output_manifest=candidates_raw_path,
    output_dir=round_dir / "images",
    total_samples=candidates_per_round,
    ...
)
```

- When `data/partitions/train_reals.jsonl` exists, candidate faces are seeded **strictly from the training partition**.
- **Identified Fragility**: The fallback `train_reals_path if train_reals_path.exists() else Path("data/raw/manifest.jsonl")` is a critical hazard. If the partition file is inadvertently moved or renamed, the pipeline silently falls back to `data/raw/manifest.jsonl`, which contains all 589 test identities. This would cause immediate, catastrophic identity leakage into the training stream (the exact vector documented in `ACTIVE_LEARNING_HARD_SAMPLE_AUDIT_REPORT.md` where `ffhq_00059` leaked into $D_2$).

---

## 4. Master Frozen Test Vault Sealing & Tamper-Proofing

### 4.1 Vault Specification & Target Sample Count

The Master Frozen Test Vault is the immutable evaluation ground truth for all detector iterations ($M_0, M_1, M_2$).

| Attribute | Specification | Production Target | Current Observed State |
| :--- | :--- | :---: | :---: |
| **Path** | `data/benchmarks/master_frozen_test.jsonl` | `data/benchmarks/master_frozen_test.jsonl` | `data/benchmarks/master_frozen_test.jsonl` |
| **Real Samples** | Authentic test images from `test_reals.jsonl` | Exactly **589** | Exactly **589** |
| **Fake Samples** | High-fidelity synthetic fakes from `test_reals` seeds | Exactly **589** (1:1 balanced) or **600** (quota) | **10 (Mock fakes from dry-run)** |
| **Total Samples** | $N_{\text{real}} + N_{\text{fake}}$ | **1,178** (or **1,189**) | **599 (CORRUPTED)** |
| **Class Ratio** | Balanced evaluation | **1.00 : 1.00** (50.0% Real, 50.0% Fake) | 58.9 : 1.0 (98.3% Real, 1.7% Fake) |
| **SHA-256 Digest** | Cryptographically locked checksum | Canonical digest computed once | Ephemeral, modified today at 08:28 |

### 4.2 Forensic Diagnosis: Why the Current Vault is Corrupted (599 Samples)

Inspection of lines 585–599 of `data/benchmarks/master_frozen_test.jsonl` revealed:
```json
{"image_id": "dry_test_fake_0", "label": 1, "manipulation_type": null, "prompt": null, "source_real_id": "ffhq_00016", "file_path": null, "data_source": "synthetic"}
...
{"image_id": "dry_test_fake_9", "label": 1, "manipulation_type": null, "prompt": null, "source_real_id": "ffhq_00093", "file_path": null, "data_source": "synthetic"}
```

**Root Cause**:
In `src/pipelines/run_block1_baseline.py` lines 260–292:
```python
if dry_run:
    mock_fakes = {
        "test": [ImageMetadata(image_id=f"dry_test_fake_{i}", label=1, ...) for i in range(test_fake_count)],
        ...
    }
    package_benchmark_d0(partitions, mock_fakes, benchmarks_dir)
```
When `run_block1_pipeline(..., dry_run=True)` was executed, `benchmarks_dir` defaulted to `data/benchmarks`. Inside `package_benchmark_d0` (lines 206–210):
```python
for path, meta_list in manifest_map.items():
    with open(path, "w", encoding="utf-8") as f:
        for m in meta_list:
            f.write(json.dumps(m.to_dict()) + "\n")
```
`open(path, "w")` clobbered the production `data/benchmarks/master_frozen_test.jsonl` file, replacing valid test fakes with 10 mock records where `file_path` is `null`!

### 4.3 Write-Once Sealing Invariant Violations

1. **No Immutability Guard**: `package_benchmark_d0` does not check if `master_frozen_test.jsonl` exists. If the file exists and is populated with real data, any subsequent execution (accidental dry-run or unit test) truncates and overwrites it.
2. **Pollution by Dry-Run Mode**: Dry-run verification should NEVER write to the canonical `data/benchmarks/` directory. It must redirect output to a temporary dry-run directory.
3. **No Filesystem Lock**: The file is not marked read-only (`chmod 444`), allowing silent overwrites by any script.

---

## 5. Cumulative Challenge Test Set Architecture & Class Balance

### 5.1 Challenge Test Purpose & Dynamics

The Cumulative Challenge Test Set (`data/active_learning/cumulative_challenge_test.jsonl`) evaluates whether retrained detectors retain generalization against hard adversarial deepfakes mined in previous active learning rounds without suffering from catastrophic forgetting.

### 5.2 Slicing & Pairing Architecture

In [`src/pipelines/run_closed_loop_active_learning.py`](src/pipelines/run_closed_loop_active_learning.py) lines 506–529:
```python
# 9. Dual-Benchmark Split: 85% Train / 15% Cumulative Challenge Test
h_train, h_test = split_hard_samples(mined_hard, test_ratio=0.15, seed=seed + r)
if h_test:
    # Pair hard fakes with matching authentic real samples from test partition for mathematically balanced evaluation
    test_reals_path = Path("data/partitions/test_reals.jsonl")
    paired_reals: List[ImageMetadata] = []
    if test_reals_path.exists():
        with open(test_reals_path, "r", encoding="utf-8") as rf:
            all_test_reals = [ImageMetadata.from_dict(json.loads(line)) for line in rf if line.strip()]
        real_rng = random.Random(seed + r * 77)
        sampled_reals = real_rng.sample(all_test_reals, min(len(h_test), len(all_test_reals)))
        paired_reals = sampled_reals

    new_challenge_items = list(h_test) + paired_reals
    challenge_test_records.extend(new_challenge_items)
    with open(challenge_manifest_path, "a", encoding="utf-8") as ch_f:
        for item in new_challenge_items:
            ch_f.write(json.dumps(item.to_dict()) + "\n")
```

#### Class Balance & ROC-AUC Validity
- **Hard Fakes**: $N_{\text{fake}} = |h_{\text{test}}|$ with label $y = 1$.
- **Paired Authentic Reals**: $N_{\text{real}} = |paired\_reals| = |h_{\text{test}}|$ with label $y = 0$, sampled from `data/partitions/test_reals.jsonl`.
- **Class Ratio**: Exactly **1.00 : 1.00** (50% Real, 50% Fake).
- **ROC-AUC Invariant**: Both class 0 and class 1 are guaranteed to be present. The Rank-Sum Wilcoxon-Mann-Whitney metric:
  $$\text{ROC-AUC} = \frac{\sum_{i \in \mathcal{P}} \text{Rank}(i) - \frac{N_{\text{pos}}(N_{\text{pos}} + 1)}{2}}{N_{\text{pos}} \times N_{\text{neg}}}$$
  has a non-zero denominator ($N_{\text{pos}} \times N_{\text{neg}} = |h_{\text{test}}|^2 > 0$), preventing undefined divisions or single-class degeneracies.

---

## 6. Vulnerabilities, Discrepancies & Hidden Failure Modes

### Vulnerability 1: Broken Identity Extraction in `verify_partition_disjointness` (CRITICAL)
In `run_closed_loop_active_learning.py` lines 128–139:
```python
train_ids = {
    json.loads(line).get("identity", json.loads(line)["image_id"])
    for line in f if line.strip()
}
```
- In the real dataset, the key `"identity"` **does not exist** in any record.
- Every line falls back to `["image_id"]`.
- In CelebA-HQ, identity `7423` appears as `celeba_hq_0` and `celeba_hq_1523`.
- If `celeba_hq_0` were in Train and `celeba_hq_1523` were in Test, `train_ids` would have `'celeba_hq_0'` and `test_ids` would have `'celeba_hq_1523'`. Their intersection would be **empty**, and the check would declare **PASS** despite severe identity leakage!

### Vulnerability 2: Missing Pre-Generation Gate in Block 1 (HIGH)
In `run_block1_baseline.py`, `verify_partition_disjointness` is neither imported nor executed. Block 1 splits raw manifests and immediately begins Stable Diffusion generation without running a pre-flight disjointness gate.

### Vulnerability 3: Cross-Round Identity Leakage in `split_hard_samples` (HIGH)
In `split_hard_samples` (lines 161–172 of `run_closed_loop_active_learning.py`):
```python
indices = list(range(len(mined_hard)))
rng.shuffle(indices)
n_test = max(1, int(round(len(mined_hard) * test_ratio)))
test_idx = set(indices[:n_test])
train_hards = [mined_hard[i] for i in range(len(mined_hard)) if i not in test_idx]
test_hards = [mined_hard[i] for i in range(len(mined_hard)) if i in test_idx]
```
- Slicing operates on individual record indices, **NOT on identity equivalence classes** (`get_group_key`).
- If multiple hard variants of the same subject are mined in Round 1, variant 1 may go to `train_hards` while variant 2 goes to `test_hards`.
- Furthermore, because candidate generation samples from `train_reals.jsonl`, all hard fakes share parent identities with `train_reals`. If subject $X$ is placed in `cumulative_challenge_test.jsonl` in Round 1, Round 2 might sample subject $X$ again, mine a new variant, and inject it into the training set for $M_2$.
- Detector $M_2$ is then trained on identity $X$ and tested on identity $X$ in `cumulative_challenge_test.jsonl`.

### Vulnerability 4: Unconditional Benchmark Overwriting & Dry-Run Contamination (CRITICAL)
`package_benchmark_d0` writes directly to `data/benchmarks/master_frozen_test.jsonl`. Running tests or dry-runs destroys production ground truth files on disk.

### Vulnerability 5: Unsafe Fallback to `data/raw/manifest.jsonl` (HIGH)
In `run_closed_loop_active_learning.py:396`:
```python
raw_manifest_path = train_reals_path if train_reals_path.exists() else Path("data/raw/manifest.jsonl")
```
A missing partition file should cause an immediate exception, not a silent fallback that introduces 589 test identities into the generation pipeline.

### Vulnerability 6: DataModule Identity Extraction Discrepancy (MEDIUM)
In `src/data/image_datamodule.py:104`:
```python
key = meta.source_real_id if meta.source_real_id else meta.image_id
```
Does not check `meta.demographics.get("identity_id")`, creating an architectural discrepancy with `splits.py:get_group_key`.

---

## 7. Production Code Remediations & Concrete Patches

### Remediation 1: Unified, Canonical `verify_partition_disjointness` in `src/data/splits.py`

Create a centralized, robust verification utility in `src/data/splits.py` that handles 3-way disjointness across Train, Val, and Test, properly extracting true identity keys:

```python
def extract_record_identity(record_dict: Dict[str, Any]) -> str:
    """Extract canonical subject identity key from raw dictionary or ImageMetadata."""
    if record_dict.get("source_real_id"):
        return str(record_dict["source_real_id"])
    demo = record_dict.get("demographics") or {}
    if demo.get("identity_id"):
        return str(demo["identity_id"])
    return str(record_dict.get("image_id", ""))


def verify_partition_disjointness(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
) -> None:
    """Assert 100% zero identity leakage across train, val, and test partition manifests.
    
    Raises:
        AssertionError: If any biometric identity spans across multiple partitions.
        FileNotFoundError: If any specified partition file does not exist.
    """
    manifests: Dict[str, Path] = {"train": Path(train_path)}
    if val_path:
        manifests["val"] = Path(val_path)
    if test_path:
        manifests["test"] = Path(test_path)

    id_to_splits: Dict[str, Set[str]] = defaultdict(set)
    sample_counts: Dict[str, int] = defaultdict(int)

    for split_name, p in manifests.items():
        if not p.is_file():
            raise FileNotFoundError(f"Partition manifest not found for {split_name}: {p}")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                rec = json.loads(line_s)
                ident = extract_record_identity(rec)
                id_to_splits[ident].add(split_name)
                sample_counts[split_name] += 1

    leaks = {ident: splits for ident, splits in id_to_splits.items() if len(splits) > 1}
    if leaks:
        leak_details = [f"Identity '{k}' leaked across {v}" for k, v in list(leaks.items())[:5]]
        raise AssertionError(
            f"FATAL IDENTITY LEAKAGE DETECTED: {len(leaks)} identities span multiple splits!\n"
            + "\n".join(leak_details)
        )
    logger.info(
        "Partition disjointness verified: %d train, %d val, %d test samples across %d unique identities with ZERO leakage.",
        sample_counts["train"],
        sample_counts.get("val", 0),
        sample_counts.get("test", 0),
        len(id_to_splits),
    )
```

### Remediation 2: Integrate Pre-Generation Assertion in `run_block1_baseline.py`

In `run_block1_pipeline` in [`src/pipelines/run_block1_baseline.py`](src/pipelines/run_block1_baseline.py):

```python
    # 1. Identity-Disjoint Partitioning
    partitions = create_identity_partitions(
        raw_manifest=raw_manifest_path,
        partitions_dir=partitions_dir,
        seed=seed,
    )

    # PRE-FLIGHT LEAKAGE GATE: Assert 100% disjointness BEFORE any synthesis begins
    verify_partition_disjointness(
        train_path=partitions_dir / "train_reals.jsonl",
        val_path=partitions_dir / "val_reals.jsonl",
        test_path=partitions_dir / "test_reals.jsonl",
    )
```

### Remediation 3: Hardened `package_benchmark_d0` with Write-Once Sealing

In `src/pipelines/run_block1_baseline.py`:

```python
def package_benchmark_d0(
    partitions: Dict[str, List[ImageMetadata]],
    curated_fakes: Dict[str, List[ImageMetadata]],
    benchmarks_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Path]:
    """Assemble Master Frozen Test Set and D0_baseline benchmark manifests with write-once sealing."""
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    d0_dir = benchmarks_dir / "D0_baseline"
    d0_dir.mkdir(parents=True, exist_ok=True)

    master_test_records = partitions["test"] + curated_fakes.get("test", [])
    d0_train_records = partitions["train"] + curated_fakes.get("train", [])
    d0_val_records = partitions["val"] + curated_fakes.get("val", [])
    d0_test_records = list(master_test_records)

    paths = {
        "master_test": benchmarks_dir / "master_frozen_test.jsonl",
        "d0_train": d0_dir / "train.jsonl",
        "d0_val": d0_dir / "val.jsonl",
        "d0_test": d0_dir / "test.jsonl",
    }

    # SEALING GUARD: Never overwrite an existing, valid Master Frozen Test Set
    master_path = paths["master_test"]
    if master_path.exists() and not dry_run:
        with open(master_path, "r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())
        if existing_count >= 1178:
            logger.info("Master Frozen Test Set already sealed with %d samples at %s. Preserving vault.", existing_count, master_path)
            # Retain existing vault records for D0 test split to guarantee identical benchmark
            with open(master_path, "r", encoding="utf-8") as f:
                d0_test_records = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

    # Write manifests
    for name, path in paths.items():
        if name == "master_test" and path.exists() and not dry_run:
            continue  # Do not overwrite sealed master vault
        with open(path, "w", encoding="utf-8") as f:
            meta_list = master_test_records if name == "master_test" else (
                d0_train_records if name == "d0_train" else (
                    d0_val_records if name == "d0_val" else d0_test_records
                )
            )
            for m in meta_list:
                f.write(json.dumps(m.to_dict()) + "\n")

    return paths
```

### Remediation 4: Dry-Run Directory Isolation

In `run_block1_pipeline`:
```python
    if dry_run:
        dry_benchmarks_dir = base_dir / "benchmarks_dry_run"
        package_benchmark_d0(partitions, mock_fakes, dry_benchmarks_dir, dry_run=True)
        # Never clobber data/benchmarks/master_frozen_test.jsonl in dry-run mode!
```

### Remediation 5: Fail-Loud Missing Partition Guard in Active Learning

In `run_closed_loop_active_learning.py` lines 394–398:
```python
    train_reals_path = Path("data/partitions/train_reals.jsonl")
    test_reals_path = Path("data/partitions/test_reals.jsonl")
    val_reals_path = Path("data/partitions/val_reals.jsonl")

    if not train_reals_path.is_file():
        raise FileNotFoundError(
            f"CRITICAL: Training seed manifest {train_reals_path} missing! "
            "Refusing fallback to raw manifest to prevent test vault contamination."
        )

    # 3-Way Disjointness Pre-Generation Gate
    verify_partition_disjointness(train_reals_path, val_reals_path, test_reals_path)
    raw_manifest_path = train_reals_path
```

### Remediation 6: Identity-Disjoint `split_hard_samples` with Quarantine

```python
def split_hard_samples(
    mined_hard: List[ImageMetadata],
    test_ratio: float = 0.15,
    seed: int = 42,
    quarantined_identities: Optional[Set[str]] = None,
) -> Tuple[List[ImageMetadata], List[ImageMetadata], Set[str]]:
    """Split mined hard samples by identity equivalence class to prevent cross-split leakage.
    
    Guarantees:
    - All variations of the same person go strictly to EITHER train buffer OR challenge test.
    - Identities already assigned to challenge test in previous rounds are quarantined from train buffer.
    """
    quarantined = set(quarantined_identities or set())
    if not mined_hard:
        return [], [], quarantined

    groups: Dict[str, List[ImageMetadata]] = defaultdict(list)
    for m in mined_hard:
        ident = m.source_real_id or (m.demographics or {}).get("identity_id") or m.image_id
        groups[str(ident)].append(m)

    unique_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_keys)

    train_hards: List[ImageMetadata] = []
    test_hards: List[ImageMetadata] = []

    # Calculate target test identities
    n_test_keys = max(1, int(round(len(unique_keys) * test_ratio))) if len(unique_keys) >= 2 else 0

    for idx, k in enumerate(unique_keys):
        if k in quarantined or idx < n_test_keys:
            test_hards.extend(groups[k])
            quarantined.add(k)
        else:
            train_hards.extend(groups[k])

    return train_hards, test_hards, quarantined
```

---

## 8. Pre-Execution Verification Protocol & Contract Sign-Off

Before launching the 9-hour, clean 10,000 production pipeline via `bash scripts/run_pipeline.sh train configs/dataset_factory.yaml` or `python scripts/run_clean_10k_production_pipeline.py`, the following 4-step verification protocol must be executed:

### Step 1: Clean Contaminated Benchmark Directory
```bash
# Remove mock-contaminated files from previous dry-run
rm -rf data/benchmarks/master_frozen_test.jsonl data/benchmarks/D0_baseline
rm -rf data/active_learning/cumulative_challenge_test.jsonl
```

### Step 2: Run Automated Partition Disjointness Check
```bash
python3 -c "
import json
from pathlib import Path
from collections import defaultdict

manifests = {
    'train': Path('data/partitions/train_reals.jsonl'),
    'val': Path('data/partitions/val_reals.jsonl'),
    'test': Path('data/partitions/test_reals.jsonl'),
}

id_map = defaultdict(set)
for split, p in manifests.items():
    with open(p) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            ident = d.get('source_real_id') or (d.get('demographics') or {}).get('identity_id') or d['image_id']
            id_map[str(ident)].add(split)

leaks = {k: v for k, v in id_map.items() if len(v) > 1}
assert len(leaks) == 0, f'LEAKAGE DETECTED: {leaks}'
print('Pre-flight check PASSED: 100% Identity Disjoint across 3,169 unique identities.')
"
```

### Step 3: Run Full PyTest Suite
```bash
make test-smoke
pytest tests/test_splits.py tests/test_block1_baseline.py tests/test_closed_loop_active_learning.py -v
```

### Step 4: Verify Master Test Count Target
Following real generation of Block 1 Master Test Fakes:
- Real samples: **589**
- Fake samples: **589** (or 600)
- Total samples: **1,178** (or 1,189)
- Command: `wc -l data/benchmarks/master_frozen_test.jsonl` must output `1178` (or `1189`).

---

### Audit Sign-Off
- **Auditor**: Subagent 6 (Benchmark Vaults, Data Splitting & Leakage Auditor)
- **Status**: Audit Completed. Critical dry-run contamination vector diagnosed. Full mathematical proofs and code remediation patches delivered. Ready for integration.
