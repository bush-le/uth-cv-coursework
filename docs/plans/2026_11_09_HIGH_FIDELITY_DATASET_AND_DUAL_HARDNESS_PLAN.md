# HIGH_FIDELITY_DATASET_AND_DUAL_HARDNESS_PLAN.md — High-Fidelity Dataset Generation & Controlled Hard-Sample Utility Plan

- **Motivation/Background**: Synthetic deepfakes frequently suffer from visible artifacts and anatomical deformities that fail to challenge modern vision backbones. Generating high-fidelity diffusion samples with identity-disjoint partitioning ensures robust and unpolluted evaluation of detector vulnerability.
- **Purpose**: Serve as the implementation plan and execution guide for high-fidelity facial diffusion generation, curation gates, and controlled 3-way utility benchmarking.
- **Overview Pipeline**: Partition Real Identities (80% Train, 20% Test) -> High-Fidelity Diffusion Generation -> Multi-Tier Curation Gate -> Baseline Training -> Hard-Sample Mining -> Proof-of-Utility Benchmark.
- **Detailed Plan**: §1 Zero-Leakage Identity-Disjoint Partitioning; §2 High-Fidelity Diffusion Engine; §3 Multi-Tier Curation Gate; §4 Supervised Hard Sample Mining; §5 3-Way Controlled Proof-of-Utility Benchmark.
- **References**: [docs/PURPOSE.md](docs/PURPOSE.md), [.agents/rules/ZERO_MOCK_DATA_RULE.md](.agents/rules/ZERO_MOCK_DATA_RULE.md), [configs/dataset_factory.yaml](configs/dataset_factory.yaml).
- **Created**: 2026-11-09T14:00:00+07:00
- **Last Updated**: 2026-11-11T18:35:00+07:00

---

# High-Fidelity Deepfake Dataset & Proof-of-Utility Benchmark Plan
## (100% Zero Data Leakage Guaranteed via Identity-Disjoint Protocol)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (Native execution chosen by user) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-fidelity synthetic facial dataset using Stable Diffusion that eliminates anatomical deformities (fooling human eyes), and execute a scientifically rigorous 3-Way Controlled Proof-of-Utility Benchmark ($D_0$ Baseline vs. $D_1 = D_0 + R$ Random Augmentation vs. $D_2 = D_0 + H_1$ Hard-Sample Augmentation) evaluated on a strictly **Identity-Disjoint Unseen Test Set** ($D_{\text{test}}$) with **100% Zero Data Leakage**.

**Architecture:**
1. **Zero-Leakage Identity-Disjoint Partitioning**: Partition the original real subjects at origin into $\text{Train\_IDs}$ ($80\%$) and $\text{Test\_IDs}$ ($20\%$) with $\text{Train\_IDs} \cap \text{Test\_IDs} = \emptyset$. All training, mining, and random augmentations are strictly generated from $\text{Train\_IDs}$. The independent evaluation set $D_{\text{test}}$ (Real + Fake) is generated solely from $\text{Test\_IDs}$.
2. **High-Fidelity Diffusion Engine**: Native $512 \times 512$ generation downsampled via Lanczos to $256 \times 256$, guided by dynamic elliptical jaw-expanded ($+18\%$) face masks with $r=16\text{px}$ Gaussian feathering to prevent duplicate mouths and boundary seams.
3. **Multi-Tier Curation Gate**: Biometric facial aspect ratio, secondary mouth/lip continuity detection, and Sobel seam gradient verification to eliminate visual anomalies before curation.
4. **Supervised Detector Bypass Mining ($H_1$)**: Define hard samples as false negatives of a trained baseline detector $M_0$: $H_1 = \{x \in \text{Synthetic Fakes} : P_{M_0}(\text{Fake} \mid x) < 0.50\}$, paired with an equal-sized random subset $R$ ($|R| = |H_1|$).
5. **3-Way Controlled Proof-of-Utility**: Train $M_0$ (on $D_0$), $M_{\text{rand}}$ (on $D_1$), and $M_{\text{hard}}$ (on $D_2$), then evaluate all three on the frozen, untouched $D_{\text{test}}$ to prove whether hard samples provide superior discriminative utility.

**Tech Stack:** PyTorch, Diffusers (Stable Diffusion v1.5), Meta DINOv2 (`facebook/dinov2-small`), OpenCV, torchvision, PIL, NumPy, Scikit-learn, Pytest.

**Spec:** `docs/PURPOSE.md`, `.agents/rules/ZERO_MOCK_DATA_RULE.md`, `configs/dataset_factory.yaml`.

---

## Global Constraints & 100% Anti-Leakage Invariants

1. **Strict Identity-Disjoint Partitioning (Zero-Leakage Invariant)**:
   - Every subject identity belongs to either $\text{Train\_IDs}$ or $\text{Test\_IDs}$.
   - **NO identity overlap**: $\text{Train\_IDs} \cap \text{Test\_IDs} = \emptyset$.
   - **No background overlap**: All Real test images come from $\text{Test\_IDs}$; all Fake test images are synthesized solely from $\text{Test\_IDs}$. The detector $M$ is NEVER exposed to any image or identity in $\text{Test\_IDs}$ during training or mining.
2. **Zero-Mock & Neural Checkpoint Integrity**:
   - Mining $H_1$ requires a verified trained checkpoint (`experiments/checkpoints/baseline_detector_best.pt`). Zero unverified random-weight or mock mining.
3. **Controlled Coursework Hardware Scale**:
   - Total dataset scale: 120–160 samples (60 Real Train, 60 Fake Train [30 initial + 30 mining pool], 20 Real Test, 20 Fake Test).
4. **VRAM Hygiene**:
   - Maintain single-pipeline lifecycle in Stable Diffusion engine (`_switch_mode`) with `gc.collect()` and `torch.cuda.empty_cache()` to guarantee zero OOM.
