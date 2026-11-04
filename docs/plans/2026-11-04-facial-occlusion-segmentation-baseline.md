# 2026-11-04-facial-occlusion-segmentation-baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the fine-grained facial occlusion segmentation components: paired Dataset/DataModule, boundary-aware metrics (mIoU and Boundary IoU), and segmentation model baseline in `src/`.

**Architecture:** Build `src/data/occlusion_dataset.py` for synchronized image-mask pairs, add vectorized `compute_miou` and `compute_boundary_iou` in `src/common/metrics.py`, build `OcclusionSegmentationBaseline` in `src/models/deep/segmentation.py`, wire it into `src/models/builder.py`, and test via unit tests in `tests/test_occlusion_segmentation.py`.

**Tech Stack:** PyTorch 2.15, Torchvision, NumPy 2.5, PyYAML, pytest.

**Spec:** [docs/coursework/TAXONOMY_AND_DATA_SPEC.md](docs/coursework/TAXONOMY_AND_DATA_SPEC.md) and [docs/phases/PHASE_02_SEGMENTATION_BASELINE_SPEC.md](docs/phases/PHASE_02_SEGMENTATION_BASELINE_SPEC.md).

## Global Constraints

- 100% Vectorized operations (zero raw pixel loops).
- Strict shape contracts: `(B, 3, H, W)` RGB float32 for images, `(B, H, W)` int64 for masks, `(B, C, H, W)` float32 for model logits.
- Zero `print()` statements in source modules; use `src.common.logger.get_logger(__name__)`.
- Lossless PNG single-channel uint8 mask handling.
- Full PEP 484 type annotations on all function signatures.

## Review Focus

- Mask resizing corrupting integer class labels: Verify that mask resizing strictly uses `InterpolationMode.NEAREST` or equivalent integer indexing to avoid interpolated class floats.
- Boundary IoU boundary calculation performance: Ensure contour/boundary extraction is fully vectorized using 2D max-pooling or difference of dilation, avoiding OpenCV unvectorized loops.
- Class weight indexing mismatch: Ensure cross-entropy class weights match num_classes ($6$) defined in the config.
- Single-channel vs multi-channel mask loading: Ensure grayscale loader converts PIL images to 2D numpy arrays of shape `(H, W)`.
- Channel dimension mismatch on batching: Ensure `images` is strictly `(B, 3, H, W)` and `masks` is `(B, H, W)`.

---

### Task 1: Segmentation Metrics (mIoU & Boundary IoU)

**Files:**
- Modify: `src/common/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `pred_masks: np.ndarray | torch.Tensor`, `target_masks: np.ndarray | torch.Tensor`, `num_classes: int`.
- Produces: `compute_miou(preds, targets, num_classes) -> dict[str, float]` and `compute_boundary_iou(preds, targets, num_classes, dilation_radius=3) -> dict[str, float]`.

- [ ] **Step 1: Write unit tests for segmentation metrics in `tests/test_metrics.py`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `compute_miou` and `compute_boundary_iou` in `src/common/metrics.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit changes**

---

### Task 2: Paired Occlusion Dataset & DataModule

**Files:**
- Create: `src/data/occlusion_dataset.py`
- Modify: `src/data/__init__.py`
- Test: `tests/test_occlusion_dataset.py`

**Interfaces:**
- Consumes: Image file paths, mask file paths, target size `(512, 512)`.
- Produces: `OcclusionDataset(image_paths, mask_paths, is_train=True)` yielding `{"image": torch.Tensor (3, H, W), "mask": torch.Tensor (H, W)}`.

- [ ] **Step 1: Write unit tests in `tests/test_occlusion_dataset.py`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `OcclusionDataset` and synchronized transforms in `src/data/occlusion_dataset.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit changes**

---

### Task 3: Segmentation Baseline Model & Builder Wiring

**Files:**
- Create: `src/models/deep/segmentation.py`
- Modify: `src/models/builder.py`
- Modify: `src/models/deep/__init__.py`
- Test: `tests/test_occlusion_model.py`

**Interfaces:**
- Consumes: `images: torch.Tensor` of shape `(B, 3, H, W)`.
- Produces: `logits: torch.Tensor` of shape `(B, 6, H, W)`.

- [ ] **Step 1: Write unit tests in `tests/test_occlusion_model.py`**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `FacialOcclusionSegmenter` in `src/models/deep/segmentation.py` and wire `architecture="occlusion_segmenter"` into `src/models/builder.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit changes**
