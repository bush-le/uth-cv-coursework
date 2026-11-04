# Phase 2: Fine-Grained Segmentation Baseline Specification

- **Motivation/Background**: Formulates the architectural standards, loss configurations, boundary metrics, and training protocol for the downstream occlusion segmentation network.
- **Purpose**: Serve as the technical contract for implementing `src/models/builder.py`, `src/common/metrics.py`, and `src/pipelines/train.py`.
- **Overview Pipeline**: Input normalization -> Hierarchical Transformer / CNN backbone -> Multiscale feature decoding -> Composite loss backpropagation -> Boundary metric evaluation.
- **Detailed Plan**: §1 Model Architectural Standards; §2 Forward Pass & Tensor Shapes; §3 Composite Loss Formulation; §4 Evaluation Metrics (Area & Boundary IoU); §5 Hyperparameters & Optimization Schedule.
- **References**: `docs/PURPOSE.md`, `docs/coursework/TAXONOMY_AND_DATA_SPEC.md`, Xie et al. (SegFormer NeurIPS 2021), Cheng et al. (Boundary IoU CVPR 2021).
- **Created**: 2026-11-04T08:20:00+07:00
- **Last Updated**: 2026-11-04T08:20:00+07:00

---

## 1. Model Architectural Standards

To ensure a balanced baseline that is computationally efficient yet capable of high-frequency boundary capture, we specify two models:

### 1.1 Primary Modern Baseline: SegFormer-B0
- **Backbone**: Hierarchical Transformer (Mix Transformer `MiT-B0`), pretrained on ImageNet-1K.
- **Advantages**: Overlapped patch merging preserves fine boundary context without positional embedding interpolation issues when scaling resolutions. Lightweight ($3.8\text{M}$ parameters, $< 10\text{ ms}$ inference).
- **Decoder**: Lightweight all-MLP decoder aggregating feature maps at $\frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}$ resolutions upsampled to $\frac{H}{4} \times \frac{W}{4}$ and projected to $C_{\text{classes}} = 6$.

### 1.2 Classical Benchmark: DeepLabV3+ (ResNet-50)
- **Backbone**: ResNet-50 with Atrous Spatial Pyramid Pooling (ASPP).
- **Purpose**: Acts as the classical convolutional benchmark against which the Transformer baseline is compared.

---

## 2. Forward Pass & Tensor Shapes

All models must adhere strictly to these contract signatures:

```python
def forward(self, images: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel class logits.

    Args:
        images: Standardized RGB image tensor of shape (B, 3, H, W).
                Expected dtype float32, normalized with ImageNet stats.

    Returns:
        logits: Unnormalized class logits of shape (B, num_classes, H, W).
                Expected dtype float32, num_classes = 6.
    """
```

- **Inference Prediction**:
  $$\hat{Y} = \operatorname{argmax}_{c} \operatorname{Softmax}(\text{logits}, \text{dim}=1) \in \{0, \dots, 5\}^{B \times H \times W}$$

---

## 3. Composite Loss Formulation

Due to severe class imbalance between the predominant background face ($> 80\%$ of pixels) and fine occlusion objects (such as thin glasses frames), standard Cross-Entropy alone produces under-segmented boundaries.

We formulate a composite objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}}$$

Where:
- **Weighted Cross-Entropy Loss**:
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=0}^5 w_c \, y_{i,c} \log(p_{i,c})$$
  with class weights $w_c$ inversely proportional to class frequency.
- **Multi-Class Soft Dice Loss**:
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \sum_{i} p_{i,c} y_{i,c} + \epsilon}{\sum_i p_{i,c}^2 + \sum_i y_{i,c}^2 + \epsilon}$$
  (Computed over foreground classes $c \in \{1, \dots, 5\}$).
- **Default Hyperparameter**: $\lambda_{\text{dice}} = 1.0$, $\epsilon = 10^{-6}$.

---

## 4. Evaluation Metrics (Area & Boundary IoU)

To validate "fine-grained" segmentation, measuring only mIoU is insufficient because area-based IoU penalizes boundary errors weakly on large objects (like masks). We enforce two metrics:

### 4.1 Mean Intersection-over-Union (mIoU)
$$\text{IoU}_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|}, \quad \text{mIoU} = \frac{1}{C} \sum_{c=0}^{C-1} \text{IoU}_c$$

### 4.2 Boundary IoU (BIoU)
Let $P_{c,d} = P_c \cap \text{Boundary}_d(P_c)$ and $G_{c,d} = G_c \cap \text{Boundary}_d(G_c)$ denote pixels within distance $d$ (default $d=3$ pixels) of the predicted and ground-truth contours:
$$\text{BIoU}_c = \frac{|(P_{c,d} \cap G) \cap (G_{c,d} \cap P)|}{|(P_{c,d} \cap G) \cup (G_{c,d} \cap P)|}$$
$$\text{mBIoU} = \frac{1}{C-1} \sum_{c=1}^{C-1} \text{BIoU}_c$$
A high mBIoU directly proves that the model captures delicate contours without boundary bleed.

---

## 5. Hyperparameters & Optimization Schedule

- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 0.01$).
- **Base Learning Rate**: $\eta = 6 \times 10^{-5}$ for Transformer backbone, $10^{-4}$ for decoder head.
- **LR Scheduler**: Cosine Annealing with 5 warmup epochs.
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast(dtype=torch.float16)`).
- **Epochs**: 40 epochs on synthetic pool ($8,000$ images), early stopping patience $= 8$ on validation mIoU.
- **Checkpointing**: Full-state checkpointing saving `best_model.pt` (highest val mIoU) and `last_model.pt`.
