# Experiments & Runtime Artifacts Layer (`/experiments`)

- **Motivation/Background**: Deep learning experiments generate voluminous runtime artifacts (model weights, training checkpoints, tensorboard events, prediction outputs, evaluation logs) that must never pollute git history but need organized namespacing for auditing.
- **Purpose**: Define standards for checkpoint saving, experiment logging, prediction dumps, and run isolation.
- **Overview Pipeline**: Training script runs -> Generates full-state checkpoints & logs into timestamped run directories -> Evaluator parses artifacts.
- **Detailed Plan**: §1 Directory Layout; §2 Checkpoint Conventions; §3 Logging & Telemetry; §4 Retention & Purge Rules.
- **References**: `.agents/rules/LOGGING_CHECKPOINT_RULES.md`, `.agents/rules/MD_CONVENTION.md`, `.agents/AI_REFERENCE.md`.
- **Created**: 2026-11-02T07:22:00+07:00
- **Last Updated**: 2026-11-02T07:22:00+07:00

---

## 1. Directory Layout

```text
experiments/
├── checkpoints/              # Model weights (*.pt, *.pth, *.ckpt) — ignored by Git
│   └── .gitkeep
├── logs/                     # Text logs, run traces, JSONL histories — ignored by Git
│   └── .gitkeep
├── predictions/              # Inferred bounding boxes, masks, visual plots — ignored by Git
│   └── .gitkeep
├── runs/                     # Tensorboard / WandB run directories — ignored by Git
│   └── .gitkeep
└── README.md                 # This governance document
```

---

## 2. Checkpoint Conventions

1. **Full-State Saving**: Checkpoints must include:
   - `epoch`: Current epoch index.
   - `model_state_dict`: Model weights.
   - `optimizer_state_dict`: Optimizer momentum/states.
   - `scheduler_state_dict`: LR scheduler state (if used).
   - `metrics`: Dictionary of validation metrics (e.g. `val_loss`, `val_map`, `val_acc`).
   - `config`: Hyperparameters and configuration snapshot.
2. **Safe Loading**: Always load with `weights_only=True` in PyTorch 2.4+ when loading weights only, or verify security when resuming full state.
3. **Naming**:
   - `best_model.pt`: Checkpoint with highest validation target metric.
   - `last_model.pt`: Most recent epoch checkpoint for uninterrupted resumption.
   - `checkpoint_epoch_<N>.pt`: Intermediate milestones.
