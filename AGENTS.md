# AGENTS.md — `uth-cv-coursework` Agent Charter (Model- & Harness-Agnostic)

Read [`.agents/README.md`](.agents/README.md), [`.agents/rules/`](.agents/rules/), [`.agents/AI_REFERENCE.md`](.agents/AI_REFERENCE.md), and [`docs/PURPOSE.md`](docs/PURPOSE.md) before modifying code.

## 1. Executable Commands
```bash
make help
make test-smoke    # Sanity gate (< 5s) after every edit
make test          # Full test suite
make lint          # Static checks & formatting
bash scripts/run_pipeline.sh train configs/config.yaml
```

## 2. Hard Invariants (Three-Tier Boundaries)
- **[ALWAYS DO]**: Keep domain logic in `src/` (notebooks `notebooks/` strictly for EDA/visualization only); document tensor shapes `(B, C, H, W)` / `(H, W, C)` and BGR<->RGB conversion; seed with `src.common.seed.seed_everything(seed)`; wrap inference in `model.eval()` + `torch.no_grad()` + `torch.cuda.amp.autocast()`; commit atomically once `make test-smoke` passes.
- **[ASK FIRST]**: Modifying dataset splits, label schemas, or training hyperparameters in `configs/`.
- **[NEVER DO]**: Training loops inside `notebooks/`; mock data or uninitialized random weights in production evaluation (`mine_hard_samples.py`, `evaluate.py` must verify real `model_state_dict` checkpoints); absolute local filesystem paths in Markdown docs; monolithic pixel loops or `print()` in `src/` (`src.common.logger.get_logger`).
