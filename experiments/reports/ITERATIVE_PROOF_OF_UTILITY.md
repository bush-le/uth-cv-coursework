# 4-Way Comparative Proof-of-Utility Benchmark Report

| Model | Training Dataset | Test Accuracy | Fake Recall (↑) | Bypass Rate (FNR ↓) | ROC-AUC (↑) | Real FPR (↓) |
|---|---|---|---|---|---|---|
| **M0_baseline** | D0 (Base Real + Fake) | 77.8% | **75.0%** | 25.0% | **0.800** | 20.0% |
| **M_rand** | D1 (D0 + Random Fakes) | 100.0% | **100.0%** | 0.0% | **1.000** | 0.0% |
| **M1_hard** | D2 (D0 + H1 Bypass) | 100.0% | **100.0%** | 0.0% | **1.000** | 0.0% |
| **M2_adaptive** | D3 (D0 + H1 + H2 Feedback) | 100.0% | **100.0%** | 0.0% | **1.000** | 0.0% |

Evaluated on frozen unseen test split: `data/benchmarks/D0_baseline/test.jsonl` (100% Zero Leakage).
