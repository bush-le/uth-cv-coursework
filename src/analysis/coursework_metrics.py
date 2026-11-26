"""Single source of truth collecting live dataset, active learning, training, and benchmark metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def fmt_int(value: int) -> str:
    """Format an integer using Vietnamese dot thousands separators."""
    return f"{int(value):,}".replace(",", ".")


def fmt_pct(value: float, decimals: int = 2, is_ratio: bool = True) -> str:
    """Format a ratio or percentage with fixed decimals and '%' suffix."""
    pct = float(value) * 100.0 if is_ratio else float(value)
    return f"{pct:.{decimals}f}%"


def _count_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s:
                records.append(json.loads(line_s))
    return records


def _split_real_fake(records: List[Dict[str, Any]]) -> Tuple[int, int]:
    reals = sum(1 for r in records if int(r.get("label", 0)) == 0)
    fakes = len(records) - reals
    return reals, fakes


def _parse_pct(text: str) -> float:
    return float(str(text).strip().rstrip("%"))


def collect_live_coursework_metrics(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load verified live metrics across manifests, Active Learning logs, Quad-Vault, 6K OOD, and Universal 20.4K."""
    root = Path(repo_root) if repo_root is not None else Path(".")

    # 1. Real partitions & synthetic manifests
    real_tr = _count_jsonl(root / "data/partitions/train_reals.jsonl")
    real_va = _count_jsonl(root / "data/partitions/val_reals.jsonl")
    real_te = _count_jsonl(root / "data/partitions/test_reals.jsonl")

    b1_raw = _count_jsonl(root / "data/synthetic/fake_manifest.jsonl")
    b1_cur = _count_jsonl(root / "data/synthetic/curated_manifest.jsonl")
    r1_raw = _count_jsonl(root / "data/active_learning/round_1/candidates_raw.jsonl")
    r1_cur = _count_jsonl(root / "data/active_learning/round_1/candidates_curated.jsonl")
    r2_raw = _count_jsonl(root / "data/active_learning/round_2/candidates_raw.jsonl")
    r2_cur = _count_jsonl(root / "data/active_learning/round_2/candidates_curated.jsonl")

    rv51_raw = _count_jsonl(root / "data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl")
    rv51_cur_path = root / "data/external_benchmarks/synthetic_face_dataset_6k/curated_manifest.jsonl"
    rv51_cur = _count_jsonl(rv51_cur_path) if rv51_cur_path.is_file() else rv51_raw

    sd15_raw_total = len(b1_raw) + len(r1_raw) + len(r2_raw)
    sd15_cur_total = len(b1_cur) + len(r1_cur) + len(r2_cur)
    fake_raw_total = sd15_raw_total + len(rv51_raw)
    fake_cur_total = sd15_cur_total + len(rv51_cur)

    # 2. Benchmark splits D0 -> D3
    d0_tr = _count_jsonl(root / "data/benchmarks/D0_baseline/train.jsonl")
    d0_va = _count_jsonl(root / "data/benchmarks/D0_baseline/val.jsonl")
    master_te = _count_jsonl(root / "data/benchmarks/master_frozen_test.jsonl")
    d1_tr = _count_jsonl(root / "data/benchmarks/D1_augmented/train.jsonl")
    d2_tr = _count_jsonl(root / "data/benchmarks/D2_hard_boosted/train.jsonl")
    d3_tr = _count_jsonl(root / "data/benchmarks/D3_clean/train.jsonl")
    d3_va = _count_jsonl(root / "data/benchmarks/D3_clean/val.jsonl")
    unseen_te = _count_jsonl(root / "data/benchmarks/D3_clean/unseen_rv51_test.jsonl")

    d0_tr_r, d0_tr_f = _split_real_fake(d0_tr)
    d0_va_r, d0_va_f = _split_real_fake(d0_va)
    mte_r, mte_f = _split_real_fake(master_te)
    d3_tr_r, d3_tr_f = _split_real_fake(d3_tr)
    d3_va_r, d3_va_f = _split_real_fake(d3_va)
    ute_r, ute_f = _split_real_fake(unseen_te)

    d3_hard_cnt = sum(1 for r in d3_tr if int(r.get("label", 0)) == 1 and bool(r.get("is_hard_sample", False)))
    d3_rv51_cnt = sum(
        1
        for r in d3_tr
        if int(r.get("label", 0)) == 1 and str(r.get("manipulation_type", "")).startswith("rv51_")
    )
    d3_sd_nonhard_cnt = d3_tr_f - d3_hard_cnt - d3_rv51_cnt

    # 3. Hard samples & Active Learning logs
    r1_hard = _count_jsonl(root / "data/active_learning/round_1/hard_samples.jsonl")
    r2_hard = _count_jsonl(root / "data/active_learning/round_2/hard_samples.jsonl")
    chal_recs = _count_jsonl(root / "data/active_learning/cumulative_challenge_test.jsonl")
    chal_reals, chal_fakes = _split_real_fake(chal_recs)
    hs_all = _count_jsonl(root / "data/hard_samples/hard_samples_manifest.jsonl")
    hs_below_05 = sum(1 for r in hs_all if float(r.get("baseline_score", 0.0)) < 0.5)
    hs_ge_05 = len(hs_all) - hs_below_05

    cl_path = root / "experiments/logs/closed_loop_metrics.json"
    cl = json.loads(cl_path.read_text(encoding="utf-8")) if cl_path.is_file() else {"rounds": [{}, {}]}
    r1_log = cl["rounds"][0] if len(cl.get("rounds", [])) > 0 else {}
    r2_log = cl["rounds"][1] if len(cl.get("rounds", [])) > 1 else {}

    r1_mstat = r1_log.get("mining_stats", {})
    r2_mstat = r2_log.get("mining_stats", {})

    # 4. Quad-Vault summary
    quad_path = root / "experiments/benchmarks/tables/benchmark_5models_quad_vault_summary.json"
    quad = json.loads(quad_path.read_text(encoding="utf-8")) if quad_path.is_file() else {}

    def _q(model_key: str, vault_key: str) -> Dict[str, Any]:
        return quad.get(model_key, {}).get(vault_key, {})

    # 5. M3 training summary
    m3_sum_path = root / "experiments/benchmarks/tables/m3_training_evaluation_summary.json"
    m3_sum = json.loads(m3_sum_path.read_text(encoding="utf-8")) if m3_sum_path.is_file() else {}

    # 6. Legacy 6K (512 & 224) + NPZ profiling
    l512_path = root / "experiments/logs/legacy_6k_benchmark_results.json"
    l224_path = root / "experiments/logs/legacy_6k_benchmark_results_224px.json"
    l512 = json.loads(l512_path.read_text(encoding="utf-8")) if l512_path.is_file() else {"models": {}}
    l224 = json.loads(l224_path.read_text(encoding="utf-8")) if l224_path.is_file() else {"models": {}}

    lnpz_path = root / "experiments/logs/legacy_6k_predictions.npz"
    prof = {"byp": 0, "unc": 0, "union": 0, "m3_byp": 0}
    if lnpz_path.is_file():
        lnpz = np.load(lnpz_path, allow_pickle=True)
        m2p = lnpz["m2_probs"]
        m3p = lnpz["m3_probs"]
        prof["byp"] = int(np.sum(m2p < 0.5))
        prof["unc"] = int(np.sum(np.abs(m2p - 0.5) <= 0.15))
        prof["union"] = int(np.sum((m2p < 0.5) | (np.abs(m2p - 0.5) <= 0.15)))
        prof["m3_byp"] = int(np.sum(m3p < 0.5)) if len(m3p) else 0

    # 7. Universal Benchmark 20.4K
    u_global: Dict[str, Dict[str, Any]] = {}
    ug_path = root / "experiments/benchmarks/tables/benchmark_5models_global_summary.csv"
    if ug_path.is_file():
        with open(ug_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u_global[row["Model"]] = row

    u_fam: Dict[str, Dict[str, str]] = {}
    uf_path = root / "experiments/benchmarks/tables/benchmark_5models_family_summary.csv"
    if uf_path.is_file():
        with open(uf_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u_fam[row["Family"]] = row

    u_meth: Dict[str, Dict[str, str]] = {}
    um_path = root / "experiments/benchmarks/tables/benchmark_5models_method_summary.csv"
    if um_path.is_file():
        with open(um_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                u_meth[row["Method"]] = row

    ff_stats = {"total": 1655, "fp_05": 0, "fpr_05": 0.0, "fp_085": 0, "fpr_085": 0.0, "ffhq_fpr_05": 0.0}
    up_path = root / "experiments/benchmarks/logs/benchmark_5models_predictions.npz"
    if up_path.is_file():
        unpz = np.load(up_path, allow_pickle=True)
        y = unpz["y_true"]
        src = unpz["sources"]
        m3u = unpz["probs_M3_dual_engine"]
        ff_mask = (y == 0) & (src == "ff++_real")
        ffhq_mask = (y == 0) & (src == "ffhq_real")
        if int(ff_mask.sum()) > 0:
            ff_p = m3u[ff_mask]
            ff_stats = {
                "total": int(ff_mask.sum()),
                "fp_05": int(np.sum(ff_p >= 0.5)),
                "fpr_05": float(np.mean(ff_p >= 0.5) * 100.0),
                "fp_085": int(np.sum(ff_p >= 0.85)),
                "fpr_085": float(np.mean(ff_p >= 0.85) * 100.0),
                "ffhq_fpr_05": float(np.mean(m3u[ffhq_mask] >= 0.5) * 100.0) if int(ffhq_mask.sum()) > 0 else 0.0,
            }

    return {
        "real": {
            "total": len(real_tr) + len(real_va) + len(real_te),
            "train": len(real_tr),
            "val": len(real_va),
            "test": len(real_te),
        },
        "curation": {
            "b1_raw": len(b1_raw),
            "b1_cur": len(b1_cur),
            "b1_rej": len(b1_raw) - len(b1_cur),
            "b1_pass": len(b1_cur) / max(1, len(b1_raw)) * 100.0,
            "r1_raw": len(r1_raw),
            "r1_cur": len(r1_cur),
            "r1_rej": len(r1_raw) - len(r1_cur),
            "r1_pass": len(r1_cur) / max(1, len(r1_raw)) * 100.0,
            "r2_raw": len(r2_raw),
            "r2_cur": len(r2_cur),
            "r2_rej": len(r2_raw) - len(r2_cur),
            "r2_pass": len(r2_cur) / max(1, len(r2_raw)) * 100.0,
            "sd15_raw": sd15_raw_total,
            "sd15_cur": sd15_cur_total,
            "sd15_rej": sd15_raw_total - sd15_cur_total,
            "sd15_pass": sd15_cur_total / max(1, sd15_raw_total) * 100.0,
            "rv51_raw": len(rv51_raw),
            "rv51_cur": len(rv51_cur),
            "rv51_rej": len(rv51_raw) - len(rv51_cur),
            "fake_raw_total": fake_raw_total,
            "fake_cur_total": fake_cur_total,
        },
        "splits": {
            "d0_train": len(d0_tr),
            "d0_train_real": d0_tr_r,
            "d0_train_fake": d0_tr_f,
            "d0_val": len(d0_va),
            "d0_val_real": d0_va_r,
            "d0_val_fake": d0_va_f,
            "master_test": len(master_te),
            "master_test_real": mte_r,
            "master_test_fake": mte_f,
            "d1_train": len(d1_tr),
            "d2_train": len(d2_tr),
            "d3_train": len(d3_tr),
            "d3_train_real": d3_tr_r,
            "d3_train_fake": d3_tr_f,
            "d3_val": len(d3_va),
            "d3_val_real": d3_va_r,
            "d3_val_fake": d3_va_f,
            "unseen_rv51_total": len(unseen_te),
            "unseen_rv51_real": ute_r,
            "unseen_rv51_fake": ute_f,
            "d3_hard": d3_hard_cnt,
            "d3_sd_nonhard": d3_sd_nonhard_cnt,
            "d3_rv51_clean": d3_rv51_cnt,
        },
        "al": {
            "r1_total": len(r1_hard),
            "r1_train": int(r1_log.get("hard_train_count", 0)),
            "r1_test": int(r1_log.get("hard_test_count", 0)),
            "r1_fn": int(r1_mstat.get("false_negative_count", 0)),
            "r1_unc": int(r1_mstat.get("uncertain_count", 0)),
            "r1_temp": float(r1_mstat.get("temperature_used", 1.0)),
            "r2_total": len(r2_hard),
            "r2_train": int(r2_log.get("hard_train_count", 0)),
            "r2_test": int(r2_log.get("hard_test_count", 0)),
            "r2_fn": int(r2_mstat.get("false_negative_count", 0)),
            "r2_unc": int(r2_mstat.get("uncertain_count", 0)),
            "r2_temp": float(r2_mstat.get("temperature_used", 1.0)),
            "total_hard": len(r1_hard) + len(r2_hard),
            "total_hard_train": int(r1_log.get("hard_train_count", 0)) + int(r2_log.get("hard_train_count", 0)),
            "total_hard_test": int(r1_log.get("hard_test_count", 0)) + int(r2_log.get("hard_test_count", 0)),
            "total_fn": int(r1_mstat.get("false_negative_count", 0)) + int(r2_mstat.get("false_negative_count", 0)),
            "total_unc": int(r1_mstat.get("uncertain_count", 0)) + int(r2_mstat.get("uncertain_count", 0)),
            "hs_below_05": hs_below_05,
            "hs_ge_05": hs_ge_05,
            "challenge_total": len(chal_recs),
            "challenge_real": chal_reals,
            "challenge_fake": chal_fakes,
            "r1_mode_errors": r1_log.get("attribution", {}).get("mode_errors", {}),
            "r2_mode_errors": r2_log.get("attribution", {}).get("mode_errors", {}),
            "r2_quotas": r2_log.get("mode_quotas", {}),
            "r2_region_errors": r2_log.get("attribution", {}).get("region_errors", {}),
            "r2_keyword_errors": r2_log.get("attribution", {}).get("keyword_errors", {}),
        },
        "quad": {
            "M0": {
                "master": _q("M0 (Baseline)", "Master_Frozen_Test"),
                "challenge": _q("M0 (Baseline)", "Challenge_Hard_Test"),
                "unseen": _q("M0 (Baseline)", "Unseen_RV51_OOD_Test"),
            },
            "M1": {
                "master": _q("M1 (Active R1)", "Master_Frozen_Test"),
                "challenge": _q("M1 (Active R1)", "Challenge_Hard_Test"),
                "unseen": _q("M1 (Active R1)", "Unseen_RV51_OOD_Test"),
            },
            "M2": {
                "master": _q("M2 (Cold-Start)", "Master_Frozen_Test"),
                "challenge": _q("M2 (Cold-Start)", "Challenge_Hard_Test"),
                "unseen": _q("M2 (Cold-Start)", "Unseen_RV51_OOD_Test"),
            },
            "M2_star": {
                "master": _q("M2* (Refined)", "Master_Frozen_Test"),
                "challenge": _q("M2* (Refined)", "Challenge_Hard_Test"),
                "unseen": _q("M2* (Refined)", "Unseen_RV51_OOD_Test"),
            },
            "M3": {
                "master": _q("M3 (Dual-Engine Champion)", "Master_Frozen_Test"),
                "challenge": _q("M3 (Dual-Engine Champion)", "Challenge_Hard_Test"),
                "unseen": _q("M3 (Dual-Engine Champion)", "Unseen_RV51_OOD_Test"),
            },
        },
        "m3_summary": m3_sum,
        "legacy6k": {
            "m512": l512.get("models", {}),
            "m224": l224.get("models", {}),
            "profiling": prof,
        },
        "universal": {
            "global": u_global,
            "family": u_fam,
            "method": u_meth,
            "ff": ff_stats,
        },
    }
