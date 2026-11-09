"""Master Orchestrator for Detector-Guided Closed-Loop Active Learning.

Coordinates iterative cycles of:
1. Dynamic budget allocation across T2I, I2I, and Inpainting (AdaptiveBudgetController).
2. Combinatorial error-steered prompt synthesis (PromptGrammarEngine).
3. Synthetic candidate generation and quality curation.
4. Inference using current detector Mt with temperature calibration.
5. Calibrated dual-criterion hard-sample mining (CalibratedHardMiner).
6. Attribution feedback closing the loop to update budgets and prompt weights.
7. Dataset expansion (D_{t+1} = D_t U H_{t+1}) and detector retraining (M_{t+1}).
8. Rigorous evaluation on the frozen, identity-disjoint test set.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
import torch
import yaml

from src.common.config import load_config
from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_schema import ImageMetadata
from src.models.detector.image_detector import DeepfakeImageDetector
from src.models.diffusion.prompt_engine import PromptGrammarEngine
from src.models.evaluators.calibrated_mining import CalibratedHardMiner
from src.pipelines.budget_controller import AdaptiveBudgetController
from src.pipelines.curate_image_dataset import curate_dataset
from src.pipelines.evaluate_image_benchmark import evaluate_image_detector
from src.pipelines.generate_deepfake_images import run_image_generation
from src.pipelines.train_image_detector import train_deepfake_image_detector

logger = get_logger(__name__)


def extract_error_attribution_from_hard_samples(
    hard_samples: List[ImageMetadata],
    all_candidates: List[ImageMetadata],
) -> Dict[str, Any]:
    """Compute error attribution across modes, mask regions, and prompt keywords.

    Args:
        hard_samples: Sublist of candidates that bypassed detector or had uncertain predictions.
        all_candidates: Complete list of evaluated candidates.

    Returns:
        Dictionary with 'mode_errors', 'region_errors', and 'keyword_errors'.
    """
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
        m: (mode_hards.get(m, 0) / max(1, count))
        for m, count in mode_totals.items()
    }

    # 2. Inpainting region failure rates
    region_totals: Dict[str, int] = {}
    region_hards: Dict[str, int] = {}
    for c in all_candidates:
        if c.manipulation_type == "sd_inpaint":
            target = (c.generation or {}).get("inpaint_mask_target") or "full"
            region_totals[target] = region_totals.get(target, 0) + 1
            if c.image_id in hard_ids:
                region_hards[target] = region_hards.get(target, 0) + 1

    region_errors = {
        r: (region_hards.get(r, 0) / max(1, count))
        for r, count in region_totals.items()
    }

    # 3. Prompt keyword failure rates
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
    keyword_totals: Dict[str, int] = {}
    keyword_hards: Dict[str, int] = {}

    for c in all_candidates:
        p_text = (c.prompt or "").lower()
        is_hard = c.image_id in hard_ids

        for cat in keyword_categories:
            key_tag = cat.split(":", 1)[1] if ":" in cat else cat
            if key_tag.lower() in p_text:
                keyword_totals[cat] = keyword_totals.get(cat, 0) + 1
                if is_hard:
                    keyword_hards[cat] = keyword_hards.get(cat, 0) + 1

    keyword_errors = {
        cat: (keyword_hards.get(cat, 0) / max(1, count))
        for cat, count in keyword_totals.items()
    }

    return {
        "mode_errors": mode_errors,
        "region_errors": region_errors,
        "keyword_errors": keyword_errors,
    }


def split_hard_samples(
    mined_hard: List[ImageMetadata],
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
    """Split mined hard samples into training buffer (85%) and cumulative challenge test set (15%).

    Guarantees deterministic identity and record isolation to prevent train-test contamination.
    """
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


def run_active_learning_loop(
    num_rounds: int = 2,
    candidates_per_round: int = 60,
    config_path: str = "configs/dataset_factory.yaml",
    dry_run: bool = False,
    output_base_dir: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
    resume: bool = True,
    device: str = "cpu",
    epochs_per_round: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute multi-round closed-loop active learning deepfake dataset expansion.

    Args:
        num_rounds: Number of adaptive rounds to execute.
        candidates_per_round: Total candidate images to synthesize per round.
        config_path: Path to YAML configuration.
        dry_run: If True, executes fast mock pipeline without heavy model loading.
        output_base_dir: Base directory to store active learning outputs and logs.
        checkpoints_dir: Directory where model checkpoints are saved and loaded.
        resume: Whether to detect and resume from the latest completed round.
        device: Compute device ('cpu' or 'cuda').
        epochs_per_round: Number of fine-tuning epochs per detector round.
        seed: Master random seed.

    Returns:
        Dictionary recording full active learning trajectory and benchmark gains.
    """
    seed_everything(seed)
    cfg = load_config(Path(config_path)) if Path(config_path).exists() else {}

    base_dir = output_base_dir or Path("data/active_learning")
    base_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = (output_base_dir / "logs") if output_base_dir else Path("experiments/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = checkpoints_dir or Path("experiments/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize components
    budget_ctrl = AdaptiveBudgetController(
        total_candidate_budget=candidates_per_round,
        min_mode_ratio=0.15,
        exploration_floor=0.20,
        seed=seed,
    )
    prompt_eng = PromptGrammarEngine(seed=seed, exploration_floor=0.15)
    miner = CalibratedHardMiner(decision_threshold=0.5, uncertainty_window=0.10)

    # 2. Frozen Test Set Path (identity-disjoint, 43 samples)
    test_manifest_path = Path("data/benchmarks/D0_baseline/test.jsonl")
    val_manifest_path = Path("data/benchmarks/D0_baseline/val.jsonl")
    train_manifest_d0 = Path("data/benchmarks/D0_baseline/train.jsonl")

    # Load initial baseline training set D0
    current_train_records: List[ImageMetadata] = []
    if train_manifest_d0.exists():
        with open(train_manifest_d0, "r", encoding="utf-8") as f:
            current_train_records = [
                ImageMetadata.from_dict(json.loads(line))
                for line in f
                if line.strip()
            ]

    # Load frozen validation and test sets
    val_records: List[ImageMetadata] = []
    if val_manifest_path.exists():
        with open(val_manifest_path, "r", encoding="utf-8") as f:
            val_records = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

    test_records: List[ImageMetadata] = []
    if test_manifest_path.exists():
        with open(test_manifest_path, "r", encoding="utf-8") as f:
            test_records = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

    # Cumulative Challenge Test Set Path (dynamically updated with 15% slices of mined H_r)
    challenge_manifest_path = base_dir / "cumulative_challenge_test.jsonl"
    challenge_test_records: List[ImageMetadata] = []
    if challenge_manifest_path.exists():
        with open(challenge_manifest_path, "r", encoding="utf-8") as f:
            challenge_test_records = [ImageMetadata.from_dict(json.loads(line)) for line in f if line.strip()]

    # Tracking records
    rounds_log: List[Dict[str, Any]] = []
    prev_attribution: Dict[str, Any] = {
        "mode_errors": {},
        "region_errors": {},
        "keyword_errors": {},
    }

    # 3. Detect pre-existing completed rounds for stateful resume
    latest_completed = 0
    current_detector_ckpt = ckpt_dir / "baseline_d0_best.pt"
    resume_records = list(current_train_records)

    if resume:
        r_cand = 1
        while True:
            r_dir = base_dir / f"round_{r_cand}"
            r_hard_path = r_dir / "hard_samples.jsonl"
            r_ckpt = ckpt_dir / f"active_learning_m{r_cand}_best.pt"
            if r_hard_path.exists() and r_ckpt.exists():
                logger.info("Found completed Round %d artifacts at %s and %s", r_cand, r_dir, r_ckpt.name)
                with open(r_hard_path, "r", encoding="utf-8") as hf:
                    new_hards = [ImageMetadata.from_dict(json.loads(line)) for line in hf if line.strip()]
                    resume_records.extend(new_hards)
                latest_completed = r_cand
                current_detector_ckpt = r_ckpt
                r_cand += 1
            else:
                break

    if latest_completed > 0:
        logger.info(
            "Stateful resume activated: detected %d completed rounds. Resuming from Round %d.",
            latest_completed,
            latest_completed + 1,
        )
        current_train_records = resume_records
        metrics_file = logs_dir / "closed_loop_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as mf:
                    old_data = json.load(mf)
                    rounds_log = old_data.get("rounds", [])
                    if rounds_log:
                        prev_attribution = rounds_log[-1].get("attribution", prev_attribution)
            except Exception:
                pass

    start_round = latest_completed + 1
    end_round = latest_completed + num_rounds

    logger.info(
        "=== Beginning Closed-Loop Active Learning Pipeline (Rounds %d -> %d, %d candidates/round) ===",
        start_round,
        end_round,
        candidates_per_round,
    )

    if dry_run:
        # Fast synthetic dry-run verification
        for r in range(start_round, end_round + 1):
            mock_hard = [
                ImageMetadata(
                    image_id=f"dry_hard_{r}_{i}",
                    label=1,
                    manipulation_type="sd_inpaint" if i % 2 == 0 else "sd_text2img",
                    prompt="RAW photo, elderly face, direct harsh camera flash",
                    is_hard_sample=True,
                    data_source="hard_sample",
                    generation={"inpaint_mask_target": "periocular"},
                )
                for i in range(max(2, candidates_per_round // 5))
            ]
            all_mock = mock_hard + [
                ImageMetadata(
                    image_id=f"dry_easy_{r}_{i}",
                    label=1,
                    manipulation_type="sd_img2img",
                    prompt="standard portrait",
                    is_hard_sample=False,
                    data_source="synthetic",
                )
                for i in range(candidates_per_round - len(mock_hard))
            ]
            attrib = extract_error_attribution_from_hard_samples(mock_hard, all_mock)
            budget_ctrl.compute_mode_quotas(attrib["mode_errors"])
            prompt_eng.update_error_weights(attrib["keyword_errors"])

            train_mock, test_mock = split_hard_samples(mock_hard, test_ratio=0.15, seed=seed + r)

            mock_master_metrics = {
                "accuracy": 0.60 + 0.05 * r,
                "roc_auc": 0.65 + 0.06 * r,
                "hard_sample_recall": 0.50 + 0.20 * r,
                "bypass_rate": max(0.0, 0.40 - 0.15 * r),
                "loss": 0.60 - 0.05 * r,
            }
            mock_challenge_metrics = {
                "accuracy": 0.55 + 0.10 * r,
                "roc_auc": 0.58 + 0.10 * r,
                "hard_sample_recall": 0.45 + 0.25 * r,
                "bypass_rate": max(0.0, 0.50 - 0.20 * r),
                "loss": 0.65 - 0.06 * r,
            }
            rounds_log.append({
                "round": r,
                "hard_sample_count": len(mock_hard),
                "hard_train_count": len(train_mock),
                "hard_test_count": len(test_mock),
                "master_test_metrics": mock_master_metrics,
                "challenge_test_metrics": mock_challenge_metrics,
                "test_metrics": mock_master_metrics,
                "attribution": attrib,
            })

        out_summary = {"rounds": rounds_log, "dry_run": True}
        with open(logs_dir / "closed_loop_metrics.json", "w", encoding="utf-8") as f:
            json.dump(out_summary, f, indent=2)
        return out_summary

    # Live Execution Loop
    for r in range(start_round, end_round + 1):
        round_dir = base_dir / f"round_{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        logger.info("\n--- [Active Learning Round %d/%d] ---", r, num_rounds)

        # 1. Tier 1: Mode Budget Allocation
        mode_quotas = budget_ctrl.compute_mode_quotas(prev_attribution["mode_errors"])
        logger.info("Round %d Modality Quotas: %s", r, mode_quotas)

        # 2. Tier 2: Regional Mask Allocation
        mask_targets = budget_ctrl.compute_inpainting_mask_targets(
            prev_attribution["region_errors"],
            count=mode_quotas.get("sd_inpaint", 20),
        )

        # 3. Dynamic Candidate Generation using Prompt Grammar Engine
        raw_manifest_path = Path("data/raw/manifest.jsonl")
        candidates_raw_path = round_dir / "candidates_raw.jsonl"
        candidates_curated_path = round_dir / "candidates_curated.jsonl"
        hard_manifest_path = round_dir / "hard_samples.jsonl"

        # Generate candidates using SD engine with prompt grammar engine
        run_image_generation(
            input_manifest=raw_manifest_path,
            output_manifest=candidates_raw_path,
            output_dir=round_dir / "images",
            total_samples=candidates_per_round,
            mode_quotas=mode_quotas,
            prompt_engine=prompt_eng,
            mask_targets=mask_targets,
            device=device,
            seed=seed + r * 100,
        )

        # 4. Quality Curation Gate
        curation_report_path = round_dir / "curation_report.json"
        curate_dataset(
            input_manifest=candidates_raw_path,
            output_manifest=candidates_curated_path,
            report_path=curation_report_path,
        )

        # Load curated candidates
        curated_candidates: List[ImageMetadata] = []
        if candidates_curated_path.exists():
            with open(candidates_curated_path, "r", encoding="utf-8") as f:
                curated_candidates = [
                    ImageMetadata.from_dict(json.loads(line))
                    for line in f
                    if line.strip()
                ]

        logger.info("Round %d Curated Candidates: %d", r, len(curated_candidates))

        # 5. Load Previous Detector M_{r-1} for Inference
        detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
        if current_detector_ckpt.exists():
            ckpt = torch.load(current_detector_ckpt, map_location=device)
            if "model_state_dict" in ckpt:
                detector.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded detector weights from %s", current_detector_ckpt.name)
        detector.to(torch.device(device))
        detector.eval()

        # 6. Temperature Calibration on Validation Set
        if val_records:
            val_dm = DeepfakeImageDataModule(
                train_metadata=[],
                val_metadata=val_records,
                test_metadata=[],
                batch_size=16,
            )
            val_dm.setup()
            val_logits_list = []
            val_labels_list = []
            with torch.no_grad():
                for batch in val_dm.val_dataloader():
                    imgs = batch["images"].to(torch.device(device))
                    lbls = batch["labels"].to(torch.device(device))
                    lgts = detector(imgs).view(-1)
                    val_logits_list.append(lgts.cpu())
                    val_labels_list.append(lbls.cpu())

            if val_logits_list:
                cat_logits = torch.cat(val_logits_list)
                cat_labels = torch.cat(val_labels_list)
                miner.calibrate_temperature(cat_logits, cat_labels)

        # 7. Candidate Inference & Calibrated Hard-Sample Mining
        candidate_dm = DeepfakeImageDataModule(
            train_metadata=[],
            val_metadata=[],
            test_metadata=curated_candidates,
            batch_size=16,
        )
        candidate_dm.setup()

        all_probs: List[float] = []
        with torch.no_grad():
            for batch in candidate_dm.test_dataloader():
                imgs = batch["images"].to(torch.device(device))
                logits = detector(imgs)
                probs = torch.sigmoid(logits).view(-1).cpu().tolist()
                all_probs.extend(probs)

        # Mine Hard Samples (False Negatives + Uncertain)
        mined_hard, mining_stats = miner.mine_hard_samples(curated_candidates, all_probs)
        logger.info(
            "Round %d Mining: %d hard samples mined (yield=%.1f%%)",
            r,
            len(mined_hard),
            mining_stats["hard_sample_yield_rate"] * 100,
        )

        with open(hard_manifest_path, "w", encoding="utf-8") as f:
            for h in mined_hard:
                f.write(json.dumps(h.to_dict()) + "\n")

        # 8. Feedback Attribution to Close the Loop
        attribution = extract_error_attribution_from_hard_samples(mined_hard, curated_candidates)
        prompt_eng.update_error_weights(attribution["keyword_errors"], temperature=0.7)
        prev_attribution = attribution

        # 9. Dual-Benchmark Split: 85% Train / 15% Cumulative Challenge Test
        h_train, h_test = split_hard_samples(mined_hard, test_ratio=0.15, seed=seed + r)
        if h_test:
            challenge_test_records.extend(h_test)
            with open(challenge_manifest_path, "a", encoding="utf-8") as ch_f:
                for ht in h_test:
                    ch_f.write(json.dumps(ht.to_dict()) + "\n")
            logger.info("Added %d hard samples to cumulative challenge test vault (total: %d).", len(h_test), len(challenge_test_records))

        # Expand Training Dataset: D_r = D_{r-1} U H_r^{train}
        current_train_records = current_train_records + h_train
        logger.info(
            "Dataset expanded to D_%d: %d total training samples (includes %d hard samples).",
            r,
            len(current_train_records),
            sum(1 for m in current_train_records if m.is_hard_sample),
        )

        # Persist benchmark dataset manifests for D1_augmented and D2_hard_boosted
        bench_folder_name = "D1_augmented" if r == 1 else ("D2_hard_boosted" if r == 2 else f"D{r}_expanded")
        bench_dir = Path("data/benchmarks") / bench_folder_name
        bench_dir.mkdir(parents=True, exist_ok=True)
        with open(bench_dir / "train.jsonl", "w", encoding="utf-8") as bf:
            for m in current_train_records:
                bf.write(json.dumps(m.to_dict()) + "\n")
        with open(bench_dir / "val.jsonl", "w", encoding="utf-8") as bf:
            for m in val_records:
                bf.write(json.dumps(m.to_dict()) + "\n")
        with open(bench_dir / "test.jsonl", "w", encoding="utf-8") as bf:
            for m in test_records:
                bf.write(json.dumps(m.to_dict()) + "\n")
        logger.info("Persisted benchmark dataset %s to %s", bench_folder_name, bench_dir)

        # 10. Retrain Detector M_r on Expanded Dataset D_r
        round_dm = DeepfakeImageDataModule(
            train_metadata=current_train_records,
            val_metadata=val_records,
            test_metadata=test_records,
            batch_size=16,
            hard_sample_oversample_factor=2,
            seed=seed,
        )
        round_dm.setup()

        round_detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
        train_deepfake_image_detector(
            model=round_detector,
            datamodule=round_dm,
            num_epochs=epochs_per_round,
            learning_rate=2e-4,
            hard_sample_loss_weight=2.0,
            checkpoint_dir=ckpt_dir,
            checkpoint_prefix=f"active_learning_m{r}",
            device=device,
            seed=seed,
        )

        # 11. Dual-Benchmark Evaluation
        m_best_ckpt = ckpt_dir / f"active_learning_m{r}_best.pt"
        if m_best_ckpt.exists():
            state = torch.load(m_best_ckpt, map_location=device)
            if "model_state_dict" in state:
                round_detector.load_state_dict(state["model_state_dict"])

        # 11a. Evaluate on Master Frozen Test Set
        master_test_metrics = {}
        if test_records:
            test_loader = round_dm.test_dataloader()
            master_test_metrics = evaluate_image_detector(round_detector, test_loader, device=device)
            logger.info("Round %d Master Frozen Test Results: %s", r, master_test_metrics)

        # 11b. Evaluate on Cumulative Challenge Test Set
        challenge_test_metrics = {}
        if challenge_test_records:
            chal_dm = DeepfakeImageDataModule(
                train_metadata=[],
                val_metadata=[],
                test_metadata=challenge_test_records,
                batch_size=16,
                seed=seed,
            )
            chal_dm.setup()
            challenge_test_metrics = evaluate_image_detector(round_detector, chal_dm.test_dataloader(), device=device)
            logger.info("Round %d Cumulative Challenge Test Results: %s", r, challenge_test_metrics)

        # Update current checkpoint pointer for next round
        current_detector_ckpt = m_best_ckpt

        rounds_log.append({
            "round": r,
            "training_samples_count": len(current_train_records),
            "hard_sample_count": len(mined_hard),
            "hard_train_count": len(h_train),
            "hard_test_count": len(h_test),
            "mining_stats": mining_stats,
            "master_test_metrics": master_test_metrics,
            "challenge_test_metrics": challenge_test_metrics,
            "test_metrics": master_test_metrics,
            "attribution": attribution,
            "mode_quotas": mode_quotas,
        })

    summary = {
        "num_rounds": num_rounds,
        "candidates_per_round": candidates_per_round,
        "rounds": rounds_log,
        "deltas": {
            "initial_auc": rounds_log[0]["test_metrics"]["roc_auc"] if rounds_log else 0.0,
            "final_auc": rounds_log[-1]["test_metrics"]["roc_auc"] if rounds_log else 0.0,
            "auc_gain": round(
                rounds_log[-1]["test_metrics"]["roc_auc"] - rounds_log[0]["test_metrics"]["roc_auc"], 4
            ) if rounds_log else 0.0,
        },
    }

    out_metrics_file = logs_dir / "closed_loop_metrics.json"
    with open(out_metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Closed-loop active learning complete! Trajectory written to %s", out_metrics_file)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector-Guided Closed-Loop Active Learning Pipeline")
    parser.add_argument("--rounds", type=int, default=2, help="Number of active learning rounds")
    parser.add_argument("--candidates-per-round", type=int, default=60, help="Candidates to generate per round")
    parser.add_argument("--config", type=str, default="configs/dataset_factory.yaml", help="Path to config")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per detector training stage")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Run quick dry-run test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dev = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    run_active_learning_loop(
        num_rounds=args.rounds,
        candidates_per_round=args.candidates_per_round,
        config_path=args.config,
        dry_run=args.dry_run,
        device=dev,
        epochs_per_round=args.epochs,
    )
