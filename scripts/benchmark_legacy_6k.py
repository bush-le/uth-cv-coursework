import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#!/usr/bin/env python3
"""Run benchmark evaluation of M0, M1, and M2 detectors on the 6k legacy synthetic dataset."""

import argparse
import json
import os
import sys
from pathlib import Path
import time
from typing import Any, Dict, List

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.common.logger import get_logger
from src.models.detector.image_detector import DeepfakeImageDetector

logger = get_logger(__name__)


class Legacy6kDataset(Dataset):
    """Dataset for 6,000 legacy synthetic face images."""

    def __init__(self, records: List[Dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = rec["image_path"]
        with Image.open(img_path) as im:
            img = im.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        return {
            "image": tensor,
            "task": rec.get("task", "unknown"),
            "sub_region": rec.get("sub_region") or "none",
            "image_path": img_path,
        }


def evaluate_model_on_6k(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate detector on 6k fake images and compute detection metrics."""
    model.eval()
    model.to(device)

    all_probs: List[float] = []
    all_tasks: List[str] = []
    all_regions: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

            all_probs.extend(probs)
            all_tasks.extend(batch["task"])
            all_regions.extend(batch["sub_region"])

    probs_arr = np.array(all_probs, dtype=np.float32)
    detected = probs_arr >= 0.5
    total = len(probs_arr)

    overall_recall = float(np.mean(detected))
    overall_bypass = float(1.0 - overall_recall)
    mean_prob = float(np.mean(probs_arr))

    # Breakdown by task
    by_task: Dict[str, Dict[str, float]] = {}
    tasks_unique = sorted(list(set(all_tasks)))
    for t in tasks_unique:
        mask = [task == t for task in all_tasks]
        task_probs = probs_arr[mask]
        t_recall = float(np.mean(task_probs >= 0.5))
        by_task[t] = {
            "count": int(np.sum(mask)),
            "recall": t_recall,
            "bypass_rate": float(1.0 - t_recall),
            "mean_fake_prob": float(np.mean(task_probs)),
        }

    # Breakdown by region (for inpaint)
    by_region: Dict[str, Dict[str, float]] = {}
    regions_unique = sorted(list(set(all_regions)))
    for r in regions_unique:
        if r == "none":
            continue
        mask = [reg == r for reg in all_regions]
        if np.sum(mask) == 0:
            continue
        reg_probs = probs_arr[mask]
        r_recall = float(np.mean(reg_probs >= 0.5))
        by_region[r] = {
            "count": int(np.sum(mask)),
            "recall": r_recall,
            "bypass_rate": float(1.0 - r_recall),
            "mean_fake_prob": float(np.mean(reg_probs)),
        }

    return {
        "total_samples": total,
        "overall_recall": overall_recall,
        "overall_bypass_rate": overall_bypass,
        "mean_fake_prob": mean_prob,
        "by_task": by_task,
        "by_region": by_region,
        "probs": probs_arr,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark detectors on 6k legacy synthetic dataset")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/external_benchmarks/synthetic_face_dataset_6k/fake_manifest.jsonl",
        help="Path to rebased fake_manifest.jsonl",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info("Running 6k Legacy Synthetic Benchmark on device: %s", device)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    logger.info("Loaded %d records from %s", len(records), manifest_path)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = Legacy6kDataset(records, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    checkpoints = {
        "M0_baseline": Path("experiments/checkpoints/baseline_d0_best.pt"),
        "M1_active_round1": Path("experiments/checkpoints/active_learning_m1_best.pt"),
        "M2_active_round2": Path("experiments/checkpoints/active_learning_m2_best.pt"),
    }

    results: Dict[str, Any] = {
        "title": "Legacy 6k Synthetic Dataset Out-of-Distribution Benchmark Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(records),
        "device": str(device),
        "models": {},
    }
    raw_probs: Dict[str, np.ndarray] = {}

    for model_name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            logger.warning("Checkpoint %s does not exist, skipping.", ckpt_path)
            continue

        logger.info("\n=== Evaluating %s (%s) on 6,000 legacy fakes ===", model_name, ckpt_path.name)
        detector = DeepfakeImageDetector(backbone="dinov2_small", num_classes=1, pretrained=True)
        state = torch.load(ckpt_path, map_location=device)
        detector.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

        t_start = time.time()
        m_results = evaluate_model_on_6k(detector, dataloader, device=device)
        t_elapsed = time.time() - t_start

        logger.info(
            "%s Results: Fake Recall=%.2f%% | Bypass Rate=%.2f%% | Mean P(Fake)=%.4f (Elapsed: %.1fs)",
            model_name,
            m_results["overall_recall"] * 100.0,
            m_results["overall_bypass_rate"] * 100.0,
            m_results["mean_fake_prob"],
            t_elapsed,
        )
        logger.info("  By Task: %s", {k: f"Recall {v['recall']*100:.1f}%" for k, v in m_results["by_task"].items()})
        raw_probs[model_name] = m_results.pop("probs")
        results["models"][model_name] = m_results

    # Save summary JSON
    out_file = Path("experiments/logs/legacy_6k_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save compressed raw predictions
    npz_file = Path("experiments/logs/legacy_6k_predictions.npz")
    np.savez_compressed(
        npz_file,
        m0_probs=raw_probs.get("M0_baseline", np.array([])),
        m1_probs=raw_probs.get("M1_active_round1", np.array([])),
        m2_probs=raw_probs.get("M2_active_round2", np.array([])),
        tasks=np.array([r.get("task", "") for r in records]),
        sub_regions=np.array([r.get("sub_region") or "none" for r in records]),
        image_paths=np.array([r["image_path"] for r in records]),
        prompts=np.array([r.get("prompt", "") for r in records]),
    )

    logger.info("\n================================================================================")
    logger.info("6K BENCHMARK COMPLETE! Metrics: %s | Raw predictions: %s", out_file, npz_file)
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
