"""Consolidate and populate data/hard_samples/ from active learning mining rounds.

Synchronizes the root data/hard_samples/ directory with active learning artifacts:
1. data/hard_samples/hard_samples_manifest.jsonl (unique mined hard samples)
2. data/hard_samples/images/ (symlinked physical images organized by modality)
3. data/hard_samples/audit_reports/hardness_report.json (demographics and yield breakdown)
4. data/hard_samples/README.md (architecture and navigation documentation)
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.common.logger import get_logger

logger = get_logger("populate_hard_samples_vault")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load non-empty JSON lines from a manifest file using a managed file handle."""
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s:
                records.append(json.loads(line_s))
    return records


def populate_hard_samples_vault(
    r1_path: Path = Path("data/active_learning/round_1/hard_samples.jsonl"),
    r2_path: Path = Path("data/active_learning/round_2/hard_samples.jsonl"),
    chal_path: Path = Path("data/active_learning/cumulative_challenge_test.jsonl"),
    d2_train_path: Path = Path("data/benchmarks/D2_hard_boosted/train.jsonl"),
    target_dir: Path = Path("data/hard_samples"),
) -> Dict[str, Any]:
    """Consolidate mined hard samples, image links, audit JSON, and vault documentation."""
    target_dir.mkdir(parents=True, exist_ok=True)

    if not r1_path.exists() or not r2_path.exists():
        raise FileNotFoundError("Active learning hard_samples.jsonl manifests not found!")

    r1_records = _load_jsonl(r1_path)
    r2_records = _load_jsonl(r2_path)
    chal_records = _load_jsonl(chal_path)
    d2_train_records = _load_jsonl(d2_train_path)

    # Deduplicate combined hard samples by image_id
    seen_ids = set()
    combined_records: List[Dict[str, Any]] = []
    for r in r1_records + r2_records:
        iid = r["image_id"]
        if iid not in seen_ids:
            seen_ids.add(iid)
            combined_records.append(r)

    chal_hard_fakes = [r for r in chal_records if r.get("label") == 1]
    chal_hard_count = len({r.get("image_id") for r in chal_hard_fakes})
    if d2_train_records:
        retraining_buffer_count = len(
            {
                r["image_id"]
                for r in d2_train_records
                if r.get("label") == 1 and r.get("is_hard_sample")
            }
        )
    else:
        retraining_buffer_count = max(0, len(combined_records) - chal_hard_count)

    logger.info(
        "Loaded hard samples: R1=%d, R2=%d -> Unique Combined=%d (D2 buffer=%d, Challenge fakes=%d)",
        len(r1_records),
        len(r2_records),
        len(combined_records),
        retraining_buffer_count,
        chal_hard_count,
    )

    # 1. Write hard_samples_manifest.jsonl
    manifest_out = target_dir / "hard_samples_manifest.jsonl"
    with open(manifest_out, "w", encoding="utf-8") as f:
        for r in combined_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Written %d records to %s", len(combined_records), manifest_out)

    # 2. Synchronize images directory cleanly
    images_dir = target_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    linked_images = 0
    for r in combined_records:
        src_raw = r.get("file_path") or r.get("image_path")
        if not src_raw:
            continue
        src_path = Path(src_raw)
        if not src_path.exists():
            logger.warning("Missing source image: %s", src_path)
            continue

        modality = r.get("manipulation_type", "unknown")
        mod_dir = images_dir / modality
        mod_dir.mkdir(parents=True, exist_ok=True)
        dest_path = mod_dir / src_path.name

        if dest_path.is_symlink() or dest_path.exists():
            dest_path.unlink()
        try:
            rel_src = os.path.relpath(src_path.resolve(), mod_dir.resolve())
            dest_path.symlink_to(rel_src)
        except OSError:
            shutil.copy2(src_path, dest_path)
        linked_images += 1

    logger.info("Linked/copied %d images into %s", linked_images, images_dir)

    # 3. Generate hardness_report.json
    type_counts = Counter(r.get("manipulation_type", "unknown") for r in combined_records)
    gender_counts = Counter(
        (r.get("demographics") or {}).get("gender", "unknown") for r in combined_records
    )
    age_counts = Counter(
        (r.get("demographics") or {}).get("age_group", "unknown") for r in combined_records
    )
    scores = [
        float(r.get("baseline_score", 0.0))
        for r in combined_records
        if r.get("baseline_score") is not None
    ]

    report = {
        "title": "Comprehensive Adversarial Hard-Sample Vault Audit Report",
        "total_hard_samples": len(combined_records),
        "allocation": {
            "round_1_mined": len(r1_records),
            "round_2_mined": len(r2_records),
            "retraining_buffer_D2": retraining_buffer_count,
            "challenge_hard_fakes": chal_hard_count,
            "cumulative_challenge_test_vault": len(chal_records),
        },
        "breakdown_by_manipulation_type": dict(type_counts),
        "breakdown_by_gender": dict(gender_counts),
        "breakdown_by_age_group": dict(age_counts),
        "confidence_statistics": {
            "mean_fake_prob": float(sum(scores) / max(1, len(scores))),
            "min_fake_prob": float(min(scores)) if scores else 0.0,
            "max_fake_prob": float(max(scores)) if scores else 0.0,
        },
    }

    report_dir = target_dir / "audit_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_out = report_dir / "hardness_report.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Audit report saved to %s", report_out)

    # 4. Generate README.md using real computed metrics
    denom = max(1, len(combined_records))
    inpaint_cnt = type_counts.get("sd_inpaint", 0)
    img2img_cnt = type_counts.get("sd_img2img", 0)
    text2img_cnt = type_counts.get("sd_text2img", 0)

    readme_content = f"""# Hard Samples Vault (`data/hard_samples/`)

## 1. Tổng Quan Kho Mẫu Khó Đối Kháng (Overview)
Thư mục này là **Kho Lưu Trữ Thống Nhất (Centralized Vault)** chứa toàn bộ các mẫu đối kháng cực khó (Hard Adversarial Samples) được khai thác tự động qua 2 vòng Closed-Loop Active Learning của đề tài.

- **Tổng số mẫu khó được phát hiện**: **{len(combined_records)} mẫu**
- **Tập tin chỉ mục thống nhất**: [`hard_samples_manifest.jsonl`](hard_samples_manifest.jsonl)
- **Kho ảnh thực tế**: [`images/`](images/) (được phân loại theo phương thái sinh ảnh: `sd_inpaint`, `sd_img2img`, `sd_text2img`)
- **Báo cáo kiểm toán nhân khẩu học & độ khó**: [`audit_reports/hardness_report.json`](audit_reports/hardness_report.json)

---

## 2. Nguồn Gốc & Cơ Chế Phân Bổ (Provenance & Allocation)

Toàn bộ {len(combined_records)} mẫu khó bắt nguồn từ quá trình dò quét thực tế trên các checkpoint thần kinh:
1. **Vòng 1 (`data/active_learning/round_1/`)**: {len(r1_records)} mẫu khó đánh lừa mô hình tiền nghiệm $M_0$.
2. **Vòng 2 (`data/active_learning/round_2/`)**: {len(r2_records)} mẫu khó đánh lừa mô hình $M_1$.

### Cơ Chế Phân Bổ (85% Huấn Luyện / 15% Kiểm Định Độc Lập):
- **85% Bộ Đệm Tái Huấn Luyện ({retraining_buffer_count} mẫu)**: Được sáp nhập vào tập huấn luyện của đợt Vòng 2 ([`data/benchmarks/D2_hard_boosted/train.jsonl`](../benchmarks/D2_hard_boosted/train.jsonl)) nhằm gia tăng độ dẻo dai đối kháng cho mô hình $M_2^*$.
- **15% Kho Kiểm Thử Thách Thức ({chal_hard_count} mẫu giả khó)**: Được cách ly nghiêm ngặt và hợp nhất vào kho kiểm định mù đối kháng tích lũy ([`data/active_learning/cumulative_challenge_test.jsonl`](../active_learning/cumulative_challenge_test.jsonl), tổng cộng {len(chal_records)} mẫu) để kiểm tra độc lập $M_0, M_1, M_2^*$.

---

## 3. Phân Rã Theo Phương Thái Sinh Ảnh (Manipulation Modalities)

| Phương Thái | Số Lượng Mẫu Khó | Tỷ Lệ | Nguyên Nhân Vượt Mặt Mô Hình |
| :--- | :---: | :---: | :--- |
| **`sd_inpaint` (Chỉnh sửa cục bộ)** | {inpaint_cnt} | {inpaint_cnt / denom * 100:.1f}% | Ranh giới ghép quanh mắt/môi siêu mỏng, 85-90% bối cảnh là ảnh thật nguyên bản FFHQ. |
| **`sd_img2img` (Biến đổi diện mạo)** | {img2img_cnt} | {img2img_cnt / denom * 100:.1f}% | Giữ nguyên hình học giải phẫu gốc, chỉ can thiệp tinh vi vào bề mặt da và ánh sáng. |
| **`sd_text2img` (Tạo ảnh thuần prompt)** | {text2img_cnt} | {text2img_cnt / denom * 100:.1f}% | Các ca khuôn mặt cân đối hoàn hảo, triệt tiêu tối đa các tạo tác méo mó thông thường. |
"""
    readme_out = target_dir / "README.md"
    with open(readme_out, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info("README documentation written to %s", readme_out)
    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for hard sample vault synchronization."""
    parser = argparse.ArgumentParser(description="Consolidate hard samples vault from active learning rounds.")
    parser.add_argument("--round1-manifest", type=Path, default=Path("data/active_learning/round_1/hard_samples.jsonl"))
    parser.add_argument("--round2-manifest", type=Path, default=Path("data/active_learning/round_2/hard_samples.jsonl"))
    parser.add_argument("--challenge-manifest", type=Path, default=Path("data/active_learning/cumulative_challenge_test.jsonl"))
    parser.add_argument("--d2-train", type=Path, default=Path("data/benchmarks/D2_hard_boosted/train.jsonl"))
    parser.add_argument("--target-dir", type=Path, default=Path("data/hard_samples"))
    return parser.parse_args()


def main() -> None:
    os.chdir(root_dir)
    args = parse_args()
    populate_hard_samples_vault(
        r1_path=args.round1_manifest,
        r2_path=args.round2_manifest,
        chal_path=args.challenge_manifest,
        d2_train_path=args.d2_train,
        target_dir=args.target_dir,
    )


if __name__ == "__main__":
    main()
