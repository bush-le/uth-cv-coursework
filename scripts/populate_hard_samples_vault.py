"""Consolidate and populate data/hard_samples/ from active learning mining rounds.

Synchronizes the root data/hard_samples/ directory with active learning artifacts:
1. data/hard_samples/hard_samples_manifest.jsonl (740 unique mined hard samples)
2. data/hard_samples/images/ (symlinked physical images organized by modality)
3. data/hard_samples/audit_reports/hardness_report.json (demographics and yield breakdown)
4. data/hard_samples/README.md (architecture and navigation documentation)
"""

import os
from pathlib import Path
import sys

# Automatic root resolution for direct script execution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
os.chdir(root_dir)


def main():
    target_dir = Path("data/hard_samples")
    target_dir.mkdir(parents=True, exist_ok=True)

    r1_path = Path("data/active_learning/round_1/hard_samples.jsonl")
    r2_path = Path("data/active_learning/round_2/hard_samples.jsonl")
    chal_path = Path("data/active_learning/cumulative_challenge_test.jsonl")

    if not r1_path.exists() or not r2_path.exists():
        raise FileNotFoundError("Active learning hard_samples.jsonl manifests not found!")

    r1_records = [json.loads(line) for line in open(r1_path, "r", encoding="utf-8") if line.strip()]
    r2_records = [json.loads(line) for line in open(r2_path, "r", encoding="utf-8") if line.strip()]
    chal_records = [json.loads(line) for line in open(chal_path, "r", encoding="utf-8") if line.strip()]

    # Deduplicate by image_id
    seen_ids = set()
    combined_records = []
    for r in r1_records + r2_records:
        iid = r["image_id"]
        if iid not in seen_ids:
            seen_ids.add(iid)
            combined_records.append(r)

    print(f"Loaded: R1={len(r1_records)}, R2={len(r2_records)} -> Unique Combined={len(combined_records)}")

    # 1. Write hard_samples_manifest.jsonl
    manifest_out = target_dir / "hard_samples_manifest.jsonl"
    with open(manifest_out, "w", encoding="utf-8") as f:
        for r in combined_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Written {len(combined_records)} records to {manifest_out}")

    # 2. Setup images directory
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    for r in combined_records:
        src_path = Path(r["file_path"])
        if not src_path.exists():
            print(f"⚠️ Missing source image: {src_path}")
            continue

        modality = r.get("manipulation_type", "unknown")
        mod_dir = images_dir / modality
        mod_dir.mkdir(parents=True, exist_ok=True)
        dest_path = mod_dir / src_path.name

        if not dest_path.exists():
            try:
                # Relative symlink
                rel_src = os.path.relpath(src_path, mod_dir)
                dest_path.symlink_to(rel_src)
            except OSError:
                shutil.copy2(src_path, dest_path)
        copied_images += 1

    print(f"✅ Linked/copied {copied_images} images into {images_dir}")

    # 3. Generate hardness_report.json
    type_counts = Counter(r.get("manipulation_type") for r in combined_records)
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
            "retraining_buffer_D2": 629,
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
    print(f"✅ Audit report saved to {report_out}")

    # 4. Generate README.md
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
1. **Vòng 1 (`data/active_learning/round_1/`)**: 553 mẫu khó đánh lừa mô hình tiền nghiệm $M_0$.
2. **Vòng 2 (`data/active_learning/round_2/`)**: 187 mẫu khó đánh lừa mô hình $M_1$.

### Cơ Chế Phân Bổ (85% Huấn Luyện / 15% Kiểm Định Độc Lập):
- **85% Bộ Đệm Tái Huấn Luyện (629 mẫu)**: Được sáp nhập vào tập huấn luyện của đợt Vòng 2 ([`data/benchmarks/D2_hard_boosted/train.jsonl`](../benchmarks/D2_hard_boosted/train.jsonl)) nhằm gia tăng độ dẻo dai đối kháng cho mô hình $M_2^*$.
- **15% Kho Kiểm Thử Thách Thức (111 mẫu)**: Được cách ly nghiêm ngặt và hợp nhất vào kho kiểm định mù đối kháng tích lũy ([`data/active_learning/cumulative_challenge_test.jsonl`](../active_learning/cumulative_challenge_test.jsonl), tổng cộng 213 mẫu) để kiểm tra độc lập $M_0, M_1, M_2^*$.

---

## 3. Phân Rã Theo Phương Thái Sinh Ảnh (Manipulation Modalities)

| Phương Thái | Số Lượng Mẫu Khó | Tỷ Lệ | Nguyên Nhân Vượt Mặt Mô Hình |
| :--- | :---: | :---: | :--- |
| **`sd_inpaint` (Chỉnh sửa cục bộ)** | {type_counts['sd_inpaint']} | {type_counts['sd_inpaint']/len(combined_records)*100:.1f}% | Ranh giới ghép quanh mắt/môi siêu mỏng, 85-90% bối cảnh là ảnh thật nguyên bản FFHQ. |
| **`sd_img2img` (Biến đổi diện mạo)** | {type_counts['sd_img2img']} | {type_counts['sd_img2img']/len(combined_records)*100:.1f}% | Giữ nguyên hình học giải phẫu gốc, chỉ can thiệp tinh vi vào bề mặt da và ánh sáng. |
| **`sd_text2img` (Tạo ảnh thuần prompt)** | {type_counts['sd_text2img']} | {type_counts['sd_text2img']/len(combined_records)*100:.1f}% | Các ca khuôn mặt cân đối hoàn hảo, triệt tiêu tối đa các tạo tác méo mó thông thường. |
"""
    readme_out = target_dir / "README.md"
    with open(readme_out, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"✅ README documentation written to {readme_out}")


if __name__ == "__main__":
    main()
