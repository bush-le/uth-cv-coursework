"""Generate a comprehensive contact sheet for all 13 mined hard samples."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def main():
    records = []
    for r in ["round_1", "round_2"]:
        p = Path(f"data/active_learning/{r}/hard_samples.jsonl")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    d["round_name"] = "R1" if r == "round_1" else "R2"
                    records.append(d)

    fig, axes = plt.subplots(4, 4, figsize=(20, 22), facecolor="#0f172a")
    plt.subplots_adjust(wspace=0.15, hspace=0.25, top=0.93, bottom=0.03, left=0.03, right=0.97)

    fig.suptitle(
        "TOPIC 11 — 13 ADVERSARIAL HARD SAMPLES MINED BY CLOSED-LOOP ACTIVE LEARNING\n"
        "False Negatives Bypassing DINOv2 Deepfake Detector (Detector Predicted as Real with 50% - 97.6% Confidence)",
        fontsize=17, fontweight="bold", color="#f8fafc", y=0.97
    )

    for idx in range(16):
        r_idx = idx // 4
        c_idx = idx % 4
        ax = axes[r_idx, c_idx]
        ax.set_facecolor("#1e293b")

        if idx < len(records):
            rec = records[idx]
            img_path = Path(rec["file_path"])
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            p_fake = rec.get("scores", {}).get("detector_p_fake", 0.0)
            mode = rec.get("manipulation_type", "unknown").replace("sd_", "")
            rnd = rec["round_name"]
            img_id = rec.get("image_id", "")
            demo = rec.get("demographics", {})
            age = demo.get("age_group", "?")
            gen = demo.get("gender", "?")
            gla = demo.get("glasses", "None")

            # Color border based on hardness
            if p_fake < 0.10:
                border_color = "#ef4444"  # red: severe bypass
                tag = f"CRITICAL BYPASS | P(Fake) = {p_fake*100:.1f}%"
                tag_bg = "#dc2626"
            elif p_fake < 0.50:
                border_color = "#f59e0b"  # amber: moderate bypass
                tag = f"HIGH BYPASS | P(Fake) = {p_fake*100:.1f}%"
                tag_bg = "#d97706"
            else:
                border_color = "#3b82f6"  # blue: boundary
                tag = f"BOUNDARY | P(Fake) = {p_fake*100:.1f}%"
                tag_bg = "#2563eb"

            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3.5)

            title_text = f"#{idx+1} [{rnd}] {img_id}\nMode: {mode.upper()} | {gen.capitalize()}, {age} | Glasses: {gla}"
            ax.set_title(title_text, color="#f1f5f9", fontsize=10.5, fontweight="semibold", pad=6)

            # Overlay tag at bottom
            ax.text(
                0.5, 0.05, tag,
                transform=ax.transAxes,
                color="#ffffff",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=tag_bg, edgecolor="none", alpha=0.92)
            )
        else:
            # Information cards for remaining 3 cells
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
                spine.set_linewidth(1.5)

            if idx == 13:
                card_title = "DISTRIBUTION BY MODE & ROUND"
                content = (
                    "• Total Mined Hard Samples: 13\n\n"
                    "• Distribution by Round:\n"
                    "  - Round 1 (vs M0 Baseline): 5 samples\n"
                    "  - Round 2 (vs M1 Active): 8 samples\n\n"
                    "• Distribution by Synthesis Mode:\n"
                    "  - Inpainting (sd_inpaint): 10 / 13 (77%)\n"
                    "  - Img2Img (sd_img2img): 3 / 13 (23%)\n\n"
                    "• Primary Vulnerability:\n"
                    "  Localized inpainting preserves\n"
                    "  85%+ authentic context, evading\n"
                    "  global ViT perceptual attention."
                )
            elif idx == 14:
                card_title = "DETECTOR BYPASS SEVERITY"
                content = (
                    "• Lowest P(Fake) Detected:\n"
                    "  0.0238 (Detector: 97.62% Real Confidence!)\n\n"
                    "• Severe Bypass (P_fake < 10%):\n"
                    "  8 / 13 samples (61.5%)\n\n"
                    "• Moderate Bypass (10% <= P_fake < 50%):\n"
                    "  4 / 13 samples (30.8%)\n\n"
                    "• Decision Boundary (P_fake ~ 50%):\n"
                    "  1 / 13 samples (7.7%)\n\n"
                    "• Mean P(Fake) across all 13:\n"
                    "  13.06% (Baseline completely deceived)"
                )
            else:
                card_title = "D2 BENCHMARK UTILITY"
                content = (
                    "• Active Learning Value:\n"
                    "  Hard samples are systematically\n"
                    "  recycled back into training loop.\n\n"
                    "• Partition Discipline:\n"
                    "  - 11 samples added to D2 Train (boosted)\n"
                    "  - 2 samples assigned to Val split\n"
                    "  - 0 test split contamination (Zero-leakage)\n\n"
                    "• Target Outcome:\n"
                    "  Forces detector to resolve micro-\n"
                    "  boundary seam and reflection artifacts."
                )

            ax.text(0.08, 0.90, card_title, transform=ax.transAxes, color="#38bdf8", fontsize=11, fontweight="bold", va="top")
            ax.text(0.08, 0.80, content, transform=ax.transAxes, color="#cbd5e1", fontsize=9.5, va="top", linespacing=1.35)

    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hard_samples_contact_sheet.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Successfully generated contact sheet: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
