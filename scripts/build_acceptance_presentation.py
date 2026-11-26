"""Build script for Topic 11 Final Acceptance Presentation HTML with Rich Visual Artifacts.

Generates docs/reports/presentation_final_acceptance_legacy.html with embedded real visual artifacts:
- Dual-Engine synthetic faces grid (SD 1.5 vs Realistic Vision 5.1)
- Soft-Mask Feathering visual comparison
- Skin gamut YCbCr distribution
- Hard-sample biological defects visual gallery (eyes, teeth, skin, hair)
- Challenge Vault recall jump curves (M0 -> M1 -> M2* -> M3)
- Zero Identity Leakage dataset partition schema
- Champion M3 Quad-Vault recall bar charts
- Universal benchmark 34-method heatmap & 8-family recall
- Real DINOv2 self-attention spatial heatmap forensic inspection
"""

import json
from pathlib import Path
import re
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.analysis.coursework_metrics import collect_live_coursework_metrics, fmt_int, fmt_pct

# Legacy presentation builder; keep separate from active presentation_final_acceptance.html
OUTPUT_PATH = Path("docs/reports/presentation_final_acceptance_legacy.html")


def _build_presentation_subs() -> dict[str, str]:
    """Build live substitutions for the slide-deck presentation."""
    m = collect_live_coursework_metrics()
    real = m["real"]
    cur = m["curation"]
    sp = m["splits"]
    al = m["al"]
    q = m["quad"]
    univ = m["universal"]
    ug = univ["global"]
    uf = univ["family"]
    ff = univ["ff"]

    u_npz = np.load("experiments/benchmarks/logs/benchmark_5models_predictions.npz", allow_pickle=True)
    y_u = u_npz["y_true"]
    taus = np.linspace(0.5, 0.999, 500)

    def _fpr1(key: str) -> tuple[float, float]:
        p = u_npz[key]
        reals_p = p[y_u == 0]
        fakes_p = p[y_u == 1]
        valid = [t for t in taus if (reals_p >= t).mean() <= 0.01]
        t_b = float(valid[0]) if valid else 0.999
        return t_b, float((fakes_p >= t_b).mean() * 100.0)

    m3_t1, m3_rec1 = _fpr1("probs_M3_dual_engine")
    m0_t1, m0_rec1 = _fpr1("probs_M0_baseline")

    d3_tr_fakes = [
        json.loads(line)
        for line in Path("data/benchmarks/D3_clean/train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["label"] == 1
    ]
    d3_rv_t2i = sum(1 for r in d3_tr_fakes if r.get("manipulation_type") == "rv51_text2img")
    d3_rv_inp = sum(1 for r in d3_tr_fakes if r.get("manipulation_type") == "rv51_inpaint")
    d3_rv_i2i = sum(1 for r in d3_tr_fakes if r.get("manipulation_type") == "rv51_img2img")

    ch_fake = max(1, al["challenge_fake"])
    m1_jump = (q["M1"]["challenge"]["recall"] - q["M0"]["challenge"]["recall"]) * 100.0
    m2s_jump = (q["M2_star"]["challenge"]["recall"] - q["M0"]["challenge"]["recall"]) * 100.0
    m3_u_jump = (q["M3"]["unseen"]["recall"] - q["M2_star"]["unseen"]["recall"]) * 100.0

    return {
        "__REAL_TR__": fmt_int(real["train"]),
        "__REAL_TE__": fmt_int(real["test"]),
        "__FAKE_RAW__": fmt_int(cur["fake_raw_total"]),
        "__SD_RAW__": fmt_int(cur["sd15_raw"]),
        "__SD_CUR__": fmt_int(cur["sd15_cur"]),
        "__B1_RAW__": fmt_int(cur["b1_raw"]),
        "__R1_RAW__": fmt_int(cur["r1_raw"]),
        "__R2_RAW__": fmt_int(cur["r2_raw"]),
        "__RV_RAW__": fmt_int(cur["rv51_raw"]),
        "__RV_CUR__": fmt_int(cur["rv51_cur"]),
        "__HARD_TOT__": fmt_int(al["total_hard"]),
        "__HARD_TR__": fmt_int(al["total_hard_train"]),
        "__CHAL_TOT__": fmt_int(al["challenge_total"]),
        "__CHAL_FAKE__": fmt_int(ch_fake),
        "__MASTER_TOT__": fmt_int(sp["master_test"]),
        "__D3_TR__": fmt_int(sp["d3_train"]),
        "__D3_SD_BASE__": fmt_int(sp["d3_sd_nonhard"]),
        "__D3_RV_CLEAN__": fmt_int(sp["d3_rv51_clean"]),
        "__D3_RV_T2I__": fmt_int(d3_rv_t2i),
        "__D3_RV_INP__": fmt_int(d3_rv_inp),
        "__D3_RV_I2I__": fmt_int(d3_rv_i2i),
        "__UNSEEN_TOT__": fmt_int(sp["unseen_rv51_total"]),
        "__UNSEEN_F__": fmt_int(sp["unseen_rv51_fake"]),
        "__M0_M_REC__": fmt_pct(q["M0"]["master"]["recall"]),
        "__M0_C_REC__": fmt_pct(q["M0"]["challenge"]["recall"]),
        "__M0_C_BYP__": fmt_pct(q["M0"]["challenge"]["bypass_rate"]),
        "__M0_C_AUC__": f"{q['M0']['challenge']['roc_auc']:.4f}",
        "__M0_U_REC__": fmt_pct(q["M0"]["unseen"]["recall"]),
        "__M1_M_REC__": fmt_pct(q["M1"]["master"]["recall"]),
        "__M1_C_REC__": fmt_pct(q["M1"]["challenge"]["recall"]),
        "__M1_U_REC__": fmt_pct(q["M1"]["unseen"]["recall"]),
        "__M1_JUMP__": f"{m1_jump:.2f}%",
        "__M2_M_REC__": fmt_pct(q["M2"]["master"]["recall"]),
        "__M2_C_REC__": fmt_pct(q["M2"]["challenge"]["recall"]),
        "__M2_U_REC__": fmt_pct(q["M2"]["unseen"]["recall"]),
        "__M2S_M_REC__": fmt_pct(q["M2_star"]["master"]["recall"]),
        "__M2S_C_REC__": fmt_pct(q["M2_star"]["challenge"]["recall"]),
        "__M2S_C_AUC__": f"{q['M2_star']['challenge']['roc_auc']:.4f}",
        "__M2S_C_DET__": str(round(q["M2_star"]["challenge"]["recall"] * ch_fake)),
        "__M2S_U_REC__": fmt_pct(q["M2_star"]["unseen"]["recall"]),
        "__M2S_JUMP__": f"{m2s_jump:.2f}%",
        "__M3_M_REC__": fmt_pct(q["M3"]["master"]["recall"]),
        "__M3_C_REC__": fmt_pct(q["M3"]["challenge"]["recall"]),
        "__M3_C_BYP__": fmt_pct(q["M3"]["challenge"]["bypass_rate"]),
        "__M3_U_REC__": fmt_pct(q["M3"]["unseen"]["recall"]),
        "__M3_U_BYP__": fmt_pct(q["M3"]["unseen"]["bypass_rate"]),
        "__M3_U_JUMP__": f"{m3_u_jump:.2f}%",
        "__U_M3_REC__": ug["M3_dual_engine"]["Fake Recall (↑)"],
        "__U_M3_SPEC__": ug["M3_dual_engine"]["Real Specificity (↑)"],
        "__UF_SD21__": uf["Latent Diffusion (SD 2.1)"]["M₃ (Dual-Engine Champion) Recall/Spec"],
        "__UF_GAN__": uf["GAN Architectures"]["M₃ (Dual-Engine Champion) Recall/Spec"],
        "__FF_FPR_085__": f"{ff['fpr_085']:.2f}%",
        "__U_M3_T1__": f"{m3_t1:.3f}",
        "__U_M3_REC1__": f"{m3_rec1:.2f}%",
        "__U_M0_T1__": f"{m0_t1:.3f}",
        "__U_M0_REC1__": f"{m0_rec1:.2f}%",
    }

def build_presentation_html() -> str:
    base_file = Path("docs/reports/presentation_5min_define_and_design.html")
    if not base_file.exists():
        raise FileNotFoundError(f"Missing base template: {base_file}")
    
    with open(base_file, "r", encoding="utf-8") as f:
        base_content = f.read()
    
    # Extract style block
    style_match = re.search(r"<style>(.*?)</style>", base_content, re.DOTALL)
    base_css = style_match.group(1) if style_match else ""
    
    # Extract script block
    script_match = re.search(r"<script>(.*?)</script>", base_content, re.DOTALL)
    base_js = script_match.group(1) if script_match else ""

    acceptance_css = """
    /* Acceptance specific badge styles */
    .badge-champion {
      background: #fef3c7;
      color: #92400e;
      border: 1px solid #f59e0b;
      font-weight: 700;
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .badge-success {
      background: #dcfce7;
      color: #166534;
      border: 1px solid #86efac;
      font-weight: 700;
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .badge-alert {
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #fca5a5;
      font-weight: 700;
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    
    /* Acceptance scorecard table */
    .scorecard-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .scorecard-table th {
      background: var(--surface-soft);
      padding: 9px 12px;
      text-align: left;
      font-weight: 700;
      color: var(--text);
      border-bottom: 2px solid var(--border-strong);
      font-family: var(--font-display);
    }
    .scorecard-table td {
      padding: 8px 12px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
    }
    .scorecard-table tr:last-child td {
      border-bottom: none;
    }
    .scorecard-table tr.highlight-champion td {
      background: rgba(30, 43, 250, 0.06);
      font-weight: 600;
    }
    .scorecard-table tr.highlight-champion td:first-child {
      border-left: 3px solid var(--primary);
    }
    
    .delta-pill-green {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      background: #dcfce7;
      color: #15803d;
      font-weight: 700;
      font-size: 11px;
      font-family: var(--font-mono);
    }
    .delta-pill-red {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      background: #fee2e2;
      color: #b91c1c;
      font-weight: 700;
      font-size: 11px;
      font-family: var(--font-mono);
    }
    
    /* Live demo launcher button */
    .btn-live-demo {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 16px;
      background: var(--primary);
      color: #ffffff;
      text-decoration: none;
      font-weight: 700;
      font-size: 13px;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(30, 43, 250, 0.25);
      transition: all 0.2s ease;
      cursor: pointer;
    }
    .btn-live-demo:hover {
      background: #1420d4;
      transform: translateY(-1px);
      box-shadow: 0 6px 16px rgba(30, 43, 250, 0.35);
    }

    /* Artifact image styling */
    .artifact-img-box {
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #ffffff;
      padding: 8px;
      border-radius: var(--radius-sm);
      overflow: hidden;
    }
    .artifact-img {
      max-width: 100%;
      max-height: 480px;
      object-fit: contain;
      border-radius: 6px;
      border: 1px solid var(--border);
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
    }
    .artifact-img-compact {
      max-width: 100%;
      max-height: 220px;
      object-fit: contain;
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    """

    slides_html = r"""
    <!-- =======================================================================
         SLIDE 00: BÁO CÁO NGHIỆM THU ĐỒ ÁN (COVER & HERO ARTIFACT)
         ======================================================================= -->
    <section class="slide active" id="slide-1" data-title="Nghiệm Thu Toàn Diện">
      <div class="slide-inner">
        <div class="cover-grid">
          <div class="cover-left">
            <div>
              <div class="cover-kicker">
                <span>UTH COMPUTER VISION COURSEWORK • ĐỀ TÀI 11 • BÁO CÁO NGHIỆM THU CHUNG CUỘC</span>
              </div>
              <h1 class="cover-title">
                Stable Diffusion-Based <span class="accent">Synthetic Facial Dataset</span> for Deepfake Defense
              </h1>
              <p class="cover-subtitle">
                Báo cáo nghiệm thu thực chứng khoa học: Hoàn tất 100% KPI cam kết với bộ dữ liệu __FAKE_RAW__ ảnh khuếch tán kép (SD 1.5 + RV5.1), két bẫy __CHAL_TOT__ Challenge Vault, mô hình Champion M3 đạt Fake Recall __M3_U_REC__ và đối chuẩn Universal Benchmark trên 20.400 ảnh.
              </p>
            </div>

            <!-- Hero Artifact Strip: The Real Golden Scorecard Artifact -->
            <div class="cover-visual-strip" style="background: #ffffff; padding: 10px; border-radius: var(--radius-md); border: 1px solid var(--border-strong);">
              <div class="panel-title-bar" style="margin-bottom: 6px;">
                <div class="panel-title" style="font-size: 14px;">
                  <span class="panel-title-dot"></span>
                  Minh Chứng Artifact Vàng: Bảng Điểm Tiến Hóa 4 Cột Mốc Mô Hình (M0 → M3)
                </div>
                <span class="panel-badge badge-champion">🏆 GOLDEN SCORECARD THỰC NGHIỆM</span>
              </div>
              <div class="artifact-img-box" style="height: 200px;">
                <img src="assets/extracted_nb/00_master_deepfake_defense_showcase_cell4_img0.png" class="artifact-img" alt="Golden Scorecard M0 to M3 Evolution" style="max-height: 195px;">
              </div>
            </div>
          </div>

          <!-- Right Column: Metadata & Committee Inspection Card -->
          <div class="cover-right">
            <div class="cover-meta-grid">
              <div class="meta-box">
                <span class="meta-label">TRƯỜNG / KHOA</span>
                <span class="meta-value">ĐH GTVT TP.HCM (UTH)</span>
                <span class="meta-tag">Khoa CNTT • Bộ Môn CV</span>
              </div>
              <div class="meta-box">
                <span class="meta-label">ĐỀ TÀI / NHÓM</span>
                <span class="meta-value">Đề Tài 11 • Nhóm 4</span>
                <span class="meta-tag">Hội Đồng Nghiệm Thu</span>
              </div>
              <div class="meta-box">
                <span class="meta-label">DỮ LIỆU SINH THỰC TẾ</span>
                <span class="meta-value">__FAKE_RAW__ Ảnh Khuếch Tán Kép</span>
                <span class="meta-tag">SD 1.5 (__SD_RAW__) + RV5.1 (__RV_RAW__)</span>
              </div>
              <div class="meta-box">
                <span class="meta-label">KÉT THÁCH THỨC ĐỘC LẬP</span>
                <span class="meta-value">__CHAL_TOT__ Mẫu Challenge Vault</span>
                <span class="meta-tag">Khóa Cứng 4 Lỗi Sinh Học</span>
              </div>
              <div class="meta-box">
                <span class="meta-label">CHAMPION M3 HIỆU NĂNG</span>
                <span class="meta-value">__M3_U_REC__ Quad-Vault Recall</span>
                <span class="meta-tag">Bypass Rate Giảm Rơi Tự Do __M3_U_BYP__</span>
              </div>
              <div class="meta-box">
                <span class="meta-label">ĐỐI CHUẨN TOÀN CẦU</span>
                <span class="meta-value">20.400 Ảnh Benchmark</span>
                <span class="meta-tag">34 Thuật Toán • 8 Họ Mô Hình</span>
              </div>
            </div>

            <!-- Direct Link to Master Showcase Notebook -->
            <div style="margin-top: 16px; background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 14px; text-align: center;">
              <span style="font-size: 12px; color: var(--text-muted); display: block; margin-bottom: 6px;">Mã nguồn & 4 Notebooks nghiệm thu trên VSCode:</span>
              <a href="../../notebooks/coursework/00_master_deepfake_defense_showcase.ipynb" class="btn-live-demo" target="_blank" style="display: inline-flex; justify-content: center; width: 100%;">
                ⚡ MỞ NOTEBOOK 00: LIVE FORENSIC INSPECTOR
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 01: VẤN ĐỀ CỐT LÕI & ARTIFACT SOFT FEATHERING
         ======================================================================= -->
    <section class="slide" id="slide-2" data-title="01 / Vấn Đề Thực Trạng">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">01 / THÁCH THỨC ĐỐI CHUẨN</span>
              <span class="section-subtitle">VÌ SAO ẢNH GIẢ KHUẾCH TÁN ĐÁNH LỪA BỘ DÒ TRUYỀN THỐNG?</span>
            </div>
            <h2 class="slide-title">
              Thực Trạng Đối Đầu: <span>Ảnh Giả Mềm Xóa Sổ Vết Cắt Biên Nhân Tạo</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Deepfake Diffusion áp dụng Soft-Mask Feathering σ=4.0 xóa nhòa hoàn toàn ranh giới ghép; Thực nghiệm chứng minh Baseline M0 bỏ sót tới __M0_C_BYP__ mẫu khó.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left Visual Artifact: Real vs Soft Feathering Comparison -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Thực Chứng: Kỹ Thuật Làm Mịn Biên Gaussian (Soft-Mask Feathering)
                </div>
                <span class="panel-badge badge-success">ARTIFACT NOTEBOOK 01</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/extracted_nb/01_diffusion_dataset_and_curation_cell8_img2.png" class="artifact-img" alt="Soft-Mask Feathering Visual Artifact">
              </div>
            </div>

            <!-- Right Visual: The Empirical Failure Data of M0 Baseline -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Minh Chứng Thực Nghiệm: Mô Hình Cơ Sở M0 Sụp Đổ Trước Mẫu Khó
                </div>
                <span class="panel-badge badge-alert">SỐ LIỆU NGHIỆM THU</span>
              </div>
              <div style="padding: 20px; display: flex; flex-direction: column; gap: 14px; height: 470px; justify-content: space-around;">
                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 14px;">Master Frozen Test (Ảnh Dễ - Phân Phối D0)</span>
                    <span class="badge-success">FAKE RECALL: __M0_M_REC__</span>
                  </div>
                  <div style="width: 100%; height: 10px; background: #e5e7eb; border-radius: 5px; overflow: hidden;">
                    <div style="width: __M0_M_REC__; height: 100%; background: #3b82f6;"></div>
                  </div>
                  <span style="font-size: 12px; color: var(--text-muted); margin-top: 6px; display: block;">M0 bắt được một nửa số ảnh giả do vẫn còn nhận diện được một số đặc trưng cơ sở.</span>
                </div>

                <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: var(--radius-md); padding: 16px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 14px; color: #9f1239;">🚨 Challenge Vault (__CHAL_TOT__ Mẫu Cực Khó)</span>
                    <span class="delta-pill-red">FAKE RECALL: __M0_C_REC__</span>
                  </div>
                  <div style="width: 100%; height: 10px; background: #e5e7eb; border-radius: 5px; overflow: hidden;">
                    <div style="width: __M0_C_REC__; height: 100%; background: #e11d48;"></div>
                  </div>
                  <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px;">
                    <span style="color: #9f1239; font-weight: 700;">Tỷ lệ lọt lưới (Bypass Rate): __M0_C_BYP__</span>
                    <span style="color: var(--text-muted);">ROC-AUC: __M0_C_AUC__</span>
                  </div>
                  <p style="font-size: 12px; color: #881337; margin-top: 6px;">
                    <strong>Nguy hại an ninh eKYC:</strong> Hơn 9/10 khuôn mặt giả tinh vi dễ dàng vượt qua cổng nhận diện ngân hàng nếu chỉ dựa vào mô hình cơ sở không qua Active Learning!
                  </p>
                </div>

                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 14px;">
                  <span style="font-weight: 700; font-size: 13px; color: var(--primary);">🎯 Hướng giải quyết của Đề Tài 11:</span>
                  <p style="font-size: 12px; color: var(--text); margin-top: 4px;">
                    Thiết lập chu trình <strong>Active Learning khép kín</strong>: Khai phá __HARD_TOT__ ca biên khó → Tinh chỉnh phân tầng Block 11 DINOv2 → Đẩy lùi đường biên quyết định → Chế ngự toàn diện các mẫu giả mạo tinh vi.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <!-- Bottom 3 Takeaway Cards -->
          <div class="card-grid-3" style="margin-top: 14px;">
            <div class="intent-card">
              <div class="intent-card-top">
                <span class="intent-index">01</span>
                <span class="intent-role">TRIỆT TIÊU ĐƯỜNG CẮT</span>
              </div>
              <h3>Xóa Sổ Biên Giả Bằng Gaussian</h3>
              <p>Soft-Mask Feathering σ=4.0 tạo chuyển tiếp mượt mà, vô hiệu hóa các thuật toán dò tìm cạnh sắc nét.</p>
            </div>
            <div class="intent-card">
              <div class="intent-card-top">
                <span class="intent-index">02</span>
                <span class="intent-role">GIỚI HẠN CỦA CNN</span>
              </div>
              <h3>CNN Học Tủ Ngữ Nghĩa Cấp Cao</h3>
              <p>Mạng ResNet/EfficientNet phụ thuộc vào hình dáng mắt/mũi/miệng nên dễ dàng bị đánh lừa bởi ảnh SD.</p>
            </div>
            <div class="intent-card">
              <div class="intent-card-top">
                <span class="intent-index">03</span>
                <span class="intent-role">CHÌA KHÓA DINOv2</span>
              </div>
              <h3>Soi Vi Kết Cấu Patch 14x14</h3>
              <p>DINOv2 SSL bảo toàn đặc trưng tần số cao vi thể, là nền tảng duy nhất bắt trọn các dị tật sinh học ẩn.</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 02: BẢNG ĐỐI CHIẾU 100% KPI CAM KẾT VÀ GLOBAL COMPARISON
         ======================================================================= -->
    <section class="slide" id="slide-3" data-title="02 / Đối Chiếu KPI Cam Kết">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">02 / NGHIỆM THU CAM KẾT</span>
              <span class="section-subtitle">ĐỐI CHIẾU 1-1 KPI THIẾT KẾ VS KẾT QUẢ THỰC TẾ</span>
            </div>
            <h2 class="slide-title">
              Nghiệm Thu Cam Kết: <span>100% Mục Tiêu Đạt Chuẩn & Vượt Kế Hoạch</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Tất cả các hạng mục kiến trúc đề ra trong Define & Design đều đã được hoàn thành đầy đủ, có mã nguồn thực thi, dữ liệu thật và báo cáo kiểm chuẩn minh bạch.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: Table of acceptance -->
            <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px; box-shadow: var(--shadow-card);">
              <table class="scorecard-table">
                <thead>
                  <tr>
                    <th>Hạng Mục Nghiệm Thu</th>
                    <th>Cam Kết Thiết Kế</th>
                    <th>Kết Quả Thực Nghiệm</th>
                    <th>Đánh Giá</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>1. Sinh Dữ Liệu</strong></td>
                    <td>SD Inpaint/Img2Img/Text2Img</td>
                    <td>__FAKE_RAW__ ảnh 2 động cơ SD 1.5 (__SD_RAW__) + RV5.1 (__RV_RAW__); Soft Feathering σ=4.0</td>
                    <td><span class="badge-success">✓ ĐẠT 100%</span></td>
                  </tr>
                  <tr>
                    <td><strong>2. CurationGate</strong></td>
                    <td>Lọc 3 tầng Hình học/Quang học/Hash</td>
                    <td>YuNet / OpenCV Haar + Laplacian phân vị 5% + Extended Audit + SHA-256/dHash; giữ __SD_CUR__ SD 1.5 &amp; __RV_CUR__ RV5.1</td>
                    <td><span class="badge-success">✓ ĐẠT 100%</span></td>
                  </tr>
                  <tr class="highlight-champion">
                    <td><strong>3. Khai Phá Mẫu Khó</strong></td>
                    <td>Active Learning vòng lặp</td>
                    <td>__HARD_TOT__ mẫu khó, Két __CHAL_TOT__ Challenge; Recall: __M0_C_REC__ → __M2S_C_REC__</td>
                    <td><span class="badge-champion">⭐ VƯỢT (+__M2S_JUMP__)</span></td>
                  </tr>
                  <tr>
                    <td><strong>4. Zero Leakage</strong></td>
                    <td>Tránh học vẹt danh tính</td>
                    <td>Cách ly __UNSEEN_F__ ảnh RV5.1 test ID; D3 cân bằng 1:1 chuẩn xác (__D3_TR__ ảnh)</td>
                    <td><span class="badge-success">✓ LIÊM CHÍNH</span></td>
                  </tr>
                  <tr class="highlight-champion">
                    <td><strong>5. Champion M3</strong></td>
                    <td>Đánh giá tổng quát hóa</td>
                    <td>Champion M3 Quad-Vault __M3_U_REC__; Universal Benchmark 20.400 ảnh</td>
                    <td><span class="badge-champion">⭐ VƯỢT CHỈ TIÊU</span></td>
                  </tr>
                </tbody>
              </table>
              <div style="margin-top: 14px; font-size: 12px; color: var(--text-muted); text-align: center;">
                Tất cả 5/5 hạng mục nghiệm thu đã được code hóa trong <code>scripts/</code> và kiểm thử tự động 100% qua pytest.
              </div>
            </div>

            <!-- Right: Real Global Comparison Benchmark Chart -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Đối Chuẩn Toàn Cầu: So Sánh Toàn Bộ Các Mô Hình
                </div>
                <span class="panel-badge badge-champion">BENCHMARK FIGURE</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/benchmarks/benchmark_5models_global_comparison.png" class="artifact-img" alt="Global Comparison Benchmark Chart">
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 03: SẢN PHẨM 1: KHO DỮ LIỆU DUAL-ENGINE __FAKE_RAW__ ẢNH & GRID MẪU THẬT
         ======================================================================= -->
    <section class="slide" id="slide-4" data-title="03 / Dữ Liệu Khuếch Tán Kép">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">03 / SẢN PHẨM 1: DỮ LIỆU</span>
              <span class="section-subtitle">ĐỘNG CƠ KHUÊCH TÁN KÉP & LÀM MỊN BIÊN GAUSSIAN</span>
            </div>
            <h2 class="slide-title">
              Kho Dữ Liệu Kép: <span>__FAKE_RAW__ Ảnh Sinh Thực Tế &amp; Lưới Đối Chuẩn Mẫu</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Kết hợp Stable Diffusion v1.5 và RealisticVision v5.1 xóa bỏ hiện tượng "học tủ" bộ sinh; kiểm soát nghiêm ngặt 3 chế độ Inpaint, Img2Img và Text2Img.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: Real Dual-Engine Portfolio Grid Artifact -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Danh Mục Sinh Kép: Real vs Inpaint vs Img2Img vs Text2Img
                </div>
                <span class="panel-badge badge-success">ARTIFACT NOTEBOOK 01</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/extracted_nb/01_diffusion_dataset_and_curation_cell10_img3.png" class="artifact-img" alt="Dual-Engine Portfolio Grid">
              </div>
            </div>

            <!-- Right: Breakdown numbers & Demographics Panel -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Cơ Cấu Danh Mục __FAKE_RAW__ Ảnh Sinh Thực Nghiệm
                </div>
                <span class="panel-badge">QUY MÔ __FAKE_RAW__ MẪU</span>
              </div>
              <div style="padding: 16px; display: flex; flex-direction: column; gap: 12px; height: 480px; justify-content: space-between;">
                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-weight: 700; color: #111111;">Động Cơ A: Stable Diffusion v1.5</span>
                    <span class="badge-success">__SD_RAW__ THÔ / __SD_CUR__ SẠCH</span>
                  </div>
                  <p style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px;">
                    Đại diện cho họ khuếch tán phổ thông mã nguồn mở, vận hành vòng lặp Active Learning nội miền (D0 → D1 → D2).
                  </p>
                  <div style="display: flex; gap: 6px; font-size: 11px; font-family: var(--font-mono);">
                    <span style="background: var(--surface-soft); padding: 2px 6px; border-radius: 4px;">D0: __B1_RAW__</span>
                    <span style="background: var(--surface-soft); padding: 2px 6px; border-radius: 4px;">AL R1: __R1_RAW__</span>
                    <span style="background: var(--surface-soft); padding: 2px 6px; border-radius: 4px;">AL R2: __R2_RAW__</span>
                  </div>
                </div>

                <div style="background: #ffffff; border: 1px solid var(--border-strong); border-radius: var(--radius-md); padding: 12px;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span style="font-weight: 700; color: var(--primary);">Động Cơ B: RealisticVision v5.1 (Photorealistic)</span>
                    <span class="badge-champion">__RV_RAW__ THÔ / __RV_CUR__ SẠCH</span>
                  </div>
                  <p style="font-size: 11px; color: var(--text-muted); margin-bottom: 6px;">
                    Đại diện cho mô hình khuếch tán tinh chỉnh cao cấp. Mô phỏng siêu thực từng chi tiết phản quang đồng tử, sợi tóc tơ và cấu trúc da chân thực.
                  </p>
                  <div style="display: flex; gap: 6px; font-size: 11px; font-family: var(--font-mono);">
                    <span style="background: var(--primary-soft); color: var(--primary); padding: 2px 6px; border-radius: 4px;">Inpaint: 2.000</span>
                    <span style="background: var(--primary-soft); color: var(--primary); padding: 2px 6px; border-radius: 4px;">Img2Img: 2.000</span>
                    <span style="background: var(--primary-soft); color: var(--primary); padding: 2px 6px; border-radius: 4px;">Text2Img: 2.000</span>
                  </div>
                </div>

                <!-- Demographics thumbnail image -->
                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 8px; text-align: center;">
                  <span style="font-size: 11px; font-weight: 700; color: var(--text); display: block; margin-bottom: 4px;">Phân phối nhân khẩu học CelebA-HQ & FFHQ (Tuổi, Giới tính, Ánh sáng):</span>
                  <img src="assets/eda/02_demographics_panel.png" class="artifact-img-compact" alt="Demographics Panel" style="max-height: 140px;">
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 04: SẢN PHẨM 2: PHỄU SÀNG LỌC CHẤT LƯỢNG 3 TẦNG & ARTIFACTS
         ======================================================================= -->
    <section class="slide" id="slide-5" data-title="04 / Phễu Lọc CurationGate">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">04 / SẢN PHẨM 2: PHỄU LỌC</span>
              <span class="section-subtitle">ĐƯỜNG ỐNG SÀNG LỌC CHẤT LƯỢNG 3 TẦNG CHUYÊN SÂU</span>
            </div>
            <h2 class="slide-title">
              CurationGate: <span>Phễu Sàng Lọc 3 Tầng Bảo Vệ Tính Toàn Vẹn</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Đảm bảo 100% ảnh nạp vào huấn luyện đạt chuẩn chân thực sinh học; loại bỏ hoàn toàn ảnh biến dạng, mờ nhòe và trùng lặp bản sao.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: Skin Gamut YCbCr Distribution Artifact -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Tầng 2: Phân Tích Phổ Màu Da Trong Không Gian YCbCr
                </div>
                <span class="panel-badge badge-success">ARTIFACT NOTEBOOK 01</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/extracted_nb/01_diffusion_dataset_and_curation_cell6_img1.png" class="artifact-img" alt="Skin Color Gamut YCbCr Analysis">
              </div>
            </div>

            <!-- Right: Quality Audit Panel Artifact -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Tầng 2 & 3: Kiểm Toán Độ Nét Quang Học & Chống Trùng Lặp
                </div>
                <span class="panel-badge">AUDIT QUALITY FIGURE</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/eda/04_image_quality_panel.png" class="artifact-img" alt="Image Quality Panel">
              </div>
            </div>
          </div>

          <!-- Bottom 3 Flow Cards -->
          <div class="card-grid-3" style="margin-top: 14px;">
            <div class="intent-card">
              <div class="intent-card-top">
                <span class="intent-index">TẦNG 1</span>
                <span class="intent-role">HÌNH HỌC KHUÔN MẶT</span>
              </div>
              <h3>YuNet / OpenCV Haar &amp; Kiểm Định Hình Học</h3>
              <p>Lọc ảnh không thấy mặt, thiếu cặp mắt hoặc đa mặt, kiểm tra Cheek FFT / CIELab ΔE / CLIP, chuẩn hóa 512×512 (DINOv2 snap 504×504).</p>
            </div>
            <div class="intent-card">
              <div class="intent-card-top">
                <span class="intent-index">TẦNG 2</span>
                <span class="intent-role">QUANG HỌC &amp; ĐỘ NÉT</span>
              </div>
              <h3>Laplacian Adaptive Quantile (q=5%)</h3>
              <p>Đo biến sai Laplacian (×1000 ≥ 1.25) hiệu chỉnh theo phân vị 5% ảnh thật, kiểm tra cháy sáng &amp; vết cắt ngang.</p>
            </div>
            <div class="intent-card">
              <div class="intent-card-top">
                <span class="intent-index">TẦNG 3</span>
                <span class="intent-role">CHỐNG TRÙNG LẶP</span>
              </div>
              <h3>SHA-256, Perceptual dHash &amp; Cosine</h3>
              <p>Khử trùng lặp băm SHA-256 + dHash kết hợp khoảng cách Cosine ≥ 0.05 đảm bảo 0 bản sao trùng lặp trong tập huấn luyện.</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 05: KHAI PHÁ __HARD_TOT__ HARD SAMPLES & ARTIFACT 4 DỊ TẬT SINH HỌC
         ======================================================================= -->
    <section class="slide" id="slide-6" data-title="05 / Khai Phá Mẫu Khó">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">05 / BẪY THỬ THÁCH MÙ</span>
              <span class="section-subtitle">TIÊU CHUẨN TOÁN HỌC & 4 DỊ TẬT SINH HỌC VI THỂ</span>
            </div>
            <h2 class="slide-title">
              Challenge Vault: <span>Bộ Sưu Tập Mẫu Khó & 4 Dị Tật Sinh Học Vi Thể</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Cô lập __CHAL_TOT__ mẫu khó nhất vào Két bẫy Challenge Vault làm thước đo bất biến kiểm chứng năng lực phá vỡ rào cản của mô hình qua các vòng học.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: The Real Biological Defect Gallery Artifact -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Mẫu Khó Thực Tế: Bộ Sưu Tập 4 Dị Tật Sinh Học (Challenge Vault)
                </div>
                <span class="panel-badge badge-alert">CHALLENGE VAULT GALLERY</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/extracted_nb/02_hard_samples_and_active_learning_cell6_img2.png" class="artifact-img" alt="Hard Samples Biological Defect Taxonomy">
              </div>
            </div>

            <!-- Right: Mathematical Criteria & Taxonomy Breakdown -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  2 Tiêu Chuẩn Toán Học & Phân Tích 4 Điểm Mù Sinh Học
                </div>
                <span class="panel-badge badge-champion">TOÁN HỌC KHAI PHÁ</span>
              </div>
              <div style="padding: 18px; display: flex; flex-direction: column; gap: 12px; height: 480px; justify-content: space-between;">
                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px;">
                  <span style="font-weight: 700; font-size: 13px; color: var(--primary);">Tiêu Chuẩn 1: Ranh Giới Mơ Hồ (Margin Ambiguity)</span>
                  <div style="background: var(--surface-soft); padding: 6px 10px; border-radius: 4px; font-family: var(--font-mono); font-size: 12px; margin: 4px 0;">
                    |P(Fake | x) - 0.50| ≤ 0.15  ⟺  P(Fake) ∈ [0.35, 0.65]
                  </div>
                  <span style="font-size: 11px; color: var(--text-muted);">Mẫu nằm sát ranh giới phân định, bộ dò ở trạng thái bất định cực đại.</span>
                </div>

                <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: var(--radius-md); padding: 12px;">
                  <span style="font-weight: 700; font-size: 13px; color: #b91c1c;">Tiêu Chuẩn 2: Đảo Chiều Tự Tin Cao (High-Confidence Inversion)</span>
                  <div style="background: #ffffff; padding: 6px 10px; border-radius: 4px; font-family: var(--font-mono); font-size: 11px; margin: 4px 0; color: #9f1239;">
                    • False Negative (FN): Fake thật nhưng P(Fake) &lt; 0.20<br>
                    • False Positive (FP): Người thật nhưng P(Fake) &gt; 0.80
                  </div>
                  <span style="font-size: 11px; color: #881337;">Mô hình cực kỳ tự tin nhưng dự đoán sai lầm hoàn toàn (bẫy eKYC nguy hại).</span>
                </div>

                <div style="background: #fdfae7; border: 1px solid rgba(17,17,17,0.1); border-radius: var(--radius-md); padding: 12px;">
                  <span style="font-weight: 700; font-size: 12px;">4 Dị Tật Sinh Học Được Xác Nhận:</span>
                  <div style="font-size: 11px; color: var(--text); line-height: 1.5; margin-top: 4px;">
                    <strong>1. Đồng tử:</strong> Phản chiếu giác mạc bất đối xứng hướng sáng.<br>
                    <strong>2. Men răng:</strong> Ranh giới kẽ răng bị hòa tan liên tục.<br>
                    <strong>3. Lỗ chân lông:</strong> Da bị làm mịn quá mức (Plastic Skin Effect).<br>
                    <strong>4. Chân tóc:</strong> Tóc tơ hòa trộn dị thường vào phông nền.
                  </div>
                </div>

                <div style="text-align: center; font-size: 11px; color: var(--text-muted);">
                  Tổng số mẫu khó khai phá: <strong>__HARD_TOT__ mẫu</strong> → Khóa cứng <strong>__CHAL_TOT__ mẫu độc lập</strong> làm Challenge Vault!
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 06: ĐỘT PHÁ VÒNG 2 ACTIVE LEARNING (M2*) & RECALL SURGE ARTIFACT
         ======================================================================= -->
    <section class="slide" id="slide-7" data-title="06 / Đột Phá Active Learning">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">06 / ĐỘT PHÁ ACTIVE LEARNING</span>
              <span class="section-subtitle">BIÊN QUYẾT ĐỊNH MỚI &amp; BƯỚC NHẢY VỌT +__M2S_JUMP__ RECALL</span>
            </div>
            <h2 class="slide-title">
              Active Learning Vòng 2: <span>M2* Nâng Fake Recall Từ __M0_C_REC__ Lên __M2S_C_REC__</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Chiến lược mở khóa có kiểm soát Block 11 của DINOv2 kết hợp khai thác mẫu khó đã tạo nên bước nhảy vọt lịch sử, triệt tiêu gần như hoàn toàn cửa lọt lưới.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: Real Challenge Recall Surge Artifact from Notebook 02 -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Thực Chứng: Biểu Đồ Cú Nhảy Vọt Fake Recall Trên __CHAL_TOT__ Challenge Samples
                </div>
                <span class="panel-badge badge-champion">ARTIFACT NOTEBOOK 02</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/extracted_nb/02_hard_samples_and_active_learning_cell9_img3.png" class="artifact-img" alt="Challenge Vault Recall Jump">
              </div>
            </div>

            <!-- Right: Precision-Recall Curves Benchmark Artifact -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Đường Cong Precision-Recall: So Sánh 5 Thế Hệ Mô Hình (Universal Benchmark 504px)
                </div>
                <span class="panel-badge badge-success">UNIVERSAL BENCHMARK</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/benchmarks/benchmark_5models_precision_recall_curves.png" class="artifact-img" alt="Precision-Recall Curves 5 Models (Universal Benchmark 504px)">
              </div>
            </div>
          </div>

          <!-- Bottom Summary Metrics -->
          <div class="card-grid-3" style="margin-top: 14px;">
            <div class="intent-card" style="text-align: center;">
              <span style="font-size: 11px; font-weight: 700; color: var(--text-muted);">M0 BASELINE D0</span>
              <div style="font-size: 26px; font-weight: 800; color: #b91c1c; font-family: var(--font-display); margin: 4px 0;">__M0_C_REC__</div>
              <span style="font-size: 11px; color: #991b1b;">Bỏ sót __M0_C_BYP__ mẫu khó</span>
            </div>
            <div class="intent-card" style="text-align: center;">
              <span style="font-size: 11px; font-weight: 700; color: var(--text-muted);">M1 ACTIVE ROUND 1</span>
              <div style="font-size: 26px; font-weight: 800; color: #f59e0b; font-family: var(--font-display); margin: 4px 0;">__M1_C_REC__</div>
              <span style="font-size: 11px; color: #b45309;">Bước nhảy đầu tiên (+__M1_JUMP__)</span>
            </div>
            <div class="intent-card" style="text-align: center; border-color: var(--primary);">
              <span style="font-size: 11px; font-weight: 700; color: var(--primary);">🏆 M2* REFINED WARM-START</span>
              <div style="font-size: 26px; font-weight: 800; color: #15803d; font-family: var(--font-display); margin: 4px 0;">__M2S_C_REC__</div>
              <span style="font-size: 11px; color: #166534; font-weight: 700;">Bắt trúng __M2S_C_DET__/__CHAL_FAKE__ mẫu giả (AUC __M2S_C_AUC__)!</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 07: BẢO TOÀN LIÊM CHÍNH: ZERO IDENTITY LEAKAGE TRÊN D3
         ======================================================================= -->
    <section class="slide" id="slide-8" data-title="07 / Liêm Chính Zero Leakage">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">07 / NGUYÊN TẮC LIÊM CHÍNH</span>
              <span class="section-subtitle">PHÁT HIỆN & CÁCH LY TRIỆT ĐỂ RÒ RỈ DANH TÍNH</span>
            </div>
            <h2 class="slide-title">
              Zero Identity Leakage: <span>Bảo Toàn 100% Liêm Chính Học Thuật</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Nhóm phát hiện __UNSEEN_F__ ảnh RealisticVision mang khuôn mặt của tập test; lập tức cách ly 100% làm tập kiểm thử mù độc lập, loại bỏ hoàn toàn nguy cơ gian lận mô hình.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: Real Zero Leakage Partition Diagram Artifact from Notebook 03 -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Phân Vùng Sạch: Sơ Đồ Cách Ly Rò Rỉ Danh Tính Tập D3
                </div>
                <span class="panel-badge badge-success">ARTIFACT NOTEBOOK 03</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/extracted_nb/03_champion_m3_and_universal_benchmark_cell3_img0.png" class="artifact-img" alt="Zero Identity Leakage Clean Partition Schema">
              </div>
            </div>

            <!-- Right: The Clean D3 Partition Solution Table -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Cấu Trúc Tập Huấn Luyện D3 Cân Bằng Chuẩn 1:1 (__D3_TR__ Ảnh)
                </div>
                <span class="panel-badge badge-champion">ZERO LEAKAGE ĐẠT CHUẨN</span>
              </div>
              <div style="padding: 16px;">
                <table class="scorecard-table">
                  <thead>
                    <tr>
                      <th>Nhóm Dữ Liệu Train D3</th>
                      <th>Số Lượng</th>
                      <th>Nguồn Gốc & Ý Nghĩa</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>Ảnh Thật (Label 0)</strong></td>
                      <td><strong>__REAL_TR__</strong></td>
                      <td>100% ảnh thật gốc tập Train (FFHQ + CelebA-HQ)</td>
                    </tr>
                    <tr>
                      <td><strong>Mẫu Khó SD 1.5 (D2_hard)</strong></td>
                      <td>__HARD_TR__</td>
                      <td>Mẫu khó từng vượt mặt M0/M1 ở Vòng 2</td>
                    </tr>
                    <tr>
                      <td><strong>Ảnh Giả SD 1.5 Cơ Sở (D0)</strong></td>
                      <td>__D3_SD_BASE__</td>
                      <td>Bảo tồn phân phối cơ sở tránh quên tri thức</td>
                    </tr>
                    <tr>
                      <td><strong>Ảnh Giả RV5.1 Sạch Danh Tính</strong></td>
                      <td>__D3_RV_CLEAN__</td>
                      <td>__D3_RV_T2I__ Text2Img + __D3_RV_INP__ Inpaint + __D3_RV_I2I__ Img2Img (Sạch ID)</td>
                    </tr>
                    <tr class="highlight-champion">
                      <td><strong>TỔNG CỘNG TẬP TRAIN D3</strong></td>
                      <td><strong>__D3_TR__</strong></td>
                      <td><strong>Cân Bằng 1:1 (__REAL_TR__ Thật : __REAL_TR__ Giả) • 0 Ảnh Rò Rỉ</strong></td>
                    </tr>
                  </tbody>
                </table>

                <div style="margin-top: 14px; background: #dcfce7; border: 1px solid #86efac; border-radius: var(--radius-md); padding: 12px;">
                  <span style="font-weight: 700; font-size: 12px; color: #166534;">✓ Cách Ly __UNSEEN_F__ Ảnh Làm Tập Kiểm Thử Mù Riêng Biệt:</span>
                  <span style="font-size: 11px; color: #14532d; display: block; margin-top: 4px;">
                    Kho <code>unseen_rv51_test.jsonl</code> (__UNSEEN_TOT__ ảnh gồm __UNSEEN_F__ fake + __REAL_TE__ test reals) được bảo vệ hoàn toàn, trở thành thước đo kiểm thử mù liên bộ sinh (Cross-Generator) khắt khe nhất của dự án!
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 08: NGHIỆM THU CHAMPION M3 & REAL QUAD-VAULT BENCHMARK FIGURE
         ======================================================================= -->
    <section class="slide" id="slide-9" data-title="08 / Champion M3 Quad-Vault">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">08 / ĐỐI CHUẨN TỨ GIÁC (QUAD-VAULT)</span>
              <span class="section-subtitle">ĐÁNH GIÁ 5 MÔ HÌNH TRÊN 3 KHO KIỂM THỬ ĐỘC LẬP</span>
            </div>
            <h2 class="slide-title">
              Champion M3: <span>Bứt Phá __M3_U_REC__ Quad-Vault Mà Không Quên SD 1.5</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Mô hình vô địch M3 Warm-Start từ M2* tạo lập kỷ lục mới toàn diện: Fake Recall đạt __M3_U_REC__ trên tập mù RV5.1 đồng thời đạt __M3_C_REC__ trên Challenge Vault.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: Real Quad-Vault Recall Comparison Chart Artifact -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Đối Chuẩn Tứ Giác: Biểu Đồ Recall 5 Mô Hình Trên 3 Kho Độc Lập
                </div>
                <span class="panel-badge badge-champion">QUAD-VAULT FIGURE</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/benchmarks/benchmark_5models_vault_recall_comparison.png" class="artifact-img" alt="Quad-Vault Recall Comparison Across 5 Models">
              </div>
            </div>

            <!-- Right: Full Quad-Vault Scorecard Table & Convergence -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Bảng Điểm Sát Hạch Nghiệm Thu Quad-Vault Toàn Diện
                </div>
                <span class="panel-badge badge-success">SỐ LIỆU ĐÃ KIỂM CHỨNG</span>
              </div>
              <div style="padding: 16px;">
                <table class="scorecard-table">
                  <thead>
                    <tr>
                      <th>Phiên Bản Mô Hình</th>
                      <th>Master Test (SD 1.5)<br><span style="font-size: 10px; font-weight: normal;">__MASTER_TOT__ ảnh</span></th>
                      <th>Challenge Vault<br><span style="font-size: 10px; font-weight: normal;">__CHAL_TOT__ mẫu khó</span></th>
                      <th>Unseen RV5.1 Vault<br><span style="font-size: 10px; font-weight: normal;">__UNSEEN_TOT__ ảnh mù</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>M0 (D0 Baseline)</strong></td>
                      <td>__M0_M_REC__</td>
                      <td>__M0_C_REC__</td>
                      <td>__M0_U_REC__</td>
                    </tr>
                    <tr>
                      <td><strong>M1 (Active Round 1)</strong></td>
                      <td>__M1_M_REC__</td>
                      <td>__M1_C_REC__</td>
                      <td>__M1_U_REC__</td>
                    </tr>
                    <tr>
                      <td><strong>M2 (Cold-Start Retrain)</strong></td>
                      <td>__M2_M_REC__</td>
                      <td>__M2_C_REC__</td>
                      <td>__M2_U_REC__</td>
                    </tr>
                    <tr>
                      <td><strong>M2* (Refined Warm-Start)</strong></td>
                      <td>__M2S_M_REC__</td>
                      <td><strong>__M2S_C_REC__</strong></td>
                      <td>__M2S_U_REC__</td>
                    </tr>
                    <tr class="highlight-champion">
                      <td><strong>🏆 M3 (Dual-Engine Champion)</strong></td>
                      <td><span class="delta-pill-green">__M3_M_REC__ (Kháng quên)</span></td>
                      <td><span class="delta-pill-green">__M3_C_REC__ (Bảo toàn)</span></td>
                      <td><span class="delta-pill-green">__M3_U_REC__ (+__M3_U_JUMP__)</span></td>
                    </tr>
                  </tbody>
                </table>

                <!-- 3 Key Findings -->
                <div style="margin-top: 14px; display: flex; flex-direction: column; gap: 8px;">
                  <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 11px;">
                    <strong>1. Không quên SD 1.5:</strong> Fake Recall Master Frozen Test đạt <strong>__M3_M_REC__</strong> (vượt xa M0 __M0_M_REC__ &amp; M2* __M2S_M_REC__).
                  </div>
                  <div style="background: #ffffff; border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; font-size: 11px;">
                    <strong>2. Bảo vệ két khó:</strong> Đạt <strong>__M3_C_REC__</strong> mẫu khó Challenge Vault (Bypass chỉ __M3_C_BYP__).
                  </div>
                  <div style="background: #eef0ff; border: 1px solid var(--border-strong); border-radius: 6px; padding: 8px 12px; font-size: 11px; color: var(--primary);">
                    <strong>3. Bứt phá liên bộ sinh:</strong> Fake Recall vọt lên <strong>__M3_U_REC__</strong> trên kho ảnh mù RV5.1 (Bypass giảm sâu __M3_U_BYP__)!
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 09: ĐỐI CHUẨN TOÀN CẦU (UNIVERSAL BENCHMARK) & HEATMAP ARTIFACT
         ======================================================================= -->
    <section class="slide" id="slide-10" data-title="09 / Universal Benchmark 20.4k">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">09 / ĐỐI CHUẨN TOÀN CẦU</span>
              <span class="section-subtitle">20.400 ẢNH QUA 34 THUẬT TOÁN & 8 HỌ CÔNG NGHỆ</span>
            </div>
            <h2 class="slide-title">
              Universal Benchmark: <span>Ma Trận Nhiệt 34 Phương Pháp & 8 Họ Công Nghệ</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Kiểm thử quy mô lớn trên 34 phương pháp sinh (GAN, Diffusion, Face-Swap, Talking Heads); M3 vượt trội ở ngưỡng nghiêm ngặt cho bài toán eKYC ngân hàng.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: Real 34-Method Heatmap Benchmark Artifact -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Ma Trận Nhiệt: Khảo Sát Chi Tiết Trên 34 Bộ Sinh Toàn Cầu
                </div>
                <span class="panel-badge badge-champion">34-METHOD HEATMAP FIGURE</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/benchmarks/benchmark_5models_method_heatmap.png" class="artifact-img" alt="Universal Benchmark 34-Method Heatmap">
              </div>
            </div>

            <!-- Right: Real Family Recall Breakdown Chart -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Phân Bổ Theo 8 Họ Công Nghệ & Khảo Sát Ngưỡng eKYC
                </div>
                <span class="panel-badge badge-success">8-FAMILY RECALL FIGURE</span>
              </div>
              <div style="padding: 12px; display: flex; flex-direction: column; justify-content: space-between; height: 480px;">
                <div class="artifact-img-box" style="height: 260px;">
                  <img src="assets/benchmarks/benchmark_5models_family_recall.png" class="artifact-img" alt="8 Technology Families Recall Chart" style="max-height: 250px;">
                </div>

                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px;">
                  <span style="font-weight: 700; font-size: 12px; color: var(--primary);">Khảo sát ngưỡng vận hành eKYC Ngân hàng (@ 512→504px):</span>
                  <div style="font-size: 11px; color: var(--text); margin-top: 4px; line-height: 1.5;">
                    • Tại ngưỡng mặc định <code>τ = 0.50</code>: M3 đạt Fake Recall toàn cầu <strong>__U_M3_REC__</strong> (SD 2.1: <strong>__UF_SD21__</strong>, GAN: <strong>__UF_GAN__</strong>), Real Spec <strong>__U_M3_SPEC__</strong>; nâng ngưỡng <code>τ = 0.85</code> hạ FPR trên video FF++ C23 còn <strong>__FF_FPR_085__</strong>.<br>
                    • Tại ngưỡng cực gắt <code>FPR ≤ 1.0%</code> (<code>τ = __U_M3_T1__</code>): Fake Recall đạt <strong>__U_M3_REC1__</strong> (M0 tại <code>τ = __U_M0_T1__</code>: __U_M0_REC1__) — thể hiện trung thực khoảng trống miền Zero-Shot ngoài Diffusion.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 10: FORENSIC LIVE INSPECTOR & DINOv2 SPATIAL ATTENTION HEATMAP
         ======================================================================= -->
    <section class="slide" id="slide-11" data-title="10 / Live Forensic Inspector">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">10 / GIẢI THÍCH MÔ HÌNH (XAI)</span>
              <span class="section-subtitle">DINOv2 SELF-ATTENTION SPATIAL HEATMAP LIVE DEMO</span>
            </div>
            <h2 class="slide-title">
              Forensic Inspector: <span>Bản Đồ Chú Ý DINOv2 Soi Vùng Can Thiệp Sâu</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Mô hình không phải là hộp đen — cơ chế trích xuất Self-Attention Head từ Block 11 định vị chính xác khu vực chỉnh sửa giả mạo trên khuôn mặt theo thời gian thực (18.5 ms).
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: Real DINOv2 Attention Heatmap Artifact from Notebook 00 -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Giám Định Trực Tiếp: Bản Đồ Chú Ý Không Gian (Spatial Attention Heatmap)
                </div>
                <span class="panel-badge badge-champion">ARTIFACT NOTEBOOK 00</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/extracted_nb/00_master_deepfake_defense_showcase_cell7_img1.png" class="artifact-img" alt="DINOv2 Spatial Attention Heatmap Forensic Inspector">
              </div>
            </div>

            <!-- Right: Direct Demo Bridge & Explanation -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Trực Tiếp Trải Nghiệm Demo Khảo Sát Trên VSCode
                </div>
                <span class="panel-badge badge-success">LIVE INFERENCE READY</span>
              </div>
              <div style="padding: 20px; display: flex; flex-direction: column; justify-content: space-around; height: 480px;">
                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px;">
                  <span style="font-weight: 700; font-size: 14px; color: var(--primary);">Khởi Chạy Notebook 00 Trực Tiếp:</span>
                  <p style="font-size: 12px; color: var(--text-muted); margin-top: 6px; line-height: 1.5;">
                    Hàm <code>interactive_forensic_inspector()</code> trong Notebook 00 cho phép Hội đồng Giảng viên chọn ngẫu nhiên bất kỳ khuôn mặt nào để soi ngay bản đồ nhiệt Self-Attention và xuất xác suất giả mạo.
                  </p>
                  <div style="margin-top: 14px;">
                    <a href="../../notebooks/coursework/00_master_deepfake_defense_showcase.ipynb" class="btn-live-demo" target="_blank" style="display: inline-flex; width: 100%; justify-content: center;">
                      🚀 MỞ NOTEBOOK 00 TRÊN VSCODE ĐỂ DEMO LIVE
                    </a>
                  </div>
                </div>

                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 14px;">
                  <span style="font-weight: 700; font-size: 13px; color: var(--text);">Thông số suy luận thời gian thực:</span>
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; font-family: var(--font-mono); font-size: 11px;">
                    <div style="background: var(--surface-soft); padding: 8px; border-radius: 4px;">Tốc Độ: ~18.5 ms/ảnh</div>
                    <div style="background: var(--surface-soft); padding: 8px; border-radius: 4px;">Bộ Nhớ: &lt; 1.2 GB VRAM</div>
                    <div style="background: var(--surface-soft); padding: 8px; border-radius: 4px;">Độ Phân Giải: 512→504px</div>
                    <div style="background: var(--surface-soft); padding: 8px; border-radius: 4px;">Kích Thước: 22M Params</div>
                  </div>
                </div>

                <div style="background: #dcfce7; border: 1px solid #86efac; border-radius: var(--radius-md); padding: 12px; text-align: center;">
                  <span style="font-size: 12px; color: #166534; font-weight: 700;">Vùng can thiệp mắt, sống mũi và viền chân tóc sáng rực điểm chú ý của DINOv2!</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =======================================================================
         SLIDE 11: TỔNG KẾT ĐỒ ÁN & HỘI TỤ HUẤN LUYỆN CHAMPION M3 ARTIFACT
         ======================================================================= -->
    <section class="slide" id="slide-12" data-title="11 / Tổng Kết Nghiệm Thu">
      <div class="slide-inner">
        <header class="slide-header">
          <div class="header-left">
            <div class="eyebrow-row">
              <span class="section-tag">11 / TỔNG KẾT & KẾT LUẬN</span>
              <span class="section-subtitle">ĐÓNG GÓP KHOA HỌC & GIÁ TRỊ THỰC TIỄN CHO eKYC NGÂN HÀNG</span>
            </div>
            <h2 class="slide-title">
              Nghiệm Thu Toàn Diện: <span>3 Đóng Góp Kỹ Thuật & Sẵn Sàng Triển Khai</span>
            </h2>
          </div>
          <div class="header-right">
            <div class="header-takeaway">
              Đề tài 11 hoàn thành 100% mục tiêu, cung cấp giải pháp phòng thủ deepfake khuếch tán thực chiến, hiệu quả cao và liêm chính tuyệt đối.
            </div>
          </div>
        </header>

        <div class="slide-body">
          <div class="split-2col">
            <!-- Left: Training convergence dynamics artifact from Notebook 03 -->
            <div class="visual-panel accent-border">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  Artifact Động Học Hội Tụ Huấn Luyện Champion M3 (Loss & AUC Qua 6 Epochs)
                </div>
                <span class="panel-badge badge-champion">CONVERGENCE FIGURE</span>
              </div>
              <div class="artifact-img-box">
                <img src="assets/extracted_nb/03_champion_m3_and_universal_benchmark_cell5_img1.png" class="artifact-img" alt="Champion M3 6-Epoch Convergence Dynamics">
              </div>
            </div>

            <!-- Right: 3 Key Pillars of Contribution -->
            <div class="visual-panel">
              <div class="panel-title-bar">
                <div class="panel-title">
                  <span class="panel-title-dot"></span>
                  3 Đóng Góp Kỹ Thuật Cốt Lõi Cho Lĩnh Vực An Ninh eKYC
                </div>
                <span class="panel-badge badge-success">ĐÓNG GÓP ĐỒ ÁN</span>
              </div>
              <div style="padding: 16px; display: flex; flex-direction: column; gap: 12px; height: 480px; justify-content: space-between;">
                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px;">
                  <span style="font-weight: 700; font-size: 13px; color: var(--primary);">1. Quy Trình Dữ Liệu Khuếch Tán Kép Soft-Feathering:</span>
                  <p style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                    Kho __FAKE_RAW__ ảnh từ SD 1.5 và RV5.1 triệt tiêu vết cắt biên nhân tạo; phễu lọc 3 tầng CurationGate mở rộng đảm bảo 100% sạch trùng lặp và đạt chuẩn chân thực sinh học.
                  </p>
                </div>

                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px;">
                  <span style="font-weight: 700; font-size: 13px; color: #15803d;">2. Active Learning Đào Sâu Mẫu Khó &amp; Challenge Vault:</span>
                  <p style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                    Khai phá __HARD_TOT__ ca biên khó, cô lập két bẫy __CHAL_TOT__ mẫu bất biến; chứng minh thực chứng bước nhảy vọt Fake Recall từ __M0_C_REC__ lên __M2S_C_REC__ (+__M2S_JUMP__).
                  </p>
                </div>

                <div style="background: #ffffff; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 12px;">
                  <span style="font-weight: 700; font-size: 13px; color: var(--primary);">3. Champion M3 &amp; Liêm Chính Zero Identity Leakage:</span>
                  <p style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                    Thích nghi liên bộ sinh xuất sắc, đạt Quad-Vault Fake Recall __M3_U_REC__ mà không làm giảm nhận thức trên SD 1.5; tốc độ suy luận 18.5 ms/ảnh sẵn sàng tích hợp cổng ngân hàng.
                  </p>
                </div>

                <div style="background: #fdfae7; border: 2px solid var(--primary); border-radius: var(--radius-md); padding: 12px; text-align: center;">
                  <span style="font-weight: 800; font-size: 13px; color: var(--primary);">
                    KÍNH CẢM ƠN HỘI ĐỒNG GIẢNG VIÊN BỘ MÔN THỊ GIÁC MÁY TÍNH UTH!
                  </span>
                  <span style="font-size: 11px; color: var(--text-muted); display: block; margin-top: 2px;">
                    Nhóm 4 kính mời Thầy Cô đặt câu hỏi phản biện kỹ thuật và đóng góp ý kiến.
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
    """

    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Topic 11 — Báo Cáo Nghiệm Thu Đồ Án | UTH Computer Vision (Nhóm 4)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
{base_css}
{acceptance_css}
  </style>
</head>
<body>
  <div class="deck-viewport">
    <div class="deck-stage" id="deck-stage">
      <div class="progress-bar" id="progress-bar"></div>
{slides_html}
      <div class="slide-counter" id="slide-counter">SLIDE 01 / 12</div>
      <div class="nav-controls" aria-label="Slide navigation">
        <button type="button" class="nav-btn" id="prev-btn" title="Previous Slide (←)">‹</button>
        <div class="nav-dots" id="nav-dots"></div>
        <button type="button" class="nav-btn" id="next-btn" title="Next Slide (→)">›</button>
      </div>
    </div>
  </div>

  <script>
{base_js}
  </script>
</body>
</html>
"""
    for tok, val in _build_presentation_subs().items():
        full_html = full_html.replace(tok, val)
    return full_html

def main():
    html_content = build_presentation_html()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Successfully generated {OUTPUT_PATH} ({len(html_content)} bytes)")

if __name__ == "__main__":
    main()
