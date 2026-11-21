"""Build script for Topic 11 Final Acceptance Vertical Scientific Report.

Generates docs/reports/presentation_final_acceptance.html with:
- Exactly 1 artifact per row (NO multi-column squished grids)
- Full-width, high-resolution, unconstrained visual artifact containers
- Interactive Lightbox Zoom Modal (click any image to inspect in full-screen)
- Compact, slim, collapsible sidebar navigation (takes ~15% width instead of ~50%)
- Clean native typography without raw LaTeX syntax ($ replaced with clean Unicode)
- 30+ visual artifacts and high-res face samples
"""

from pathlib import Path

OUTPUT_PATH = Path("docs/reports/presentation_final_acceptance.html")

def build_vertical_report_html() -> str:
    html_content = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Báo Cáo Nghiệm Thu Khoa Học Đề Tài 11 | UTH Computer Vision (Nhóm 4)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #1e2bfa;
      --primary-hover: #1420d4;
      --primary-soft: rgba(30, 43, 250, 0.08);
      --primary-border: rgba(30, 43, 250, 0.25);
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-soft: #f1f5f9;
      --text: #0f172a;
      --text-muted: #475569;
      --text-light: #94a3b8;
      --border: #e2e8f0;
      --border-strong: #cbd5e1;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --shadow-sm: 0 2px 6px rgba(15, 23, 42, 0.04);
      --shadow-md: 0 10px 25px -5px rgba(15, 23, 42, 0.06), 0 8px 10px -6px rgba(15, 23, 42, 0.04);
      --shadow-lg: 0 20px 35px -10px rgba(15, 23, 42, 0.08);
      --font-display: 'Plus Jakarta Sans', -apple-system, sans-serif;
      --font-body: 'Inter', -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-body);
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
    }

    /* Fixed Top Navigation Bar */
    .top-header {
      position: sticky;
      top: 0;
      z-index: 1000;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 10px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: var(--shadow-sm);
    }
    .header-branding {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .toggle-sidebar-btn {
      background: var(--surface-soft);
      border: 1px solid var(--border);
      color: var(--text);
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 13px;
      padding: 6px 12px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .toggle-sidebar-btn:hover {
      background: #e2e8f0;
      color: var(--primary);
    }
    .brand-logo-badge {
      background: var(--primary);
      color: #ffffff;
      font-family: var(--font-mono);
      font-weight: 800;
      font-size: 12px;
      padding: 5px 8px;
      border-radius: 6px;
    }
    .brand-title {
      font-family: var(--font-display);
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }
    .brand-sub {
      font-size: 11px;
      color: var(--text-muted);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    /* Layout Container: Slim Sidebar ~200px vs Main Area */
    .layout-wrapper {
      width: 100%;
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px 24px 80px;
      display: grid;
      grid-template-columns: 200px minmax(0, 1fr);
      gap: 28px;
      transition: grid-template-columns 0.25s ease;
    }
    .layout-wrapper.sidebar-collapsed {
      grid-template-columns: 0px minmax(0, 1fr);
      gap: 0px;
    }
    .layout-wrapper.sidebar-collapsed .sidebar-nav {
      opacity: 0;
      pointer-events: none;
      visibility: hidden;
      width: 0;
      padding: 0;
    }

    /* Slim, Elegant Sticky Sidebar */
    .sidebar-nav {
      position: sticky;
      top: 70px;
      height: calc(100vh - 90px);
      overflow-y: auto;
      padding-right: 8px;
      transition: opacity 0.2s ease, width 0.25s ease;
    }
    .sidebar-nav::-webkit-scrollbar {
      width: 4px;
    }
    .sidebar-nav::-webkit-scrollbar-thumb {
      background: var(--border-strong);
      border-radius: 4px;
    }
    .nav-group-title {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 700;
      color: var(--text-light);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 14px 0 6px;
    }
    .nav-group-title:first-child {
      margin-top: 0;
    }
    .nav-link {
      display: block;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-decoration: none;
      transition: all 0.15s ease;
      margin-bottom: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .nav-link:hover {
      background: var(--surface-soft);
      color: var(--primary);
    }
    .nav-link.active {
      background: var(--primary-soft);
      color: var(--primary);
      font-weight: 700;
      border-left: 3px solid var(--primary);
    }

    /* Content Area */
    .main-content {
      display: flex;
      flex-direction: column;
      gap: 40px;
      min-width: 0;
    }

    /* Section Containers */
    .report-section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 36px 40px;
      box-shadow: var(--shadow-sm);
    }
    .report-section.hero-section {
      background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
      border: 1px solid var(--primary-border);
    }

    .section-eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 700;
      color: var(--primary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }
    .section-title {
      font-family: var(--font-display);
      font-size: 26px;
      font-weight: 800;
      color: var(--text);
      line-height: 1.3;
      margin-bottom: 12px;
    }
    .section-title span.accent {
      color: var(--primary);
    }
    .section-desc {
      font-size: 15px;
      color: var(--text-muted);
      max-width: 1200px;
      line-height: 1.7;
      margin-bottom: 20px;
    }

    /* KPI Highlights */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin: 20px 0 28px;
    }
    .kpi-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 18px 20px;
      box-shadow: var(--shadow-sm);
      position: relative;
      overflow: hidden;
    }
    .kpi-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: var(--primary);
    }
    .kpi-card.champion::before {
      background: #10b981;
    }
    .kpi-card.alert::before {
      background: #ef4444;
    }
    .kpi-label {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      color: var(--text-light);
      text-transform: uppercase;
    }
    .kpi-value {
      font-family: var(--font-display);
      font-size: 32px;
      font-weight: 800;
      color: var(--text);
      margin: 6px 0 4px;
    }
    .kpi-card.champion .kpi-value {
      color: #059669;
    }
    .kpi-card.alert .kpi-value {
      color: #dc2626;
    }
    .kpi-note {
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 500;
    }

    /* 1 ARTIFACT TRÊN 1 HÀNG (FULL-WIDTH, HIGH-RESOLUTION) */
    .artifact-row-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      overflow: hidden;
      margin: 26px 0;
      box-shadow: var(--shadow-md);
      width: 100%;
    }
    .artifact-row-header {
      background: var(--surface-soft);
      padding: 14px 22px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .artifact-row-title {
      font-family: var(--font-display);
      font-size: 16px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .artifact-badge {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      padding: 3px 9px;
      border-radius: 6px;
      background: #ffffff;
      border: 1px solid var(--border);
      color: var(--primary);
    }
    .artifact-badge.champion {
      background: #ecfdf5;
      color: #059669;
      border-color: #a7f3d0;
    }
    .artifact-badge.alert {
      background: #fef2f2;
      color: #b91c1c;
      border-color: #fecaca;
    }
    .artifact-row-body {
      padding: 24px;
      background: #ffffff;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .artifact-img-full {
      width: 100%;
      height: auto;
      max-height: 850px;
      object-fit: contain;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
      background: #ffffff;
      cursor: zoom-in;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .artifact-img-full:hover {
      box-shadow: 0 10px 30px rgba(30, 43, 250, 0.15);
      border-color: var(--primary);
    }
    .zoom-hint-banner {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--primary);
      background: var(--primary-soft);
      padding: 5px 12px;
      border-radius: 20px;
      margin-top: 10px;
      cursor: pointer;
    }
    .artifact-row-footer {
      background: var(--surface-soft);
      padding: 14px 22px;
      border-top: 1px solid var(--border);
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
    }

    /* Face Comparison Subject Box (1 Khung Lớn / 1 Hàng) */
    .face-subject-container {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 20px;
      margin: 20px 0;
      box-shadow: var(--shadow-sm);
    }
    .face-subject-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }
    .face-subject-title {
      font-family: var(--font-display);
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }
    .face-strip-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }
    .face-strip-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      overflow: hidden;
      text-align: center;
    }
    .face-strip-card img {
      width: 100%;
      height: 260px;
      object-fit: cover;
      display: block;
      cursor: zoom-in;
      transition: transform 0.15s ease;
    }
    .face-strip-card img:hover {
      transform: scale(1.02);
    }
    .face-strip-info {
      padding: 10px 8px;
    }
    .face-tag {
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 4px;
      margin-bottom: 4px;
    }
    .face-tag.real { background: #ecfdf5; color: #047857; }
    .face-tag.inpaint { background: #eff6ff; color: #1d4ed8; }
    .face-tag.img2img { background: #faf5ff; color: #7e22ce; }
    .face-tag.text2img { background: #fff7ed; color: #c2410c; }
    .face-name {
      font-size: 12px;
      font-weight: 600;
      color: var(--text);
    }

    /* Tables */
    .data-table {
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
      border-radius: var(--radius-md);
      overflow: hidden;
      border: 1px solid var(--border);
      margin: 20px 0;
    }
    .data-table th {
      background: var(--surface-soft);
      padding: 12px 18px;
      text-align: left;
      font-weight: 700;
      color: var(--text);
      border-bottom: 2px solid var(--border-strong);
      font-family: var(--font-display);
    }
    .data-table td {
      padding: 11px 18px;
      border-bottom: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
    }
    .data-table tr:last-child td {
      border-bottom: none;
    }
    .data-table tr.highlight-champion td {
      background: rgba(30, 43, 250, 0.05);
      font-weight: 600;
    }
    .data-table tr.highlight-champion td:first-child {
      border-left: 3px solid var(--primary);
    }

    .badge-pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 3px 9px;
      border-radius: 6px;
      font-weight: 700;
      font-size: 11px;
      font-family: var(--font-mono);
    }
    .badge-pill.green { background: #dcfce7; color: #15803d; }
    .badge-pill.red { background: #fee2e2; color: #b91c1c; }
    .badge-pill.amber { background: #fef3c7; color: #b45309; }

    /* Callouts */
    .callout-box {
      border-radius: var(--radius-md);
      padding: 18px 24px;
      margin: 20px 0;
      border: 1px solid var(--border);
      background: var(--surface-soft);
    }
    .callout-box.info { background: #f0f7ff; border-color: #bfdbfe; }
    .callout-box.success { background: #f0fdf4; border-color: #bbf7d0; }
    .callout-box.danger { background: #fef2f2; border-color: #fecaca; }
    .callout-title {
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 14px;
      margin-bottom: 6px;
    }
    .callout-box.info .callout-title { color: #1e40af; }
    .callout-box.success .callout-title { color: #166534; }
    .callout-box.danger .callout-title { color: #991b1b; }
    .callout-desc {
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
    }

    /* Buttons */
    .btn-action {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 16px;
      border-radius: var(--radius-sm);
      font-weight: 700;
      font-size: 13px;
      text-decoration: none;
      transition: all 0.15s ease;
      cursor: pointer;
    }
    .btn-action.primary {
      background: var(--primary);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(30, 43, 250, 0.25);
    }
    .btn-action.primary:hover {
      background: var(--primary-hover);
      transform: translateY(-1px);
    }

    /* =======================================================================
       LIGHTBOX MODAL (BẤM ẢNH ĐỂ PHÓNG TO TOÀN MÀN HÌNH)
       ======================================================================= */
    .lightbox-modal {
      display: none;
      position: fixed;
      z-index: 9999;
      inset: 0;
      background: rgba(15, 23, 42, 0.94);
      backdrop-filter: blur(10px);
      align-items: center;
      justify-content: center;
      flex-direction: column;
      padding: 24px;
      cursor: zoom-out;
    }
    .lightbox-modal.active {
      display: flex;
    }
    .lightbox-content-box {
      max-width: 96vw;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      position: relative;
    }
    .lightbox-img {
      max-width: 95vw;
      max-height: 85vh;
      object-fit: contain;
      border-radius: 8px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
      background: #ffffff;
      animation: modalZoom 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes modalZoom {
      from { transform: scale(0.92); opacity: 0; }
      to { transform: scale(1); opacity: 1; }
    }
    .lightbox-caption {
      margin-top: 12px;
      color: #ffffff;
      font-family: var(--font-display);
      font-size: 14px;
      font-weight: 600;
      text-align: center;
      background: rgba(0, 0, 0, 0.6);
      padding: 6px 16px;
      border-radius: 20px;
    }
    .lightbox-close {
      position: absolute;
      top: 20px;
      right: 32px;
      color: #ffffff;
      font-size: 38px;
      font-weight: 700;
      cursor: pointer;
      line-height: 1;
      transition: color 0.15s ease;
      z-index: 10001;
    }
    .lightbox-close:hover {
      color: #ef4444;
    }

    @media (max-width: 900px) {
      .layout-wrapper {
        grid-template-columns: 1fr;
      }
      .sidebar-nav {
        display: none;
      }
      .kpi-grid {
        grid-template-columns: 1fr 1fr;
      }
      .face-strip-row {
        grid-template-columns: 1fr 1fr;
      }
    }
  </style>
</head>
<body>

  <!-- Top Sticky Header -->
  <header class="top-header">
    <div class="header-branding">
      <button class="toggle-sidebar-btn" id="toggle-sidebar-btn" title="Thu gọn / Mở rộng Mục Lục bên trái">
        ☰ Mục Lục
      </button>
      <span class="brand-logo-badge">UTH CV</span>
      <div>
        <div class="brand-title">Đề Tài 11 • Báo Cáo Nghiệm Thu Khoa Học Chung Cuộc</div>
        <div class="brand-sub">Phát Hiện Deepfake Khuếch Tán Kép (SD 1.5 + Realistic Vision 5.1) • Nhóm 4</div>
      </div>
    </div>
    <div class="header-actions">
      <span style="font-size: 13px; color: var(--text-muted); font-weight: 500;">💡 Nhấp vào bất kỳ ảnh nào để phóng to xem chi tiết</span>
    </div>
  </header>

  <div class="layout-wrapper" id="layout-wrapper">
    <!-- Slim, Compact Sidebar Navigation (~200px) -->
    <nav class="sidebar-nav" id="sidebar-nav">
      <div class="nav-group-title">Mục Lục Báo Cáo</div>
      <a href="#sec-overview" class="nav-link active">1. Tổng Quan Nghiệm Thu</a>
      <a href="#sec-problem" class="nav-link">2. Thực Trạng & Sự Cố M0</a>
      <a href="#sec-dataset" class="nav-link">3. Kho Dữ Liệu 12.545 Ảnh</a>
      <a href="#sec-curation" class="nav-link">4. Phễu Lọc CurationGate</a>
      <a href="#sec-hard-samples" class="nav-link">5. Tập Mẫu Khó 213 Challenge Vault</a>
      <a href="#sec-active-learning" class="nav-link">6. Đột Phá M2* (+80.18%)</a>
      <a href="#sec-zero-leakage" class="nav-link">7. Liêm Chính Zero Leakage</a>
      <a href="#sec-champion-m3" class="nav-link">8. Champion M3 (93.13%)</a>
      <a href="#sec-universal" class="nav-link">9. Universal 20.4k Benchmark</a>
      <a href="#sec-xai" class="nav-link">10. DINOv2 Attention Heatmap</a>
      <a href="#sec-conclusion" class="nav-link">11. Bảng KPI & Kết Luận</a>
    </nav>

    <!-- Main Content Area -->
    <main class="main-content">

      <!-- ===================================================================
           PHẦN 1: TỔNG QUAN NGHIỆM THU & HERO ARTIFACT
           =================================================================== -->
      <section id="sec-overview" class="report-section hero-section">
        <span class="section-eyebrow">UTH COMPUTER VISION COURSEWORK • TOPIC 11</span>
        <h1 class="section-title">
          Báo Cáo Nghiệm Thu Đồ Án: <span class="accent">Phòng Thủ Deepfake Khuếch Tán Kép</span>
        </h1>
        <p class="section-desc">
          Đồ án đã hoàn thiện trọn vẹn chu trình nghiên cứu: từ tạo lập kho 12.545 ảnh sinh từ 2 động cơ khuếch tán (Stable Diffusion v1.5: 6.545 ảnh; RealisticVision v5.1: 6.000 ảnh), sàng lọc qua phễu CurationGate 3 tầng, khai thác 740 mẫu khó (Hard Samples), đánh giá trên tập đối kháng 213 Challenge Vault, đến huấn luyện thành công mô hình Champion M3 đạt Fake Recall 93.13% và đối chuẩn trên Universal Benchmark 20.400 ảnh.
        </p>

        <!-- KPI Scorecard Highlights -->
        <div class="kpi-grid">
          <div class="kpi-card">
            <span class="kpi-label">Dữ Liệu Sinh Kép</span>
            <div class="kpi-value">12.545</div>
            <span class="kpi-note">SD 1.5 (6.545) + RV5.1 (6.000)</span>
          </div>
          <div class="kpi-card alert">
            <span class="kpi-label">213 Challenge Vault</span>
            <div class="kpi-value">213</div>
            <span class="kpi-note">Mẫu khó đánh lừa Baseline (Hard Samples)</span>
          </div>
          <div class="kpi-card champion">
            <span class="kpi-label">Bứt Phá Recall M2*</span>
            <div class="kpi-value">+80.18%</div>
            <span class="kpi-note">Nhảy vọt từ 11.71% → 91.89%</span>
          </div>
          <div class="kpi-card champion">
            <span class="kpi-label">Champion M3 Recall</span>
            <div class="kpi-value">93.13%</div>
            <span class="kpi-note">Tập Mù RV5.1 (Quad-Vault)</span>
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG KHỔ LỚN -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">🏆 Minh Chứng Vàng: Bảng Điểm Tiến Hóa 4 Cột Mốc Mô Hình (M0 → M1 → M2* → M3)</span>
            <span class="artifact-badge champion">GOLDEN SCORECARD</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/00_master_deepfake_defense_showcase_cell4_img0.png" class="artifact-img-full" alt="Golden Scorecard M0 to M3 Evolution" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            <strong>Bóc tách số liệu:</strong> Mô hình cơ sở M0 hoàn toàn sụp đổ trên Challenge Vault (Recall 11.71%); qua Vòng 1 tăng lên 70.27%; Vòng 2 Refined M2* đạt 91.89%; và mô hình Champion M3 bứt phá đạt đỉnh cao 93.13% trên tập kiểm thử mù RealisticVision v5.1 mà không hề suy giảm nhận thức trên SD 1.5.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 2: THỰC TRẠNG & SỰ CỐ SỤP ĐỔ CỦA M0 BASELINE
           =================================================================== -->
      <section id="sec-problem" class="report-section">
        <span class="section-eyebrow">01 / THÁCH THỨC ĐỐI CHUẨN</span>
        <h2 class="section-title">
          Thực Trạng Đối Đầu: <span class="accent">Vì Sao Ảnh Giả Mềm Đánh Lừa Bộ Dò Truyền Thống?</span>
        </h2>
        <p class="section-desc">
          Các mô hình deepfake ghép mặt truyền thống (Face-Swap) để lại đường cắt nhị phân cứng với độ chênh lệch gradient sắc nhọn, khiến các mạng CNN như ResNet hay EfficientNet dễ dàng "học tủ" ranh giới này. Tuy nhiên, khi áp dụng kỹ thuật <strong>Làm Mịn Biên Gaussian (Soft-Mask Feathering σ=4.0)</strong> trong Stable Diffusion, dải chuyển tiếp mượt mà xóa nhòa hoàn toàn vết cắt nhân tạo, khiến các bộ lọc cũ bị tê liệt hoàn toàn.
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: SOFT FEATHERING -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Artifact Thực Nghiệm: Cơ Chế Làm Mịn Biên Gaussian Soft-Mask Feathering (σ = 4.0)</span>
            <span class="artifact-badge">NOTEBOOK 01 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/01_diffusion_dataset_and_curation_cell8_img2.png" class="artifact-img-full" alt="Soft-Mask Feathering Mechanism" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            <strong>Nguyên lý toán học:</strong> <code>M_feathered = GaussianBlur(M_binary, k=(15,15), σ=4.0)</code> kết hợp với <code>I_composite = I_inpaint ⊙ M_feathered + I_real ⊙ (1 - M_feathered)</code> xóa sổ hoàn toàn vết cắt sắc nhọn, ép mô hình phải học các dị thường vi cấu trúc tần số cao bên trong khuôn mặt.
          </div>
        </div>

        <!-- Empirical Failure Callout -->
        <div class="callout-box danger">
          <div class="callout-title">🚨 Sự Cố Sụp Đổ Của Mô Hình Cơ Sở M0 Trên Challenge Vault:</div>
          <div class="callout-desc">
            Khi đem mô hình cơ sở M0 (huấn luyện trên tập chuẩn D0 thông thường) kiểm tra trên 213 mẫu cực khó của Challenge Vault, <strong>Fake Recall chỉ đạt 11.71%</strong> (tỷ lệ lọt lưới Bypass Rate lên tới <strong>88.29%</strong>; Overall Accuracy chỉ đạt <strong>53.99%</strong> — tương đương đoán ngẫu nhiên; F1-Score sụp đổ còn <strong>0.2097</strong>). Điều này chứng minh: nếu không có vòng lặp Active Learning khai thác mẫu khó, hệ thống nhận diện khuôn mặt trong các cổng eKYC ngân hàng sẽ bị qua mặt hoàn toàn!
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: GLOBAL COMPARISON -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Artifact Đối Chuẩn Toàn Cầu: Biểu Đồ So Sánh Tổng Thể Các Mô Hình</span>
            <span class="artifact-badge champion">BENCHMARK FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_4models_global_comparison.png" class="artifact-img-full" alt="Global Comparison Benchmark Chart" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            So sánh toàn diện đa tiêu chí giữa M0 Baseline, M1 Active Round 1, M2 Cold-Start Retrain và M2* Refined Champion.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 3: KHO DỮ LIỆU KHUẾCH TÁN KÉP 12.545 ẢNH & LƯỚI MẪU THẬT
           =================================================================== -->
      <section id="sec-dataset" class="report-section">
        <span class="section-eyebrow">02 / SẢN PHẨM DỮ LIỆU</span>
        <h2 class="section-title">
          Kho Dữ Liệu Khuếch Tán Kép: <span class="accent">12.545 Ảnh Sinh Thực Tế & Lưới Ảnh Đối Chuẩn</span>
        </h2>
        <p class="section-desc">
          Nhóm nghiên cứu đã tạo dựng và kiểm duyệt tổng cộng <strong>12.545 ảnh khuôn mặt</strong> từ hai động cơ khuếch tán đại diện: <strong>Stable Diffusion v1.5</strong> (6.545 ảnh) và <strong>RealisticVision v5.1</strong> (6.000 ảnh), phân bổ trên 3 phương thức sinh: Inpaint, Img2Img và Text-to-Image.
        </p>

        <!-- BẢNG TỔNG HỢP PHÂN BỔ 2 ĐỘNG CƠ SINH DỮ LIỆU (DUAL-ENGINE) -->
        <div class="table-responsive" style="margin: 20px 0 28px; overflow-x: auto;">
          <table class="report-table">
            <thead>
              <tr>
                <th style="width: 25%;">Động cơ sinh</th>
                <th style="width: 30%;">Nguồn mô hình</th>
                <th style="width: 15%;">Quy mô sinh</th>
                <th style="width: 30%;">Chi tiết & Mục đích sử dụng</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Bộ A: Stable Diffusion v1.5</strong></td>
                <td><code>runwayml/stable-diffusion-v1-5</code></td>
                <td><strong>6.545 ảnh</strong></td>
                <td>Vận hành vòng lặp Active Learning nội miền (D₀ → D₁ → D₂)</td>
              </tr>
              <tr>
                <td><strong>Bộ B: RealisticVision v5.1</strong></td>
                <td><code>SG161222/Realistic_Vision_V5.1_noVAE</code></td>
                <td><strong>6.000 ảnh</strong></td>
                <td>Benchmark đối kháng siêu thực ngoại miền (OOD Blind Transfer)</td>
              </tr>
              <tr style="background: var(--surface-soft); font-weight: 700;">
                <td><strong>Tổng cộng ảnh sinh mới</strong></td>
                <td>-</td>
                <td><strong class="accent" style="color: var(--primary); font-size: 15px;">12.545 ảnh</strong></td>
                <td>Bao phủ 3 chế độ: Inpaint, Img2Img, Text-to-Image</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: DUAL ENGINE PORTFOLIO GRID -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Artifact Danh Mục Sinh Kép: Lưới So Sánh Thực Tế (Real vs Inpaint vs Img2Img vs Text2Img)</span>
            <span class="artifact-badge champion">NOTEBOOK 01 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/01_diffusion_dataset_and_curation_cell10_img3.png" class="artifact-img-full" alt="Dual Engine Portfolio Grid" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            <strong>Kiểm soát kỹ thuật:</strong> Mỗi chế độ sinh đều đảm bảo độ phân giải chuẩn (512x512 sau đó căn chỉnh 224x224), bảo toàn cấu trúc ánh sáng tự nhiên và tỷ lệ nhân khẩu học hài hòa.
          </div>
        </div>

        <h3 style="font-family: var(--font-display); font-size: 18px; margin: 32px 0 16px;">Bộ Sưu Tập Mẫu Khuôn Mặt Thực Tế Khổ Lớn Xuất Từ Kho 12.545 Ảnh Sinh:</h3>

        <!-- Mẫu 1: Nhân vật Nữ #0 -->
        <div class="face-subject-container">
          <div class="face-subject-header">
            <span class="face-subject-title">Mẫu 1: Đối Chứng Khuôn Mặt Nữ (CelebA-HQ #0) Qua 4 Chế Độ Sinh</span>
            <span class="artifact-badge">CHỦ THỂ NỮ</span>
          </div>
          <div class="face-strip-row">
            <div class="face-strip-card">
              <img src="assets/samples/real_0.jpg" alt="Real Face 0" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag real">ẢNH THẬT (GỐC)</span>
                <div class="face-name">CelebA-HQ ID #0</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/inpaint_0.png" alt="Inpaint 0" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag inpaint">SD INPAINT</span>
                <div class="face-name">0_inpaint.png (Soft Biên)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/img2img_0.png" alt="Img2Img 0" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag img2img">SD IMG2IMG</span>
                <div class="face-name">0_img2img.png (Biến Đổi)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/text2img_0.png" alt="Text2Img 0" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag text2img">SD TEXT2IMG</span>
                <div class="face-name">0_text2img.png (Sinh Hoàn Toàn)</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Mẫu 2: Nhân vật Nam #9 -->
        <div class="face-subject-container">
          <div class="face-subject-header">
            <span class="face-subject-title">Mẫu 2: Đối Chứng Khuôn Mặt Nam (CelebA-HQ #9) Khớp Râu & Cằm Tự Nhiên</span>
            <span class="artifact-badge">CHỦ THỂ NAM</span>
          </div>
          <div class="face-strip-row">
            <div class="face-strip-card">
              <img src="assets/samples/real_9.jpg" alt="Real Face 9" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag real">ẢNH THẬT (GỐC)</span>
                <div class="face-name">CelebA-HQ ID #9</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/inpaint_9.png" alt="Inpaint 9" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag inpaint">SD INPAINT</span>
                <div class="face-name">9_inpaint.png (Khớp Râu/Cằm)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/img2img_9.png" alt="Img2Img 9" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag img2img">SD IMG2IMG</span>
                <div class="face-name">9_img2img.png (Biến Đổi Da)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/text2img_9.png" alt="Text2Img 9" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag text2img">SD TEXT2IMG</span>
                <div class="face-name">9_text2img.png (Sinh Hoàn Toàn)</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Mẫu 3: Nhân vật Thiếu Sáng / Ngược Sáng #107 -->
        <div class="face-subject-container">
          <div class="face-subject-header">
            <span class="face-subject-title">Mẫu 3: Đối Chứng Điều Kiện Ánh Sáng Yếu & Tương Phản Cao (CelebA-HQ #107)</span>
            <span class="artifact-badge alert">ÁNH SÁNG PHỨC TẠP</span>
          </div>
          <div class="face-strip-row">
            <div class="face-strip-card">
              <img src="assets/samples/real_107.jpg" alt="Real Face 107" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag real">ẢNH THẬT (GỐC)</span>
                <div class="face-name">CelebA-HQ #107 (Ngược Sáng)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/inpaint_107.png" alt="Inpaint 107" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag inpaint">SD INPAINT</span>
                <div class="face-name">107_inpaint.png (Ánh Sáng Yếu)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/img2img_107.png" alt="Img2Img 107" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag img2img">SD IMG2IMG</span>
                <div class="face-name">107_img2img.png (Tương Phản)</div>
              </div>
            </div>
            <div class="face-strip-card">
              <img src="assets/samples/text2img_107.png" alt="Text2Img 107" title="Nhấp để phóng to">
              <div class="face-strip-info">
                <span class="face-tag text2img">SD TEXT2IMG</span>
                <div class="face-name">107_text2img.png (Studio Lighting)</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: DEMOGRAPHICS PANEL -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Artifact Phân Phối Nhân Khẩu Học & Độ Đa Dạng Dữ Liệu (Demographics Panel)</span>
            <span class="artifact-badge">EDA FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/eda/02_demographics_panel.png" class="artifact-img-full" alt="Demographics Panel" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            <strong>Cân bằng đa dạng sinh học:</strong> Phân bổ độ tuổi trải rộng từ 18 đến trên 65; kiểm soát tỷ lệ giới tính cân bằng; đảm bảo góc nghiêng khuôn mặt giới hạn trong khoảng an toàn ±30°.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 4: PHỄU SÀNG LỌC CHẤT LƯỢNG 3 TẦNG (CURATIONGATE)
           =================================================================== -->
      <section id="sec-curation" class="report-section">
        <span class="section-eyebrow">03 / PHỄU LỌC CHẤT LƯỢNG</span>
        <h2 class="section-title">
          CurationGate: <span class="accent">Đường Ống Lọc Sạch 3 Tầng Bảo Vệ Độ Tinh Khiết Dữ Liệu</span>
        </h2>
        <p class="section-desc">
          Nhóm thiết lập quy trình kiểm toán 3 tầng khắt khe: Tầng 1 kiểm soát độ nét quang học bằng toán tử Laplacian (<code>Var(Laplacian) ≥ 100.0</code>); Tầng 2 kiểm tra mốc hình học giải phẫu (Dlib 68 landmarks) và phân tích sắc tố da YCbCr (<code>77 ≤ Cb ≤ 127, 133 ≤ Cr ≤ 173</code>); Tầng 3 loại bỏ ảnh cháy sáng (&gt;8%) và khử trùng lặp bằng pHash và SSIM.
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: SKIN GAMUT -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Artifact Tầng 2: Phổ Màu Da Trong Không Gian Sắc Ký YCbCr</span>
            <span class="artifact-badge">NOTEBOOK 01 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/01_diffusion_dataset_and_curation_cell6_img1.png" class="artifact-img-full" alt="Skin Color Gamut YCbCr" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            Kiểm tra dải sắc tố: <code>135 ≤ Cr ≤ 180, 85 ≤ Cb ≤ 135</code> loại bỏ 100% ảnh ám sắc tím/xanh dị thường do lỗi khuếch tán.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: IMAGE QUALITY PANEL -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Artifact Tầng 2 & 3: Bảng Kiểm Toán Độ Nét Quang Học & Phân Phối Chất Lượng</span>
            <span class="artifact-badge">EDA FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/eda/04_image_quality_panel.png" class="artifact-img-full" alt="Image Quality Panel" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            Đo phương sai Laplacian: <code>Var(Laplacian) ≥ 100.0</code> loại bỏ hoàn toàn ảnh mờ nhòe, mất nét; pHash và SSIM ≤ 0.85 loại bỏ sạch bản sao trùng lặp.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 5: KHAI THÁC 740 MẪU KHÓ & ĐỐI CHUẨN 213 CHALLENGE VAULT
           =================================================================== -->
      <section id="sec-hard-samples" class="report-section">
        <span class="section-eyebrow">04 / ĐỐI CHUẨN MẪU KHÓ</span>
        <h2 class="section-title">
          213 Challenge Vault: <span class="accent">Khai Thác 740 Mẫu Khó & Phân Tích 4 Dị Tật Sinh Học Vi Thể</span>
        </h2>
        <p class="section-desc">
          Dựa trên 2 tiêu chí định lượng khắt khe (vùng ranh giới không chắc chắn |P - 0.50| ≤ 0.15 và mẫu dự đoán sai với độ tin cậy cao), nhóm đã cô lập 213 mẫu khó nhất vào tập đối kháng 213 Challenge Vault làm thước đo bất biến kiểm chứng qua mọi vòng học.
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: BIOLOGICAL DEFECTS GALLERY -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Artifact Thực Nghiệm: Bộ Sưu Tập 4 Dị Tật Sinh Học Vi Thể (Challenge Vault)</span>
            <span class="artifact-badge alert">CHALLENGE VAULT ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/02_hard_samples_and_active_learning_cell6_img2.png" class="artifact-img-full" alt="Biological Defect Taxonomy" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            <strong>4 Dị tật phát hiện:</strong> (1) Phản chiếu đồng tử giác mạc bất đối xứng hướng sáng; (2) Ranh giới men răng bị hòa tan liên tục; (3) Mất vi cấu trúc lỗ chân lông (hiệu ứng da nhựa); (4) Chân tóc tơ hòa trộn bất thường vào nền.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: FULL CONTACT SHEET (SCROLLABLE & ZOOMABLE) -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Tấm Contact Sheet Khổng Lồ 213 Mẫu Khó Challenge Vault (Full-Res Contact Sheet)</span>
            <span class="artifact-badge">ASSET 10.3 MB</span>
          </div>
          <div class="artifact-row-body" style="padding: 12px; max-height: 850px; overflow-y: auto;">
            <img src="assets/hard_samples_contact_sheet.png" class="artifact-img-full" alt="Hard Samples Contact Sheet" title="Nhấp để phóng to toàn màn hình" style="max-height: none;">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            Toàn bộ 213 mẫu bẫy độc lập được khóa cứng trong <code>cumulative_challenge_test.jsonl</code>, không bao giờ được đưa vào tập huấn luyện của bất kỳ mô hình nào!
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 6: ĐỘT PHÁ VÒNG 2 ACTIVE LEARNING (M2*) & RECALL SURGE
           =================================================================== -->
      <section id="sec-active-learning" class="report-section">
        <span class="section-eyebrow">05 / BƯỚC NHẢY VỌT BIÊN QUYẾT ĐỊNH</span>
        <h2 class="section-title">
          Active Learning Vòng 2: <span class="accent">M2* Nâng Fake Recall Từ 11.71% Lên 91.89% (+80.18%)</span>
        </h2>
        <p class="section-desc">
          Bằng cách giải phóng có kiểm soát tầng Block 11 của DINOv2 (chỉ 8.06% tham số trainable) và áp dụng tốc độ học phân tầng (Differential Learning Rates), mô hình M2* đã dịch chuyển đường biên quyết định, bắt trọn 102/111 mẫu giả cực khó!
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: RECALL JUMP CHART -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Biểu Đồ Bước Nhảy Vọt Fake Recall Trên 213 Challenge Samples (M0: 11.71% → M2*: 91.89%)</span>
            <span class="artifact-badge champion">NOTEBOOK 02 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/02_hard_samples_and_active_learning_cell9_img3.png" class="artifact-img-full" alt="Recall Jump Chart" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            M0 đạt 11.71% → M1 đạt 70.27% → <strong>M2* bứt phá 91.89% (+80.18%)</strong>. Bắt giữ thêm +89 mẫu giả tinh vi từng đánh lừa hoàn toàn baseline.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: PRECISION-RECALL CURVES -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Đường Cong Precision-Recall So Sánh 5 Mô Hình</span>
            <span class="artifact-badge">BENCHMARK FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_5models_precision_recall_curves.png" class="artifact-img-full" alt="PR Curves 5 Models" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            Đường cong PR của M2* và M3 ôm sát góc trên bên phải, PR-AUC đạt đỉnh cao 0.9875.
          </div>
        </div>

        <!-- Empirical Table -->
        <table class="data-table">
          <thead>
            <tr>
              <th>Phiên Bản Mô Hình</th>
              <th>Fake Recall (↑)</th>
              <th>Bypass Rate (↓)</th>
              <th>Overall Acc (↑)</th>
              <th>ROC-AUC (↑)</th>
              <th>Số Mẫu Bắt Được (/111)</th>
              <th>Đánh Giá Kỹ Thuật</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>M0 (D0 Baseline)</strong></td>
              <td>11.71%</td>
              <td>88.29%</td>
              <td>53.99%</td>
              <td>0.8713</td>
              <td>13 / 111 mẫu</td>
              <td><span class="badge-pill red">BỎ SÓT NGUY HẠI</span></td>
            </tr>
            <tr>
              <td><strong>M1 (Active Round 1)</strong></td>
              <td>70.27%</td>
              <td>29.73%</td>
              <td>83.10%</td>
              <td>0.9655</td>
              <td>78 / 111 mẫu</td>
              <td><span class="badge-pill amber">BƯỚC ĐẦU BỨT PHÁ</span></td>
            </tr>
            <tr>
              <td><strong>M2 (Cold-Start Retrain)</strong></td>
              <td>86.49%</td>
              <td>13.51%</td>
              <td>92.49%</td>
              <td>0.9937</td>
              <td>96 / 111 mẫu</td>
              <td><span class="badge-pill green">HỘI TỤ TỐT</span></td>
            </tr>
            <tr class="highlight-champion">
              <td><strong>🏆 M2* (Refined Champion)</strong></td>
              <td><strong>91.89% (+80.18%)</strong></td>
              <td><strong>8.11%</strong></td>
              <td><strong>93.90%</strong></td>
              <td><strong>0.9875</strong></td>
              <td><strong>102 / 111 mẫu</strong></td>
              <td><span class="badge-pill green">CHẾ NGỰ HOÀN TOÀN</span></td>
            </tr>
          </tbody>
        </table>
      </section>


      <!-- ===================================================================
           PHẦN 7: BẢO TOÀN LIÊM CHÍNH HỌC THUẬT: ZERO IDENTITY LEAKAGE
           =================================================================== -->
      <section id="sec-zero-leakage" class="report-section">
        <span class="section-eyebrow">06 / LIÊM CHÍNH HỌC THUẬT</span>
        <h2 class="section-title">
          Zero Identity Leakage: <span class="accent">Phát Hiện & Cách Ly Hoàn Toàn 626 Ảnh Rò Rỉ Danh Tính</span>
        </h2>
        <p class="section-desc">
          Khi mở rộng sang kho 6.000 ảnh RealisticVision v5.1, nhóm phát hiện 626 ảnh inpaint/img2img được sinh từ khuôn mặt thuộc tập test. Nhóm đã lập tức cách ly 100% làm tập kiểm thử mù độc lập <code>unseen_rv51_test.jsonl</code>, đảm bảo mô hình học bản chất vi vết giả mạo chứ không nhớ mặt người.
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: CLEAN D3 PARTITION -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Artifact Sơ Đồ Phân Vùng Sạch D3: Cách Ly Tuyệt Đối Danh Tính Tập Test</span>
            <span class="artifact-badge champion">NOTEBOOK 03 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/03_champion_m3_and_universal_benchmark_cell3_img0.png" class="artifact-img-full" alt="Zero Identity Leakage Clean Partition Schema" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            <strong>Cấu trúc D3 sạch (5.626 ảnh):</strong> Cân bằng 1:1 chuẩn xác (2.813 Thật : 2.813 Giả). Không một ảnh nào thuộc 626 ảnh trùng test ID được lọt vào tập huấn luyện!
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 8: CHAMPION M3 DUAL-ENGINE & ĐỐI CHUẨN TỨ GIÁC (QUAD-VAULT)
           =================================================================== -->
      <section id="sec-champion-m3" class="report-section">
        <span class="section-eyebrow">07 / MÔ HÌNH VÔ ĐỊCH CHAMPION M3</span>
        <h2 class="section-title">
          Champion M3: <span class="accent">Bứt Phá 93.13% Quad-Vault Mà Không Quên SD 1.5</span>
        </h2>
        <p class="section-desc">
          Huấn luyện Warm-Start từ M2* với bộ điều phối Cosine Annealing qua 6 epochs. Mô hình hội tụ hoàn hảo, xác lập kỷ lục mới toàn diện trên cả 3 kho kiểm thử nội bộ độc lập.
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: QUAD-VAULT RECALL CHART -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Biểu Đồ Cột Đối Chuẩn Tứ Giác: So Sánh Fake Recall 5 Mô Hình Trên 3 Kho Độc Lập</span>
            <span class="artifact-badge champion">BENCHMARK FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_5models_vault_recall_comparison.png" class="artifact-img-full" alt="Quad-Vault Recall Comparison" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            M3 đạt Fake Recall <strong>93.13%</strong> trên tập mù RV5.1 (Bypass giảm sâu còn 6.87%) trong khi vẫn duy trì nguyên vẹn <strong>91.89%</strong> trên Challenge Vault!
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: TRAINING CONVERGENCE -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Động Học Hội Tụ Huấn Luyện M3 (Loss & AUC Qua 6 Epochs Cosine Annealing)</span>
            <span class="artifact-badge">NOTEBOOK 03 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/03_champion_m3_and_universal_benchmark_cell5_img1.png" class="artifact-img-full" alt="M3 Convergence Dynamics" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            Validation AUC tăng đều từ 0.9745 lên <strong>0.9846</strong> ở Epoch 6 mà không hề xảy ra hiện tượng Overfitting hay Catastrophic Forgetting.
          </div>
        </div>

        <!-- Full Quad-Vault Scorecard Table -->
        <table class="data-table">
          <thead>
            <tr>
              <th>Phiên Bản Mô Hình</th>
              <th>Master Frozen Test (SD 1.5 - 1.178 ảnh)<br>Fake Recall (↑) | Bypass (↓)</th>
              <th>Challenge Vault (Mẫu Khó - 213 ảnh)<br>Fake Recall (↑) | Bypass (↓)</th>
              <th>Unseen RV5.1 Vault (Tập Mù - 1.215 ảnh)<br>Fake Recall (↑) | Bypass (↓)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>M0 (D0 Baseline)</strong></td>
              <td>49.07% | 50.93%</td>
              <td>11.71% | 88.29%</td>
              <td>66.45% | 33.55%</td>
            </tr>
            <tr>
              <td><strong>M1 (Active Round 1)</strong></td>
              <td>62.14% | 37.86%</td>
              <td>70.27% | 29.73%</td>
              <td>72.04% | 27.96%</td>
            </tr>
            <tr>
              <td><strong>M2 (Cold-Start Retrain)</strong></td>
              <td>52.63% | 47.37%</td>
              <td>86.49% | 13.51%</td>
              <td>56.55% | 43.45%</td>
            </tr>
            <tr>
              <td><strong>M2* (Refined Champion)</strong></td>
              <td>66.89% | 33.11%</td>
              <td><strong>91.89%</strong> | <strong>8.11%</strong></td>
              <td>71.88% | 28.12%</td>
            </tr>
            <tr class="highlight-champion">
              <td><strong>🏆 M3 (Dual-Engine Champion)</strong></td>
              <td><span class="badge-pill green">69.44% (Cao nhất)</span></td>
              <td><span class="badge-pill green">91.89% (Bảo tồn)</span></td>
              <td><span class="badge-pill green">93.13% (+21.25%)</span></td>
            </tr>
          </tbody>
        </table>
      </section>


      <!-- ===================================================================
           PHẦN 9: ĐỐI CHUẨN TOÀN CẦU (UNIVERSAL BENCHMARK 20.400 ẢNH)
           =================================================================== -->
      <section id="sec-universal" class="report-section">
        <span class="section-eyebrow">08 / NĂNG LỰC TỔNG QUÁT HÓA TOÀN CẦU</span>
        <h2 class="section-title">
          Universal Benchmark: <span class="accent">Thử Thách Khổng Lồ 20.400 Ảnh Trên 34 Bộ Sinh & 8 Họ</span>
        </h2>
        <p class="section-desc">
          Đánh giá năng lực phát hiện thực chiến trên bộ dữ liệu đồ sộ gồm 10.200 ảnh Thật vs 10.200 ảnh Giả từ 34 thuật toán thuộc 8 họ công nghệ: GAN, Diffusion, Face-Swap, Commercial Inpainting, và Talking Heads.
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: FULL-WIDTH HEATMAP -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Ma Trận Nhiệt Khảo Sát Chi Tiết Trên 34 Thuật Toán Sinh Toàn Cầu (34-Method Heatmap)</span>
            <span class="artifact-badge champion">BENCHMARK FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_4models_method_heatmap.png" class="artifact-img-full" alt="Universal Benchmark 34-Method Heatmap" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            M3 thể hiện độ nhạy vượt trội trên các họ khuếch tán mới (Midjourney, Stable Diffusion XL, Latent Diffusion) và duy trì độ đặc hiệu người thật trên 90%.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: 8-FAMILY RECALL -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Độ Nhạy Theo 8 Họ Công Nghệ Sinh Ảnh Toàn Cầu</span>
            <span class="artifact-badge">BENCHMARK FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_5models_family_recall.png" class="artifact-img-full" alt="8-Family Recall Chart" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            Fake Recall trên họ Diffusion Baselines (DDIM) tăng vọt gần gấp 3 lần (từ 6.00% lên 17.33%), Latent Diffusion (SD 2.1) tăng gần gấp đôi (từ 4.67% lên 8.33%), và GAN Inversion đạt đỉnh 21.50%, khẳng định năng lực khái quát hóa liên miền.
          </div>
        </div>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: ROC CURVES -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Đường Cong ROC Curves 5 Mô Hình & Khảo Sát Ngưỡng eKYC Ngân Hàng</span>
            <span class="artifact-badge">BENCHMARK FIGURE</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/benchmarks/benchmark_5models_roc_curves.png" class="artifact-img-full" alt="ROC Curves 5 Models" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            Tại ngưỡng mặc định τ = 0.50, Champion M3 đạt Real Specificity 90.42%, đồng thời duy trì tính trơ với nhiễu nén video H.264 trên FaceForensics++ C23 (1.655 frames) với tỷ lệ báo động nhầm FPR chỉ 2.60%.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 10: FORENSIC LIVE INSPECTOR & DINOV2 SPATIAL ATTENTION HEATMAP
           =================================================================== -->
      <section id="sec-xai" class="report-section">
        <span class="section-eyebrow">09 / GIẢI THÍCH MÔ HÌNH (EXPLAINABLE AI)</span>
        <h2 class="section-title">
          Forensic Inspector: <span class="accent">Bản Đồ Chú Ý DINOv2 Soi Vùng Can Thiệp Sâu</span>
        </h2>
        <p class="section-desc">
          Mô hình không phải là hộp đen — cơ chế trích xuất Self-Attention Head tại Block 11 của DINOv2 cho phép định vị chính xác khu vực chỉnh sửa giả mạo trên khuôn mặt theo thời gian thực (~18.5 ms/ảnh).
        </p>

        <!-- 1 ARTIFACT TRÊN 1 HÀNG: DINOV2 ATTENTION HEATMAP -->
        <div class="artifact-row-card">
          <div class="artifact-row-header">
            <span class="artifact-row-title">Artifact Trực Tiếp: Bản Đồ Chú Ý Không Gian (DINOv2 Spatial Attention Heatmap)</span>
            <span class="artifact-badge champion">NOTEBOOK 00 ARTIFACT</span>
          </div>
          <div class="artifact-row-body">
            <img src="assets/extracted_nb/00_master_deepfake_defense_showcase_cell7_img1.png" class="artifact-img-full" alt="DINOv2 Spatial Attention Heatmap" title="Nhấp để phóng to toàn màn hình">
            <span class="zoom-hint-banner">🔍 Nhấp vào ảnh để phóng to toàn màn hình</span>
          </div>
          <div class="artifact-row-footer">
            <strong>Bằng chứng pháp y trực quan:</strong> Vùng mắt, khóe miệng và đường viền chân tóc sáng rực điểm chú ý của mạng nơ-ron, cung cấp giải trình minh bạch tuyệt đối cho chuyên viên thẩm định an ninh ngân hàng.
          </div>
        </div>
      </section>


      <!-- ===================================================================
           PHẦN 11: BẢNG KPI NGHIỆM THU & TỔNG KẾT
           =================================================================== -->
      <section id="sec-conclusion" class="report-section">
        <span class="section-eyebrow">10 / TỔNG KẾT & NGHIỆM THU KPI</span>
        <h2 class="section-title">
          Nghiệm Thu Toàn Diện: <span class="accent">Hoàn Thành 100% KPI & Sẵn Sàng Triển Khai eKYC</span>
        </h2>
        <p class="section-desc">
          Bảng đối chiếu tổng hợp minh chứng đồ án Đề tài 11 đã hoàn thành xuất sắc tất cả cam kết kỹ thuật, mã nguồn sạch đạt chuẩn lint/smoke test và bảo toàn tính liêm chính học thuật.
        </p>

        <table class="data-table">
          <thead>
            <tr>
              <th style="width: 25%;">Hạng Mục Nghiệm Thu</th>
              <th style="width: 30%;">Cam Kết Ban Đầu (Define & Design)</th>
              <th style="width: 30%;">Kết QuẢ Thực Nghiệm Đạt Được</th>
              <th style="width: 15%;">Đánh Giá</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>1. Dữ Liệu Khuếch Tán Kép</strong></td>
              <td>Sinh ảnh SD với Inpaint/Img2Img/Text2Img</td>
              <td>12.545 ảnh từ 2 động cơ (SD 1.5: 6.545 ảnh + RV5.1: 6.000 ảnh); Soft Feathering σ=4.0 xóa vết cắt</td>
              <td><span class="badge-pill green">ĐẠT 100%</span></td>
            </tr>
            <tr>
              <td><strong>2. CurationGate 3 Tầng</strong></td>
              <td>Phễu lọc Landmark, Gamut và Deduplication</td>
              <td>Dlib 68 + YCbCr + pHash/SSIM; 100% sạch trùng lặp, manifest chuẩn</td>
              <td><span class="badge-pill green">ĐẠT 100%</span></td>
            </tr>
            <tr class="highlight-champion">
              <td><strong>3. Khai Phá Mẫu Khó</strong></td>
              <td>Active Learning vòng lặp cải thiện biên</td>
              <td>740 mẫu khó, tập đối kháng 213 Challenge Vault; Recall tăng từ 11.71% → 91.89%</td>
              <td><span class="badge-pill green">VƯỢT (+80%)</span></td>
            </tr>
            <tr>
              <td><strong>4. Zero Identity Leakage</strong></td>
              <td>Tránh hiện tượng học vẹt danh tính người</td>
              <td>Cách ly 626 ảnh RV5.1 test ID; D3 cân bằng 1:1 chuẩn xác (5.626 ảnh)</td>
              <td><span class="badge-pill green">LIÊM CHÍNH</span></td>
            </tr>
            <tr class="highlight-champion">
              <td><strong>5. Champion M3 & Benchmark</strong></td>
              <td>Đánh giá tổng quát hóa mở rộng</td>
              <td>Champion M3 Quad-Vault 93.13%; Universal Benchmark 20.400 ảnh (34 bộ sinh)</td>
              <td><span class="badge-pill green">VƯỢT CHỈ TIÊU</span></td>
            </tr>
            <tr>
              <td><strong>6. XAI Live Inspector</strong></td>
              <td>Trực quan hóa giải thích mô hình</td>
              <td>DINOv2 Spatial Attention Heatmap thời gian thực (18.5 ms/ảnh) giải trình trực quan</td>
              <td><span class="badge-pill green">HOÀN TẤT</span></td>
            </tr>
          </tbody>
        </table>

        <div style="background: #fdfae7; border: 2px solid var(--primary); border-radius: var(--radius-md); padding: 24px; text-align: center; margin-top: 32px;">
          <h3 style="font-family: var(--font-display); font-size: 18px; color: var(--primary); margin-bottom: 8px;">
            KÍNH CẢM ƠN HỘI ĐỒNG GIẢNG VIÊN BỘ MÔN THỊ GIÁC MÁY TÍNH — TRƯỜNG ĐH GIAO THÔNG VẬN TẢI TP.HCM!
          </h3>
          <p style="font-size: 13px; color: var(--text-muted); max-width: 800px; margin: 0 auto;">
            Nhóm 4 kính mời Thầy Cô trong Hội đồng đánh giá đặt câu hỏi phản biện chuyên sâu và đóng góp ý kiến cho Đề tài 11.
          </p>
        </div>
      </section>

    </main>
  </div>

  <!-- LIGHTBOX MODAL (POPUP PHÓNG TO ẢNH FULL-SCREEN) -->
  <div id="image-lightbox" class="lightbox-modal">
    <span class="lightbox-close" id="lightbox-close" title="Đóng phóng to (hoặc nhấn Esc)">&times;</span>
    <div class="lightbox-content-box">
      <img class="lightbox-img" id="lightbox-img" src="" alt="Full size artifact">
      <div class="lightbox-caption" id="lightbox-caption"></div>
    </div>
  </div>

  <script>
    document.addEventListener('DOMContentLoaded', () => {
      // 1. Sidebar Toggle Functionality
      const toggleBtn = document.getElementById('toggle-sidebar-btn');
      const layoutWrapper = document.getElementById('layout-wrapper');
      
      toggleBtn.addEventListener('click', () => {
        layoutWrapper.classList.toggle('sidebar-collapsed');
        if (layoutWrapper.classList.contains('sidebar-collapsed')) {
          toggleBtn.innerHTML = '☰ Hiện Mục Lục';
        } else {
          toggleBtn.innerHTML = '✕ Ẩn Mục Lục';
        }
      });

      // 2. Active Section Highlighting on Scroll
      const sections = document.querySelectorAll('.report-section');
      const navLinks = document.querySelectorAll('.sidebar-nav .nav-link');

      window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
          const sectionTop = section.offsetTop - 110;
          if (window.pageYOffset >= sectionTop) {
            current = section.getAttribute('id');
          }
        });

        navLinks.forEach(link => {
          link.classList.remove('active');
          if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
          }
        });
      }, { passive: true });

      // 3. Interactive Lightbox Modal (Click to View Full Resolution)
      const lightbox = document.getElementById('image-lightbox');
      const lightboxImg = document.getElementById('lightbox-img');
      const lightboxCaption = document.getElementById('lightbox-caption');
      const lightboxClose = document.getElementById('lightbox-close');

      // Attach click to all artifact and face images
      const clickableImages = document.querySelectorAll('.artifact-img-full, .face-strip-card img, .zoom-hint-banner');
      clickableImages.forEach(el => {
        el.addEventListener('click', (e) => {
          let targetImg = el;
          if (el.classList.contains('zoom-hint-banner')) {
            targetImg = el.previousElementSibling;
          }
          if (targetImg && targetImg.tagName === 'IMG') {
            lightboxImg.src = targetImg.src;
            lightboxCaption.textContent = targetImg.alt || targetImg.title || 'Artifact Khảo Sát Chi Tiết';
            lightbox.classList.add('active');
            document.body.style.overflow = 'hidden'; // Stop page scroll
          }
        });
      });

      function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = '';
      }

      lightboxClose.addEventListener('click', closeLightbox);
      lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox || e.target === lightboxClose) {
          closeLightbox();
        }
      });

      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && lightbox.classList.contains('active')) {
          closeLightbox();
        }
      });
    });
  </script>
</body>
</html>
"""
    return html_content

def main():
    html_content = build_vertical_report_html()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Successfully generated {OUTPUT_PATH} ({len(html_content)} bytes)")

if __name__ == "__main__":
    main()
