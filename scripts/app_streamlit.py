"""Streamlit Hybrid 4-Mode Visual Showcase & Live X-Ray Inspector (Topic 11)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.coursework_metrics import collect_live_coursework_metrics  # noqa: E402
from src.analysis.ui_showcase import (  # noqa: E402
    CurationMetricItem,
    DetectorBundle,
    InspectionResult,
    ShowcaseCatalogEntry,
    apply_interactive_thresholds,
    get_golden_presets,
    inspect_pipeline_sample,
    launch_live_sd_worker,
    load_curated_showcase_catalog,
    load_detector_bundle,
    render_latent_noise_preview,
    unload_detector_bundle,
)

TAB_CONFIGS: List[Tuple[str, str, float]] = [
    ("sd_text2img", "1. Text2Img", 0.20),
    ("sd_img2img", "2. Img2Img", 0.12),
    ("sd_inpaint", "3. Inpaint", 0.12),
    ("ood_and_real", "4. OOD RV5.1 & Real FFHQ", 0.20),
]
TAB_DEFAULT_CLIP: Dict[str, float] = {tab_id: clip_val for tab_id, _, clip_val in TAB_CONFIGS}


@st.cache_resource(show_spinner="Đang nạp bộ 3 detector thật (M0, M2*, M3)...")
def get_detector_bundle() -> Optional[DetectorBundle]:
    """Load and cache the real M0/M2*/M3 detector ensemble."""
    try:
        return load_detector_bundle()
    except Exception as exc:
        st.warning(f"Không thể nạp detector checkpoints: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def get_clip_scorer() -> Optional[Any]:
    """Load and cache CPU CLIP scorer once to avoid reloading per interaction."""
    try:
        from src.common.metrics import ClipScorer

        return ClipScorer(device="cpu")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_showcase_catalog() -> Dict[str, List[ShowcaseCatalogEntry]]:
    """Load held-out showcase catalog entries grouped by 4 presentation tabs."""
    return load_curated_showcase_catalog()


@st.cache_data(show_spinner=False)
def get_coursework_metrics() -> Dict[str, Any]:
    """Collect live Quad-Vault & Universal 20.4K benchmark metrics."""
    return collect_live_coursework_metrics(REPO_ROOT)


def _init_session_state(
    catalog: Dict[str, List[ShowcaseCatalogEntry]],
    presets: List[Dict[str, Any]],
) -> None:
    """Initialize Streamlit session state keys deterministically on first boot."""
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "sd_text2img"
    if "min_sharpness" not in st.session_state:
        st.session_state.min_sharpness = 1.3
    if "min_cheek_fft" not in st.session_state:
        st.session_state.min_cheek_fft = 0.0050
    if "min_clip" not in st.session_state:
        st.session_state.min_clip = TAB_DEFAULT_CLIP[st.session_state.active_tab]
    if "tau" not in st.session_state:
        st.session_state.tau = 0.50
    if "selected_sample_id" not in st.session_state:
        first_preset = presets[0] if presets else None
        if first_preset is not None:
            st.session_state.selected_sample_id = first_preset["primary_sample_id"]
        else:
            entries = catalog.get(st.session_state.active_tab, [])
            st.session_state.selected_sample_id = entries[0].sample_id if entries else None
    if "active_preset_id" not in st.session_state:
        st.session_state.active_preset_id = presets[0]["id"] if presets else None
    if "custom_target" not in st.session_state:
        st.session_state.custom_target = None
    if "gallery_limit" not in st.session_state:
        st.session_state.gallery_limit = 12


def _switch_tab(tab_id: str, sample_id: Optional[str] = None) -> None:
    """Switch active mode tab and reset Min CLIP slider to mode-appropriate default."""
    st.session_state.active_tab = tab_id
    st.session_state.min_clip = TAB_DEFAULT_CLIP.get(tab_id, 0.20)
    st.session_state.custom_target = None
    st.session_state.gallery_limit = 12
    if sample_id is not None:
        st.session_state.selected_sample_id = sample_id


def _render_scorecard_expander(metrics: Dict[str, Any]) -> None:
    """Render global Proof-of-Utility Scorecard from live coursework metrics."""
    with st.expander("Scorecard Toàn Cục M0 → M1 → M2* → M3 (Proof of Utility)", expanded=False):
        quad = metrics.get("quad", {})
        curation = metrics.get("curation", {})
        al = metrics.get("al", {})
        u_global = metrics.get("universal", {}).get("global", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "SD 1.5 Đạt Curation Gate",
            f"{curation.get('sd15_cur', 0):,}",
            f"{curation.get('sd15_pass', 0.0):.1f}% pass",
        )
        c2.metric(
            "Challenge Hard Vault",
            f"{al.get('challenge_total', 0):,}",
            f"Hard Fake: {al.get('challenge_fake', 0):,}",
        )
        m0_chal = quad.get("M0", {}).get("challenge", {})
        m3_chal = quad.get("M3", {}).get("challenge", {})
        c3.metric(
            "Challenge AUC (M0 → M3)",
            f"{m3_chal.get('auc', 0.0):.4f}",
            f"+{(m3_chal.get('auc', 0.0) - m0_chal.get('auc', 0.0)):.4f} vs M0",
        )
        m3_unseen = quad.get("M3", {}).get("unseen", {})
        m0_unseen = quad.get("M0", {}).get("unseen", {})
        c4.metric(
            "Unseen RV5.1 OOD Recall (M3)",
            f"{m3_unseen.get('recall', 0.0) * 100.0:.2f}%",
            f"+{(m3_unseen.get('recall', 0.0) - m0_unseen.get('recall', 0.0)) * 100.0:.2f}% vs M0",
        )

        quad_rows: List[Dict[str, Any]] = []
        for label, key in (
            ("M0 (Baseline)", "M0"),
            ("M1 (Active R1)", "M1"),
            ("M2* (Refined R2)", "M2_star"),
            ("M3 (Champion)", "M3"),
        ):
            m_data = quad.get(key, {})
            master = m_data.get("master", {})
            chal = m_data.get("challenge", {})
            unseen = m_data.get("unseen", {})
            quad_rows.append(
                {
                    "Model": label,
                    "Master AUC": round(float(master.get("auc", 0.0)), 4),
                    "Master F1": round(float(master.get("f1", 0.0)), 4),
                    "Challenge AUC": round(float(chal.get("auc", 0.0)), 4),
                    "Challenge Recall": f"{float(chal.get('recall', 0.0)) * 100.0:.2f}%",
                    "Unseen RV5.1 AUC": round(float(unseen.get("auc", 0.0)), 4),
                    "Unseen RV5.1 Recall": f"{float(unseen.get('recall', 0.0)) * 100.0:.2f}%",
                }
            )
        st.dataframe(quad_rows, use_container_width=True, hide_index=True)

        if u_global:
            st.caption("Universal Benchmark 20.4K — Tổng hợp đa nguồn:")
            uni_rows = [
                {
                    "Model": model_name,
                    "AUC": row.get("AUC", row.get("auc", "")),
                    "F1": row.get("F1", row.get("f1", "")),
                    "Accuracy": row.get("Accuracy", row.get("accuracy", "")),
                    "Recall": row.get("Recall", row.get("recall", "")),
                }
                for model_name, row in u_global.items()
            ]
            st.dataframe(uni_rows, use_container_width=True, hide_index=True)


def _evaluate_catalog_entry(
    entry: ShowcaseCatalogEntry,
    min_sharpness: float,
    min_cheek_fft: float,
    min_clip: float,
) -> Tuple[bool, List[str]]:
    """Re-evaluate a catalog entry against current interactive slider thresholds."""
    measured = dict(entry.measured)
    if entry.cheek_fft_ratio is not None and "cheek_fft_ratio" not in measured:
        measured["cheek_fft_ratio"] = entry.cheek_fft_ratio
    return apply_interactive_thresholds(
        measured,
        mode=entry.mode,
        min_sharpness=min_sharpness,
        min_cheek_fft=min_cheek_fft,
        min_clip=min_clip,
        max_delta_e=float(measured.get("effective_max_delta_e", 18.0)),
    )


def _metric_progress_ratio(item: CurationMetricItem) -> float:
    """Normalize a CurationMetricItem value into [0.0, 1.0] for st.progress display."""
    if item.value is None:
        return 0.0
    val = float(item.value)
    thr = float(item.threshold) if item.threshold is not None else 1.0
    if item.name == "sharpness":
        return max(0.0, min(1.0, val / max(1.0, thr * 4.0)))
    if item.name == "cheek_fft":
        return max(0.0, min(1.0, val / max(1e-4, thr * 2.5)))
    if item.name == "clip_score":
        return max(0.0, min(1.0, val / 0.40))
    if item.name == "geometry":
        return 1.0 if item.passed else 0.25
    if item.name == "delta_e_lab":
        return max(0.0, min(1.0, 1.0 - (val / max(1.0, thr * 2.0))))
    if item.name == "seam_diff":
        return max(0.0, min(1.0, 1.0 - (val / max(1.0, thr * 2.0))))
    return 1.0 if item.passed else 0.0


def _collect_ffhq_source_choices(catalog: Dict[str, List[ShowcaseCatalogEntry]]) -> List[str]:
    """Collect unique valid FFHQ source image paths from catalog entries."""
    seen: set[str] = set()
    choices: List[str] = []
    for entries in catalog.values():
        for entry in entries:
            cand = entry.image_path if entry.mode == "real_ffhq" else entry.source_image_path
            if cand and cand not in seen and Path(cand).is_file():
                seen.add(cand)
                choices.append(cand)
    return sorted(choices)


def _render_live_sandbox(
    catalog: Dict[str, List[ShowcaseCatalogEntry]],
    bundle: Optional[DetectorBundle],
) -> Optional[DetectorBundle]:
    """Render collapsible Upload Ảnh Lạ & Live SD 1.5 generator sandbox."""
    with st.expander("Upload Ảnh Lạ / Sinh thử 1 ảnh Live (SD 1.5)", expanded=False):
        col_up, col_sd = st.columns(2)

        with col_up:
            st.markdown("**Kiểm định ảnh upload bất kỳ**")
            up_target = st.file_uploader(
                "Chọn ảnh kiểm định",
                type=["png", "jpg", "jpeg", "webp"],
                key="upload_target_file",
            )
            up_source = st.file_uploader(
                "Ảnh gốc FFHQ đối chứng (tùy chọn cho Img2Img/Inpaint)",
                type=["png", "jpg", "jpeg", "webp"],
                key="upload_source_file",
            )
            up_prompt = st.text_input(
                "Prompt đánh giá CLIP (tùy chọn)",
                value="portrait photo of a person, natural skin texture",
                key="upload_prompt",
            )
            force_det = st.checkbox(
                "Chạy thử Detector kể cả khi ảnh bị loại",
                value=False,
                key="force_detector_checkbox",
            )
            if st.button("Phân tích ảnh upload", key="btn_analyze_upload", use_container_width=True):
                if up_target is None:
                    st.warning("Vui lòng chọn file ảnh trước khi phân tích.")
                else:
                    target_pil = ImageOps.exif_transpose(Image.open(up_target)).convert("RGB")
                    source_pil = (
                        ImageOps.exif_transpose(Image.open(up_source)).convert("RGB")
                        if up_source is not None
                        else None
                    )
                    upload_mode = "sd_img2img" if source_pil is not None else "custom_upload"
                    st.session_state.custom_target = {
                        "kind": "upload",
                        "title": f"Upload: {up_target.name}",
                        "mode": upload_mode,
                        "target_image": target_pil,
                        "source_image": source_pil,
                        "mask_region": None,
                        "prompt": up_prompt,
                        "seed": None,
                        "steps": None,
                        "guidance_scale": None,
                        "strength": None,
                        "ground_truth_label": None,
                        "run_detector_if_rejected": bool(force_det),
                    }
                    st.rerun()

        with col_sd:
            st.markdown("**Sinh thử 1 ảnh Live (SD 1.5 — 4GB VRAM Safe)**")
            ffhq_choices = _collect_ffhq_source_choices(catalog)
            with st.form("live_sd_generator_form"):
                sd_mode = st.selectbox(
                    "Chế độ sinh ảnh",
                    options=["text2img", "img2img", "inpaint"],
                    index=0,
                )
                sd_source = st.selectbox(
                    "Ảnh nguồn FFHQ (cho Img2Img / Inpaint)",
                    options=ffhq_choices if ffhq_choices else ["(Không có ảnh FFHQ cục bộ)"],
                    index=0,
                )
                sd_mask_region = st.selectbox(
                    "Vùng Mask Inpaint",
                    options=["lips", "periocular", "full"],
                    index=0,
                )
                sd_prompt = st.text_input(
                    "Prompt sinh ảnh",
                    value="studio headshot portrait of a person, natural skin pores, realistic lighting",
                )
                sd_seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
                submitted = st.form_submit_button("Sinh thử 1 ảnh Live (SD 1.5)", use_container_width=True)

            if submitted:
                if sd_mode in ("img2img", "inpaint") and not Path(sd_source).is_file():
                    st.error("Chế độ Img2Img / Inpaint yêu cầu ảnh nguồn FFHQ hợp lệ.")
                else:
                    # Strictly release detector GPU cache before spawning SD 1.5 worker (4GB VRAM ceiling)
                    get_detector_bundle.clear()
                    unload_detector_bundle(bundle)
                    try:
                        with st.spinner("Đang giải phóng VRAM & sinh ảnh SD 1.5 trong worker cô lập..."):
                            worker_res = launch_live_sd_worker(
                                mode=sd_mode,
                                prompt=sd_prompt,
                                source_image=Path(sd_source) if sd_mode in ("img2img", "inpaint") else None,
                                mask_region=sd_mask_region,
                                seed=int(sd_seed),
                            )
                        canonical_mode = {
                            "text2img": "sd_text2img",
                            "img2img": "sd_img2img",
                            "inpaint": "sd_inpaint",
                        }[sd_mode]
                        st.session_state.active_tab = canonical_mode
                        st.session_state.min_clip = TAB_DEFAULT_CLIP[canonical_mode]
                        st.session_state.custom_target = {
                            "kind": "live_sd",
                            "title": f"Live SD 1.5 ({sd_mode}, seed={int(sd_seed)})",
                            "mode": canonical_mode,
                            "target_image": Path(worker_res["output_path"]),
                            "source_image": (
                                Path(worker_res["input_path"]) if "input_path" in worker_res else None
                            ),
                            "mask_path": Path(worker_res["mask_path"]) if "mask_path" in worker_res else None,
                            "mask_region": worker_res.get("mask_region", sd_mask_region),
                            "prompt": worker_res.get("prompt", sd_prompt),
                            "seed": int(worker_res.get("seed", sd_seed)),
                            "steps": worker_res.get("steps"),
                            "guidance_scale": worker_res.get("guidance_scale"),
                            "strength": worker_res.get("strength"),
                            "ground_truth_label": 1,
                            "run_detector_if_rejected": bool(
                                st.session_state.get("force_detector_checkbox", False)
                            ),
                            "worker_latency_ms": worker_res.get("latency_ms"),
                        }
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Sinh ảnh Live SD 1.5 thất bại: {exc}")
                        bundle = get_detector_bundle()

    return bundle


def _render_xray_inspector(
    inspection: InspectionResult,
    *,
    title: str,
    prompt: str,
    seed: Optional[int],
    steps: Optional[int],
    guidance_scale: Optional[float],
    strength: Optional[float],
    mask_region: Optional[str],
    tau: float,
) -> None:
    """Render Khối 2 — 4-Stage X-Ray Inspector (Cột A, Cột B, Cột C)."""
    st.subheader(f"Khối 2 — Bảng Soi X-Ray 4 Tầng: {title}")
    col_a, col_b, col_c = st.columns([1.15, 1.0, 1.15], gap="medium")

    # Cột A: Tầng 1 — Mode-Native Input -> Output
    with col_a:
        st.markdown("#### Cột A · Tầng 1: Đầu vào → Đầu ra")
        show_overlay = st.checkbox(
            "Hiện lớp phủ YuNet / ROI da má / Mask",
            value=True,
            key="xray_show_overlay",
        )
        display_img = (
            inspection.overlays.get("face_audit", inspection.image_512_rgb)
            if show_overlay
            else inspection.image_512_rgb
        )

        if inspection.mode == "sd_text2img":
            noise_img = inspection.overlays.get("latent_noise")
            if noise_img is None and seed is not None:
                noise_img = render_latent_noise_preview(int(seed), size=256)
            in_col, out_col = st.columns(2)
            with in_col:
                if noise_img is not None:
                    st.image(noise_img, caption=f"Latent Noise (seed={seed})", use_container_width=True)
                else:
                    st.info("Text2Img Latent Noise")
                st.caption(f"Steps: {steps or 20} | CFG: {guidance_scale or 6.2}")
            with out_col:
                st.image(display_img, caption="Output (Text2Img)", use_container_width=True)
        elif inspection.mode == "sd_img2img":
            in_col, out_col = st.columns(2)
            with in_col:
                if inspection.source_512_rgb is not None:
                    st.image(inspection.source_512_rgb, caption="Input FFHQ Gốc", use_container_width=True)
                st.caption(f"Denoising Strength: {strength if strength is not None else 0.30}")
            with out_col:
                st.image(display_img, caption="Output (Img2Img)", use_container_width=True)
        elif inspection.mode == "sd_inpaint":
            in_col, out_col = st.columns(2)
            with in_col:
                src_vis = inspection.overlays.get("source_with_mask", inspection.source_512_rgb)
                if src_vis is not None:
                    st.image(
                        src_vis,
                        caption=f"Input FFHQ + Mask ({mask_region or 'auto'})",
                        use_container_width=True,
                    )
            with out_col:
                st.image(display_img, caption="Output (Inpaint)", use_container_width=True)
        else:
            if inspection.mode == "rv51_ood":
                st.info("Unseen RV5.1 OOD — nguồn Kaggle CelebA-HQ không lưu cục bộ (chỉ đơn ảnh).")
            elif inspection.mode == "real_ffhq":
                st.info("Held-out FFHQ Real đối chứng (ground_truth = 0).")
            st.image(display_img, caption=f"Ảnh kiểm định ({inspection.mode})", use_container_width=True)

        if prompt:
            st.caption(f"**Prompt:** {prompt[:220]}")

        if inspection.before_after_html is not None:
            st.caption("Kéo thanh rèm để soi Trước ↔ Sau:")
            components.html(inspection.before_after_html, height=340)

    # Cột B: Tầng 2 — Phiếu Thẩm Định Data Gate
    with col_b:
        st.markdown("#### Cột B · Tầng 2: Phiếu Thẩm Định Data Gate")
        if inspection.gate_passed:
            st.success("✓ ĐẠT CHUẨN — ĐƯỢC CHỌN VÀO DATASET")
        else:
            st.error("✕ BỊ LOẠI TẠI CỔNG DATA")
            reasons_to_show = inspection.rejection_reasons_vi or inspection.rejection_reasons
            for reason in reasons_to_show:
                st.markdown(f"- {reason}")

        for item in inspection.metrics:
            icon = "✓" if item.passed else "✕"
            val_str = f"{item.value:.4f}" if item.name == "cheek_fft" and item.value is not None else (
                f"{item.value:.3f}" if item.value is not None else "N/A"
            )
            thr_str = (
                f"{item.threshold:.4f}"
                if item.name == "cheek_fft" and item.threshold is not None
                else (f"{item.threshold:.2f}" if item.threshold is not None else "N/A")
            )
            st.markdown(f"**{icon} {item.label}**: `{val_str}` (ngưỡng `{item.direction} {thr_str}`)")
            st.progress(_metric_progress_ratio(item))
            if item.detail:
                st.caption(item.detail)

        if inspection.warnings:
            for warn in inspection.warnings:
                st.warning(warn)

    # Cột C: Tầng 3 & 4 — M0 Mining -> M0..M3 & Explainability
    with col_c:
        st.markdown("#### Cột C · Tầng 3 & 4: M0 Mining & M0..M3")
        if not inspection.gate_passed and not inspection.detector_scores:
            st.warning("Ảnh bị loại tại Cổng Data — Detector không chấm (bật tùy chọn ép chấm nếu muốn xem).")
        elif not inspection.detector_scores:
            st.warning("Không có điểm Detector — kiểm tra tải checkpoint / VRAM.")
        else:
            m0_prob = inspection.detector_scores.get("M0", 0.0)
            m3_prob = inspection.detector_scores.get("M3", 0.0)

            if inspection.mode == "real_ffhq" or inspection.ground_truth_label == 0:
                if m3_prob < tau:
                    st.success(f"✓ True Negative (Ảnh thật): M3 P(fake)={m3_prob:.3f} < τ={tau:.2f}")
                else:
                    st.error(f"✕ False Positive (Báo nhầm ảnh thật): M3 P(fake)={m3_prob:.3f} ≥ τ={tau:.2f}")
            elif inspection.is_hard_sample:
                st.warning(f"🔥 HARD SAMPLE — BỎ LỌT M0 (M0={m0_prob:.3f} < τ={tau:.2f})")
            else:
                st.info(f"✓ M0 BẮT NGAY (M0={m0_prob:.3f} ≥ τ={tau:.2f})")

            for model_key, label in (
                ("M0", "M0 (Baseline)"),
                ("M2*", "M2* (Active R2)"),
                ("M3", "M3 (Champion)"),
            ):
                if model_key in inspection.detector_scores:
                    prob = float(inspection.detector_scores[model_key])
                    pred_txt = "FAKE" if prob >= tau else "REAL"
                    st.markdown(f"**{label}**: `P(fake) = {prob:.4f}` → `{pred_txt}`")
                    st.progress(max(0.0, min(1.0, prob)))

            if "M0" in inspection.heatmaps and "M3" in inspection.heatmaps:
                st.caption("Bản đồ nhạy cảm Patch DINOv2 (M0 vs M3):")
                hm_l, hm_r = st.columns(2)
                with hm_l:
                    st.image(inspection.heatmaps["M0"], caption="M0 Patch Map", use_container_width=True)
                with hm_r:
                    st.image(inspection.heatmaps["M3"], caption="M3 Patch Map", use_container_width=True)

        if inspection.latencies_ms:
            lat_parts = [f"{k}: {v:.0f}ms" for k, v in inspection.latencies_ms.items()]
            st.caption("Độ trễ: " + " · ".join(lat_parts))


def main() -> None:
    """Render the 1-screen Streamlit Hybrid visual showcase application."""
    st.set_page_config(
        page_title="Topic 11 — Synthetic Facial Dataset & Deepfake Detection",
        layout="wide",
    )

    catalog = get_showcase_catalog()
    presets = get_golden_presets(catalog)
    coursework_metrics = get_coursework_metrics()
    bundle = get_detector_bundle()
    clip_scorer = get_clip_scorer()

    _init_session_state(catalog, presets)

    # Header & Presenter Bar
    st.title("Topic 11 — SD 1.5 Synthetic Facial Dataset & Deepfake Detection")
    if bundle is not None:
        vram_txt = f"{bundle.free_vram_mb:.0f} MiB free" if bundle.free_vram_mb is not None else "CPU mode"
        st.caption(f"Device: `{bundle.device}` ({vram_txt}) · Backbone: `{bundle.backbone}` · Held-Out Catalog")
    else:
        st.caption("Device: `CPU` · Detector checkpoints unavailable")

    st.markdown("**Kịch bản Vàng 1-Click (Golden Presets)**")
    preset_cols = st.columns(len(presets))
    for col, preset in zip(preset_cols, presets):
        with col:
            if st.button(preset["label"], key=f"preset_btn_{preset['id']}", use_container_width=True):
                st.session_state.active_preset_id = preset["id"]
                _switch_tab(preset["tab"], sample_id=preset["primary_sample_id"])
                st.rerun()

    active_preset = next((p for p in presets if p["id"] == st.session_state.active_preset_id), None)
    if active_preset is not None and active_preset["tab"] == st.session_state.active_tab:
        p_col1, p_col2 = st.columns([3, 1])
        p_col1.caption(active_preset["description"])
        if p_col2.button(
            "Soi ca đối chứng (Fail / Real)",
            key="btn_preset_secondary",
            use_container_width=True,
        ):
            st.session_state.custom_target = None
            st.session_state.selected_sample_id = active_preset["secondary_sample_id"]
            st.rerun()

    _render_scorecard_expander(coursework_metrics)

    # Khối 1: Mode Selector, Sliders & Gallery
    st.divider()
    st.subheader("Khối 1 — Chọn Chế Độ, Ngưỡng Data Gate & Kho Mẫu Held-Out")
    mode_cols = st.columns(len(TAB_CONFIGS))
    for col, (tab_id, tab_label, _) in zip(mode_cols, TAB_CONFIGS):
        with col:
            is_active = st.session_state.active_tab == tab_id
            if st.button(
                tab_label,
                key=f"mode_btn_{tab_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                _switch_tab(tab_id)
                st.rerun()

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    min_sharpness = s_col1.slider(
        "Min Sharpness",
        min_value=0.5,
        max_value=10.0,
        step=0.1,
        key="min_sharpness",
    )
    min_cheek_fft = s_col2.slider(
        "Min Cheek FFT",
        min_value=0.0010,
        max_value=0.0300,
        step=0.0005,
        format="%.4f",
        key="min_cheek_fft",
    )
    min_clip = s_col3.slider(
        "Min CLIP",
        min_value=0.05,
        max_value=0.40,
        step=0.01,
        key="min_clip",
    )
    tau = s_col4.slider(
        "Ngưỡng τ",
        min_value=0.10,
        max_value=0.90,
        step=0.05,
        key="tau",
    )

    active_entries = catalog.get(st.session_state.active_tab, [])
    evaluated_entries: List[Tuple[ShowcaseCatalogEntry, bool, List[str]]] = [
        (entry, *_evaluate_catalog_entry(entry, min_sharpness, min_cheek_fft, min_clip))
        for entry in active_entries
    ]

    if st.session_state.active_tab != "ood_and_real":
        total_cnt = len(evaluated_entries)
        pass_cnt = sum(1 for _, ok, _ in evaluated_entries if ok)
        fail_cnt = total_cnt - pass_cnt
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Tổng mẫu", total_cnt)
        f2.metric("✓ Vượt qua", pass_cnt)
        f3.metric("✕ Bị loại", fail_cnt)
        f4.metric("Tỷ lệ đạt", f"{(pass_cnt / max(1, total_cnt)) * 100.0:.1f}%")
        filter_choice = st.radio(
            "Lọc lưới mẫu",
            options=["Tất cả", "✓ Vượt qua", "✕ Bị loại"],
            horizontal=True,
            key="gallery_filter_sd",
        )
        filtered = [
            item
            for item in evaluated_entries
            if filter_choice == "Tất cả"
            or (filter_choice == "✓ Vượt qua" and item[1])
            or (filter_choice == "✕ Bị loại" and not item[1])
        ]
    else:
        ood_cnt = sum(1 for e, _, _ in evaluated_entries if e.mode == "rv51_ood")
        real_cnt = sum(1 for e, _, _ in evaluated_entries if e.label == 0)
        f1, f2, f3 = st.columns(3)
        f1.metric("Tổng mẫu Tab 4", len(evaluated_entries))
        f2.metric("RV5.1 OOD", ood_cnt)
        f3.metric("FFHQ Real (label=0)", real_cnt)
        filter_choice = st.radio(
            "Lọc nguồn kiểm định",
            options=["Tất cả", "RV5.1 OOD", "FFHQ Real"],
            horizontal=True,
            key="gallery_filter_ood",
        )
        filtered = [
            item
            for item in evaluated_entries
            if filter_choice == "Tất cả"
            or (filter_choice == "RV5.1 OOD" and item[0].mode == "rv51_ood")
            or (filter_choice == "FFHQ Real" and item[0].label == 0)
        ]

    # Ensure selected_sample_id belongs to current tab when no custom upload/live sample is active
    tab_ids = {e.sample_id for e in active_entries}
    if st.session_state.selected_sample_id not in tab_ids and active_entries:
        st.session_state.selected_sample_id = filtered[0][0].sample_id if filtered else active_entries[0].sample_id

    visible_items = filtered[: st.session_state.gallery_limit]
    if visible_items:
        cols_per_row = 4
        for start in range(0, len(visible_items), cols_per_row):
            row_items = visible_items[start : start + cols_per_row]
            row_cols = st.columns(cols_per_row)
            for col, (entry, ok, _) in zip(row_cols, row_items):
                with col:
                    if Path(entry.image_path).is_file():
                        st.image(entry.image_path, use_container_width=True)
                    badge = "✓ Đạt" if ok else "✕ Loại"
                    if entry.label == 0:
                        badge = "FFHQ Real"
                    elif entry.mode == "rv51_ood":
                        badge = "OOD RV5.1"
                    m3_sc = entry.scores.get("M3", 0.0)
                    st.caption(f"{badge} · `{entry.sample_id}` · M3={m3_sc:.2f}")
                    if st.button("Chọn soi X-Ray", key=f"select_{entry.sample_id}", use_container_width=True):
                        st.session_state.custom_target = None
                        st.session_state.selected_sample_id = entry.sample_id
                        st.rerun()

        if len(filtered) > len(visible_items):
            if st.button("Hiện thêm mẫu", key="btn_show_more_gallery"):
                st.session_state.gallery_limit += 12
                st.rerun()

    bundle = _render_live_sandbox(catalog, bundle)

    # Khối 2: 4-Stage X-Ray Inspector
    st.divider()
    custom = st.session_state.custom_target
    if custom is not None:
        mask_arr = None
        if custom.get("mask_path") is not None and Path(custom["mask_path"]).is_file():
            mask_arr = Image.open(custom["mask_path"]).convert("L")
        inspection = inspect_pipeline_sample(
            custom["target_image"],
            bundle=bundle,
            source_image=custom.get("source_image"),
            mask=mask_arr,
            mask_region=custom.get("mask_region"),
            mode=custom["mode"],
            prompt=custom.get("prompt", ""),
            seed=custom.get("seed"),
            min_sharpness=min_sharpness,
            min_cheek_fft=min_cheek_fft,
            min_clip=min_clip,
            tau=tau,
            ground_truth_label=custom.get("ground_truth_label"),
            run_detector_if_rejected=bool(st.session_state.get("force_detector_checkbox", False)),
            clip_scorer=clip_scorer,
            compute_heatmaps=True,
        )
        _render_xray_inspector(
            inspection,
            title=custom["title"],
            prompt=custom.get("prompt", ""),
            seed=custom.get("seed"),
            steps=custom.get("steps"),
            guidance_scale=custom.get("guidance_scale"),
            strength=custom.get("strength"),
            mask_region=custom.get("mask_region"),
            tau=tau,
        )
    else:
        entry = next((e for e in active_entries if e.sample_id == st.session_state.selected_sample_id), None)
        if entry is None and active_entries:
            entry = active_entries[0]
        if entry is not None:
            inspection = inspect_pipeline_sample(
                Path(entry.image_path),
                bundle=bundle,
                source_image=Path(entry.source_image_path) if entry.source_image_path else None,
                mask_region=entry.mask_region,
                mode=entry.mode,
                prompt=entry.prompt,
                seed=entry.seed,
                min_sharpness=min_sharpness,
                min_cheek_fft=min_cheek_fft,
                min_clip=min_clip,
                tau=tau,
                ground_truth_label=entry.label,
                run_detector_if_rejected=bool(st.session_state.get("force_detector_checkbox", False)),
                clip_scorer=clip_scorer,
                compute_heatmaps=True,
            )
            _render_xray_inspector(
                inspection,
                title=f"{entry.sample_id} ({entry.mode})",
                prompt=entry.prompt,
                seed=entry.seed,
                steps=entry.steps,
                guidance_scale=entry.guidance_scale,
                strength=entry.strength,
                mask_region=entry.mask_region,
                tau=tau,
            )


if __name__ == "__main__":
    main()
