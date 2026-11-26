"""Streamlit Hybrid Studio v2 — Visual Showcase, Auto Batch SD 1.5 & 4-Stage X-Ray Inspector."""

from __future__ import annotations

import hashlib
import html
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
    evaluate_batch_with_bundle,
    get_golden_presets,
    inspect_pipeline_sample,
    launch_live_sd_batch_worker,
    launch_live_sd_worker,
    load_curated_showcase_catalog,
    load_detector_bundle,
    render_latent_noise_preview,
    sample_random_batch_configs,
    sample_random_live_config,
    unload_detector_bundle,
)

TAB_CONFIGS: List[Tuple[str, str, float]] = [
    ("sd_text2img", "1. Text2Img", 0.20),
    ("sd_img2img", "2. Img2Img", 0.12),
    ("sd_inpaint", "3. Inpaint", 0.12),
    ("ood_and_real", "4. OOD RV5.1 & Real FFHQ", 0.20),
]
TAB_DEFAULT_CLIP: Dict[str, float] = {tab_id: clip_val for tab_id, _, clip_val in TAB_CONFIGS}
CANONICAL_SD_MODE_MAP: Dict[str, str] = {
    "text2img": "sd_text2img",
    "img2img": "sd_img2img",
    "inpaint": "sd_inpaint",
    "mixed": "sd_text2img",
}
DEFAULT_LIVE_PROMPT = "studio headshot portrait of a person, natural skin pores, realistic lighting"
XRAY_CACHE_MAX_ITEMS = 24
GALLERY_PAGE_SIZE = 6


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
    """Initialize session state and apply pending programmatic updates before widgets mount."""
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "sd_text2img"

    # Apply _pending_* keys cleanly before widget instantiation to avoid StreamlitAPIException
    if "_pending_min_clip" in st.session_state:
        st.session_state.min_clip = float(st.session_state.pop("_pending_min_clip"))
    if "_pending_live_sd_mode" in st.session_state:
        st.session_state.live_sd_mode = str(st.session_state.pop("_pending_live_sd_mode"))
    if "_pending_live_sd_prompt" in st.session_state:
        st.session_state.live_sd_prompt = str(st.session_state.pop("_pending_live_sd_prompt"))
    if "_pending_live_sd_seed" in st.session_state:
        st.session_state.live_sd_seed = int(st.session_state.pop("_pending_live_sd_seed"))
    if "_pending_live_sd_source" in st.session_state:
        pending_src = st.session_state.pop("_pending_live_sd_source")
        if pending_src is not None:
            st.session_state.live_sd_source = str(pending_src)
    if "_pending_live_sd_mask_region" in st.session_state:
        st.session_state.live_sd_mask_region = str(st.session_state.pop("_pending_live_sd_mask_region"))
    if "_pending_force_detector" in st.session_state:
        forced = bool(st.session_state.pop("_pending_force_detector"))
        st.session_state.force_detector_flag = forced
        st.session_state.force_detector_checkbox = forced

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
    if "gallery_page" not in st.session_state:
        st.session_state.gallery_page = 0
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None
    if "_xray_cache" not in st.session_state:
        st.session_state._xray_cache = {}
    if "force_detector_flag" not in st.session_state:
        st.session_state.force_detector_flag = False
    if "live_sd_mode" not in st.session_state:
        st.session_state.live_sd_mode = "text2img"
    if "live_sd_prompt" not in st.session_state:
        st.session_state.live_sd_prompt = DEFAULT_LIVE_PROMPT
    if "live_sd_seed" not in st.session_state:
        st.session_state.live_sd_seed = 42
    if "live_sd_mask_region" not in st.session_state:
        st.session_state.live_sd_mask_region = "lips"


def _on_force_detector_change() -> None:
    """Persist inline force-detector toggle across reruns even when widget unmounts."""
    st.session_state.force_detector_flag = bool(st.session_state.get("force_detector_checkbox", False))


def _switch_tab(tab_id: str, sample_id: Optional[str] = None) -> None:
    """Switch active mode tab and reset Min CLIP slider via pending key."""
    st.session_state.active_tab = tab_id
    st.session_state["_pending_min_clip"] = TAB_DEFAULT_CLIP.get(tab_id, 0.20)
    st.session_state.custom_target = None
    st.session_state.gallery_page = 0
    if sample_id is not None:
        st.session_state.selected_sample_id = sample_id


def _render_scorecard_expander(metrics: Dict[str, Any]) -> None:
    """Render global Proof-of-Utility Scorecard from live coursework metrics."""
    with st.expander("Scorecard Toàn Cục M0 → M1 → M2* → M3 (Proof of Utility)", expanded=False):
        quad = metrics.get("quad", {})
        curation = metrics.get("curation", {})
        al = metrics.get("al", {})
        u_global = metrics.get("universal", {}).get("global", {})

        c1, c2 = st.columns(2)
        c1.metric(
            "SD 1.5 Đạt Gate",
            f"{curation.get('sd15_cur', 0):,}",
            f"{curation.get('sd15_pass', 0.0):.1f}% pass",
        )
        c2.metric(
            "Hard Vault",
            f"{al.get('challenge_total', 0):,}",
            f"Fake: {al.get('challenge_fake', 0):,}",
        )

        m0_chal = quad.get("M0", {}).get("challenge", {})
        m3_chal = quad.get("M3", {}).get("challenge", {})
        m3_unseen = quad.get("M3", {}).get("unseen", {})
        m0_unseen = quad.get("M0", {}).get("unseen", {})

        c3, c4 = st.columns(2)
        c3.metric(
            "Challenge AUC (M3)",
            f"{m3_chal.get('auc', 0.0):.4f}",
            f"+{(m3_chal.get('auc', 0.0) - m0_chal.get('auc', 0.0)):.4f} vs M0",
        )
        c4.metric(
            "Unseen RV5.1 Recall",
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
                    "Challenge AUC": round(float(chal.get("auc", 0.0)), 4),
                    "Challenge Recall": f"{float(chal.get('recall', 0.0)) * 100.0:.2f}%",
                    "Unseen RV5.1 Recall": f"{float(unseen.get('recall', 0.0)) * 100.0:.2f}%",
                }
            )
        st.dataframe(quad_rows, width="stretch", hide_index=True)

        if u_global:
            st.caption("Universal Benchmark 20.4K — Tổng hợp đa nguồn:")
            uni_rows = [
                {
                    "Model": model_name,
                    "AUC": row.get("AUC", row.get("auc", "")),
                    "F1": row.get("F1", row.get("f1", "")),
                    "Recall": row.get("Recall", row.get("recall", "")),
                }
                for model_name, row in u_global.items()
            ]
            st.dataframe(uni_rows, width="stretch", hide_index=True)


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
    """Normalize a CurationMetricItem value into [0.0, 1.0] for progress display."""
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


def _render_semantic_bar_html(
    ratio: float,
    color: str,
    label_left: str,
    label_right: str,
) -> str:
    """Render a color-coded HTML semantic progress bar (Green=#16a34a, Red=#dc2626, Gray=#64748b)."""
    pct = max(0, min(100, int(round(float(ratio) * 100.0))))
    safe_left = html.escape(str(label_left))
    safe_right = html.escape(str(label_right))
    safe_color = html.escape(str(color))
    return (
        "<div style='margin:4px 0 8px 0;'>"
        "<div style='display:flex;justify-content:space-between;font-size:0.84rem;margin-bottom:3px;'>"
        f"<span style='font-weight:600;color:#e2e8f0;'>{safe_left}</span>"
        f"<span style='font-family:monospace;color:{safe_color};font-weight:600;'>{safe_right}</span>"
        "</div>"
        "<div style='width:100%;height:9px;background:#1e293b;border-radius:999px;overflow:hidden;border:1px solid #334155;'>"
        f"<div style='width:{pct}%;height:100%;background:{safe_color};border-radius:999px;transition:width 0.2s ease;'></div>"
        "</div>"
        "</div>"
    )


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


def _compute_target_sha256(target_image: Any) -> str:
    """Compute a stable SHA-256 cache key for a target Path or PIL Image."""
    if isinstance(target_image, Path):
        if target_image.is_file():
            return hashlib.sha256(target_image.read_bytes()).hexdigest()[:16]
        return hashlib.sha256(str(target_image).encode("utf-8")).hexdigest()[:16]
    if isinstance(target_image, Image.Image):
        return hashlib.sha256(target_image.convert("RGB").tobytes()).hexdigest()[:16]
    return hashlib.sha256(str(target_image).encode("utf-8")).hexdigest()[:16]


def _inspect_with_cache(
    target_image: Any,
    *,
    bundle: Optional[DetectorBundle],
    source_image: Any = None,
    mask: Any = None,
    mask_region: Optional[str] = None,
    mode: str = "sd_text2img",
    prompt: str = "",
    seed: Optional[int] = None,
    min_sharpness: float = 1.3,
    min_cheek_fft: float = 0.0050,
    min_clip: float = 0.20,
    tau: float = 0.50,
    ground_truth_label: Optional[int] = None,
    run_detector_if_rejected: bool = False,
    clip_scorer: Optional[Any] = None,
) -> InspectionResult:
    """Inspect a sample using bounded LRU session cache (`_xray_cache`, max 24 entries)."""
    cache: Dict[Tuple[Any, ...], Dict[str, Any]] = st.session_state._xray_cache
    bundle_dev = str(bundle.device) if bundle is not None else "none"
    img_hash = _compute_target_sha256(target_image)
    src_hash = _compute_target_sha256(source_image) if source_image is not None else "none"
    mask_hash = _compute_target_sha256(mask) if mask is not None else "none"
    cache_key = (
        img_hash,
        src_hash,
        mask_hash,
        str(mode),
        str(prompt or ""),
        str(mask_region or ""),
        bundle_dev,
    )
    cached_payload = cache.pop(cache_key, None)
    if cached_payload is not None:
        cache[cache_key] = cached_payload

    inspection = inspect_pipeline_sample(
        target_image,
        bundle=bundle,
        source_image=source_image,
        mask=mask,
        mask_region=mask_region,
        mode=mode,
        prompt=prompt,
        seed=seed,
        min_sharpness=min_sharpness,
        min_cheek_fft=min_cheek_fft,
        min_clip=min_clip,
        tau=tau,
        ground_truth_label=ground_truth_label,
        run_detector_if_rejected=run_detector_if_rejected,
        clip_scorer=clip_scorer,
        compute_heatmaps=True,
        precomputed=cached_payload,
    )

    # Cache expensive intermediate outputs once computed (bounded LRU)
    should_store = cached_payload is None or (
        bool(inspection.detector_scores) and not cached_payload.get("detector_scores")
    )
    if should_store:
        if cache_key in cache:
            del cache[cache_key]
        elif len(cache) >= XRAY_CACHE_MAX_ITEMS:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
        cache[cache_key] = {
            "rgb_512": inspection.image_512_rgb,
            "source_512": inspection.source_512_rgb,
            "mask_512": inspection.mask_512,
            "yunet_face": inspection.yunet_face,
            "cheek_boxes": inspection.cheek_boxes,
            "biometrics": inspection.biometrics,
            "measured_base": dict(inspection.measured),
            "before_after_html": inspection.before_after_html,
            "source_with_mask_overlay": inspection.overlays.get("source_with_mask"),
            "detector_scores": dict(inspection.detector_scores),
            "heatmaps": dict(inspection.heatmaps),
            "warnings": list(inspection.warnings),
        }

    return inspection


def _render_sidebar(
    catalog: Dict[str, List[ShowcaseCatalogEntry]],
    presets: List[Dict[str, Any]],
    coursework_metrics: Dict[str, Any],
    bundle: Optional[DetectorBundle],
) -> Tuple[float, float, float, float]:
    """Render Studio Sidebar with Device/VRAM pill, 4 modes, presets, sliders, and scorecard."""
    with st.sidebar:
        st.markdown("## Studio Controls")
        if bundle is not None:
            vram_txt = f"{bundle.free_vram_mb:.0f} MiB free" if bundle.free_vram_mb is not None else "CPU"
            pill_color = "#16a34a" if bundle.device.type == "cuda" else "#d97706"
            st.markdown(
                f"<div style='padding:6px 10px;border-radius:8px;background:#0f172a;border:1px solid {pill_color};"
                f"font-size:0.82rem;margin-bottom:10px;'>"
                f"<strong style='color:{pill_color};'>● {html.escape(str(bundle.device).upper())}</strong> · "
                f"{html.escape(vram_txt)} · <code>{html.escape(bundle.backbone)}</code>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Detector checkpoints unavailable (CPU fallback)")

        st.markdown("**1. Chế độ Pipeline**")
        for tab_id, tab_label, _ in TAB_CONFIGS:
            is_active = st.session_state.active_tab == tab_id
            if st.button(
                tab_label,
                key=f"mode_btn_{tab_id}",
                type="primary" if is_active else "secondary",
                width="stretch",
            ):
                _switch_tab(tab_id)
                st.rerun()

        st.divider()
        st.markdown("**2. Kịch bản Vàng (Golden Presets)**")
        for preset in presets:
            if st.button(preset["label"], key=f"preset_btn_{preset['id']}", width="stretch"):
                st.session_state.active_preset_id = preset["id"]
                _switch_tab(preset["tab"], sample_id=preset["primary_sample_id"])
                st.rerun()

        active_preset = next((p for p in presets if p["id"] == st.session_state.active_preset_id), None)
        if active_preset is not None and active_preset["tab"] == st.session_state.active_tab:
            st.caption(active_preset["description"])
            if st.button("Soi ca đối chứng (Fail / Real)", key="btn_preset_secondary", width="stretch"):
                st.session_state.custom_target = None
                st.session_state.selected_sample_id = active_preset["secondary_sample_id"]
                st.rerun()

        st.divider()
        st.markdown("**3. Ngưỡng Kiểm Định & Quyết Định**")
        if st.session_state.active_tab == "ood_and_real":
            tau = st.slider(
                "Ngưỡng τ",
                min_value=0.10,
                max_value=0.90,
                step=0.05,
                key="tau",
            )
            with st.expander("Ngưỡng tham chiếu Curation (tùy chọn)", expanded=False):
                min_sharpness = st.slider(
                    "Min Sharpness",
                    min_value=0.5,
                    max_value=10.0,
                    step=0.1,
                    key="min_sharpness",
                )
                min_cheek_fft = st.slider(
                    "Min Cheek FFT",
                    min_value=0.0010,
                    max_value=0.0300,
                    step=0.0005,
                    format="%.4f",
                    key="min_cheek_fft",
                )
                min_clip = st.slider(
                    "Min CLIP",
                    min_value=0.05,
                    max_value=0.40,
                    step=0.01,
                    key="min_clip",
                )
        else:
            min_sharpness = st.slider(
                "Min Sharpness",
                min_value=0.5,
                max_value=10.0,
                step=0.1,
                key="min_sharpness",
            )
            min_cheek_fft = st.slider(
                "Min Cheek FFT",
                min_value=0.0010,
                max_value=0.0300,
                step=0.0005,
                format="%.4f",
                key="min_cheek_fft",
            )
            min_clip = st.slider(
                "Min CLIP",
                min_value=0.05,
                max_value=0.40,
                step=0.01,
                key="min_clip",
            )
            tau = st.slider(
                "Ngưỡng τ",
                min_value=0.10,
                max_value=0.90,
                step=0.05,
                key="tau",
            )

        st.divider()
        _render_scorecard_expander(coursework_metrics)

    return float(min_sharpness), float(min_cheek_fft), float(min_clip), float(tau)


def _render_heldout_tab(
    catalog: Dict[str, List[ShowcaseCatalogEntry]],
    min_sharpness: float,
    min_cheek_fft: float,
    min_clip: float,
) -> List[ShowcaseCatalogEntry]:
    """Render Held-Out 1-row 6-card strip with pagination and active highlight."""
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

    tab_ids = {e.sample_id for e in active_entries}
    if st.session_state.selected_sample_id not in tab_ids and active_entries:
        st.session_state.selected_sample_id = filtered[0][0].sample_id if filtered else active_entries[0].sample_id

    total_pages = max(1, (len(filtered) + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    page = max(0, min(int(st.session_state.gallery_page), total_pages - 1))
    st.session_state.gallery_page = page
    start_idx = page * GALLERY_PAGE_SIZE
    page_items = filtered[start_idx : start_idx + GALLERY_PAGE_SIZE]

    if page_items:
        cols = st.columns(GALLERY_PAGE_SIZE)
        for col, (entry, ok, _) in zip(cols, page_items):
            is_selected = (
                st.session_state.custom_target is None
                and st.session_state.selected_sample_id == entry.sample_id
            )
            with col:
                if Path(entry.image_path).is_file():
                    st.image(entry.image_path, width="stretch")
                badge = "✓ Đạt" if ok else "✕ Loại"
                if entry.label == 0:
                    badge = "FFHQ Real"
                elif entry.mode == "rv51_ood":
                    badge = "OOD RV5.1"
                if is_selected:
                    st.markdown("**▶ ĐANG SOI**")
                m3_sc = entry.scores.get("M3", 0.0)
                st.caption(f"{badge} · `{entry.sample_id}` · M3={m3_sc:.2f}")
                if st.button(
                    "▶ Đang soi" if is_selected else "Soi X-Ray",
                    key=f"select_{entry.sample_id}",
                    type="primary" if is_selected else "secondary",
                    width="stretch",
                ):
                    st.session_state.custom_target = None
                    st.session_state.selected_sample_id = entry.sample_id
                    st.rerun()

        nav_l, nav_mid, nav_r = st.columns([1, 2, 1])
        with nav_l:
            if st.button("< Trước", key="btn_gallery_prev", disabled=(page <= 0), width="stretch"):
                st.session_state.gallery_page = max(0, page - 1)
                st.rerun()
        with nav_mid:
            st.caption(f"Trang {page + 1} / {total_pages} ({len(filtered)} mẫu)")
        with nav_r:
            if st.button("Sau >", key="btn_gallery_next", disabled=(page >= total_pages - 1), width="stretch"):
                st.session_state.gallery_page = min(total_pages - 1, page + 1)
                st.rerun()

    return active_entries


def _launch_single_live_sd_and_set_target(
    *,
    sd_mode: str,
    prompt: str,
    source_image: Optional[Path],
    mask_region: str,
    seed: int,
    bundle: Optional[DetectorBundle],
) -> Optional[DetectorBundle]:
    """Unload detectors, run single SD 1.5 worker, and populate custom_target."""
    get_detector_bundle.clear()
    unload_detector_bundle(bundle)
    try:
        with st.spinner("Đang giải phóng VRAM & sinh ảnh SD 1.5 trong worker cô lập..."):
            worker_res = launch_live_sd_worker(
                mode=sd_mode,
                prompt=prompt,
                source_image=source_image,
                mask_region=mask_region,
                seed=int(seed),
            )
        canonical_mode = CANONICAL_SD_MODE_MAP.get(sd_mode, "sd_text2img")
        st.session_state.active_tab = canonical_mode
        st.session_state["_pending_min_clip"] = TAB_DEFAULT_CLIP[canonical_mode]
        st.session_state.custom_target = {
            "kind": "live_sd",
            "title": f"Live SD 1.5 ({sd_mode}, seed={int(seed)})",
            "mode": canonical_mode,
            "target_image": Path(worker_res["output_path"]),
            "source_image": Path(worker_res["input_path"]) if "input_path" in worker_res else None,
            "mask_path": Path(worker_res["mask_path"]) if "mask_path" in worker_res else None,
            "mask_region": worker_res.get("mask_region", mask_region),
            "prompt": worker_res.get("prompt", prompt),
            "seed": int(worker_res.get("seed", seed)),
            "steps": worker_res.get("steps"),
            "guidance_scale": worker_res.get("guidance_scale"),
            "strength": worker_res.get("strength"),
            "ground_truth_label": 1,
            "run_detector_if_rejected": bool(st.session_state.get("force_detector_flag", False)),
            "worker_latency_ms": worker_res.get("latency_ms"),
        }
        st.rerun()
    except Exception as exc:
        st.error(f"Sinh ảnh Live SD 1.5 thất bại: {exc}")
        return get_detector_bundle()
    return bundle


def _render_live_and_batch_tab(
    catalog: Dict[str, List[ShowcaseCatalogEntry]],
    bundle: Optional[DetectorBundle],
    clip_scorer: Optional[Any],
    min_sharpness: float,
    min_cheek_fft: float,
    min_clip: float,
    tau: float,
) -> Optional[DetectorBundle]:
    """Render Live SD 1.5 Generator + Random Config + 1-Click Random + Auto Batch (N=1..12)."""
    ffhq_choices = _collect_ffhq_source_choices(catalog)

    act_c1, act_c2 = st.columns(2)
    with act_c1:
        if st.button("🎲 Random cấu hình", key="btn_random_sd_config", width="stretch"):
            cur_mode = st.session_state.get("live_sd_mode", "text2img")
            req_mode = None if cur_mode == "mixed" else cur_mode
            if req_mode in ("img2img", "inpaint") and not ffhq_choices:
                req_mode = "text2img"
            cfg = sample_random_live_config(catalog, mode=req_mode)
            st.session_state["_pending_live_sd_mode"] = cfg["mode"]
            st.session_state["_pending_live_sd_prompt"] = cfg["prompt"]
            st.session_state["_pending_live_sd_seed"] = int(cfg["seed"])
            st.session_state["_pending_live_sd_mask_region"] = cfg["mask_region"]
            if cfg.get("source_image"):
                st.session_state["_pending_live_sd_source"] = cfg["source_image"]
            st.rerun()

    with act_c2:
        if st.button("⚡ Sinh ngẫu nhiên ngay (1-Click)", key="btn_random_sd_oneclick", width="stretch"):
            cur_mode = st.session_state.get("live_sd_mode", "text2img")
            req_mode = None if cur_mode == "mixed" else cur_mode
            if req_mode in ("img2img", "inpaint") and not ffhq_choices:
                req_mode = "text2img"
            cfg = sample_random_live_config(catalog, mode=req_mode)
            st.session_state["_pending_live_sd_mode"] = cfg["mode"]
            st.session_state["_pending_live_sd_prompt"] = cfg["prompt"]
            st.session_state["_pending_live_sd_seed"] = int(cfg["seed"])
            st.session_state["_pending_live_sd_mask_region"] = cfg["mask_region"]
            if cfg.get("source_image"):
                st.session_state["_pending_live_sd_source"] = cfg["source_image"]
            src_arg = Path(cfg["source_image"]) if cfg.get("source_image") else None
            bundle = _launch_single_live_sd_and_set_target(
                sd_mode=str(cfg["mode"]),
                prompt=str(cfg["prompt"]),
                source_image=src_arg,
                mask_region=str(cfg["mask_region"]),
                seed=int(cfg["seed"]),
                bundle=bundle,
            )

    cfg_col, batch_col = st.columns([1.25, 1.0], gap="medium")
    with cfg_col:
        sd_mode = st.selectbox(
            "Chế độ sinh ảnh SD 1.5",
            options=["text2img", "img2img", "inpaint", "mixed"],
            key="live_sd_mode",
        )
        sd_source = ""
        if sd_mode in ("img2img", "inpaint", "mixed"):
            src_sel_col, src_thumb_col = st.columns([2.2, 1.0])
            with src_sel_col:
                sd_source = st.selectbox(
                    "Ảnh nguồn FFHQ (cho Img2Img / Inpaint)",
                    options=ffhq_choices if ffhq_choices else ["(Không có ảnh FFHQ cục bộ)"],
                    key="live_sd_source",
                )
            with src_thumb_col:
                if sd_source and Path(sd_source).is_file():
                    with Image.open(sd_source) as im:
                        thumb = im.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
                    st.image(thumb, caption="FFHQ 128×128", width=128)
        else:
            st.caption("Text2Img sinh ảnh mới từ nhiễu Gauss + Prompt (không dùng ảnh nguồn).")

        if sd_mode in ("inpaint", "mixed"):
            sd_mask_region = st.selectbox(
                "Vùng Mask Inpaint",
                options=["lips", "periocular", "nose", "full"],
                key="live_sd_mask_region",
            )
        else:
            sd_mask_region = "lips"

        with st.form("live_sd_generator_form"):
            sd_prompt = st.text_input("Prompt sinh ảnh", key="live_sd_prompt")
            sd_seed = st.number_input("Seed", min_value=0, max_value=999999, step=1, key="live_sd_seed")
            submitted = st.form_submit_button("Sinh thử 1 ảnh Live (SD 1.5)", width="stretch")

        if submitted:
            eff_mode = "text2img" if sd_mode == "mixed" else sd_mode
            if eff_mode in ("img2img", "inpaint") and not Path(sd_source).is_file():
                st.error("Chế độ Img2Img / Inpaint yêu cầu ảnh nguồn FFHQ hợp lệ.")
            else:
                bundle = _launch_single_live_sd_and_set_target(
                    sd_mode=eff_mode,
                    prompt=sd_prompt,
                    source_image=Path(sd_source) if eff_mode in ("img2img", "inpaint") else None,
                    mask_region=sd_mask_region,
                    seed=int(sd_seed),
                    bundle=bundle,
                )

    with batch_col:
        st.markdown("**Auto Batch Pipeline (N = 1..12)**")
        batch_n = st.slider(
            "Số lượng mẫu N",
            min_value=1,
            max_value=12,
            value=4,
            step=1,
            key="batch_count",
        )
        if st.button("🚀 Auto Sinh & Kiểm Định Lô (N mẫu)", key="btn_auto_batch_run", width="stretch"):
            try:
                configs = sample_random_batch_configs(catalog, count=int(batch_n), mode=sd_mode)
                get_detector_bundle.clear()
                unload_detector_bundle(bundle)
                with st.spinner(f"Đang sinh lô {batch_n} ảnh SD 1.5 trong 1 worker cô lập..."):
                    worker_batch = launch_live_sd_batch_worker(configs)
                bundle = get_detector_bundle()
                eval_res = evaluate_batch_with_bundle(
                    worker_batch.get("items", []),
                    bundle=bundle,
                    min_sharpness=min_sharpness,
                    min_cheek_fft=min_cheek_fft,
                    min_clip=None if sd_mode == "mixed" else min_clip,
                    tau=tau,
                    clip_scorer=clip_scorer,
                )
                st.session_state.batch_results = eval_res
                if eval_res["items"]:
                    first_entry = eval_res["items"][0]
                    first_item = first_entry["item"]
                    first_mode = CANONICAL_SD_MODE_MAP.get(str(first_item.get("mode", "text2img")), "sd_text2img")
                    st.session_state.active_tab = first_mode
                    st.session_state["_pending_min_clip"] = TAB_DEFAULT_CLIP[first_mode]
                    st.session_state.custom_target = {
                        "kind": "batch_item",
                        "batch_index": 0,
                        "title": f"Auto Batch #1 ({first_item.get('mode', 'text2img')}, seed={first_item.get('seed')})",
                        "mode": first_mode,
                        "target_image": Path(first_item["output_path"]),
                        "source_image": Path(first_item["input_path"]) if first_item.get("input_path") else None,
                        "mask_path": Path(first_item["mask_path"]) if first_item.get("mask_path") else None,
                        "mask_region": first_item.get("mask_region", "lips"),
                        "prompt": first_item.get("prompt", ""),
                        "seed": first_item.get("seed"),
                        "steps": first_item.get("steps"),
                        "guidance_scale": first_item.get("guidance_scale"),
                        "strength": first_item.get("strength"),
                        "ground_truth_label": 1,
                        "run_detector_if_rejected": bool(st.session_state.get("force_detector_flag", False)),
                        "worker_latency_ms": first_item.get("latency_ms"),
                    }
                st.rerun()
            except Exception as exc:
                st.error(f"Auto Batch SD 1.5 thất bại: {exc}")
                bundle = get_detector_bundle()

        batch_res = st.session_state.get("batch_results")
        if batch_res is not None:
            funnel = batch_res.get("funnel", {})
            st.caption("Phễu Pipeline Lô:")
            bf1, bf2, bf3, bf4 = st.columns(4)
            bf1.metric("Sinh", funnel.get("generated", 0))
            bf2.metric("Đạt Gate", funnel.get("gate_passed", 0))
            bf3.metric("M0 Bỏ Lọt", funnel.get("hard_samples", 0))
            bf4.metric("M3 Cứu", funnel.get("m3_rescued", 0))

    batch_res = st.session_state.get("batch_results")
    if batch_res is not None and batch_res.get("items"):
        st.markdown("**Danh sách mẫu Auto Batch (nhấn để soi X-Ray):**")
        b_items = batch_res["items"]
        cols_per_row = min(6, max(1, len(b_items)))
        for start in range(0, len(b_items), cols_per_row):
            chunk = b_items[start : start + cols_per_row]
            b_cols = st.columns(cols_per_row)
            for offset, (col, b_entry) in enumerate(zip(b_cols, chunk)):
                idx = start + offset
                item = b_entry["item"]
                out_p = Path(str(item.get("output_path", "")))
                is_active_batch = (
                    st.session_state.custom_target is not None
                    and st.session_state.custom_target.get("kind") == "batch_item"
                    and st.session_state.custom_target.get("batch_index") == idx
                )
                with col:
                    if out_p.is_file():
                        st.image(str(out_p), width="stretch")
                    status_lbl = "✓ Gate" if b_entry["gate_passed"] else "✕ Gate"
                    if b_entry["is_hard_sample"]:
                        status_lbl += " · 🔥Hard"
                    m3_v = b_entry["detector_scores"].get("M3")
                    m3_txt = f" · M3={m3_v:.2f}" if m3_v is not None else ""
                    st.caption(f"#{idx + 1} {item.get('mode', '')} · {status_lbl}{m3_txt}")
                    if st.button(
                        "▶ Đang soi" if is_active_batch else f"Soi #{idx + 1}",
                        key=f"select_batch_{idx}",
                        type="primary" if is_active_batch else "secondary",
                        width="stretch",
                    ):
                        b_mode = CANONICAL_SD_MODE_MAP.get(str(item.get("mode", "text2img")), "sd_text2img")
                        st.session_state.active_tab = b_mode
                        st.session_state["_pending_min_clip"] = TAB_DEFAULT_CLIP[b_mode]
                        st.session_state.custom_target = {
                            "kind": "batch_item",
                            "batch_index": idx,
                            "title": f"Auto Batch #{idx + 1} ({item.get('mode', 'text2img')}, seed={item.get('seed')})",
                            "mode": b_mode,
                            "target_image": out_p,
                            "source_image": Path(item["input_path"]) if item.get("input_path") else None,
                            "mask_path": Path(item["mask_path"]) if item.get("mask_path") else None,
                            "mask_region": item.get("mask_region", "lips"),
                            "prompt": item.get("prompt", ""),
                            "seed": item.get("seed"),
                            "steps": item.get("steps"),
                            "guidance_scale": item.get("guidance_scale"),
                            "strength": item.get("strength"),
                            "ground_truth_label": 1,
                            "run_detector_if_rejected": bool(st.session_state.get("force_detector_flag", False)),
                            "worker_latency_ms": item.get("latency_ms"),
                        }
                        st.rerun()

    return bundle


def _render_upload_tab() -> None:
    """Render Upload Ảnh Lạ tab for arbitrary target & optional FFHQ source inspection."""
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        up_target = st.file_uploader(
            "Chọn ảnh kiểm định",
            type=["png", "jpg", "jpeg", "webp"],
            key="upload_target_file",
        )
    with u_col2:
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
    if st.button("Phân tích ảnh upload", key="btn_analyze_upload", width="stretch"):
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
                "run_detector_if_rejected": bool(st.session_state.get("force_detector_flag", False)),
            }
            st.rerun()


def _render_pipeline_ribbon(
    inspection: InspectionResult,
    *,
    seed: Optional[int],
    tau: float,
) -> None:
    """Render 4-stage Pipeline Flow Ribbon (Input -> Data Gate -> M0 Mining -> M3 Verdict)."""
    # Stage 1: Input & Synthesis
    s1_color = "#3b82f6"
    s1_status = f"Mode: {inspection.mode}" + (f" · seed={seed}" if seed is not None else "")

    # Stage 2: Data Curation Gate
    passed_cnt = sum(1 for m in inspection.metrics if m.passed)
    total_cnt = len(inspection.metrics)
    if inspection.gate_passed:
        s2_color = "#16a34a"
        s2_status = f"✓ Đạt {passed_cnt}/{total_cnt}"
    else:
        s2_color = "#dc2626"
        first_reason = (inspection.rejection_reasons_vi or inspection.rejection_reasons or ["Không đạt"])[0]
        s2_status = f"✕ Loại: {first_reason[:36]}"

    # Stage 3: M0 Hard-Sample Mining
    if not inspection.detector_scores:
        s3_color = "#64748b"
        s3_status = "Skipped (Gate loại)" if not inspection.gate_passed else "Chưa nạp Detector"
    elif inspection.is_hard_sample:
        s3_color = "#d97706"
        s3_status = f"🔥 M0 Bỏ Lọt ({inspection.detector_scores.get('M0', 0.0):.3f} < {tau:.2f})"
    else:
        s3_color = "#16a34a"
        s3_status = f"✓ M0={inspection.detector_scores.get('M0', 0.0):.3f}"

    # Stage 4: Detector M3 & Explainability
    if not inspection.detector_scores:
        s4_color = "#64748b"
        s4_status = "Chưa chấm"
    else:
        m3_prob = float(inspection.detector_scores.get("M3", 0.0))
        if inspection.mode == "real_ffhq" or inspection.ground_truth_label == 0:
            s4_color = "#16a34a" if m3_prob < tau else "#dc2626"
            s4_status = f"{'✓ REAL' if m3_prob < tau else '✕ FP'} · P(fake)={m3_prob:.3f}"
        elif inspection.is_hard_sample and m3_prob >= tau:
            s4_color = "#16a34a"
            s4_status = f"✓ M0→M3 Cứu · P={m3_prob:.3f}"
        else:
            s4_color = "#dc2626" if m3_prob >= tau else "#d97706"
            s4_status = f"{'FAKE' if m3_prob >= tau else 'REAL'} · P={m3_prob:.3f}"

    stages = [
        ("① Input & Synthesis", s1_status, s1_color),
        ("② Data Gate", s2_status, s2_color),
        ("③ M0 Hard Mining", s3_status, s3_color),
        ("④ M3 Verdict & XAI", s4_status, s4_color),
    ]
    cards_html = []
    for title, status, color in stages:
        cards_html.append(
            "<div style='flex:1;min-width:170px;background:#0f172a;border:1px solid "
            f"{color};border-left:4px solid {color};border-radius:8px;padding:8px 10px;'>"
            f"<div style='font-size:0.78rem;font-weight:700;color:#cbd5e1;'>{html.escape(title)}</div>"
            f"<div style='font-size:0.84rem;font-weight:600;color:{color};margin-top:2px;'>{html.escape(status)}</div>"
            "</div>"
        )
    ribbon_html = (
        "<div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:6px 0 12px 0;'>"
        + "<div style='color:#64748b;font-weight:700;'>➔</div>".join(cards_html)
        + "</div>"
    )
    st.markdown(ribbon_html, unsafe_allow_html=True)


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
    """Render 4-Stage Pipeline Ribbon + Row 1 (3 Columns) + Row 2 (Full-Width M0 vs M3 Heatmaps)."""
    st.subheader(f"Bảng Soi X-Ray 4 Tầng: {title}")
    _render_pipeline_ribbon(inspection, seed=seed, tau=tau)

    col_a, col_b, col_c = st.columns([1.15, 1.0, 1.15], gap="medium")

    # Col A: Non-duplicated Input -> Output / Wipe
    with col_a:
        st.markdown("#### Cột A · Input → Output")
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

        if inspection.before_after_html is not None and not show_overlay:
            st.caption("Kéo thanh rèm để soi Trước ↔ Sau:")
            components.html(inspection.before_after_html, height=340)
        elif inspection.mode == "sd_text2img":
            noise_img = inspection.overlays.get("latent_noise")
            if noise_img is None and seed is not None:
                noise_img = render_latent_noise_preview(int(seed), size=256)
            in_col, out_col = st.columns(2)
            with in_col:
                if noise_img is not None:
                    st.image(noise_img, caption=f"Latent Noise (seed={seed})", width="stretch")
                else:
                    st.info("Text2Img Latent Noise")
            with out_col:
                st.image(display_img, caption="Output (Text2Img)", width="stretch")
        elif inspection.mode == "sd_img2img":
            in_col, out_col = st.columns(2)
            with in_col:
                if inspection.source_512_rgb is not None:
                    st.image(inspection.source_512_rgb, caption="Input FFHQ Gốc", width="stretch")
            with out_col:
                st.image(display_img, caption="Output (Img2Img)", width="stretch")
        elif inspection.mode == "sd_inpaint":
            in_col, out_col = st.columns(2)
            with in_col:
                src_vis = inspection.overlays.get("source_with_mask", inspection.source_512_rgb)
                if src_vis is not None:
                    st.image(src_vis, caption=f"Input + Mask ({mask_region or 'auto'})", width="stretch")
            with out_col:
                st.image(display_img, caption="Output (Inpaint)", width="stretch")
        else:
            if inspection.mode == "rv51_ood":
                st.info("Unseen RV5.1 OOD — nguồn Kaggle CelebA-HQ không lưu cục bộ.")
            elif inspection.mode == "real_ffhq":
                st.info("Held-out FFHQ Real đối chứng (ground_truth = 0).")
            st.image(display_img, caption=f"Ảnh kiểm định ({inspection.mode})", width="stretch")

        meta_parts = []
        if seed is not None:
            meta_parts.append(f"Seed: `{seed}`")
        if steps is not None:
            meta_parts.append(f"Steps: `{steps}`")
        if guidance_scale is not None:
            meta_parts.append(f"CFG: `{guidance_scale}`")
        if strength is not None and inspection.mode == "sd_img2img":
            meta_parts.append(f"Strength: `{strength}`")
        if meta_parts:
            st.caption(" · ".join(meta_parts))
        if prompt:
            st.caption(f"**Prompt:** {prompt[:220]}")

    # Col B: Semantic Bars for 6 Data Gate Metrics
    with col_b:
        st.markdown("#### Cột B · Phiếu Thẩm Định Data Gate")
        if inspection.gate_passed:
            st.success("✓ ĐẠT CHUẨN — ĐƯỢC CHỌN VÀO DATASET")
        else:
            st.error("✕ BỊ LOẠI TẠI CỔNG DATA")
            reasons_to_show = inspection.rejection_reasons_vi or inspection.rejection_reasons
            for reason in reasons_to_show:
                st.markdown(f"- {reason}")

        for item in inspection.metrics:
            icon = "✓" if item.passed else "✕"
            val_str = (
                f"{item.value:.4f}"
                if item.name == "cheek_fft" and item.value is not None
                else (f"{item.value:.3f}" if item.value is not None else "N/A")
            )
            thr_str = (
                f"{item.threshold:.4f}"
                if item.name == "cheek_fft" and item.threshold is not None
                else (f"{item.threshold:.2f}" if item.threshold is not None else "N/A")
            )
            if item.value is None:
                bar_color = "#64748b"
                right_lbl = "N/A"
            elif item.passed:
                bar_color = "#16a34a"
                right_lbl = f"✓ {val_str} ({item.direction} {thr_str})"
            else:
                bar_color = "#dc2626"
                right_lbl = f"✕ {val_str} ({item.direction} {thr_str})"

            st.markdown(
                _render_semantic_bar_html(
                    _metric_progress_ratio(item),
                    bar_color,
                    f"{icon} {item.label}",
                    right_lbl,
                ),
                unsafe_allow_html=True,
            )
            if item.detail:
                st.caption(item.detail)

        if inspection.warnings:
            for warn in inspection.warnings:
                st.warning(warn)

    # Col C: Inline Force Toggle + Semantic Probability Bars + Verdict
    with col_c:
        st.markdown("#### Cột C · M0 Mining & Detector M3")
        if not inspection.gate_passed:
            st.warning("Ảnh bị loại tại Data Gate.")
            st.checkbox(
                "Ép chạy Detector kể cả khi ảnh bị loại",
                value=bool(st.session_state.get("force_detector_flag", False)),
                key="force_detector_checkbox",
                on_change=_on_force_detector_change,
            )

        if not inspection.detector_scores:
            st.info("Detector chưa chấm điểm cho mẫu này.")
        else:
            m0_prob = float(inspection.detector_scores.get("M0", 0.0))
            m3_prob = float(inspection.detector_scores.get("M3", 0.0))

            if inspection.mode == "real_ffhq" or inspection.ground_truth_label == 0:
                if m3_prob < tau:
                    st.success(f"✓ True Negative (Ảnh thật): M3 P(fake)={m3_prob:.3f} < τ={tau:.2f}")
                else:
                    st.error(f"✕ False Positive (Báo nhầm ảnh thật): M3 P(fake)={m3_prob:.3f} ≥ τ={tau:.2f}")
            elif inspection.is_hard_sample:
                if m3_prob >= tau:
                    st.success(
                        f"🔥 HARD SAMPLE — M0 BỎ LỌT ({m0_prob:.3f} < τ) → M3 CỨU ĐÚNG ({m3_prob:.3f} ≥ τ)"
                    )
                else:
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
                    is_fake = prob >= tau
                    pred_txt = "FAKE" if is_fake else "REAL"
                    bar_color = "#dc2626" if is_fake else "#16a34a"
                    st.markdown(
                        _render_semantic_bar_html(
                            max(0.0, min(1.0, prob)),
                            bar_color,
                            label,
                            f"P(fake)={prob:.4f} → {pred_txt}",
                        ),
                        unsafe_allow_html=True,
                    )

    # Row 2 (Full Width [1, 1]): Enlarged M0 vs M3 DINOv2 Patch Sensitivity Heatmaps
    if "M0" in inspection.heatmaps and "M3" in inspection.heatmaps:
        st.markdown("#### Tầng 4 · Bản đồ nhạy cảm Patch DINOv2 (M0 Baseline vs M3 Champion)")
        hm_l, hm_r = st.columns(2, gap="medium")
        with hm_l:
            st.image(inspection.heatmaps["M0"], caption="M0 (Baseline) Patch Sensitivity", width="stretch")
        with hm_r:
            st.image(inspection.heatmaps["M3"], caption="M3 (Champion) Patch Sensitivity", width="stretch")

    if inspection.latencies_ms:
        lat_parts = [f"{k}: {v:.0f}ms" for k, v in inspection.latencies_ms.items()]
        st.caption("Độ trễ các tầng: " + " · ".join(lat_parts))


def main() -> None:
    """Render Streamlit Studio Workspace v2."""
    st.set_page_config(
        page_title="Topic 11 — Studio v2: Synthetic Facial Dataset & Deepfake Detection",
        layout="wide",
    )

    catalog = get_showcase_catalog()
    presets = get_golden_presets(catalog)
    coursework_metrics = get_coursework_metrics()
    bundle = get_detector_bundle()
    clip_scorer = get_clip_scorer()

    _init_session_state(catalog, presets)
    min_sharpness, min_cheek_fft, min_clip, tau = _render_sidebar(
        catalog, presets, coursework_metrics, bundle
    )

    st.title("Topic 11 — Studio v2: SD 1.5 Data Curation & Active Deepfake Detection")

    tab_heldout, tab_live, tab_upload = st.tabs(
        ["📂 Kho Held-Out", "⚡ Sinh Live & Auto Batch (SD 1.5)", "📤 Upload Ảnh Lạ"]
    )
    with tab_heldout:
        active_entries = _render_heldout_tab(catalog, min_sharpness, min_cheek_fft, min_clip)
    with tab_live:
        bundle = _render_live_and_batch_tab(
            catalog,
            bundle,
            clip_scorer,
            min_sharpness,
            min_cheek_fft,
            min_clip,
            tau,
        )
    with tab_upload:
        _render_upload_tab()

    # Active Source Banner above X-Ray Inspector
    custom = st.session_state.custom_target
    if custom is not None:
        ban_l, ban_r = st.columns([3.5, 1.0])
        with ban_l:
            st.info(f"Đang soi nguồn ngoài kho: **{custom['title']}** (chế độ `{custom['mode']}`)")
        with ban_r:
            if st.button("Quay lại Kho Held-Out", key="btn_back_to_heldout", width="stretch"):
                st.session_state.custom_target = None
                st.rerun()

    st.divider()
    force_det = bool(st.session_state.get("force_detector_flag", False))

    if custom is not None:
        mask_arr = None
        if custom.get("mask_path") is not None and Path(custom["mask_path"]).is_file():
            mask_arr = Image.open(custom["mask_path"]).convert("L")
        inspection = _inspect_with_cache(
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
            run_detector_if_rejected=force_det,
            clip_scorer=clip_scorer,
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
            inspection = _inspect_with_cache(
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
                run_detector_if_rejected=force_det,
                clip_scorer=clip_scorer,
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
