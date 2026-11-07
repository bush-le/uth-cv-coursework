from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
DECK_PATH = REPO_ROOT / "docs/reports/presentation_5min_define_and_design.html"


def _extract_visible_text(html: str) -> str:
    no_style = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    no_script = re.sub(r"<script[\s\S]*?</script>", " ", no_style, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", no_script)
    return re.sub(r"\s+", " ", text).strip()


def test_deck_has_eight_slides_and_fixed_stage():
    html = DECK_PATH.read_text(encoding="utf-8")
    assert "width: 1920px;" in html and "height: 1080px;" in html
    slides = re.findall(r'<section[^>]+class="[^"]*slide[^"]*"[^>]*>', html)
    assert len(slides) == 8
    for idx in range(1, 9):
        assert f'id="slide-{idx}"' in html


def test_deck_enforces_blue_professional_single_accent_palette():
    html = DECK_PATH.read_text(encoding="utf-8").lower()
    assert "#fdfae7" in html
    assert "#1e2bfa" in html
    for forbidden_hex in ["#d97706", "#dc2626", "#059669", "#9333ea", "#1e293b", "#111827"]:
        assert forbidden_hex not in html, f"Forbidden non-Blue-Professional color {forbidden_hex}"
    assert "notes-drawer" not in html
    assert "<pre" not in html


def test_deck_contains_zero_unverified_numbers_or_code_clutter():
    html = DECK_PATH.read_text(encoding="utf-8")
    visible = _extract_visible_text(html)
    forbidden_patterns = [
        r"\b95%\b",
        r"\b62%\b",
        r"\b92%\b",
        r"\b85%\b",
        r"\b50\.0\b",
        r"&ge;\s*50",
        r"≥\s*50",
        r"\b0\.28\b",
        r"\b0\.30\b",
        r"\b0\.85\b",
        r"\b78\.4\b",
        r"\b0\.312\b",
        r"\b3,000\b",
        r"\b5,000\b",
        r"\b256\b",
        r"\b512\b",
        r"\b70%\b",
        r"\b15%\b",
        r"\b6GB\b",
        r"\b50 steps\b",
        r"\b30 steps\b",
        r"\bSHA-256\b",
        r"\bFP16\b",
        r"\b1e-4\b",
        r"\bsrc/",
        r"\bdata/",
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, visible, flags=re.I), f"Forbidden unverified detail matched: {pat}"


def test_every_slide_has_inline_svg_visual_diagram():
    html = DECK_PATH.read_text(encoding="utf-8")
    parts = re.split(r'<section[^>]+class="[^"]*slide[^"]*"[^>]*>', html)[1:]
    assert len(parts) == 8
    for idx, slide_html in enumerate(parts, start=1):
        assert "<svg" in slide_html, f"Slide {idx} is missing an inline <svg> visual diagram"


def test_combines_all_three_visual_approaches_across_eight_slides():
    html = DECK_PATH.read_text(encoding="utf-8")
    # Approach 1: Domain-Specific CV Vector Diagrams
    for cv_term in [
        "Decision Boundary",
        "Text-to-Image",
        "Facial Inpainting",
        "Image-to-Image",
        "Zero Identity Leakage",
        "Data Acquisition",
        "Face Preprocessing",
        "Stable Diffusion Generation",
        "Quality Control",
        "Labeling &amp; Metadata",
        "Dataset Export",
        "ROC-AUC",
    ]:
        assert cv_term in html, f"Missing Approach 1 CV visual element: {cv_term}"

    # Approach 2: Embedded Illustrative Face Portrait Thumbnails & Overlays
    assert "MINH HỌA Ý TƯỞNG" in html
    assert 'id="face-portrait-base"' in html or 'class="portrait-thumb"' in html

    # Approach 3: Animated Data-Flow & Visual Scanning Pulses
    for anim_name in ["@keyframes flowPulse", "@keyframes scanLine", "@keyframes pulseNode"]:
        assert anim_name in html, f"Missing Approach 3 animation: {anim_name}"

    # Group 4 Attribution on Slide 1, Slide 4, and Slide 5
    slide_1_part = html.split('id="slide-1"')[1].split('id="slide-2"')[0]
    slide_4_part = html.split('id="slide-4"')[1].split('id="slide-5"')[0]
    slide_5_part = html.split('id="slide-5"')[1].split('id="slide-6"')[0]
    assert "Nhóm thực hiện: Nhóm 4" in slide_1_part
    assert "Nhóm thực hiện: Nhóm 4" in slide_4_part
    assert "Nhóm thực hiện: Nhóm 4" in slide_5_part


def test_companion_markdown_guides_contain_zero_unverified_numbers_and_eight_slides():
    for rel_path in [
        "docs/reports/PRESENTATION_5MIN_DEFINE_AND_DESIGN.md",
        "docs/reports/SLIDE_PRESENTATION_STRUCTURE_5MIN.md",
    ]:
        content = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert "Slide 07" in content and "Slide 08" in content, f"{rel_path} must document all 8 slides"
        for pat in [
            r"\b95%\b",
            r"\b62%\b",
            r"\b92%\b",
            r"\b85%\b",
            r"\b0\.28\b",
            r"\b0\.85\b",
            r"\b78\.4\b",
            r"\b3,000\b",
            r"\b5,000\b",
            r"\b256\b",
            r"\b512\b",
            r"\b6GB\b",
        ]:
            assert not re.search(pat, content, flags=re.I), f"Forbidden number {pat} in {rel_path}"


def test_svg_text_elements_respect_min_font_size_and_no_overflow():
    html = DECK_PATH.read_text(encoding="utf-8")
    svgs = re.findall(r"<svg[\s\S]*?</svg>", html, flags=re.I)
    assert len(svgs) >= 8
    for svg in svgs:
        for size_str in re.findall(r'font-size="([0-9.]+)"', svg):
            assert float(size_str) >= 13.0, f"SVG text font-size {size_str} is below 13px minimum"
        for text_content in re.findall(r"<text[^>]*>(.*?)</text>", svg):
            clean = re.sub(r"&[a-zA-Z0-9#]+;", "X", text_content).strip()
            assert len(clean) <= 58, f"SVG <text> too long ({len(clean)} chars), risks clipping: {clean}"


def test_chrome_controls_are_anchored_inside_deck_stage():
    html = DECK_PATH.read_text(encoding="utf-8")
    stage_split = html.split('id="deck-stage"')[1]
    assert 'id="slide-counter"' in stage_split
    assert 'class="nav-controls"' in stage_split
    assert 'id="progress-bar"' in stage_split
    assert "margin-top: auto" in html


def test_rendered_dom_and_svg_geometry_in_headless_chrome():
    import base64
    import json
    import os
    import shutil
    import socket
    import struct
    import subprocess
    import time
    import urllib.request

    if not shutil.which("google-chrome"):
        return

    port = 9246
    proc = subprocess.Popen(
        [
            "google-chrome",
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--remote-debugging-port={port}",
            "--window-size=1920,1080",
            f"file://{DECK_PATH}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1.8)
        targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read().decode())
        ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page")
        rest = ws_url.split("ws://")[1]
        host_port, path = rest.split("/", 1)
        host, p = host_port.split(":")
        sock = socket.create_connection((host, int(p)), timeout=15)
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall(
            (
                f"GET /{path} HTTP/1.1\r\nHost: {host_port}\r\n"
                f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += sock.recv(4096)

        def cdp_call(msg_id: int, method: str, params: dict | None = None) -> dict:
            payload = json.dumps({"id": msg_id, "method": method, "params": params or {}}).encode()
            header = bytearray([0x81])
            n = len(payload)
            if n < 126:
                header.append(0x80 | n)
            elif n < 65536:
                header.append(0x80 | 126)
                header.extend(struct.pack("!H", n))
            else:
                header.append(0x80 | 127)
                header.extend(struct.pack("!Q", n))
            mask = os.urandom(4)
            header.extend(mask)
            sock.sendall(header + bytearray(b ^ mask[i % 4] for i, b in enumerate(payload)))

            def recv_exact(k: int) -> bytes:
                buf = b""
                while len(buf) < k:
                    buf += sock.recv(k - len(buf))
                return buf

            while True:
                b1, b2 = recv_exact(2)
                length = b2 & 0x7F
                if length == 126:
                    length = struct.unpack("!H", recv_exact(2))[0]
                elif length == 127:
                    length = struct.unpack("!Q", recv_exact(8))[0]
                raw = recv_exact(length)
                data = json.loads(raw.decode("utf-8", errors="replace"))
                if data.get("id") == msg_id:
                    return data

        cdp_call(1, "Runtime.enable")
        cdp_call(
            2,
            "Emulation.setDeviceMetricsOverride",
            {"width": 1920, "height": 1080, "deviceScaleFactor": 1, "mobile": False},
        )
        time.sleep(0.5)

        js_check = r"""
        (function() {
          const slides = Array.from(document.querySelectorAll('.slide'));
          const errors = [];
          if (slides.length !== 8) {
            errors.push(`Expected 8 slides, found ${slides.length}`);
          }
          slides.forEach((slide, idx) => {
            slides.forEach((s, i) => s.classList.toggle('active', i === idx));
            const title = slide.querySelector('.slide-title');
            if (title && title.clientHeight > 54) {
              errors.push(`${slide.id}: .slide-title wrapped to 2 lines (${title.clientHeight}px)`);
            }
            slide.querySelectorAll('.card-grid-4 .intent-card h3').forEach(h3 => {
              if (h3.clientHeight > 30) {
                errors.push(`${slide.id}: .card-grid-4 h3 wrapped to 2 lines (${h3.innerText})`);
              }
            });
            slide.querySelectorAll('.svg-stage').forEach(stage => {
              const sr = stage.getBoundingClientRect();
              const svg = stage.querySelector('svg');
              if (svg && svg.viewBox.baseVal.width > 0) {
                const vb = svg.viewBox.baseVal;
                const mismatch = Math.abs((sr.width / sr.height) - (vb.width / vb.height)) / (vb.width / vb.height);
                if (mismatch > 0.22) {
                  errors.push(`${slide.id}: SVG aspect mismatch ${Math.round(mismatch * 100)}%`);
                }
              }
            });
            slide.querySelectorAll('svg').forEach(svg => {
              const rects = Array.from(svg.querySelectorAll('rect')).filter(r => !r.classList.contains('decor-rect')).map(r => {
                const b = r.getBBox();
                const m = svg.getCTM().inverse().multiply(r.getCTM());
                return { x: b.x + m.e, y: b.y + m.f, w: b.width, h: b.height, right: b.x + m.e + b.width, bottom: b.y + m.f + b.height };
              });
              svg.querySelectorAll('text').forEach(t => {
                const b = t.getBBox();
                const m = svg.getCTM().inverse().multiply(t.getCTM());
                const tx = b.x + m.e, ty = b.y + m.f, tr = tx + b.width, tb = ty + b.height;
                const tcx = (tx + tr) / 2, tcy = (ty + tb) / 2;
                const cands = rects.filter(r => r.w >= 45 && r.h >= 20 && tcx >= r.x && tcx <= r.right && tcy >= r.y && tcy <= r.bottom).sort((a, b) => (a.w * a.h) - (b.w * b.h));
                if (cands.length > 0) {
                  const enc = cands[0];
                  if (tx < enc.x + 2 || tr > enc.right - 2) {
                    errors.push(`${slide.id}: SVG text '${t.textContent.trim()}' (${Math.round(b.width)}px) overflows rect (${Math.round(enc.w)}px)`);
                  }
                }
              });
            });
          });
          return errors;
        })()
        """
        res = cdp_call(3, "Runtime.evaluate", {"expression": js_check, "returnByValue": True})
        errors = res["result"]["result"]["value"]
        assert errors == [], f"Presentation layout bugs detected: {errors}"
    finally:
        proc.terminate()
