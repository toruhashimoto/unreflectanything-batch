"""UnReflect Batch — simple Streamlit GUI (Phase 2).

A thin wrapper over the same engine the CLI uses (src.unreflect_batch.run_batch),
so behaviour is identical. Launch with run_app.bat, or:
    .venv\\Scripts\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.image_io import SUPPORTED_EXTS  # noqa: E402
from src.unreflect_batch import (  # noqa: E402
    BatchConfig, run_batch, resolve_device, load_model,
    WeightsMissingError, ModelLoadError, weights_status, download_weights,
)


@st.cache_resource(show_spinner=False)
def _cached_model(device_resolved: str):
    """Load the UnReflectModel once per resolved device and reuse across reruns
    (avoids reloading the 3.44 GB checkpoint every time you press Run)."""
    return load_model(device_resolved)

st.set_page_config(page_title="UnReflect Batch", page_icon="🪞", layout="wide")
st.title("🪞 UnReflect Batch")
st.caption(
    "Specular-reflection / highlight removal for 3DGS & photogrammetry input photos. "
    "Originals are never modified; output goes to a separate folder."
)
st.info(
    "Evaluation-only: single-image reflection removal has **no multi-view consistency "
    "guarantee**. Use it to *improve a problematic photo set*, and A/B test your SfM/3DGS "
    "with vs. without the cleaned images before trusting it.",
    icon="⚠️",
)


def _browse() -> str | None:
    """Open a native folder picker (works for the locally-run server)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory()
        root.destroy()
        return path or None
    except Exception as e:  # noqa: BLE001
        st.warning(f"Folder dialog unavailable ({e}); paste the path instead.")
        return None


c1, c2 = st.columns(2)
with c1:
    if st.button("📂 Browse input…"):
        p = _browse()
        if p:
            st.session_state["input_dir"] = p
    input_dir = st.text_input("Input folder", key="input_dir", placeholder=r"D:\photo_input")
with c2:
    if st.button("📂 Browse output…"):
        p = _browse()
        if p:
            st.session_state["output_dir"] = p
    output_dir = st.text_input("Output folder", key="output_dir", placeholder=r"D:\photo_unreflect")

with st.sidebar:
    st.header("Options")
    device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    recursive = st.checkbox("Recurse into sub-folders", value=True)
    make_preview = st.checkbox("Save before/after previews", value=True)
    heatmap = st.checkbox("Save diff heatmaps", value=False)
    emit_mask = st.checkbox("Save change masks (COLMAP exclusion)", value=False)
    composite = st.checkbox("Composite (model, ~448px)", value=False,
                            help="Model's internal composite; whole image still softened on resize-back.")
    mask_composite = st.checkbox("Full-res composite (keep detail outside highlights)", value=False,
                                 help="Best for high-res input: only blown highlights are replaced; rest stays full-res.")
    use_exiftool = st.checkbox("Full metadata copy (exiftool)", value=False,
                               help="Copy ALL metadata via exiftool if installed; otherwise fast piexif/PIL EXIF.")
    overwrite = st.checkbox("Overwrite existing outputs", value=False)
    st.divider()
    limit = st.number_input("Test mode: first N images (0 = all)", min_value=0, value=0, step=1)
    max_size = st.number_input("Quick mode: downscale longest side px (0 = off)", min_value=0, value=0, step=128,
                               help="Changes output dimensions — for quick tests only, not COLMAP input.")
    st.divider()
    threshold = st.slider("Highlight threshold", 0.0, 1.0, 0.3, 0.05)
    dilation = st.slider("Mask dilation (px)", 0, 120, 40, 5)
    jpeg_quality = st.slider("JPEG quality", 80, 100, 95, 1)
    exts = st.multiselect("Extensions", list(SUPPORTED_EXTS), default=list(SUPPORTED_EXTS))

    st.divider()
    with st.expander("Model weights (~5.9 GB, required once)"):
        if st.button("⬇ Download model weights"):
            with st.spinner("Downloading weights (one time, ~5.9 GB)…"):
                try:
                    wdir = download_weights(progress=False)
                    st.success(f"Weights ready: {wdir}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Download failed: {e}")
        st.caption("Or run `unreflectanything download --weights` in a terminal. "
                   "There is no automatic download.")

run = st.button("▶ Run batch", type="primary", use_container_width=True)

if run:
    if not input_dir or not output_dir:
        st.error("Please set both an input and an output folder.")
        st.stop()

    cfg = BatchConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        recursive=recursive,
        exts=tuple(exts) if exts else SUPPORTED_EXTS,
        device=device,
        overwrite=overwrite,
        make_preview=make_preview,
        heatmap=heatmap,
        emit_mask=emit_mask,
        limit=int(limit) or None,
        max_size=int(max_size) or None,
        jpeg_quality=int(jpeg_quality),
        threshold=float(threshold),
        dilation=int(dilation),
        composite=composite,
        mask_composite=mask_composite,
        use_exiftool=use_exiftool,
    )

    # Resolve device and load (cached) model once per session.
    dev_resolved, dev_note = resolve_device(device)
    try:
        with st.spinner(f"Loading model on {dev_resolved} (cached after first run)…"):
            model = _cached_model(dev_resolved)
    except (WeightsMissingError, ModelLoadError) as e:
        st.error(f"Setup problem:\n\n{e}")
        st.stop()

    progress = st.progress(0.0, text="Starting…")

    def on_progress(i: int, total: int, rec: dict):
        progress.progress(i / max(1, total), text=f"{i}/{total} — {Path(rec['source']).name} [{rec['status']}]")

    try:
        with st.spinner(f"Processing on {dev_resolved} ({dev_note})…"):
            summary = run_batch(cfg, progress=False, on_progress=on_progress, model=model)
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()

    progress.progress(1.0, text="Done")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Processed", summary.get("processed_ok", 0))
    m2.metric("Skipped", summary.get("skipped", 0))
    m3.metric("Errors", summary.get("errors", 0))
    m4.metric("Device", summary.get("device", "?"))
    st.success(f"Output: {summary.get('output_dir')}  •  logs in /logs")

    # Show a few previews if available.
    prev_dir = Path(output_dir) / "preview_compare"
    if prev_dir.exists():
        previews = sorted(prev_dir.rglob("*.jpg"))[:6]
        if previews:
            st.subheader("Before / after samples")
            for p in previews:
                st.image(str(p), caption=p.relative_to(prev_dir).as_posix(), use_container_width=True)

    errs = Path(output_dir) / "logs" / "errors.csv"
    if summary.get("errors", 0) and errs.exists():
        st.subheader("Errors")
        st.code(errs.read_text(encoding="utf-8"))
