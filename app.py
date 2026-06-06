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
    mask_level = st.slider("  ↳ mask level (higher = tighter, less blur)", 200, 255, 248, 1,
                           help="Only pixels brighter than this (luma 0-255) are replaced. Raise if the result looks blurry.")
    mask_dilation = st.slider("  ↳ mask grow px (keep small)", 0, 40, 0, 1,
                              help="Grows the replaced area. Large values blur the subject — keep at 0-2.")
    st.divider()
    realityscan = st.checkbox("RealityScan alignment masks", value=False,
                              help="Emit a ready-to-import folder under realityscan/: a copy of each ORIGINAL image + a '<name>.mask.png' exclusion mask (black=removed reflection=excluded). Best as an alignment pre-process — the originals stay untouched and only reflections are ignored.")
    rs_copy_originals = st.checkbox("  ↳ copy original images next to masks", value=True,
                                    help="RealityScan must import photos + masks together. Keep on to get a self-contained importable folder; turn off for masks-only.")
    rs_gate = st.slider("  ↳ only mask pixels this bright (orig luma)", 0, 255, 250, 1,
                        help="Only original pixels at/above this brightness can be masked (0 disables the gate). Tight by default so diffuse-bright surfaces (sky, white paint/bodywork) are NOT excluded; lower to ~240 for a genuinely glary set.")
    rs_drop = st.slider("  ↳ min luma drop to count as reflection", 1, 80, 12, 1,
                        help="A pixel is excluded only if the model darkened it by at least this much.")
    rs_dilation = st.slider("  ↳ grow excluded region px", 0, 20, 2, 1,
                            help="Covers reflection halos. Keep small — over-masking fragments the reconstruction.")
    rs_open = st.slider("  ↳ remove specks radius", 0, 10, 1, 1,
                        help="Morphological open: drops isolated tiny excluded specks.")
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
        mask_composite_level=float(mask_level),
        mask_composite_dilation=int(mask_dilation),
        realityscan=realityscan,
        rs_copy_originals=rs_copy_originals,
        rs_drop_level=float(rs_drop),
        rs_highlight_gate=float(rs_gate),
        rs_dilation=int(rs_dilation),
        rs_open=int(rs_open),
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

    # RealityScan masks: show where they went + how to use them, plus samples.
    if realityscan:
        rs_dir = Path(output_dir) / "realityscan"
        st.subheader("RealityScan alignment masks")
        st.success(f"Mask folder: {rs_dir}")
        mean_excl = summary.get("realityscan_mean_excluded_pct")
        if mean_excl is not None:
            st.metric("Avg pixels excluded", f"{mean_excl:.2f}%")
            if mean_excl > 12:
                st.warning(
                    f"Excluding {mean_excl:.1f}% of pixels on average — likely over-masking "
                    "diffuse-bright areas (sky, white paint/bodywork), not just reflections. "
                    "Raise the brightness gate, or this set may simply not need masking "
                    "(originals often reconstruct best for mild glare).",
                    icon="⚠️",
                )
        if rs_copy_originals:
            st.markdown(
                f"**How to use in RealityScan** (masks exclude reflections from alignment; "
                f"black = excluded, white = kept):\n"
                f"1. In RealityScan, **WORKFLOW → Inputs → Folder** and pick `{rs_dir}` "
                f"(this loads the photos **and** the `.mask.png` masks together so they auto-attach).\n"
                f"2. Select the images → **Selected Input → Image Layers** → tick "
                f"**“Enable masks for alignment.”**\n"
                f"3. Run alignment. (Originals were copied unmodified; your input folder is untouched.)"
            )
        else:
            st.markdown(
                f"**Masks-only output** (black = excluded, white = kept). This folder has the "
                f"`.mask.png` files but **no images**, so it is not directly importable:\n"
                f"1. Merge each `‹name›.mask.png` into the **same folder as its photo**.\n"
                f"2. Import the photos **and** masks together (**WORKFLOW → Inputs → Folder**).\n"
                f"3. Select the images → **Selected Input → Image Layers** → tick "
                f"**“Enable masks for alignment”** → run alignment."
            )
        rs_masks = sorted(rs_dir.rglob("*.mask.png"))[:4]
        if rs_masks:
            st.caption("Sample masks (black = reflection excluded):")
            for p in rs_masks:
                st.image(str(p), caption=p.relative_to(rs_dir).as_posix(), use_container_width=True)

    errs = Path(output_dir) / "logs" / "errors.csv"
    if summary.get("errors", 0) and errs.exists():
        st.subheader("Errors")
        st.code(errs.read_text(encoding="utf-8"))
