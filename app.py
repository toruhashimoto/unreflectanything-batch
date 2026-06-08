"""ReflectMask for RealityScan — Streamlit GUI.

Mask-first workflow: generate tight binary RealityScan alignment masks (white = kept,
black = excluded reflection) from your photos, using UnReflectAnything as the
reflection-detection backend. Originals are never modified. This is a thin wrapper over
the same engine the CLI uses (src.unreflect_batch.run_batch), so behaviour is identical.

Launch with run_app.bat, or:
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


st.set_page_config(page_title="ReflectMask for RealityScan", page_icon="🎭", layout="wide")
st.title("🎭 ReflectMask for RealityScan")
st.caption(
    "Generate **tight RealityScan alignment masks** that exclude specular reflections / "
    "blown highlights, protecting valid features for high-detail photogrammetry. "
    "UnReflectAnything is the reflection-detection backend. Originals are never modified."
)

# --------------------------------------------------------------------------- #
# Mode (primary control)                                                        #
# --------------------------------------------------------------------------- #
MODE_LABELS = {
    "reflectmask": "🎭 ReflectMask — RealityScan alignment masks (recommended)",
    "diagnostic": "🔬 Diagnostic preview — inspect masks & before/after",
    "clean": "🧪 Cleaned image export (experimental)",
}
mode = st.radio(
    "Mode", list(MODE_LABELS), index=0, format_func=lambda m: MODE_LABELS[m],
)

if mode == "reflectmask":
    st.success(
        "**Output:** a RealityScan-ready folder under `realityscan/` — a byte-exact copy "
        "of each original image + a `‹name›.mask.png` exclusion mask (white = kept, "
        "black = excluded reflection). Import the folder into RealityScan and enable "
        "*“masks for alignment.”* **No cleaned images are written** — the originals stay "
        "the primary input.",
        icon="🎭",
    )
elif mode == "diagnostic":
    st.info(
        "**Output:** the RealityScan masks **plus** before/after previews and diff "
        "heatmaps, so you can inspect what the backend flags before committing to a mask "
        "set. Still mask-first — no cleaned images replace your originals.",
        icon="🔬",
    )
else:
    st.warning(
        "**Experimental:** writes reflection-removed (cleaned) **images** instead of "
        "masks. For high-detail RealityScan alignment, prefer **ReflectMask** — feeding "
        "cleaned images can soften features and usually hurts alignment more than masking. "
        "Originals are still never modified.",
        icon="🧪",
    )

# --------------------------------------------------------------------------- #
# Input / output folders                                                        #
# --------------------------------------------------------------------------- #
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
    output_dir = st.text_input("Output folder", key="output_dir", placeholder=r"D:\rs_reflectmask")

# --------------------------------------------------------------------------- #
# Sidebar — RealityScan mask settings front and centre, the rest folded away    #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("RealityScan mask")
    st.caption("Keep it **tight**: exclude only true reflections, not bright diffuse "
               "surfaces. Over-masking removes valid features and hurts alignment.")
    rs_gate = st.slider("Only mask pixels this bright (original luma)", 0, 255, 250, 1,
                        help="Only original pixels at/above this brightness can be masked "
                             "(0 disables the gate). Tight by default so diffuse-bright "
                             "surfaces (sky, white paint/bodywork) are NOT excluded; lower "
                             "to ~240 for a genuinely glary set.")
    rs_drop = st.slider("Min luma drop to count as reflection", 1, 80, 12, 1,
                        help="A pixel is excluded only if the backend darkened it by at "
                             "least this much.")
    rs_dilation = st.slider("Grow excluded region (px)", 0, 20, 2, 1,
                            help="Covers reflection halos. Keep small — over-masking "
                                 "fragments the reconstruction.")
    rs_open = st.slider("Remove specks radius (px)", 0, 10, 1, 1,
                        help="Morphological open: drops isolated tiny excluded specks.")
    rs_copy_originals = st.checkbox("Copy original images next to masks", value=True,
                                    help="RealityScan must import photos + masks together. "
                                         "Keep on for a self-contained importable folder; "
                                         "off = masks-only.")

    st.divider()
    st.subheader("Run")
    device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    backend = st.selectbox(
        "Detection backend", ["unreflect", "luma"], index=0,
        format_func=lambda b: {"unreflect": "UnReflectAnything (AI, GPU + weights)",
                               "luma": "Pure luma (no GPU / no weights)"}[b],
        help="Pure luma needs no model — it uses the brightness gate above as the luma "
             "level. Use it when you have no GPU/weights, or as an A/B baseline.",
    )
    recursive = st.checkbox("Recurse into sub-folders", value=True)
    overwrite = st.checkbox("Overwrite existing outputs", value=False)
    limit = st.number_input("Test mode: first N images (0 = all)", min_value=0, value=0, step=1)
    exts = st.multiselect("Extensions", list(SUPPORTED_EXTS), default=list(SUPPORTED_EXTS))

    with st.expander("Backend / detection (advanced)"):
        model_max_size = st.number_input("Model input cap px (mask modes; 0 = full res)",
                                         min_value=0, value=2048, step=256,
                                         help="The model is ~448px internally; capping its "
                                              "input avoids wasted full-res I/O (~5x faster on "
                                              "50MP). Mask + original copy stay native, so the "
                                              "RealityScan deliverable is unchanged. 0 = full "
                                              "resolution (slow). No effect in Cleaned-export mode.")
        threshold = st.slider("Highlight threshold (model)", 0.0, 1.0, 0.3, 0.05)
        dilation = st.slider("Highlight mask dilation (px, model)", 0, 120, 40, 5)
        max_size = st.number_input("Quick mode: downscale longest side px (0 = off)",
                                   min_value=0, value=0, step=128,
                                   help="Changes output dimensions — for quick tests only, "
                                        "not real RealityScan/COLMAP input.")
        use_exiftool = st.checkbox("Full metadata copy (exiftool)", value=False,
                                   help="Copy ALL metadata via exiftool if installed; "
                                        "otherwise fast piexif/PIL EXIF.")

    with st.expander("Cleaned image (only used in Cleaned-export mode)"):
        composite = st.checkbox("Composite (model, ~448px)", value=False,
                                help="Model's internal composite; whole image still softened on resize-back.")
        mask_composite = st.checkbox("Full-res composite (keep detail outside highlights)", value=False,
                                     help="Only blown highlights are replaced; rest stays full-res.")
        mask_level = st.slider("↳ mask level (higher = tighter, less blur)", 200, 255, 248, 1)
        mask_dilation = st.slider("↳ mask grow px (keep small)", 0, 40, 0, 1)
        jpeg_quality = st.slider("JPEG quality", 80, 100, 95, 1)

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

run = st.button("▶ Run", type="primary", use_container_width=True)

# --------------------------------------------------------------------------- #
# Run                                                                           #
# --------------------------------------------------------------------------- #
if run:
    if not input_dir or not output_dir:
        st.error("Please set both an input and an output folder.")
        st.stop()

    # Mask-first defaults (realityscan / previews / no cleaned image) are resolved by
    # BatchConfig.__post_init__ from the mode; pass the raw toggles here.
    cfg = BatchConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        recursive=recursive,
        exts=tuple(exts) if exts else SUPPORTED_EXTS,
        device=device,
        mode=mode,
        backend=backend,
        overwrite=overwrite,
        make_preview=False,
        heatmap=False,
        limit=int(limit) or None,
        max_size=int(max_size) or None,
        model_max_size=int(model_max_size) or None,
        jpeg_quality=int(jpeg_quality),
        threshold=float(threshold),
        dilation=int(dilation),
        composite=composite,
        mask_composite=mask_composite,
        mask_composite_level=float(mask_level),
        mask_composite_dilation=int(mask_dilation),
        realityscan=False,
        rs_copy_originals=rs_copy_originals,
        rs_drop_level=float(rs_drop),
        rs_highlight_gate=float(rs_gate),
        rs_dilation=int(rs_dilation),
        rs_open=int(rs_open),
        use_exiftool=use_exiftool,
    )

    # Backend B (luma) needs no model / weights / GPU; only the AI backend loads it.
    if backend == "luma":
        model = None
        dev_resolved, dev_note = "cpu", "pure-luma backend (no model)"
    else:
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
    st.success(f"Mode: **{summary.get('mode')}**  •  Output: {summary.get('output_dir')}  •  logs in /logs")

    # RealityScan masks: the primary product in the mask-first modes.
    if cfg.realityscan:
        rs_dir = Path(output_dir) / "realityscan"
        st.subheader("RealityScan alignment masks")
        st.success(f"Mask folder: {rs_dir}")
        mean_excl = summary.get("realityscan_mean_excluded_pct")
        if mean_excl is not None:
            st.metric("Avg pixels excluded", f"{mean_excl:.2f}%")
            warn_n = summary.get("realityscan_warn_images", 0)
            danger_n = summary.get("realityscan_danger_images", 0)
            if warn_n or danger_n:
                st.caption(f"Per-image mask-area flags: **{warn_n}** warning, **{danger_n}** danger.")
            if mean_excl > 12:
                st.error(
                    f"⛔ Excluding {mean_excl:.1f}% of pixels on average — likely "
                    "over-masking diffuse-bright areas (sky, white paint/bodywork), not "
                    "just reflections. Raise the brightness gate, or this set may simply "
                    "not need masking (originals often reconstruct best for mild glare).",
                    icon="⛔",
                )
            elif mean_excl > 5:
                st.warning(
                    f"⚠️ Excluding {mean_excl:.1f}% of pixels on average — getting high. "
                    "Check the masks aren't catching bright diffuse surfaces; raise the "
                    "brightness gate if so.",
                    icon="⚠️",
                )
        if rs_copy_originals:
            st.markdown(
                f"**How to use in RealityScan** (white = kept, black = excluded reflection):\n"
                f"1. **WORKFLOW → Inputs → Folder** and pick `{rs_dir}` "
                f"(loads photos **and** `.mask.png` masks together so they auto-attach).\n"
                f"2. Select the images → **Selected Input → Image Layers** → tick "
                f"**“Enable masks for alignment.”**\n"
                f"3. Run alignment. (Originals were copied unmodified; your input folder is untouched.)"
            )
        else:
            st.markdown(
                f"**Masks-only output** (white = kept, black = excluded). This folder has "
                f"the `.mask.png` files but **no images**, so it is not directly importable:\n"
                f"1. Merge each `‹name›.mask.png` into the **same folder as its photo**.\n"
                f"2. Import photos **and** masks together (**WORKFLOW → Inputs → Folder**).\n"
                f"3. Select the images → **Image Layers** → **“Enable masks for alignment”** → align."
            )
        rs_masks = sorted(rs_dir.rglob("*.mask.png"))[:4]
        if rs_masks:
            st.caption("Sample masks (black = reflection excluded):")
            for p in rs_masks:
                st.image(str(p), caption=p.relative_to(rs_dir).as_posix(), use_container_width=True)

    # Diagnostic / cleaned previews.
    prev_dir = Path(output_dir) / "preview_compare"
    if prev_dir.exists():
        previews = sorted(prev_dir.rglob("*.jpg"))[:6]
        if previews:
            st.subheader("Before / after samples")
            for p in previews:
                st.image(str(p), caption=p.relative_to(prev_dir).as_posix(), use_container_width=True)

    if mode == "clean":
        st.info(f"Cleaned images written under {Path(output_dir)} (mirroring the input tree).")

    errs = Path(output_dir) / "logs" / "errors.csv"
    if summary.get("errors", 0) and errs.exists():
        st.subheader("Errors")
        st.code(errs.read_text(encoding="utf-8"))
