# UnReflect Batch

[![CI](https://github.com/toruhashimoto/unreflectanything-batch/actions/workflows/ci.yml/badge.svg)](https://github.com/toruhashimoto/unreflectanything-batch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Upstream: UnReflectAnything](https://img.shields.io/badge/upstream-UnReflectAnything-blue)](https://github.com/alberto-rota/UnReflectAnything)
[![3DGS: LichtFeld Studio](https://img.shields.io/badge/3DGS-LichtFeld%20Studio-orange)](https://github.com/MrNeRF/LichtFeld-Studio)

**English** · [日本語 (Japanese)](README.ja.md)

> **Independent wrapper.** This project is an independent batch wrapper for
> [UnReflectAnything](https://github.com/alberto-rota/UnReflectAnything) — it is **not
> affiliated with or endorsed by** the original authors, and bundles **none** of their
> code (the model is installed from PyPI and called via its public API/CLI).

Batch-remove **specular reflections / blown-out highlights** from input photos using
[**UnReflectAnything**](https://alberto-rota.github.io/UnReflectAnything/), as a
**pre-processing step for 3D Gaussian Splatting (3DGS) and photogrammetry**
(RealityScan, Postshot, Nerfstudio, COLMAP, …).

It is a thin, safe **wrapper** around the `unreflectanything` package — it does not
modify the research code. Originals are never touched; cleaned images plus
before/after previews, diff heatmaps and per-image logs are written to a separate
output folder, with **file names and sub-folder structure preserved** so the result
drops straight into your existing SfM/3DGS pipeline.

> ⚠️ **Evaluation-only.** Single-image reflection removal has **no multi-view
> consistency guarantee** — the network can inpaint specular regions differently in
> different views, which *can hurt* SfM feature matching. Treat the output as a tool
> to **improve a problematic photo set for visualization/quality experiments**, not as
> measurement ground truth. Always A/B test your reconstruction *with vs. without* the
> cleaned images. See [Recommended workflow](#recommended-workflow-for-3dgs--photogrammetry).

---

## Demo

![before / after / diff](examples/demo_before_after.jpg)

*A synthetic scene with specular glare → `--mask-composite` → diff heatmap. Originals are
never modified; only blown-highlight regions change, and the heatmap shows exactly what.
More (and how to reproduce) in [`examples/`](examples/).*

---

## 1. Requirements

| | |
|---|---|
| OS | Windows 10/11 (developed & verified on Windows 11) |
| Python | **3.11+** (3.11 verified; the engine pins exact deps) |
| GPU (optional) | NVIDIA CUDA GPU. **RTX 50-series (Blackwell/sm_120) verified** (RTX 5070 Ti). CPU fallback works but is slow. |
| Disk | ~3 GB (PyTorch) + **~5.9 GB model weights** + your images |

No GPU? It still runs on CPU automatically — just slower.

---

## 2. Setup

### Easiest — let the launcher do everything
Double-click **`run_app.bat`** (GUI) or **`run_batch_example.bat`** (CLI). On first run
it creates the virtual environment, installs everything, and downloads the weights.

### Or run the setup script directly
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Gui
```
`-Gui` also installs Streamlit. Add `-SkipWeights` to defer the 5.9 GB download, or
`-CudaIndex https://download.pytorch.org/whl/cu130` to match a very new driver.

### Or do it manually (and understand the two gotchas)
```powershell
# 1) Create & activate a venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# 2) TRAP #1 — install CUDA PyTorch FIRST from the PyTorch index.
#    The default PyPI `torch` wheel on Windows is CPU-ONLY, and Blackwell (sm_120)
#    needs the cu128 build.
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# 3) TRAP #2 — requirements.txt pins `transformers` to the exact commit the
#    checkpoint needs. A plain `pip install unreflectanything` pulls the latest
#    transformers, whose DINOv3 keys don't match -> inference fails with
#    "Error(s) in loading state_dict ... Missing key(s) ... dinov3".
pip install -r requirements.txt
```

### Verify the GPU (optional)
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), 'sm_120' in torch.cuda.get_arch_list())"
# expect:  2.9.1+cu128 True True
```

---

## 3. Download the model weights (required, one time)

There is **no auto-download**; you must fetch the weights once (~5.9 GB):
```powershell
.\.venv\Scripts\unreflectanything.exe download --weights
.\.venv\Scripts\unreflectanything.exe verify --weights   # optional sanity check
```
Weights are cached under `%LOCALAPPDATA%\unreflectanything\weights` and reused by every
run. (The setup script does this for you unless you passed `-SkipWeights`.)

You can also let the tools fetch them: pass **`--download-weights`** to `main.py`, or click
**"Download model weights"** in the GUI sidebar. If weights are missing, the tools fail
fast with a clear message pointing at the exact command and cache location (no silent
mid-run crash).

---

## 4. Usage — GUI

```powershell
run_app.bat
```
Pick an input and output folder, tweak options in the sidebar, press **Run batch**.
You get live progress, a summary, and before/after samples. The GUI calls the exact
same engine as the CLI.

## 5. Usage — CLI

### Batch (the main tool)
```powershell
python main.py --input "D:\photo_input" --output "D:\photo_unreflect" --recursive --make-preview --device cuda
```

Full options:

| Flag | Default | Description |
|---|---|---|
| `--input, -i` | — | Input image folder (**required**) |
| `--output, -o` | — | Output folder, **must be outside input** (**required**) |
| `--recursive, -r` | off | Recurse into sub-folders (structure mirrored in output) |
| `--device, -d` | `auto` | `auto` \| `cuda` \| `cpu` (auto = GPU if a *working* one is present) |
| `--extensions` | `.jpg,.jpeg,.png,.tif,.tiff` | Comma-separated extensions to process |
| `--make-preview` | off | Side-by-side before/after into `preview_compare/` |
| `--heatmap` | off | Per-image luma-difference heatmaps into `heatmap/` |
| `--emit-mask` | off | Approx. changed-region masks into `masks/` (COLMAP exclusion masks) |
| `--overwrite` | off | Replace existing outputs (default: **skip**, never overwrite) |
| `--jpeg-quality` | `95` | JPEG quality (min 95 enforced, 4:4:4) |
| `--threshold` | `0.3` | Highlight-detection threshold (model) |
| `--dilation` | `40` | Highlight-mask dilation in px (model) |
| `--composite` | off | Model's internal composite: blends diffuse into highlight regions at the model's ~448 px resolution (whole image is still softened on resize-back) |
| `--mask-composite` | off | **Wrapper full-res composite**: keeps the original full-resolution pixels everywhere except blown highlights — **best for high-resolution SfM/3DGS input** |
| `--exiftool` | off | Copy **all** metadata via exiftool when available (maker notes/GPS/XMP, all formats; slower, per-file). Default = fast piexif/PIL EXIF |
| `--verbose` | off | Show the engine's own per-image output |
| `--limit N` | — | **Test mode**: process only the first N images |
| `--max-size PX` | — | **Quick mode**: downscale longest side before processing (⚠ changes output dims — not for COLMAP input) |
| `--download-weights` | off | Download the ~5.9 GB weights first if missing, then run |
| `--dry-run` | off | List what would be processed, run nothing |
| `--no-progress` | off | Disable the progress bar |

### Single image (straight from the engine, no wrapper needed)
```powershell
.\.venv\Scripts\unreflectanything.exe inference "in.jpg" -o "out.jpg" -d cuda
```

---

## 6. Output structure

```
<output>/
├── <original tree, original filenames>   # cleaned images (format/size/EXIF preserved)
│   ├── img001.jpg
│   └── day2/img777.png
├── preview_compare/                      # [Original | UnReflect | (Diff)] strips  (--make-preview)
├── heatmap/                              # luma-difference heatmaps               (--heatmap)
├── masks/                                # changed-region masks                   (--emit-mask)
└── logs/
    ├── process_log.jsonl                 # one detailed JSON record per image
    ├── process_log.csv                   # flat summary table
    ├── errors.csv                        # failed images only
    └── run_summary.json                  # run totals + full config
```

Every record carries `processed_by: "UnReflectAnything"`, plus the source reference,
timestamp, model name/version, device, input/output size, processing time, the
parameters used, the evaluation metrics, and any error.

---

## 7. Evaluation features

- **Mean-luminance delta** (before→after) — how much overall brightness dropped.
- **Highlight-pixel ratio** (before/after) — fraction of near-clipped pixels; the core
  signal for "did it remove blown highlights".
- **Diff heatmap** (`--heatmap`) — where, and how strongly, the image was altered.
- **Change mask** (`--emit-mask`) — a binary mask of altered regions you can hand to
  COLMAP to **exclude** those pixels from feature matching (safer than trusting
  hallucinated fill for geometry).
- **Test mode** (`--limit N`) — try a handful of images first.
- **Quick mode** (`--max-size PX`) — downscale to test speed/quality fast (not for real
  reconstruction — it changes dimensions).

---

## 8. How it stays COLMAP / 3DGS-compatible

- **Originals are never modified.** Output is a separate tree; same-name files are
  skipped unless `--overwrite`.
- **Dimensions preserved.** Output width×height == input (COLMAP derives focal length
  in pixels from EXIF + image size; a resize would corrupt intrinsics).
- **EXIF preserved** — especially `FocalLengthIn35mmFilm` / `FocalLength` (full EXIF
  transplant for JPEG→JPEG; best-effort for TIFF/PNG), plus the ICC profile. Add
  `--exiftool` (with `exiftool` on PATH) for an exhaustive all-metadata copy
  (maker notes, GPS, XMP, IPTC) across every format.
- **Format preserved** — JPEG→JPEG (quality ≥95, 4:4:4, single re-encode), PNG/TIFF stay
  lossless. No double-JPEG.
- **Uniform processing** — the whole set runs through one model/version/parameter set,
  avoiding photometric discontinuities across views.

---

## 9. Recommended workflow for 3DGS / photogrammetry

1. **Capture-time fixes beat software.** For shiny subjects, cross-polarization (polarizer
   on lights + lens), dulling spray, soft diffuse lighting, and locked exposure are the
   gold standard. Use AI removal to **salvage sets you can't re-shoot**.
2. Run UnReflect Batch with `--make-preview --heatmap` and **look at the previews** — does
   it cleanly remove highlights, or invent texture? Invented detail that differs per view
   hurts SfM.
3. Consider **`--composite`** (changes only highlight regions) and/or **`--emit-mask`**
   (feed the mask to COLMAP to exclude altered regions from matching).
4. **A/B test:** reconstruct with the originals and with the cleaned set; keep whichever
   gives better registration / fewer artifacts. Note that some 3DGS variants *model*
   reflections rather than needing them removed. Use the bundled harness:
   ```powershell
   python tools\ab_colmap.py --work ab_work --colmap <path-to>\colmap.exe ^
       --set original "D:\photo_input" --set cleaned "D:\photo_unreflect"
   ```
   It runs a COLMAP sparse reconstruction on each set and reports **registered
   images, 3D points, mean track length and reprojection error** side by side, plus
   `ab_work\ab_report.md`. (COLMAP not bundled — get it from
   <https://github.com/colmap/colmap/releases>.)

---

## 9b. A/B evaluation pipelines (optional)

Two optional harnesses under `tools/` let you **measure** whether cleaning actually
helps your reconstruction, instead of guessing. Both resolve external tools from a
flag, an environment variable, or `PATH` — nothing is hard-coded.

**External tools (not bundled):**
- [COLMAP](https://github.com/colmap/colmap/releases) — `--colmap` or `$COLMAP_EXE`
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) (for the 3DGS harness) — `--lichtfeld` or `$LICHTFELD_EXE`

### SfM A/B — `tools/ab_colmap.py`
Runs a COLMAP sparse reconstruction on each image set and reports registered images,
3D points, track length and reprojection error.
```powershell
python tools\ab_colmap.py --work ab_work --matcher sequential --max-image-size 2000 ^
    --set original "D:\photo_input" --set cleaned "D:\photo_unreflect"
```

### 3DGS A/B — `tools/ab_3dgs.py`
Full pipeline per set: **COLMAP → LichtFeld Studio headless training → eval renders →
same-viewpoint comparison figures + report** (PSNR / SSIM / #Gaussians).
```powershell
$env:LICHTFELD_EXE = "C:\path\to\LichtFeld-Studio.exe"
python tools\ab_3dgs.py --work ab3dgs_out ^
    --set original "D:\photo_input" --set cleaned "D:\photo_unreflect" ^
    --shared-poses original --steps-scaler 0.5 --resize-factor 2
```
- `--shared-poses NAME` trains every set on `NAME`'s camera poses so renders are
  directly comparable per frame (requires identically named, pixel-aligned images
  across sets — exactly the original-vs-cleaned case). Omit it for a fully independent
  pipeline per set.
- COLMAP `SIMPLE_RADIAL` cameras carry lens distortion, so LichtFeld is invoked with
  `--undistort` automatically (`--no-undistort` to disable).
- Output: `<work>/compare/*.jpg` (per-frame `GT | 3DGS-per-set` strips) and
  `<work>/report.md`.

> **Reading the numbers honestly.** PSNR/SSIM are computed against each set's *own*
> ground-truth images. When the GT differs between sets (glare vs. cleaned), they
> indicate *reconstruct-ability*, not absolute quality — always read them together with
> the visual figures. In practice, removing view-dependent glare tends to raise PSNR/SSIM
> and *lower* the Gaussian count (the model no longer wastes splats on specular
> highlights), and `--mask-composite` is usually the cleaning mode that helps rather than
> hurts (default removal softens high-res input and can degrade SfM).

---

## 10. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `Error(s) in loading state_dict … Missing key(s) … dinov3` | Wrong `transformers`. Install the pinned commit: `pip install "transformers @ https://github.com/huggingface/transformers/archive/2fe43376cdde02b7ffcf117e6eb9aa4375fb2dd1.zip"` |
| `torch.cuda.is_available()` is `False`, or `no kernel image is available` | CPU-only / wrong torch wheel. Reinstall: `pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall` |
| `FileNotFoundError … Run 'unreflectanything download --weights' first` | Weights not downloaded yet — see [section 3](#3-download-the-model-weights-required-one-time). |
| `UnicodeEncodeError: 'cp932' …` when running `unreflectanything --help` | Windows non-UTF-8 console. Set `PYTHONUTF8=1` (the app and `.bat` files already do this). |
| CUDA out of memory on big images | Use `--max-size` for a quick test, or `--device cpu`, or process fewer at a time. |
| One image fails | It's logged to `logs/errors.csv` and the batch keeps going by design. |

---

## 11. Project layout

```
UnReflectAnything/
├── main.py                 # CLI entry point
├── app.py                  # Streamlit GUI (Phase 2)
├── requirements.txt
├── run_app.bat             # double-click: GUI (first run = auto-setup)
├── run_batch_example.bat   # double-click: example CLI batch
├── scripts/setup_env.ps1   # environment installer (venv + cu128 torch + deps + weights)
├── src/
│   ├── image_io.py         # discovery + EXIF/format-preserving I/O
│   ├── metrics.py          # luminance / highlight metrics, heatmap, change mask
│   ├── preview.py          # before/after compare composites
│   ├── logger.py           # JSONL / CSV / errors / summary
│   └── unreflect_batch.py  # engine: device select, model load, per-image pipeline
├── tools/
│   ├── ab_colmap.py        # A/B COLMAP sparse-reconstruction comparison (original vs cleaned)
│   └── ab_3dgs.py          # A/B 3D Gaussian Splatting comparison (COLMAP -> LichtFeld -> figures)
└── tests/                  # fast unit tests (no torch needed)
```

## 12. Testing
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
The unit tests cover the torch-independent utilities (discovery, I/O, metrics, preview,
logging) and run in well under a second.

## 13. Notes & license

- This wrapper is provided as-is for your pipeline. **UnReflectAnything** is MIT-licensed,
  but its frozen vision backbone is **DINOv3**, governed by **Meta's DINOv3 License**
  (not open source; requires "Built with DINOv3" attribution and has use restrictions).
  Review that license before any redistribution or commercial use.
- **High-resolution inputs are softened.** The model resizes internally to ~448 px and
  upscales the diffuse result back to the original dimensions, so on e.g. 4K input the
  *whole image* loses high-frequency detail — which can **hurt** SfM feature matching.
  For high-res photogrammetry use **`--mask-composite`**: it keeps the original full-res
  pixels everywhere except blown highlights, preserving SfM features while still
  suppressing glare. (Measured on 4K video frames: default cleaning cut image sharpness
  — Laplacian variance — by ~94%, while `--mask-composite` retained most of it.)
- Because of the internal resize, fine texture in altered regions is *reconstructed*,
  not measured. This is why the output is for **evaluation**, not metrology.

### Acknowledgments / third-party

This wrapper **does not bundle or redistribute** any of the following — it invokes them
as installed dependencies or external tools. Respect each project's license:

- [UnReflectAnything](https://github.com/alberto-rota/UnReflectAnything) (MIT) — the
  reflection-removal model. Its frozen **DINOv3** backbone is governed by **Meta's DINOv3
  License** (not open source; attribution + use restrictions).
- [COLMAP](https://github.com/colmap/colmap) (BSD) — Structure-from-Motion, used by the
  A/B harnesses.
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) (GPLv3) — 3D Gaussian
  Splatting trainer, invoked as an external process by `tools/ab_3dgs.py` (not bundled
  or linked into this project).
- [PyTorch](https://pytorch.org), [Pillow](https://python-pillow.org),
  [piexif](https://github.com/hMatoba/Piexif), [Streamlit](https://streamlit.io).

This project is licensed under the **MIT License** (see [`LICENSE`](LICENSE)); see
[`NOTICE.md`](NOTICE.md) for the full third-party tool/model terms (incl. DINOv3 and
LichtFeld's GPL-3.0).
