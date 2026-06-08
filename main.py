"""UnReflect Batch — command-line entry point.

Example:
    python main.py --input "D:\\photo_input" --output "D:\\photo_unreflect" \\
        --recursive --make-preview --device cuda

Always writes to a separate output folder; never modifies the originals.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Defensive: unreflectanything's banner/rich output uses non-ASCII; on a Windows
# cp932 console that can crash. Force UTF-8 before anything heavy is imported.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# Allow `python main.py` from any cwd.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.image_io import SUPPORTED_EXTS  # noqa: E402


class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show option defaults AND keep the raw (multi-line) epilog formatting."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reflectmask",
        description=(
            "ReflectMask for RealityScan — generate tight binary alignment masks that "
            "exclude specular reflections / blown highlights from RealityScan feature "
            "detection, protecting valid features for high-detail photogrammetry. "
            "UnReflectAnything is used as the reflection-detection backend. Originals "
            "are never modified."
        ),
        epilog=(
            "modes (optional leading subcommand; default = reflectmask):\n"
            "  reflectmask  original images + <name>.mask.png exclusion masks  (DEFAULT, primary)\n"
            "  diagnostic   reflectmask + before/after previews & diff heatmaps for inspection\n"
            "  clean        EXPERIMENTAL: export reflection-removed (cleaned) images instead\n"
            "\n"
            "examples:\n"
            "  python main.py reflectmask -i \"D:\\photo_input\" -o \"D:\\rs_reflectmask\" -r\n"
            "  python main.py -i \"D:\\in\" -o \"D:\\out\" -r          (flat form = reflectmask)\n"
            "  python main.py clean -i \"D:\\in\" -o \"D:\\out\" -r      (cleaned images, experimental)\n"
        ),
        formatter_class=_Fmt,
    )
    p.add_argument("--input", "-i", required=True, type=Path, help="input image folder")
    p.add_argument("--output", "-o", required=True, type=Path, help="output folder (must be OUTSIDE input)")
    p.add_argument("--recursive", "-r", action="store_true", help="recurse into sub-folders")
    p.add_argument(
        "--device", "-d", choices=["auto", "cuda", "cpu"], default="auto",
        help="auto = use CUDA if a working GPU is present, else CPU",
    )
    p.add_argument(
        "--backend", choices=["unreflect", "luma"], default="unreflect",
        help="reflection-detection backend: 'unreflect' (AI; needs GPU + weights) or "
             "'luma' (pure brightness gate; no model/GPU/weights, --rs-gate is the luma level)",
    )
    p.add_argument(
        "--workers", default="auto", metavar="N",
        help="parallel worker processes: 'auto' (from CPU cores / free VRAM / the Windows 61 "
             "cap) or an integer; 1 = sequential. The per-image work is CPU/I-O bound and the "
             "GPU is mostly idle, so on a many-core box this is the big throughput lever.",
    )
    p.add_argument(
        "--extensions", default=",".join(SUPPORTED_EXTS),
        help="comma-separated list of extensions to process",
    )

    # Output artifacts / evaluation.
    p.add_argument("--make-preview", action="store_true", help="save side-by-side before/after into preview_compare/")
    p.add_argument("--heatmap", action="store_true", help="save per-image luma-difference heatmaps into heatmap/")
    p.add_argument("--emit-mask", action="store_true", help="save approximate changed-region masks into masks/ (255=changed; visualization)")

    # RealityScan alignment masks.
    p.add_argument("--realityscan", action="store_true",
                   help="emit a RealityScan-ready folder under realityscan/: a copy of each ORIGINAL image plus a '<name>.mask.png' exclusion mask (black=removed reflection=excluded, white=kept). Import the folder into RealityScan and enable 'masks for alignment'.")
    p.add_argument("--rs-masks-only", action="store_true",
                   help="RealityScan: write only the .mask.png files (don't copy the originals). NOTE: the folder is then NOT directly importable — each .mask.png must be merged into the same folder as its image before import")
    p.add_argument("--rs-drop", type=float, default=12.0,
                   help="RealityScan mask: min luma the model must darken a pixel by to count it as a removed reflection")
    p.add_argument("--rs-gate", type=float, default=250.0,
                   help="RealityScan mask: only mask pixels whose ORIGINAL luma was >= this (0-255; 0 disables). Tight by default so diffuse-bright surfaces (sky, white paint/bodywork) are NOT excluded; lower to ~240 for a genuinely glary set")
    p.add_argument("--rs-dilation", type=int, default=2,
                   help="RealityScan mask: grow the excluded region by N px to cover reflection halos (keep small)")
    p.add_argument("--rs-open", type=int, default=1,
                   help="RealityScan mask: remove specks smaller than this radius (morphological open)")
    p.add_argument("--rs-separator", default=".", choices=[".", "_", "@", "#", "!"],
                   help="RealityScan mask name separator before 'mask' (e.g. '.' -> name.ext.mask.png)")
    p.add_argument("--mask-warn", type=float, default=5.0,
                   help="warn when the average %% of pixels excluded by the masks exceeds this")
    p.add_argument("--mask-danger", type=float, default=12.0,
                   help="flag DANGER (likely over-masking valid features) when the average %% excluded exceeds this")

    # Safety / behaviour.
    p.add_argument("--overwrite", action="store_true", help="overwrite existing outputs (default: skip)")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG output quality (>=95 enforced)")

    # Model parameters (passed through to UnReflectAnything).
    p.add_argument("--threshold", type=float, default=0.3, help="highlight detection threshold")
    p.add_argument("--dilation", type=int, default=40, help="highlight mask dilation (px)")
    p.add_argument("--batch-size", type=int, default=4, help="model batch size")
    p.add_argument("--composite", action="store_true", help="model's internal composite: blend diffuse into highlight regions at the model's ~448px resolution")
    p.add_argument("--mask-composite", action="store_true", help="wrapper FULL-RES composite: keep original detail everywhere except blown highlights (best for high-res SfM/3DGS input)")
    p.add_argument("--mask-level", type=float, default=248.0, help="mask-composite: only replace pixels brighter than this luma 0-255 (higher = tighter = less blur)")
    p.add_argument("--mask-dilation", type=int, default=0, help="mask-composite: grow the replaced region by N px (keep small; large values blur the subject)")
    p.add_argument("--mask-feather", type=float, default=1.0, help="mask-composite: edge feather sigma in px")
    p.add_argument("--exiftool", action="store_true", help="copy ALL metadata via exiftool when available (maker notes/GPS/XMP, all formats; slower, per-file). Default uses fast piexif/PIL EXIF.")
    p.add_argument("--verbose", action="store_true", help="show the engine's own per-image stdout")
    p.add_argument("--download-weights", action="store_true", help="download the ~5.9 GB model weights first if they are missing, then run")

    # Test / quick modes.
    p.add_argument("--limit", type=int, default=None, metavar="N", help="test mode: process only the first N images")
    p.add_argument("--max-size", type=int, default=None, metavar="PX", help="quick mode: downscale longest side to PX before processing (CHANGES output dims — not for COLMAP input)")
    p.add_argument("--model-max-size", type=int, default=2048, metavar="PX",
                   help="mask-first modes: cap the MODEL's working resolution to PX (the model is ~448px internally; feeding full 50MP is wasted I/O). The mask is upscaled to native and the original copy stays native, so the RealityScan deliverable's resolution is UNCHANGED. 0 = feed full resolution (slow). No effect in 'clean' mode.")
    p.add_argument("--dry-run", action="store_true", help="list what would be processed without running the model")
    p.add_argument("--no-progress", action="store_true", help="disable the tqdm progress bar")
    return p


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    # Import the engine lazily so --help works before torch is installed (importing the
    # engine pulls numpy/PIL but NOT torch — that import stays lazy inside the engine).
    from src.unreflect_batch import (
        BatchConfig, run_batch, WeightsMissingError, ModelLoadError,
        weights_status, download_weights, MODES, DEFAULT_MODE,
    )

    # An optional leading subcommand selects the product mode (default = reflectmask).
    # The legacy flat form and the back-compat `--realityscan` flag still parse as before.
    # The mode's mask-first defaults are resolved inside BatchConfig.__post_init__.
    mode = DEFAULT_MODE
    if raw and raw[0] in MODES:
        mode = raw.pop(0)
    args = build_parser().parse_args(raw)

    _w = str(args.workers).strip().lower()
    if _w in ("auto", "0", ""):
        workers = None
    else:
        try:
            workers = max(1, int(_w))
        except ValueError:
            print(f"\n[ERROR] --workers must be 'auto' or an integer (got {args.workers!r})", file=sys.stderr)
            return 2

    exts = tuple(e for e in (args.extensions or "").replace(" ", ",").split(",") if e)
    cfg = BatchConfig(
        input_dir=args.input,
        output_dir=args.output,
        recursive=args.recursive,
        exts=exts,
        device=args.device,
        mode=mode,
        backend=args.backend,
        overwrite=args.overwrite,
        make_preview=args.make_preview,
        heatmap=args.heatmap,
        emit_mask=args.emit_mask,
        limit=args.limit,
        max_size=args.max_size,
        model_max_size=(args.model_max_size or None),
        workers=workers,
        jpeg_quality=args.jpeg_quality,
        threshold=args.threshold,
        dilation=args.dilation,
        batch_size=args.batch_size,
        composite=args.composite,
        mask_composite=args.mask_composite,
        mask_composite_level=args.mask_level,
        mask_composite_dilation=args.mask_dilation,
        mask_composite_feather=args.mask_feather,
        realityscan=args.realityscan,
        rs_copy_originals=not args.rs_masks_only,
        rs_separator=args.rs_separator,
        rs_drop_level=args.rs_drop,
        rs_highlight_gate=args.rs_gate,
        rs_dilation=args.rs_dilation,
        rs_open=args.rs_open,
        mask_warn_pct=args.mask_warn,
        mask_danger_pct=args.mask_danger,
        use_exiftool=args.exiftool,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    if args.download_weights:
        ok, _, detail = weights_status()
        if ok:
            print(f"[weights] already present: {detail}")
        else:
            try:
                wdir = download_weights()
                print(f"[weights] downloaded to {wdir}")
            except Exception as e:  # noqa: BLE001
                print(f"\n[ERROR] weights download failed: {e}", file=sys.stderr)
                return 3

    try:
        summary = run_batch(cfg, progress=not args.no_progress)
    except (WeightsMissingError, ModelLoadError) as e:
        print(f"\n[SETUP ERROR] {e}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[ABORTED] interrupted by user (partial logs were written).", file=sys.stderr)
        return 130

    print("\n=== ReflectMask for RealityScan — summary ===")
    print(f"  mode        : {summary.get('mode')}  (backend: {summary.get('backend')})")
    print(f"  device      : {summary.get('device')}  ({summary.get('device_note')})")
    print(f"  workers     : {summary.get('workers')}")
    print(f"  candidates  : {summary.get('num_candidates')}")
    print(f"  processed   : {summary.get('processed_ok', 0)}")
    print(f"  skipped     : {summary.get('skipped', 0)}")
    print(f"  errors      : {summary.get('errors', 0)}")
    print(f"  output      : {summary.get('output_dir')}")
    print(f"  logs        : {Path(summary.get('output_dir', '.')) / 'logs'}")
    if cfg.realityscan:
        rs_dir = Path(summary.get('output_dir', '.')) / 'realityscan'
        mean_excl = summary.get('realityscan_mean_excluded_pct')
        n_masks = summary.get('realityscan_masks_written', 0)
        if not n_masks:
            print("  RealityScan : [!] no masks generated "
                  f"({summary.get('realityscan_warning', '')})")
        else:
            print(f"  RealityScan : {rs_dir}  ({n_masks} masks, avg {mean_excl:.2f}% excluded)")
            warn_n = summary.get('realityscan_warn_images', 0)
            danger_n = summary.get('realityscan_danger_images', 0)
            if warn_n or danger_n:
                print(f"     mask-area  : {warn_n} warning (> {cfg.mask_warn_pct:.0f}%), "
                      f"{danger_n} danger (> {cfg.mask_danger_pct:.0f}%) of {n_masks} images")
            if cfg.rs_copy_originals:
                print("     -> import this folder into RealityScan (photos + masks together),")
                print("        then enable 'masks for alignment' in Selected Input > Image Layers.")
            else:
                print("     -> masks-only: place each .mask.png next to its image (same folder),")
                print("        import together, then enable 'masks for alignment'.")
            if mean_excl is not None and mean_excl > cfg.mask_danger_pct:
                print(f"     [!] excluding {mean_excl:.1f}% of pixels on average -- likely over-masking "
                      "diffuse-bright areas, not just reflections.")
                print("         Raise --rs-gate (e.g. 252), or this set may simply not need masking "
                      "(originals often reconstruct best for mild glare).")
    if summary.get("errors", 0):
        print(f"  -> see {Path(summary.get('output_dir', '.')) / 'logs' / 'errors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
