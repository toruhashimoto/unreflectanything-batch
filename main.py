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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unreflect-batch",
        description=(
            "Batch-remove specular reflections / highlights from photos using "
            "UnReflectAnything, as an evaluation pre-process for 3D Gaussian "
            "Splatting / photogrammetry. Originals are never modified."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, type=Path, help="input image folder")
    p.add_argument("--output", "-o", required=True, type=Path, help="output folder (must be OUTSIDE input)")
    p.add_argument("--recursive", "-r", action="store_true", help="recurse into sub-folders")
    p.add_argument(
        "--device", "-d", choices=["auto", "cuda", "cpu"], default="auto",
        help="auto = use CUDA if a working GPU is present, else CPU",
    )
    p.add_argument(
        "--extensions", default=",".join(SUPPORTED_EXTS),
        help="comma-separated list of extensions to process",
    )

    # Output artifacts / evaluation.
    p.add_argument("--make-preview", action="store_true", help="save side-by-side before/after into preview_compare/")
    p.add_argument("--heatmap", action="store_true", help="save per-image luma-difference heatmaps into heatmap/")
    p.add_argument("--emit-mask", action="store_true", help="save approximate changed-region masks into masks/ (COLMAP exclusion masks)")

    # Safety / behaviour.
    p.add_argument("--overwrite", action="store_true", help="overwrite existing outputs (default: skip)")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG output quality (>=95 enforced)")

    # Model parameters (passed through to UnReflectAnything).
    p.add_argument("--threshold", type=float, default=0.3, help="highlight detection threshold")
    p.add_argument("--dilation", type=int, default=40, help="highlight mask dilation (px)")
    p.add_argument("--batch-size", type=int, default=4, help="model batch size")
    p.add_argument("--composite", action="store_true", help="model's internal composite: blend diffuse into highlight regions at the model's ~448px resolution")
    p.add_argument("--mask-composite", action="store_true", help="wrapper FULL-RES composite: keep original detail everywhere except blown highlights (best for high-res SfM/3DGS input)")
    p.add_argument("--exiftool", action="store_true", help="copy ALL metadata via exiftool when available (maker notes/GPS/XMP, all formats; slower, per-file). Default uses fast piexif/PIL EXIF.")
    p.add_argument("--verbose", action="store_true", help="show the engine's own per-image stdout")

    # Test / quick modes.
    p.add_argument("--limit", type=int, default=None, metavar="N", help="test mode: process only the first N images")
    p.add_argument("--max-size", type=int, default=None, metavar="PX", help="quick mode: downscale longest side to PX before processing (CHANGES output dims — not for COLMAP input)")
    p.add_argument("--dry-run", action="store_true", help="list what would be processed without running the model")
    p.add_argument("--no-progress", action="store_true", help="disable the tqdm progress bar")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Import the engine lazily so --help works even before torch is installed.
    from src.unreflect_batch import BatchConfig, run_batch, WeightsMissingError, ModelLoadError

    exts = tuple(e for e in (args.extensions or "").replace(" ", ",").split(",") if e)
    cfg = BatchConfig(
        input_dir=args.input,
        output_dir=args.output,
        recursive=args.recursive,
        exts=exts,
        device=args.device,
        overwrite=args.overwrite,
        make_preview=args.make_preview,
        heatmap=args.heatmap,
        emit_mask=args.emit_mask,
        limit=args.limit,
        max_size=args.max_size,
        jpeg_quality=args.jpeg_quality,
        threshold=args.threshold,
        dilation=args.dilation,
        batch_size=args.batch_size,
        composite=args.composite,
        mask_composite=args.mask_composite,
        use_exiftool=args.exiftool,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

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

    print("\n=== UnReflect Batch summary ===")
    print(f"  device      : {summary.get('device')}  ({summary.get('device_note')})")
    print(f"  candidates  : {summary.get('num_candidates')}")
    print(f"  processed   : {summary.get('processed_ok', 0)}")
    print(f"  skipped     : {summary.get('skipped', 0)}")
    print(f"  errors      : {summary.get('errors', 0)}")
    print(f"  output      : {summary.get('output_dir')}")
    print(f"  logs        : {Path(summary.get('output_dir', '.')) / 'logs'}")
    if summary.get("errors", 0):
        print(f"  -> see {Path(summary.get('output_dir', '.')) / 'logs' / 'errors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
