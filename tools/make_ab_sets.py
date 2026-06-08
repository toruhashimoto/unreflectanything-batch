"""Build the four A/B comparison variants for a photo set, in one tidy workspace.

The whole point of ReflectMask is *measurable* alignment quality, not looks -- so the
honest way to choose a strategy is to reconstruct the same scene several ways and compare.
This orchestrator produces, under ``<work>/``, four ready-to-use sets:

    <work>/original/      copies of the input photos            (baseline: NO mask)
    <work>/reflectmask/   originals + <name>.mask.png           (AI backend)
    <work>/luma/          originals + <name>.mask.png           (pure-luma backend, no model)
    <work>/cleaned/       reflection-removed images             (experimental)

plus ``ab_sets_report.md`` / ``.json`` with per-set stats (image count, mean % of pixels
excluded, over-masking warnings) and the exact next-step commands. Each masked set is a
self-contained RealityScan-importable folder; import each set, run alignment, and compare
registered-image count / detail / artifacts.

The model is loaded ONCE and reused for the two AI sets. If the weights aren't downloaded
(or you pass --skip-model), only ``original`` and ``luma`` are built -- so this still works
with no GPU and no weights.

    python tools/make_ab_sets.py -i "D:/photo_input" -o "D:/ab_work" --recursive
    python tools/make_ab_sets.py -i "D:/photo_input" -o "D:/ab_work" --skip-model   # no GPU
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import image_io  # noqa: E402
from src.unreflect_batch import (  # noqa: E402
    BatchConfig, run_batch, resolve_device, load_model, weights_status,
)


def copy_originals(input_dir: Path, out_dir: Path, recursive: bool) -> int:
    """Copy every input image into ``out_dir`` preserving the sub-folder layout."""
    imgs = image_io.iter_images(input_dir, recursive, image_io.SUPPORTED_EXTS)
    for p in imgs:
        dst = image_io.relative_output_path(p, input_dir, out_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    return len(imgs)


def _set_from_summary(name: str, out_dir: Path, summary: dict) -> dict:
    return {
        "set": name,
        "dir": str(out_dir),
        "import_dir": summary.get("realityscan_dir") or str(out_dir),
        "mode": summary.get("mode"),
        "backend": summary.get("backend"),
        "images": summary.get("processed_ok", 0),
        "errors": summary.get("errors", 0),
        "masks": bool(summary.get("realityscan_masks_written")),
        "mean_excluded_pct": summary.get("realityscan_mean_excluded_pct"),
        "warn_images": summary.get("realityscan_warn_images", 0),
        "danger_images": summary.get("realityscan_danger_images", 0),
    }


def build_report(work: Path, input_dir: Path, results: list) -> tuple:
    """Return ``(markdown, json_obj)`` summarising the variant sets and how to compare."""
    def cell(v):
        return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    lines = [
        "# A/B comparison sets",
        "",
        f"Input: `{input_dir}`",
        f"Workspace: `{work}`",
        "",
        "| set | images | masks | mean % excluded | warn | danger | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        note = r.get("skipped") or ("baseline, no mask" if r["set"] == "original" else "")
        lines.append(
            f"| `{r['set']}` | {cell(r.get('images'))} | {'yes' if r.get('masks') else 'no'} "
            f"| {cell(r.get('mean_excluded_pct'))} | {cell(r.get('warn_images'))} "
            f"| {cell(r.get('danger_images'))} | {note} |"
        )
    lines += ["", "Import folders (masked sets are the `realityscan/` sub-folder):"]
    for r in results:
        lines.append(f"- `{r['set']}` -> `{r.get('import_dir', r.get('dir'))}`")
    lines += [
        "",
        "## Compare in RealityScan (primary)",
        "For each masked set (`reflectmask`, `luma`) import its folder so photos + `.mask.png`",
        "load together, tick **Selected Input > Image Layers > Enable masks for alignment**, and",
        "align. For `original` and `cleaned`, import the folder and align with no mask. Compare",
        "**registered-image count, component connectivity, and final detail / artifacts**.",
        "",
        "## Compare in COLMAP (optional, scripted)",
        "`tools/ab_colmap.py` runs a COLMAP sparse reconstruction per set and tabulates",
        "registered images / 3D points / track length / reprojection error:",
        "```",
        f'python tools/ab_colmap.py --work "{work / "_colmap"}" ^',
        f'    --set original "{work / "original"}" --set cleaned "{work / "cleaned"}"',
        "```",
        "(COLMAP reads masks with its own naming, so for masked COLMAP runs generate them with",
        "`tools/make_colmap_masks.py` and pass `--set-masked`; the RealityScan masks here are",
        "named for RealityScan import.)",
    ]
    obj = {"input": str(input_dir), "work": str(work), "sets": results}
    return "\n".join(lines), obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the original / reflectmask / luma / cleaned A/B variant sets in one workspace.")
    ap.add_argument("--input", "-i", required=True, type=Path, help="input image folder")
    ap.add_argument("--work", "-o", required=True, type=Path, help="output workspace (gets one sub-folder per set)")
    ap.add_argument("--recursive", "-r", action="store_true", help="recurse into sub-folders")
    ap.add_argument("--device", "-d", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None, metavar="N", help="only the first N images (quick test)")
    ap.add_argument("--rs-gate", type=float, default=250.0, help="AI reflectmask: original-luma gate")
    ap.add_argument("--luma-level", type=float, default=243.0, help="pure-luma set: brightness threshold")
    ap.add_argument("--skip-model", action="store_true",
                    help="only build original + luma (no GPU / weights needed)")
    args = ap.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"input folder does not exist: {args.input}")
    args.work.mkdir(parents=True, exist_ok=True)
    results: list = []

    # 1) original -- the no-mask baseline (a copy, so the workspace is self-contained).
    t0 = time.perf_counter()
    n = copy_originals(args.input, args.work / "original", args.recursive)
    print(f"[original] copied {n} images ({time.perf_counter() - t0:.1f}s)")
    results.append({"set": "original", "dir": str(args.work / "original"),
                    "import_dir": str(args.work / "original"), "images": n, "masks": False})

    # 2) luma -- needs no model / GPU / weights.
    luma_cfg = BatchConfig(
        input_dir=args.input, output_dir=args.work / "luma", recursive=args.recursive,
        mode="reflectmask", backend="luma", rs_highlight_gate=args.luma_level, limit=args.limit,
    )
    results.append(_set_from_summary("luma", args.work / "luma", run_batch(luma_cfg, progress=True)))

    # 3+4) AI sets -- reflectmask + cleaned, sharing one loaded model.
    have_weights = weights_status()[0]
    if args.skip_model or not have_weights:
        why = "skipped (--skip-model)" if args.skip_model else "skipped (weights not downloaded)"
        print(f"[reflectmask/cleaned] {why}")
        for nm in ("reflectmask", "cleaned"):
            imp = str(args.work / nm / "realityscan") if nm == "reflectmask" else str(args.work / nm)
            results.append({"set": nm, "dir": str(args.work / nm), "import_dir": imp,
                            "images": 0, "skipped": why})
    else:
        device, _ = resolve_device(args.device)
        model = load_model(device)
        rm_cfg = BatchConfig(
            input_dir=args.input, output_dir=args.work / "reflectmask", recursive=args.recursive,
            mode="reflectmask", backend="unreflect", rs_highlight_gate=args.rs_gate, limit=args.limit,
        )
        results.append(_set_from_summary("reflectmask", args.work / "reflectmask",
                                         run_batch(rm_cfg, progress=True, model=model)))
        cl_cfg = BatchConfig(
            input_dir=args.input, output_dir=args.work / "cleaned", recursive=args.recursive,
            mode="clean", backend="unreflect", limit=args.limit,
        )
        results.append(_set_from_summary("cleaned", args.work / "cleaned",
                                         run_batch(cl_cfg, progress=True, model=model)))

    md, obj = build_report(args.work, args.input, results)
    (args.work / "ab_sets_report.md").write_text(md, encoding="utf-8")
    (args.work / "ab_sets_report.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print("\n" + md)
    print(f"\nReport: {args.work / 'ab_sets_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
