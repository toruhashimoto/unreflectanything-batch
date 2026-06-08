"""Generate RealityScan alignment masks that EXCLUDE bright reflection (highlight)
regions — WITHOUT a GPU or the UnReflectAnything model (pure OpenCV/numpy luma gate).

This is the no-model counterpart of the GUI/CLI ``--realityscan`` feature: where that
uses the model's before/after diff to target the reflections it actually removed, this
tool simply excludes pixels brighter than ``--level``. Use it when you have no GPU /
haven't downloaded the weights, or want a fast deterministic baseline.

RealityScan convention (see src/realityscan.py for citations): a mask is **black (0)
where excluded, white (255) where kept**, named ``<image.ext>.mask.png`` and placed in
the SAME folder as the image, imported together with the photos. This tool writes a
ready-to-import folder: a copy of each original image plus its mask.

    python tools/make_realityscan_masks.py -i "D:/imgs" -o "D:/rs_project" --level 240 --dilation 2
    # then in RealityScan: WORKFLOW > Folder > pick D:/rs_project, enable masks for alignment.

Keep the gate TIGHT — over-masking (excluding large bright-but-diffuse areas like sky or
white surfaces) removes feature-rich regions and can fragment the reconstruction. Exclude
only the actual reflections (raise --level / keep --dilation small).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import realityscan, metrics  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate RealityScan alignment masks (luma-gated, no model) that "
                    "exclude bright reflection regions, plus a copy of each image.")
    ap.add_argument("--input", "-i", required=True, type=Path, help="image folder")
    ap.add_argument("--output", "-o", required=True, type=Path,
                    help="output folder (gets a copy of each image + its .mask.png)")
    ap.add_argument("--level", type=float, default=240.0,
                    help="exclude pixels with luma >= this (0-255). Higher = tighter.")
    ap.add_argument("--dilation", type=int, default=2,
                    help="grow the excluded region by N px (keep small; large fragments SfM)")
    ap.add_argument("--open", dest="open_radius", type=int, default=1,
                    help="remove specks smaller than this radius (morphological open)")
    ap.add_argument("--masks-only", action="store_true",
                    help="write only the .mask.png files (don't copy the originals). NOTE: the "
                         "folder is then NOT directly importable - merge each .mask.png into the "
                         "same folder as its image before importing into RealityScan")
    ap.add_argument("--separator", default=".", choices=list(realityscan.VALID_SEPARATORS),
                    help="mask name separator before 'mask' (e.g. '.' -> name.ext.mask.png)")
    ap.add_argument("--recursive", "-r", action="store_true", help="recurse into sub-folders")
    args = ap.parse_args(argv)

    globber = args.input.rglob("*") if args.recursive else args.input.glob("*")
    imgs = sorted(p for p in globber if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    excl = []
    for p in imgs:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        # Same pure-luma exclusion mask the GUI/CLI use with --backend luma.
        mask, stats = metrics.luma_exclusion_mask(
            rgb, level=args.level, dilation=args.dilation,
            open_radius=args.open_radius, return_stats=True,
        )
        excl.append(stats["final_mask_ratio"])

        rel = p.relative_to(args.input)
        mask_dst = args.output / rel.parent / realityscan.mask_filename(p.name, args.separator)
        realityscan.save_mask_png(mask, mask_dst)
        if not args.masks_only:
            realityscan.copy_source_image(p, args.output / rel)

    mean_excl = float(np.mean(excl)) if excl else 0.0
    print(f"wrote {len(imgs)} RealityScan masks to {args.output}  "
          f"(luma>={args.level:.0f}, dilation={args.dilation}; "
          f"excluded {mean_excl:.2f}% of pixels on average)")
    if not args.masks_only:
        print("  (originals copied alongside the masks - import this folder into RealityScan)")
    if mean_excl > 15:
        print("WARNING: excluding >15% of pixels - likely too aggressive (bright diffuse "
              "areas, not just reflections). Raise --level / lower --dilation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
