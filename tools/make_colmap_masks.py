"""Generate COLMAP / RealityScan feature-extraction masks that EXCLUDE bright
reflection (highlight) regions — WITHOUT altering the images.

For SfM/photogrammetry alignment, masking out reflective regions is often better than
inpainting them: the images stay untouched, so the surrounding features keep their exact
original descriptors and geometric accuracy, while only the unreliable, view-dependent
reflection pixels are ignored. (Measured on real footage, tight masking preserved more
3D points AND the original reprojection accuracy, vs. inpaint-style removal.)

COLMAP convention: a mask lives at ``<mask_dir>/<image_name>.png`` (e.g. the mask for
``frame-00000.png`` is ``frame-00000.png.png``). Black (0) pixels are ignored; non-zero
(255) pixels are used. RealityScan also accepts per-image masks.

Use the resulting masks with the A/B harness:
    python tools/ab_colmap.py --work ab --matcher sequential \
        --set original "D:/imgs" \
        --set-masked masked "D:/imgs" "D:/masks"

Keep the gate TIGHT — over-masking (excluding large bright-but-diffuse areas like sky or
white surfaces) removes feature-rich regions and can fragment the reconstruction. Exclude
only the actual reflections (raise --level / keep --dilation small).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate COLMAP/RealityScan masks that "
                                             "exclude bright reflection regions.")
    ap.add_argument("--input", "-i", required=True, type=Path, help="image folder")
    ap.add_argument("--output", "-o", required=True, type=Path, help="output mask folder")
    ap.add_argument("--level", type=float, default=240.0,
                    help="exclude pixels with luma >= this (0-255). Higher = tighter.")
    ap.add_argument("--dilation", type=int, default=2,
                    help="grow the excluded region by N px (keep small; large fragments SfM)")
    ap.add_argument("--recursive", "-r", action="store_true", help="recurse into sub-folders")
    args = ap.parse_args(argv)

    globber = args.input.rglob("*") if args.recursive else args.input.glob("*")
    imgs = sorted(p for p in globber if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    excl = []
    for p in imgs:
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        luma = rgb @ _LUMA
        refl = (luma >= args.level).astype(np.uint8)
        if args.dilation > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilation + 1,) * 2)
            refl = cv2.dilate(refl, k)
        mask = np.where(refl > 0, np.uint8(0), np.uint8(255))  # 0 = ignore reflection
        excl.append(float((mask == 0).mean()) * 100)
        rel = p.relative_to(args.input)
        dst = args.output / rel.parent / (p.name + ".png")  # COLMAP: <image_name>.png
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, "L").save(dst)

    print(f"wrote {len(imgs)} masks to {args.output}  "
          f"(luma>={args.level:.0f}, dilation={args.dilation}; "
          f"excluded {np.mean(excl):.2f}% of pixels on average)")
    if np.mean(excl) > 15:
        print("WARNING: excluding >15% of pixels — likely too aggressive (bright diffuse "
              "areas, not just reflections). Raise --level / lower --dilation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
