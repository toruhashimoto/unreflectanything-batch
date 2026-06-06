"""RealityScan (formerly RealityCapture) image-layer mask helpers.

RealityScan applies per-image masks during alignment / meshing / texturing. The
conventions below are taken verbatim from the official Capturing Reality / Epic
documentation (verified for RealityScan 2.x, unchanged across the RealityCapture
rebrand):

* **Polarity** — *"In a mask, white areas will be used in processing, while black
  areas are excluded."* So a reflection you want ignored must be painted **black (0)**;
  everything kept is **white (255)**. There is no per-stage inversion and no import-time
  "invert" toggle, so the file must already be in this polarity.
  (https://rshelp.capturingreality.com/en-US/tools/mask.htm)

* **Naming (same-folder form)** — the mask for an image ``IMG_1234.jpg`` is
  ``IMG_1234.jpg.mask.png`` — the full original file name *including its extension*,
  then the ``.mask`` layer keyword, then the mask's own ``.png`` extension. The
  separator before ``mask`` may be one of ``. _ @ # !``. The mask must sit **in the
  same folder as the image** and be imported **together** with the photos, or
  RealityScan loads it as an extra image instead of attaching it.
  (https://rshelp.capturingreality.com/en-US/tools/imglayers.htm)

* **Format** — an 8-bit grayscale PNG, strictly binary (0/255, hard edges; gray is
  accepted but officially discouraged), at the source image's exact resolution.

To use: import the folder (photos + masks together) via the WORKFLOW tab, select the
images, then enable **"Enable masks for alignment"** in Selected Input > Image Layers.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# The layer keyword is the literal string "mask" (not user-configurable). The
# separator that precedes it may be any one of these symbols.
MASK_LAYER = "mask"
VALID_SEPARATORS = (".", "_", "@", "#", "!")


def mask_filename(image_name: str, separator: str = ".") -> str:
    """Return the same-folder RealityScan mask file name for an image file name.

    >>> mask_filename("IMG_1234.jpg")
    'IMG_1234.jpg.mask.png'
    >>> mask_filename("frame-00000.png", separator="_")
    'frame-00000.png_mask.png'
    """
    if separator not in VALID_SEPARATORS:
        raise ValueError(
            f"separator must be one of {VALID_SEPARATORS!r} (got {separator!r})"
        )
    return f"{image_name}{separator}{MASK_LAYER}.png"


def save_mask_png(mask: np.ndarray, dst: Path, like_size: tuple[int, int] | None = None) -> None:
    """Write ``mask`` as a strictly-binary 8-bit grayscale PNG (only 0 and 255).

    ``like_size`` is an optional ``(width, height)`` of the source image; if given and
    the mask differs, it is resized with nearest-neighbour (preserving the hard binary
    edges) so the mask is 1:1 with its image, as RealityScan expects.
    """
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    # Enforce strict binary (RealityScan discourages intermediate gray values).
    arr = np.where(arr >= 128, np.uint8(255), np.uint8(0))
    img = Image.fromarray(arr, "L")
    if like_size is not None and img.size != tuple(like_size):
        img = img.resize(tuple(like_size), Image.NEAREST)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG")


def copy_source_image(src: Path, dst: Path) -> None:
    """Byte-exact copy of the original image next to its mask.

    A plain copy (not a re-encode) keeps the pixels, dimensions, EXIF, ICC and format
    identical to the original, which RealityScan needs for correct intrinsics — and
    keeps this tool's guarantee that the *input* is never modified (we only write into
    the separate output folder)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
