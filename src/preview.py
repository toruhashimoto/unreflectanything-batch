"""Side-by-side before/after (and optional diff) preview composites.

Saved under ``<output>/preview_compare/`` mirroring the input tree. Previews are
*downscaled* (they are only for human visual judgement, not for reconstruction),
keeping their file size small even for large source photos.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

_LABEL_BAR = 22
_GAP = 8
_BG = (245, 245, 245)
_FG = (20, 20, 20)


def _resampling():
    # Pillow >=9.1 enum; fall back for older.
    return getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def _fit_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    w = max(1, round(img.width * target_h / img.height))
    return img.resize((w, target_h), _resampling())


def _label(panel: Image.Image, text: str) -> Image.Image:
    """Add a small caption bar above a panel."""
    out = Image.new("RGB", (panel.width, panel.height + _LABEL_BAR), _BG)
    out.paste(panel, (0, _LABEL_BAR))
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), text, fill=_FG)
    return out


def make_compare(
    before: Image.Image,
    after: Image.Image,
    heatmap: Image.Image | None = None,
    max_height: int = 720,
) -> Image.Image:
    """Compose a horizontal [Original | UnReflect | (Diff)] strip."""
    target_h = min(max_height, before.height)
    panels = [
        _label(_fit_height(before.convert("RGB"), target_h), "Original"),
        _label(_fit_height(after.convert("RGB"), target_h), "UnReflect"),
    ]
    if heatmap is not None:
        panels.append(_label(_fit_height(heatmap.convert("RGB"), target_h), "Diff (luma)"))

    total_w = sum(p.width for p in panels) + _GAP * (len(panels) - 1)
    total_h = max(p.height for p in panels)
    canvas = Image.new("RGB", (total_w, total_h), _BG)
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + _GAP
    return canvas
