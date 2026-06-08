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


def make_grid(panels, cols: int = 3, max_height: int = 320) -> Image.Image:
    """Lay out labelled panels row-major in a grid of ``cols`` columns.

    ``panels`` is a list of ``(label, PIL.Image)``. Each panel is height-normalised and
    captioned; cells are sized to the widest/tallest panel and left-aligned within.
    """
    if not panels:
        raise ValueError("make_grid: no panels")
    tiles = [_label(_fit_height(img.convert("RGB"), max_height), text) for text, img in panels]
    cell_w = max(t.width for t in tiles)
    cell_h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    total_w = cols * cell_w + _GAP * (cols - 1)
    total_h = rows * cell_h + _GAP * (rows - 1)
    canvas = Image.new("RGB", (total_w, total_h), _BG)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas.paste(t, (c * (cell_w + _GAP), r * (cell_h + _GAP)))
    return canvas


def make_diagnostic(
    original: Image.Image,
    cleaned: Image.Image,
    heatmap: Image.Image,
    candidate: Image.Image,
    final_mask: Image.Image,
    overlay: Image.Image,
    max_height: int = 300,
) -> Image.Image:
    """A single 6-panel inspection sheet for the Diagnostic mode.

    Row 1: Original | Cleaned (backend) | Diff heatmap
    Row 2: Threshold candidate | Final mask (black = excluded) | Overlay (red = excluded)
    """
    return make_grid(
        [
            ("Original", original),
            ("Cleaned (backend)", cleaned),
            ("Diff heatmap", heatmap),
            ("Threshold candidate", candidate),
            ("Final mask (black=excluded)", final_mask),
            ("Overlay (red=excluded)", overlay),
        ],
        cols=3,
        max_height=max_height,
    )
