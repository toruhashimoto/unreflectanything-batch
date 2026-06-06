"""Image discovery and EXIF/ICC-preserving I/O.

Design goals (driven by COLMAP / 3DGS requirements):
  * Never touch the originals.
  * Output dimensions must equal input dimensions (intrinsics depend on it).
  * Preserve EXIF — especially FocalLengthIn35mmFilm / FocalLength — and the ICC
    profile, because COLMAP derives focal length in pixels from EXIF.
  * Preserve the file format (jpeg->jpeg, png->png, tiff->tiff). Lossless formats
    stay lossless; JPEG is re-encoded at high quality (>=95, 4:4:4) only once.

Only `PIL`/`piexif` are imported here (no torch), so this module stays light and
unit-testable without the heavy ML stack.
"""
from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

# Make Pillow tolerant of slightly-truncated JPEGs rather than crashing a batch.
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_JPEG_EXTS = (".jpg", ".jpeg")
_TIFF_EXTS = (".tif", ".tiff")


def normalize_exts(exts: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize a list of extensions to lowercase, dot-prefixed tuple."""
    if not exts:
        return SUPPORTED_EXTS
    out: list[str] = []
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return tuple(out) or SUPPORTED_EXTS


def iter_images(
    input_root: Path,
    recursive: bool,
    exts: Sequence[str] = SUPPORTED_EXTS,
) -> list[Path]:
    """Return a deterministically sorted list of image files under ``input_root``.

    Skips an ``output``-style sibling automatically only if it is *inside* the
    input root; callers are still responsible for choosing a separate output dir.
    """
    input_root = Path(input_root)
    exts = normalize_exts(exts)
    globber = input_root.rglob("*") if recursive else input_root.glob("*")
    found = [
        p
        for p in globber
        if p.is_file() and p.suffix.lower() in exts
    ]
    # Deterministic order so logs / preview indices are stable across runs.
    return sorted(found, key=lambda p: str(p).lower())


def relative_output_path(src: Path, input_root: Path, output_root: Path) -> Path:
    """Mirror the input sub-folder structure under the output root, keeping the
    original file name unchanged (so downstream RealityScan/COLMAP see the same
    relative layout)."""
    src = Path(src)
    rel = src.relative_to(Path(input_root))
    return Path(output_root) / rel


def read_metadata(path: Path) -> dict:
    """Read lightweight metadata without decoding pixels twice."""
    with Image.open(path) as im:
        exif_bytes = b""
        try:
            exif = im.getexif()
            if exif:
                exif_bytes = exif.tobytes()
        except Exception:  # noqa: BLE001 - corrupt EXIF must not break the batch
            exif_bytes = b""
        return {
            "size": im.size,  # (width, height)
            "mode": im.mode,
            "format": im.format,
            "icc_profile": im.info.get("icc_profile"),
            "exif_bytes": exif_bytes,
        }


def load_rgb(path: Path) -> Image.Image:
    """Load an image and convert to RGB (drops alpha / palette / CMYK)."""
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


@functools.lru_cache(maxsize=1)
def find_exiftool() -> str | None:
    """Locate an ``exiftool`` executable: PATH first, then a bundled ``.tools`` copy.

    When present (and enabled), exiftool does a full ``-TagsFromFile`` copy — maker
    notes, GPS, lens/IPTC/XMP — which is stronger than the PIL/piexif fallback,
    especially for PNG/TIFF. Returns ``None`` if not found.
    """
    p = shutil.which("exiftool")
    if p:
        return p
    root = Path(__file__).resolve().parents[1]
    for cand in (root / ".tools" / "exiftool").rglob("exiftool.exe"):
        return str(cand)
    return None


def save_processed(
    src_path: Path,
    img: Image.Image,
    dst_path: Path,
    jpeg_quality: int = 95,
    use_exiftool: bool = False,
) -> None:
    """Save ``img`` to ``dst_path`` preserving format/EXIF/ICC from ``src_path``.

    ``dst_path`` extension is expected to match ``src_path`` extension (the batch
    layer enforces format preservation). JPEG is written at quality>=95 with 4:4:4
    chroma (subsampling=0).

    Metadata strategy:
      * ``use_exiftool=True`` and exiftool is available -> full ``-TagsFromFile``
        copy of *all* metadata (best fidelity, all formats), run per file.
      * otherwise -> PIL writes the standard EXIF block, and for JPEG->JPEG a
        faithful ``piexif.transplant`` copies the full EXIF segment (fast default).
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ext = dst_path.suffix.lower()

    # Pull EXIF + ICC from the source.
    with Image.open(src_path) as src_im:
        try:
            exif = src_im.getexif()
            exif_bytes = exif.tobytes() if exif else None
        except Exception:  # noqa: BLE001
            exif_bytes = None
        icc = src_im.info.get("icc_profile")

    save_kwargs: dict = {}
    if icc:
        save_kwargs["icc_profile"] = icc
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes

    if ext in _JPEG_EXTS:
        img.save(
            dst_path,
            format="JPEG",
            quality=max(95, int(jpeg_quality)),
            subsampling=0,  # 4:4:4 — no chroma loss
            optimize=True,
            **save_kwargs,
        )
    elif ext in _TIFF_EXTS:
        img.save(dst_path, format="TIFF", compression="tiff_lzw", **save_kwargs)
    elif ext == ".png":
        img.save(dst_path, format="PNG", compress_level=6, **save_kwargs)
    else:
        # Unknown extension: fall back to a lossless PNG next to it.
        dst_path = dst_path.with_suffix(".png")
        img.save(dst_path, format="PNG", **save_kwargs)

    # --- Metadata copy ------------------------------------------------------
    et = find_exiftool() if use_exiftool else None
    if et:
        try:
            subprocess.run(
                [et, "-q", "-q", "-overwrite_original", "-TagsFromFile", str(src_path),
                 "-all:all", "-icc_profile", str(dst_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=False,
            )
        except Exception:  # noqa: BLE001 - metadata copy must never break the batch
            pass
    elif dst_path.suffix.lower() in _JPEG_EXTS and src_path.suffix.lower() in _JPEG_EXTS:
        try:
            import piexif

            piexif.transplant(str(src_path), str(dst_path))
        except Exception:  # noqa: BLE001 - keep the PIL-written EXIF if this fails
            pass
