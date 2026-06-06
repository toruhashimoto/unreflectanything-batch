"""Tests for the RealityScan mask helpers and the model-based exclusion mask.

Torch-free: these exercise only numpy / PIL / OpenCV logic, mirroring the rest of the
unit suite. They lock in the two facts that matter for RealityScan interop — the exact
mask file name and the white=keep / black=exclude polarity — so a regression can't
silently produce masks RealityScan ignores or inverts.
"""
import numpy as np
import pytest
from PIL import Image

from src import metrics, realityscan


def _solid(value, size=(10, 10)):
    return np.full((size[1], size[0], 3), value, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Naming convention                                                            #
# --------------------------------------------------------------------------- #
def test_mask_filename_same_folder_form():
    # Full original name INCLUDING extension, then .mask.png.
    assert realityscan.mask_filename("IMG_1234.jpg") == "IMG_1234.jpg.mask.png"
    assert realityscan.mask_filename("frame-00000.png") == "frame-00000.png.mask.png"


def test_mask_filename_alternate_separators():
    assert realityscan.mask_filename("a.jpg", separator="_") == "a.jpg_mask.png"
    for sep in realityscan.VALID_SEPARATORS:
        assert realityscan.mask_filename("a.jpg", separator=sep) == f"a.jpg{sep}mask.png"


def test_mask_filename_rejects_bad_separator():
    with pytest.raises(ValueError):
        realityscan.mask_filename("a.jpg", separator="-")


# --------------------------------------------------------------------------- #
# PNG writing: strictly binary, correct size                                   #
# --------------------------------------------------------------------------- #
def test_save_mask_png_is_binary_grayscale(tmp_path):
    # A soft/gray mask must be hard-binarised on write (RealityScan discourages gray).
    soft = np.linspace(0, 255, 10 * 10, dtype=np.uint8).reshape(10, 10)
    dst = tmp_path / "m.png"
    realityscan.save_mask_png(soft, dst)
    out = Image.open(dst)
    assert out.mode == "L"
    vals = set(np.unique(np.asarray(out)).tolist())
    assert vals.issubset({0, 255})


def test_save_mask_png_resizes_to_like_size(tmp_path):
    mask = np.zeros((4, 6), dtype=np.uint8)  # (h=4, w=6)
    dst = tmp_path / "m.png"
    realityscan.save_mask_png(mask, dst, like_size=(12, 8))  # (w=12, h=8)
    assert Image.open(dst).size == (12, 8)


def test_copy_source_image_is_byte_exact(tmp_path):
    src = tmp_path / "in.png"
    Image.fromarray(_solid(123), "RGB").save(src)
    dst = tmp_path / "sub" / "in.png"
    realityscan.copy_source_image(src, dst)
    assert dst.read_bytes() == src.read_bytes()


# --------------------------------------------------------------------------- #
# Model-based exclusion mask: polarity + gating                                #
# --------------------------------------------------------------------------- #
def test_exclusion_mask_no_change_keeps_everything():
    # Model changed nothing -> nothing excluded -> all white (kept).
    img = _solid(255)
    m = metrics.reflection_exclusion_mask(img, img)
    assert int(m.min()) == 255 and int(m.max()) == 255


def test_exclusion_mask_marks_removed_highlight_black():
    # A bright region the model darkened must become black (excluded); the rest white.
    before = _solid(255, (10, 10))
    after = before.copy()
    after[:, :5] = 100  # left half darkened by the model (a removed highlight)
    m = metrics.reflection_exclusion_mask(
        before, after, drop_level=12, highlight_gate=200, dilation=0, open_radius=0
    )
    assert int(m[:, :5].max()) == 0      # excluded
    assert int(m[:, 5:].min()) == 255    # kept


def test_exclusion_mask_brightness_gate_protects_dark_areas():
    # Same luma drop, but on a DARK original: with the gate on it must NOT be excluded.
    before = _solid(80, (10, 10))
    after = _solid(40, (10, 10))  # darkened by 40, but original was dark
    gated = metrics.reflection_exclusion_mask(
        before, after, drop_level=12, highlight_gate=200, dilation=0, open_radius=0
    )
    assert int(gated.min()) == 255  # nothing excluded (below the brightness gate)
    # With the gate disabled, the same drop IS excluded.
    ungated = metrics.reflection_exclusion_mask(
        before, after, drop_level=12, highlight_gate=0, dilation=0, open_radius=0
    )
    assert int(ungated.max()) == 0


def test_exclusion_mask_is_strictly_binary():
    before = _solid(255, (16, 16))
    after = before.copy()
    after[4:12, 4:12] = 90
    m = metrics.reflection_exclusion_mask(before, after, highlight_gate=200)
    assert set(np.unique(m).tolist()).issubset({0, 255})
