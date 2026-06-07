"""Tests for the v0.2 product modes (ReflectMask is the primary product) — torch-free.

These lock the three facts the reorganization must not regress on:
  * the mode -> config mapping (reflectmask/diagnostic are mask-first; clean exports images),
  * the mask-first no-overwrite skip logic (keys off the mask, not the cleaned image),
  * the mask-area warning classifier (over-masking is a danger signal, not success).
"""
from pathlib import Path

import pytest

from src.unreflect_batch import (
    BatchConfig, MODES, DEFAULT_MODE,
    resolve_mode_defaults, pending_outputs, mask_ratio_warning_level,
)


def _cfg(**kw) -> BatchConfig:
    kw.setdefault("input_dir", "in")
    kw.setdefault("output_dir", "out")
    return BatchConfig(**kw)


# --------------------------------------------------------------------------- #
# mode -> config defaults                                                       #
# --------------------------------------------------------------------------- #
def test_default_mode_is_reflectmask():
    assert DEFAULT_MODE == "reflectmask"
    assert set(MODES) == {"reflectmask", "diagnostic", "clean"}


def test_reflectmask_mode_is_mask_first():
    d = resolve_mode_defaults("reflectmask")
    assert d["realityscan"] is True
    assert d["write_cleaned"] is False
    # The cleaned image is NOT written: the deliverable is the RealityScan mask.
    assert _cfg(mode="reflectmask").write_cleaned is False


def test_diagnostic_mode_adds_previews_but_no_cleaned():
    d = resolve_mode_defaults("diagnostic")
    assert d["realityscan"] is True and d["write_cleaned"] is False
    assert d["make_preview"] is True and d["heatmap"] is True
    assert _cfg(mode="diagnostic").write_cleaned is False


def test_clean_mode_writes_cleaned_and_no_masks_by_default():
    d = resolve_mode_defaults("clean")
    assert d["write_cleaned"] is True
    assert d["realityscan"] is False
    assert _cfg(mode="clean").write_cleaned is True


def test_unknown_mode_falls_back_to_reflectmask():
    assert resolve_mode_defaults("nope")["write_cleaned"] is False


def test_explicit_write_cleaned_overrides_mode():
    assert _cfg(mode="reflectmask", write_cleaned=True).write_cleaned is True
    assert _cfg(mode="clean", write_cleaned=False).write_cleaned is False


# --------------------------------------------------------------------------- #
# mask-first skip logic                                                         #
# --------------------------------------------------------------------------- #
def test_pending_outputs_reflectmask_keys_off_mask():
    dst = Path("out/IMG.jpg")
    mask = Path("out/realityscan/IMG.jpg.mask.png")
    # Mask-first: the (never-written) cleaned image must not count toward 'done'.
    assert pending_outputs(False, True, dst, mask) == [mask]


def test_pending_outputs_clean_keys_off_cleaned():
    dst = Path("out/IMG.jpg")
    assert pending_outputs(True, False, dst, None) == [dst]


def test_pending_outputs_clean_plus_masks_requires_both():
    dst = Path("out/IMG.jpg")
    mask = Path("out/realityscan/IMG.jpg.mask.png")
    assert pending_outputs(True, True, dst, mask) == [dst, mask]


def test_pending_outputs_never_empty():
    dst = Path("out/IMG.jpg")
    assert pending_outputs(False, False, dst, None) == [dst]


# --------------------------------------------------------------------------- #
# mask-area warning classifier (warn > 5%, danger > 12%)                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pct,level", [
    (0.0, "ok"), (5.0, "ok"), (5.01, "warning"),
    (12.0, "warning"), (12.5, "danger"), (40.0, "danger"),
])
def test_mask_ratio_warning_level(pct, level):
    assert mask_ratio_warning_level(pct) == level


def test_mask_ratio_warning_level_custom_thresholds():
    assert mask_ratio_warning_level(7.0, warn=10.0, danger=20.0) == "ok"
    assert mask_ratio_warning_level(15.0, warn=10.0, danger=20.0) == "warning"
    assert mask_ratio_warning_level(25.0, warn=10.0, danger=20.0) == "danger"


def test_config_has_default_mask_thresholds():
    c = _cfg()
    assert c.mask_warn_pct == 5.0 and c.mask_danger_pct == 12.0
