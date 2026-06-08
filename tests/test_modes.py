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
    resolve_mode_defaults, pending_outputs, mask_ratio_warning_level, resolve_workers,
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


# --- mode defaults are enforced in BatchConfig.__post_init__ (self-consistent) --- #
def test_reflectmask_config_enables_realityscan():
    c = _cfg(mode="reflectmask")
    assert c.realityscan is True and c.write_cleaned is False


def test_diagnostic_config_enables_previews_and_masks():
    c = _cfg(mode="diagnostic")
    assert c.realityscan is True and c.make_preview is True and c.heatmap is True
    assert c.write_cleaned is False


def test_clean_config_has_no_masks_unless_requested():
    assert _cfg(mode="clean").realityscan is False
    assert _cfg(mode="clean", realityscan=True).realityscan is True


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


def test_backend_defaults_to_unreflect():
    assert _cfg().backend == "unreflect"
    assert _cfg(backend="luma").backend == "luma"


def test_config_model_max_size_default():
    assert _cfg().model_max_size == 2048
    assert _cfg(model_max_size=None).model_max_size is None


# --- parallel worker count resolution (pure / injectable) --- #
# signature: resolve_workers(requested, n_images, backend, cores, free_vram_gb, free_ram_gb)
def test_resolve_workers_explicit_capped_by_resources():
    assert resolve_workers(8, 100, "luma", 192, None, None) == 8       # explicit, plenty of room
    assert resolve_workers(100, 100, "luma", 192, None, None) == 61    # Windows 61 cap
    assert resolve_workers(100, 10, "luma", 192, None, None) == 10     # <= n_images
    assert resolve_workers(8, 100, "luma", 4, None, None) == 4         # <= cores


def test_resolve_workers_auto_luma_targets_io_knee():
    assert resolve_workers(None, 1000, "luma", 192, None, None) == 32


def test_resolve_workers_auto_ai_resource_derived():
    # big batch: VRAM-derived (free/4.5), capped by the conservative ceiling (16)
    assert resolve_workers(None, 1000, "unreflect", 192, 88, None) == 16   # 88/4.5=19 -> ceiling 16
    assert resolve_workers(None, 1000, "unreflect", 192, 20, None) == 4    # 20/4.5=4 (VRAM binds)
    assert resolve_workers(None, 1000, "unreflect", 192, None, None) == 16  # no VRAM info -> ceiling


def test_resolve_workers_ai_scales_with_batch():
    # >= ~6 images/worker; tiny batches stay (near-)sequential -> no regression
    assert resolve_workers(None, 5, "unreflect", 192, 88, None) == 1       # <6 -> 1
    assert resolve_workers(None, 16, "unreflect", 192, 88, None) == 2      # 16//6 = 2
    assert resolve_workers(None, 24, "unreflect", 192, 88, None) == 4      # 24//6 = 4
    assert resolve_workers(None, 48, "unreflect", 192, 88, None) == 8      # 48//6 = 8 (measured 2.9x)
    assert resolve_workers(None, 200, "unreflect", 192, 88, None) == 16    # capped at the ceiling


def test_resolve_workers_capped_by_free_ram():
    # RAM cap (free_ram / 1.5) protects small machines from OOM (both auto and explicit)
    assert resolve_workers(None, 1000, "luma", 192, None, 6.0) == 4        # 6/1.5 = 4
    assert resolve_workers(None, 1000, "unreflect", 192, 88, 3.0) == 2     # 3/1.5 = 2
    assert resolve_workers(32, 1000, "luma", 192, None, 6.0) == 4          # explicit also RAM-capped


def test_resolve_workers_explicit_never_ooms():
    assert resolve_workers(32, 500, "unreflect", 192, 20, None) == 4       # explicit capped by VRAM
    assert resolve_workers(6, 500, "unreflect", 192, 88, None) == 6        # explicit respected under VRAM


def test_resolve_workers_single_image_is_serial():
    assert resolve_workers(None, 1, "luma", 192, None, None) == 1
    assert resolve_workers(16, 1, "unreflect", 192, 88, None) == 1
