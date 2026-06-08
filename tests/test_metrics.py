import numpy as np

from src import metrics


def _solid(value, size=(10, 10)):
    return np.full((size[1], size[0], 3), value, dtype=np.uint8)


def test_mean_luminance_extremes():
    assert metrics.mean_luminance(_solid(0)) == 0.0
    assert metrics.mean_luminance(_solid(255)) == 255.0


def test_highlight_ratio():
    assert metrics.highlight_ratio(_solid(255)) == 1.0
    assert metrics.highlight_ratio(_solid(0)) == 0.0
    # half highlight, half dark
    img = np.concatenate([_solid(255, (10, 5)), _solid(0, (10, 5))], axis=0)
    assert abs(metrics.highlight_ratio(img) - 0.5) < 1e-6


def test_compute_pair_metrics_signs():
    before = _solid(200)
    after = _solid(150)  # darker -> highlights reduced
    m = metrics.compute_pair_metrics(before, after)
    assert m["mean_luma_before"] == 200.0
    assert m["mean_luma_after"] == 150.0
    assert m["mean_luma_delta"] == -50.0
    assert m["mean_abs_diff"] == 50.0


def test_compute_pair_metrics_handles_shape_mismatch():
    before = _solid(100, (10, 10))
    after = _solid(100, (8, 8))
    m = metrics.compute_pair_metrics(before, after)  # must not raise
    assert m["mean_abs_diff"] == 0.0


def test_diff_heatmap_shape_and_dtype():
    before = _solid(100)
    after = _solid(120)
    hm = metrics.diff_heatmap(before, after)
    assert hm.shape == (10, 10, 3)
    assert hm.dtype == np.uint8


def test_luminance_composite_gating():
    # Dark original (below highlight level) -> keep original, ignore diffuse.
    out = metrics.luminance_composite(_solid(50), _solid(200), level=235.0, dilation=0, feather=0)
    assert int(round(out.mean())) == 50
    # Fully-bright original -> replace with diffuse.
    out2 = metrics.luminance_composite(_solid(255), _solid(100), level=235.0, dilation=0, feather=0)
    assert int(round(out2.mean())) == 100


def test_change_mask_identical_is_empty():
    before = _solid(100)
    assert int(metrics.change_mask(before, before).max()) == 0
    after = _solid(200)
    assert int(metrics.change_mask(before, after).max()) == 255


def test_reflection_candidate_mask_polarity():
    before = _solid(255, (10, 10))
    after = before.copy()
    after[:, :5] = 100  # left half darkened (a removed highlight)
    c = metrics.reflection_candidate_mask(before, after, drop_level=12, highlight_gate=200)
    assert int(c[:, :5].min()) == 255   # candidate reflection
    assert int(c[:, 5:].max()) == 0     # not a candidate


def test_luma_candidate_mask_polarity():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :4] = 255
    c = metrics.luma_candidate_mask(img, level=200)
    assert int(c[:, :4].min()) == 255 and int(c[:, 4:].max()) == 0


def test_mask_overlay_tints_only_excluded():
    img = np.full((6, 6, 3), 100, dtype=np.uint8)
    mask = np.full((6, 6), 255, dtype=np.uint8)
    mask[:, :3] = 0  # left half excluded
    out = metrics.mask_overlay(img, mask, color=(255, 0, 0), alpha=0.5)
    assert (out[:, 3:] == 100).all()          # kept region unchanged
    assert out[:, :3, 0].mean() > 100         # excluded tinted toward red (R up)
    assert out[:, :3, 1].mean() < 100         # G down


def test_morph_to_mask_matches_reflection_exclusion_mask():
    # Equivalence: candidate + morph_to_mask == the all-in-one mask (no downscale path).
    before = _solid(255, (20, 20))
    after = before.copy()
    after[5:15, 5:15] = 100
    direct = metrics.reflection_exclusion_mask(
        before, after, drop_level=12, highlight_gate=200, dilation=2, open_radius=1)
    cand = metrics.reflection_candidate_mask(before, after, drop_level=12, highlight_gate=200)
    composed = metrics.morph_to_mask(cand, dilation=2, open_radius=1)
    assert (direct == composed).all()


def test_morph_to_mask_matches_luma_exclusion_mask():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[2:10, 2:10] = 255
    direct = metrics.luma_exclusion_mask(img, level=200, dilation=2, open_radius=1)
    cand = metrics.luma_candidate_mask(img, level=200)
    composed = metrics.morph_to_mask(cand, dilation=2, open_radius=1)
    assert (direct == composed).all()


def test_morph_to_mask_polarity_and_stats():
    cand = np.zeros((10, 10), dtype=np.uint8)
    cand[:, :5] = 255  # left half candidate
    m, st = metrics.morph_to_mask(cand, dilation=0, open_radius=0, return_stats=True)
    assert int(m[:, :5].max()) == 0 and int(m[:, 5:].min()) == 255
    assert abs(st["candidate_pixel_ratio"] - 50.0) < 1e-6
    assert abs(st["final_mask_ratio"] - 50.0) < 1e-6
