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
