from PIL import Image

from src import preview


def test_make_compare_two_panels():
    before = Image.new("RGB", (100, 50), (10, 10, 10))
    after = Image.new("RGB", (100, 50), (200, 200, 200))
    out = preview.make_compare(before, after, max_height=50)
    # two equal panels (width 100 each) + one gap of 8
    assert out.width == 100 + 100 + preview._GAP
    # each panel is image height + the label bar
    assert out.height == 50 + preview._LABEL_BAR


def test_make_compare_with_heatmap_three_panels():
    before = Image.new("RGB", (40, 40))
    after = Image.new("RGB", (40, 40))
    heat = Image.new("RGB", (40, 40))
    out = preview.make_compare(before, after, heat, max_height=40)
    assert out.width == 40 * 3 + preview._GAP * 2


def test_make_compare_downscales_to_max_height():
    before = Image.new("RGB", (400, 400))
    after = Image.new("RGB", (400, 400))
    out = preview.make_compare(before, after, max_height=100)
    assert out.height == 100 + preview._LABEL_BAR


def test_make_grid_six_panels_two_rows():
    imgs = [(f"p{i}", Image.new("RGB", (40, 40))) for i in range(6)]
    out = preview.make_grid(imgs, cols=3, max_height=40)
    cell_h = 40 + preview._LABEL_BAR
    assert out.width == 40 * 3 + preview._GAP * 2   # 3 columns
    assert out.height == cell_h * 2 + preview._GAP  # 2 rows


def test_make_diagnostic_is_three_by_two():
    p = Image.new("RGB", (40, 40))
    gray = Image.new("L", (40, 40))  # masks come in as L; must be accepted
    out = preview.make_diagnostic(p, p, p, gray, gray, p, max_height=40)
    cell_h = 40 + preview._LABEL_BAR
    assert out.width == 40 * 3 + preview._GAP * 2
    assert out.height == cell_h * 2 + preview._GAP
