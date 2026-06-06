from pathlib import Path

from PIL import Image

from src import image_io


def _make(path: Path, size=(8, 8), color=(120, 120, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_normalize_exts_defaults_and_dotting():
    assert image_io.normalize_exts(None) == image_io.SUPPORTED_EXTS
    assert image_io.normalize_exts(["JPG", ".PnG"]) == (".jpg", ".png")


def test_iter_images_non_recursive_filters_extensions(tmp_path):
    _make(tmp_path / "a.jpg")
    _make(tmp_path / "b.PNG")
    (tmp_path / "notes.txt").write_text("x")
    _make(tmp_path / "sub" / "c.tif")

    found = image_io.iter_images(tmp_path, recursive=False)
    names = {p.name for p in found}
    assert names == {"a.jpg", "b.PNG"}  # sub/ excluded, .txt excluded


def test_iter_images_recursive(tmp_path):
    _make(tmp_path / "a.jpg")
    _make(tmp_path / "sub" / "c.tiff")
    found = image_io.iter_images(tmp_path, recursive=True)
    assert {p.name for p in found} == {"a.jpg", "c.tiff"}


def test_relative_output_path_mirrors_tree(tmp_path):
    src = tmp_path / "in" / "day1" / "img.jpg"
    out = image_io.relative_output_path(src, tmp_path / "in", tmp_path / "out")
    assert out == tmp_path / "out" / "day1" / "img.jpg"


def test_save_processed_preserves_format_and_dims(tmp_path):
    src = tmp_path / "in.jpg"
    _make(src, size=(16, 10))
    out = tmp_path / "out.jpg"
    img = Image.new("RGB", (16, 10), (10, 200, 10))
    image_io.save_processed(src, img, out, jpeg_quality=95)
    assert out.exists()
    with Image.open(out) as o:
        assert o.format == "JPEG"
        assert o.size == (16, 10)


def test_save_processed_png_roundtrip(tmp_path):
    src = tmp_path / "in.png"
    _make(src, size=(12, 12))
    out = tmp_path / "nested" / "out.png"
    image_io.save_processed(src, Image.new("RGB", (12, 12)), out)
    assert out.exists()
    with Image.open(out) as o:
        assert o.format == "PNG"
        assert o.size == (12, 12)
