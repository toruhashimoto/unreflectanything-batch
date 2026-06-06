# Contributing

Thanks for your interest! This project is a small, focused **wrapper** around the
[`unreflectanything`](https://github.com/alberto-rota/UnReflectAnything) package for
3D Gaussian Splatting / photogrammetry pre-processing. Contributions that improve the
wrapper, docs, tests, or the A/B harnesses are welcome.

## Scope

- This repo wraps and *calls* the model; it does **not** vendor or modify upstream
  source. Changes to the removal model itself belong upstream.
- Keep the originals-are-sacred guarantees intact: never modify inputs; preserve
  dimensions, EXIF (focal length), ICC, and file format; continue-on-error.

## Dev setup

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Gui
```

Two install traps to be aware of (see README §2):
1. **torch** must come from the CUDA index (`--index-url .../whl/cu128`) on Windows;
   the default PyPI wheel is CPU-only and Blackwell needs cu128.
2. **transformers** must be the pinned commit in `requirements.txt`, or the checkpoint
   fails to load (`Missing key(s) ... dinov3`).

## Tests

The unit tests are intentionally **torch-free** (the heavy imports are lazy), so they
run fast and in CI without a GPU:

```bash
python -m pip install pillow numpy pytest
python -m pytest
```

Please add/extend tests under `tests/` for any change to the pure-Python utilities
(`src/image_io.py`, `src/metrics.py`, `src/preview.py`, `src/logger.py`). CI runs the
suite on Ubuntu + Windows × Python 3.11/3.12.

## Style

- Match the surrounding code (naming, comment density, type hints).
- Keep heavy imports (`torch`, `unreflectanything`) lazy/inside functions so the
  utility modules and tests stay importable without the ML stack.

## Pull requests

- Keep PRs focused; describe the user-facing change and how you tested it.
- Update `README.md` / `examples/` if behavior changes.
- By contributing, you agree your contributions are licensed under the project's
  [MIT License](LICENSE).
