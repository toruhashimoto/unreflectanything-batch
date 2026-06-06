"""Generate the README demo image from a SYNTHETIC specular scene (reproducible).

Run from the repo root with the project venv (and weights already downloaded via
`unreflectanything download --weights`):

    python examples/make_demo.py

It synthesizes a textured scene with bright specular "glare" blobs, runs this tool
(`--mask-composite --make-preview --heatmap`), and writes the side-by-side strip
[Original | UnReflect | Diff] to examples/demo_before_after.jpg.

Synthetic input is used deliberately so nothing private is published; results on real
photos look better.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)


def synth_scene(w: int = 840, h: int = 560) -> Image.Image:
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Wall (top) -> floor (bottom) vertical gradient.
    grad = 70 + 120 * (yy / h)
    # Fine texture so the scene has matchable detail.
    noise = rng.normal(0, 10, (h, w)).astype(np.float32)
    base = np.clip(grad + noise, 0, 255)
    img = np.stack([base * 0.92, base * 0.95, base], axis=-1)  # slightly cool
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(pil)
    horizon = int(h * 0.42)
    # Perspective floor grid.
    for i in range(-6, 7):
        x0 = w / 2 + i * 26
        draw.line([(w / 2 + i * 110, horizon), (x0 * 0.2 + w * 0.4, h)], fill=(60, 64, 70), width=2)
    yk = horizon
    step = 10
    while yk < h:
        draw.line([(0, yk), (w, yk)], fill=(60, 64, 70), width=1)
        step = int(step * 1.18) + 2
        yk += step
    # A couple of "objects" (boxes) on the wall for structure.
    draw.rectangle([90, 120, 220, 250], outline=(40, 40, 48), width=3)
    draw.rectangle([560, 90, 690, 210], outline=(40, 40, 48), width=3)

    # Bright specular glares (near-white, soft) — the highlights to be removed.
    glare = Image.new("L", (w, h), 0)
    gd = ImageDraw.Draw(glare)
    for (cx, cy, rx, ry) in [(180, 180, 46, 30), (620, 150, 38, 26),
                              (410, 360, 60, 34), (300, 470, 34, 22)]:
        gd.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    glare = glare.filter(ImageFilter.GaussianBlur(7))
    arr = np.asarray(pil).astype(np.float32)
    m = (np.asarray(glare).astype(np.float32) / 255.0)[..., None]
    arr = arr * (1 - m) + np.float32(255) * m  # blow out to white in glare regions
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def main() -> int:
    src = ASSETS / "demo_input.png"
    synth_scene().save(src)
    print("synthetic scene ->", src)

    out = Path(tempfile.mkdtemp(prefix="unreflect_demo_"))
    cmd = [sys.executable, str(ROOT / "main.py"), "-i", str(ASSETS), "-o", str(out),
           "--make-preview", "--heatmap", "--mask-composite", "--no-progress"]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=False)

    strips = list((out / "preview_compare").glob("demo_input*.jpg"))
    if not strips:
        print("ERROR: no preview produced (are the weights downloaded?)")
        return 1
    strip = Image.open(strips[0]).convert("RGB")
    if strip.width > 1100:
        strip = strip.resize((1100, round(strip.height * 1100 / strip.width)),
                             getattr(Image, "Resampling", Image).LANCZOS)
    dst = HERE / "demo_before_after.jpg"
    strip.save(dst, quality=88)
    print("demo image ->", dst, strip.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
