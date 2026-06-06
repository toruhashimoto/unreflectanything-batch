"""A/B 3D Gaussian Splatting comparison.

End-to-end, for two or more named image sets (e.g. `original` vs a reflection-removed
`cleaned`):

    image set  ->  COLMAP sparse SfM  ->  LichtFeld Studio headless 3DGS training
               ->  eval renders       ->  same-viewpoint comparison figures + report

So you can SEE and measure whether reflection removal produces a better/cleaner 3D
Gaussian Splatting scene (less baked-in glare, fewer Gaussians) before committing.

This is environment-agnostic: external tools are resolved from flags, environment
variables, or PATH — nothing is hard-coded.

REQUIREMENTS (external, not bundled):
  * COLMAP            https://github.com/colmap/colmap/releases   (--colmap / $COLMAP_EXE)
  * LichtFeld Studio  https://github.com/MrNeRF/LichtFeld-Studio  (--lichtfeld / $LICHTFELD_EXE)
  * Python deps: pillow, numpy (already required by this project)

EXAMPLE:
    python tools/ab_3dgs.py --work ab3dgs_out \
        --set original  "D:/photos/original" \
        --set cleaned   "D:/photos/unreflect" \
        --shared-poses original \
        --lichtfeld "C:/path/LichtFeld-Studio.exe" \
        --steps-scaler 0.5 --resize-factor 2

`--shared-poses NAME` trains every set on NAME's camera poses (requires identically
named, pixel-aligned images across sets, e.g. original vs cleaned versions of the same
frames). This isolates *appearance* and makes renders directly comparable per frame.
Omit it to run a full independent pipeline per set.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_colmap import find_colmap, reconstruct  # noqa: E402  (sibling module)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# Tool / path helpers                                                           #
# --------------------------------------------------------------------------- #
def find_lichtfeld(explicit: str | None) -> str:
    """Resolve LichtFeld-Studio: --lichtfeld / $LICHTFELD_EXE / PATH (file or dir)."""
    for cand in (explicit, os.environ.get("LICHTFELD_EXE"), os.environ.get("LICHTFELD_STUDIO")):
        if cand:
            p = Path(cand)
            if p.is_dir():
                hits = list(p.rglob("LichtFeld-Studio*")) or list(p.rglob("lichtfeld-studio*"))
                hits = [h for h in hits if h.is_file() and h.suffix.lower() in ("", ".exe")]
                if hits:
                    return str(hits[0])
            return cand
    for name in ("LichtFeld-Studio", "lichtfeld-studio", "LichtFeld-Studio.exe"):
        p = shutil.which(name)
        if p:
            return p
    raise SystemExit("LichtFeld Studio not found — pass --lichtfeld <path> or set "
                     "$LICHTFELD_EXE (https://github.com/MrNeRF/LichtFeld-Studio)")


def link_or_copy_dir(src: Path, dst: Path) -> None:
    """Make dst point at src's contents cheaply: directory junction/symlink, else copy."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            # Junction needs no admin rights, unlike a Windows symlink.
            subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            os.symlink(src, dst, target_is_directory=True)
        return
    except Exception:  # noqa: BLE001 - fall back to a real copy
        pass
    shutil.copytree(src, dst)


def count_images(d: Path) -> int:
    return sum(1 for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS)


# --------------------------------------------------------------------------- #
# Training                                                                      #
# --------------------------------------------------------------------------- #
def assemble_dataset(work: Path, name: str, images_dir: Path, sparse_dir: Path) -> Path:
    """Build a LichtFeld/COLMAP dataset: <ds>/images + <ds>/sparse/0."""
    ds = work / "datasets" / name
    if (ds / "sparse" / "0").exists():
        shutil.rmtree(ds / "sparse", ignore_errors=True)
    (ds / "sparse").mkdir(parents=True, exist_ok=True)
    shutil.copytree(sparse_dir, ds / "sparse" / "0")
    link_or_copy_dir(Path(images_dir), ds / "images")
    return ds


def train(lichtfeld: str, data_path: Path, out_path: Path, *, iters: int | None,
          steps_scaler: float | None, resize_factor: int, test_every: int,
          undistort: bool, extra: list[str], log_path: Path) -> int:
    cmd = [lichtfeld, "--headless", "--data-path", str(data_path),
           "--output-path", str(out_path), "--resize_factor", str(resize_factor),
           "--eval", "--save-eval-images", "--test-every", str(test_every)]
    if undistort:
        cmd.append("--undistort")
    if iters:
        cmd += ["--iter", str(iters)]
    if steps_scaler:
        cmd += ["--steps-scaler", str(steps_scaler)]
    cmd += extra
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1")
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


_METRIC_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")


def parse_metrics(out_path: Path) -> dict:
    """Read the final PSNR/SSIM/#Gaussians row from LichtFeld's metrics_report.txt."""
    report = out_path / "metrics_report.txt"
    last = None
    if report.exists():
        for line in report.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _METRIC_ROW.match(line)
            if m:
                last = m
    if not last:
        return {"iter": None, "psnr": None, "ssim": None, "gaussians": None}
    return {"iter": int(last.group(1)), "psnr": float(last.group(2)),
            "ssim": float(last.group(3)), "gaussians": int(last.group(5))}


def latest_eval_dir(out_path: Path) -> Path | None:
    evs = sorted(out_path.glob("eval_step_*"), key=lambda p: int(p.name.split("_")[-1]))
    return evs[-1] if evs else None


# --------------------------------------------------------------------------- #
# Comparison figures (frames matched by GT content, not by index)              #
# --------------------------------------------------------------------------- #
_GAP, _BAR, _BG, _FG = 8, 22, (245, 245, 245), (20, 20, 20)


def _resample():
    return getattr(Image, "Resampling", Image).LANCZOS


def _eval_halves(p: Path):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    m = w // 2
    return im.crop((0, 0, m, h)), im.crop((m, 0, w, h))  # (GT, render)


def _fit(im: Image.Image, h: int) -> Image.Image:
    if im.height == h:
        return im
    return im.resize((max(1, round(im.width * h / im.height)), h), _resample())


def _label(panel: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (panel.width, panel.height + _BAR), _BG)
    out.paste(panel, (0, _BAR))
    ImageDraw.Draw(out).text((4, 4), text, fill=_FG)
    return out


def _strip(items, panel_h: int) -> Image.Image:
    panels = [_label(_fit(im, panel_h), lab) for im, lab in items]
    w = sum(p.width for p in panels) + _GAP * (len(panels) - 1)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (w, h), _BG)
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + _GAP
    return canvas


def _gt_signature(p: Path) -> np.ndarray:
    gt, _ = _eval_halves(p)
    return np.asarray(gt.resize((96, 54)).convert("L"), dtype=np.float32)


def build_comparison(eval_by_name: dict[str, Path], anchor: str, out_dir: Path,
                     panel_h: int = 380) -> int:
    """For each anchor val frame, match the same frame in the other sets by GT content
    and compose [GT(anchor) | render(set1) | render(set2) | ...]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [anchor] + [n for n in eval_by_name if n != anchor]
    anchor_imgs = sorted(eval_by_name[anchor].glob("*.png"), key=lambda p: int(p.stem))
    sigs = {n: {p: _gt_signature(p) for p in eval_by_name[n].glob("*.png")} for n in names}

    used = {n: set() for n in names}
    n_written = 0
    for ap in anchor_imgs:
        row = []
        gt_a, ren_a = _eval_halves(ap)
        row.append((gt_a, f"GT ({anchor})"))
        row.append((ren_a, f"3DGS: {anchor}"))
        ok = True
        for n in names[1:]:
            best, bd = None, 1e18
            for cp, sg in sigs[n].items():
                if cp in used[n]:
                    continue
                d = float(((sigs[anchor][ap] - sg) ** 2).mean())
                if d < bd:
                    bd, best = d, cp
            if best is None:
                ok = False
                break
            used[n].add(best)
            _, ren = _eval_halves(best)
            row.append((ren, f"3DGS: {n}"))
        if not ok:
            continue
        _strip(row, panel_h).save(out_dir / f"cmp_{n_written:02d}.jpg", quality=92)
        n_written += 1
    return n_written


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A/B 3D Gaussian Splatting comparison "
                                             "(COLMAP -> LichtFeld -> same-view figures)")
    ap.add_argument("--work", required=True, type=Path, help="working/output directory")
    ap.add_argument("--set", nargs=2, action="append", metavar=("NAME", "DIR"), required=True,
                    help="a named image set (repeatable), e.g. --set original D:/imgs")
    ap.add_argument("--shared-poses", metavar="NAME", default=None,
                    help="train all sets on NAME's COLMAP poses (needs aligned, same-named images)")
    ap.add_argument("--anchor", default=None, help="reference set for the comparison figures "
                                                   "(default: --shared-poses set, else first --set)")
    ap.add_argument("--lichtfeld", default=None, help="path to LichtFeld-Studio (else $LICHTFELD_EXE/PATH)")
    ap.add_argument("--colmap", default=None, help="path to colmap (else $COLMAP_EXE/PATH)")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--matcher", default="sequential",
                    help="COLMAP matcher: sequential (video frames) | exhaustive | ...")
    ap.add_argument("--colmap-max-image-size", type=int, default=2000)
    ap.add_argument("--iter", type=int, default=None, help="training iterations (default: LichtFeld default 30000)")
    ap.add_argument("--steps-scaler", type=float, default=None, help="scale all training steps (e.g. 0.5)")
    ap.add_argument("--resize-factor", type=int, default=2, help="LichtFeld resize factor (1/2/4/8)")
    ap.add_argument("--test-every", type=int, default=12, help="hold out every Nth image for eval")
    ap.add_argument("--no-undistort", action="store_true", help="do not pass --undistort to LichtFeld")
    ap.add_argument("--lichtfeld-arg", action="append", default=[], help="extra raw LichtFeld arg (repeatable)")
    ap.add_argument("--force-colmap", action="store_true", help="re-run COLMAP even if sparse exists")
    ap.add_argument("--force-train", action="store_true", help="re-train even if eval output exists")
    args = ap.parse_args(argv)

    sets = [(n, Path(d)) for n, d in args.set]
    names = [n for n, _ in sets]
    if len(set(names)) != len(names):
        raise SystemExit("--set names must be unique")
    for n, d in sets:
        if not d.is_dir() or count_images(d) == 0:
            raise SystemExit(f"set '{n}': no images in {d}")
    if args.shared_poses and args.shared_poses not in names:
        raise SystemExit(f"--shared-poses '{args.shared_poses}' is not one of {names}")
    anchor = args.anchor or args.shared_poses or names[0]
    if anchor not in names:
        raise SystemExit(f"--anchor '{anchor}' is not one of {names}")

    colmap = find_colmap(args.colmap)
    lichtfeld = find_lichtfeld(args.lichtfeld)
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"colmap    : {colmap}")
    print(f"lichtfeld : {lichtfeld}")
    print(f"sets      : {', '.join(names)}  | anchor: {anchor}  | shared-poses: {args.shared_poses}")

    # 1) COLMAP sparse for each set (or once, for the shared-poses set).
    colmap_root = args.work / "colmap"
    sparse_for = {}
    to_reconstruct = [args.shared_poses] if args.shared_poses else names
    for n in to_reconstruct:
        sdir = colmap_root / n / "sparse" / "0"
        if sdir.exists() and not args.force_colmap:
            print(f"[colmap] reuse existing sparse for '{n}'")
        else:
            print(f"[colmap] reconstructing '{n}' ...")
            imgs = dict(sets)[n]
            r = reconstruct(n, imgs, colmap_root, colmap, args.camera_model,
                            matcher=args.matcher, max_image_size=args.colmap_max_image_size)
            if "error" in r or not sdir.exists():
                raise SystemExit(f"COLMAP failed for '{n}': {r.get('error', 'no sparse/0 produced')}")
            print(f"[colmap] '{n}': registered {r.get('registered')}/{r.get('n_input')} | "
                  f"points {r.get('points')}")
    for n in names:
        sparse_for[n] = (colmap_root / (args.shared_poses or n) / "sparse" / "0")

    # 2) Assemble datasets + 3) train each set.
    results = {}
    for n, imgs in sets:
        ds = assemble_dataset(args.work, n, imgs, sparse_for[n])
        out = args.work / "train" / n
        if latest_eval_dir(out) is not None and not args.force_train:
            print(f"[train] reuse existing 3DGS output for '{n}'")
        else:
            print(f"[train] training 3DGS for '{n}' ...")
            code = train(lichtfeld, ds, out, iters=args.iter, steps_scaler=args.steps_scaler,
                         resize_factor=args.resize_factor, test_every=args.test_every,
                         undistort=not args.no_undistort, extra=args.lichtfeld_arg,
                         log_path=args.work / "logs" / f"train_{n}.log")
            if code != 0:
                print(f"[train] WARNING: LichtFeld exited {code} for '{n}' "
                      f"(see logs/train_{n}.log)")
        results[n] = parse_metrics(out)

    # 4) Comparison figures (matched by GT content).
    eval_by_name = {}
    for n in names:
        ev = latest_eval_dir(args.work / "train" / n)
        if ev is not None:
            eval_by_name[n] = ev
    n_fig = 0
    if anchor in eval_by_name and len(eval_by_name) >= 2:
        n_fig = build_comparison(eval_by_name, anchor, args.work / "compare")
    else:
        print("[compare] not enough eval outputs to build comparison figures")

    # 5) Report.
    lines = ["# 3DGS A/B report", "",
             f"- COLMAP: `{colmap}`", f"- LichtFeld: `{lichtfeld}`",
             f"- shared-poses: `{args.shared_poses}`  | anchor: `{anchor}`",
             f"- resize_factor: {args.resize_factor}  | test_every: {args.test_every}"
             f"  | iter: {args.iter or 'default'}  | steps_scaler: {args.steps_scaler}", "",
             "| Set | PSNR | SSIM | #Gaussians | (vs anchor) |",
             "|---|---|---|---|---|"]
    base_g = results.get(anchor, {}).get("gaussians")
    for n in names:
        r = results[n]
        dg = ""
        if base_g and r.get("gaussians"):
            dg = f"{(r['gaussians'] / base_g - 1) * 100:+.0f}% gaussians"
        lines.append(f"| {n} | {r.get('psnr')} | {r.get('ssim')} | {r.get('gaussians')} | {dg} |")
    lines += ["",
              f"Same-viewpoint comparison figures: `{(args.work / 'compare').as_posix()}` "
              f"({n_fig} frames). Each row: `GT(anchor) | 3DGS render per set`.",
              "",
              "> Note: PSNR/SSIM are measured against each set's OWN ground-truth images, so",
              "> they indicate *reconstruct-ability*, not absolute quality, when GT differs",
              "> between sets (e.g. glare vs cleaned). Read them with the visual figures."]
    (args.work / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nReport: {args.work / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
