"""A/B reconstruction comparison with COLMAP.

Runs a COLMAP **sparse** reconstruction (feature extraction -> exhaustive matching
-> incremental mapping) on two or more image folders and reports the registration
metrics side by side, so you can judge whether reflection-removal *helps or hurts*
Structure-from-Motion before committing to it for 3DGS.

This deliberately uses CPU SIFT + CPU matching so it runs anywhere (no GPU/display
needed). It reads camera intrinsics from EXIF, so it is a fair comparison only if
the cleaned images preserved EXIF + dimensions (this tool's whole point).

Usage:
    python tools/ab_colmap.py --work <workdir> \
        --set original "D:\\ab_demo\\original" \
        --set cleaned  "D:\\ab_demo\\cleaned" \
        [--colmap "D:\\...\\colmap.exe"] [--camera-model SIMPLE_RADIAL]

Writes <workdir>/ab_report.md and prints a summary table.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_colmap(explicit: str | None) -> str:
    """Resolve the colmap executable: --colmap / $COLMAP_EXE / PATH / local .tools."""
    import os

    for cand in (explicit, os.environ.get("COLMAP_EXE")):
        if cand:
            p = Path(cand)
            if p.is_dir():  # accept a directory containing colmap(.exe)
                for sub in p.rglob("colmap*"):
                    if sub.is_file() and sub.stem.lower() == "colmap":
                        return str(sub)
            return cand
    p = shutil.which("colmap")
    if p:
        return p
    here = Path(__file__).resolve().parents[1]
    for pat in ("colmap.exe", "colmap"):
        for c in (here / ".tools" / "colmap").rglob(pat):
            return str(c)
    raise SystemExit("colmap not found — pass --colmap <path> or set $COLMAP_EXE "
                     "(download: https://github.com/colmap/colmap/releases)")


def run(cmd: list[str], log: Path, timeout: int = 3600) -> tuple[int, str]:
    """Run a command, tee combined output to a log file, return (code, output)."""
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", timeout=timeout)
        fh.write(proc.stdout or "")
    return proc.returncode, proc.stdout or ""


_METRICS = {
    "images": r"Images:\s*(\d+)",
    "registered": r"Registered images:\s*(\d+)",
    "points": r"Points:\s*(\d+)",
    "observations": r"Observations:\s*(\d+)",
    "track_length": r"Mean track length:\s*([\d.]+)",
    "obs_per_image": r"Mean observations per image:\s*([\d.]+)",
    "reproj_error": r"Mean reprojection error:\s*([\d.]+)",
}


def parse_analyzer(text: str) -> dict:
    out: dict = {}
    for key, pat in _METRICS.items():
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    return out


_MATCHERS = {
    "exhaustive": "exhaustive_matcher",
    "sequential": "sequential_matcher",   # best for ordered video frames
    "spatial": "spatial_matcher",
    "vocab_tree": "vocab_tree_matcher",
}


def reconstruct(name: str, images: Path, work: Path, colmap: str, camera_model: str,
                matcher: str = "exhaustive", max_image_size: int = 0,
                mask_path: Path | None = None) -> dict:
    sdir = work / name
    sdir.mkdir(parents=True, exist_ok=True)
    log = sdir / "colmap.log"
    log.write_text("", encoding="utf-8")
    db = sdir / "database.db"
    sparse = sdir / "sparse"
    sparse.mkdir(exist_ok=True)
    if db.exists():
        db.unlink()

    n_imgs = len([p for p in images.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")])
    t0 = time.perf_counter()

    # COLMAP 4.x renamed the GPU toggles to Feature{Extraction,Matching}.use_gpu
    # (older 3.x used Sift{Extraction,Matching}.use_gpu). We target 4.x here.
    fe_cmd = [colmap, "feature_extractor", "--database_path", str(db),
              "--image_path", str(images),
              "--ImageReader.single_camera", "1",
              "--ImageReader.camera_model", camera_model,
              "--FeatureExtraction.use_gpu", "0"]
    if max_image_size and max_image_size > 0:
        fe_cmd += ["--FeatureExtraction.max_image_size", str(max_image_size)]
    if mask_path is not None:
        # COLMAP ignores black (0) pixels in <mask_path>/<image_name>.png — i.e. no
        # features are detected there. Used to exclude reflection regions WITHOUT
        # altering the image (keeps surrounding features at their original descriptors).
        fe_cmd += ["--ImageReader.mask_path", str(mask_path)]
    code, _ = run(fe_cmd, log)
    if code != 0:
        return {"name": name, "n_input": n_imgs, "error": "feature_extractor failed", "seconds": round(time.perf_counter() - t0, 1)}

    matcher_cmd = _MATCHERS.get(matcher, "exhaustive_matcher")
    code, _ = run([colmap, matcher_cmd, "--database_path", str(db),
                   "--FeatureMatching.use_gpu", "0"], log)
    if code != 0:
        return {"name": name, "n_input": n_imgs, "error": f"{matcher_cmd} failed", "seconds": round(time.perf_counter() - t0, 1)}

    code, _ = run([colmap, "mapper", "--database_path", str(db),
                   "--image_path", str(images), "--output_path", str(sparse)], log)
    seconds = round(time.perf_counter() - t0, 1)
    if code != 0:
        return {"name": name, "n_input": n_imgs, "error": "mapper failed", "seconds": seconds}

    # Pick the sub-model with the most registered images.
    best: dict = {"registered": -1}
    n_models = 0
    for sub in sorted(sparse.glob("*")):
        if not sub.is_dir():
            continue
        n_models += 1
        _, out = run([colmap, "model_analyzer", "--path", str(sub)], log, timeout=300)
        stats = parse_analyzer(out)
        if (stats.get("registered") or 0) > (best.get("registered") or -1):
            best = stats
            best["model_dir"] = str(sub)
    best.update({"name": name, "n_input": n_imgs, "n_models": n_models, "seconds": seconds})
    return best


def fmt(v, suffix=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}{suffix}"
    return f"{v}{suffix}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A/B COLMAP sparse reconstruction comparison")
    ap.add_argument("--work", required=True, type=Path, help="working dir for COLMAP outputs")
    ap.add_argument("--set", nargs=2, action="append", metavar=("NAME", "DIR"),
                    default=[], help="a named image set, e.g. --set original D:\\imgs")
    ap.add_argument("--set-masked", nargs=3, action="append", default=[],
                    metavar=("NAME", "IMAGES", "MASKS"),
                    help="a named image set with a COLMAP mask dir (excludes masked "
                         "regions from feature detection without altering the images)")
    ap.add_argument("--colmap", default=None, help="path to colmap.exe (else PATH or .tools)")
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--matcher", default="exhaustive", choices=list(_MATCHERS),
                    help="feature matcher (use 'sequential' for ordered video frames)")
    ap.add_argument("--max-image-size", type=int, default=0,
                    help="cap SIFT input resolution for speed (e.g. 2000); 0 = COLMAP default")
    args = ap.parse_args(argv)

    colmap = find_colmap(args.colmap)
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"colmap: {colmap}")

    jobs = [(n, Path(d), None) for n, d in args.set]
    jobs += [(n, Path(img), Path(msk)) for n, img, msk in args.set_masked]
    if not jobs:
        raise SystemExit("provide at least one --set or --set-masked")

    results = []
    for name, d, mask in jobs:
        tag = f"{d}" + (f"  (mask: {mask})" if mask else "")
        print(f"\n=== reconstructing '{name}' from {tag} ===")
        try:
            r = reconstruct(name, d, args.work, colmap, args.camera_model,
                            matcher=args.matcher, max_image_size=args.max_image_size,
                            mask_path=mask)
        except Exception as e:  # noqa: BLE001 - one set failing must not kill the rest
            r = {"name": name, "n_input": None, "error": f"{type(e).__name__}: {e}", "seconds": None}
        if "error" in r:
            print(f"  [FAILED] {r['error']} ({r['seconds']}s)")
        else:
            print(f"  registered {fmt(r.get('registered'))}/{r['n_input']} | "
                  f"points {fmt(r.get('points'))} | reproj {fmt(r.get('reproj_error'),'px')} | {r['seconds']}s")
        results.append(r)

    # Markdown report.
    cols = [
        ("Set", "name"),
        ("Input imgs", "n_input"),
        ("Registered", "registered"),
        ("3D points", "points"),
        ("Observations", "observations"),
        ("Mean track len", "track_length"),
        ("Mean reproj err (px)", "reproj_error"),
        ("Sub-models", "n_models"),
        ("Seconds", "seconds"),
    ]
    lines = ["# COLMAP A/B reconstruction report", "",
             "Sparse SfM (CPU SIFT + exhaustive matching). Higher *Registered* and",
             "*3D points*, longer *track length*, and lower *reprojection error* are better.",
             "", "| " + " | ".join(h for h, _ in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in results:
        row = [fmt(r.get(key)) if key != "name" else str(r.get(key)) for _, key in cols]
        if "error" in r:
            row = [str(r.get("name")), fmt(r.get("n_input"))] + [f"ERROR: {r['error']}"] + ["—"] * (len(cols) - 3)
        lines.append("| " + " | ".join(row) + " |")
    lines += ["",
              "> Note: single-image reflection removal has no multi-view consistency guarantee.",
              "> Use this A/B on YOUR photo set to decide whether to clean before SfM/3DGS."]
    report = args.work / "ab_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {report}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
