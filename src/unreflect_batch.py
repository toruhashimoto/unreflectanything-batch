"""Batch reflection-removal engine.

Wraps the `unreflectanything` Python API (loads the model ONCE and reuses it for
every image — the 3.44 GB checkpoint is far too expensive to reload per file).

Heavy imports (torch / unreflectanything) are done lazily inside functions so the
utility modules and unit tests stay importable without the ML stack.

Hard guarantees enforced here (for COLMAP / 3DGS safety):
  * originals are never written to;
  * outputs go to a separate output tree, original filenames preserved;
  * same-name outputs are skipped unless --overwrite;
  * output dimensions == processed-input dimensions (resize_output=True);
  * one failed image is logged and the batch continues.
"""
from __future__ import annotations

import contextlib
import os
import time
import tempfile
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from . import image_io
from . import metrics as metrics_mod
from . import preview as preview_mod
from .logger import RunLogger, PROCESSED_BY

MODEL_NAME = "UnReflectAnything"


class WeightsMissingError(RuntimeError):
    """Raised when the pretrained weights have not been downloaded yet."""


class ModelLoadError(RuntimeError):
    """Raised when the checkpoint cannot be loaded (e.g. transformers mismatch)."""


WEIGHTS_DOWNLOAD_CMD = "unreflectanything download --weights"


def weights_status() -> tuple[bool, Optional[Path], str]:
    """Return (available, weights_dir, human_detail) for the pretrained weights."""
    try:
        import unreflectanything as unreflect
    except Exception as e:  # noqa: BLE001
        return False, None, f"unreflectanything is not importable: {e}"
    try:
        wdir = Path(unreflect.cache("weights"))
    except Exception as e:  # noqa: BLE001
        return False, None, f"could not resolve the weights cache dir: {e}"
    pts = sorted(wdir.glob("*.pt")) if wdir.exists() else []
    if pts:
        total_mb = sum(p.stat().st_size for p in pts) / 1e6
        return True, wdir, f"{len(pts)} weight file(s), {total_mb:.0f} MB, in {wdir}"
    return False, wdir, f"no .pt weights found in {wdir}"


def weights_missing_message(wdir: Optional[Path] = None) -> str:
    """A friendly, actionable message for when weights aren't downloaded yet."""
    if wdir is None:
        wdir = weights_status()[1]
    loc = f"\n  Cache location : {wdir}" if wdir else ""
    return (
        "Pretrained model weights are not downloaded yet (~5.9 GB, one time).\n"
        f"  Download them : {WEIGHTS_DOWNLOAD_CMD}\n"
        "  Or pass --download-weights to fetch them now (CLI), or use the GUI's "
        '"Download model weights" button.'
        f"{loc}\n"
        "  (Weights are required to run — there is NO automatic download.)"
    )


def download_weights(progress: bool = True) -> Path:
    """Download the pretrained weights via the package API. Returns the weights dir."""
    import unreflectanything as unreflect

    if progress:
        print(f"Downloading UnReflectAnything weights (~5.9 GB) — one time...", flush=True)
    unreflect.download("weights")
    return Path(unreflect.cache("weights"))


@dataclass
class BatchConfig:
    input_dir: Path
    output_dir: Path
    recursive: bool = False
    exts: tuple[str, ...] = image_io.SUPPORTED_EXTS
    device: str = "auto"  # auto | cuda | cpu
    overwrite: bool = False
    make_preview: bool = False
    heatmap: bool = False
    emit_mask: bool = False
    limit: Optional[int] = None
    max_size: Optional[int] = None  # quick mode: downscale longest side (changes dims!)
    jpeg_quality: int = 95
    threshold: float = 0.3
    dilation: int = 40
    batch_size: int = 4
    composite: bool = False  # model's internal composite (blended at ~448px)
    mask_composite: bool = False  # wrapper full-res highlight-gated composite
    # mask-composite gate (kept TIGHT: only truly-blown pixels are replaced, so the
    # rest of the image stays at full original sharpness). The model's --dilation (40)
    # is for its own inpaint mask and must NOT be reused here, or it balloons the
    # replaced region and blurs the subject.
    mask_composite_level: float = 248.0
    mask_composite_dilation: int = 0
    mask_composite_feather: float = 1.0
    use_exiftool: bool = False  # full metadata copy via exiftool (if available)
    verbose: bool = False  # let the engine's own stdout through
    highlight_level: float = metrics_mod.DEFAULT_HIGHLIGHT_LEVEL
    dry_run: bool = False

    def __post_init__(self):
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.exts = image_io.normalize_exts(self.exts)

    def params_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "dilation": self.dilation,
            "batch_size": self.batch_size,
            "jpeg_quality": self.jpeg_quality,
            "composite": self.composite,
            "mask_composite": self.mask_composite,
            "mask_composite_level": self.mask_composite_level,
            "mask_composite_dilation": self.mask_composite_dilation,
            "mask_composite_feather": self.mask_composite_feather,
            "max_size": self.max_size,
            "use_exiftool": self.use_exiftool,
        }


@contextlib.contextmanager
def _suppress_stdout(enabled: bool = True):
    """Silence the engine's per-image ``Using device: ...`` prints during inference."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


# --------------------------------------------------------------------------- #
# Device                                                                        #
# --------------------------------------------------------------------------- #
def resolve_device(requested: str = "auto") -> tuple[str, str]:
    """Return (device, human_note). Falls back to CPU cleanly when CUDA can't run.

    A mere ``is_available()`` is not enough on Blackwell: a CPU-only or wrong-arch
    torch build can pass ``is_available()`` and only fail at the first kernel
    launch. We therefore run a tiny matmul to prove kernels execute.
    """
    import torch

    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu", "forced CPU"

    if not torch.cuda.is_available():
        if requested == "cuda":
            return "cpu", "CUDA requested but torch.cuda.is_available() is False -> CPU"
        return "cpu", "no CUDA available -> CPU"

    try:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        archs = torch.cuda.get_arch_list()
        needed = f"sm_{cap[0]}{cap[1]}"
        if needed not in archs:
            return "cpu", (
                f"{name} needs {needed} but this torch build lacks it "
                f"({archs}); reinstall torch from the cu128 index -> using CPU"
            )
        a = torch.randn(256, 256, device="cuda")
        _ = a @ a
        torch.cuda.synchronize()
        return "cuda", f"{name} ({needed})"
    except Exception as e:  # noqa: BLE001
        return "cpu", f"CUDA probe failed ({e}) -> CPU"


# --------------------------------------------------------------------------- #
# Model                                                                         #
# --------------------------------------------------------------------------- #
def load_model(device: str):
    """Load the pretrained UnReflectAnything model once, with friendly errors."""
    import unreflectanything as unreflect

    try:
        return unreflect.model(pretrained=True, device=device)
    except FileNotFoundError as e:
        raise WeightsMissingError(weights_missing_message()) from e
    except RuntimeError as e:
        msg = str(e)
        if "state_dict" in msg or "Missing key" in msg or "Unexpected key" in msg:
            raise ModelLoadError(
                "Failed to load the checkpoint — this is almost always a "
                "`transformers` version mismatch. Install the pinned commit:\n"
                "    pip install \"transformers @ https://github.com/huggingface/"
                "transformers/archive/2fe43376cdde02b7ffcf117e6eb9aa4375fb2dd1.zip\"\n"
                f"(original error: {msg.splitlines()[0]})"
            ) from e
        raise


# --------------------------------------------------------------------------- #
# Per-image pipeline                                                            #
# --------------------------------------------------------------------------- #
def _run_inference_to_file(model, src: Path, dst_png: Path, cfg: BatchConfig) -> None:
    import unreflectanything as unreflect

    with _suppress_stdout(not cfg.verbose):
        unreflect.inference(
            str(src),
            output=str(dst_png),
            model=model,
            threshold=cfg.threshold,
            dilation=cfg.dilation,
            resize_output=True,  # output dims == input dims (COLMAP intrinsics safety)
            composite=cfg.composite,
        )


def process_one(
    src: Path,
    cfg: BatchConfig,
    model,
    device: str,
    tmp_dir: Path,
    subdirs: dict[str, Path],
) -> dict:
    """Process a single image. Returns a log record; never raises for image-level
    problems (those become an ``error`` record)."""
    t0 = time.perf_counter()
    dst = image_io.relative_output_path(src, cfg.input_dir, cfg.output_dir)
    rel = src.relative_to(cfg.input_dir)
    record: dict = {
        "status": "ok",
        "processed_by": PROCESSED_BY,
        "source": str(src),
        "output": str(dst),
        "model": MODEL_NAME,
        "model_version": _pkg_version(),
        "device": device,
        "params": cfg.params_dict(),
    }

    try:
        meta = image_io.read_metadata(src)
        record["input_size"] = list(meta["size"])  # (w, h)

        # No-overwrite default.
        if dst.exists() and not cfg.overwrite:
            record["status"] = "skipped"
            record["error"] = "output exists (use --overwrite to replace)"
            record["duration_sec"] = round(time.perf_counter() - t0, 4)
            return record

        if cfg.dry_run:
            record["status"] = "skipped"
            record["error"] = "dry-run"
            record["duration_sec"] = round(time.perf_counter() - t0, 4)
            return record

        # Determine the actual input fed to the model (optionally downscaled).
        proc_input = src
        if cfg.max_size:
            small = image_io.load_rgb(src)
            small.thumbnail((cfg.max_size, cfg.max_size), preview_mod._resampling())
            proc_input = tmp_dir / (rel.as_posix().replace("/", "__") + ".in.png")
            proc_input.parent.mkdir(parents=True, exist_ok=True)
            small.save(proc_input, format="PNG")

        before_pil = image_io.load_rgb(proc_input)

        # Run the model -> lossless temp PNG (avoids a double JPEG encode).
        tmp_out = tmp_dir / (rel.as_posix().replace("/", "__") + ".out.png")
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        _run_inference_to_file(model, proc_input, tmp_out, cfg)
        after_pil = image_io.load_rgb(tmp_out)

        # Full-res highlight-gated composite: keep original detail everywhere
        # except blown highlights (preserves SfM features on high-res inputs).
        if cfg.mask_composite:
            comp = metrics_mod.luminance_composite(
                np.asarray(before_pil), np.asarray(after_pil),
                level=cfg.mask_composite_level,
                dilation=cfg.mask_composite_dilation,
                feather=cfg.mask_composite_feather,
            )
            after_pil = Image.fromarray(comp, "RGB")

        before_arr = np.asarray(before_pil)
        after_arr = np.asarray(after_pil)
        record["output_size"] = [after_pil.width, after_pil.height]
        record["metrics"] = metrics_mod.compute_pair_metrics(
            before_arr, after_arr, cfg.highlight_level
        )

        # Save the cleaned image (format/EXIF/ICC preserved from the original src).
        image_io.save_processed(
            src, after_pil, dst, jpeg_quality=cfg.jpeg_quality, use_exiftool=cfg.use_exiftool
        )

        # Optional artifacts.
        if cfg.heatmap or cfg.make_preview:
            hm_arr = metrics_mod.diff_heatmap(before_arr, after_arr)
            hm_pil = Image.fromarray(hm_arr, "RGB")
            if cfg.heatmap:
                hm_path = subdirs["heatmap"] / rel.with_suffix(".png")
                hm_path.parent.mkdir(parents=True, exist_ok=True)
                hm_pil.save(hm_path, format="PNG")
                record["heatmap"] = str(hm_path)
            if cfg.make_preview:
                cmp_img = preview_mod.make_compare(
                    before_pil, after_pil, hm_pil if cfg.heatmap else None
                )
                cmp_path = subdirs["preview"] / rel.with_suffix(".jpg")
                cmp_path.parent.mkdir(parents=True, exist_ok=True)
                cmp_img.save(cmp_path, format="JPEG", quality=90)
                record["preview"] = str(cmp_path)

        if cfg.emit_mask:
            mask_arr = metrics_mod.change_mask(before_arr, after_arr)
            mask_path = subdirs["masks"] / rel.with_suffix(".png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_arr, "L").save(mask_path, format="PNG")
            record["mask"] = str(mask_path)

        record["duration_sec"] = round(time.perf_counter() - t0, 4)
        return record

    except Exception as e:  # noqa: BLE001 - image-level failures must not stop the batch
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc(limit=4)
        record["duration_sec"] = round(time.perf_counter() - t0, 4)
        return record


def _pkg_version() -> str:
    try:
        from importlib.metadata import version

        return version("unreflectanything")
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# Batch driver                                                                  #
# --------------------------------------------------------------------------- #
def run_batch(
    cfg: BatchConfig,
    progress: bool = True,
    on_progress: Optional[Callable[[int, int, dict], None]] = None,
    model=None,
) -> dict:
    """Process every matching image under ``cfg.input_dir``.

    Pass ``model`` to reuse an already-loaded UnReflectModel (the GUI caches one
    across reruns so the 3.44 GB checkpoint is loaded once per session).

    Returns a run summary dict. Raises only for fatal, whole-run problems
    (bad paths, missing weights, model load failure). Per-image errors are logged
    and the batch keeps going.
    """
    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"input folder does not exist: {cfg.input_dir}")
    # Never let output nest inside input (would risk re-processing outputs / overwrite).
    out_res, in_res = cfg.output_dir.resolve(), cfg.input_dir.resolve()
    if out_res == in_res or in_res in out_res.parents:
        raise ValueError(
            "output folder must NOT be the input folder or inside it "
            f"(input={in_res}, output={out_res})"
        )

    images = image_io.iter_images(cfg.input_dir, cfg.recursive, cfg.exts)
    if cfg.limit is not None:
        images = images[: max(0, cfg.limit)]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    subdirs = {
        "preview": cfg.output_dir / "preview_compare",
        "heatmap": cfg.output_dir / "heatmap",
        "masks": cfg.output_dir / "masks",
        "logs": cfg.output_dir / "logs",
    }
    logger = RunLogger(subdirs["logs"])

    device, device_note = resolve_device(cfg.device)
    summary: dict = {
        "started_at": _now(),
        "model": MODEL_NAME,
        "model_version": _pkg_version(),
        "device": device,
        "device_note": device_note,
        "exiftool": image_io.find_exiftool() if cfg.use_exiftool else None,
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "recursive": cfg.recursive,
        "num_candidates": len(images),
        "config": _safe_config(cfg),
        "note": (
            "Evaluation-only output for 3DGS/photogrammetry quality experiments — "
            "NOT measurement ground truth. Single-image reflection removal has no "
            "multi-view consistency guarantee; A/B test SfM with vs. without."
        ),
    }

    if not images:
        summary.update(logger_counts(logger))
        logger.finalize(summary)
        return summary

    if model is None and not cfg.dry_run:
        ok, wdir, _ = weights_status()  # friendly, fast preflight before the slow load
        if not ok:
            raise WeightsMissingError(weights_missing_message(wdir))
        model = load_model(device)  # may raise WeightsMissingError / ModelLoadError

    tmp_dir = Path(tempfile.mkdtemp(prefix="unreflect_", dir=str(cfg.output_dir / "logs")))
    iterator = enumerate(images, start=1)
    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(images), unit="img", desc=f"UnReflect[{device}]")
        except Exception:  # noqa: BLE001
            bar = None

    try:
        for i, src in iterator:
            rec = process_one(src, cfg, model, device, tmp_dir, subdirs)
            logger.log(rec)
            if bar is not None:
                bar.update(1)
                bar.set_postfix_str(rec["status"])
            if on_progress is not None:
                on_progress(i, len(images), rec)
    finally:
        if bar is not None:
            bar.close()
        _rmtree_quiet(tmp_dir)

    summary.update(logger_counts(logger))
    logger.finalize(summary)
    return summary


def logger_counts(logger: RunLogger) -> dict:
    c = logger.counts
    return {"processed_ok": c["ok"], "skipped": c["skipped"], "errors": c["error"]}


def _safe_config(cfg: BatchConfig) -> dict:
    d = asdict(cfg)
    d["input_dir"] = str(cfg.input_dir)
    d["output_dir"] = str(cfg.output_dir)
    d["exts"] = list(cfg.exts)
    return d


def _now() -> str:
    from .logger import utc_now_iso

    return utc_now_iso()


def _rmtree_quiet(path: Path) -> None:
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass
