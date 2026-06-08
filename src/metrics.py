"""Evaluation metrics for reflection-removal quality.

These are *evaluation-only* signals to help judge whether a cleaned set is likely
to help (or hurt) a downstream 3DGS / SfM reconstruction — NOT ground-truth
measurements. All functions operate on uint8 RGB numpy arrays of shape (H, W, 3).
"""
from __future__ import annotations

import numpy as np

# Rec.601 luma weights.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# A pixel whose luma is at/above this (0-255) is treated as a probable specular
# highlight / blown-out region. 235 ~= 0.92 of full scale.
DEFAULT_HIGHLIGHT_LEVEL = 235.0


def to_luma(rgb: np.ndarray) -> np.ndarray:
    """Return an (H, W) float32 luma image in [0, 255]."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return arr[..., :3] @ _LUMA


def mean_luminance(rgb: np.ndarray) -> float:
    return float(to_luma(rgb).mean())


def highlight_ratio(rgb: np.ndarray, level: float = DEFAULT_HIGHLIGHT_LEVEL) -> float:
    """Fraction of pixels at/above ``level`` luma — a proxy for highlight area."""
    luma = to_luma(rgb)
    if luma.size == 0:
        return 0.0
    return float((luma >= level).mean())


def compute_pair_metrics(
    before: np.ndarray,
    after: np.ndarray,
    highlight_level: float = DEFAULT_HIGHLIGHT_LEVEL,
) -> dict:
    """Before/after evaluation metrics.

    ``after`` is resized-back to original dims by the engine, so shapes match; if
    not, the after image is centre-aligned by truncation to be safe.
    """
    before = np.asarray(before)
    after = np.asarray(after)
    if before.shape != after.shape:
        h = min(before.shape[0], after.shape[0])
        w = min(before.shape[1], after.shape[1])
        before = before[:h, :w]
        after = after[:h, :w]

    lb = mean_luminance(before)
    la = mean_luminance(after)
    hb = highlight_ratio(before, highlight_level)
    ha = highlight_ratio(after, highlight_level)
    mad = float(np.abs(before.astype(np.float32) - after.astype(np.float32)).mean())
    return {
        "mean_luma_before": round(lb, 4),
        "mean_luma_after": round(la, 4),
        "mean_luma_delta": round(la - lb, 4),
        "highlight_ratio_before": round(hb, 6),
        "highlight_ratio_after": round(ha, 6),
        "highlight_ratio_delta": round(ha - hb, 6),
        "mean_abs_diff": round(mad, 4),
    }


def diff_heatmap(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB heatmap of the per-pixel luma difference.

    Uses OpenCV's TURBO colormap when available, else a numpy fallback. Bright =
    large change (region the network altered).
    """
    before = np.asarray(before)
    after = np.asarray(after)
    if before.shape != after.shape:
        h = min(before.shape[0], after.shape[0])
        w = min(before.shape[1], after.shape[1])
        before = before[:h, :w]
        after = after[:h, :w]

    diff = np.abs(to_luma(before) - to_luma(after))
    dmax = float(diff.max())
    norm = (diff / dmax) if dmax > 1e-6 else np.zeros_like(diff)
    gray = (norm * 255.0).astype(np.uint8)

    try:
        import cv2

        bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        return bgr[..., ::-1].copy()  # BGR -> RGB
    except Exception:  # noqa: BLE001 - numpy fallback colormap
        t = norm
        r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def luminance_composite(
    original: np.ndarray,
    diffuse: np.ndarray,
    level: float = 248.0,
    dilation: int = 0,
    feather: float = 1.0,
) -> np.ndarray:
    """Full-resolution, highlight-gated composite.

    Keeps the *original* pixels everywhere except where the original is a bright
    highlight (luma >= ``level``), and only there blends in the model's diffuse
    result. This preserves full-resolution detail (important for SfM/3DGS) while
    still suppressing blown highlights — a workaround for the model's ~448 px
    internal resolution softening high-res inputs. Returns (H, W, 3) uint8.

    Keep this gate TIGHT. ``dilation`` grows the replaced region by that many pixels
    in every direction, so a large value (e.g. the model's inpaint dilation of 40)
    balloons even small bright specks into big soft blobs and blurs the subject.
    Defaults replace only near-blown pixels with a 1 px feather. Lower ``level`` and
    raise ``dilation`` only if you specifically need to clean larger glare areas.
    """
    orig = np.asarray(original, dtype=np.float32)
    diff = np.asarray(diffuse, dtype=np.float32)
    if orig.shape != diff.shape:
        h = min(orig.shape[0], diff.shape[0])
        w = min(orig.shape[1], diff.shape[1])
        orig, diff = orig[:h, :w], diff[:h, :w]

    m = (to_luma(orig) >= level).astype(np.float32)
    try:
        import cv2

        if dilation and dilation > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
            m = cv2.dilate(m, k)
        if feather and feather > 0:
            m = cv2.GaussianBlur(m, (0, 0), float(feather))
    except Exception:  # noqa: BLE001 - mask refinement is best-effort
        pass

    m3 = np.clip(m, 0.0, 1.0)[..., None]
    out = orig * (1.0 - m3) + diff * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def change_mask(
    before: np.ndarray,
    after: np.ndarray,
    level: float = 12.0,
) -> np.ndarray:
    """Binary (H, W) uint8 mask (0/255) of regions the network changed.

    This approximates the "removed-reflection" region from the before/after diff.
    255 = changed (where the network altered the image), 0 = unchanged. Note the
    polarity is the *opposite* of a COLMAP/RealityScan exclusion mask — for that,
    use :func:`reflection_exclusion_mask`, which marks reflections black (excluded).
    """
    before = np.asarray(before)
    after = np.asarray(after)
    diff = np.abs(to_luma(before) - to_luma(after))
    return np.where(diff >= level, np.uint8(255), np.uint8(0))


def _morph_clean(refl: np.ndarray, open_radius: int, dilation: int) -> np.ndarray:
    """Despeckle (morphological open) then grow (dilate) a 0/1 uint8 mask.

    Shared by the model-based and pure-luma exclusion masks. Best-effort: if OpenCV
    is unavailable the (still valid) un-morphed binary mask is returned.
    """
    try:
        import cv2

        if open_radius and open_radius > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_radius + 1,) * 2)
            refl = cv2.morphologyEx(refl, cv2.MORPH_OPEN, k)
        if dilation and dilation > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)
            refl = cv2.dilate(refl, k)
    except Exception:  # noqa: BLE001 - morphology is best-effort; binary result still valid
        pass
    return refl


def reflection_exclusion_mask(
    before: np.ndarray,
    after: np.ndarray,
    drop_level: float = 12.0,
    highlight_gate: float = 250.0,
    dilation: int = 2,
    open_radius: int = 1,
    return_stats: bool = False,
):
    """Binary (H, W) uint8 *exclusion* mask from a model before/after pair, in the
    polarity RealityScan / COLMAP expect: ``0`` (black) = **exclude** (a reflection
    the model removed), ``255`` (white) = **keep**.

    A pixel is treated as a removed specular reflection when the model *darkened* it
    by at least ``drop_level`` luma **and** (if ``highlight_gate`` > 0) the original
    pixel was at least that bright. The default gate is deliberately TIGHT (250 ≈
    near-blown) because luma alone cannot distinguish a blown specular from a bright
    *diffuse* surface (overcast sky, white paint, a light-coloured car body); a low gate
    masks all of them and fragments the SfM reconstruction. Lower the gate (e.g. 240) to
    catch more reflection on a genuinely glary set, raise it (e.g. 252) to preserve more
    features. If the excluded fraction is large, the set probably does not need masking
    at all (originals reconstruct best for mild glare).

    The output is **strictly binary with hard edges**: RealityScan accepts up to 256
    gray shades but officially discourages them ("may interfere with processing and
    produce inconsistent results"), so no feather is applied. ``open_radius`` removes
    tiny specks (morphological open) and ``dilation`` grows the excluded region to
    cover reflection halos — keep both small (large values fragment SfM).
    """
    cand = reflection_candidate_mask(before, after, drop_level, highlight_gate) > 0
    candidate_ratio = float(cand.mean()) * 100.0  # candidate area BEFORE morphology

    refl = _morph_clean(cand.astype(np.uint8), open_radius, dilation)
    mask = np.where(refl > 0, np.uint8(0), np.uint8(255))  # reflection -> black (excluded)
    if return_stats:
        return mask, {
            "candidate_pixel_ratio": round(candidate_ratio, 3),
            "final_mask_ratio": round(float((mask == 0).mean()) * 100.0, 3),
        }
    return mask


def luma_exclusion_mask(
    rgb: np.ndarray,
    level: float = 243.0,
    dilation: int = 2,
    open_radius: int = 1,
    return_stats: bool = False,
):
    """Pure-luminance RealityScan exclusion mask — **no model, no GPU** (Backend B).

    Marks pixels brighter than ``level`` (luma 0-255) as excluded reflection, in the
    same polarity as :func:`reflection_exclusion_mask`: ``0`` (black) = **exclude**,
    ``255`` (white) = **keep**. A fast, deterministic fallback for when the
    UnReflectAnything weights / GPU aren't available, or as an A/B baseline.

    Luma alone cannot tell a blown specular from a bright *diffuse* surface (sky, white
    paint, light bodywork), so keep ``level`` HIGH (tight) to avoid masking those.
    ``open_radius`` despeckles and ``dilation`` covers reflection halos — keep both small.
    With ``return_stats`` returns ``(mask, stats)`` (candidate ratio before morphology,
    final ratio after).
    """
    cand = luma_candidate_mask(rgb, level) > 0
    candidate_ratio = float(cand.mean()) * 100.0

    refl = _morph_clean(cand.astype(np.uint8), open_radius, dilation)
    mask = np.where(refl > 0, np.uint8(0), np.uint8(255))  # bright reflection -> black (excluded)
    if return_stats:
        return mask, {
            "candidate_pixel_ratio": round(candidate_ratio, 3),
            "final_mask_ratio": round(float((mask == 0).mean()) * 100.0, 3),
        }
    return mask


def reflection_candidate_mask(
    before: np.ndarray,
    after: np.ndarray,
    drop_level: float = 12.0,
    highlight_gate: float = 250.0,
) -> np.ndarray:
    """Pre-morphology reflection candidate as 0/255 uint8 (255 = candidate reflection).

    This is exactly what :func:`reflection_exclusion_mask` gates on *before* the
    morphological open/dilate; exposed so the diagnostic view can show what the
    threshold selected, separately from the cleaned-up final mask.
    """
    before = np.asarray(before)
    after = np.asarray(after)
    if before.shape != after.shape:
        h = min(before.shape[0], after.shape[0])
        w = min(before.shape[1], after.shape[1])
        before = before[:h, :w]
        after = after[:h, :w]
    lb = to_luma(before)
    la = to_luma(after)
    refl = (lb - la) >= drop_level
    if highlight_gate and highlight_gate > 0:
        refl &= lb >= highlight_gate
    return (refl.astype(np.uint8) * 255)


def luma_candidate_mask(rgb: np.ndarray, level: float = 243.0) -> np.ndarray:
    """Pre-morphology pure-luma candidate as 0/255 uint8 (255 = pixel brighter than ``level``)."""
    return ((to_luma(np.asarray(rgb)) >= level).astype(np.uint8) * 255)


def mask_overlay(
    rgb: np.ndarray,
    exclusion_mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Tint the EXCLUDED region (``exclusion_mask == 0``) of an RGB image for the eye.

    Returns a uint8 RGB copy where pixels the RealityScan mask excludes (black) are
    blended toward ``color`` by ``alpha`` and kept pixels are unchanged — so you can
    see exactly what alignment will ignore, overlaid on the photo.
    """
    rgb = np.asarray(rgb).astype(np.float32)
    m = np.asarray(exclusion_mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != rgb.shape[:2]:
        h = min(m.shape[0], rgb.shape[0])
        w = min(m.shape[1], rgb.shape[1])
        rgb = rgb[:h, :w]
        m = m[:h, :w]
    excl = (m == 0)[..., None]
    tint = np.array(color, dtype=np.float32)
    out = np.where(excl, rgb * (1.0 - alpha) + tint * alpha, rgb)
    return np.clip(out, 0, 255).astype(np.uint8)
