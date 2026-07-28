#!/usr/bin/env python3
"""Count wrinkles from a trained density U-Net checkpoint.

Reads the predicted density map two independent ways:

  * PEAK count     = number of plateau-collapsed local maxima above a RELATIVE
                     threshold (``peak_thresh`` * this crop's heatmap max, with
                     an absolute ``peak_floor``)
  * INTEGRAL count = sum(predicted map inside mask) / one-bump volume

If the two disagree by more than ``disagree_thresh`` the result is flagged
(``disagree=True``) as uncertain -- typically dense/merged wrinkles (peaks
under-count) or diffuse noise (integral over-counts).

Usage as a library:
    import count_wrinkle_unet as cw
    model, meta = cw.load_model("wrinkle_unet.pt")
    result = cw.count_crop(model, meta, "crystal_db/crystals/seq3942__gds29.png")
    cw.render_debug(crop_gray, mask, result, "out/debug.png")

CLI:
    python3 count_wrinkle_unet.py wrinkle_unet.pt crop.png --debug out/dbg.png
"""

import argparse
import os

import numpy as np

import wrinkle_unet as wu

# RELATIVE peak threshold: a candidate peak must exceed this FRACTION of the
# crop's OWN heatmap max (not an absolute value). This de-plateaus dense crops,
# whose targets are sums of many overlapping bumps and so reach values >> 1.0.
# Paired with an absolute floor so near-empty crops don't turn noise into peaks.
# Tuned on the frame-held-out 5-fold CV (dense sigma=3, 155 crops): this setting
# fixes the dense-bin peak blow-up (6+ MAE 19.3 -> 6.2) with NO sparse regression
# (0-2 MAE stays 0.39). See find_peaks for the plateau-collapsing NMS.
DEFAULT_PEAK_THRESH = 0.30      # RELATIVE: fraction of THIS crop's heatmap max
DEFAULT_PEAK_FLOOR = 0.35       # absolute floor; suppresses noise on empty crops
DEFAULT_MIN_DISTANCE = 3        # min separation (px) between counted peaks
DEFAULT_DISAGREE_THRESH = 2.0   # |peak - integral| above this => uncertain


# ---------------------------------------------------------------------------
# Model loading + inference
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, device=None):
    """Load a trained checkpoint -> (model.eval(), meta dict)."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = ckpt.get("base", 16)
    meta = {
        "base": base,
        "size": ckpt.get("size", 256),
        "sigma": ckpt.get("sigma", wu.DEFAULT_SIGMA),
    }
    model = wu.build_unet(in_ch=1, base=base, out_ch=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    meta["device"] = str(device)
    return model, meta


def predict_heatmap(model, meta, gray, mask):
    """Run the U-Net on a single (gray, mask) pair -> predicted density map.

    ``gray`` and ``mask`` must already be square and sized to meta["size"].
    Returns a (H, W) float32 density map (masked).
    """
    import torch
    device = meta.get("device", "cpu")
    img = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(img)[0, 0].cpu().numpy().astype(np.float32)
    pred *= (np.asarray(mask) > 0).astype(np.float32)
    return pred


# ---------------------------------------------------------------------------
# Reading counts off the heatmap
# ---------------------------------------------------------------------------

def find_peaks(heatmap, peak_thresh=DEFAULT_PEAK_THRESH,
               min_distance=DEFAULT_MIN_DISTANCE, peak_floor=DEFAULT_PEAK_FLOOR):
    """Count peaks with a RELATIVE, plateau-aware detector.

    ``peak_thresh`` is a FRACTION of this crop's own heatmap max (not an
    absolute level); the effective cut is ``max(peak_thresh * hm.max(),
    peak_floor)``. Candidate maxima are local maxima within a
    ``2*min_distance+1`` window. Crucially, a broad flat plateau (common on
    dense crops, where many wrinkle bumps overlap into one high region) has
    *every* plateau pixel tie the local max; naive ``hm == maxf`` counts them
    all and explodes the count. We collapse each connected plateau/blob to a
    SINGLE peak via connected-component labelling, so one bright region -> one
    count. Returns an array of (x, y) representative peak coords.
    """
    from scipy import ndimage
    hm = np.asarray(heatmap, dtype=np.float32)
    mx = float(hm.max())
    if mx <= 0.0:
        return np.zeros((0, 2), dtype=np.int32)
    thr = max(float(peak_thresh) * mx, float(peak_floor))
    size = 2 * int(min_distance) + 1
    maxf = ndimage.maximum_filter(hm, size=size, mode="constant", cval=0.0)
    peak_mask = (hm == maxf) & (hm >= thr)
    # collapse connected plateau pixels to one peak each (8-connectivity)
    labels, n = ndimage.label(peak_mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros((0, 2), dtype=np.int32)
    # one representative coordinate (centroid) per labelled region
    centers = ndimage.center_of_mass(peak_mask, labels, range(1, n + 1))
    xs = np.array([c[1] for c in centers])
    ys = np.array([c[0] for c in centers])
    return np.round(np.stack([xs, ys], axis=1)).astype(np.int32)


def integral_count(heatmap, sigma):
    """INTEGRAL count = summed density / one unit-bump volume."""
    return float(np.asarray(heatmap).sum() / wu.bump_volume(sigma))


# Below this peak count we trust the PEAK reading; at/above it we fall back to
# INTEGRAL. On the frame-held-out 5-fold CV (dense sigma=3, 155 crops) this
# hybrid scored the best overall count-MAE (2.44) -- it takes peak's precision
# on the sparse majority (0-2 MAE 0.39) and integral's robustness on dense
# crops where peaks merge (6+ MAE 5.28), beating either metric alone
# (peak 2.74, integral 2.57).
HYBRID_PEAK_CUTOFF = 2


def hybrid_count(peak_count, integral, cutoff=HYBRID_PEAK_CUTOFF):
    """Single best-estimate count: peak when sparse, integral when dense.

    Returns a rounded integer. ``peak_count`` is trusted for low counts
    (<= ``cutoff``); otherwise the (rounded) ``integral`` count is used, since
    on dense crops peaks merge and integral tracks the true count better.
    """
    if int(peak_count) <= int(cutoff):
        return int(peak_count)
    return int(round(float(integral)))


def count_from_heatmap(heatmap, sigma, peak_thresh=DEFAULT_PEAK_THRESH,
                       min_distance=DEFAULT_MIN_DISTANCE,
                       disagree_thresh=DEFAULT_DISAGREE_THRESH,
                       peak_floor=DEFAULT_PEAK_FLOOR,
                       hybrid_cutoff=HYBRID_PEAK_CUTOFF):
    """Read peak + integral counts off a heatmap.

    Returns both raw readings plus a single ready-to-use ``count`` (the hybrid
    best estimate, see ``hybrid_count``) and a ``disagree`` uncertainty flag
    (|peak - integral| > ``disagree_thresh``), which fires mainly on dense
    crops where peaks merge -- exactly the cases the hybrid routes to integral.
    """
    peaks = find_peaks(heatmap, peak_thresh, min_distance, peak_floor)
    peak_count = int(len(peaks))
    integ = integral_count(heatmap, sigma)
    return {
        "count": hybrid_count(peak_count, integ, hybrid_cutoff),
        "peak_count": peak_count,
        "integral_count": integ,
        "peaks": peaks,
        "disagree": bool(abs(peak_count - integ) > disagree_thresh),
        "heatmap": heatmap,
    }


def count_crop(model, meta, crop_path, peak_thresh=DEFAULT_PEAK_THRESH,
               min_distance=DEFAULT_MIN_DISTANCE,
               disagree_thresh=DEFAULT_DISAGREE_THRESH,
               peak_floor=DEFAULT_PEAK_FLOOR,
               hybrid_cutoff=HYBRID_PEAK_CUTOFF):
    """Full path: load an RGBA crop PNG, letterbox, infer, read counts.

    Returns the count_from_heatmap dict (including the ready-to-use ``count``
    field and ``disagree`` flag) plus the preprocessed ``gray`` and ``mask``
    used (handy for render_debug).
    """
    size = meta.get("size", 256)
    gray, mask = wu.load_crop_gray(crop_path)
    gray, mask, _ = wu.letterbox_to_square(gray, mask, [], size)
    heatmap = predict_heatmap(model, meta, gray, mask)
    result = count_from_heatmap(heatmap, meta.get("sigma", wu.DEFAULT_SIGMA),
                                peak_thresh, min_distance, disagree_thresh,
                                peak_floor, hybrid_cutoff)
    result["gray"] = gray
    result["mask"] = mask
    return result


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def _colorize(heatmap):
    """Map a density heatmap to an RGB overlay (blue->green->red)."""
    hm = np.asarray(heatmap, dtype=np.float32)
    hi = float(hm.max())
    norm = hm / hi if hi > 1e-6 else hm
    # simple 3-stop colormap
    r = np.clip(1.5 * norm - 0.5, 0, 1)
    g = np.clip(1.0 - np.abs(norm - 0.5) * 2.0, 0, 1)
    b = np.clip(1.0 - 1.5 * norm, 0, 1)
    return np.stack([r, g, b], axis=-1)


def render_debug(gray, mask, result, out_path, alpha=0.55):
    """Overlay the predicted heatmap + marked peaks on the crop; save a PNG.

    ``gray`` is the single-channel input (square, [0,1]); ``result`` is the
    dict from count_crop / count_from_heatmap.
    """
    from PIL import Image, ImageDraw
    gray = np.asarray(gray, dtype=np.float32)
    heatmap = np.asarray(result["heatmap"], dtype=np.float32)

    base = np.stack([gray, gray, gray], axis=-1)          # grayscale RGB
    heat = _colorize(heatmap)
    hm_alpha = (heatmap / heatmap.max() if heatmap.max() > 1e-6
                else heatmap)[..., None] * alpha
    blended = np.clip(base * (1 - hm_alpha) + heat * hm_alpha, 0, 1)
    img = Image.fromarray((blended * 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(img)
    for (x, y) in result.get("peaks", []):
        x, y = int(x), int(y)
        r = 4
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 0), width=1)
        draw.line([x - r - 2, y, x + r + 2, y], fill=(255, 255, 0), width=1)
        draw.line([x, y - r - 2, x, y + r + 2], fill=(255, 255, 0), width=1)

    label = "peak=%d  integral=%.1f  %s" % (
        result["peak_count"], result["integral_count"],
        "DISAGREE" if result["disagree"] else "ok")
    draw.rectangle([0, 0, 8 + 7 * len(label), 14], fill=(0, 0, 0))
    draw.text((3, 2), label, fill=(255, 255, 0))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", help="trained wrinkle_unet.pt")
    p.add_argument("crop", help="RGBA crop PNG")
    p.add_argument("--peak-thresh", type=float, default=DEFAULT_PEAK_THRESH,
                   help="RELATIVE: fraction of this crop's heatmap max")
    p.add_argument("--peak-floor", type=float, default=DEFAULT_PEAK_FLOOR,
                   help="absolute floor for the peak threshold")
    p.add_argument("--min-distance", type=int, default=DEFAULT_MIN_DISTANCE)
    p.add_argument("--disagree-thresh", type=float, default=DEFAULT_DISAGREE_THRESH)
    p.add_argument("--debug", default=None, help="write an overlay PNG here")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is not available. Run inference on the GPU box "
              "(pip install torch).")
        raise SystemExit(1)

    model, meta = load_model(args.checkpoint, device=args.device)
    result = count_crop(model, meta, args.crop,
                        peak_thresh=args.peak_thresh,
                        min_distance=args.min_distance,
                        disagree_thresh=args.disagree_thresh,
                        peak_floor=args.peak_floor)
    print("count          = %d%s" % (result["count"],
          "  (uncertain)" if result["disagree"] else ""))
    print("peak_count     = %d" % result["peak_count"])
    print("integral_count = %.2f" % result["integral_count"])
    print("disagree       = %s" % result["disagree"])
    if args.debug:
        out = render_debug(result["gray"], result["mask"], result, args.debug)
        print("debug overlay  -> %s" % out)


if __name__ == "__main__":
    main()
