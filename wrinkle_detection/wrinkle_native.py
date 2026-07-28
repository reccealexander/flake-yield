#!/usr/bin/env python3
"""Native-resolution wrinkle-counting density U-Net.

The production model (``count_wrinkle_unet`` + ``wrinkle_unet_sigma3_dense.pt``)
letterboxes every crop to 128x128 before inference. With a median long side of
~1324 px that is a ~10x downscale, which smears thin wrinkle lines (3-8 px wide)
into nothing -- the "resolution pathology" this module fixes.

Approach
--------
Train and infer at NATIVE resolution using fixed-size patches instead of a
single letterboxed frame:

  * Preprocessing is identical in spirit to ``wrinkle_unet.load_crop_gray``:
    green channel, 2-98 percentile stretch computed INSIDE the crystal mask over
    the WHOLE crop (before any patching), background set to ``BG_VALUE``. We call
    ``wu.load_crop_gray`` directly so the convention is byte-for-byte the same.
  * Training samples random ``PATCH`` x ``PATCH`` native windows whose centre
    falls inside (or within a small margin of) the crystal mask. The target is a
    sum of unit Gaussian bumps at the annotated points that land in the window,
    rendered at NATIVE scale with ``NATIVE_SIGMA``. Loss is masked MSE (mask =
    the crystal alpha inside the window). Crops smaller than ``PATCH`` on a side
    are padded with ``BG_VALUE`` (mask 0).
  * Inference tiles the whole crop with a ``STRIDE`` = PATCH/2 sliding window and
    blends overlapping predictions with a smooth (Hann) weight window, then
    multiplies by the crystal mask -> a full-crop, native-resolution density map.

Sigma choice (NATIVE_SIGMA = 7 px)
----------------------------------
Wrinkle lines are ~3-8 px wide at native resolution. A point-supervised density
bump must be:
  * wide enough that the network can actually regress it (a 1-2 px spike is
    below what a base-16 U-Net with 4 poolings reproduces cleanly), and
  * narrow enough that neighbouring wrinkles stay separable as distinct peaks.
sigma = 7 gives a bump FWHM of 2.355*7 ~ 16.5 px, comparable to the line width
plus a margin, and lets the peak detector separate wrinkles that are >= ~7 px
apart. one-bump volume = 2*pi*sigma^2 ~ 307.9. This is recorded in the
checkpoint meta so counting always uses the sigma the model was trained with.
"""

import math
import os

import numpy as np

import wrinkle_unet as wu
import count_wrinkle_unet as cw

# --- native-scale constants (recorded into the checkpoint) --------------------
PATCH = 256                 # native-resolution training/inference window (px)
STRIDE = 128                # sliding-window stride for tiled inference (PATCH/2)
NATIVE_SIGMA = 7.0          # Gaussian bump sigma in NATIVE pixels (see module doc)
BG_VALUE = wu.BG_VALUE      # background value outside the mask
CENTER_MARGIN = 32          # patch centre may fall this many px outside the mask

# Peak detection at native scale: peaks must be >= ~sigma apart. The production
# find_peaks uses min_distance=3 tuned for the 128-px letterboxed maps; at native
# scale that would split one bump into several maxima, so we widen it to ~sigma.
NATIVE_MIN_DISTANCE = int(round(NATIVE_SIGMA))     # 7 px
NATIVE_PEAK_THRESH = cw.DEFAULT_PEAK_THRESH        # relative frac of crop max (0.30)
# The 128 model's absolute peak_floor (0.35) is WRONG at native scale: the native
# model regresses broad sigma=7 bumps and systematically under-shoots amplitude
# (measured native heatmap maxima span ~0.14-0.55, not ~1.0), so a 0.35 floor
# zeros the whole peak mask on low-amplitude crops -> peak_count 0 -> the ported
# hybrid collapsed accurate integral estimates to 0. Lower the floor to sit below
# the observed amplitude range; the relative 0.30*max cut still de-plateaus.
NATIVE_PEAK_FLOOR = 0.08
# At native scale the INTEGRAL is the calibrated count (density regression: sum /
# one-bump volume tracks true counts on wrinkled crops, e.g. int 8.4 for true 8,
# 17.7 for true 15), whereas the low, broad native peaks merge and under-count on
# dense crops. So the native primary ``count`` is round(integral); peak_count is
# still exposed for the sparse cross-check and the disagree flag.
NATIVE_COUNT_MODE = "integral"
# The native model lays down a faint DIFFUSE density baseline over featureless
# crystal area (per-pixel << the real-bump peaks). Integrated over a large or
# high-aspect crop this hallucinated mass inflates the count (empty crops read
# 4-6). We remove it by zeroing heatmap pixels below a RELATIVE floor -- a
# fraction of THIS crop's own heatmap max -- before integrating. Relative (not
# absolute) because native amplitude varies ~3-4x across crops (max 0.14-0.55);
# a fixed absolute floor that clears an empty crop would gut a low-amplitude
# DENSE crop. Swept on val, rel=0.05 minimises overall count-MAE (2.36) while
# keeping the sparse bin clean, so it is a background subtraction, not a fragile
# tuned constant.
NATIVE_INTEGRAL_REL_FLOOR = 0.05


# ---------------------------------------------------------------------------
# Whole-crop preprocessing (native resolution) -- reuse load_crop_gray
# ---------------------------------------------------------------------------

def load_crop_native(path, bg_value=BG_VALUE):
    """Load an RGBA crop at NATIVE resolution -> (gray, mask).

    Thin wrapper over ``wrinkle_unet.load_crop_gray`` so the normalization
    convention (green channel, 2-98 stretch inside the mask over the whole crop,
    background = bg_value) is exactly the production one. Returns full-size
    arrays -- no letterboxing.
    """
    return wu.load_crop_gray(path, bg_value=bg_value)


# ---------------------------------------------------------------------------
# Patch extraction (torch-free)
# ---------------------------------------------------------------------------

def _pad_to_patch(gray, mask, patch=PATCH, bg_value=BG_VALUE):
    """Pad a possibly-smaller-than-patch crop up to at least patch x patch.

    Returns (gray_p, mask_p, ox, oy) where (ox, oy) is the top-left offset at
    which the original crop was placed (0 unless padding was needed on that
    axis; centred padding). Padding uses ``bg_value`` for gray and 0 for mask.
    """
    H, W = gray.shape
    padH = max(0, patch - H)
    padW = max(0, patch - W)
    if padH == 0 and padW == 0:
        return gray, mask, 0, 0
    oy, ox = padH // 2, padW // 2
    gp = np.full((H + padH, W + padW), float(bg_value), dtype=np.float32)
    mp = np.zeros((H + padH, W + padW), dtype=np.float32)
    gp[oy:oy + H, ox:ox + W] = gray
    mp[oy:oy + H, ox:ox + W] = mask
    return gp, mp, ox, oy


def extract_patch(gray, mask, points, cx, cy, patch=PATCH, bg_value=BG_VALUE):
    """Cut a patch x patch window centred at (cx, cy) in the (padded) crop.

    Coordinates cx, cy and the input ``points`` are in the SAME frame as
    ``gray``/``mask`` (already padded if needed). The window is clamped to lie
    inside the array. Returns (gray_win, mask_win, pts_win) with pts_win the
    points that fall inside the window, expressed in window-local coords.
    """
    H, W = gray.shape
    x0 = int(round(cx)) - patch // 2
    y0 = int(round(cy)) - patch // 2
    x0 = min(max(0, x0), max(0, W - patch))
    y0 = min(max(0, y0), max(0, H - patch))
    x1, y1 = x0 + patch, y0 + patch
    gwin = np.full((patch, patch), float(bg_value), dtype=np.float32)
    mwin = np.zeros((patch, patch), dtype=np.float32)
    gh, gw = min(y1, H) - y0, min(x1, W) - x0
    gwin[:gh, :gw] = gray[y0:y0 + gh, x0:x0 + gw]
    mwin[:gh, :gw] = mask[y0:y0 + gh, x0:x0 + gw]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts):
        local = pts - np.array([x0, y0], dtype=np.float32)
        keep = ((local[:, 0] >= 0) & (local[:, 0] < patch)
                & (local[:, 1] >= 0) & (local[:, 1] < patch))
        local = local[keep]
    else:
        local = np.zeros((0, 2), dtype=np.float32)
    return gwin, mwin, local


def sample_patch_center(mask, rng, margin=CENTER_MARGIN):
    """Pick a random (cx, cy) whose location is inside the mask (+/- margin).

    Returns integer (cx, cy). If the mask is empty, returns the array centre.
    """
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        H, W = mask.shape
        return W // 2, H // 2
    i = int(rng.integers(0, len(xs)))
    cx = int(xs[i]) + int(rng.integers(-margin, margin + 1))
    cy = int(ys[i]) + int(rng.integers(-margin, margin + 1))
    return cx, cy


# ---------------------------------------------------------------------------
# Patch dataset (torch imported lazily inside the factory)
# ---------------------------------------------------------------------------

def make_patch_dataset_class():
    """Return the ``WrinklePatchDataset`` class (torch required)."""
    from torch.utils.data import Dataset
    import torch

    class WrinklePatchDataset(Dataset):
        """Random native-resolution patches with Gaussian-bump density targets.

        Each __getitem__ loads a crop (cached), pads it up to at least PATCH,
        samples a mask-centred window, renders the native-scale target for the
        points inside it, optionally augments, and returns
        (img, target, mask) each shaped (1, PATCH, PATCH) float32.

        ``patches_per_crop`` sets how many random windows make up one epoch per
        crop (dataset length = n_crops * patches_per_crop).
        """

        def __init__(self, items, patch=PATCH, sigma=NATIVE_SIGMA, augment=True,
                     seed=0, patches_per_crop=8, max_angle=12.0,
                     center_margin=CENTER_MARGIN):
            self.items = list(items)
            self.patch = int(patch)
            self.sigma = float(sigma)
            self.augment = bool(augment)
            self._base_seed = int(seed)
            self.patches_per_crop = int(patches_per_crop)
            self.max_angle = float(max_angle)
            self.center_margin = int(center_margin)
            self._cache = {}

        def __len__(self):
            return len(self.items) * self.patches_per_crop

        def _load(self, ci):
            if ci not in self._cache:
                path, points = self.items[ci]
                gray, mask = load_crop_native(path)
                gray, mask, ox, oy = _pad_to_patch(gray, mask, self.patch)
                pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
                if len(pts):
                    pts = pts + np.array([ox, oy], dtype=np.float32)
                self._cache[ci] = (gray, mask, pts)
            return self._cache[ci]

        def __getitem__(self, i):
            ci = i % len(self.items)
            gray, mask, pts = self._load(ci)
            rng = np.random.default_rng(
                (self._base_seed * 1000003 + i * 9176
                 + np.random.randint(1 << 30)) & 0x7FFFFFFF)
            cx, cy = sample_patch_center(mask, rng, self.center_margin)
            gwin, mwin, lpts = extract_patch(gray, mask, pts, cx, cy, self.patch)
            if self.augment:
                gwin, mwin, lpts = wu.augment_sample(
                    gwin, mwin, lpts, rng, max_angle=self.max_angle)
            target = wu.make_target(lpts, (self.patch, self.patch), mwin,
                                    self.sigma)
            img_t = torch.from_numpy(gwin).unsqueeze(0).float()
            tgt_t = torch.from_numpy(target).unsqueeze(0).float()
            msk_t = torch.from_numpy(mwin).unsqueeze(0).float()
            return img_t, tgt_t, msk_t

    return WrinklePatchDataset


# ---------------------------------------------------------------------------
# Tiled sliding-window inference
# ---------------------------------------------------------------------------

def _hann2d(patch):
    """2-D Hann weight window, floored so it is strictly positive everywhere.

    The floor (small epsilon) guarantees the overlap-blend denominator never
    hits zero at a crop border pixel that happens to coincide with a window
    edge, while the raised-cosine shape suppresses seam discontinuities.
    """
    w = np.hanning(patch).astype(np.float32)
    w = np.maximum(w, 1e-3)
    win = np.outer(w, w).astype(np.float32)
    return win


def _tile_starts(length, patch, stride):
    """Start offsets covering [0, length) with the last tile flush to the end."""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def predict_heatmap_native(model, meta, gray, mask, patch=PATCH, stride=STRIDE):
    """Tiled sliding-window inference -> full-crop native density map (masked).

    ``gray``/``mask`` are native-resolution whole-crop arrays. Overlapping tile
    predictions are averaged with a Hann weight window; the result is multiplied
    by the crystal mask.
    """
    import torch
    device = meta.get("device", "cpu")
    patch = int(meta.get("patch", patch))
    stride = int(stride)
    gray_p, mask_p, ox, oy = _pad_to_patch(gray, mask, patch)
    Hp, Wp = gray_p.shape
    acc = np.zeros((Hp, Wp), dtype=np.float64)
    wsum = np.zeros((Hp, Wp), dtype=np.float64)
    win = _hann2d(patch).astype(np.float64)

    model.eval()
    ys = _tile_starts(Hp, patch, stride)
    xs = _tile_starts(Wp, patch, stride)
    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                g = gray_p[y0:y0 + patch, x0:x0 + patch]
                inp = torch.from_numpy(np.ascontiguousarray(g))
                inp = inp.unsqueeze(0).unsqueeze(0).float().to(device)
                pred = model(inp)[0, 0].cpu().numpy().astype(np.float64)
                acc[y0:y0 + patch, x0:x0 + patch] += pred * win
                wsum[y0:y0 + patch, x0:x0 + patch] += win
    out = np.zeros_like(acc)
    nz = wsum > 0
    out[nz] = acc[nz] / wsum[nz]
    out = out * (mask_p > 0)
    # crop back off any padding
    H, W = gray.shape
    out = out[oy:oy + H, ox:ox + W]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Counting at native scale
# ---------------------------------------------------------------------------

def _group_peaks_by_ridge(hm, peaks, rel_thresh=0.30):
    """Collapse bead-peaks along each connected heatmap ridge to ONE.

    ``rel_thresh`` is a fraction of the crop's heatmap max (scale-invariant,
    matching the peak finder's relative convention). Returns (count,
    representatives): one representative coordinate per ridge component --
    the strongest peak on it -- so debug overlays draw exactly ``count``
    markers. A peak on background (component of its own) represents itself.
    """
    import numpy as _np
    peaks = _np.asarray(peaks, dtype=int).reshape(-1, 2)
    if len(peaks) == 0:
        return 0, peaks
    from scipy import ndimage
    mask = hm > (float(rel_thresh) * float(hm.max()))
    labels, _ = ndimage.label(mask, structure=_np.ones((3, 3)))
    best = {}                                  # component -> (value, (x, y))
    reps = []
    for i, (x, y) in enumerate(peaks):
        lab = int(labels[y, x])
        val = float(hm[y, x])
        if lab == 0:
            reps.append((x, y))                # background single: itself
        elif lab not in best or val > best[lab][0]:
            best[lab] = (val, (x, y))
    reps.extend(xy for _, xy in best.values())
    reps = _np.asarray(sorted(reps), dtype=_np.int32).reshape(-1, 2)
    return int(len(reps)), reps


def count_from_heatmap_native(heatmap, sigma=NATIVE_SIGMA,
                              peak_thresh=NATIVE_PEAK_THRESH,
                              min_distance=NATIVE_MIN_DISTANCE,
                              peak_floor=NATIVE_PEAK_FLOOR,
                              disagree_thresh=cw.DEFAULT_DISAGREE_THRESH,
                              count_mode=NATIVE_COUNT_MODE,
                              integral_rel_floor=NATIVE_INTEGRAL_REL_FLOOR):
    """Read peak + integral counts off a native-scale heatmap.

    Reuses ``count_wrinkle_unet.find_peaks``/``integral_count`` but with a
    native-scale ``min_distance`` (~sigma, so one broad bump is not split into
    several maxima) and a native-scale ``peak_floor`` (the native model's peaks
    are low-amplitude). Before integrating, heatmap pixels below
    ``integral_rel_floor * heatmap.max()`` are zeroed to remove the diffuse
    background baseline (relative -> scale-invariant across crops). The primary
    ``count`` is the (floored) INTEGRAL rounded when ``count_mode == "integral"``
    -- the calibrated readout at native scale -- otherwise the grouped-peak
    count.

    MARKERS (``peaks`` / ``peak_count``) come from ridge-INSTANCE extraction
    (native_ridge): each physical wrinkle is one extended ridge polyline drawn
    as a single marker at its densest vertex, so overlays show one mark per
    wrinkle instead of the isotropic peak finder's bead-string fragmentation.
    The ``disagree`` uncertainty flag is DECOUPLED and still computed from the
    grouped isotropic-peak count vs the integral (``group_peak_count`` is
    exposed), because that disagreement was holdout-validated as a review queue
    (~3.4x error ratio flagged/unflagged) whereas the ridge count tracks the
    integral too closely to flag well.
    """
    import numpy as _np
    hm = _np.asarray(heatmap, dtype=_np.float32)
    # --- grouped isotropic peaks: retained ONLY to drive the disagree flag ---
    # A wrinkle is a LINE: the native heatmap renders it as one elongated ridge,
    # on which the isotropic maxima finder strings MANY "peaks" like beads.
    # _group_peaks_by_ridge collapses bead-peaks per connected heatmap component
    # but still SPLITS one wrinkle into several markers when its ridge sags
    # below the rel threshold mid-line. It is kept here because the disagree
    # flag was holdout-validated on grouped-peak-vs-integral disagreement (28%
    # flag rate, ~3.4x error ratio flagged/unflagged); driving the flag off the
    # ridge instance count instead collapses that discrimination to ~1.2x, so we
    # DECOUPLE: the flag stays on ``group_peak_count``, while the displayed
    # markers come from ridge instances (below).
    gpeaks = cw.find_peaks(hm, peak_thresh, min_distance, peak_floor)
    group_peak_count, _gp_reps = _group_peaks_by_ridge(hm, gpeaks)

    thr = float(integral_rel_floor) * float(hm.max())
    hm_int = _np.where(hm > thr, hm, 0.0)
    integ = cw.integral_count(hm_int, sigma)
    if count_mode == "peak":
        count = group_peak_count
    else:
        count = int(round(integ))

    # --- markers: ridge INSTANCES (one representative per physical wrinkle) ---
    # Retuned for native heatmaps (see native_ridge.NATIVE_RIDGE_PARAMS). Each
    # wrinkle -> one extended ridge polyline -> one marker at its densest vertex,
    # so ``peaks`` draws exactly one mark per wrinkle (holdout marker F1 0.66 vs
    # the grouped-peaks' 0.49 under polyline matching, dense-crop precision
    # 0.47 -> 0.90). ``peak_count`` = number of ridge instances.
    import native_ridge as _nr
    _rr = _nr.extract_ridge_instances(hm, gray=None, mask=None)
    peaks = _rr["peaks"]
    peak_count = _rr["peak_count"]

    # RELATIVE disagree (unchanged, on the grouped-peak count): an absolute gap
    # of 2 is noise on a 30-wrinkle flake. gap > max(thresh, 0.7*integral).
    gap = abs(group_peak_count - integ)
    return {
        "count": count,
        "peak_count": peak_count,
        "group_peak_count": group_peak_count,
        "integral_count": integ,
        "peaks": peaks,
        "instances": _rr["instances"],
        "disagree": bool(gap > max(float(disagree_thresh), 0.7 * integ)),
        "heatmap": heatmap,
    }


def count_crop_native(model, meta, path, peak_thresh=NATIVE_PEAK_THRESH,
                      min_distance=None, peak_floor=NATIVE_PEAK_FLOOR,
                      disagree_thresh=cw.DEFAULT_DISAGREE_THRESH,
                      count_mode=NATIVE_COUNT_MODE):
    """Full path: load an RGBA crop, tiled-infer at native res, read counts.

    Returns the count dict (``count``/``peak_count``/``integral_count``/
    ``peaks``/``disagree``/``heatmap``) plus the ``gray`` and ``mask`` used.
    """
    sigma = float(meta.get("sigma", NATIVE_SIGMA))
    if min_distance is None:
        min_distance = int(meta.get("min_distance", max(1, int(round(sigma)))))
    gray, mask = load_crop_native(path)
    heatmap = predict_heatmap_native(model, meta, gray, mask)
    result = count_from_heatmap_native(
        heatmap, sigma, peak_thresh=peak_thresh, min_distance=min_distance,
        peak_floor=peak_floor, disagree_thresh=disagree_thresh,
        count_mode=count_mode)
    result["gray"] = gray
    result["mask"] = mask
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_native(checkpoint_path, device=None):
    """Load a native checkpoint -> (model.eval(), meta dict)."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = ckpt.get("base", 16)
    meta = {
        "base": base,
        "patch": ckpt.get("patch", PATCH),
        "sigma": ckpt.get("sigma", NATIVE_SIGMA),
        "min_distance": ckpt.get("min_distance", NATIVE_MIN_DISTANCE),
        "norm": ckpt.get("norm", "green_2_98_inside_mask_whole_crop"),
    }
    model = wu.build_unet(in_ch=1, base=base, out_ch=1)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model.to(device)
    meta["device"] = str(device)
    return model, meta


if __name__ == "__main__":
    # torch-free self-check of the target volume / patch geometry
    vol = wu.bump_volume(NATIVE_SIGMA)
    tgt = wu.make_target([(128, 128)], (PATCH, PATCH), None, NATIVE_SIGMA)
    print("native sigma=%.1f  bump_volume=%.2f  single-bump integral=%.4f"
          % (NATIVE_SIGMA, vol, tgt.sum() / vol))
    print("tile starts for 1324px:", _tile_starts(1324, PATCH, STRIDE))
