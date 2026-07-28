#!/usr/bin/env python3
"""Ridge-instance extraction tuned for NATIVE-resolution (sigma=7) heatmaps.

Why a wrapper instead of ``ridge_count`` defaults
-------------------------------------------------
``ridge_count``'s defaults were frozen for 128px sigma-3 letterboxed maps whose
amplitude sits near 1.0. Native maps are broad, smooth and LOW amplitude -- max
0.028-0.387 across crops (median 0.20). Two absolute gates then collapse the
pipeline (verified by stage instrumentation):

  * ``floor=0.35`` on the HIGH hysteresis level. ``hi = max(hi_frac*max, floor)``
    becomes 0.35 > crop max on nearly every native crop -> empty filament mask
    -> 0 instances even on 27-wrinkle crops.
  * ``inst_floor=0.25`` absolute vs native mean densities ~0.02-0.4 -> drops
    every surviving instance.

Fix: normalize the heatmap to its own max BEFORE ridge extraction, so hi_frac /
floor / inst_floor all act as fractions of the crop's own amplitude (robust
across the ~14x amplitude spread). Coherence is already amplitude-invariant.
Geometry (tensor/crest/gap/lengths) is scaled to the sigma-7 bump. The dense
integral fallback is DISABLED here -- native counting keeps the calibrated
INTEGRAL as its primary count; ridge extraction is only the MARKER source, so
the number of instances is the peak_count and each instance's densest vertex is
its representative marker point.

Parameters were grid-tuned on pre-4213 tuning crops optimizing click-to-polyline
F1 at tol 25 (tiebreak: instance-count MAE); see eval_ridge_tune.py.
"""
import numpy as np

import ridge_count as rc

# Frozen native ridge parameters (grid-tuned). ``geom`` is fixed at the sigma-7
# scaling (factor 7/3 over the sigma-3 defaults); the gates below were searched.
# Placeholders here are overwritten by eval_ridge_tune.py's winner via
# ``set_params`` / by editing this dict after the freeze.
NATIVE_RIDGE_PARAMS = {
    # --- fixed sigma-7 geometry (pixels) ---
    "sigma": 7.0,
    "tensor_sigma": 4.667,      # DEFAULT_TENSOR_SIGMA * 7/3
    "crest_step": 3.5,          # DEFAULT_CREST_STEP * 7/3
    "gap_px": 23.333,           # DEFAULT_GAP_PX * 7/3
    "gap_angle_deg": 30.0,
    "merge_angle_deg": 30.0,
    "min_ridge_len": 16.0,      # DEFAULT_MIN_RIDGE_LEN * 7/3, rounded
    # --- grid-tuned gates (fractions of the crop's own max after normalize) ---
    # Frozen from a grid + boundary refinement on 60 pre-4213 tuning crops
    # (15 sparse / 14 medium / 31 dense) optimizing click-to-polyline F1 @ tol
    # 25 (tiebreak: instance-count MAE). Winner F1=0.711 p=0.794 r=0.643
    # inst-MAE=2.57. Notes: inst_floor beyond ~0.2 craters recall (dense-crop
    # crests are low-density); min_instance_len is a de-fragmentation knob --
    # raising it prunes spurious short strokes inside dense filament regions
    # while the per-region fallback keeps every real wrinkle at >=1 instance, so
    # recall is flat 44->120 and F1 plateaus by ~90.
    "hi_frac": 0.50,
    "lo_frac": 0.15,
    "floor": 0.0,               # absolute floor disabled; hi_frac governs
    "coh_min": 0.30,
    "inst_floor": 0.20,
    "min_instance_len": 90.0,
}


def _representatives(instances, hm):
    """One (x, y) marker per instance: the polyline vertex of max heatmap density."""
    H, W = hm.shape
    reps = []
    for inst in instances:
        poly = np.asarray(inst["polyline"], dtype=float).reshape(-1, 2)
        xi = np.clip(np.round(poly[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(poly[:, 1]).astype(int), 0, H - 1)
        k = int(np.argmax(hm[yi, xi]))
        reps.append((float(poly[k, 0]), float(poly[k, 1])))
    return np.asarray(reps, dtype=float).reshape(-1, 2)


def extract_ridge_instances(heatmap, gray=None, mask=None,
                            params=NATIVE_RIDGE_PARAMS):
    """Native ridge-instance extraction -> dict.

    Normalizes ``heatmap`` to its own max, then runs
    ``ridge_count.count_from_heatmap_ridge`` (dense fallback OFF) with the frozen
    native gates. Returns::

        {"instances": [...], "polylines": [ (N,2) xy, ... ],
         "peaks": (K,2) float,   # one representative marker per instance
         "peak_count": int}      # number of ridge instances
    """
    hm = np.asarray(heatmap, dtype=np.float32)
    if mask is not None:
        hm = hm * (np.asarray(mask) > 0)
    mx = float(hm.max())
    if mx <= 0:
        return {"instances": [], "polylines": [],
                "peaks": np.zeros((0, 2)), "peak_count": 0}
    hmn = hm / mx
    p = dict(params)
    res = rc.count_from_heatmap_ridge(
        hmn, gray=gray, mask=None, dense_fallback=False,
        hi_frac=p["hi_frac"], lo_frac=p["lo_frac"], floor=p["floor"],
        merge_angle_deg=p["merge_angle_deg"], gap_px=p["gap_px"],
        gap_angle_deg=p["gap_angle_deg"], min_ridge_len=p["min_ridge_len"],
        min_instance_len=p["min_instance_len"], tensor_sigma=p["tensor_sigma"],
        crest_step=p["crest_step"], coh_min=p["coh_min"], sigma=p["sigma"],
        inst_floor=p["inst_floor"])
    instances = res["instances"]
    polylines = [np.asarray(i["polyline"], dtype=float).reshape(-1, 2)
                 for i in instances]
    peaks = _representatives(instances, hmn)
    return {"instances": instances, "polylines": polylines,
            "peaks": peaks, "peak_count": len(instances)}
