#!/usr/bin/env python3
"""Far-field registration refinement by edge-snapping from a single anchor.

Once ONE crystal is registered (correct identity, orientation, uniform scale),
the anchor predicts where every other crystal's edges should land to within the
tape strain (~a few %). We don't need to detect a second shape: sample each GDS
edge, search PERPENDICULAR to it for the nearest real image edge (gradient ridge),
and refit the 6-parameter affine over all those edge correspondences. Iterating
(shrinking the search window) tightens the anisotropy. The anchor corners are
pinned with high weight so the solution can't walk off the confirmed crystal.

This is the standard step AFTER the anchor fit (see shape_match.py / the flipped,
proper-rotation, mirror-partner anchor). Validated on seq4045: 2.8% strain,
6 px median edge residual, faint target crystals correctly located. User-confirmed.

Key rules (see memory gds-canonical-orientation):
  - GDS is always x-flipped (F = diag(1,-1)); anchor is a proper rotation.
  - Snap to EDGES (perpendicular gradient ridge), not blobs/centroids.
  - The template says where to look; each crystal still needs an independent
    border check before its occupancy/pass counts.
"""

from pathlib import Path

import cv2
import gdstk
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from itertools import permutations

FRAME = Path(__file__).resolve().parent.parent / "flake_imgs" / "new" / "seq4045.jpg"


def _find_gds():
    """Locate the GDS design, portably. Priority: FLAKE_GDS env var; then a copy
    bundled in the repo (gds/tas_v3_design.gds) so it travels with an rsync to the
    DGX; then the original ~/Downloads location for back-compat on this Mac."""
    import os
    env = os.environ.get("FLAKE_GDS")
    if env:
        return Path(env)
    bundled = Path(__file__).resolve().parent.parent / "gds" / "tas_v3_design.gds"
    if bundled.exists():
        return bundled
    return Path.home() / "Downloads" / "tas_v3_design.gds"


GDS = _find_gds()
SCALE = 13.7
EDGE_STEP = 22          # px between edge samples along each GDS edge
EDGE_MIN = 0.12         # min normalized gradient response to accept a snap
WINDOWS = [60, 45, 35, 28]   # perpendicular search half-width, coarse -> fine
ANCHOR_WEIGHT = 8       # replication weight to pin the anchor corners


def ccw(P):
    c = P.mean(0)
    return P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]


def load_gds_flipped():
    lib = gdstk.read_gds(str(GDS))
    cell = [c for c in lib.cells if c.name == "SharpTipCompactTiling"][0]
    polys = [np.asarray(q.points) for q in cell.get_polygons()]
    ctr = np.vstack(polys).mean(0)
    F = np.array([[1, 0], [0, -1]])            # tape mirror; canonical orientation
    fpolys = [(F @ (q - ctr).T).T for q in polys]
    return polys, fpolys


def _mask_to_poly(seg):
    """Fit a 3/4/6-gon to a boolean mask's outer contour; None if none fits."""
    seg = seg.astype(np.uint8)
    cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.04, 0.015, 0.05):
        ap = cv2.approxPolyDP(c, eps * peri, True).reshape(-1, 2)
        if len(ap) in (3, 4, 6):
            return ap.astype(float)
    return None


def detect_bright(green):
    """Threshold-based bright-polygon detector (fixed exposure; fragile across
    frames). Kept for reference/comparison; the pipeline uses detect_sam."""
    flat = green - ndi.gaussian_filter(green, 120)
    mask = remove_small_objects(flat > threshold_otsu(flat), 10_000)
    mask = ndi.binary_fill_holes(mask)
    lab, _ = ndi.label(mask)
    dets = []
    for pr in regionprops(lab):
        if not (10_000 <= pr.area <= 500_000):
            continue
        if pr.axis_major_length / max(pr.axis_minor_length, 1e-6) < 2.0:
            continue
        ap = _mask_to_poly(lab == pr.label)
        if ap is not None:
            dets.append(ap)
    return dets


SAM_MIN_AREA = 8_000          # keep small shapes; corner-match does the selecting
SAM_MAX_FRAME_FRACTION = 0.70  # above this a mask is the background


SAM_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "sam"


def sam_shapes(green, cache_key=None):
    """Run loose SAM once; return (masks, polys) and cache both per frame.

    Deliberately NOT sam_segment.find_crystals: its elongation/brightness filters
    (built to pick material crystals) strip out the clean triangles that make the
    best registration anchors. We want every well-shaped mask as a candidate and
    let the anchor selection pick; non-crystal masks simply won't score well. This
    mirrors experiments/sam_segment_only.py, which yields ~22 masks on seq4046
    (incl. the center triangles) vs 6 from find_crystals.

    masks (per-crystal boolean arrays) are what the refinement snaps to; polys are
    the corner polygons the anchor is fit from. SAM is slow, so both are cached
    PERSISTENTLY in cache/sam/ (compressed npz; /tmp gets wiped on reboot, which
    silently re-ran SAM on already-processed frames). Legacy /tmp caches are
    migrated on first touch.
    """
    import sys
    from pathlib import Path as _P
    npz = SAM_CACHE_DIR / f"sam_{cache_key}.npz" if cache_key is not None else None
    if npz is not None and npz.exists():
        d = np.load(npz, allow_pickle=True)
        masks = [m for m in d["masks"].astype(bool)]
        polys = [np.asarray(p, float) for p in d["polys"]]
        return masks, polys
    if cache_key is not None:                      # migrate a legacy /tmp cache
        mcache = _P("/tmp") / f"sammask_{cache_key}.npy"
        pcache = _P("/tmp") / f"sampoly_{cache_key}.npy"
        if mcache.exists() and pcache.exists():
            masks = [np.asarray(m, bool) for m in np.load(mcache, allow_pickle=True)]
            polys = [np.asarray(p, float) for p in np.load(pcache, allow_pickle=True)]
            _save_sam_cache(npz, masks, polys)
            return masks, polys
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from flakeyield import sam_segment
    gen = sam_segment._generator(sam_segment.DEFAULT_CHECKPOINT, sam_segment.POINTS_PER_SIDE)
    raw = gen.generate(sam_segment.to_rgb(green))
    masks, polys = [], []
    for m in sorted(raw, key=lambda m: -m["area"]):
        if not (SAM_MIN_AREA <= m["area"] <= SAM_MAX_FRAME_FRACTION * green.size):
            continue
        seg = m["segmentation"].astype(bool)
        ap = _mask_to_poly(seg)
        if ap is None:
            continue
        masks.append(seg)
        polys.append(ap)
    if npz is not None:
        _save_sam_cache(npz, masks, polys)
    return masks, polys


def _save_sam_cache(npz, masks, polys):
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, masks=np.array(masks, dtype=bool),
                        polys=np.array(polys, dtype=object))


def detect_sam(green, cache_key=None):
    """Loose-SAM shape polygons (see sam_shapes); the anchor detector."""
    return sam_shapes(green, cache_key)[1]


def anchor_fit(det_tri, gds_tri_flipped):
    """Proper-rotation Kabsch of a flipped GDS triangle onto a detection."""
    d, gv = ccw(det_tri), ccw(gds_tri_flipped)
    best = None
    for perm in permutations(range(3)):
        s = gv[list(perm)]
        ms, mt = s.mean(0), d.mean(0)
        M = (d - mt).T @ (SCALE * (s - ms))
        U, S, Vt = np.linalg.svd(M)
        R = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        t = mt - SCALE * (R @ ms)
        res = np.sqrt(((SCALE * (s @ R.T) + t - d) ** 2).sum(1)).mean()
        if best is None or res < best[0]:
            best = (res, SCALE * R, t, s, d)
    return best  # res, A, b, anchor_src(gds), anchor_tgt(img)


def edge_map(green):
    gs = ndi.gaussian_filter(green, 1.2)
    E = np.hypot(ndi.sobel(gs, axis=1), ndi.sobel(gs, axis=0))
    return E / (np.percentile(E, 99) + 1e-6)


def snap_correspondences(fpolys, A, b, E, win):
    H, W = E.shape
    SRC, TGT = [], []
    for fq in fpolys:
        for i in range(len(fq)):
            gA, gB = fq[i], fq[(i + 1) % len(fq)]
            iA, iB = gA @ A.T + b, gB @ A.T + b
            iseg = iB - iA
            iL = np.hypot(*iseg)
            if iL < EDGE_STEP:
                continue
            u = iseg / iL
            n = np.array([-u[1], u[0]])
            for tt in np.linspace(0.1, 0.9, max(2, int(iL // EDGE_STEP))):
                gp = gA + tt * (gB - gA)
                ip = iA + tt * iseg
                offs = np.arange(-win, win + 1, 1.0)
                line = ip[None, :] + offs[:, None] * n[None, :]
                xs, ys = line[:, 0], line[:, 1]
                ok = (xs > 1) & (xs < W - 2) & (ys > 1) & (ys < H - 2)
                if ok.sum() < 5:
                    continue
                vals = np.where(ok, ndi.map_coordinates(E, [ys, xs], order=1, mode="constant"), 0)
                k = int(np.argmax(vals))
                if vals[k] < EDGE_MIN:
                    continue
                SRC.append(gp)
                TGT.append(line[k])
    return np.array(SRC), np.array(TGT)


def refit_affine(SRC, TGT, anchor_src, anchor_tgt):
    src = np.vstack([SRC] + [anchor_src] * ANCHOR_WEIGHT)
    tgt = np.vstack([TGT] + [anchor_tgt] * ANCHOR_WEIGHT)
    X = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(X, tgt, rcond=None)
    A, b = sol[:2].T, sol[2]
    r = np.sqrt(((SRC @ A.T + b - TGT) ** 2).sum(1))          # trim worst 25%, refit
    keep = r < np.percentile(r, 75)
    src = np.vstack([SRC[keep]] + [anchor_src] * ANCHOR_WEIGHT)
    tgt = np.vstack([TGT[keep]] + [anchor_tgt] * ANCHOR_WEIGHT)
    X = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(X, tgt, rcond=None)
    return sol[:2].T, sol[2], np.median(r[keep])


def refine(green, anchor_det_idx=3, anchor_gds_tri=10):
    polys, fpolys = load_gds_flipped()
    dets = detect_bright(green)
    res, A, b, asrc, atgt = anchor_fit(dets[anchor_det_idx], fpolys[anchor_gds_tri])
    A0, b0 = A.copy(), b.copy()
    E = edge_map(green)
    for win in WINDOWS:
        SRC, TGT = snap_correspondences(fpolys, A, b, E, win)
        A, b, med = refit_affine(SRC, TGT, asrc, atgt)
    sv = np.linalg.svd(A, compute_uv=False)
    aniso = 100 * (sv[0] - sv[1]) / sv[1]
    return dict(A=A, b=b, A0=A0, b0=b0, polys=polys, fpolys=fpolys,
                aniso=aniso, sv=sv, med=med, anchor_res=res, atgt=atgt)


def main():
    g = np.asarray(Image.open(FRAME).convert("RGB")).astype(np.float32)[:, :, 1]
    H, W = g.shape
    r = refine(g)
    print(f"anchor resid {r['anchor_res']:.1f}px  ->  aniso {r['aniso']:.1f}% "
          f"({r['sv'][0]:.2f}/{r['sv'][1]:.2f} px/um), edge resid {r['med']:.1f}px")

    def n8(x, lo, hi):
        return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    vis = np.repeat(n8(g, np.percentile(g, 2), np.percentile(g, 98))[:, :, None], 3, axis=2).astype(np.uint8)
    img = Image.fromarray(vis)
    dr = ImageDraw.Draw(img)
    for fq in r["fpolys"]:
        p0 = fq @ r["A0"].T + r["b0"]
        dr.line([tuple(p) for p in p0] + [tuple(p0[0])], fill=(90, 90, 200), width=1)
    for fq, q in zip(r["fpolys"], r["polys"]):
        pts = fq @ r["A"].T + r["b"]
        col = {3: (255, 40, 40), 4: (60, 200, 255), 6: (90, 220, 90)}[len(q)]
        dr.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=col, width=3)
    for p in r["atgt"]:
        dr.ellipse([p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10], outline=(255, 0, 255), width=4)
    dr.rectangle([0, 0, W, 30], fill=(0, 0, 0))
    dr.text((10, 8), f"edge-snap anisotropy {r['aniso']:.1f}%  (thin blue = uniform-scale anchor start)", fill=(240, 240, 240))
    out = Path(__file__).resolve().parent.parent / "out" / f"{FRAME.stem}_aniso.png"
    img.resize((W // 2, H // 2), Image.LANCZOS).save(out)
    print(f"written {out}")


if __name__ == "__main__":
    main()
