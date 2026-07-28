#!/usr/bin/env python3
"""Standard GDS registration pipeline: SAM -> anchor -> edge-ICP refinement.

Validated on seq4045 (clean), seq4046 (messy/debris), seq4050 (huge, strained).
Every step here exists because a specific frame broke a simpler version:

  1. DETECT -- loose SAM (edge_anisotropy.sam_shapes): every mask 8k..0.70*frame,
     no elongation/brightness filter. find_crystals strips the triangle anchors.

  2. ANCHOR -- corner-match every detection to its GDS twin (x-flipped GDS,
     PROPER rotation, fixed 13.7 px/um), refine with the global ICP, and rank by
     MASK-COVERAGE (fraction of detected crystals the template lands on), NOT by
     corner/edge residual (SAM's corners are rough; a shifted or off-frame-parked
     pose games edge-coverage but leaves most crystals uncovered). Structure:
       - TIER 1: triangles only, single anchor. Reliable, fast, keeps the good
         frames (seq4045/46/50) exactly.
       - TIER 2 (only if tier-1 mask-coverage < ~30%, i.e. no clean triangle --
         seq4057's triangles are faint/SAM-invisible): ALL shapes, each seed
         expanded into a MULTI-SHAPE joint anchor (fit several visible crystals'
         corners together, like the manual tri+quad, so the pose is pinned and
         cannot over-warp). Keep tier-2 only if it covers more crystals.
       - Within a tier: coarse-rank seeds by raw-pose mask-coverage, refine the
         top few, then MASK-COVERAGE gates + edge RESIDUAL selects (kills mirror
         partners at equal coverage), strain must be plausible.
     This resolves every observed failure: one-cell shifts and off-frame parks
     (low mask-coverage), mirror partners (residual), and faint-triangle frames
     (multishape on the visible lens crystals).

  3. GLOBAL REFINE -- polarity-aware ICP of every template edge to the SAM crystal
     boundaries; refit ONE affine over all correspondences, no anchor pin. Drives
     the fit by ~1000 crystal edges, capturing the frame-wide anisotropic strain.
     SCALE and ANISOTROPY are clamped each iteration (scale near the 13.7 px/um
     hardware constant, strain <=8%): a fit must not grow the template off-frame
     or over-warp to fake coverage, which makes mask-coverage an honest signal.

  4. PER-SHAPE REFINE -- each shape fits its OWN small affine (polarity-aware
     mini-ICP) to absorb LOCAL non-affine strain a single global affine cannot
     (on seq4050 a bowtie's two bases otherwise collapse onto one crystal edge).
     HEAVILY GUARDED, because unguarded it over-fits faint/weak-edged crystals
     and flips or drifts them onto noise (seq4045). A per-shape fit is kept only
     if it (a) does not flip, (b) rotates <8 deg and scales within 0.7..1.4x of
     the global pose, (c) translates <half the shape's size, and (d) actually
     IMPROVES that shape's edge coverage. Otherwise the shape keeps the global
     affine untouched -- so an already-perfect global fit is left alone.

polarity-aware = a template edge only matches an image boundary with the crystal
on the edge's INTERIOR side (from the polygon winding). Without it, adjacent
crystals' near-parallel edges swap and bases collapse.

Output: a per-shape affine (A_i, b_i) for every GDS polygon i. The registered
outline of crystal i is  fpolys[i] @ A_i.T + b_i. The GDS supplies which crystals
exist (the yield denominator); each still needs an independent border check
before its occupancy counts. See REGISTRATION.md and memory gds-*/edge-*/sam-*.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import edge_anisotropy as ea
from auto_anchor import anchor_fit, CAND_RES_TOL

SCALE = ea.SCALE

# global ICP: coarse->fine perpendicular search half-widths (px). Starts wide
# enough to reach a base offset ~70px away (seq4050) that a 40px window missed.
GLOBAL_WINDOWS = [100, 70, 50, 35, 25, 16, 12]
# per-shape ICP windows
SHAPE_WINDOWS = [60, 40, 25, 15, 10]
EDGE_MIN = 0.20        # min normalized boundary response to accept a snap
POLARITY_PROBE = 7.0   # px along the inward normal to test crystal interior
ANCHOR_TOPK = 8        # refine this many top global-score anchors
ANISO_MAX = 8.0        # % strain above which a refined anchor is an implausible warp
COVERAGE_GATE = 0.08   # keep anchors within this of the max coverage, then pick by residual
ABS_COVERAGE_MIN = 0.15  # a correct pose puts shapes on crystals; below this the fit is
                         # almost surely wrong even if residual/margins look fine (seq4057)
TIER2_TRIGGER = 0.30     # if the triangle tier covers fewer crystals than this, also run
                         # the all-shape multishape tier and keep whichever covers more
SHAPE_SCALE_GUARD = (0.7, 1.4)  # per-shape singular values must stay in this band
MCOV_TIE_GATE = 0.03     # only candidates within this of the top mask-coverage are a
                         # true tie (mirror partners share mcov exactly); residual
                         # disambiguates ONLY these. A candidate covering meaningfully
                         # fewer crystals must not win on a tighter residual alone.
MULTI_TILE_ENABLED = True   # re-enabled after ground-truth calibration on seq3889:
                            # neighbouring fields are 180-ROTATED copies, which the
                            # search now covers. Extra-copy crystals remain tagged
                            # needs_review in the database.
MULTI_TILE_MAX = 3       # fit at most this many copies of the template per frame
TILE_MIN_RESIDUAL = 2    # need at least this many big uncovered crystals to try another copy
TILE_ROT_TOL = 10.0      # deg: an extra copy must agree with tile 1's rotation (same wafer)
TILE_MIN_OFFSET = 1200.0  # px: ...but be placed genuinely elsewhere, not re-find tile 1
                          # (a near-identical placement ~800px off re-claims tile-1's own
                          # residual junk -- seen on seq4045)
TILE_CENTER_TOL = 220.0  # px: a residual crystal whose centroid sits this close to a
                         # placed site is ON that site (growth is centre-outward, so a
                         # partially grown crystal still centres on its site)
TILE_CENTER_MIN = 3      # accept a copy only if >= this many residual crystals centre.
                         # NO fraction-of-residual requirement: junk masks inside the
                         # copy's footprint (4152: six of nine, all >400px from any site)
                         # inflated it while every user-rejected tile passed it -- the
                         # absolute gates (match dist, edge verification) do the work.
TILE_MATCH_DIST = 100.0  # px: the matched crystals must sit this close on average.
                         # Calibrated on user verdicts: confirmed copies polish to
                         # 14-45px; every rejected one sat at 78-134px.
TILE_MAX_AREA = 1_200_000  # px^2: residual masks above this are background/merged junk,
                           # not crystals (largest designed shape is ~750k px^2)
TILE_OVERLAP_MAX = 0.30  # bounding-box overlap allowed between copies. The 180-rotated
                         # neighbour NESTS into the original's concave gaps (the 'compact'
                         # tiling): measured 0.19 bbox overlap on seq3889's ground truth,
                         # with the actual shapes never colliding.
TILE_STEP_MAX = 1.6      # a neighbour is at most this many cell extents away
TILE_VIS_AREA_MIN = 25_000  # px^2: a site's frame-visible part must be at least this big
                            # for its (clipped) centroid to take part in matching
TILE_NALIGN_MIN = 2      # accept a copy only if >= this many of its shapes' edges hug a
                         # residual crystal at >=50% per-shape coverage. Calibrated on the
                         # user's verdicts: correct copies score 2-4+, every wrong one 0-1
                         # (centroid metrics overlap; edges are the honest separator).
SPLIT_MIN_AREA = 50_000  # only masks this big are checked for merged crystals
SPLIT_SEAM_DROP = 4.0    # the seam must be at least this much darker than the crystals'
                         # own 25th-percentile level (measured: seq4139's gap is ~6-8
                         # grey levels below the two diamonds it separates)
SPLIT_COVER_MIN = 0.70   # the bright parts must jointly cover this fraction of the mask,
                         # else the "split" is eating real crystal, not removing a seam
CONST_MIN_AREA = 40_000  # constellation tier: only masks this big vote (small masks are
                         # debris/partials whose centroids would pollute the vote)
CONST_SIZE_TOL = 0.35    # pair a det with a GDS shape only if lengths agree within this
CONST_INLIER_TOL = 120.0  # px: a det centroid this close to a placed GDS centroid is an inlier
CONST_TOPK = 10          # refine this many top-voted constellation poses
TRI_FILL_MIN = 0.80      # a mask that fills its min-enclosing triangle this well is
                         # (possibly truncated) triangular -> reconstruct the full apex
COV_CONF_MIN = 0.18      # edge-coverage floor. A correct pose's template edges LIE ON
                         # crystal boundaries: the 7 validated frames all sit >=0.22,
                         # every wrong pose (junk-blob anchor, one-cell shift) <=0.14.
                         # Used both to pick among candidates and to flag confidence.


def reconstruct_triangles(masks, dets):
    """Recover the true triangle for each mask that is a TRUNCATED sharp-tip triangle.

    The design is 'SharpTipCompactTiling': its triangles are long sharp slivers. A
    crystal whose tip didn't fully grow (or that SAM clipped) is segmented as a
    trapezoid -- fitting that 4-gon as a 4-gon is meaningless, and the frame ends up
    anchored on some junk blob instead. cv2.minEnclosingTriangle extends the two
    converging legs back to their apex, reconstructing the sliver. Returned as
    (mask_index, triangle) so the reused index keeps masks/dedup/reporting intact.

    Only non-triangle masks are reconstructed (a native 3-gon is already a triangle,
    and reconstructing it would just collide on the same (det,gds) pair). A mask that
    is genuinely triangular fills its enclosing triangle well; a fat quad or a lens
    does not (or, if it squeaks past the fill gate, produces a large anchor residual
    and is dropped in build)."""
    out = []
    for di, m in enumerate(masks):
        if len(dets[di]) == 3:
            continue
        m = np.asarray(m, bool).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area, tri = cv2.minEnclosingTriangle(c.astype(np.float32))
        if tri is None or area <= 0:
            continue
        if cv2.contourArea(c) / area >= TRI_FILL_MIN:
            out.append((di, tri.reshape(-1, 2).astype(float)))
    return out


def _seam_split(m, green_sm):
    """Split one mask at internal dark seams; None if it isn't a merged pair.

    Two crystals growing toward each other leave a gap a few grey levels darker
    than either crystal. SAM sometimes bridges it into one mask (seq4139: two
    in-spec diamonds, 29um and 21um wide, returned as one 58um-wide mask, which
    then matches NO designed shape and poisons every anchor tier). Recover the
    parts: threshold inside the mask a hair below the crystals' own level, keep
    the large bright components, and assign the seam pixels to the nearest part."""
    if m.sum() < SPLIT_MIN_AREA:
        return None
    v = green_sm[m]
    bright = m & (green_sm > np.percentile(v, 25) - SPLIT_SEAM_DROP)
    bright = ndi.binary_opening(bright, iterations=3)
    lab, n = ndi.label(bright)
    if n < 2:
        return None
    sizes = ndi.sum(bright, lab, range(1, n + 1))
    big = [i + 1 for i in range(n) if sizes[i] >= ea.SAM_MIN_AREA]
    if len(big) < 2 or sum(sizes[i - 1] for i in big) < SPLIT_COVER_MIN * m.sum():
        return None
    keep = np.where(np.isin(lab, big), lab, 0)
    _, (iy, ix) = ndi.distance_transform_edt(keep == 0, return_indices=True)
    filled = keep[iy, ix]
    return [(filled == b) & m for b in big]


def split_merged_masks(masks, green):
    """Rebuild (masks, dets) with merged masks split at dark seams. A split is kept
    only if every part yields a corner polygon (the downstream lists are parallel).
    Returns (masks, dets, changed)."""
    green_sm = ndi.uniform_filter(green, 9)
    out_m, out_d, changed = [], [], False
    for m in masks:
        m = np.asarray(m, bool)
        parts = _seam_split(m, green_sm)
        polys = None if parts is None else [ea._mask_to_poly(p.astype(np.uint8)) for p in parts]
        if polys is not None and all(p is not None for p in polys):
            out_m.extend(parts)
            out_d.extend(polys)
            changed = True
        else:
            ap = ea._mask_to_poly(m.astype(np.uint8))
            if ap is not None:
                out_m.append(m)
                out_d.append(ap)
    return out_m, out_d, changed


def _axis_length_angle(pts):
    """Principal-axis extent and orientation (deg) of a point set."""
    c = pts.mean(0)
    Q = pts - c
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    u = Vt[0]
    proj = Q @ u
    return c, float(proj.max() - proj.min()), float(np.degrees(np.arctan2(u[1], u[0])))


def constellation_seeds(masks, fpolys):
    """Rescue anchoring from mask CENTROIDS + long-axis orientations -- no corners.

    On diamond/lens-dominated frames, _mask_to_poly's corners are unreliable (blunt
    tips, merged side vertices), so every corner-based anchor fit fails even though
    SAM's masks are excellent (seq4139..4151). And a single diamond cannot reveal its
    own identity anyway (four rhombi + eight lenses look alike locally; a rhombus is
    also 180-degree symmetric). Centroids and axis directions survive all of that:
    hypothesise (det crystal <-> GDS shape, axis-aligned rotation or its 180 flip),
    which fixes the pose at the known 13.7 px/um scale, then let the WHOLE tiling
    vote -- count how many other detected centroids land on a placed GDS centroid.
    The correct identity aligns the constellation; a wrong one leaves it misplaced.
    Returns the top-voted poses as (score, di, gi, A, b), best first."""
    dm = []
    for di, m in enumerate(masks):
        m = np.asarray(m, bool)
        if m.sum() < CONST_MIN_AREA:
            continue
        ys, xs = np.nonzero(m)
        c, L, ang = _axis_length_angle(np.stack([xs, ys], 1).astype(float))
        dm.append((di, c, L, ang))
    gm = []
    for gi, fq in enumerate(fpolys):
        c, L, ang = _axis_length_angle(np.asarray(fq, float))
        gm.append((gi, c, L * SCALE, ang))
    cands = []
    for di, cd, Ld, ad in dm:
        for gi, cg, Lg, ag in gm:
            if abs(Ld - Lg) / max(Lg, 1) > CONST_SIZE_TOL:
                continue
            for extra in (0.0, 180.0):        # an axis has no sign -> two rotations
                th = np.radians(ad - ag + extra)
                A = SCALE * np.array([[np.cos(th), -np.sin(th)],
                                      [np.sin(th),  np.cos(th)]])
                b = cd - A @ cg
                placed = np.array([A @ c2 + b for _, c2, _, _ in gm])
                score = sum(np.linalg.norm(placed - cd2, axis=1).min() < CONST_INLIER_TOL
                            for _, cd2, _, _ in dm)
                cands.append((score, di, gi, A, b))
    cands.sort(key=lambda t: -t[0])
    return cands


def boundary_maps(masks, shape):
    """From SAM masks build (Bsoft, Msm): a soft boundary-ridge map to snap to,
    and a smoothed crystal-interior map for the polarity test."""
    H, W = shape
    B = np.zeros((H, W), np.float32)
    M = np.zeros((H, W), np.float32)
    for m in masks:
        m = np.asarray(m, bool)
        B[m ^ ndi.binary_erosion(m, iterations=2)] = 1.0
        M[m] = 1.0
    Bsoft = ndi.gaussian_filter(B, 3.0)
    Bsoft /= (Bsoft.max() + 1e-6)
    Msm = ndi.gaussian_filter(M, 2.0)
    return Bsoft, Msm


def _inward_normal(iA, iB, cimg):
    seg = iB - iA
    L = np.hypot(*seg)
    if L < 1e-6:
        return None, None, 0.0
    u = seg / L
    n = np.array([-u[1], u[0]])
    mid = 0.5 * (iA + iB)
    return u, (n if np.dot(cimg - mid, n) > 0 else -n), L


def _snap_edge(ip, n, n_in, win, Bsoft, Msm, polarity):
    """Best polarity-correct boundary point along the perpendicular through ip."""
    H, W = Bsoft.shape
    offs = np.arange(-win, win + 1, 1.0)
    line = ip[None, :] + offs[:, None] * n[None, :]
    xs, ys = line[:, 0], line[:, 1]
    ok = (xs > 2) & (xs < W - 3) & (ys > 2) & (ys < H - 3)
    if ok.sum() < 5:
        return None
    vals = np.where(ok, ndi.map_coordinates(Bsoft, [ys, xs], order=1, mode="constant"), 0)
    for k in np.argsort(-vals)[:10]:
        if vals[k] < EDGE_MIN:
            break
        p = line[k]
        if polarity:
            ins = ndi.map_coordinates(Msm, [[p[1] + POLARITY_PROBE * n_in[1]],
                                            [p[0] + POLARITY_PROBE * n_in[0]]], order=1)[0]
            out = ndi.map_coordinates(Msm, [[p[1] - POLARITY_PROBE * n_in[1]],
                                            [p[0] - POLARITY_PROBE * n_in[0]]], order=1)[0]
            if not (ins > 0.5 and out < 0.5):
                continue
        return p
    return None


def _edge_samples(fq, A, b, step=16, min_len=18):
    """Yield (gds_pt, img_pt, normal, inward_normal) along a shape's edges."""
    cimg = fq.mean(0) @ A.T + b
    for i in range(len(fq)):
        gA, gB = fq[i], fq[(i + 1) % len(fq)]
        iA, iB = gA @ A.T + b, gB @ A.T + b
        u, n_in, L = _inward_normal(iA, iB, cimg)
        if u is None or L < min_len:
            continue
        n = np.array([-u[1], u[0]])
        for tt in np.linspace(0.08, 0.92, max(2, int(L // step))):
            yield gA + tt * (gB - gA), iA + tt * (iB - iA), n, n_in


def _fit_affine(SRC, TGT, trim=0.80):
    X = np.hstack([SRC, np.ones((len(SRC), 1))])
    sol, *_ = np.linalg.lstsq(X, TGT, rcond=None)
    A, b = sol[:2].T, sol[2]
    res = np.sqrt(((SRC @ A.T + b - TGT) ** 2).sum(1))
    keep = res < np.percentile(res, 100 * trim)
    X = np.hstack([SRC[keep], np.ones((keep.sum(), 1))])
    sol, *_ = np.linalg.lstsq(X, TGT[keep], rcond=None)
    A, b = sol[:2].T, sol[2]
    return A, b, float(np.median(res[keep]))


def global_edge_score(fpolys, A, b, Bsoft, win=18):
    """Mean boundary-ridge response under all in-frame template edges."""
    H, W = Bsoft.shape
    tot, cnt = 0.0, 0
    for fq in fpolys:
        P = fq @ A.T + b
        if P[:, 0].max() < 0 or P[:, 0].min() > W or P[:, 1].max() < 0 or P[:, 1].min() > H:
            continue
        for _, ip, n, _ in _edge_samples(fq, A, b):
            offs = np.arange(-win, win + 1, 2.0)
            line = ip[None, :] + offs[:, None] * n[None, :]
            xs, ys = line[:, 0], line[:, 1]
            ok = (xs > 1) & (xs < W - 2) & (ys > 1) & (ys < H - 2)
            if ok.sum() < 4:
                continue
            vals = np.where(ok, ndi.map_coordinates(Bsoft, [ys, xs], order=1, mode="constant"), 0)
            tot += vals.max()
            cnt += 1
    return tot / max(cnt, 1)


def coverage(fpolys, A, b, Bsoft, Msm, tol=8.0):
    """Fraction of in-frame template edge samples that find a polarity-correct
    crystal boundary within tol px. The correct pose lands shapes ON crystals
    (high coverage); a one-cell-shifted pose drops shapes onto gaps (low), even
    when its edge residual looks fine -- this is what residual alone can't tell."""
    H, W = Bsoft.shape
    hit = tot = 0
    for fq in fpolys:
        P = fq @ A.T + b
        if P[:, 0].max() < 0 or P[:, 0].min() > W or P[:, 1].max() < 0 or P[:, 1].min() > H:
            continue
        for _, ip, n, n_in in _edge_samples(fq, A, b):
            tot += 1
            if _snap_edge(ip, n, n_in, tol, Bsoft, Msm, polarity=True) is not None:
                hit += 1
    return hit / max(tot, 1)


def _clamp_aniso(A, target=SCALE, scale_tol=0.12, max_pct=ANISO_MAX):
    """Clamp an affine's SCALE (near the 13.7 px/um hardware constant) AND its
    anisotropy (<= max_pct). The overall scale is fixed by the objective+camera,
    so a fit must not grow/shrink the template -- otherwise it games coverage by
    scaling most shapes off-frame and aligning the few that remain (seq4057). And
    the real tape strain is a few %, so larger anisotropy is over-warp. Together
    these make coverage honest: a wrong anchor can buy neither scale nor strain."""
    U, s, Vt = np.linalg.svd(A)             # s[0] >= s[1]
    g = np.clip(np.sqrt(s[0] * s[1]), target * (1 - scale_tol), target * (1 + scale_tol))
    ratio = min(s[0] / max(s[1], 1e-9), 1 + max_pct / 100)
    sr = np.sqrt(ratio)
    return U @ np.diag([g * sr, g / sr]) @ Vt


def mask_boundaries(masks):
    """Per-mask boundary pixel coordinates (ys, xs). Pose-INDEPENDENT, so compute
    once per frame: the anchor stage scores hundreds of candidate poses, and
    re-deriving these full-frame erosions inside every mask_coverage call was 88%
    of registration runtime (132s of 184s on seq4061)."""
    bnds = []
    for m in masks:
        m = np.asarray(m, bool)
        bnd = m ^ ndi.binary_erosion(m, iterations=2)
        ys, xs = np.nonzero(bnd)
        if len(ys) >= 10:
            bnds.append((ys, xs))
    return bnds


def mask_coverage(fpolys, A, b, bnds, shape, tol=6, min_frac=0.25):
    """Fraction of SAM crystal masks that have a template edge running along their
    boundary. Unlike edge-coverage (fraction of in-frame template edges), this
    cannot be gamed by a shifted pose that parks a few shapes on the frame edge:
    such a pose leaves most detected crystals uncovered. The correct pose lands a
    template shape on (nearly) every crystal -> high mask-coverage.

    bnds is the precomputed mask_boundaries(masks). The dilation kernel is an L1
    ball of radius tol -- identical to ndi.binary_dilation(iterations=tol) with the
    default cross structure, but one cv2 pass instead of tol full-frame sweeps."""
    H, W = shape
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    for fq in fpolys:
        P = fq @ A.T + b
        dr.line([tuple(p) for p in P] + [tuple(P[0])], fill=255, width=1)
    ax = np.abs(np.arange(-tol, tol + 1))
    kern = ((ax[:, None] + ax[None, :]) <= tol).astype(np.uint8)
    tmpl = cv2.dilate((np.array(im) > 0).astype(np.uint8), kern) > 0
    covered = 0
    for ys, xs in bnds:
        if tmpl[ys, xs].sum() / len(ys) > min_frac:
            covered += 1
    return covered / max(len(bnds), 1)


def icp_global(fpolys, A, b, Bsoft, Msm):
    """Polarity-aware ICP of all template edges to the SAM boundaries -> one affine.
    Anisotropy is clamped each iteration so the fit cannot over-warp."""
    med = np.inf
    for win in GLOBAL_WINDOWS:
        SRC, TGT = [], []
        for fq in fpolys:
            for gp, ip, n, n_in in _edge_samples(fq, A, b):
                p = _snap_edge(ip, n, n_in, win, Bsoft, Msm, polarity=True)
                if p is not None:
                    SRC.append(gp)
                    TGT.append(p)
        if len(SRC) < 6:
            break
        A, b, med = _fit_affine(np.array(SRC), np.array(TGT))
        A = _clamp_aniso(A)
    return A, b, med


def _shape_coverage(fq, A, b, Bsoft, Msm, tol=6.0):
    hit = tot = 0
    for _, ip, n, n_in in _edge_samples(fq, A, b, step=10, min_len=12):
        tot += 1
        if _snap_edge(ip, n, n_in, tol, Bsoft, Msm, polarity=True) is not None:
            hit += 1
    return hit / max(tot, 1), tot


def icp_shape(fq, A0, b0, Bsoft, Msm):
    """Fit one shape\'s own small affine to its polarity-correct edges -- GUARDED.

    Per-shape refinement fixes local strain (seq4050 bowtie) but over-fits where
    a shape\'s edges are faint/ambiguous (seq4045): unguarded it flips or drifts a
    shape onto noise. So each per-shape fit must (a) not flip or rotate/scale far
    from the global pose, (b) not translate more than ~half the shape\'s size, and
    (c) actually IMPROVE that shape\'s edge coverage -- else the shape keeps the
    global affine untouched. A perfect global fit is thus left alone."""
    base_cov, ntot = _shape_coverage(fq, A0, b0, Bsoft, Msm)
    if ntot < 4:
        return A0, b0
    A0inv = np.linalg.inv(A0)
    sv0 = np.linalg.svd(A0, compute_uv=False)
    extent = np.ptp(fq @ A0.T + b0, axis=0).max()
    A, b = A0.copy(), b0.copy()
    for win in SHAPE_WINDOWS:
        SRC, TGT = [], []
        for gp, ip, n, n_in in _edge_samples(fq, A, b, step=10, min_len=12):
            p = _snap_edge(ip, n, n_in, win, Bsoft, Msm, polarity=True)
            if p is not None:
                SRC.append(gp)
                TGT.append(p)
        if len(SRC) < 8:
            break
        # NO trimming (trim=1.0): a genuinely off edge (seq4050 bowtie base, ~48px
        # from its true crystal edge) is the LARGEST residual, so trimming would
        # discard exactly the correspondences that must move the shape, leaving the
        # base collapsed on its neighbour. Polarity rejects bad snaps; the
        # coverage-improve accept test guards against noise.
        An, bn, _ = _fit_affine(np.array(SRC), np.array(TGT), trim=1.0)
        rel = An @ A0inv
        ang = abs(np.arctan2(rel[1, 0] - rel[0, 1], rel[0, 0] + rel[1, 1]))
        sv = np.linalg.svd(An, compute_uv=False)
        cshift = np.linalg.norm((fq.mean(0) @ An.T + bn) - (fq.mean(0) @ A0.T + b0))
        if (np.sign(np.linalg.det(An)) != np.sign(np.linalg.det(A0))
                or ang > np.radians(8)
                or sv.max() / sv0.max() > SHAPE_SCALE_GUARD[1]
                or sv.min() / sv0.min() < SHAPE_SCALE_GUARD[0]
                or cshift > 0.5 * extent):
            break
        A, b = An, bn
    new_cov, _ = _shape_coverage(fq, A, b, Bsoft, Msm)
    if new_cov < base_cov + 0.05:
        return A0, b0
    return A, b


def multishape_fit(dets, fpolys, A0, b0, match_radius=70.0, min_shapes=2):
    """Given a rough seed pose, gather EVERY detection that agrees on a consistent
    GDS placement and fit ONE affine jointly over all their corners.

    A single anchor lets the global fit over-warp to fake coverage (seq4057: a lone
    lens hexagon warps to 30.6% strain). Multiple visible crystals pinned together
    rigidly constrain the pose -- exactly what the manual tri+quad 7-point fit did.
    Each detection is assigned to the nearest same-vertex GDS shape under the seed
    pose; corners are paired by assignment within match_radius; shapes contributing
    too few in-radius corners are dropped. Returns (A, b, n_shapes) or None."""
    from scipy.optimize import linear_sum_assignment
    placed = [(gi, fpolys[gi] @ A0.T + b0) for gi in range(len(fpolys))]
    SRC, TGT, nsh = [], [], 0
    for dp in dets:
        dp = np.asarray(dp, float)
        same = [(np.linalg.norm(dp.mean(0) - P.mean(0)), gi, P)
                for gi, P in placed if len(fpolys[gi]) == len(dp)]
        if not same:
            continue
        dist, gi, P = min(same, key=lambda t: t[0])
        Dm = np.linalg.norm(P[:, None] - dp[None], axis=2)
        ri, ci = linear_sum_assignment(Dm)
        pairs = [(r, c) for r, c in zip(ri, ci) if Dm[r, c] < match_radius]
        if len(pairs) < max(3, len(dp) - 1):     # most corners must land in-radius
            continue
        for r, c in pairs:
            SRC.append(fpolys[gi][r])
            TGT.append(dp[c])
        nsh += 1
    if nsh < min_shapes or len(SRC) < 6:
        return None
    A, b, _ = _fit_affine(np.array(SRC), np.array(TGT), trim=0.9)
    return A, b, nsh


def register(green, cache_key=None, triangles_only=True, _split=None, rot_prior=None):
    """Full pipeline. Returns dict with per-shape affines and diagnostics.

    _split is internal: the split-retry passes its seam-split (masks, dets) here to
    rerun the cascade on them (and blocks a second retry). rot_prior=(deg, tol):
    only consider candidate poses within tol of that rotation -- used by the
    multi-tile pass, where tile 1 already fixed the wafer orientation."""
    polys, fpolys = ea.load_gds_flipped()
    if _split is not None:
        masks, dets = _split
    else:
        masks, dets = ea.sam_shapes(green, cache_key=cache_key)
    Bsoft, Msm = boundary_maps(masks, green.shape)
    bnds = mask_boundaries(masks)          # pose-independent; hundreds of poses score against these

    # 2. anchor candidates. Prefer TRIANGLES: every confirmed anchor across frames
    # is a triangle (the design's distinctive, asymmetric-enough shape), whereas
    # quad/hex anchors on these crystals yield tight-residual-but-shifted poses
    # that hijack the ranking. Fall back to all shapes only if no triangle is
    # detected. Coarse-rank by global edge score to pick which to refine.
    def build(cand_list, nsides_ok, res_tol=None):
        out = []
        for di, dp in cand_list:
            if not nsides_ok(len(dp)):
                continue
            for gi, gv in enumerate(fpolys):
                if len(gv) != len(dp):
                    continue
                res, A, b, asrc, atgt = anchor_fit(dp, gv)
                # reconstructed-triangle candidates are res-filtered: a lens or fat
                # quad forced into a triangle mismatches its corners (large res) and
                # is dropped, so only genuine truncated triangles survive. Native
                # detections are NOT res-filtered (SAM corners are rough; the pose,
                # not the corners, is judged downstream).
                if res_tol is not None and res > res_tol:
                    continue
                # coarse score = mask-coverage of the RAW anchor pose. This is what
                # the final selection cares about (does the template land on the
                # detected crystals), so it surfaces the right candidate into the
                # refine set -- global_edge_score did not (it missed seq4057's
                # correct anchor, which sat outside the top-K).
                out.append(dict(res=res, A=A, b=b, di=di, gi=gi,
                                gs=mask_coverage(fpolys, A, b, bnds, green.shape)))
        return out

    def rot_ok(A):
        if rot_prior is None:
            return True
        th = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
        return abs((th - rot_prior[0] + 180) % 360 - 180) <= rot_prior[1]

    def evaluate(seeds, use_multishape):
        """Refine the top-K seeds (optionally via a multi-shape joint anchor) and
        return them scored with cov/aniso/med."""
        seeds = sorted(seeds, key=lambda c: -c["gs"])
        out, seen = [], set()
        for c in seeds:
            if not rot_ok(c["A"]):       # wrong wafer orientation (multi-tile prior)
                continue
            if (c["di"], c["gi"]) in seen:
                continue
            seen.add((c["di"], c["gi"]))
            As, bs, meds = icp_global(fpolys, c["A"], c["b"], Bsoft, Msm)
            cov, A, b, med, nsh = coverage(fpolys, As, bs, Bsoft, Msm), As, bs, meds, 1
            if use_multishape:
                ms = multishape_fit(dets, fpolys, c["A"], c["b"])
                if ms:
                    Am, bm, medm = icp_global(fpolys, ms[0], ms[1], Bsoft, Msm)
                    covm = coverage(fpolys, Am, bm, Bsoft, Msm)
                    if covm > cov:
                        cov, A, b, med, nsh = covm, Am, bm, medm, ms[2]
            sv = np.linalg.svd(A, compute_uv=False)
            c.update(A=A, b=b, med=med, cov=cov, nshapes=nsh,
                     mcov=mask_coverage(fpolys, A, b, bnds, green.shape),
                     aniso=100 * (sv[0] - sv[1]) / sv[1])
            out.append(c)
            if len(out) >= ANCHOR_TOPK:
                break
        return out

    def select(top):
        # MASK-COVERAGE gates (how many detected crystals the template lands on --
        # robust to shifted poses that game edge-coverage), residual SELECTS, strain
        # must be plausible. A one-cell shift (seq4050) or an off-frame-parked pose
        # (seq4057) covers few masks; a mirror partner (seq4046 GDS#33 vs #34)
        # covers the same but fits worse. Gate near max mask-coverage, pick tightest.
        plausible = [c for c in top if c["aniso"] <= ANISO_MAX] or top
        # EDGE-COVERAGE FLOOR: a correct pose lays its template edges ON crystal
        # boundaries (high edge-coverage). A wrong pose -- anchored on a junk blob or
        # shifted a cell -- can still SPRAY shapes across the frame and graze enough
        # mask boundaries to score decent mask-coverage, yet almost none of its own
        # edges lie on a crystal (low edge-coverage; seq4079 mcov 0.31 but cov 0.09).
        # So restrict to poses that clear the floor; fall back only if none do (a
        # genuinely bad frame), which then gets flagged low-confidence below.
        on_crystal = [c for c in plausible if c["cov"] >= COV_CONF_MIN]
        pool = on_crystal or plausible
        max_mcov = max(c["mcov"] for c in pool)
        # MASK-COVERAGE decides; edge RESIDUAL only breaks TRUE ties. The top-coverage
        # pose lands template shapes on the most real crystals. Residual then picks
        # among candidates that cover essentially the same crystals -- notably the
        # mirror partners (GDS#33/#34), which share mcov exactly and differ only in
        # fit quality. A candidate covering fewer crystals (outside MCOV_TIE_GATE) is
        # NOT a tie and cannot win on a tighter-but-sparser residual (seq4046, 4079).
        tie = [c for c in pool if c["mcov"] >= max_mcov - MCOV_TIE_GATE]
        ranked = sorted(tie, key=lambda c: c["med"])
        best = ranked[0]
        gated = [c for c in pool if c["mcov"] >= max_mcov - COVERAGE_GATE]
        mcov2 = sorted((c["mcov"] for c in pool), reverse=True)
        mcov_margin = (mcov2[0] - mcov2[1]) if len(mcov2) > 1 else 1.0
        if len(ranked) > 1:
            res_margin = (ranked[1]["med"] - best["med"]) / max(ranked[1]["med"], 1e-6)
            low_conf = res_margin < 0.15
        else:
            res_margin, low_conf = 1.0, False
        conf = dict(mcov=mcov_margin, res=res_margin, n_gated=len(gated),
                    abs_mcov=best["mcov"], cov=best["cov"])
        if best["mcov"] < ABS_COVERAGE_MIN:     # too few crystals covered -> wrong pose
            low_conf = True
        if best["cov"] < COV_CONF_MIN:          # template doesn't lie on crystals -> wrong pose
            low_conf = True
        return best, conf, low_conf

    # TIERED anchoring. Tier 1: triangles only, single-anchor -- reliable and keeps
    # the good frames (seq4045/46/50) exactly. Tier 2 (only if the best triangle
    # pose covers too little, i.e. no clean triangle exists -- seq4057, whose
    # triangles are faint/SAM-invisible): all shapes, expanded into a multi-shape
    # joint anchor so the visible lens crystals pin the pose without over-warp.
    # native SAM detections, plus triangles reconstructed from truncated sharp-tip
    # crystals (edge_anisotropy's corner approximation collapses a thin lens into a
    # fake triangle and clips a real triangle's apex, so the good anchor is often
    # missing until reconstructed -- seq4079).
    base = list(enumerate(dets))
    recon = reconstruct_triangles(masks, dets)
    tri_seeds = (build(base, lambda n: n == 3)
                 + build(recon, lambda n: n == 3, res_tol=CAND_RES_TOL))
    best = conf = low_conf = None
    if tri_seeds:
        best, conf, low_conf = select(evaluate(tri_seeds, use_multishape=False))
    # Run the all-shape multishape tier too if the triangle tier covers too few
    # crystals (seq4057: only a small faint triangle detects, tier-1 -> 15%; the
    # visible lens crystals via multishape reach ~62%). Keep whichever covers more.
    if best is None or best["mcov"] < TIER2_TRIGGER:
        all_top = evaluate(build(base, lambda n: True)
                           + build(recon, lambda n: n == 3, res_tol=CAND_RES_TOL),
                           use_multishape=True)
        if all_top:
            b2, c2, l2 = select(all_top)
            if best is None or b2["mcov"] > best["mcov"]:
                best, conf, low_conf = b2, c2, l2
    # TIER 3 (constellation rescue) -- ONLY when tiers 1-2 end low-confidence, so no
    # currently-successful frame can change. On diamond/lens frames the corner-based
    # anchors all fail (bad corners + identity ambiguity); centroid+axis constellation
    # voting recovers the pose from the tiling's global structure instead.
    if best is None or low_conf:
        seeds3 = [dict(res=0.0, A=A3, b=b3, di=di, gi=gi,
                       gs=mask_coverage(fpolys, A3, b3, bnds, green.shape))
                  for _, di, gi, A3, b3 in constellation_seeds(masks, fpolys)[:CONST_TOPK]]
        t3 = evaluate(seeds3, use_multishape=True) if seeds3 else []
        if t3:
            b3, c3, l3 = select(t3)
            # adopt only on improvement: it turns the frame confident, or it stays
            # flagged but covers more crystals (still tagged needs_review downstream).
            if best is None or (not l3) or (low_conf and b3["mcov"] > best["mcov"]):
                best, conf, low_conf = b3, c3, l3
    # SPLIT RETRY -- ONLY when the whole cascade ends low-confidence, so confident
    # frames cannot change. SAM sometimes merges adjacent crystals across their dark
    # gap into one mask (seq4139), which poisons every tier: the merged corners match
    # no GDS shape and the merged centroid sits between two template sites. Split the
    # merged masks at their seams and rerun the full cascade on the clean set.
    if (best is None or low_conf) and _split is None:
        sm, sd, split_changed = split_merged_masks(masks, green)
        if split_changed:
            alt = register(green, cache_key=None, triangles_only=triangles_only,
                           _split=(sm, sd))
            if best is None or (not alt["low_confidence"]) \
                    or alt["mask_coverage"] > best["mcov"]:
                return alt
    if best is None:
        raise RuntimeError("no anchor candidates (SAM found no matchable shapes)")

    # 4. per-shape refine for local strain
    shape_AB = [icp_shape(fq, best["A"], best["b"], Bsoft, Msm) for fq in fpolys]
    sv = np.linalg.svd(best["A"], compute_uv=False)
    return dict(polys=polys, fpolys=fpolys, masks=masks,
                global_A=best["A"], global_b=best["b"], shape_AB=shape_AB,
                anchor=(best["di"], best["gi"]), global_med=best["med"],
                coverage=best["cov"], mask_coverage=best["mcov"],
                aniso=100 * (sv[0] - sv[1]) / sv[1], conf=conf,
                low_confidence=low_conf,
                Bsoft=Bsoft, Msm=Msm)


def registered_outlines(reg):
    """Per-crystal registered outlines in image px (one polygon per GDS shape)."""
    return [fq @ A.T + b for fq, (A, b) in zip(reg["fpolys"], reg["shape_AB"])]


def covered_mask_indices(shape_AB, fpolys, masks, shape, tol=6, min_frac=0.25):
    """Indices of SAM masks that a placed template lies along (same test as
    mask_coverage, but returning WHICH masks instead of the fraction)."""
    H, W = shape
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    for fq, (A, b) in zip(fpolys, shape_AB):
        P = fq @ np.asarray(A).T + np.asarray(b)
        dr.line([tuple(p) for p in P] + [tuple(P[0])], fill=255, width=1)
    ax = np.abs(np.arange(-tol, tol + 1))
    kern = ((ax[:, None] + ax[None, :]) <= tol).astype(np.uint8)
    tmpl = cv2.dilate((np.array(im) > 0).astype(np.uint8), kern) > 0
    out = set()
    for i, m in enumerate(masks):
        m = np.asarray(m, bool)
        bnd = m ^ ndi.binary_erosion(m, iterations=2)
        if bnd.sum() < 10:
            continue
        ys, xs = np.nonzero(bnd)
        if tmpl[ys, xs].sum() / len(ys) > min_frac:
            out.add(i)
    return out


def sam_backed_sites(green_shape, fpolys, shape_AB, masks):
    """Sites with a size-compatible SAM crystal centred on them (clip-aware).
    Same evidence rule as the multi-tile fitter's matching, applied to any placed
    template: a coherent SAM segment centred on a site, with dimensions the site
    could produce, marks that site as genuinely occupied -- downstream, dimness
    alone must not reject it (seq3890's bottom-right triangle: SAM mask dead-on
    at 6px, but green-excess +2.4 sits under the cross-frame floor)."""
    H, W = green_shape
    rc, rdim = [], []
    for m in masks:
        m = np.asarray(m, bool)
        if not (CONST_MIN_AREA <= m.sum() <= TILE_MAX_AREA):
            continue
        ys, xs = np.nonzero(m)
        c = np.array([xs.mean(), ys.mean()])
        Q = np.stack([xs - c[0], ys - c[1]], 1).astype(float)
        _, _, Vt = np.linalg.svd(Q / np.sqrt(len(Q)), full_matrices=False)
        rc.append(c)
        rdim.append([np.ptp(Q @ Vt[0]), np.ptp(Q @ Vt[1])])
    if not rc:
        return []
    out = []
    for gi, (fq, (A, b)) in enumerate(zip(fpolys, shape_AB)):
        fq = np.asarray(fq, float)
        P = fq @ np.asarray(A).T + np.asarray(b)
        c, a = _clip_centroid(P, W, H)
        if c is None or a < TILE_VIS_AREA_MIN:
            continue
        Q = fq - fq.mean(0)
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        L, Wd = np.ptp(Q @ Vt[0]) * SCALE, np.ptp(Q @ Vt[1]) * SCALE
        for c2, (l2, w2) in zip(rc, rdim):
            if np.linalg.norm(np.asarray(c2) - c) >= TILE_CENTER_TOL:
                continue
            Lr, Wr = max(l2, w2), min(l2, w2)
            if 0.5 * Wd <= Wr <= 1.7 * Wd and 0.30 * L <= Lr <= 1.20 * L:
                out.append(gi)
                break
    return out


def register_tiles(green, cache_key=None):
    """Multi-tile registration: the design cell repeats across the wafer, and one
    frame can show crystals from more than one copy (a frame is smaller than a
    cell, so it may straddle two). A rigid lattice shift of tile 1 does NOT land
    on the neighbour (the tape strain fans non-uniformly -- measured on seq4061),
    so each copy needs its own registration: fit tile 1 as usual, mask out the
    crystals it covers, and rerun the whole anchor cascade on the residual. An
    extra copy is accepted only if it is CONFIDENT on its own crystals, agrees
    with tile 1's rotation (same wafer), and sits genuinely elsewhere.

    Returns the tile-1 dict (fully backward compatible) with reg["tiles"]: a list
    of {shape_AB, anchor, mask_coverage, coverage, low_confidence} per copy."""
    reg = register(green, cache_key=cache_key)
    reg["tiles"] = [dict(shape_AB=reg["shape_AB"], anchor=reg["anchor"],
                         mask_coverage=reg["mask_coverage"], coverage=reg["coverage"],
                         low_confidence=reg["low_confidence"], _b=tuple(reg["global_b"]),
                         sam_backed=sam_backed_sites(green.shape, reg["fpolys"],
                                                     reg["shape_AB"], reg["masks"]))]
    if reg["low_confidence"] or not MULTI_TILE_ENABLED:
        return reg                       # don't extend an already-doubtful base
    masks, fpolys = reg["masks"], reg["fpolys"]
    covered = covered_mask_indices(reg["shape_AB"], fpolys, masks, green.shape)
    A1 = np.asarray(reg["global_A"])
    rot1 = np.degrees(np.arctan2(A1[1, 0], A1[0, 0]))
    for _ in range(MULTI_TILE_MAX - 1):
        rm = []
        for i, m in enumerate(masks):
            m = np.asarray(m, bool)
            if i in covered or not (CONST_MIN_AREA <= m.sum() <= TILE_MAX_AREA):
                continue
            rm.append(m)
        if len(rm) < TILE_MIN_RESIDUAL:
            break
        alt = _fit_tile_offset(green, fpolys, np.asarray(reg["global_A"]),
                               np.asarray(reg["global_b"]), rm)
        if alt is None:
            break
        A2 = np.asarray(alt["A"])
        rot2 = np.degrees(np.arctan2(A2[1, 0], A2[0, 0]))
        drot = abs((rot2 - rot1 + 180) % 360 - 180)
        # neighbouring fields are either same-oriented or 180-rotated (ground
        # truth: seq3889's neighbour is flipped); both are legitimate.
        if min(drot, abs(180 - drot)) > TILE_ROT_TOL:
            break
        # copies must be pairwise adjacent: the new copy may not overlap ANY
        # accepted tile's footprint (not just tile 1's)
        A1inv = np.linalg.inv(np.asarray(reg["global_A"]))
        ext = np.vstack([np.asarray(fq, float) for fq in fpolys])
        cw, ch = float(np.ptp(ext[:, 0])), float(np.ptp(ext[:, 1]))
        clash = False
        for t in reg["tiles"]:
            dd = A1inv @ (np.asarray(alt["b"]) - np.asarray(t["_b"]))
            ox, oy = max(0.0, cw - abs(dd[0])), max(0.0, ch - abs(dd[1]))
            if ox * oy > TILE_OVERLAP_MAX * cw * ch:
                clash = True
                break
        if clash:
            break
        reg["tiles"].append(dict(shape_AB=alt["shape_AB"], anchor=("tile", "offset"),
                                 mask_coverage=alt["mcov"], coverage=alt["cov"],
                                 db=alt["db"], match_dist=alt["match_dist"],
                                 sam_backed=alt["sam_backed"],
                                 low_confidence=False, _b=tuple(np.asarray(alt["b"]))))
        covered |= covered_mask_indices(alt["shape_AB"], fpolys, masks, green.shape)
        # claim by SITE-CENTERING too (visible-part centroids, matching the fit):
        # a partially grown or frame-clipped crystal on one of this copy's sites
        # has no edges near the outline, so the edge test above won't claim it,
        # and the loop would keep re-fitting the same crystals.
        H2, W2 = green.shape
        placed = []
        for fq in fpolys:
            c, a = _clip_centroid(np.asarray(fq, float) @ A2.T + np.asarray(alt["b"]), W2, H2)
            if c is not None and a >= TILE_VIS_AREA_MIN:
                placed.append(c)
        placed = np.array(placed) if placed else np.zeros((0, 2))
        for i, m in enumerate(masks):
            if i in covered or not len(placed):
                continue
            m = np.asarray(m, bool)
            if m.sum() < CONST_MIN_AREA:
                continue
            ys, xs = np.nonzero(m)
            c = np.array([xs.mean(), ys.mean()])
            if np.linalg.norm(placed - c, axis=1).min() < TILE_CENTER_TOL:
                covered.add(i)
    return reg


def _clip_centroid(P, W, H):
    """Centroid and area of a polygon clipped to the frame rectangle (Sutherland-
    Hodgman), or (None, 0) if (almost) nothing is visible. An edge crystal and its
    predicted shape are clipped IDENTICALLY by the frame, so matching visible-part
    centroids removes the clip bias that full-shape centres suffer."""
    poly = [tuple(p) for p in P]
    edges = [(lambda p: p[0] >= 0, 0, 0), (lambda p: p[0] <= W, 0, W),
             (lambda p: p[1] >= 0, 1, 0), (lambda p: p[1] <= H, 1, H)]
    for inside, ax, v in edges:
        if not poly:
            return None, 0.0
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            ia, ib = inside(a), inside(b)
            if ia:
                out.append(a)
            if ia != ib:
                t = (v - a[ax]) / (b[ax] - a[ax])
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        poly = out
    if len(poly) < 3:
        return None, 0.0
    Q = np.array(poly)
    x, y = Q[:, 0], Q[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = cross.sum() / 2.0
    if abs(A) < 1.0:
        return None, 0.0
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * A)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * A)
    return np.array([cx, cy]), abs(A)


def _fit_tile_offset(green, fpolys, A1, b1, rm):
    """Fit the ADJACENT template copy to the residual crystals.

    Ground truth (seq3889, user-labelled): neighbouring fields are ROTATED 180
    degrees relative to tile 1 -- so BOTH orientations (A1 and -A1) are searched,
    translation free within each. Physical constraints honoured:
      - copies are stamped edge-to-edge, never interleaved: a candidate offset is
        valid only if the two cell footprints are (near-)disjoint and at most
        ~1.6 cell extents apart.
      - an edge crystal is frame-clipped, and so is the neighbour's predicted
        shape: matching uses the centroid of each site's VISIBLE (frame-clipped)
        part, which removes the clip bias of full-shape centres.
    An offset must centre >= TILE_CENTER_MIN in-footprint residual crystals
    (>= 50% of them) with a tight match; edge-ICP then refines the alignment
    against the residual crystal boundaries (centroids alone leave ~80px error)."""
    H, W = green.shape
    ext = np.vstack([np.asarray(fq, float) for fq in fpolys])
    cw, ch = float(np.ptp(ext[:, 0])), float(np.ptp(ext[:, 1]))
    A1inv = np.linalg.inv(A1)
    rc, rdim = [], []
    for m in rm:
        ys, xs = np.nonzero(np.asarray(m, bool))
        c = np.array([xs.mean(), ys.mean()])
        rc.append(c)
        Q = np.stack([xs - c[0], ys - c[1]], 1).astype(float)
        _, _, Vt = np.linalg.svd(Q / np.sqrt(len(Q)), full_matrices=False)
        pu, pv = Q @ Vt[0], Q @ Vt[1]
        rdim.append([np.ptp(pu), np.ptp(pv)])
    rc, rdim = np.array(rc), np.array(rdim)
    gdim = []
    for fq in fpolys:
        Q = np.asarray(fq, float) - np.asarray(fq, float).mean(0)
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        pu, pv = Q @ Vt[0], Q @ Vt[1]
        gdim.append([np.ptp(pu) * SCALE, np.ptp(pv) * SCALE])
    gdim = np.array(gdim)
    # crystal-vs-site size compatibility: partial growth SHORTENS a crystal but
    # keeps its full width (measured: 4139's diamonds match design widths at 100%
    # while lengths run 50-100%). A sliver crystal must not "match" a rhombus site.
    compat = np.ones((len(rc), len(fpolys)), bool)
    for i in range(len(rc)):
        L, Wd = max(rdim[i]), min(rdim[i])
        compat[i] = ((Wd >= 0.5 * gdim[:, 1]) & (Wd <= 1.7 * gdim[:, 1])
                     & (L >= 0.30 * gdim[:, 0]) & (L <= 1.20 * gdim[:, 0]))
    xm0, xm1 = ext[:, 0].min() - 30, ext[:, 0].max() + 30
    ym0, ym1 = ext[:, 1].min() - 30, ext[:, 1].max() + 30

    def adjacent_ok(db):
        dd = A1inv @ db
        ox, oy = max(0.0, cw - abs(dd[0])), max(0.0, ch - abs(dd[1]))
        return (ox * oy <= TILE_OVERLAP_MAX * cw * ch
                and abs(dd[0]) <= TILE_STEP_MAX * cw and abs(dd[1]) <= TILE_STEP_MAX * ch)

    def search(Aor):
        P0 = [np.asarray(fq, float) @ Aor.T + b1 for fq in fpolys]
        Aorinv = np.linalg.inv(Aor)

        def visible_sites(db):
            out = []
            for gi, P in enumerate(P0):
                c, a = _clip_centroid(P + db, W, H)
                if c is not None and a >= TILE_VIS_AREA_MIN:
                    out.append((gi, c))
            return out

        def match(db):
            """(count, mean_dist, need, pairs): residual crystals on visible sites;
            only residuals INSIDE this copy's footprint may match (partially grown
            tile-1 leftovers elsewhere cannot belong to this copy). pairs is the
            matched (crystal_idx, gds_idx) list -- downstream, those sites are
            SAM-backed: a segmented crystal sits on them."""
            vs = visible_sites(db)
            if not vs:
                return 0, 1e9, TILE_CENTER_MIN, []
            dd = (rc - b1 - db) @ Aorinv.T
            inft = ((dd[:, 0] >= xm0) & (dd[:, 0] <= xm1)
                    & (dd[:, 1] >= ym0) & (dd[:, 1] <= ym1))
            need = TILE_CENTER_MIN
            cs = np.array([c for _, c in vs])
            gis = [gi for gi, _ in vs]
            dist = np.linalg.norm(rc[:, None, :] - cs[None, :, :], axis=2)
            dist = np.where(compat[:, gis], dist, 1e9)   # size-incompatible pairings forbidden
            dmin = dist.min(1)
            js = dist.argmin(1)
            sel = (dmin < TILE_CENTER_TOL) & inft
            if not sel.any():
                return 0, 1e9, need, []
            pairs = [(int(i), int(gis[js[i]])) for i in np.nonzero(sel)[0]]
            return int(sel.sum()), float(dmin[sel].mean()), need, pairs

        gc0 = np.array([P.mean(0) for P in P0])
        dbs = (rc[:, None, :] - gc0[None, :, :]).reshape(-1, 2)
        dbs = [db for db in dbs if np.linalg.norm(db) >= TILE_MIN_OFFSET and adjacent_ok(db)]
        tried, out = [], []
        scored = sorted(((match(db)[0], -match(db)[1], tuple(db)) for db in dbs), reverse=True)
        for n0, negd, db in scored[:40]:
            if n0 == 0 or len(tried) >= 6:
                break
            db = np.array(db)
            if any(np.linalg.norm(db - t) < 300 for t in tried):
                continue
            tried.append(db)
            for step, span in ((120, 480), (30, 120)):
                grid = [(dxy, match(db + dxy)) for dxy in
                        (np.array([dx, dy]) for dx in range(-span, span + 1, step)
                         for dy in range(-span, span + 1, step))
                        if adjacent_ok(db + dxy)]
                if not grid:
                    break
                dxy, _ = max(grid, key=lambda t: (t[1][0], -t[1][1]))
                db = db + dxy
            n, mdist, needc, pairs = match(db)
            if n < needc or mdist > TILE_MATCH_DIST:
                continue
            out.append(dict(db=db, n=n, mdist=mdist, match=match, pairs=pairs))
        return out

    # collect EVERY candidate that passes the centroid gates, from both
    # orientations: the centroid-best is often not the edge-verified one (4152's
    # correct copy ranked behind a spurious pose on centroids alone).
    cands = []
    for Aor in (A1, -A1):                 # neighbouring fields may be 180-rotated
        for c in search(Aor):
            c["Aor"] = Aor
            cands.append(c)
    if not cands:
        return None
    cands.sort(key=lambda c: (-c["n"], c["mdist"]))
    cands = cands[:6]
    # Both orientations can tie on centroids (the flipped pose covers the same
    # sites approximately); EDGES break the tie -- only the true orientation wraps
    # each crystal's actual outline. Finalise both, count edge-hugging shapes.
    Bsoft2, Msm2 = boundary_maps(rm, green.shape)
    final = []
    for c in cands:
        Aor, db, match = c["Aor"], c["db"], c["match"]
        As, bs, med = icp_global(fpolys, Aor, b1 + db, Bsoft2, Msm2)
        dbi = bs - b1
        n2, mdist2, need2, pairs2 = (match(dbi) if adjacent_ok(dbi)
                                     else (0, 1e9, TILE_CENTER_MIN, []))
        if n2 < need2:
            As, bs = Aor, b1 + db          # keep the matched pose, drop the ICP drift
            n2, mdist2, pairs2 = c["n"], c["mdist"], c["pairs"]
        shape_AB = [icp_shape(fq, As, bs, Bsoft2, Msm2) for fq in fpolys]
        nalign = 0
        for fq, (Af, bf) in zip(fpolys, shape_AB):
            sc, tot = _shape_coverage(fq, Af, bf, Bsoft2, Msm2)
            if tot >= 4 and sc >= 0.5:
                nalign += 1
        if nalign < TILE_NALIGN_MIN:       # not edge-verified -> not a real copy
            continue
        final.append(dict(A=As, b=bs, med=med, ncent=n2, mdist=mdist2, db=db,
                          nalign=nalign, shape_AB=shape_AB,
                          sam_backed=sorted({gi for _, gi in pairs2})))
    if not final:
        return None
    best = max(final, key=lambda c: (c["nalign"], c["ncent"], -c["mdist"]))
    frac = best["ncent"] / max(len(rc), 1)
    return dict(A=best["A"], b=best["b"], med=best["med"], cov=frac, mcov=frac,
                ncent=best["ncent"], db=tuple(best["db"]),
                match_dist=best["mdist"], shape_AB=best["shape_AB"],
                sam_backed=best["sam_backed"])


def _render(green, reg, out_path):
    H, W = green.shape

    def n8(x, lo, hi):
        return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    rgb = np.repeat(n8(green, np.percentile(green, 2), np.percentile(green, 98))[:, :, None], 3, 2).astype(np.float32)
    ov = np.zeros((H, W, 3), np.float32)
    for m in reg["masks"]:
        ov[np.asarray(m, bool)] = [0, 150, 0]
    rgb = (0.78 * rgb + 0.22 * ov).astype(np.uint8)
    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img)
    tiles = reg.get("tiles") or [dict(shape_AB=reg["shape_AB"])]
    for ti, tile in enumerate(tiles):
        for q, fq, (A, b) in zip(reg["polys"], reg["fpolys"], tile["shape_AB"]):
            pts = fq @ np.asarray(A).T + np.asarray(b)
            col = {3: (255, 60, 60), 4: (60, 160, 255), 6: (255, 220, 0)}[len(q)]
            if ti > 0:      # extra tile copies: same scheme, lightened to tell apart
                col = tuple(min(255, c + 120) for c in col)
            dr.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=col, width=3)
    di, gi = reg["anchor"]
    flag = "  LOW-CONF" if reg["low_confidence"] else ""
    ntile = f"  {len(tiles)} tiles" if len(tiles) > 1 else ""
    dr.rectangle([0, 0, W, 30], fill=(0, 0, 0))
    dr.text((10, 8), f"det#{di}->GDS#{gi}  global {reg['global_med']:.1f}px  "
                     f"aniso {reg['aniso']:.1f}%{ntile}{flag}", fill=(240, 240, 240))
    img.resize((W // 2, H // 2), Image.LANCZOS).save(out_path)


def main():
    import sys
    frame = Path(sys.argv[1]) if len(sys.argv) > 1 else ea.FRAME
    if not frame.is_absolute():
        frame = ea.FRAME.parent / frame
    g = np.asarray(Image.open(frame).convert("RGB")).astype(np.float32)[:, :, 1]
    reg = register(g, cache_key=frame.stem)
    di, gi = reg["anchor"]
    c = reg["conf"] or {}
    print(f"{frame.name}: anchor det#{di}->GDS#{gi}  global {reg['global_med']:.1f}px  "
          f"aniso {reg['aniso']:.1f}%  mask-cov {100*reg['mask_coverage']:.0f}%  "
          f"margin(mcov +{100*c.get('mcov',0):.0f}pt / res {100*c.get('res',0):.0f}%)"
          f"{'  *** LOW CONFIDENCE ***' if reg['low_confidence'] else ''}")
    out = Path(__file__).resolve().parent.parent / "out" / f"{frame.stem}_registered.png"
    _render(g, reg, out)
    print(f"written {out}")


if __name__ == "__main__":
    main()
