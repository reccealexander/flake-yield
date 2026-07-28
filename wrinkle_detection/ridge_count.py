#!/usr/bin/env python3
"""Ridge-aware wrinkle counting from an existing density-U-Net heatmap.

Motivation
----------
The production readout (``count_wrinkle_unet.count_from_heatmap``) reads the
U-Net density map two ways -- isotropic local-maxima PEAK counting and the
INTEGRAL -- and hybridises them. On dense / merged wrinkles the Gaussian bumps
fuse into *elongated plateaus running along the wrinkle direction*: peak
counting under-counts (one plateau = one peak) and the plateau-collapse can
merge two parallel wrinkles into one. Integral counting is unbiased in
expectation but noisy per-crop.

This module instead treats the heatmap as an image of thin bright *ridges* and
counts ridge INSTANCES: each physical wrinkle should be one 1-D curve. We

  1. hysteresis-threshold the heatmap (relative to its own max, using the same
     ``peak_thresh`` / ``peak_floor`` conventions as the peak counter) into
     filament regions,
  2. skeletonize each region to a 1-px medial axis,
  3. build a node/edge graph of the skeleton (endpoints + junctions as nodes,
     the deg-2 runs between them as edges),
  4. bridge dashed gaps (near + collinear endpoints) with virtual edges,
  5. trace *strokes* through the graph -- at every junction a stroke continues
     into the edge that best keeps going straight (turn < ``merge_angle_deg``),
     so an X-crossing yields the two through-lines rather than four stubs and a
     bent (L-shaped) wrinkle stays a single stroke,
  6. collapse near-isotropic blobs (single sparse wrinkles / lone bumps) to one
     instance each so the sparse case never fragments.

Count = number of ridge instances. Each instance also carries its polyline,
arc length and mean heatmap density.

A local orientation field (structure tensor of the heatmap) is computed and
used both to align the dashed-gap bridging and as returned debug info.

Torch-free: uses numpy, scipy, skimage (skeletonize) and cv2. Import-safe on a
machine without torch, so ``tests/test_ridge_count.py`` runs on synthetic
heatmaps with no checkpoint.

Public API
----------
``count_from_heatmap_ridge(heatmap, gray=None, ...)`` -> dict with keys
``count``, ``instances`` (list of dicts: ``polyline`` (N,2) xy, ``length``,
``mean_density``), ``skeleton``, ``orientation`` and the resolved params.

``render_ridge_debug(gray, mask, result, out_path)`` draws the numbered
polylines over the crop in the style of ``count_wrinkle_unet.render_debug``.
"""

import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# Default hyper-parameters.
#
# Frozen from a small grid search on the TRAINING-portion crops (seed-42 split,
# holdout seq4213+ excluded) -- see eval_ridge_count.py. At these values RIDGE
# beats PEAK and HYBRID on the (biased) full 127-crop set: overall count-MAE
# 1.60 vs 1.76 / 1.73, sparse bin 0-2 0.38 vs 0.52, dense bin 6+ 4.09 vs 4.38;
# on the best-effort val split 1.19 vs 1.57 / 1.43. FLOOR / INST_FLOOR echo the
# peak counter's peak_floor so a ridge must clear a comparable brightness bar.
# ---------------------------------------------------------------------------
DEFAULT_HI_FRAC = 0.25       # hysteresis HIGH = frac of this crop's heatmap max
DEFAULT_LO_FRAC = 0.10       # hysteresis LOW  = frac of max (region growth)
DEFAULT_FLOOR = 0.35         # absolute floor on the HIGH level (empty-crop guard)
DEFAULT_MERGE_ANGLE_DEG = 25.0   # max turn to continue a stroke through a node
DEFAULT_GAP_PX = 10.0        # max endpoint gap to bridge a dashed ridge
DEFAULT_GAP_ANGLE_DEG = 30.0     # max misalignment to bridge a dashed gap
DEFAULT_MIN_RIDGE_LEN = 7.0  # region diameter below this -> one blob instance
DEFAULT_MIN_INSTANCE_LEN = 3.0   # drop shorter stub strokes (never a whole region)
DEFAULT_TENSOR_SIGMA = 2.0   # structure-tensor smoothing scale (~ 0.7 * bump sigma)
DEFAULT_CREST_STEP = 1.5     # across-ridge sampling offset (px) for crest NMS
DEFAULT_COH_MIN = 0.05       # min structure-tensor coherence to accept a crest px
DEFAULT_SIGMA = 3.0          # heatmap bump sigma (for the region integral fallback)
DEFAULT_DENSE_MIN = 5.0      # a region's integral must reach this to fall back
DEFAULT_DENSE_MARGIN = 1.0   # ... AND exceed its instance count by this much
DEFAULT_INST_FLOOR = 0.25    # drop instances whose mean density < this (~ peak_floor)


# ---------------------------------------------------------------------------
# Orientation field (structure tensor)
# ---------------------------------------------------------------------------

def structure_tensor_orientation(img, sigma=DEFAULT_TENSOR_SIGMA):
    """Local orientation + coherence of ``img`` via the structure tensor.

    Returns ``(theta, coherence)`` where ``theta`` in [0, pi) is the direction
    of the ridge (the low-gradient / along-ridge axis) at each pixel and
    ``coherence`` in [0, 1] measures how anisotropic (ridge-like) the neighbour-
    hood is. Uses a Gaussian derivative + Gaussian tensor smoothing.
    """
    from scipy import ndimage
    img = np.asarray(img, dtype=np.float32)
    gy, gx = np.gradient(img)
    Jxx = ndimage.gaussian_filter(gx * gx, sigma)
    Jyy = ndimage.gaussian_filter(gy * gy, sigma)
    Jxy = ndimage.gaussian_filter(gx * gy, sigma)
    # eigen-decomposition of [[Jxx,Jxy],[Jxy,Jyy]]
    diff = Jxx - Jyy
    tmp = np.sqrt(diff * diff + 4.0 * Jxy * Jxy)
    lam1 = 0.5 * (Jxx + Jyy + tmp)   # larger eigenvalue -> across-ridge gradient
    lam2 = 0.5 * (Jxx + Jyy - tmp)
    # gradient (across-ridge) direction is eigenvector of lam1; ridge is +90 deg
    grad_theta = 0.5 * np.arctan2(2.0 * Jxy, diff)
    theta = np.mod(grad_theta + math.pi / 2.0, math.pi)
    coh = np.where((lam1 + lam2) > 1e-12, (lam1 - lam2) / (lam1 + lam2 + 1e-12), 0.0)
    return theta.astype(np.float32), np.clip(coh, 0.0, 1.0).astype(np.float32)


def ridge_crest_mask(hm, theta, coh, region, lo_level, coh_min=DEFAULT_COH_MIN,
                     step=DEFAULT_CREST_STEP):
    """Across-ridge non-maximum-suppression -> thin ridge-crest lines.

    A pixel is a crest if the heatmap there is >= the heatmap sampled one
    ``step`` to either side ALONG THE ACROSS-RIDGE direction (theta + 90 deg).
    Unlike the isotropic peak detector, this keeps two close *parallel* ridges
    apart -- each has its own crest even when their density plateaus merge, so
    an elongated fused plateau is split into one line per wrinkle instead of a
    single blob. Restricted to ``region`` (the hysteresis filament mask), to
    pixels above ``lo_level`` and to anisotropic (coherent) neighbourhoods.
    """
    from scipy import ndimage
    hm = np.asarray(hm, dtype=np.float32)
    H, W = hm.shape
    ax = -np.sin(theta)          # across-ridge unit vector = (cos(theta+90), sin)
    ay = np.cos(theta)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def sample(sx, sy):
        return ndimage.map_coordinates(hm, [yy + sy, xx + sx], order=1,
                                       mode="nearest").astype(np.float32)

    vp = sample(step * ax, step * ay)
    vm = sample(-step * ax, -step * ay)
    eps = 1e-6
    crest = (hm + eps >= vp) & (hm + eps >= vm) & (hm >= lo_level) \
        & (coh >= coh_min) & (np.asarray(region) > 0)
    return crest


# ---------------------------------------------------------------------------
# Skeleton graph
# ---------------------------------------------------------------------------

_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbor_count(skel):
    from scipy import ndimage
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return ndimage.convolve(skel.astype(np.uint8), k, mode="constant", cval=0)


def _skeletonize(mask):
    """1-px skeleton of a binary mask (skimage if present, else cv2 thinning)."""
    mask = np.asarray(mask) > 0
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask).astype(bool)
    except Exception:
        # Fallback: Zhang-Suen thinning via cv2.ximgproc, or morphological.
        import cv2
        m = (mask.astype(np.uint8)) * 255
        try:
            thin = cv2.ximgproc.thinning(m)
            return thin > 0
        except Exception:
            # crude morphological skeleton
            skel = np.zeros_like(m)
            el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            cur = m.copy()
            while True:
                op = cv2.morphologyEx(cur, cv2.MORPH_OPEN, el)
                tmp = cv2.subtract(cur, op)
                er = cv2.erode(cur, el)
                skel = cv2.bitwise_or(skel, tmp)
                cur = er
                if cv2.countNonZero(cur) == 0:
                    break
            return skel > 0


class _Graph:
    """Node/edge graph extracted from a skeleton.

    Nodes are endpoint/junction pixel-clusters (plus isolated dots). Edges are
    ordered pixel paths; each edge records unit outgoing directions at both
    ends for stroke tracing.
    """

    def __init__(self):
        self.node_xy = []       # (x, y) centroid per node
        self.node_edges = []    # list of incident edge ids per node
        self.edges = []         # list of dicts

    def add_node(self, xy):
        self.node_xy.append(np.asarray(xy, dtype=np.float64))
        self.node_edges.append([])
        return len(self.node_xy) - 1

    def add_edge(self, u, v, path_xy, virtual=False):
        path = np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
        eid = len(self.edges)
        du = _end_dir(path, at_start=True, node_xy=self.node_xy[u])
        dv = _end_dir(path, at_start=False, node_xy=self.node_xy[v])
        length = _polyline_length(path)
        self.edges.append({"u": u, "v": v, "path": path, "du": du, "dv": dv,
                           "length": length, "virtual": virtual})
        self.node_edges[u].append(eid)
        self.node_edges[v].append(eid)
        return eid


def _polyline_length(path):
    if len(path) < 2:
        return 0.0
    d = np.diff(path, axis=0)
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def _end_dir(path, at_start, node_xy, k=4):
    """Unit vector pointing from the node end of ``path`` into the polyline."""
    path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    if len(path) == 1:
        return np.array([0.0, 0.0])
    if at_start:
        seg = path[:min(k, len(path))]
        d = seg[-1] - node_xy
    else:
        seg = path[max(0, len(path) - k):]
        d = seg[0] - node_xy
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([0.0, 0.0])


def build_skeleton_graph(skel):
    """Skeleton -> (_Graph, comp_labels, ncomp).

    ``comp_labels`` labels every skeleton pixel with its 8-connected component;
    used later to collapse near-isotropic blob components.
    """
    from scipy import ndimage
    skel = np.asarray(skel) > 0
    g = _Graph()
    comp_labels, ncomp = ndimage.label(skel, structure=np.ones((3, 3)))
    if not skel.any():
        return g, comp_labels, ncomp

    deg = _neighbor_count(skel) * skel
    node_px = skel & ((deg == 1) | (deg >= 3) | (deg == 0))
    # cluster node pixels into nodes (8-conn)
    node_lab, nnodes = ndimage.label(node_px, structure=np.ones((3, 3)))
    pix_to_node = {}
    for nid in range(1, nnodes + 1):
        ys, xs = np.where(node_lab == nid)
        gidx = g.add_node((xs.mean(), ys.mean()))
        for y, x in zip(ys, xs):
            pix_to_node[(y, x)] = gidx

    H, W = skel.shape
    visited_path = np.zeros_like(skel, dtype=bool)

    def is_node(y, x):
        return (y, x) in pix_to_node

    # trace edges starting from every node pixel
    for (ny, nx), u in list(pix_to_node.items()):
        for dy, dx in _NB8:
            y, x = ny + dy, nx + dx
            if not (0 <= y < H and 0 <= x < W) or not skel[y, x]:
                continue
            if is_node(y, x):
                # direct node-node adjacency (short link): make a 2-pt edge once
                v = pix_to_node[(y, x)]
                if v == u:
                    continue
                key = (min((ny, nx), (y, x)), max((ny, nx), (y, x)))
                if visited_path[y, x] and visited_path[ny, nx]:
                    pass
                # avoid duplicate: only add from the lexicographically smaller px
                if (ny, nx) < (y, x):
                    g.add_edge(u, v, [(nx, ny), (x, y)])
                continue
            if visited_path[y, x]:
                continue
            # walk the deg-2 corridor to the next node
            path = [(nx, ny), (x, y)]
            visited_path[y, x] = True
            py, px = ny, nx
            cy, cx = y, x
            reached = None
            while True:
                nxts = []
                for ddy, ddx in _NB8:
                    yy, xx = cy + ddy, cx + ddx
                    if not (0 <= yy < H and 0 <= xx < W) or not skel[yy, xx]:
                        continue
                    if (yy, xx) == (py, px):
                        continue
                    nxts.append((yy, xx))
                # prefer a node neighbour to terminate
                node_nbrs = [p for p in nxts if is_node(*p)]
                if node_nbrs:
                    ry, rx = node_nbrs[0]
                    path.append((rx, ry))
                    reached = pix_to_node[(ry, rx)]
                    break
                cont = [p for p in nxts if not visited_path[p]]
                if not cont:
                    reached = u  # dead end back (shouldn't usually happen)
                    break
                ny2, nx2 = cont[0]
                visited_path[ny2, nx2] = True
                path.append((nx2, ny2))
                py, px = cy, cx
                cy, cx = ny2, nx2
            if reached is not None:
                g.add_edge(u, reached, path)

    return g, comp_labels, ncomp


# ---------------------------------------------------------------------------
# Dashed-gap bridging + stroke tracing
# ---------------------------------------------------------------------------

def _angle_between(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return math.pi
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.acos(c)


def _bridge_gaps(g, gap_px, gap_angle_deg):
    """Add virtual edges between endpoint nodes that are near + collinear.

    An endpoint node (degree 1, real edges only) extends outward along
    ``-du`` of its single edge. Two endpoints A, B are bridged when |AB| is
    small AND the connecting segment is aligned with BOTH outward directions
    (dashed continuation of one physical wrinkle).
    """
    gap_ang = math.radians(gap_angle_deg)
    endpoints = []
    for nid, eids in enumerate(g.node_edges):
        real = [e for e in eids if not g.edges[e]["virtual"]]
        if len(real) == 1:
            e = g.edges[real[0]]
            d = e["du"] if e["u"] == nid else e["dv"]
            outward = -d  # wrinkle continues away from the node
            endpoints.append((nid, np.asarray(g.node_xy[nid]), outward))

    used = set()
    for i in range(len(endpoints)):
        ni, pi, oi = endpoints[i]
        best = None
        for j in range(len(endpoints)):
            if i == j:
                continue
            nj, pj, oj = endpoints[j]
            if ni == nj:
                continue
            seg = pj - pi
            dist = np.linalg.norm(seg)
            if dist < 1e-6 or dist > gap_px:
                continue
            # segment must align with A's outward dir and with -B's outward dir
            if _angle_between(seg, oi) > gap_ang:
                continue
            if _angle_between(-seg, oj) > gap_ang:
                continue
            if best is None or dist < best[0]:
                best = (dist, nj, pj)
        if best is not None:
            nj = best[1]
            key = (min(ni, nj), max(ni, nj))
            if key in used:
                continue
            used.add(key)
            g.add_edge(ni, nj, [g.node_xy[ni], g.node_xy[nj]], virtual=True)


def _trace_strokes(g, merge_angle_deg):
    """Greedy stroke decomposition of the graph.

    Each stroke starts at an unused edge (preferring one at a degree-1 node) and
    extends through nodes into the unused incident edge that best continues the
    current travel direction (turn < ``merge_angle_deg``). Returns a list of
    ordered (M,2) xy polylines.
    """
    max_turn = math.radians(merge_angle_deg)
    n_edges = len(g.edges)
    used = [False] * n_edges

    def node_degree(nid):
        return len(g.node_edges[nid])

    def out_dir(eid, node):
        e = g.edges[eid]
        return e["du"] if e["u"] == node else e["dv"]

    def other(eid, node):
        e = g.edges[eid]
        return e["v"] if e["u"] == node else e["u"]

    def edge_path_from(eid, node):
        e = g.edges[eid]
        p = e["path"]
        return p if e["u"] == node else p[::-1]

    # order: prefer starting edges attached to endpoints, then longer edges
    order = sorted(range(n_edges),
                   key=lambda e: (min(node_degree(g.edges[e]["u"]),
                                      node_degree(g.edges[e]["v"])) != 1,
                                  -g.edges[e]["length"]))

    strokes = []
    for start in order:
        if used[start]:
            continue
        used[start] = True
        e = g.edges[start]
        # build initial path; choose orientation so we can extend both ends
        path = list(map(tuple, e["path"]))
        head_node, tail_node = e["u"], e["v"]
        head_path = list(map(tuple, edge_path_from(start, head_node)))  # tail->? no
        # We'll extend from tail_node forward and head_node backward.

        def extend(node, incoming_travel, append_side):
            nonlocal path
            cur_node = node
            travel = incoming_travel
            while True:
                cands = [eid for eid in g.node_edges[cur_node]
                         if not used[eid]]
                best_eid, best_align = None, -2.0
                for eid in cands:
                    od = out_dir(eid, cur_node)
                    if np.linalg.norm(od) < 1e-9:
                        continue
                    align = float(np.dot(travel, od))
                    if align > best_align:
                        best_align, best_eid = align, eid
                if best_eid is None or best_align < math.cos(max_turn):
                    break
                used[best_eid] = True
                seg = list(map(tuple, edge_path_from(best_eid, cur_node)))
                if append_side == "tail":
                    path = path + seg[1:]
                else:
                    path = seg[::-1][:-1] + path
                nxt = other(best_eid, cur_node)
                travel = -out_dir(best_eid, nxt)
                cur_node = nxt

        # travel direction arriving at tail_node = -dv
        extend(tail_node, -e["dv"], "tail")
        extend(head_node, -e["du"], "head")
        strokes.append(np.asarray(path, dtype=np.float64))

    return strokes


# ---------------------------------------------------------------------------
# Top-level ridge count
# ---------------------------------------------------------------------------

def _hysteresis_mask(heatmap, hi, lo):
    """Region-grow: keep LOW-connected components that contain a HIGH pixel."""
    from scipy import ndimage
    hm = np.asarray(heatmap, dtype=np.float32)
    low_mask = hm >= lo
    high_mask = hm >= hi
    if not high_mask.any():
        return np.zeros_like(hm, dtype=bool)
    lab, n = ndimage.label(low_mask, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(lab[high_mask])] = True
    keep[0] = False
    return keep[lab]


def _component_diameter(pts_xy):
    """Approx max pairwise distance of a small pixel set (via bbox diagonal for
    speed, exact for tiny sets)."""
    if len(pts_xy) <= 1:
        return 0.0
    mn = pts_xy.min(axis=0)
    mx = pts_xy.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def count_from_heatmap_ridge(heatmap, gray=None, mask=None,
                             hi_frac=DEFAULT_HI_FRAC, lo_frac=DEFAULT_LO_FRAC,
                             floor=DEFAULT_FLOOR,
                             merge_angle_deg=DEFAULT_MERGE_ANGLE_DEG,
                             gap_px=DEFAULT_GAP_PX,
                             gap_angle_deg=DEFAULT_GAP_ANGLE_DEG,
                             min_ridge_len=DEFAULT_MIN_RIDGE_LEN,
                             min_instance_len=DEFAULT_MIN_INSTANCE_LEN,
                             tensor_sigma=DEFAULT_TENSOR_SIGMA,
                             crest_step=DEFAULT_CREST_STEP,
                             coh_min=DEFAULT_COH_MIN,
                             sigma=DEFAULT_SIGMA,
                             dense_fallback=True,
                             dense_min=DEFAULT_DENSE_MIN,
                             dense_margin=DEFAULT_DENSE_MARGIN,
                             inst_floor=DEFAULT_INST_FLOOR):
    """Ridge-aware wrinkle count from a density heatmap.

    Pipeline: hysteresis filament REGIONS (``max(hi_frac*max, floor)`` /
    ``lo_frac*max``, echoing the peak counter's thresholds) -> across-ridge
    directional NMS crest lines inside those regions (this is what separates
    close parallel wrinkles a medial-axis skeleton would fuse) -> skeleton
    graph -> dashed-gap bridging -> stroke tracing through junctions -> one
    instance per stroke. Bookkeeping is done per hysteresis REGION so every
    filament region yields at least one instance (a lone isotropic bump, whose
    crest is empty, still counts as 1) and compact regions collapse to a single
    blob instance.

    ``gray`` is retained for debug overlays; orientation / crests are computed
    on the heatmap itself. Returns a dict compatible in spirit with
    ``count_wrinkle_unet.count_from_heatmap``::

        {
          "count": int,                 # number of ridge instances
          "instances": [                # one per physical wrinkle
             {"polyline": (M,2) xy, "length": float, "mean_density": float}, ...
          ],
          "peaks": (K,2) int,           # instance midpoints (for render reuse)
          "skeleton": bool HxW,         # the crest skeleton actually traced
          "orientation": (theta, coherence),
          "heatmap": heatmap,
          "params": {...},
        }
    """
    from scipy import ndimage
    hm = np.asarray(heatmap, dtype=np.float32)
    if mask is not None:
        hm = hm * (np.asarray(mask) > 0)
    params = dict(hi_frac=hi_frac, lo_frac=lo_frac, floor=floor,
                  merge_angle_deg=merge_angle_deg, gap_px=gap_px,
                  gap_angle_deg=gap_angle_deg, min_ridge_len=min_ridge_len,
                  min_instance_len=min_instance_len, tensor_sigma=tensor_sigma,
                  crest_step=crest_step, coh_min=coh_min)
    theta, coh = structure_tensor_orientation(hm, tensor_sigma)

    empty = {"count": 0, "count_pure": 0, "instances": [],
             "peaks": np.zeros((0, 2), np.int32),
             "skeleton": np.zeros(hm.shape, bool),
             "orientation": (theta, coh), "heatmap": hm, "params": params}

    mx = float(hm.max())
    if mx <= 0.0:
        return empty
    hi = max(hi_frac * mx, floor)
    lo = min(lo_frac * mx, hi)          # LOW must not exceed HIGH
    if lo <= 0:
        lo = 0.5 * hi

    fil = _hysteresis_mask(hm, hi, lo)
    if not fil.any():
        return empty

    # across-ridge NMS crest lines, then thin to 1-px skeleton
    crest = ridge_crest_mask(hm, theta, coh, fil, lo, coh_min, crest_step)
    skel = _skeletonize(crest)

    strokes = []
    if skel.any():
        g, _, _ = build_skeleton_graph(skel)
        _bridge_gaps(g, gap_px, gap_angle_deg)
        strokes = _trace_strokes(g, merge_angle_deg)

    # Bookkeeping per hysteresis REGION (not per crest component): guarantees
    # every filament region contributes >= 1 instance even if its crest is empty.
    region_labels, nreg = ndimage.label(fil, structure=np.ones((3, 3)))
    region_pix = {}
    ys, xs = np.where(fil)
    for y, x in zip(ys, xs):
        region_pix.setdefault(int(region_labels[y, x]), []).append((x, y))
    region_diam = {r: _component_diameter(np.asarray(p, dtype=np.float64))
                   for r, p in region_pix.items()}
    small_regions = {r for r, d in region_diam.items() if d < min_ridge_len}

    H, W = hm.shape

    def stroke_regions(poly):
        xi = np.clip(np.round(poly[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(poly[:, 1]).astype(int), 0, H - 1)
        return set(int(r) for r in np.unique(region_labels[yi, xi]) if r != 0)

    def midpoint_region(poly):
        mid = poly[len(poly) // 2]
        xi = min(max(int(round(mid[0])), 0), W - 1)
        yi = min(max(int(round(mid[1])), 0), H - 1)
        r = int(region_labels[yi, xi])
        if r != 0:
            return r
        rs = stroke_regions(poly)
        return next(iter(rs)) if rs else 0

    instances = []                       # each tagged with ["region"]
    covered = set()
    for s in strokes:
        rs = stroke_regions(s)
        if rs and rs.issubset(small_regions):
            continue                     # compact region -> handled as one blob
        if _polyline_length(s) < min_instance_len:
            continue
        inst = _make_instance(s, hm)
        inst["region"] = midpoint_region(s)
        instances.append(inst)
        covered |= rs

    # each compact region -> exactly one blob instance
    for r in small_regions:
        pts = np.asarray(region_pix[r], dtype=np.float64)
        poly = pts.mean(0)[None, :] if len(pts) < 2 else _order_blob(pts)
        inst = _make_instance(poly, hm)
        inst["region"] = r
        instances.append(inst)
        covered.add(r)

    # any region with no surviving stroke (sparse wrinkle whose crest was
    # filtered / too short) -> one fallback instance so nothing vanishes
    for r, p in region_pix.items():
        if r in covered:
            continue
        pts = np.asarray(p, dtype=np.float64)
        poly = pts.mean(0)[None, :] if len(pts) < 2 else _order_blob(pts)
        inst = _make_instance(poly, hm)
        inst["region"] = r
        instances.append(inst)
        covered.add(r)

    # Absolute density floor (mirrors the peak counter's peak_floor): a real
    # wrinkle's crest sits near a bump peak (~1); low-density instances are
    # noise / model hallucination on near-empty crops. Drop them so the sparse
    # bin does not over-count.
    if inst_floor > 0:
        instances = [inst for inst in instances
                     if inst["mean_density"] >= inst_floor]

    peaks = np.array([inst["polyline"][len(inst["polyline"]) // 2]
                      for inst in instances], dtype=np.float64) \
        if instances else np.zeros((0, 2))
    peaks = np.round(peaks).astype(np.int32) if len(peaks) else \
        np.zeros((0, 2), np.int32)

    # ---- region-wise dense fallback -----------------------------------
    # Where the model fused many wrinkles into a saturated bright REGION,
    # instance extraction under-counts; there the region's integral (density
    # mass / one-bump volume) tracks the true count far better. Mirrors the
    # production peak/integral hybrid, but decided per region and using ridge
    # instances (not isotropic peaks) for the non-saturated majority.
    bump_vol = 2.0 * math.pi * float(sigma) * float(sigma)
    inst_per_region = {}
    for inst in instances:
        inst_per_region[inst["region"]] = inst_per_region.get(inst["region"], 0) + 1
    count_pure = len(instances)
    count = count_pure
    if dense_fallback:
        total = 0
        for r, k in inst_per_region.items():
            if r == 0:
                total += k
                continue
            rp = np.asarray(region_pix[r])       # columns: (x, y)
            ig_r = float(hm[rp[:, 1], rp[:, 0]].sum() / bump_vol)
            if ig_r >= dense_min and ig_r > k + dense_margin:
                total += int(round(ig_r))
            else:
                total += k
        count = total

    return {"count": count, "count_pure": count_pure, "instances": instances,
            "peaks": peaks, "skeleton": skel, "orientation": (theta, coh),
            "heatmap": hm, "params": params}


def _order_blob(pts):
    """Order a tiny blob's pixels into a short polyline along its long axis."""
    c = pts.mean(0)
    d = pts - c
    # principal axis via SVD
    try:
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        axis = vt[0]
    except Exception:
        axis = np.array([1.0, 0.0])
    t = d @ axis
    order = np.argsort(t)
    return pts[order]


def _make_instance(poly, hm):
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    length = _polyline_length(poly)
    H, W = hm.shape
    xi = np.clip(np.round(poly[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(poly[:, 1]).astype(int), 0, H - 1)
    dens = hm[yi, xi]
    return {"polyline": poly, "length": float(length),
            "mean_density": float(dens.mean()) if len(dens) else 0.0}


# ---------------------------------------------------------------------------
# Debug rendering (extends count_wrinkle_unet.render_debug's style)
# ---------------------------------------------------------------------------

def _colorize(heatmap):
    hm = np.asarray(heatmap, dtype=np.float32)
    hi = float(hm.max())
    norm = hm / hi if hi > 1e-6 else hm
    r = np.clip(1.5 * norm - 0.5, 0, 1)
    g = np.clip(1.0 - np.abs(norm - 0.5) * 2.0, 0, 1)
    b = np.clip(1.0 - 1.5 * norm, 0, 1)
    return np.stack([r, g, b], axis=-1)


_PALETTE = [(255, 60, 60), (60, 200, 255), (120, 255, 120), (255, 220, 60),
            (255, 120, 255), (120, 160, 255), (255, 170, 70), (170, 255, 220)]


def render_ridge_debug(gray, mask, result, out_path, alpha=0.45, extra_label=""):
    """Overlay heatmap + numbered ridge polylines on the crop; save a PNG.

    Mirrors ``count_wrinkle_unet.render_debug``: greyscale base, blue->red
    heatmap tint, plus each instance's polyline drawn in a distinct colour with
    its index label near the polyline midpoint.
    """
    from PIL import Image, ImageDraw
    gray = np.asarray(gray, dtype=np.float32)
    heatmap = np.asarray(result["heatmap"], dtype=np.float32)

    base = np.stack([gray, gray, gray], axis=-1)
    heat = _colorize(heatmap)
    hm_alpha = (heatmap / heatmap.max() if heatmap.max() > 1e-6
                else heatmap)[..., None] * alpha
    blended = np.clip(base * (1 - hm_alpha) + heat * hm_alpha, 0, 1)
    # upscale x3 so thin polylines are legible
    scale = 3
    img = Image.fromarray((blended * 255).astype(np.uint8), "RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    for i, inst in enumerate(result.get("instances", [])):
        col = _PALETTE[i % len(_PALETTE)]
        poly = np.asarray(inst["polyline"], dtype=np.float64) * scale
        if len(poly) >= 2:
            draw.line([tuple(p) for p in poly], fill=col, width=2, joint="curve")
            for p in (poly[0], poly[-1]):
                draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=col)
        else:
            p = poly[0]
            r = 3
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=col, width=2)
        mid = poly[len(poly) // 2]
        draw.text((mid[0] + 3, mid[1] - 5), str(i + 1), fill=col)

    label = "ridge=%d" % result["count"]
    cp = result.get("count_pure")
    if cp is not None and cp != result["count"]:
        label += "(pure=%d,dense-fb)" % cp   # count came from the region integral
    if extra_label:
        label += "  " + extra_label
    draw.rectangle([0, 0, 8 + 7 * len(label), 14], fill=(0, 0, 0))
    draw.text((3, 2), label, fill=(255, 255, 0))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    # tiny self-check on a synthetic two-ridge heatmap
    from scipy import ndimage
    hm = np.zeros((128, 128), np.float32)
    hm[40:90, 50] = 1.0
    hm[40:90, 70] = 1.0
    hm = ndimage.gaussian_filter(hm, 2.0)
    hm /= hm.max()
    res = count_from_heatmap_ridge(hm)
    print("two parallel ridges -> count =", res["count"])
