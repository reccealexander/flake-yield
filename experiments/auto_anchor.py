#!/usr/bin/env python3
"""Fully automatic anchor selection for GDS registration -- no hand-picking.

The problem: a lone symmetric triangle can't reveal its own identity (mirror
partners GDS#10/#11 both fit a detection at 10px). The disambiguator is the
WHOLE tiling: only the correct (detection -> GDS-shape) identity places the rest
of the template onto real crystals. So:

  1. detect bright polygons; for each detection x each same-vertex GDS shape,
     compute the anchor transform (flipped GDS, PROPER rotation only).
  2. keep low-residual candidates; score each by GLOBAL edge alignment (mean
     gradient ridge under all in-frame template edges).
  3. refine the top-K with the edge-snap anisotropy step; the correct anchor
     settles to a tight globally-consistent fit, the wrong partner cannot.
  4. pick the min refined-residual candidate. Report the margin to the runner-up
     as a CONFIDENCE; flag thin margins for a quick human confirm.

The compact tiling has a near-translational symmetry, so a wrong partner can
still refine to a "reasonable" residual -- hence the confidence flag rather than
a silent pick. Validated on seq4045: auto-selects det#3 -> GDS#10 (the
user-confirmed anchor). See memory: gds-canonical-orientation, edge-snap-anisotropy.
"""

from pathlib import Path
from itertools import permutations

import numpy as np
from PIL import Image, ImageDraw

import edge_anisotropy as ea   # same directory

SCALE = ea.SCALE
CAND_RES_TOL = 25.0   # px, max anchor-corner residual to consider a candidate
TOP_K = 4             # refine this many top global-score candidates
CONF_MARGIN = 0.25    # refined-residual ratio below which we flag low confidence


def ccw(P):
    c = P.mean(0)
    return P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]


def anchor_fit(det_poly, gds_flipped):
    """Proper-rotation Kabsch, best over corner permutations. Returns
    (res, A, b, src_gds, tgt_img)."""
    d, gg = ccw(det_poly), ccw(gds_flipped)
    best = None
    for perm in permutations(range(len(gg))):
        s = gg[list(perm)]
        ms, mt = s.mean(0), d.mean(0)
        M = (d - mt).T @ (SCALE * (s - ms))
        U, S, Vt = np.linalg.svd(M)
        R = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        t = mt - SCALE * (R @ ms)
        res = np.sqrt(((SCALE * (s @ R.T) + t - d) ** 2).sum(1)).mean()
        if best is None or res < best[0]:
            best = (res, SCALE * R, t, s, d)
    return best


def global_score(fpolys, A, b, E, win=18):
    """Mean gradient-ridge response under all in-frame template edges."""
    H, W = E.shape
    tot, cnt = 0.0, 0
    from scipy import ndimage as ndi
    for fq in fpolys:
        P = fq @ A.T + b
        if P[:, 0].max() < 0 or P[:, 0].min() > W or P[:, 1].max() < 0 or P[:, 1].min() > H:
            continue
        for i in range(len(P)):
            a2, b2 = P[i], P[(i + 1) % len(P)]
            seg = b2 - a2
            L = np.hypot(*seg)
            if L < 20:
                continue
            u = seg / L
            n = np.array([-u[1], u[0]])
            for tt in np.linspace(0.1, 0.9, max(2, int(L // 22))):
                ip = a2 + tt * seg
                offs = np.arange(-win, win + 1, 2.0)
                line = ip[None, :] + offs[:, None] * n[None, :]
                xs, ys = line[:, 0], line[:, 1]
                ok = (xs > 1) & (xs < W - 2) & (ys > 1) & (ys < H - 2)
                if ok.sum() < 4:
                    continue
                vals = np.where(ok, ndi.map_coordinates(E, [ys, xs], order=1, mode="constant"), 0)
                tot += vals.max()
                cnt += 1
    return tot / max(cnt, 1)


def refine_from(fpolys, A, b, asrc, atgt, E):
    for win in ea.WINDOWS:
        SRC, TGT = ea.snap_correspondences(fpolys, A, b, E, win)
        A, b, med = ea.refit_affine(SRC, TGT, asrc, atgt)
    sv = np.linalg.svd(A, compute_uv=False)
    return A, b, med, 100 * (sv[0] - sv[1]) / sv[1]


def auto_register(green, cache_key=None):
    polys, fpolys = ea.load_gds_flipped()
    E = ea.edge_map(green)
    dets = ea.detect_sam(green, cache_key=cache_key)

    # enumerate candidates and coarse-score
    cands = []
    for di, dp in enumerate(dets):
        for gi, gv in enumerate(fpolys):
            if len(gv) != len(dp):
                continue
            res, A, b, asrc, atgt = anchor_fit(dp, gv)
            if res > CAND_RES_TOL:
                continue
            cands.append(dict(res=res, A=A, b=b, asrc=asrc, atgt=atgt, di=di, gi=gi))
    for c in cands:
        c["gscore"] = global_score(fpolys, c["A"], c["b"], E)
    cands.sort(key=lambda c: -c["gscore"])

    # refine top-K, rank by refined edge residual (correct anchor -> tightest)
    top = cands[:TOP_K]
    for c in top:
        c["A"], c["b"], c["med"], c["aniso"] = refine_from(
            fpolys, c["A"], c["b"], c["asrc"], c["atgt"], E)
    top.sort(key=lambda c: c["med"])

    best = top[0]
    conf = None
    if len(top) > 1:
        conf = (top[1]["med"] - best["med"]) / max(top[1]["med"], 1e-6)  # relative margin
    return dict(best=best, ranked=top, polys=polys, fpolys=fpolys, conf=conf,
                low_confidence=(conf is not None and conf < CONF_MARGIN))


def main():
    import sys
    frame = Path(sys.argv[1]) if len(sys.argv) > 1 else ea.FRAME
    if not frame.is_absolute():
        frame = ea.FRAME.parent / frame
    g = np.asarray(Image.open(frame).convert("RGB")).astype(np.float32)[:, :, 1]
    H, W = g.shape
    r = auto_register(g, cache_key=frame.stem)
    b = r["best"]
    print(f"AUTO anchor: det#{b['di']} -> GDS#{b['gi']}  "
          f"(refined edge resid {b['med']:.1f}px, aniso {b['aniso']:.1f}%)")
    print("ranked candidates (refined edge residual, lower=better):")
    for c in r["ranked"]:
        print(f"  det#{c['di']}->GDS#{c['gi']}: {c['med']:.1f}px  aniso {c['aniso']:.1f}%  gscore {c['gscore']:.3f}")
    if r["conf"] is not None:
        flag = "  *** LOW CONFIDENCE - confirm by eye ***" if r["low_confidence"] else ""
        print(f"confidence margin over runner-up: {100*r['conf']:.0f}%{flag}")

    def n8(x, lo, hi):
        return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    vis = np.repeat(n8(g, np.percentile(g, 2), np.percentile(g, 98))[:, :, None], 3, axis=2).astype(np.uint8)
    img = Image.fromarray(vis)
    dr = ImageDraw.Draw(img)
    for fq, q in zip(r["fpolys"], r["polys"]):
        pts = fq @ b["A"].T + b["b"]
        col = {3: (255, 40, 40), 4: (60, 200, 255), 6: (90, 220, 90)}[len(q)]
        dr.line([tuple(p) for p in pts] + [tuple(pts[0])], fill=col, width=3)
    for p in b["atgt"]:
        dr.ellipse([p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10], outline=(255, 0, 255), width=4)
    dr.rectangle([0, 0, W, 30], fill=(0, 0, 0))
    dr.text((10, 8), f"AUTO: det#{b['di']}->GDS#{b['gi']}  resid {b['med']:.1f}px  aniso {b['aniso']:.1f}%", fill=(240, 240, 240))
    out = Path(__file__).resolve().parent.parent / "out" / f"{frame.stem}_auto_anchor.png"
    img.resize((W // 2, H // 2), Image.LANCZOS).save(out)
    print(f"written {out}")


if __name__ == "__main__":
    main()
