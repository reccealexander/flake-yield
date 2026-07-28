#!/usr/bin/env python3
"""Build a database of registered crystals from the processed frames.

For each frame that registers confidently, crop EVERY GDS-outlined crystal (even
if it hangs over the frame edge) from the ORIGINAL colour image, then keep only
crystals that pass two constraints before storing:

  1. NOT a bright crystal  -- interior mean brightness minus the frame background
     median is below BRIGHT_MAX_EXCESS. The bright distractor crystals scatter far
     harder than the faint target crystals; this drops them.
  2. UNIFORM in colour (ignoring wrinkles) -- chromaticity R/(R+G) is brightness-
     invariant, so a wrinkle (a brightness/topography change) barely moves it;
     it is median-smoothed to erase thin wrinkle lines, then its spread over the
     interior must be below HUE_SPREAD_MAX. A genuinely multi-colour crystal
     (regions of different thickness/type) is rejected; a wrinkled single-colour
     one is kept.

Stored per crystal: the ORIGINAL-colour crop (lossless PNG), the polygon mask,
and metadata incl. the exact corresponding GDS shape (design coords), the
registered image outline, bbox, and the measured brightness/colour stats.

The two thresholds are uncalibrated (no labels yet), so the program prints the
full distribution of both measurements -- calibrate BRIGHT_MAX_EXCESS and
HUE_SPREAD_MAX from that, then re-run.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import edge_anisotropy as ea
import sam_register as sr

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "crystal_db"
REGCACHE = DB / "reg_cache"

# --- constraints (calibrate from the printed distributions) ---
# A crystal is a target if it is (a) not bright, (b) uniform in colour, and (c) the
# right colour. The target colour SHIFTS between samples (chroma 0.37 on seq4050,
# 0.54 on chip3_50x_2), so it is not a fixed threshold -- it is AUTO-DERIVED per
# frame. What is frame-invariant is the colour OFFSET between a monolayer target
# and the bright thick-crystal distractors: the target is more olive (higher
# R/(R+G)) than the bright crystals by a repeatable amount. So per frame we take
# the bright crystals' median chroma and keep crystals in a band above it.
BRIGHT_MAX_EXCESS = 25.0   # interior mean green - local ring background; above => bright crystal
HUE_SPREAD_MAX = 0.020     # spread of median-smoothed R/(R+G); above => multi-colour
TARGET_OFFSET_LO = 0.030   # target chroma is at least this much more olive than the bright crystals.
                           # NOTE: lowering this to recover marginally-olive valid crystals (seq4026)
                           # admits equally-olive FALSE positives (seq4025) -- the two overlap in
                           # chroma/brightness/uniformity, so a looser band can't separate them.
                           # The colour-band misses need a better discriminator, not a threshold move.
TARGET_OFFSET_HI = 0.120   # ...but not more than this (beyond => a different material, e.g. orange)
HUE_MIN_FALLBACK = 0.350   # if a frame has no bright crystals to derive from, use this lower bound

# LUMINANCE regime. The colour selection needs a strong red channel: the validated
# olive frames have mean R ~= 70 (chroma R/(R+G) ~ 0.35, reliable). Some frames are
# imaged with a weak red channel (mean R ~ 13, chroma ~ 0.10) or none at all (green-
# LUT, R ~ 1), and there R/(R+G) is noise -- uniform crystals read as "multi-colour"
# and the chroma band no longer transfers. For those we select on the two signals
# that DO survive: green-channel brightness (thick distractors are much brighter than
# monolayers) and green-channel TONE uniformity (a multi-thickness/broken crystal
# varies in green level; a wrinkled monolayer, median-smoothed, does not). This is the
# user's original "not bright, uniform" pair measured on luminance instead of colour.
# Thresholds are absolute green-level excess over the local ring background, tuned to
# reproduce the hand-counted valid crystals on seq4102/04/08/10 and seq3875/76/78/81.
LOWCOLOR_R_MAX = 40.0     # mean red below this => chroma unreliable, use the luminance path
LUM_EXCESS_MIN = 3.5      # green excess over local background below this => bare substrate.
                          # 3.5 recovers seq3878's faintest valid flake (excess 3.9) while
                          # still rejecting seq4102's bare substrate (3.2); the gap between
                          # a faint monolayer and bare gold here is genuinely this thin.
# SAM-BACKED sites: the multi-tile fitter records which sites had a segmented SAM
# crystal matched onto them (tile["sam_backed"]). For those, DIMNESS (only dimness)
# is forgiven on the luminance path -- a coherent SAM segment centred on a site is
# stronger evidence than a 3-grey-level excess (seq4152's tile-2 crystals sit at
# +2.7/+3.0, below another frame's bare-gold reading of 3.2). A blanket SAM-union
# overlap does NOT work: junk masks covering empty sites admitted false crystals.

# MIRROR-PAIR dim rescue: the design places every triangle/trapezoid with a mirror
# twin. When BOTH members of a pair are in-frame, tone-uniform, and dimly positive,
# that joint evidence beats the per-site confusion limit (each ~10% likely on the
# measured null -> ~1% jointly; seq4152's true pair reads +2.7/+3.0 while seq4102's
# bare-gold trap can't fire -- its partner site is bright). Only dimness is forgiven.
PAIR_DIM_MIN = 2.0        # both pair members must exceed this excess...
MIRROR_PAIRS = [(0, 1), (2, 3), (6, 7), (8, 9), (10, 11), (14, 15), (18, 19),
                (21, 22), (23, 24), (27, 28), (30, 31), (33, 34)]
LUM_BRIGHT_MAX = 30.0     # green excess above this => thick/multilayer distractor
LUM_TONE_SMOOTH = 11      # median window on green to erase wrinkle lines before uniformity
LUM_SPREAD_MAX = 25.0     # legacy p90-p10 tone spread; kept as metadata only -- it counts
                          # illumination GRADIENTS as nonuniformity (user-labelled valid
                          # crystals measured 32-34 while a genuinely patchy one hit 7)
# DETRENDED uniformity (luminance path): fit a plane to the smoothed interior tone,
# judge the residual. A smooth gradient (illumination) detrends to nothing; a real
# multi-tone crystal keeps large off-level patches. Calibrated on user labels:
# valid crystals measure patch<=0.12/sd<=6.3, genuinely patchy ones >=0.34/16.7.
LUM_REL_BRIGHT = 1.8      # frame-relative bright ceiling: a kept crystal may be at most
                          # this multiple of the frame's median kept excess (monolayer
                          # contrast is uniform per frame; ~2x = a thicker crystal)
# NOTE: an absolute floor on the relative ceiling (only cut crystals >15 excess) would
# rescue the validated seq3933 gds17, but it also re-admits a batch of uniform bright
# FALSE positives at the same 10-15 excess -- they overlap gds17 in every feature. So
# the bright-rel coupling stays as-is; gds17 is a known pre-existing loss (dim-median frame).
LUM_PATCH_MAX = 0.38      # largest off-level patch may cover at most this interior fraction.
                          # WRINKLES create off-level patches too (bright fold streaks), and a
                          # wrinkled-but-single-tone crystal is valid (seq3896 gds5/6/7: patch
                          # 0.23-0.34, tsd 8-12); genuinely multi-tone shapes sit higher (patch
                          # 0.4-0.6, tsd 17-24), so the gap is here.
LUM_TONE_SD_MAX = 14.0    # detrended residual std above this => multi-tone (see LUM_PATCH_MAX)
LUM_PATCH_THR = 6.0       # grey levels off the plane that count as "off-level"

MIN_IN_FRAME_FRACTION = 0.70  # the visible crystal must be at least 70% the size of its
                              # GDS shape (user rule). Rejects template positions at the
                              # frame edge whose sliver catches target colour on background
                              # (seq4061 gds30, seq4071 gds25: both 1% in-frame) and
                              # heavily cut-off partials too incomplete to judge.
EDGE_MARGIN = 6            # erode the outline this far before measuring (skip the border)
HUE_SMOOTH = 11           # median-filter window on the chromaticity map (erases thin wrinkles)
BG_INNER, BG_OUTER = 8, 25  # local-background ring: dilate(outer) minus dilate(inner)
MIN_INTERIOR_PX = 300     # need this many in-frame interior px to judge the constraints
PAD = 6                   # px padding around the crop bbox

# Colour 50x brightfield frames only. The green-LUT frames (seq3941/seq3942) are
# excluded: with no colour channel the target-vs-distractor selection can't be
# automated (see the green-LUT notes), and their entries are curated by hand.
DEFAULT_FRAMES = ["seq4045", "seq4046", "seq4050", "seq4057",
                  "seq4052", "chip3_50x", "chip3_50x_2"]

# --- ML RECALL-ASSIST (optional layer, default on) --------------------------
# Rules stay primary; the learned crystal selector (out/crystal_selector.pt) only
# RESCUES shapes the rules REJECTED for appearance reasons -- recovering the dim/
# colour misses the rules drop, without touching rule-admitted decisions. It never
# rescues off-frame slivers or bright/thick distractors. Rescued crystals are
# flagged ml_rescued + needs_review. Set ML_RESCUE=False to fully disable (the
# pipeline then behaves exactly as before). Degrades gracefully if torch or the
# checkpoint is absent.
ML_RESCUE = True
ML_RESCUE_REASONS = ("too-dim", "too-green", "off-colour", "nonuniform")
_ML = {"model": None, "tried": False}


def _get_ml_model():
    if not ML_RESCUE:
        return None
    if _ML["tried"]:
        return _ML["model"]
    _ML["tried"] = True
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import crystal_selector as CS
        m, _ = CS.load_model(ROOT / "out" / "crystal_selector.pt")
        _ML["model"] = m
        print("ML recall-assist: loaded out/crystal_selector.pt")
    except Exception as e:
        print(f"ML recall-assist: disabled ({e})")
        _ML["model"] = None
    return _ML["model"]


def get_registration(stem, green):
    """register() is expensive (tier-2); cache the per-shape affines per frame."""
    cache = REGCACHE / f"{stem}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    reg = sr.register_tiles(green, cache_key=stem)
    d = dict(shape_AB=reg["shape_AB"], low_confidence=reg["low_confidence"],
             anchor=reg["anchor"], mask_coverage=reg["mask_coverage"],
             tiles=reg["tiles"])
    REGCACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(d, open(cache, "wb"))
    return d


def polygon_mask(outline_local, shape):
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in outline_local], fill=255)
    return np.array(im) > 0


def frame_chroma(rgb):
    """Brightness-invariant chromaticity R/(R+G), median-smoothed so thin wrinkle
    lines are erased but large multi-colour regions survive."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    return ndi.median_filter(r / (r + g + 1e-6), HUE_SMOOTH)


def frame_green_tone(rgb):
    """Median-smoothed green channel -- the luminance analogue of frame_chroma. Thin
    wrinkle lines are erased so a wrinkled monolayer reads as uniform, while genuine
    multi-thickness regions survive as a tone step (the luminance-regime uniformity)."""
    return ndi.median_filter(rgb[:, :, 1].astype(np.float32), LUM_TONE_SMOOTH)


def measure(green, chroma_sm, green_sm, interior, ring):
    """Mean chromaticity (target olive vs bright green), colour spread (uniform),
    brightness excess over the local gold ring, legacy tone spread (metadata), and
    the DETRENDED tone uniformity pair (patch fraction, residual sd) -- the
    luminance-regime signal for frames whose red channel is too weak for chroma."""
    bg = float(np.median(green[ring])) if ring.sum() > 50 else float(np.median(green))
    brightness_excess = float(green[interior].mean() - bg)
    vals = chroma_sm[interior]
    mean_chroma = float(vals.mean())
    hue_spread = float(np.percentile(vals, 90) - np.percentile(vals, 10))
    tone = green_sm[interior]
    tone_spread = float(np.percentile(tone, 90) - np.percentile(tone, 10))
    # detrend: least-squares plane over the interior; a smooth illumination
    # gradient vanishes, genuine multi-tone patches survive
    ys, xs = np.nonzero(interior)
    X = np.stack([xs, ys, np.ones_like(xs)], 1).astype(float)
    coef, *_ = np.linalg.lstsq(X, tone, rcond=None)
    resid = tone - X @ coef
    tone_sd = float(resid.std())
    off = np.abs(resid) > LUM_PATCH_THR
    patch = 0.0
    if off.any():
        m = np.zeros(green.shape, bool)
        m[ys[off], xs[off]] = True
        lab, n = ndi.label(m)
        patch = float(ndi.sum(m, lab, range(1, n + 1)).max()) / max(len(ys), 1)
    return brightness_excess, mean_chroma, hue_spread, tone_spread, patch, tone_sd


def merge_records(records, frames):
    """Write the crystal manifest, replacing entries for the given frames. Kept
    separate from build() so a parallel batch can write the manifest ONCE from the
    main process (workers writing it concurrently would race and corrupt it)."""
    man_path = DB / "manifest.json"
    existing = json.load(open(man_path)) if man_path.exists() else []
    existing = [r for r in existing if r["frame"] not in set(frames)]
    DB.mkdir(parents=True, exist_ok=True)
    json.dump(existing + records, open(man_path, "w"), indent=1)


def build(frames, store=True, write_manifest=True):
    polys, fpolys = ea.load_gds_flipped()
    if store:
        (DB / "crystals").mkdir(parents=True, exist_ok=True)
    records, stats = [], []
    for stem in frames:
        img_path = ea.FRAME.parent / f"{stem}.jpg"
        if not img_path.exists():
            print(f"{stem}: image not found, skipping")
            continue
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        H, W = rgb.shape[:2]
        green = rgb[:, :, 1].astype(np.float32)
        chroma_sm = frame_chroma(rgb)
        green_sm = frame_green_tone(rgb)
        reg = get_registration(stem, green)
        low_conf = reg["low_confidence"]
        if low_conf:
            # Do NOT skip low-confidence frames: some register fine but trip the
            # (conservative) confidence flag, and dropping them silently discards
            # real crystals (seq4065's gds29). Extract them, but tag needs_review.
            print(f"{stem}: LOW-CONFIDENCE registration (mask-cov "
                  f"{100*reg['mask_coverage']:.0f}%) -- extracting but tagging crystals needs_review")

        # PASS 1 -- measure every in-frame shape from every registered TILE copy (a
        # frame can straddle two copies of the periodic template). Tiles are ordered
        # primary-first; a later tile's shape that mostly overlaps an already-claimed
        # one (a boundary crystal both copies reach) is skipped, not double-counted.
        tiles = reg.get("tiles") or [dict(shape_AB=reg["shape_AB"])]
        shapes = []
        claimed = np.zeros((H, W), bool)
        for ti, tile in enumerate(tiles, 1):
            backed = set(tile.get("sam_backed", []))
            for gi, ((A, b), fq, poly) in enumerate(zip(tile["shape_AB"], fpolys, polys)):
                A, b = np.asarray(A), np.asarray(b)
                outline = fq @ A.T + b
                fmask = polygon_mask(outline, (H, W))
                interior = ndi.binary_erosion(fmask, iterations=EDGE_MARGIN)
                if interior.sum() < MIN_INTERIOR_PX:
                    continue                             # off-frame or too little to judge
                if ti > 1 and (fmask & claimed).sum() > 0.5 * fmask.sum():
                    continue                             # both tiles reached the same crystal
                ring = (ndi.binary_dilation(fmask, iterations=BG_OUTER)
                        & ~ndi.binary_dilation(fmask, iterations=BG_INNER))
                be, mc, hs, ts, patch, tsd = measure(green, chroma_sm, green_sm, interior, ring)
                claimed |= fmask
                shapes.append(dict(gi=gi, tile=ti, poly=poly, outline=outline,
                                   mc=mc, hs=hs, be=be, ts=ts, patch=patch, tsd=tsd,
                                   backed=(gi in backed),
                                   area=int(fmask.sum()),
                                   in_frac=float(fmask.sum()) / _full_area(outline)))

        # PICK THE REGIME per frame. The colour selection needs a strong red channel;
        # when mean R is weak (or absent) chroma is noise, so select on luminance --
        # a green-brightness band (not bright, not bare substrate) plus green-tone
        # uniformity. Otherwise auto-derive the chroma band from the bright distractors.
        lowcolor = float(rgb[:, :, 0].mean()) < LOWCOLOR_R_MAX
        if lowcolor:
            lo, hi = LUM_EXCESS_MIN, LUM_BRIGHT_MAX
            print(f"{stem}: LOW-COLOUR (mean R {rgb[:,:,0].mean():.0f}) -> luminance path, "
                  f"green-excess band [{lo:.0f}, {hi:.0f}], tone-spread <= {LUM_SPREAD_MAX:.0f}")
        else:
            bright = [s["mc"] for s in shapes if s["be"] > BRIGHT_MAX_EXCESS]
            if bright:
                bc = float(np.median(bright))
                lo, hi = bc + TARGET_OFFSET_LO, bc + TARGET_OFFSET_HI
            else:
                bc, lo, hi = None, HUE_MIN_FALLBACK, 1.0  # no bright crystals to derive from
            print(f"{stem}: {len(bright)} bright distractors, chroma {bc if bc is None else round(bc,3)}"
                  f" -> target band [{lo:.3f}, {hi:.3f}]")

        # mirror-pair dim rescue (luminance path): mark shapes whose mirror twin in
        # the SAME tile is also in-frame, uniform, and dimly positive
        pair_ok = set()
        if lowcolor:
            by_key = {(s["tile"], s["gi"]): s for s in shapes}
            for a, b in MIRROR_PAIRS:
                for ti in {s["tile"] for s in shapes}:
                    sa, sb = by_key.get((ti, a)), by_key.get((ti, b))
                    if sa is None or sb is None:
                        continue
                    if all(s["in_frac"] >= MIN_IN_FRAME_FRACTION
                           and s["patch"] <= LUM_PATCH_MAX and s["tsd"] <= LUM_TONE_SD_MAX
                           and PAIR_DIM_MIN <= s["be"] <= LUM_BRIGHT_MAX
                           for s in (sa, sb)):
                        pair_ok.add((ti, a))
                        pair_ok.add((ti, b))

        # PASS 2a -- per-shape gate decisions (no storing yet)
        for s in shapes:
            gi, mc, hs, be = s["gi"], s["mc"], s["hs"], s["be"]
            in_frame = s["in_frac"] >= MIN_IN_FRAME_FRACTION   # else it is an edge sliver, not a crystal
            if lowcolor:                                  # gate on luminance, no reliable colour
                # dimness (be < lo) is forgiven when the tile fitter matched a SAM
                # crystal onto this site, or when its mirror twin corroborates it;
                # the bright ceiling and (detrended) uniformity still apply
                dim_ok = be >= lo or s["backed"] or (s["tile"], gi) in pair_ok
                uniform = s["patch"] <= LUM_PATCH_MAX and s["tsd"] <= LUM_TONE_SD_MAX
                passed = in_frame and dim_ok and (be <= hi) and uniform
                reason = ("" if passed else " ".join(w for w in [
                          "off-frame" if not in_frame else "",
                          "too-dim" if not dim_ok else "",
                          "bright-lum" if be > hi else "",
                          "nonuniform" if not uniform else ""] if w))
            else:                                         # gate on colour
                passed = in_frame and (be <= BRIGHT_MAX_EXCESS) and (hs <= HUE_SPREAD_MAX) and (lo <= mc <= hi)
                reason = ("" if passed else " ".join(w for w in [
                          "off-frame" if not in_frame else "",
                          "bright-lum" if be > BRIGHT_MAX_EXCESS else "",
                          "multicolour" if hs > HUE_SPREAD_MAX else "",
                          ("too-green" if mc < lo else "off-colour") if not (lo <= mc <= hi) else ""] if w))
            s["_pass"], s["_reason"] = passed, reason

        # PASS 2b -- frame-RELATIVE bright ceiling (luminance path): monolayer
        # contrast is consistent within a frame, so a crystal ~2x brighter than the
        # frame's own kept cluster is a thicker one even when it clears the absolute
        # ceiling (seq3889 gds32 at +17.3 among valid +8/+9). Median needs >= 3 keeps.
        if lowcolor:
            ex = [s["be"] for s in shapes if s["_pass"]]
            if len(ex) >= 3:
                relmax = LUM_REL_BRIGHT * float(np.median(ex))
                for s in shapes:
                    if s["_pass"] and s["be"] > relmax:
                        s["_pass"] = False
                        s["_reason"] = "bright-rel(%.0f>%.0f)" % (s["be"], relmax)

        # PASS 2b.5 -- ML RECALL-ASSIST: rescue rule-rejected in-frame shapes the
        # learned selector calls valid. Only appearance-based rejections of in-frame
        # shapes are eligible (never off-frame slivers or bright/thick distractors),
        # so rescue can only ADD recall, never remove a rule-kept crystal.
        model = _get_ml_model()
        if model is not None:
            import crystal_selector_hybrid as hy
            for s in shapes:
                if s["_pass"] or s["in_frac"] < MIN_IN_FRAME_FRACTION:
                    continue
                if not any(r in s["_reason"] for r in ML_RESCUE_REASONS):
                    continue
                outline = s["outline"]
                rx0 = max(0, int(np.floor(outline[:, 0].min())) - PAD)
                ry0 = max(0, int(np.floor(outline[:, 1].min())) - PAD)
                rx1 = min(W, int(np.ceil(outline[:, 0].max())) + PAD)
                ry1 = min(H, int(np.ceil(outline[:, 1].max())) + PAD)
                if rx1 <= rx0 or ry1 <= ry0:
                    continue
                fmask = polygon_mask(outline, (H, W))
                rgba = np.dstack([rgb[ry0:ry1, rx0:rx1],
                                  (fmask[ry0:ry1, rx0:rx1] * 255).astype(np.uint8)])
                dec = hy.decide(model, rgba, excess=s["be"])
                if dec["valid"]:
                    s["_pass"] = True
                    s["_ml_rescued"] = True
                    s["_ml_p"] = dec["p_ml"]
                    s["_reason"] = "ml-rescued(%.2f)" % dec["p_ml"]

        # PASS 2c -- store
        kept = 0
        for s in shapes:
            gi, mc, hs, be, ts = s["gi"], s["mc"], s["hs"], s["be"], s["ts"]
            # tile 1 keeps the historical id (stem__gdsN); extra template copies get
            # a _tK suffix so a boundary frame can store the same gds index twice.
            sid = f"{stem}__gds{gi}" if s["tile"] == 1 else f"{stem}__gds{gi}_t{s['tile']}"
            passed, reason = s["_pass"], s["_reason"]
            stats.append((stem, gi, len(s["poly"]), mc, hs, be, s["in_frac"], passed, reason))
            if not passed:
                continue
            outline = s["outline"]
            x0 = max(0, int(np.floor(outline[:, 0].min())) - PAD)
            y0 = max(0, int(np.floor(outline[:, 1].min())) - PAD)
            x1 = min(W, int(np.ceil(outline[:, 0].max())) + PAD)
            y1 = min(H, int(np.ceil(outline[:, 1].max())) + PAD)
            fmask = polygon_mask(outline, (H, W))
            crop = rgb[y0:y1, x0:x1]
            rgba = np.dstack([crop, (fmask[y0:y1, x0:x1] * 255).astype(np.uint8)])
            rec = dict(id=sid, frame=stem, gds_index=gi, tile=s["tile"],
                       gds_nsides=len(s["poly"]),
                       gds_polygon=s["poly"].tolist(),
                       image_outline=outline.tolist(),
                       crop_bbox=[x0, y0, x1, y1],
                       outline_in_crop=(outline - [x0, y0]).tolist(),
                       mean_chroma=mc, hue_spread=hs, brightness_excess=be,
                       tone_spread=round(ts, 1), tone_patch=round(s["patch"], 2),
                       tone_sd=round(s["tsd"], 1),
                       regime="luminance" if lowcolor else "colour",
                       target_band=[round(lo, 3), round(hi, 3)],
                       in_frame_fraction=round(s["in_frac"], 3),
                       area_px=s["area"],
                       # surface for human review: extra-copy crystals AND any crystal
                       # admitted through a dim rescue (SAM-backed or mirror-pair)
                       needs_review=low_conf or s["tile"] > 1
                                    or (lowcolor and be < lo)
                                    or s.get("_ml_rescued", False),
                       ml_rescued=s.get("_ml_rescued", False),
                       ml_p=s.get("_ml_p"),
                       image_file=f"crystals/{sid}.png")
            if store:
                Image.fromarray(rgba, "RGBA").save(DB / rec["image_file"])
            records.append(rec)
            kept += 1
        if store:
            # Remove stale crops from earlier runs: a crystal rejected under the
            # current gates must not linger in the folder (seq4061 gds30 survived
            # its manifest removal as an orphan PNG).
            keep_ids = {r["id"] for r in records if r["frame"] == stem}
            for p in (DB / "crystals").glob(f"{stem}__gds*.png"):
                if p.stem not in keep_ids:
                    p.unlink()
                    print(f"{stem}: removed stale crop {p.name}")
        print(f"{stem}: kept {kept} crystals")

    if store and write_manifest:
        merge_records(records, frames)                       # idempotent per frame
    _report(stats)
    print(f"\nstored {len(records)} crystals -> {DB}/  (manifest.json)")
    return records, stats


def _full_area(outline):
    x, y = outline[:, 0], outline[:, 1]
    return max(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))), 1.0)


def _report(stats):
    if not stats:
        return
    mc = np.array([s[3] for s in stats])
    hs = np.array([s[4] for s in stats])
    npass = sum(s[7] for s in stats)
    print("\n--- calibration (all cropped GDS shapes) ---")
    print(f"  {len(stats)} shapes measured, {npass} pass both constraints")
    print(f"  mean_chroma R/(R+G): min {mc.min():.3f}  p50 {np.median(mc):.3f}  "
          f"p90 {np.percentile(mc,90):.3f}  max {mc.max():.3f}   (target band auto-derived per frame)")
    print(f"  hue_spread:          min {hs.min():.3f}  p50 {np.median(hs):.3f}  "
          f"p90 {np.percentile(hs,90):.3f}  max {hs.max():.3f}   (HUE_SPREAD_MAX={HUE_SPREAD_MAX})")
    print("  rejections:")
    for stem, gi, ns, m, h, b, frac, ok, why in stats:
        if not ok:
            print(f"    {stem} gds{gi}({ns}gon): chroma {m:.3f} spread {h:.3f} (ex {b:+.1f})  -> {why}")


def main():
    frames = sys.argv[1:] or DEFAULT_FRAMES
    build(frames, store=True)


if __name__ == "__main__":
    main()
