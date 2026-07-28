# GDS registration of MoS2 crystals — process reference

How we located the exfoliated crystals in a 50x frame by registering the known
GDS template to it, rather than detecting the crystals directly. Worked out on
`flake_imgs/new/seq4057.jpg`. This documents the reasoning and the working
recipe so it can be reproduced and extended.

## Why not just detect the crystals

The target crystals are **0.9 sigma per pixel** against the gold background
(measured from hand-marked borders: border ~1.5 grey levels below its
surroundings, against ~1.8 of local scatter). Every appearance-based method
tried failed at this contrast, and the failures were consistent, not incidental:

- Thresholding, top-hat, Frangi ridge filter, watershed, Hough on the raw
  band-pass, bimodality, template-on-intensity — all landed at 0.5-1.4 sigma.
- SAM (ViT-B) could not segment them: at 0.9 sigma there is no *appearance* to
  group. Turning up sensitivity (points_per_side 16->32, lower iou/stability
  thresholds, CLAHE) surfaced more debris, not the faint crystals.

The reason is structural, not a tuning miss. The crystal contrast (~4 levels)
sits at the *same spatial scale* (~200 px) as the background illumination swing
(~4.6 levels), so the region/DC channel is confusion-limited at ~1 sigma and no
estimator escapes it. A human sees the crystals because the visual system
integrates along the whole outline (0.9 sigma/px over a ~400 px edge is ~18
sigma) and uses the global lattice — neither of which a per-pixel operator does.

**Conclusion: don't detect the crystals. Place the known template.** The GDS
says where every crystal is; registration is a 4-6 parameter fit, not a
detection problem. This also fixes a correctness issue: yield = passing /
*designed*, and only the template supplies the denominator (crystals that never
transferred are invisible to any detector but must be counted).

## What the frame is

- 2880x2048, green/olive display LUT. **Only the green channel carries border
  signal**; red is a near-constant ~70 (olive tint), blue is dead (~2, crushed).
- Colour is NOT useless: hue (R/(R+G)) separates crystal *types* — bright
  crystals sit at hue ~0.33, target crystals ~0.37 (~6 sigma). Use colour for
  the bright/multicoloured *classification*, not for border detection (borders
  gain nothing from colour — the LUT channels are correlated).
- JPEG quantization tables are all 1s (quality ~100). Compression is a non-issue.
- Measured noise floor: white ~2.0, not the ~8.4 first (wrongly) measured off a
  patch that straddled structure. Re-measure noise on a *clean* flat patch.

## The GDS

`~/Downloads/tas_v3_design.gds`, cell `SharpTipCompactTiling`: 36 polygons
(8 triangles, 20 quads, 8 hexagons), a compact tiling ~422x445 um. Shapes are
self-similar at 1.5x size steps (e.g. 40x20 and 60x30 um triangles), which
makes scale ambiguous from a single shape — see below. Sharp tips run to ~9.5
degrees. No alignment marks, no text, one layer.

## The registration recipe (what worked)

Coordinate convention: image px, GDS in um, `scale` in px/um.

### 1. Border darkness map
Band-pass the green channel: `gaussian(g,1.5) - gaussian(g,25)`. The sigma=25
subtraction removes the 200 px background (the confound); sigma=1.5 keeps a ~3 px
line. Borders come out dark (negative); use `darkness = max(-bp, 0)`. For
refinement, zero out "loud" regions (bright crystals, off-sample bands, debris
rings) so snaps don't chase them:
```
sm = gaussian(g,15); med = median(sm)
loud = dilate((sm>med+6)|(sm<med-25), 25) | dilated_specks
darkness[loud] = 0
```

### 2. Rotation (automatic, robust)
Edge-orientation histogram of the band-passed image vs the GDS's length-weighted
edge-orientation histogram; or the FFT edge-registration sweep. Both give
**theta ~ 87 deg** independently. (Do NOT trust rotation derived from 2 mark
centroids — approximate marks over a 500 px baseline give ~15 deg error.)

### 3. Scale (from one marked shape)
Have the user mark corners of one clean shape and give one physical dimension.
Here: triangle base = 20 um, measured 274 px -> **13.7 px/um**. The triangle was
the 40x20 um GDS shape (sides 41.2 um -> 564 px confirmed). Note this pins scale
only up to the design's 1.5x self-similarity; the physical dimension breaks that.
`scale = px/um` is a hardware constant (objective + camera pixel); ideally read
it from the microscope calibration once instead of per image.

### 4. THE FLIP (critical, was the blocker)
The exfoliation tape mirrors the pattern when peeled, so the GDS is **reflected
across the x-axis** relative to the image. Apply `F = diag(1,-1)` to the GDS
coords (centred) before rotating. Without the flip, the mark-to-GDS fit residual
was ~130 px (garbage); with it, ~14 px (locked). A least-squares fit will *not*
reliably find the flip on its own from approximate marks (it fit no-flip and
flip about equally) — the user identified it by eye. When in doubt, try both
handednesses and compare residuals, or ask.

### 5. Correspondence + pose (Procrustes over marked corners)
User marked 3 shapes: a triangle (3 dots), a trapezoid/quad (4 dots), and a
triangle base (2 dots). Using the two fully-marked shapes (7 corner points):
- Order each shape's corners CCW (canonical).
- Search all (GDS-triangle, GDS-quad) correspondences x cyclic vertex orderings.
- For each, Procrustes/Kabsch fit (fixed scale) on the flipped GDS; keep min
  residual. This jointly finds *which* GDS shapes the marks are (the correct
  sub-region — the frame shows only ~1/6 of the tiling) and the pose.
- Result: residual 14 px, correspondence L-tri=poly#34, C-quad=poly#28.

The 7-point Procrustes is far more robust than a 2-centroid vector method, which
thrashed.

### 6. Affine refinement (tape strain)
The user noticed alignment error *growing with distance from the anchor* — the
signature of anisotropic scale. Free-scale (similarity) fit still gave 13.71 and
13.6 px residual, but an **affine** fit (6 params, allows shear/anisotropy)
dropped residual to **10 px** and revealed **4.8% anisotropy** (14.19 vs 13.55
px/um along perpendicular axes). This is real, mild non-rigid strain from the
peel. **Fit affine, not rigid+scale.** A single uniform scale is dead-on at the
anchor and drifts ~5% of the distance away.
```
X = hstack([gds_flipped_pts, ones]);  A_all = lstsq(X, mark_pts)
A = A_all[:2].T;  b = A_all[2];  placed = gds_flipped @ A.T + b
```

### 7. Per-shape gentle snap (optional, for the last few px)
From the affine pose, each polygon may nudge <=8 px locally to maximize border
darkness under its edges (on the *masked* darkness map, requiring ~15%
improvement so it doesn't chase debris). Absorbs residual per-crystal
non-rigidity. Do NOT do an unconstrained global darkness-maximization — it chases
the strong bright-crystal/off-sample edges and breaks a good fit.

## Final pose for seq4057
flip across x-axis, affine A with singular values 14.19/13.55 px/um, ~87 deg
rotation, ~4.8% anisotropy, 10 px mark residual. Confirmed visually accurate by
the user (small residual on some crystals, acceptable for now).

## What this unlocks
With the pose known, every crystal's outline is known by construction:
- **Per-crystal masks** read off the registered template (no detection).
- **Wrinkle/fracture detection** (already works at ~2 sigma) run *inside* each
  known mask -> PASS/FAIL per crystal.
- **Yield** = passing / (crystals designed in the frame), denominator from GDS.
- **HQ-SAM** (ViT-B, boundary-refinement, needs a prompt) optionally refines
  exact outlines from the template boxes -- it improves thin boundaries but does
  NOT detect; the template supplies the location.

### Automatic anchor selection (DONE — no hand-picking)

`experiments/auto_anchor.py`. The anchor identity is chosen automatically per
frame by using the whole tiling as the disambiguator:
1. detect bright polygons; enumerate every (detection -> same-vertex GDS shape)
   candidate; anchor-fit each (flipped GDS, PROPER rotation), keep residual < 25px.
2. coarse-rank by GLOBAL edge score (mean gradient ridge under all in-frame
   template edges).
3. refine the top-K with the edge-snap anisotropy step; rank by REFINED edge
   residual. The correct anchor settles tight (6px); the wrong mirror partner
   fights the edges (9.5px).
4. pick the min-residual candidate; report the margin to the runner-up as a
   confidence and flag thin margins (< 25% relative) for a quick eyeball.

Validated on seq4045: auto-selects det#3 -> GDS#10 (the user-confirmed anchor),
6.0px / 2.8% aniso, 37% confidence margin. Renders identically to the manual fit.
The near-translational tiling symmetry means confidence can be thin on some
frames -- hence the flag, not a silent pick.

**Detector: loose SAM, not find_crystals.** The anchor detector is SAM with only
a minimal area filter (`edge_anisotropy.detect_sam`, mirroring
`sam_segment_only.py`): keep every mask 8k px..0.70*frame, drop background, no
elongation/brightness filter. `sam_segment.find_crystals`' material-crystal
filters strip out exactly the clean triangles that make the best anchors (on
seq4046 it left 6 shapes, no triangle; loose SAM gives ~22 incl. the center
triangles). Feed ALL loosely-segmented shapes to corner-matching and let the
global-tiling score select; non-crystal masks just fail CAND_RES_TOL. A fixed
BRIGHT-THRESHOLD detector is NOT exposure-robust (worked on seq4045, found
nothing on seq4046) -- `detect_bright` is kept only for reference.

Validated on seq4046 (a messy frame: heavy debris, off-sample bands): fully
automatic, det#7 -> GDS#34, 6.7px / 3.3% aniso, 35% margin. User-confirmed
"perfect" (2026-07-16). Note the bright distractor crystals ("meant to be
ignored") are correctly NOT covered -- the template lands on the tiling sites
(incl. faint ones), not the bright contamination.

### Standard pipeline: `experiments/sam_register.py` (DONE)

The full refinement chain, folded into one module. Every step earns its place on
a frame that broke the simpler version (seq4045 clean, seq4046 messy, seq4050
huge+strained). `register(green, cache_key)` returns a per-shape affine for every
GDS polygon; the registered outline of crystal i is `fpolys[i] @ A_i.T + b_i`.

1. **Detect** — loose SAM (`ea.sam_shapes`): masks + corner polygons, area filter
   only. See the loose-SAM note above.
2. **Anchor** — corner-match every detection to every same-vertex GDS shape
   (x-flipped, PROPER rotation, fixed 13.7), refine with the global ICP, and rank
   by **MASK-COVERAGE** (fraction of SAM-detected crystals the template lands on),
   NOT edge/corner residual. Structure:
   - **Tier 1: triangles only, single anchor.** Reliable and fast; keeps the good
     frames (seq4045/46/50) exactly.
   - **Tier 2 (only if tier-1 mask-coverage < ~30%): all shapes + multishape.**
     seq4057's triangles are faint/SAM-invisible, so it has no clean triangle
     anchor; the visible LENS crystals (hexagons/quads) are fit *together* into a
     multi-shape joint anchor (like the manual tri+quad 7-point fit), which pins
     the pose and prevents over-warp. Keep tier-2 only if it covers more crystals.
   - Within a tier: coarse-rank seeds by raw-pose mask-coverage, refine the top
     few, then **mask-coverage GATES, edge residual SELECTS** (residual separates
     mirror partners at equal coverage), strain must be plausible.
   - **Why mask-coverage, not edge-coverage/residual:** on the near-periodic
     tiling a one-cell SHIFT keeps a low residual (seq4050 bogus quad 3.1px) and
     an off-frame-PARKED pose games *edge*-coverage (seq4057: 26% edge-cov but
     only 4/36 shapes in-frame). Mask-coverage — did the template land on the
     detected crystals — can't be gamed either way (8% for those wrong poses,
     32-83% for the correct ones).
3. **Global refine** — polarity-aware ICP of all template edges to the SAM
   boundaries, one affine, no anchor pin (windows 100->12 px so a ~70px base
   offset is reachable). Captures the frame-wide anisotropic strain. **Scale and
   anisotropy are clamped every iteration** (scale near the 13.7 px/um hardware
   constant, strain <=8%): a fit must not grow the template off-frame or over-warp
   to fake coverage. Without this a wrong anchor scaled the template up so few
   shapes stayed in-frame and gamed coverage (seq4057).
4. **Per-shape refine** — each shape fits its own small affine (polarity-aware
   mini-ICP) to absorb LOCAL non-affine strain (seq4050 bowtie: top-tri base
   503->564 px onto the real edge at 571; the two bases separate onto their own
   crystals). **Heavily guarded** — unguarded it over-fits faint crystals and
   flips/drifts them onto noise (seq4045 regressed badly). A per-shape fit is
   kept only if it does NOT flip, rotates <8 deg, scales within 0.7..1.4x,
   translates <half the shape size, AND improves that shape's edge coverage;
   else the shape keeps the global affine. So a clean frame where the global fit
   is already perfect is left untouched, while genuine local strain is still
   fixed. **The per-shape fit uses plain least-squares, NO residual trimming:** a
   genuinely off edge (the seq4050 bowtie base, ~48px from its true crystal edge)
   is the LARGEST residual, so trimming would discard exactly the correspondences
   that must move the shape -- leaving the base collapsed on its neighbour.
   Polarity + the coverage-improve accept test are what reject noise here, not
   trimming. (This regressed once when the per-shape fit was folded onto the
   trimmed global fitter; keep it untrimmed.)

**Polarity-aware matching** (used in 3 and 4): a template edge only matches an
image boundary with the crystal (SAM mask) on the edge's INTERIOR side, from the
polygon winding. Without it, adjacent crystals' near-parallel edges swap and two
triangle bases collapse onto one edge. Probe 7 px along the inward normal;
require interior=crystal, exterior=background.

The pipeline reports confidence and flags a frame for a quick eyeball when the
residual margin over the runner-up is thin (<15%) OR the winning pose's
**absolute mask-coverage is below 15%** (too few crystals covered -> wrong pose).

**Status: all four frames auto-register correctly, no hand marks.** seq4045 68%,
seq4046 32%, seq4050 83%, seq4057 62% mask-coverage (manual seq4057 was 69%).
seq4057 -- faint SAM-invisible triangles + heavy circular artifacts -- was the
frame that forced the mask-coverage / scale-clamp / multishape / tiered design;
it now registers via the tier-2 multishape anchor on its visible lens crystals.

## Open items for later
- **Per-frame automation.** DONE via `sam_register.py` (above) — no marks, no
  hand-picking, works clean->messy->strained. Remaining nicety: microscope
  **stage coordinates** (if recorded) would map the frame to the tiling directly
  and remove even the confidence-flag ambiguity.
- **Design/scale per sample.** The pipeline assumes tas_v3_design at 13.7 px/um.
  If a frame's shapes don't match at any consistent scale it is a different
  design (angles vary between samples) and needs its own GDS -- detect this
  (all-triangle free-scale fits disagree) and pick the design from metadata.
- **Anisotropy per frame.** 4.8% strain may vary between chips/frames; measure it
  per frame from >=3 spread anchors, or treat as a stable peel-process constant.
- **Design selection.** Designs vary between samples; the correct GDS must be
  chosen per sample (metadata).
- **Higher-order warp.** If affine leaves some crystals off, thin-plate spline on
  more anchors. 10 px residual here suggests affine suffices for now.
- **Verification discipline.** Never let the template alone decide occupancy —
  each sited crystal must still pass an independent border/darkness check. The
  lattice says where to look, not what is there.

## Scripts
- `experiments/gds_register.py` — FFT edge registration (rotation + coarse
  translation). Set PX_PER_UM.
- `experiments/refine_pose.py` — Procrustes from marks + snap (edit for affine).
- The affine fit and overlays were produced inline; see this file's step 6.
Reference overlays in `out/`: `seq4057_gds_flipfit.png` (14 px lock),
`seq4057_gds_affine.png` (final, 10 px).


## Automated registration by corner matching (in progress)

Replacing hand-marks with auto-detection. The working idea (user's): match
detected shape CORNERS to a database of the GDS shapes, not centroids. Each GDS
shape's vertex geometry is a fingerprint (with scale known, absolute size
separates the self-similar 40x20 vs 60x30 triangles), so aligning >=3 corners of
one clean detection to its GDS twin fixes identity + rotation + position from a
single match. `experiments/shape_match.py`.

VALIDATED: on seq4045, the one shape that corner-matched (a triangle, ~20px
residual) was placed exactly on its real crystal -- identity, angle, position all
correct from corners alone. Centroid matching never achieved this.

### Handedness: RESOLVED (canonical flipped orientation, proper rotation only)

A long detour: fitting the symmetric anchor triangle with a *free* reflection
(`R = U@Vt`, det unconstrained) flip-flopped the whole template unpredictably.
Root cause: an isoceles triangle scores ~10px in MULTIPLE mirror orientations,
so a free fit returns an arbitrary handedness. Composing hacks (reflect-about-
the-anchor) only made the pose un-reproducible.

The fix (user's diagnosis): keep the GDS in ONE canonical orientation while
fitting -- **x-flipped**, `F = diag(1,-1)` applied first (the tape mirror) --
and fit as a **proper rotation, det=+1**. Because the pattern is flipped, the
triangle that actually lands on a detection is the **mirror partner** of the one
you'd naively pick: on seq4045, GDS#11's partner **GDS#10** fits det#3 at 10px
as a proper rotation. Transform is then a fully determined `flip-x -> rotate ->
scale` with NO reflection to choose. Confirmed correct by the user (2026-07-16).

Rule for all future frames: never fit a free reflection; flip the GDS across x
once, fit the mirror-partner shape with det forced to +1. Encoded in
`shape_match.py::match_corners`.

### Far-field from ONE anchor by edge-snapping (no 2nd detection needed)

A single symmetric triangle anchor cannot supply a 2nd asymmetric shape to lock
the anisotropy, and detecting a clean quad on the hard frames failed. The fix
(user's idea): once one crystal is registered, the anchor *predicts* where every
other crystal's edges land to within the few-% tape strain. So don't detect a
2nd shape -- **snap predicted GDS edges to real image edges and refit the affine.**

Recipe (`experiments/edge_anisotropy.py`, validated on seq4045):
1. Anchor: flipped GDS#10 -> det#3, proper rotation, uniform scale (10px).
2. Edge map: gradient magnitude of lightly-smoothed green (Sobel on gaussian 1.2),
   normalized by its 99th pct.
3. For every GDS edge, sample points ~22px apart; at each, search PERPENDICULAR
   to the edge (+/- window) for the max gradient ridge. Accept if response > 0.12.
   These (gds_pt -> found_img_pt) pairs are edge correspondences across the frame.
4. Refit the 6-param affine over all correspondences, with the anchor's 3 corners
   replicated (weight 8) so the solution stays pinned to the confirmed crystal.
   Trim the worst 25% of edge residuals and refit (robust to bad snaps).
5. Iterate windows 60->45->35->28 px (coarse-to-fine).

Result on seq4045: converged to **2.8% anisotropy** (14.08/13.69 px/um), median
edge residual 17->6 px, anchor held ~12px. The faint ~0.9-sigma central crystals
(invisible to any detector) are correctly located by the template. User-confirmed
"perfect". This is the standard far-field step after the anchor fit; it replaces
the need for multiple clean detections. Snap to EDGES (perpendicular ridge), not
blobs. The template still only says *where to look* -- each crystal needs its own
border check before its occupancy counts.

BOTTLENECK: detection quality. Both SAM and a bright-threshold detector gave only
ONE clean corner-match on seq4045 (huge, overexposed, touching crystals -> rough
polygons). One match anchors the pose but it drifts far from the anchor; locking
the full affine (rotation + ~6% anisotropy + translation) needs >=2-3 clean
matches SPREAD across the frame, then a consensus vote (which also resolves the
self-similar-shape aliasing) + affine refine. The pipeline for that is in
shape_match.py; it just needs enough clean detections to feed it.

Next: better shape detection that separates touching crystals and extracts clean
corners; and/or test on cleaner (sliver-style) frames where more shapes detect
cleanly. Corner matching is the right primitive -- proven -- it is starved of
clean inputs on the hardest frames.
