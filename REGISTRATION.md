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

## Open items for later
- **Per-frame automation.** This frame was solved with user marks. For unattended
  runs each frame needs an initial anchor. Best: microscope **stage coordinates**
  (if recorded) map the frame to the tiling directly, zero ambiguity, no marks.
  Otherwise: detect the bright crystals as fiducials (they sit at known GDS
  sites) — needs the anisotropy known first.
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
