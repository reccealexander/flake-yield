# flake-yield

Per-crystal yield inspection for gold-template-assisted exfoliated 2D material
(MoS2 on patterned gold, after Yoon et al. and Wu et al.).

**The rule:** a successful crystal is entirely uniform. Any fracture line means
failure. Missing material at an edge is not a defect.

    python run.py '~/Downloads/seq*.jpg' --overlays out/ --csv out/yield.csv

    seq3941.jpg: 2/2 crystals pass
        crystal 0: PASS  [171182 px, 0 fracture(s)]
        crystal 1: PASS  [170679 px, 0 fracture(s)]
    seq3942.jpg: 0/1 crystals pass
        crystal 0: FAIL  [272452 px, 3 fracture(s)]

Always look at the overlay. Blue outlines what was searched; red marks a
detected fracture.

## Scope, honestly

Calibrated and verified on **four crystals** across three frames: `seq3941` (two
clean), `seq3942` (one fractured), `chip3_50x_hc_2` (one clean, carrying
specular artifacts). `tests/test_separation.py` pins that result.

It is **not** verified on the raw brightfield series (`Captured.jpg`), which is
a different problem: material reads *darker* than background there rather than
brighter, 22% of each crystal is clipped to zero, and per-pixel noise is 44%
against 10.6% here. Expect to rebuild `segment.py` for that regime, not tune it.

## Why not machine learning

The gold template makes exfoliation deterministic: you already know where the
material is and what N is, because you patterned it. The two questions a model
like MaskTerial spends its capacity on -- *find the flakes*, *guess the
thickness* -- are already answered. What remains is inspection.

And at 20-50 hand-taken images there is no training set to do otherwise.

## Why a top-hat and not a ridge filter

Frangi is the textbook choice for finding curvilinear structure. It does not
work here, and the failure is structural rather than a tuning miss. Scored
against a hand-annotated crack:

| Detector | AUC |
|---|---|
| Frangi (raw) | 0.451 |
| Frangi (smoothed) | 0.424 |
| Sato / Meijering | 0.507 / 0.419 |
| **chance** | **0.500** |

Two separate reasons, both worth knowing:

**A crack is sometimes an edge, not a ridge.** On `Captured_1` the annotated
"crack" turned out to be the seam between clean material and damaged material --
the two sides differ by 15 grey levels and 250x in clipped fraction. Frangi
rejects step edges by construction, so it scored *below* chance. Match the
operator to what the defect physically is.

**Fractures live at ~1-2 per-pixel SNR.** Frangi's output on `seq3942` is a
field of noise worms with the real lines buried in it (see
`out/diagnostics/seq3942_frangi_raw.png`). Speckle fakes a ridge inside any
6-pixel neighbourhood, which is all a Hessian filter ever sees.

What makes these fractures obvious to a human is not local contrast --- it is
**coherence**: they stay connected for hundreds of pixels. So the test that
works is not "is this pixel ridge-like" but "is this a *connected, elongated*
run of pixels brighter than their surroundings". Noise fakes the first
constantly and the second essentially never. The size and elongation filters in
`fractures.py` are not tidying-up; they are the detector.

## The margin

The detection threshold can move over a 2x range without either frame changing
verdict:

| threshold mult | seq3941 (clean) | seq3942 (fractured) | chip3 (clean, artifacts) |
|---|---|---|---|
| 1.6 | 0 curves | 3 | **1 -- false positive** |
| 1.8 | 0 | 3 | 0 |
| 2.2 (default) | 0 | 3 | 0 |
| 2.6 | 0 | 2 | 0 |
| 3.0 | 0 | 2 | 0 |
| 3.5 | 0 | **0 -- missed** | 0 |

The usable window is 1.8-3.0, and that window -- not the default constant -- is
the reason to believe any of this.

Watch how it has moved. On two frames it looked like 1.4-3.0. Adding chip3, which
carries specular artifacts, lifted the floor to 1.6. Swapping in SAM, whose
boundaries include the crystal tips the threshold mask truncated, lifted it to
1.8 -- leaving the old default sitting exactly on the edge. **Every time more
information has arrived, the slack has shrunk.** The earlier figures were not
wrong so much as measured against too little. Treat this as an upper bound.

## The searched fraction

The boundary margin is eroded away before looking for fractures, because the
crystal's own edge is a bright line that reads as a defect. That erosion is not
free -- it is crystal never inspected, and these slivers are narrow:

| margin | verdicts | searched (seq3941) |
|---|---|---|
| 3 | **false positive** on a clean crystal | 92% |
| 5 | correct | 87% |
| 8 (default) | correct | 80% |
| 10 | correct | 75% |
| 15 (old) | correct | **64%** |

15 was inherited from the threshold-mask era, where the boundary was fuzzy and
the margin was compensating for it. SAM's boundary is crisp. At 15, **36% of
seq3941's crystals were never examined -- including the sharp tips**, which are
stress concentrators and therefore where cracks start. The worst possible place
to be blind.

A verdict is only ever "clean over the fraction we looked at". `Inspection`
reports it; the overlay draws it (blue outlines the searched interior, which is
deliberately *inside* the crystal boundary).

## Measured facts

Re-measure before trusting any of these on a new batch.

| Quantity | Green LUT (`seq*`) | Raw brightfield (`Captured.jpg`) |
|---|---|---|
| Signal channel | green (R mean 1.8, B mean 2.1) | green (R std 6.2, B 59% at zero) |
| Material polarity | **brighter** than background | **darker** than background |
| Clipping | **none** (G spans 98-203) | **22% of each crystal** |
| Per-pixel noise | ~10.6% | ~44% |
| Crystal/background step | ~20 levels vs ~15 noise | -- |

## Pipeline

1. `normalize.py` -- take green; flag clipped pixels as unreadable.
2. `sam_segment.py` -- SAM (Segment Anything), zero-shot. No training, no labels.
   It separates touching crystals, which Otsu and watershed both cannot, and
   sizes seq3941's two identical slivers to within 0.3% against 11% for
   thresholding. SAM has no concept of "crystal", so this module selects among
   its masks on four physical criteria: area (rejects the ~98% background mask),
   elongation >= 2.5 (rejects chip3's 248x191 specular blob at 1.3), brightness
   above the frame median (rejects the dark off-sample bands), and a max frame
   fraction.
3. `fractures.py` -- TV denoise, white top-hat, then keep only connected
   elongated runs.
4. `viz.py` -- render the verdict for eyeball audit.

`segment.py` is the previous threshold pipeline, still selectable via
`Config(use_sam=False)`. It needs no 358MB checkpoint, so it is a real fallback
rather than dead code -- but it merges touching crystals and truncates tips.

## Traps worth remembering

Each of these cost real time and will bite again:

- **Exclude off-sample pixels before thresholding.** `seq3942` has a dark strip
  at one edge; left in, Otsu splits dark-vs-everything at -19 and the mask
  swallows 88% of the frame instead of finding the crystal.
- **Do not morphologically close the crystal mask.** Adjacent slivers sit
  ~20-40px apart; a 21x21 closing welds them into one 432k px blob and you lose
  per-crystal yield.
- **Erode the boundary before looking for fractures.** The crystal edge is a
  strong bright line and reads as a defect. An earlier version's "9x
  separation" was measuring boundary sharpness, nothing more.
- **A black top-hat is not a free win.** Added to catch dark fractures, it
  produced a false positive on a clean crystal and found nothing extra on the
  fractured one. Bright-only is deliberate; dark fractures are currently missed.
- **Dust is bright, thin and elongated too.** A specular streak satisfies every
  geometric test a fracture does, and chip3 was failed on one. Only the
  brightness cap separates them: real fractures measure +5.4 to +6.9 over the
  local median, that streak +19.1. A fracture is subtle topography in an
  atomically thin sheet; a scatterer sitting on the gold is not.
- **Check mask coverage in the overlay, not the pixel count.** Three crystals
  all came out near 130k px under `flatten_sigma=120`, which looked like
  reassuring agreement between template shapes. It was an artifact: the
  threshold truncates every mask at a similar area regardless of the crystal's
  true extent. seq3942's mask covered a lobe of its lens and the fractures were
  found by luck. Equal-looking areas are not evidence; the overlay is.

## Known gaps

- **Bright fractures only** (see above).
- **Fragments, not networks.** It detects ~2 segments of a visibly larger
  network. Fine for pass/fail, useless as a quantitative fracture-length metric.
- **Four crystals of ground truth.** Three clean, one fractured. Encouraging,
  not proof -- and every frame added so far has found a new failure mode.
- **The brightness cap is set against one artifact and three fractures.** A
  genuinely bright fracture would be rejected by it.
- **SAM runs on CPU**, minutes per frame rather than seconds: its automatic mask
  generator builds float64 point tensors and Apple's MPS has no float64. At 50
  frames that is an hour-plus.
- **The brightness rule assumes the frame median is background.** True here
  (crystals cover 15-30%), but it inverts on a zoomed-out mosaic packed with
  crystals -- exactly the eventual whole-chip goal.
- **Segmentation untested on frames with many shapes.**
