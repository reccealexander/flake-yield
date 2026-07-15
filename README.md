# flake-yield

Yield inspection for gold-template-assisted exfoliated 2D material (MoS2 on
patterned gold, after Yoon et al. and Wu et al.).

A site fails if the crystal carries an artifact, **except** a crack near one of
the template's sharp tips, which is expected and tolerated.

    from flakeyield import inspect_image, Config
    from flakeyield.viz import render

    r = inspect_image("Captured.jpg", Config(exclude_boxes=[(1800, 1990, 2380, 2760)]))
    print(r.summary())        # FAIL (1 wrinkle, ...)
    render(r).save("overlay.png")

Always look at the overlay. Grey was searched and judged clean, blue could not
be read, green is the tolerated tip zone, red is a detected wrinkle.

## Why this is not a machine learning problem

The gold template makes exfoliation deterministic: you already know where the
material is and what N is, because you patterned it. So the two questions a
model like MaskTerial spends its capacity on -- *find the flakes*, *guess the
thickness* -- are already answered. What's left is inspection: did it transfer,
is it intact, is it flat.

That is classical CV, and at 20-50 hand-taken images there is no training set to
do anything else with anyway.

## Measured facts about this imaging regime

All from the reference frame (`Captured.jpg`, 2880x2048). Re-measure before
trusting any of it on a new batch.

| Quantity | Value | Consequence |
|---|---|---|
| Red channel | mean 225, **std 6.2** | Saturated flat. Carries no information. |
| Blue channel | mean 1.9, **59% at zero** | Crushed. Gold absorbs blue; the rest is clipping. |
| Green channel | mean 57, 12% at zero | The only usable channel. `normalize.py` uses it alone. |
| Gold background | G = 109 | The per-image reference. |
| Crystal noise | mean 41.7, **std 18.5** | ~44% per-pixel noise. Defeats local filters. |
| Clipped inside crystal | **22%** | A blind spot. Reported, never scored as clean. |
| Clipping structure | 91% in 3 regions >1000px | Big saturated areas, not speckle -- so maskable. |

Two of these drive the whole design:

**The green channel is the signal.** Red is pinned near saturation and blue is
gone, so the 3-vector of optical contrast that MaskTerial's classifier relies on
collapses to one dimension here.

**Clipping does more than erase.** Where the sensor bottoms out it creates a
hard artificial cliff from G~40 to G=0. Measured: **70% of the strongest Frangi
ridges sat within 20px of a clipped region**, versus 44% of the search area.
The clipping manufactures decoys that outrank the real defect. Hence
`blind_mask`, which masks clipped regions *plus a margin*.

## Why the Radon transform

The obvious tool is a Hessian ridge filter. It does not work here, and the
failure is instructive rather than a tuning problem:

| Approach | Wrinkle vs threshold | Verdict |
|---|---|---|
| Frangi, raw | 0.78x | misses |
| Frangi + Gaussian denoise | 1.00x | misses |
| White/black top-hat | 0.90x / 0.48x | misses, 100% of hits at clip edges |
| Frangi + clip margin masked | 1.33x | marginal, fragments into 634 pieces |
| **Integrate along the line** | **11.8 sigma** | **detects** |

Every local operator looks at a handful of neighbouring pixels, where the
per-pixel SNR is ~1-2, and discards the one property that makes the wrinkle
obvious to a human: it stays coherent over ~1400 pixels. Integrating along its
own direction averages the noise down as sqrt(N) while the line accumulates.
That is the Radon transform, and it recovers the reference wrinkle at 11.1
sigma with the angle correct to 0.2 degrees.

## Pipeline

1. `normalize.py` -- green channel over the gold background mode. The mode
   search **must** skip the clipping floor: G=0 is the single most populated bin
   in the histogram, so a naive `argmax` returns 0 and the normalisation divides
   by zero. (MaskTerial's `calculate_mean_background_values` has this bug for
   this regime: it searches [20, 230] and the true blue mode is below 20.)
2. `segment.py` -- material vs gold by threshold, then erode to an interior.
3. `geometry.py` -- tips from each observed contour, **not** registered from the
   GDS, because the tip angles change between samples.
4. `wrinkles.py` -- Radon detection over the searchable area.
5. `viz.py` -- render the decision for eyeball audit.

## Known limitations

- **Straight wrinkles only.** A strongly curved fold smears across Radon
  offsets. The reference fold is dead straight and folds in exfoliated material
  commonly are, but this is a real gap. Tiling the transform would fix it.
- **Tip detection is unverified.** On the reference frame it does not find the
  template's 9.5-degree tips; it finds corners of the merged material+clipped
  blob. Needs work, and needs images where a single template shape is framed.
- **Sidelobes.** A strong line lifts neighbouring Radon angles. Erasing the
  detected chord between passes helps but does not fully suppress them, so the
  extra detections on the reference frame are probably not real.
- **Calibrated on one image.** `tip_radius_px` and `z_threshold` are guesses
  until there are marked examples to set them from.
- **Slow.** ~75s/frame, dominated by rotations. Fine for 50 images.
- **Burned-in scale bar.** The microscope stamps "20 um" into the pixels and it
  reads as a bright structure on the crystal. Pass `exclude_boxes`.

## The upstream fix

The single highest-leverage change is not in this code. Exposure is destroying
22% of every crystal and manufacturing the false edges that broke the first
three approaches. One afternoon fixing white balance and exposure on the
microscope recovers all of it. That cannot be done in post: clipping is
irreversible.
