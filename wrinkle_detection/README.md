# Wrinkle detection

The deployed wrinkle counter: model code, training script, trained weights and
the hand-clicked annotations. All paths resolve inside this folder (override
with the `FLAKE_ROOT` environment variable).

```
pip install -r requirements.txt
```

## What's here

**Model (inference)**
| file | role |
|---|---|
| `wrinkle_native.py` | the deployed model: native-res U-Net, tiled inference, integral-count readout |
| `native_ridge.py` | ridge-instance extraction (overlay crosshair markers) |
| `ridge_count.py` | ridge-tracing primitives `native_ridge` builds on |
| `count_wrinkle_unet.py` | shared readout helpers (`find_peaks`, `integral_count`, `render_debug`) |
| `wrinkle_unet.py` | core building blocks (crop normalization, U-Net definition) |

**Training / verification**
| file | role |
|---|---|
| `train_wrinkle_native.py` | trains the model from scratch on the annotations |
| `verify_accuracy.py` | scores the deployed weights on the held-out test set |

**Data**
| path | what |
|---|---|
| `crystal_db/wrinkle_points.json` | hand-clicked wrinkle labels used for **training** (150 crops, 830 clicks) |
| `crystal_db/wrinkle_points_holdout.json` | hand-clicked labels for the reserved **test** set, seq4213–4276 (136 crops, 653 clicks) |
| `out/wrinkle_native_prod.pt` | **the trained weights** (~1.95M parameters + the config to load them) |
| `crystal_db/crystals/*.png` | the input crystal crops (RGBA; alpha = crystal mask). **Not in git** -- they are cut from the raw frames by `experiments/build_crystal_db.py` |

## Reproducing

Both need the crystal crops in `crystal_db/crystals/`.

**Verify the accuracy claim** (fast -- the headline number):
```
python3 verify_accuracy.py
```
Expected: n=113, MAE 0.98, bias −0.49 on crops the model never trained on.

**Retrain the model from scratch** (~50 min):
```
python3 train_wrinkle_native.py --include-master-holdout --crop-split \
        --epochs 150 --seed 7 --output out/retrained.pt
```
That is the exact recipe behind `wrinkle_native_prod.pt`. Training is
stochastic, so a rerun lands near, not exactly on, the shipped weights (the
recipe measured 1.11 ± 0.09 MAE across three seeds).

## How the count works

The network outputs a density **heatmap**; the count is that heatmap's integral
(summed after zeroing pixels below 5% of the crop's max, to remove background
haze) divided by one Gaussian bump's volume -- so each wrinkle contributes ~1.
The crosshair markers in the overlays come from ridge tracing and are a visual
aid only; they do **not** set the count. Frames seq4213–4276 are the reserved
test set: never train or tune on them.
