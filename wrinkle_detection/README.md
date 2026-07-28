# Wrinkle detection — reproducibility package

Everything needed to reproduce the wrinkle results: the model code, the
training script, the trained weights, the hand-labelled annotations, and the
input crystal crops. Self-contained — all paths resolve inside this folder
(override with the `FLAKE_ROOT` environment variable).

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

**Pipeline / training / verification**
| file | role |
|---|---|
| `add_wrinkle_counts.py` | regenerates `out/wrinkle_counts.csv`, the overlay PNGs, and the GDS columns |
| `calibrate_gds_csvs.py` | aggregate de-bias + presence threshold → the `_cal` columns |
| `train_wrinkle_native.py` | trains the model from scratch on the annotations |
| `frame_conditions.py` | imaging-condition clustering (only for `--stratified-split` training) |
| `verify_accuracy.py` | scores the deployed weights on the held-out test set |

**Data**
| path | what |
|---|---|
| `crystal_db/crystals/*.png` | the input crystal crops (RGBA; alpha = crystal mask) |
| `crystal_db/wrinkle_points.json` | hand-clicked wrinkle labels used for **training** (150 crops, 830 clicks) |
| `crystal_db/wrinkle_points_holdout.json` | hand-clicked labels for the reserved **test** set, seq4213–4276 (136 crops, 653 clicks) |
| `gds/gds_shapes.csv` | designed-shape table the aggregate columns are written into |
| `out/wrinkle_native_prod.pt` | **the trained weights** (~1.95M parameters + the config to load them) |
| `out/wrinkle_calibration.json` | the fitted de-bias scale and presence threshold |
| `expected_results/` | the delivered outputs, to diff your regenerated ones against |

## Three things you can reproduce

**1. Verify the accuracy claim** (fast — the headline number):
```
python3 verify_accuracy.py
```
Expected: n=113, MAE 0.98, bias −0.49 on crops the model never trained on.

**2. Regenerate the counts + overlay images** (~45 min on CPU for 873 crops):
```
python3 add_wrinkle_counts.py --root .
```
Writes `out/wrinkle_counts.csv`, `out/wrinkle_overlays/*.png`, and the wrinkle +
`_cal` columns in `gds/gds_shapes*.csv`. Diff against `expected_results/`.

**3. Retrain the model from scratch** (~50 min; reproduces the weights):
```
python3 train_wrinkle_native.py --include-master-holdout --crop-split \
        --epochs 150 --seed 7 --output out/retrained.pt
```
That is the exact recipe behind `wrinkle_native_prod.pt`. Note training is
stochastic — a rerun lands near, not exactly on, the shipped weights (the
recipe measured 1.11 ± 0.09 MAE across three seeds).

## How the count works

The network outputs a density **heatmap**; the count is that heatmap's integral
(summed after zeroing pixels below 5% of the crop's max, to remove background
haze) divided by one Gaussian bump's volume — so each wrinkle contributes ~1.
The crosshair markers in the overlays come from ridge tracing and are a visual
aid only; they do **not** set the count. Frames seq4213–4276 are the reserved
test set and are excluded from production counting by default.
