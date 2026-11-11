# flake-yield

Per-crystal yield inspection for gold-template-assisted exfoliated MoS2. A
microscope frame goes in; out come the crystals the GDS template says should be
there, each cropped, filtered, and counted for wrinkles.

The full walk-through of every stage is `yield_calculation_doc.pdf`.

## The pipeline

1. **Segment** -- `flakeyield/sam_segment.py`. SAM (ViT-B), zero-shot, on the
   green channel. Masks are cached per frame in `cache/sam/`, so re-processing a
   frame is instant.
2. **Register** -- `experiments/sam_register.py` (anchor selection in
   `auto_anchor.py`, GDS loading and SAM detection in `edge_anisotropy.py`).
   Don't detect the faint crystals; place the known template. Each SAM detection
   is corner-matched to its twin in the x-flipped GDS layout, anchor poses are
   ranked by mask coverage, the winner is refined by ICP, and small guarded
   per-shape affines absorb local strain. Implausible fits are flagged
   low-confidence. Why it works this way: `REGISTRATION.md`.
3. **Extract** -- `experiments/build_crystal_db.py`. Crops every GDS-outlined
   crystal from the original colour frame and keeps it only if it is not a bright
   distractor and is uniform in chromaticity (wrinkled single-colour crystals
   pass, multi-thickness ones do not). Crops land in `crystal_db/crystals/`,
   metadata in `crystal_db/manifest.json`.
4. **Count wrinkles** -- `wrinkle_detection/`. A native-resolution, point-
   supervised density U-Net; the count is the heatmap integral. Held-out
   accuracy n=113, MAE 0.98, bias -0.49. Details, retraining and verification:
   `wrinkle_detection/README.md`.

`pipeline.py` runs stages 1-3 on one image with no GUI imports. `app.py` and
`batch.py` both call it.

## Running it

    pip install -r requirements.txt     # plus torch/torchvision and SAM, see the file
    curl -L -o models/sam_vit_b.pth \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Desktop app -- open one or many frames, browse SAM / registration / crystals /
wrinkles galleries:

    python app.py

Headless, over a folder or glob (e.g. on the DGX; rsync `out/` and `crystal_db/`
back and browse them in the app):

    python batch.py flake_imgs/new_716
    python batch.py 'flake_imgs/new/*.jpg' --workers 4

Wrinkle counts for every stored crystal (fills the app's Wrinkles tab and the
wrinkle columns of `gds/gds_shapes*.csv`):

    python wrinkle_detection/add_wrinkle_counts.py

SAM uses CUDA when available (seconds per frame) and otherwise the CPU (1-2
minutes); Apple MPS is deliberately not used. `FLAKE_DEVICE` overrides.
`FLAKE_GDS` points at a different design file than `gds/tas_v3_design.gds`.

## Layout

| path | what |
|---|---|
| `pipeline.py`, `app.py`, `batch.py` | the pipeline core and its two front ends |
| `flakeyield/sam_segment.py` | SAM segmentation |
| `experiments/` | registration and crystal extraction (the name is historical) |
| `crystal_selector.py`, `crystal_selector_hybrid.py` | optional ML recall-assist for extraction -- active only if `out/crystal_selector.pt` exists, which it does not by default |
| `wrinkle_detection/` | the deployed wrinkle counter, as a self-contained package |
| `gds/` | the GDS design and the per-shape result tables |
| `crystal_db/` | crystal manifest, wrinkle labels, reserved test-frame ranges |
| `flake_imgs/` | the raw frames, as captured. Do not re-save them: the wrinkle signal lives at 3-8 grey levels |
| `out/wrinkle_native_prod.pt` | the deployed wrinkle checkpoint |

Everything else under `out/`, `cache/` and `crystal_db/crystals/` is regenerated
by the commands above.

## What the numbers do and do not mean

Every accuracy figure measures one stage against inputs the earlier stages had
already selected. The wrinkle MAE, for instance, is measured on crystals that
imaging, SAM, registration and extraction had already let through. Treat a
frame-level yield as only as good as its registration: check the overlay in
`out/pipeline_final/`, and treat low-confidence frames as unverified.
