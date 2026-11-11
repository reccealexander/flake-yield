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

`pipeline.py` runs stages 1-3 on one image with no GUI imports; `batch.py` runs
it over many.

## Running it

    pip install -r requirements.txt     # plus torch/torchvision and SAM, see the file
    curl -L -o models/sam_vit_b.pth \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Two inputs are not in this repository and have to be supplied locally:

- **The GDS design**, at `gds/tas_v3_design.gds` (or point `FLAKE_GDS` at it).
  Registration cannot run without it.
- **The microscope frames.** Use them as captured and do not re-save them:
  the wrinkle signal lives at 3-8 grey levels, and a lossy re-encode destroys
  it. Processed frames are kept in `flake_imgs/new/`.

Over a folder or glob (headless, so it also runs on the DGX; rsync `out/` and
`crystal_db/` back to browse the results):

    python batch.py path/to/frames/
    python batch.py 'path/to/frames/*.jpg' --workers 4

SAM uses CUDA when available (seconds per frame) and otherwise the CPU (1-2
minutes); Apple MPS is deliberately not used. `FLAKE_DEVICE` overrides.

## Layout

| path | what |
|---|---|
| `pipeline.py`, `batch.py` | the pipeline core and its batch runner |
| `flakeyield/sam_segment.py` | SAM segmentation |
| `experiments/` | registration and crystal extraction (the name is historical) |
| `wrinkle_detection/` | the deployed wrinkle counter |
| `crystal_db/` | hand-clicked wrinkle labels; the pipeline writes the crystal crops and `manifest.json` here |
| `out/wrinkle_native_prod.pt` | the deployed wrinkle checkpoint |

Everything under `out/`, `cache/` and `crystal_db/` other than those files is
written by the pipeline and is not tracked.

## What the numbers do and do not mean

Every accuracy figure measures one stage against inputs the earlier stages had
already selected. The wrinkle MAE, for instance, is measured on crystals that
imaging, SAM, registration and extraction had already let through. Treat a
frame-level yield as only as good as its registration: check the overlay in
`out/pipeline_final/`, and treat low-confidence frames as unverified.
