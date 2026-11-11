#!/usr/bin/env python3
"""Headless processing pipeline -- the core batch.py runs on every frame.

One image in -> SAM masks + registration overlay + extracted crystals, written to
the project's output folders. No GUI dependencies, so it runs over SSH / in a
container on the DGX exactly as it does on a desktop.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

import edge_anisotropy as ea          # noqa: E402
import sam_register as sr             # noqa: E402
import build_crystal_db as bdb        # noqa: E402

IMG_DIR = ROOT / "flake_imgs" / "new"
OUT = ROOT / "out"
SAMCACHE = OUT / "sam_cache"
PIPELINE_FINAL = OUT / "pipeline_final"          # clean registration overlays, one per frame
CRYSTALS = ROOT / "crystal_db" / "crystals"
THUMBS = OUT / "thumb_cache"                      # small disk thumbnails for browsing results

PALETTE = np.array([
    [230, 60, 60], [60, 160, 240], [90, 210, 90], [250, 170, 40],
    [190, 110, 230], [70, 220, 200], [240, 120, 180], [180, 190, 60],
    [120, 130, 250], [230, 90, 130], [80, 200, 140], [220, 150, 90],
], np.float32)


def _n8(x, lo, hi):
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def render_sam(rgb, masks, path):
    g = rgb[:, :, 1].astype(np.float32)
    base = np.repeat(_n8(g, np.percentile(g, 1), np.percentile(g, 99))[:, :, None], 3, 2).astype(np.float32)
    for i, m in enumerate(masks):
        m = np.asarray(m, bool)
        base[m] = 0.55 * base[m] + 0.45 * PALETTE[i % len(PALETTE)]
    im = Image.fromarray(base.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for i, m in enumerate(masks):
        ys, xs = np.where(np.asarray(m, bool))
        if len(xs):
            dr.text((xs.mean(), ys.mean()), str(i), fill=(255, 255, 255))
    dr.rectangle([0, 0, im.width, 30], fill=(0, 0, 0))
    dr.text((8, 9), f"{len(masks)} SAM masks", fill=(240, 240, 240))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def process_image(src_path, log=lambda m: None, write_manifest=True):
    """Run the full pipeline on one image. Writes the three outputs; returns info.

    write_manifest=False skips the manifest write (the crystal crops are still
    written) so a parallel batch can merge the manifest once from the main process
    -- the returned dict includes this frame's `records` for that merge.
    """
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(src_path).stem)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    dst = IMG_DIR / f"{stem}.jpg"
    src = Path(src_path)
    if src.resolve() != dst.resolve():
        # quality=95 and no chroma subsampling, NOT PIL's default 75: these are
        # already-JPEG microscope frames, so this is a second lossy pass over a
        # signal that lives at 3-8 grey levels. When the source IS the frame in
        # IMG_DIR, skip the copy entirely rather than re-encode it in place.
        Image.open(src).convert("RGB").save(dst, quality=95, subsampling=0)
    rgb = np.asarray(Image.open(dst).convert("RGB"))
    green = rgb[:, :, 1].astype(np.float32)

    sam_cached = ((ea.SAM_CACHE_DIR / f"sam_{stem}.npz").exists()
                  or (Path("/tmp") / f"sammask_{stem}.npy").exists())
    log("SAM already cached — skipping segmentation, registering…" if sam_cached
        else "Running SAM segmentation + registration…")
    reg = sr.register_tiles(green, cache_key=stem)

    render_sam(rgb, reg["masks"], SAMCACHE / f"{stem}_cachedmasks.png")
    PIPELINE_FINAL.mkdir(parents=True, exist_ok=True)
    sr._render(green, reg, PIPELINE_FINAL / f"{stem}_registered.png")

    bdb.REGCACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(shape_AB=reg["shape_AB"], low_confidence=reg["low_confidence"],
                     anchor=reg["anchor"], mask_coverage=reg["mask_coverage"],
                     tiles=reg["tiles"]),
                open(bdb.REGCACHE / f"{stem}.pkl", "wb"))
    log("Extracting crystals…")
    records, _ = bdb.build([stem], store=True, write_manifest=write_manifest)
    nkept = sum(r["frame"] == stem for r in records)

    return dict(stem=stem, low_conf=reg["low_confidence"], mask_cov=reg["mask_coverage"],
                anchor=reg["anchor"], nkept=nkept, records=records)
