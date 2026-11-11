#!/usr/bin/env python3
"""Headless batch runner -- process a folder or glob of images through the full
pipeline (SAM -> registration -> crystal extraction). No GUI, so it runs over SSH
on the DGX. Outputs land in out/sam_cache, out/pipeline_final and crystal_db, so
you can rsync those back and browse them locally.

    python batch.py flake_imgs/new_716
    python batch.py 'flake_imgs/new/*.jpg' --workers 4

SAM uses the GPU automatically when CUDA is available (see flakeyield/sam_segment).
"""

import argparse
import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def resolve(images):
    p = Path(images)
    if p.is_dir():
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in EXTS)
    return sorted(glob.glob(images))


def run_one(src, write_manifest=True):
    import pipeline
    t = time.time()
    try:
        res = pipeline.process_image(src, write_manifest=write_manifest)
        return (src, res, time.time() - t, None)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (src, None, time.time() - t, f"{type(e).__name__}: {e}")


def _log(i, n, r):
    src, res, dt, err = r
    stem = Path(src).stem
    if err:
        print(f"[{i}/{n}] {stem}: ERROR {err}  ({dt:.0f}s)", flush=True)
    else:
        flag = "  *LOW-CONF*" if res["low_conf"] else ""
        print(f"[{i}/{n}] {stem}: {res['nkept']} crystals, "
              f"{100*res['mask_cov']:.0f}% cov, {dt:.0f}s{flag}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run the flake pipeline over a folder/glob of images.")
    ap.add_argument("images", help="a folder, or a quoted glob like 'dir/*.jpg'")
    ap.add_argument("--workers", type=int, default=1,
                    help="process this many frames in parallel (default 1). Uses more CPU; "
                         "with a single GPU the frames share it, so raise cautiously.")
    args = ap.parse_args()

    files = resolve(args.images)
    if not files:
        print(f"no images found at: {args.images}")
        return

    try:
        from flakeyield import sam_segment
        print(f"device: {sam_segment.DEVICE}   images: {len(files)}   workers: {args.workers}", flush=True)
    except Exception:
        print(f"images: {len(files)}   workers: {args.workers}", flush=True)

    t0, results = time.time(), []
    if args.workers > 1:
        # Workers do NOT write the shared manifest (that would race); they only
        # write their own crops and return records. The main process merges the
        # manifest once, below.
        import functools
        import multiprocessing as mp
        ctx = mp.get_context("spawn")          # spawn, not fork -- CUDA-safe
        with ctx.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(
                    functools.partial(run_one, write_manifest=False), files), 1):
                results.append(r); _log(i, len(files), r)
        import pipeline
        recs = [rec for (_, res, _, _) in results if res for rec in res["records"]]
        frames = [res["stem"] for (_, res, _, _) in results if res]
        pipeline.bdb.merge_records(recs, frames)
        print(f"merged manifest: {len(recs)} crystals across {len(frames)} frames", flush=True)
    else:
        for i, src in enumerate(files, 1):
            r = run_one(src, write_manifest=True); results.append(r); _log(i, len(files), r)

    ok = sum(r[1] is not None for r in results)
    low = sum(r[1] is not None and r[1]["low_conf"] for r in results)
    ncry = sum(r[1]["nkept"] for r in results if r[1] is not None)
    print(f"\ndone: {ok}/{len(files)} processed ({low} low-confidence), "
          f"{ncry} crystals stored, in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
