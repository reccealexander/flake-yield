#!/usr/bin/env python3
"""Train the NATIVE-resolution wrinkle-counting U-Net.

Fixes the letterbox resolution pathology of the 128-px production model by
training on random 256-px NATIVE-resolution patches (see ``wrinkle_native``).

Key properties
--------------
  * FRAME-GROUPED train/val split: every crop of a given frame stays on the same
    side, so a frame's crystals never straddle the split. ~20% of frames go to
    val. Fixed seed.
  * Reserved holdout: frames seq4213 and above are dropped entirely -- never
    trained, tuned, or evaluated on here.
  * Honest val metric: masked-MSE val loss every epoch, plus val count-MAE
    (integral and peak) computed via FULL TILED INFERENCE on the val crops every
    ``--eval-every`` epochs. Best checkpoint is chosen by tiled integral
    count-MAE (the metric we actually care about), not patch MSE.
  * Device: CUDA -> MPS -> CPU, but MPS is sanity-checked with a tiny
    forward/backward first and we fall back to CPU if it NaNs or errors.

Just works:  python3 train_wrinkle_native.py
"""

import argparse
import os
import random
import re
import sys

import numpy as np

def _find_data_root():
    """Locate the data root (the directory containing crystal_db/).

    Works both in the repo (scripts one level down, data at the parent) and in
    a self-contained bundle (data sitting beside the scripts). Override with
    the FLAKE_ROOT environment variable."""
    d = os.path.dirname(os.path.abspath(__file__))
    for cand in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(cand, "crystal_db")):
            return cand
    return os.path.dirname(d)


ROOT = os.environ.get("FLAKE_ROOT") or _find_data_root()
DEFAULT_POINTS = os.path.join(ROOT, "crystal_db", "wrinkle_points.json")
DEFAULT_CROPS = os.path.join(ROOT, "crystal_db", "crystals")
DEFAULT_OUTPUT = os.path.join(ROOT, "out", "wrinkle_native.pt")

HOLDOUT_MIN_SEQ = 4213      # frames seq4213+ are the reserved holdout


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=ROOT, help="main checkout (data lives here)")
    p.add_argument("--points", default=None)
    p.add_argument("--crops-dir", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--sigma", type=float, default=None,
                   help="native bump sigma (default wrinkle_native.NATIVE_SIGMA)")
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--patches-per-crop", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-angle", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-master-holdout", action="store_true",
                   help="keep master-confirmed seq4213+ crops in training "
                        "(match production's data budget; clean-subset eval only)")
    p.add_argument("--crop-split", action="store_true",
                   help="production-style crop-level split instead of "
                        "frame-grouped (full coverage; holdout-eval only)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=3,
                   help="run full tiled val inference every N epochs")
    p.add_argument("--device", default=None)
    return p.parse_args()


def frame_of(cid):
    return cid.split("__")[0]


def seq_of(cid):
    m = re.match(r"seq(\d+)", frame_of(cid))
    return int(m.group(1)) if m else None


def is_holdout(cid):
    s = seq_of(cid)
    return s is not None and s >= HOLDOUT_MIN_SEQ


def crop_level_split(items, val_frac, seed):
    """PRODUCTION-STYLE split: shuffle individual crops; ~val_frac to val.

    Crops from one frame can land on both sides, so training touches nearly
    every frame's imaging conditions (full split coverage) -- but internal
    val metrics are flattered (val crops have same-frame siblings in train).
    Judge such models on the seq4213+ holdout only. Same return shape as
    frame_grouped_split; the frame lists are derived (and may overlap).
    """
    import random as _random
    idx = list(range(len(items)))
    _random.Random(seed).shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    val_i = set(idx[:n_val])
    train = [it for i, it in enumerate(items) if i not in val_i]
    val = [it for i, it in enumerate(items) if i in val_i]
    def frames(sel):
        return sorted({frame_of(os.path.splitext(os.path.basename(p))[0])
                       for p, _ in sel})
    return train, val, frames(train), frames(val)


def frame_grouped_split(items, val_frac, seed):
    """Split (path, pts) items by FRAME; ~val_frac of frames to val. Fixed seed.

    Returns (train_items, val_items, train_frames, val_frames).
    """
    def cid_of(path):
        return os.path.splitext(os.path.basename(path))[0]
    frames = sorted({frame_of(cid_of(p)) for p, _ in items})
    rng = random.Random(seed)
    rng.shuffle(frames)
    n_val = max(1, int(round(len(frames) * val_frac)))
    val_frames = set(frames[:n_val])
    train, val = [], []
    for p, pts in items:
        (val if frame_of(cid_of(p)) in val_frames else train).append((p, pts))
    return train, val, sorted(set(frames) - val_frames), sorted(val_frames)


def sanity_check_device(torch, device, base):
    """Tiny forward/backward on ``device``; return True if finite, else False."""
    import wrinkle_unet as wu
    try:
        m = wu.build_unet(in_ch=1, base=base).to(device)
        x = torch.randn(1, 1, 64, 64, device=device)
        y = m(x).sum()
        y.backward()
        ok = torch.isfinite(y).item() and all(
            torch.isfinite(p.grad).all().item()
            for p in m.parameters() if p.grad is not None)
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print("  device %s failed sanity check: %s" % (device, exc))
        return False


def pick_device(torch, requested, base):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps and mps.is_available():
        print("Sanity-checking MPS with a tiny forward/backward...")
        if sanity_check_device(torch, torch.device("mps"), base):
            print("  MPS OK.")
            return torch.device("mps")
        print("  MPS produced NaN/error -> falling back to CPU.")
    return torch.device("cpu")


def main():
    args = parse_args()
    args.points = args.points or os.path.join(args.root, "crystal_db",
                                              "wrinkle_points.json")
    args.crops_dir = args.crops_dir or os.path.join(args.root, "crystal_db",
                                                     "crystals")
    args.output = args.output or os.path.join(args.root, "out",
                                              "wrinkle_native.pt")

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not available. Install it and rerun.")
        sys.exit(1)
    from torch.utils.data import DataLoader

    import wrinkle_unet as wu
    import wrinkle_native as wn
    import count_wrinkle_unet as cw

    sigma = args.sigma if args.sigma is not None else wn.NATIVE_SIGMA
    min_distance = max(1, int(round(sigma)))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    all_items, missing = wu.build_index(args.points, args.crops_dir,
                                        confirmed_only=True)
    if missing:
        print("Warning: %d confirmed ids have no crop PNG on disk" % missing)
    # drop reserved holdout frames -- unless matching the production model's
    # data budget: --include-master-holdout keeps the seq4213+ crops that are
    # confirmed in the MASTER points file (the 27 the production checkpoint
    # trained on). Any model trained this way must be evaluated ONLY on the
    # clean holdout subset (crops absent from the master file).
    if args.include_master_holdout:
        items = list(all_items)
        print("Confirmed crops: %d total, master seq4213+ INCLUDED "
              "(clean-subset eval only!)" % len(items))
    else:
        items = [(p, pts) for p, pts in all_items
                 if not is_holdout(os.path.splitext(os.path.basename(p))[0])]
        n_hold = len(all_items) - len(items)
        print("Confirmed crops: %d total, %d dropped as holdout (seq>=%d), %d usable"
              % (len(all_items), n_hold, HOLDOUT_MIN_SEQ, len(items)))

    split_fn = crop_level_split if args.crop_split else frame_grouped_split
    train_items, val_items, train_frames, val_frames = split_fn(
        items, args.val_frac, args.seed)
    mode = ("CROP-level (production-style)" if args.crop_split
            else "frame-grouped")
    print("Split mode: %s" % mode)
    print("Frame-grouped split: %d train crops / %d val crops "
          "(%d train frames / %d val frames)"
          % (len(train_items), len(val_items), len(train_frames), len(val_frames)))
    print("Val frames: %s" % ", ".join(val_frames))

    DatasetCls = wn.make_patch_dataset_class()
    train_ds = DatasetCls(train_items, patch=args.patch, sigma=sigma,
                          augment=True, seed=args.seed,
                          patches_per_crop=args.patches_per_crop,
                          max_angle=args.max_angle)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)

    device = pick_device(torch, args.device, args.base)
    print("Device: %s" % device)

    model = wu.build_unet(in_ch=1, base=args.base, out_ch=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    volume = wu.bump_volume(sigma)
    meta = {"base": args.base, "patch": args.patch, "sigma": sigma,
            "min_distance": min_distance, "device": str(device),
            "norm": "green_2_98_inside_mask_whole_crop"}

    def tiled_val():
        """Full tiled inference on val crops -> (masked-MSE, int-MAE, peak-MAE)."""
        model.eval()
        mse_sum, mse_n = 0.0, 0
        int_err, peak_err, n = 0.0, 0.0, 0
        for path, pts in val_items:
            gray, mask = wn.load_crop_native(path)
            hm = wn.predict_heatmap_native(model, meta, gray, mask)
            tgt = wu.make_target(pts, gray.shape, mask, sigma)
            m = mask > 0
            if m.sum() > 0:
                mse_sum += float(((hm - tgt) ** 2)[m].mean())
                mse_n += 1
            res = wn.count_from_heatmap_native(
                hm, sigma, min_distance=min_distance)
            true = len(pts)
            int_err += abs(res["integral_count"] - true)
            peak_err += abs(res["peak_count"] - true)
            n += 1
        return (mse_sum / max(1, mse_n), int_err / max(1, n),
                peak_err / max(1, n))

    best_mae = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for img, tgt, msk in train_loader:
            img, tgt, msk = img.to(device), tgt.to(device), msk.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = ((pred - tgt) ** 2 * msk).sum() / msk.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                print("  non-finite loss at epoch %d -- aborting" % epoch)
                sys.exit(2)
            loss.backward()
            optimizer.step()
            running += loss.item() * img.size(0)
            n += img.size(0)
        scheduler.step()
        train_loss = running / max(1, n)

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs) \
            or (epoch == 1)
        if do_eval:
            val_mse, int_mae, peak_mae = tiled_val()
            history.append((epoch, train_loss, val_mse, int_mae, peak_mae))
            marker = ""
            if int_mae < best_mae:
                best_mae = int_mae
                torch.save({
                    "model_state": model.state_dict(),
                    "base": args.base,
                    "patch": args.patch,
                    "sigma": sigma,
                    "min_distance": min_distance,
                    "norm": "green_2_98_inside_mask_whole_crop",
                    "val_masked_mse": val_mse,
                    "val_int_count_mae": int_mae,
                    "val_peak_count_mae": peak_mae,
                    "val_frames": val_frames,
                    "train_frames": train_frames,
                    "holdout_min_seq": HOLDOUT_MIN_SEQ,
                    "args": vars(args),
                }, args.output)
                marker = "  <- saved best"
            print("epoch %3d  train %.6f  val_mse %.6f  int-MAE %.3f  "
                  "peak-MAE %.3f%s" % (epoch, train_loss, val_mse, int_mae,
                                       peak_mae, marker))
        else:
            print("epoch %3d  train %.6f" % (epoch, train_loss))

    print("\nBest val integral count-MAE: %.3f  (checkpoint: %s)"
          % (best_mae, args.output))
    print("Done.")


if __name__ == "__main__":
    main()
