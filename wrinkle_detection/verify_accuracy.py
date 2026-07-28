#!/usr/bin/env python3
"""Reproduce the deployed model's accuracy on the held-out test set.

Scores out/wrinkle_native_prod.pt against the CLEAN holdout: the crops in
crystal_db/wrinkle_points_holdout.json that were confirmed by hand AND are not
also present as seq4213+ entries in the master training annotations (those few
leaked into an early training pool, so they are excluded to keep the test
honest). Prints count MAE and bias overall and per density bin.

Expected (as published): n=113, MAE 0.98, bias -0.49
  (0-2 wrinkles: 0.33 | 3-5: 1.52 | 6+: 2.43)

Usage:
    python3 verify_accuracy.py                 # data root auto-detected
    FLAKE_ROOT=/path/to/data python3 verify_accuracy.py
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wrinkle_native as wn


def _find_data_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for cand in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        if os.path.isdir(os.path.join(cand, "crystal_db")):
            return cand
    return os.path.dirname(d)


ROOT = os.environ.get("FLAKE_ROOT") or _find_data_root()
HOLDOUT_MIN_SEQ = 4213
BINS = (("0-2", 0, 2), ("3-5", 3, 5), ("6+", 6, 10 ** 9))


def main():
    crops = os.path.join(ROOT, "crystal_db", "crystals")
    ckpt = os.path.join(ROOT, "out", "wrinkle_native_prod.pt")
    hold_path = os.path.join(ROOT, "crystal_db", "wrinkle_points_holdout.json")
    master_path = os.path.join(ROOT, "crystal_db", "wrinkle_points.json")
    for p in (crops, ckpt, hold_path, master_path):
        if not os.path.exists(p):
            sys.exit("ERROR: missing %s" % p)

    # crops that leaked into training -> excluded from the honest test set
    master = json.load(open(master_path))
    contaminated = {
        k for k, v in master.items()
        if v.get("confirmed")
        and (m := re.match(r"seq(\d+)__", k))
        and int(m.group(1)) >= HOLDOUT_MIN_SEQ
    }
    hold = json.load(open(hold_path))
    truth = {k: len(v["points"]) for k, v in hold.items()
             if v.get("confirmed") and k not in contaminated
             and os.path.exists(os.path.join(crops, k + ".png"))}
    print("clean holdout: %d crops (never trained on)" % len(truth))

    model, meta = wn.load_model_native(ckpt)
    errs, per_bin = [], {b[0]: [] for b in BINS}
    for i, (stem, t) in enumerate(sorted(truth.items()), 1):
        pred = wn.count_crop_native(model, meta,
                                    os.path.join(crops, stem + ".png"))["count"]
        e = pred - t
        errs.append(e)
        for name, lo, hi in BINS:
            if lo <= t <= hi:
                per_bin[name].append(e)
        if i % 25 == 0:
            print("  %d/%d" % (i, len(truth)), flush=True)

    errs = np.array(errs, dtype=float)
    print("\nOVERALL  n=%d  MAE %.2f  bias %+.2f"
          % (len(errs), np.mean(np.abs(errs)), np.mean(errs)))
    for name, _, _ in BINS:
        e = np.array(per_bin[name], dtype=float)
        if len(e):
            print("  density %-4s n=%-3d MAE %.2f  bias %+.2f"
                  % (name, len(e), np.mean(np.abs(e)), np.mean(e)))


if __name__ == "__main__":
    main()
