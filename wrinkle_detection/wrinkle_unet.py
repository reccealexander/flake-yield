#!/usr/bin/env python3
"""Point-supervised wrinkle *counting* via a small U-Net density regressor.

Idea
----
Every wrinkle is annotated with ONE clicked point. For a crop we render a
target "density" map: the sum of unit-amplitude Gaussian bumps (sigma ~ 4 px)
placed at the points and masked to the crystal. A small U-Net regresses this
map. The count is then read two independent ways:

  * PEAK count     = number of local maxima above a threshold
  * INTEGRAL count = sum(predicted map) / integral-of-one-unit-bump

Their disagreement is a cheap uncertainty flag (see count_wrinkle_unet.py).

This module is a *library*: torch is imported lazily so the non-torch helpers
(``make_target``, ``bump_volume``, ``load_crop_gray``) and the geometric
augmentation can be imported and unit-tested on a machine without torch.

Expected annotation file  (crystal_db/wrinkle_points.json)
----------------------------------------------------------
A JSON object keyed by crop id (the crop PNG basename without extension)::

    {
      "seq3942__gds29": {
        "confirmed": true,
        "points": [[x, y], [x, y], ...]     # crop-pixel coords, x=col y=row
      },
      ...
    }

Only entries with ``confirmed == true`` are used for training. A few common
shape variants are tolerated (see ``load_points_file``): a bare list of points,
points given as ``{"x":..,"y":..}`` dicts, or a ``{"points":[...] }`` wrapper.
"""

import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CRYSTAL_DB = os.path.join(ROOT, "crystal_db")
DEFAULT_CROPS = os.path.join(CRYSTAL_DB, "crystals")
DEFAULT_POINTS = os.path.join(CRYSTAL_DB, "wrinkle_points.json")

BG_VALUE = 0.0          # background (outside mask) value for the input channel
DEFAULT_SIGMA = 4.0     # Gaussian bump sigma in output pixels
TRUNCATE = 4.0          # render each bump out to TRUNCATE * sigma


# ---------------------------------------------------------------------------
# Target rendering  (pure numpy, torch-free)
# ---------------------------------------------------------------------------

def bump_volume(sigma=DEFAULT_SIGMA):
    """Analytic integral (volume) of one unit-amplitude 2-D Gaussian bump.

    A bump is ``exp(-(dx^2 + dy^2) / (2 sigma^2))`` with peak value 1, whose
    integral over the plane is ``2 * pi * sigma^2``. INTEGRAL count divides the
    summed map by this number.
    """
    return 2.0 * math.pi * float(sigma) * float(sigma)


def make_target(points, shape, mask=None, sigma=DEFAULT_SIGMA):
    """Render the density target: sum of unit Gaussian bumps at ``points``.

    Parameters
    ----------
    points : iterable of (x, y)
        Point coordinates in crop pixels (x = column, y = row). Points outside
        the image are ignored.
    shape : (H, W)
        Output map shape.
    mask : (H, W) array or None
        If given, the final map is multiplied by (mask > 0) so density outside
        the crystal is removed.
    sigma : float
        Bump standard deviation in pixels.

    Returns
    -------
    (H, W) float32 density map.
    """
    H, W = int(shape[0]), int(shape[1])
    target = np.zeros((H, W), dtype=np.float32)
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    half = int(math.ceil(TRUNCATE * sigma))
    two_s2 = 2.0 * sigma * sigma

    for p in points:
        px, py = float(p[0]), float(p[1])
        # integer window around the point, clipped to the image
        x0 = int(math.floor(px)) - half
        x1 = int(math.floor(px)) + half + 1
        y0 = int(math.floor(py)) - half
        y1 = int(math.floor(py)) + half + 1
        x0c, x1c = max(0, x0), min(W, x1)
        y0c, y1c = max(0, y0), min(H, y1)
        if x0c >= x1c or y0c >= y1c:
            continue  # bump falls entirely outside the image
        xs = np.arange(x0c, x1c, dtype=np.float32) - px
        ys = np.arange(y0c, y1c, dtype=np.float32) - py
        gx = np.exp(-(xs * xs) / two_s2)
        gy = np.exp(-(ys * ys) / two_s2)
        bump = np.outer(gy, gx).astype(np.float32)
        target[y0c:y1c, x0c:x1c] += bump

    if mask is not None:
        target *= (np.asarray(mask) > 0).astype(np.float32)
    return target


# ---------------------------------------------------------------------------
# Crop loading  (pure numpy, torch-free)
# ---------------------------------------------------------------------------

def load_crop_gray(path, bg_value=BG_VALUE):
    """Load an RGBA crop -> (gray, mask).

    The single input channel is the *contrast-normalized green channel* masked
    to the crystal:

      * take the RGBA crop; alpha > 0 is the crystal mask,
      * percentile 2-98 stretch of the green channel *within the mask*,
      * scale to [0, 1], zero the background.

    Returns
    -------
    gray : (H, W) float32 in [0, 1], background = bg_value
    mask : (H, W) float32 in {0, 1}
    """
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGBA"), dtype=np.float32)
    green = arr[..., 1]
    mask = (arr[..., 3] > 0).astype(np.float32)
    inside = green[mask > 0]
    if inside.size:
        lo, hi = np.percentile(inside, (2.0, 98.0))
    else:
        lo, hi = 0.0, 255.0
    if hi - lo < 1.0:
        hi = lo + 1.0
    gray = np.clip((green - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out = np.full_like(gray, float(bg_value))
    out[mask > 0] = gray[mask > 0]
    return out, mask


# ---------------------------------------------------------------------------
# Points file loading + filtering  (torch-free)
# ---------------------------------------------------------------------------

def _coerce_points(raw):
    """Turn a variety of point encodings into a list of (x, y) float tuples."""
    pts = []
    if raw is None:
        return pts
    if isinstance(raw, dict):
        raw = raw.get("points", raw.get("pts", []))
    for item in raw:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                pts.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pts.append((float(item[0]), float(item[1])))
    return pts


def load_points_file(points_path=DEFAULT_POINTS, confirmed_only=True):
    """Load wrinkle_points.json -> {crop_id: [(x, y), ...]}.

    When ``confirmed_only`` is True (default) only entries whose ``confirmed``
    field is truthy are kept. Entry shapes accepted per crop id:

      * ``{"confirmed": bool, "points": [[x,y], ...]}``  (canonical)
      * ``{"confirmed": bool, "points": [{"x":..,"y":..}, ...]}``
      * a bare ``[[x,y], ...]``  (treated as confirmed)
    """
    with open(points_path, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("wrinkle_points.json must be a JSON object keyed by crop id")
    out = {}
    for cid, entry in data.items():
        if isinstance(entry, dict):
            if confirmed_only and not entry.get("confirmed", False):
                continue
            pts = _coerce_points(entry)
        else:
            # bare list -> treated as confirmed points
            pts = _coerce_points(entry)
        out[str(cid)] = pts
    return out


# ---------------------------------------------------------------------------
# Geometric augmentation  (torch-free; SAME transform for image + points)
# ---------------------------------------------------------------------------

def _center_affine(size, flip_x, flip_y, angle_deg):
    """Build a 3x3 homogeneous affine about the center of a size x size grid.

    Composition (applied to a column vector [x, y, 1]):
        translate(+c) . rotate(angle) . scale(flip) . translate(-c)
    """
    c = (size - 1) / 2.0
    to_origin = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], dtype=np.float64)
    scale = np.array([[-1 if flip_x else 1, 0, 0],
                      [0, -1 if flip_y else 1, 0],
                      [0, 0, 1]], dtype=np.float64)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    rot = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
    from_origin = np.array([[1, 0, c], [0, 1, c], [0, 0, 1]], dtype=np.float64)
    return from_origin @ rot @ scale @ to_origin


def apply_affine_to_points(points, M):
    """Apply a 3x3 (or 2x3) affine to an (N,2) array of (x, y) points."""
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([pts, ones], axis=1)          # (N,3)
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (2, 3):
        M = np.vstack([M, [0, 0, 1]])
    out = (M @ homog.T).T[:, :2]
    return out.astype(np.float32)


def augment_sample(gray, mask, points, rng, max_angle=12.0,
                   brightness=0.15, contrast=0.25, geom=True, photo=True):
    """Apply matched geometric + photometric augmentation.

    The SAME geometric affine (random rot90 x flip x small-angle rotation) is
    applied to ``gray``, ``mask`` and the point coordinates, so points stay
    pinned to image features. Photometric jitter (brightness/contrast) touches
    only the image. Inputs must be square (H == W).

    Returns (gray_aug, mask_aug, points_aug).
    """
    import cv2
    H, W = gray.shape
    if H != W:
        raise ValueError("augment_sample expects a square crop (H == W)")
    size = H
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if geom:
        flip_x = rng.random() < 0.5
        flip_y = rng.random() < 0.5
        angle = 90.0 * rng.integers(0, 4) + rng.uniform(-max_angle, max_angle)
        M = _center_affine(size, flip_x, flip_y, angle)
        M23 = M[:2, :].astype(np.float32)
        gray = cv2.warpAffine(gray, M23, (size, size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=BG_VALUE)
        mask = cv2.warpAffine(mask, M23, (size, size),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        pts = apply_affine_to_points(pts, M)

    if photo:
        b = rng.uniform(-brightness, brightness)
        c = 1.0 + rng.uniform(-contrast, contrast)
        gray = np.clip(gray * c + b, 0.0, 1.0).astype(np.float32)

    mask = (mask > 0.5).astype(np.float32)
    gray = gray * mask                       # keep background clean
    return gray, mask, pts


def letterbox_to_square(gray, mask, points, size):
    """Isotropic-scale + pad a (possibly non-square) crop to size x size.

    Keeps Gaussian bumps circular (single scale factor for both axes). Returns
    (gray_sq, mask_sq, points_sq) with points mapped into the padded frame.
    """
    import cv2
    H, W = gray.shape
    s = float(size) / float(max(H, W))
    newW, newH = max(1, int(round(W * s))), max(1, int(round(H * s)))
    g = cv2.resize(gray, (newW, newH), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (newW, newH), interpolation=cv2.INTER_NEAREST)
    ox = (size - newW) // 2
    oy = (size - newH) // 2
    gray_sq = np.full((size, size), BG_VALUE, dtype=np.float32)
    mask_sq = np.zeros((size, size), dtype=np.float32)
    gray_sq[oy:oy + newH, ox:ox + newW] = g
    mask_sq[oy:oy + newH, ox:ox + newW] = (m > 0.5)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts):
        pts = pts * s + np.array([ox, oy], dtype=np.float32)
    return gray_sq, mask_sq, pts


# ---------------------------------------------------------------------------
# Dataset  (torch imported lazily inside the class)
# ---------------------------------------------------------------------------

def build_index(points_path=DEFAULT_POINTS, crops_dir=DEFAULT_CROPS,
                confirmed_only=True):
    """Return [(crop_png_path, [(x,y),...]), ...] for confirmed crops on disk."""
    by_id = load_points_file(points_path, confirmed_only=confirmed_only)
    items, missing = [], 0
    for cid, pts in sorted(by_id.items()):
        path = os.path.join(crops_dir, cid + ".png")
        if os.path.exists(path):
            items.append((path, pts))
        else:
            missing += 1
    return items, missing


def make_dataset_class():
    """Return the ``WrinklePointDataset`` class (torch required)."""
    from torch.utils.data import Dataset
    import torch

    class WrinklePointDataset(Dataset):
        """Loads crops + confirmed points, renders density targets, augments.

        Each item is (image_tensor, target_tensor, mask_tensor), all shaped
        (1, size, size) float32.
        """

        def __init__(self, points_path=DEFAULT_POINTS, crops_dir=DEFAULT_CROPS,
                     size=256, sigma=DEFAULT_SIGMA, augment=True, seed=0,
                     items=None, confirmed_only=True, max_angle=12.0):
            if items is None:
                items, _ = build_index(points_path, crops_dir, confirmed_only)
            self.items = list(items)
            self.size = int(size)
            self.sigma = float(sigma)
            self.augment = bool(augment)
            self.max_angle = float(max_angle)
            self._base_seed = int(seed)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, points = self.items[i]
            gray, mask = load_crop_gray(path)
            gray, mask, pts = letterbox_to_square(gray, mask, points, self.size)
            if self.augment:
                # deterministic-per-item stream that still varies across epochs
                rng = np.random.default_rng(
                    (self._base_seed * 1000003 + i * 9176 + np.random.randint(1 << 30))
                    & 0x7FFFFFFF)
                gray, mask, pts = augment_sample(
                    gray, mask, pts, rng, max_angle=self.max_angle)
            target = make_target(pts, (self.size, self.size), mask, self.sigma)
            img_t = torch.from_numpy(gray).unsqueeze(0).float()
            tgt_t = torch.from_numpy(target).unsqueeze(0).float()
            msk_t = torch.from_numpy(mask).unsqueeze(0).float()
            return img_t, tgt_t, msk_t

    return WrinklePointDataset


# ---------------------------------------------------------------------------
# The U-Net  (torch required; factory keeps torch import lazy)
# ---------------------------------------------------------------------------

def make_unet_class():
    """Return the ``WrinkleUNet`` class (torch required)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv_block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class WrinkleUNet(nn.Module):
        """Small encoder-decoder U-Net, single-channel in, single-channel out.

        4 down / 4 up levels with skip connections. Output is passed through
        softplus so the predicted density is non-negative (matches the target).
        """

        def __init__(self, in_ch=1, base=16, out_ch=1):
            super().__init__()
            self.enc1 = conv_block(in_ch, base)
            self.enc2 = conv_block(base, base * 2)
            self.enc3 = conv_block(base * 2, base * 4)
            self.enc4 = conv_block(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = conv_block(base * 8, base * 16)
            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
            self.dec4 = conv_block(base * 16, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = conv_block(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = conv_block(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = conv_block(base * 2, base)
            self.head = nn.Conv2d(base, out_ch, 1)

        @staticmethod
        def _cat(up, skip):
            # pad the upsampled feature to the skip size if off by a pixel
            if up.shape[-2:] != skip.shape[-2:]:
                dh = skip.shape[-2] - up.shape[-2]
                dw = skip.shape[-1] - up.shape[-1]
                up = F.pad(up, [0, dw, 0, dh])
            return torch.cat([up, skip], dim=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(self._cat(self.up4(b), e4))
            d3 = self.dec3(self._cat(self.up3(d4), e3))
            d2 = self.dec2(self._cat(self.up2(d3), e2))
            d1 = self.dec1(self._cat(self.up1(d2), e1))
            out = self.head(d1)
            return F.softplus(out)

    return WrinkleUNet


def build_unet(in_ch=1, base=16, out_ch=1):
    """Convenience constructor: returns an instantiated WrinkleUNet."""
    return make_unet_class()(in_ch=in_ch, base=base, out_ch=out_ch)


if __name__ == "__main__":
    # tiny self-check of the torch-free core
    pts = [(50, 50), (100, 120), (200, 180)]
    tgt = make_target(pts, (256, 256), None, DEFAULT_SIGMA)
    est = tgt.sum() / bump_volume(DEFAULT_SIGMA)
    print("make_target: %d points -> integral estimate %.4f "
          "(bump_volume=%.3f)" % (len(pts), est, bump_volume(DEFAULT_SIGMA)))
