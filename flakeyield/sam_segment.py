"""Segment crystals with SAM (Segment Anything), zero-shot.

Replaces the hand-tuned thresholding in `segment.py`, which needed six constants
-- a dark floor, a bright percentile, two sigmas, a min-area, a hole size -- each
added in response to a measured failure, and each of which a new imaging
condition then broke.

Measured against the same three frames (see experiments/try_sam.py):

  seq3941 -- the two slivers are the same template shape. SAM sizes them at
  171182 and 170679 px, agreeing to 0.3%; the threshold pipeline said 137118 and
  123284, differing by 11%. Same object twice, so self-consistency is evidence.

  chip3 -- the threshold pipeline merges the crystal with its neighbour into one
  blob. The GDS is a compact tiling, so shapes touch, and watershed cannot cut
  them: there is no gradient ridge along the join. SAM separates them.

SAM has no concept of "crystal" -- it segments everything it sees, including the
background as one ~98% mask. Choosing which masks are crystals is this module's
real work, and its main risk.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.measure import regionprops

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "sam_vit_b.pth"
MODEL_TYPE = "vit_b"

# SAM runs on CPU: SamAutomaticMaskGenerator builds float64 point tensors and
# Apple's MPS backend has no float64.
DEVICE = "cpu"

# Sampling density of the automatic mask generator. 16 is enough for objects
# this large and keeps CPU runtime tolerable.
POINTS_PER_SIDE = 16

MIN_CRYSTAL_PX = 20_000

# Above this fraction of the frame a mask is the background, which SAM returns
# as an object like any other (~98% on these frames).
MAX_FRAME_FRACTION = 0.70

# Crystals are slivers: the template's acute tips run to ~9.5 degrees, and every
# real crystal measured lands between 4.7x and 12x elongation. chip3's specular
# blob is 1.3x. This is what separates a crystal from a bright artifact.
MIN_ELONGATION = 2.5

# A crystal is brighter than the gold background under this LUT -- the same
# physics segment.py relies on. SAM has no such concept: it returns the dark
# off-sample bands at frame edges as objects too, since they are perfectly good
# regions, just not material.
#
# Measured, as mean green minus the frame median: real crystals +5.6, +12.7,
# +15.7, +17.4; the dark bands -6.5 and -112.2. Zero separates them cleanly.
#
# Assumes the frame median is background, i.e. that crystals do not cover most of
# the frame. True here (15-30%); revisit for a zoomed-out mosaic, where it fails.
MIN_BRIGHTNESS_OVER_MEDIAN = 0.0

_model_cache: dict = {}


@dataclass
class SamCrystal:
    mask: np.ndarray
    area_px: int
    elongation: float
    brightness_over_median: float
    predicted_iou: float
    stability: float


def to_rgb(green: np.ndarray) -> np.ndarray:
    """SAM expects a natural photo: 3-channel, full dynamic range.

    These frames are single-channel and low-contrast (the crystal/background
    step is ~20 grey levels), so green is stretched across 0-255 and replicated.
    Without the stretch SAM sees a nearly flat field.
    """
    lo, hi = np.percentile(green, 1), np.percentile(green, 99)
    stretched = np.clip((green - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return np.repeat(stretched[:, :, None], 3, axis=2).astype(np.uint8)


def _generator(checkpoint: Path, points_per_side: int):
    """Load SAM once; the checkpoint is 358MB and loading dominates a single frame."""
    key = (str(checkpoint), points_per_side)
    if key not in _model_cache:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at {checkpoint}. Download with:\n"
                f"  curl -sL -o {checkpoint} "
                f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )
        sam = sam_model_registry[MODEL_TYPE](checkpoint=str(checkpoint)).to(DEVICE)
        _model_cache[key] = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)
    return _model_cache[key]


def find_crystals(
    green: np.ndarray,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    points_per_side: int = POINTS_PER_SIDE,
    min_crystal_px: int = MIN_CRYSTAL_PX,
    max_frame_fraction: float = MAX_FRAME_FRACTION,
    min_elongation: float = MIN_ELONGATION,
    min_brightness: float = MIN_BRIGHTNESS_OVER_MEDIAN,
) -> list[np.ndarray]:
    """Boolean mask per crystal, largest first."""
    return [c.mask for c in describe_crystals(
        green, checkpoint, points_per_side, min_crystal_px,
        max_frame_fraction, min_elongation, min_brightness,
    )]


def describe_crystals(
    green: np.ndarray,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    points_per_side: int = POINTS_PER_SIDE,
    min_crystal_px: int = MIN_CRYSTAL_PX,
    max_frame_fraction: float = MAX_FRAME_FRACTION,
    min_elongation: float = MIN_ELONGATION,
    min_brightness: float = MIN_BRIGHTNESS_OVER_MEDIAN,
) -> list[SamCrystal]:
    """As find_crystals, but keeping SAM's own quality scores for inspection."""
    generator = _generator(Path(checkpoint), points_per_side)
    raw = generator.generate(to_rgb(green))
    frame_median = float(np.median(green))

    crystals = []
    for m in raw:
        area = int(m["area"])
        if area < min_crystal_px:
            continue
        if area > max_frame_fraction * green.size:
            continue  # the background, which SAM returns as an object

        mask = m["segmentation"]
        props = regionprops(mask.astype(np.uint8))
        if not props:
            continue
        p = props[0]
        elongation = p.axis_major_length / max(p.axis_minor_length, 1e-6)
        if elongation < min_elongation:
            continue

        brightness = float(green[mask].mean() - frame_median)
        if brightness < min_brightness:
            continue  # a dark region, not material

        crystals.append(
            SamCrystal(
                mask=mask,
                area_px=area,
                elongation=float(elongation),
                brightness_over_median=brightness,
                predicted_iou=float(m.get("predicted_iou", float("nan"))),
                stability=float(m.get("stability_score", float("nan"))),
            )
        )

    crystals.sort(key=lambda c: -c.area_px)
    return crystals
