"""Image preprocessing and augmentation.

Pipeline order:
    load -> crop_roi -> clahe -> resize -> [augment] -> normalise -> CHW tensor

The first four steps are deterministic, so ``dataset`` runs them once and caches
the result; only augmentation and normalisation happen per epoch.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

# OpenCV spawns its own thread pool, which fights with the torch pool during
# training and makes wall-clock times noisy.
cv2.setNumThreads(0)


def load_image(path) -> np.ndarray:
    """Read an image as uint8 RGB HWC. Handles .ppm, .png, .jpg."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def crop_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, margin: float = 0.10) -> np.ndarray:
    """Crop to the annotated sign box, expanded by ``margin`` on each side.

    GTSRB images carry roughly a 10% border around the sign. Cropping tight to
    the ROI removes background clutter; the margin keeps the sign's outer rim,
    which is a genuine class cue (red triangle vs blue circle).
    """
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(round(bw * margin)), int(round(bh * margin))

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    if x2 <= x1 or y2 <= y1:  # degenerate annotation; fall back to the full frame
        return img
    return img[y1:y2, x1:x2]


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 4) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel.

    GTSRB frames are dashcam captures with severe under/over-exposure. CLAHE is
    applied in LAB and only to luminance -- running it per RGB channel shifts
    hue, which matters here because sign colour is class-discriminative.

    A 4x4 tile grid (not OpenCV's 8x8 default) suits these small crops, which
    average roughly 50px per side.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size, picking the interpolation that suits the direction."""
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def preprocess_deterministic(
    img: np.ndarray, roi: tuple | None, size: int, use_clahe: bool = True
) -> np.ndarray:
    """The cacheable part of the pipeline. Returns uint8 HWC."""
    if roi is not None:
        img = crop_roi(img, *roi)
    if use_clahe:
        img = apply_clahe(img)
    return resize(img, size)


# ---------------------------------------------------------------------------
# Augmentation
#
# NO HORIZONTAL OR VERTICAL FLIP, AND NO 90-DEGREE ROTATION.
#
# Mirroring changes what a traffic sign means. Class 33 (turn right ahead) maps
# onto class 34 (turn left ahead); 19/20 are dangerous-curve-left/right; 36/37
# are go-straight-or-right/left; and every speed limit digit becomes nonsense.
# Flip augmentation is standard for natural images and actively wrong here.
# ---------------------------------------------------------------------------


def augment_affine(
    img: np.ndarray,
    rng: np.random.Generator,
    max_rotation: float = 12.0,
    max_translate: float = 0.10,
    scale_range: tuple = (0.9, 1.1),
    max_shear: float = 0.10,
) -> np.ndarray:
    """Random rotation + scale + shear + translation in a single warp.

    Composed into one 2x3 matrix on purpose: chaining three separate warpAffine
    calls interpolates three times, which visibly smears a 32px image.

    BORDER_REPLICATE rather than a constant fill -- black corners would give the
    model a border artefact to latch onto that correlates with augmentation.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    angle = rng.uniform(-max_rotation, max_rotation)
    scale = rng.uniform(*scale_range)
    shear = rng.uniform(-max_shear, max_shear)
    tx = rng.uniform(-max_translate, max_translate) * w
    ty = rng.uniform(-max_translate, max_translate) * h

    m = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # Fold a shear about the centre into the rotation/scale matrix.
    shear_m = np.array([[1.0, shear, -shear * cy], [0.0, 1.0, 0.0]], dtype=np.float64)
    m3 = np.vstack([m, [0, 0, 1]]) @ np.vstack([shear_m, [0, 0, 1]])
    m = m3[:2]

    m[0, 2] += tx
    m[1, 2] += ty

    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def augment_photometric(
    img: np.ndarray,
    rng: np.random.Generator,
    brightness: tuple = (0.7, 1.3),
    contrast: tuple = (0.8, 1.2),
    noise_prob: float = 0.2,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Random brightness/contrast jitter with occasional gaussian noise."""
    x = img.astype(np.float32) / 255.0

    x = x * rng.uniform(*brightness)
    mean = x.mean()
    x = (x - mean) * rng.uniform(*contrast) + mean

    if rng.random() < noise_prob:
        x = x + rng.normal(0.0, noise_std, x.shape).astype(np.float32)

    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return augment_photometric(augment_affine(img, rng), rng)


def to_tensor(img: np.ndarray, mean, std) -> torch.Tensor:
    """uint8 HWC -> normalised float32 CHW."""
    x = img.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def denormalise(t: torch.Tensor, mean, std) -> np.ndarray:
    """Inverse of to_tensor, for plotting. Returns uint8 HWC."""
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)
