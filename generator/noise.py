"""Small helpers for random smooth fields (used for folds, dirt, textures, shadows)."""
import math

import cv2
import numpy as np


def smooth_noise(h: int, w: int, cell_px: float, rng: np.random.Generator,
                 octaves: int = 1, persistence: float = 0.5) -> np.ndarray:
    """Return a smooth random field of shape (h, w), mean ~0, std ~1.

    cell_px = typical feature size in pixels. Each octave halves the feature size.
    Made by drawing a small grid of random numbers and enlarging it with
    bicubic interpolation: cheap and good enough for textures.
    """
    out = np.zeros((h, w), np.float32)
    amp, cell = 1.0, float(cell_px)
    for _ in range(octaves):
        cell = max(cell, 1.0)
        gh = int(math.ceil(h / cell)) + 3
        gw = int(math.ceil(w / cell)) + 3
        grid = rng.standard_normal((gh, gw)).astype(np.float32)
        big = cv2.resize(grid, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)
        oy = int(rng.integers(0, max(1, big.shape[0] - h)))
        ox = int(rng.integers(0, max(1, big.shape[1] - w)))
        out += amp * big[oy:oy + h, ox:ox + w]
        amp *= persistence
        cell /= 2
    out -= out.mean()
    out /= out.std() + 1e-6
    return out


def uniform(rng: np.random.Generator, lo_hi) -> float:
    """Sample uniformly from a [lo, hi] list taken from the config."""
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))
