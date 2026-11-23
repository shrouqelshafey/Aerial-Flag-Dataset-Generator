"""Give a pasted flag the camera look of a REAL background photo.

A real drone photo already went through a camera: lens blur, sensor noise, haze
that lifts the blacks. A rendered flag has none of that. If we paste it as it is,
the flag is the only perfectly clean patch in the picture, and a detector learns
that shortcut ("clean patch = flag") instead of the flag. On real frames, where
the flag is just as blurry and noisy as the ground, the shortcut fails.

So we measure the photo once and give every flag the same three things:

  black level  0.5th percentile of each colour channel = haze / airlight of the whole
               photo.  flag = black + flag * (1 - black)
  noise        sensor noise sigma, Immerkaer's method on the smoothest blocks of the
               photo (texture is not mistaken for noise)
  blur         a small Gaussian, sigma drawn from blur_sigma_px (lens + demosaicing)

The background itself is never re-processed: it is already a real photo.
"""
import math

import cv2
import numpy as np

from .noise import uniform

_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def estimate_noise(gray, block=32):
    """Noise sigma of a float image (0..1). Immerkaer (1996), per block, 10th percentile.

    The kernel cancels smooth gradients, so on flat ground only noise is left; the
    quietest 10 % of the blocks are flat ground, the rest also contain texture.
    """
    conv = np.abs(cv2.filter2D(gray, cv2.CV_32F, _K))[1:-1, 1:-1]
    h, w = conv.shape
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return 0.0
    means = conv[:hb * block, :wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3))
    return float(math.sqrt(math.pi / 2) / 6 * np.percentile(means, 10))


def black_level(img, cap=0.25):
    """Per-channel 0.5th percentile (BGR), capped: the haze floor of the photo."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    return np.minimum(np.percentile(small.reshape(-1, 3), 0.5, axis=0), cap).astype(np.float32)


def measure(img, cfg):
    """Camera look of one photo (float32 BGR 0..1). cfg = generator.batch.match_camera."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "black": black_level(img) if cfg.get("haze", True) else None,
        "noise": min(estimate_noise(gray), 0.04) if cfg.get("noise", True) else 0.0,
    }


def for_image(look, rng, cfg):
    """The match dict for one image: the photo's look + a blur drawn for this image."""
    return {**look, "blur_sigma": uniform(rng, cfg.get("blur_sigma_px", [0.0, 0.0]))}
