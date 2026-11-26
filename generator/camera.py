"""Stage 7 - camera simulation, applied to the whole frame in PHYSICAL order:

  atmosphere (haze) -> exposure (brightness / contrast) -> lens (vignetting,
  defocus) -> motion during exposure -> sensor noise -> image processing (gamma)

The final JPEG compression happens once, when the frame is saved
(image_effects.compression.jpeg_quality).

Each step is switched by config/config.yaml: image_effects.*.enabled for the
spec's effects, generator.layers for the rest. A switched-off step leaves the
image untouched. Order matters: noise added before blur would be blurred away,
which a real camera never does.
"""
import math

import cv2
import numpy as np

from .noise import uniform

HAZE_BGR = np.array([0.78, 0.84, 0.88], np.float32)


def motion_kernel(length_px: float, angle_rad: float) -> np.ndarray:
    k = int(math.ceil(length_px)) + 2
    k += (k % 2 == 0)
    ker = np.zeros((k, k), np.float32)
    c = k // 2
    for t in np.linspace(-length_px / 2, length_px / 2, max(2, int(length_px * 4))):
        x, y = c + t * math.cos(angle_rad), c + t * math.sin(angle_rad)
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0                       # bilinear "splat" of one sample
        ker[y0, x0] += (1 - fx) * (1 - fy)
        ker[y0, x0 + 1] += fx * (1 - fy)
        ker[y0 + 1, x0] += (1 - fx) * fy
        ker[y0 + 1, x0 + 1] += fx * fy
    return ker / ker.sum()


def apply_blur(img, gsd, rng, c):
    """Lens defocus, then motion during the exposure, on the WHOLE picture (flags and
    ground together, so a flag is never sharper than the ground around it).

    c: defocus_prob, defocus_sigma_px [lo, hi], motion_prob, speed_mps [lo, hi],
       exposure_s [lo, hi]. Motion length (px) = speed (m/s) * exposure (s) / GSD (m/px).
    """
    p = {}
    if rng.random() < c["defocus_prob"]:
        sigma = uniform(rng, c["defocus_sigma_px"])
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        p["defocus_sigma_px"] = round(sigma, 3)

    if rng.random() < c["motion_prob"]:
        speed = uniform(rng, c["speed_mps"])
        lo, hi = c["exposure_s"]
        exposure = math.exp(rng.uniform(math.log(lo), math.log(hi)))   # log-uniform
        length = speed * exposure / gsd
        if length >= 0.5:
            img = cv2.filter2D(img, -1, motion_kernel(length, rng.uniform(0, 2 * np.pi)),
                               borderType=cv2.BORDER_REFLECT)
        p["motion_px"] = round(length, 3)
    return img, p


def apply_camera(img, gsd, rng, c):
    """img: float32 BGR 0..1. Returns (uint8 BGR image, dict of what was applied)."""
    h, w = img.shape[:2]
    p = {}

    if rng.random() < c["haze_prob"]:
        k = uniform(rng, c["haze_strength"])
        img = img * (1 - k) + HAZE_BGR * k
        p["haze"] = round(k, 3)

    b, ct = uniform(rng, c["brightness"]), uniform(rng, c["contrast"])
    if b != 1.0 or ct != 1.0:
        mean = float(img.mean())
        img = ((img - mean) * ct + mean) * b
        p["brightness"], p["contrast"] = round(b, 3), round(ct, 3)

    s = uniform(rng, c["vignette_strength"])
    if s > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        img = img * (1 - s * r2)[..., None]
        p["vignette"] = round(s, 3)

    img, pb = apply_blur(img, gsd, rng, c)
    p.update(pb)

    read, shot = uniform(rng, c["read_noise"]), uniform(rng, c["shot_noise"])
    img = np.clip(img, 0, 1)
    if read > 0 or shot > 0:
        noise_sigma = np.sqrt(read ** 2 + shot * img)
        img = img + rng.standard_normal(img.shape, dtype=np.float32) * noise_sigma
        p["read_noise"], p["shot_noise"] = round(read, 4), round(shot, 4)

    g = uniform(rng, c["gamma"])
    if g != 1.0:
        img = np.clip(img, 0, 1) ** (1 / g)
        p["gamma"] = round(g, 3)

    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return out, p
