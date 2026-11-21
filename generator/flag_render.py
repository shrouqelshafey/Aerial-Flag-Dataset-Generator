"""Stage 3 - render one flat object (flag or distractor) as a supersampled RGBA canvas.

The object is drawn `ss` times bigger than its final pixel size, with all cloth
effects, so that the later shrink (cv2.INTER_AREA) blends stripe borders the way a
real lens + sensor does. The function also returns where the 4 corners of the
object are on the canvas; compose.py pushes those corners through the same
homography as the pixels, which is how the label stays exact.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .noise import smooth_noise, uniform

SAND_BGR = np.array([0.55, 0.68, 0.78], np.float32)


@dataclass
class RenderedObject:
    canvas: np.ndarray        # (h, w, 4) float32, PREMULTIPLIED colour + alpha
    corners: np.ndarray       # (4, 2) float32 corners on the canvas: TL, TR, BR, BL
    long_px: float            # object long side on the canvas (supersampled px)
    short_px: float
    effects: dict


def load_texture(path) -> np.ndarray:
    """Read a flag/distractor image as float32 BGR 0..1 (alpha, if any, over white)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    if img.shape[2] == 4:
        a = img[..., 3:4]
        img = img[..., :3] * a + (1 - a)
    return img[..., :3]


def render_object(tex, size_m, gsd, ss, rng, fx) -> RenderedObject:
    """tex: float BGR image (long side horizontal). size_m: (long, short) in metres."""
    L = max(4, int(round(size_m[0] / gsd * ss)))
    S = max(2, int(round(size_m[1] / gsd * ss)))
    interp = cv2.INTER_AREA if tex.shape[1] > L else cv2.INTER_LINEAR
    t = cv2.resize(tex, (L, S), interpolation=interp)        # stretched to the real aspect
    eff = {}

    # --- sun fading: less saturation, a bit lighter ---------------------------
    if rng.random() < fx["fade_prob"]:
        k = uniform(rng, fx["fade_strength"])
        gray = t.mean(axis=2, keepdims=True)
        t = gray + (t - gray) * (1 - k)
        t = t + (1 - t) * k * 0.4
        eff["fade"] = round(k, 3)

    # --- dirt / sand patches -----------------------------------------------------
    if rng.random() < fx["dirt_prob"]:
        k = uniform(rng, fx["dirt_strength"])
        n = smooth_noise(S, L, max(2.0, S / 3), rng, octaves=3)
        m = np.clip(n * 0.5 + 0.1, 0, 1)[..., None] * k
        t = t * (1 - m) + SAND_BGR * m
        eff["dirt"] = round(k, 3)

    # --- put it on a transparent canvas with padding (room for folds) ------------
    pad = int(math.ceil(0.12 * S)) + 2
    Hc, Wc = S + 2 * pad, L + 2 * pad
    canvas = np.zeros((Hc, Wc, 4), np.float32)
    canvas[pad:pad + S, pad:pad + L, :3] = t
    canvas[pad:pad + S, pad:pad + L, 3] = 1.0

    # --- folds and wrinkles: move pixels (remap) + shade the slopes -------------
    if rng.random() < fx["fold_prob"]:
        amp = uniform(rng, fx["fold_amplitude"]) * S
        cell = S * rng.uniform(0.4, 1.2)
        yy, xx = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
        map_x = xx + amp * smooth_noise(Hc, Wc, cell, rng, octaves=2)
        map_y = yy + amp * smooth_noise(Hc, Wc, cell, rng, octaves=2)
        canvas = cv2.remap(canvas, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # height field = soft noise + a few straight creases
        hgt = 0.6 * smooth_noise(Hc, Wc, cell, rng, octaves=3)
        for _ in range(int(rng.integers(1, 4))):
            a = rng.uniform(0, np.pi)
            lam = S * rng.uniform(0.3, 1.0)
            hgt += 0.7 * np.sin(2 * np.pi * (xx * np.cos(a) + yy * np.sin(a)) / lam
                                + rng.uniform(0, 2 * np.pi))
        sun = rng.uniform(0, 2 * np.pi)
        gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3)
        slope = gx * np.cos(sun) + gy * np.sin(sun)
        slope /= np.abs(slope).max() + 1e-6
        k = uniform(rng, fx["fold_shading"])
        canvas[..., :3] *= (1 + k * slope)[..., None]
        eff["fold_amp_px"] = round(amp / ss, 2)
        eff["fold_shading"] = round(k, 3)

    # --- soften the edge a little (fabric edges are never razor sharp) ----------
    # The canvas is PREMULTIPLIED from the start (alpha is exactly 0 or 1 when it is
    # filled), and remap/shading keep it that way. So blur all 4 channels together:
    # blurring alpha alone would spread alpha onto black pixels = a dark halo.
    if fx["edge_feather_px"] > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), fx["edge_feather_px"] * ss)

    # Premultiplied colour can never be brighter than its alpha.
    canvas[..., :3] = np.clip(canvas[..., :3], 0, canvas[..., 3:4])
    corners = np.array([[pad, pad], [pad + L, pad], [pad + L, pad + S], [pad, pad + S]],
                       np.float32)
    return RenderedObject(canvas, corners, float(L), float(S), eff)
