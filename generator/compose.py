"""Stages 4 + 5 - placement and compositing.

place_object() builds ONE homography that maps the supersampled canvas to the
frame (rotation + scale + small perspective tilt + position), warps the pixels with
it, and returns the 4 object corners pushed through the very same matrix.
Those corners become the label.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .noise import uniform


@dataclass
class Scene:
    tint: np.ndarray      # (3,) colour cast of the background, applied to pasted objects
    light: float          # sun strength for pasted objects
    sun_dir: np.ndarray   # (2,) unit vector, direction shadows fall


def make_scene(frame, rng, cc) -> Scene:
    mean = frame.reshape(-1, 3).mean(axis=0)
    tint = (mean / (mean.mean() + 1e-6)) ** cc["tint_strength"]
    a = rng.uniform(0, 2 * np.pi)
    return Scene(tint.astype(np.float32), uniform(rng, cc["light_gain"]),
                 np.array([np.cos(a), np.sin(a)], np.float32))


def object_corners(center, angle_deg, long_px, short_px, rng, tilt):
    """Ideal rectangle corners (TL, TR, BR, BL) in frame pixels + small random jitter."""
    L, S = long_px, short_px
    base = np.array([[-L / 2, -S / 2], [L / 2, -S / 2], [L / 2, S / 2], [-L / 2, S / 2]], np.float32)
    t = math.radians(angle_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], np.float32)
    jitter = rng.uniform(-tilt, tilt, size=(4, 2)).astype(np.float32) * S
    return base @ R.T + jitter + np.asarray(center, np.float32)


def place_object(frame, obj, dst, ss, rng, cc, scene, gsd, shadow=True, match=None):
    """Warp `obj` (RenderedObject) so its corners land on `dst` and blend it into `frame`.

    frame: float32 BGR, modified in place.  dst: (4, 2) corners in frame pixels.
    match: optional camera look of a REAL background photo (see generator/match.py):
           {"blur_sigma": px, "black": (3,) black level, "noise": sigma}. The object
           gets the same lens blur, haze and sensor noise as the photo around it.
    """
    h, w = frame.shape[:2]
    Hm = cv2.getPerspectiveTransform(obj.corners, (dst * ss).astype(np.float32))

    # Where does the whole canvas (object + padding) land in the frame?
    Hc, Wc = obj.canvas.shape[:2]
    cc_pts = np.array([[[0, 0]], [[Wc, 0]], [[Wc, Hc]], [[0, Hc]]], np.float32)
    box = cv2.perspectiveTransform(cc_pts, Hm).reshape(-1, 2) / ss
    blur = float(match.get("blur_sigma", 0.0)) if match else 0.0
    m = 1 + int(math.ceil(3 * blur))                         # room for the blur to spread
    x0 = max(0, int(math.floor(box[:, 0].min())) - m)
    y0 = max(0, int(math.floor(box[:, 1].min())) - m)
    x1 = min(w, int(math.ceil(box[:, 0].max())) + m)
    y1 = min(h, int(math.ceil(box[:, 1].max())) + m)
    if x1 <= x0 or y1 <= y0:
        return {"drawn": False}

    # Warp at supersampled resolution into a small patch, then shrink with INTER_AREA.
    T = np.array([[1, 0, -x0 * ss], [0, 1, -y0 * ss], [0, 0, 1]], np.float64)
    pw, ph = x1 - x0, y1 - y0
    patch = cv2.warpPerspective(obj.canvas, T @ Hm, (pw * ss, ph * ss), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_AREA)
    info = {"drawn": True}
    if blur > 0:          # lens softness; blurring premultiplied RGBA = no dark halo
        patch = cv2.GaussianBlur(patch, (0, 0), blur)
        info["blur_sigma_px"] = round(blur, 3)
    alpha = patch[..., 3:4]
    rgb = patch[..., :3] * scene.tint * scene.light          # premultiplied colour
    if match is not None and match.get("black") is not None:
        # haze / airlight: the photo's darkest value is lifted, so is the flag's
        b = np.asarray(match["black"], np.float32)
        rgb = rgb * (1 - b) + alpha * b

    region = frame[y0:y1, x0:x1]
    if shadow and rng.random() < cc["contact_shadow_prob"]:
        off = uniform(rng, cc["contact_shadow_offset_m"]) / gsd
        k = uniform(rng, cc["contact_shadow_strength"])
        M = np.float32([[1, 0, scene.sun_dir[0] * off], [0, 1, scene.sun_dir[1] * off]])
        sh = cv2.warpAffine(alpha[..., 0], M, (pw, ph), borderValue=0)
        sh = cv2.GaussianBlur(sh, (0, 0), max(0.6, off))
        region *= (1 - k * sh)[..., None]
        info["contact_shadow"] = round(k, 3)

    region[:] = rgb + (1 - alpha) * region                   # premultiplied "over"
    if match is not None and match.get("noise", 0) > 0:
        # same sensor noise as the photo, only where the object is
        region += rng.standard_normal(region.shape, dtype=np.float32) * (match["noise"] * alpha)
        info["noise_sigma"] = round(float(match["noise"]), 4)
    return info


def cast_shadows(frame, gsd, rng, cc):
    """Large soft shadows from things outside the picture (buildings, poles, aircraft)."""
    if rng.random() >= cc["cast_shadow_prob"]:
        return []
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.float32)
    kinds = []
    for _ in range(int(rng.integers(1, 3))):
        kind = rng.choice(["edge", "pole", "wing"])
        a = rng.uniform(0, np.pi)
        d = np.array([np.cos(a), np.sin(a)])
        n = np.array([-d[1], d[0]])
        c = np.array([rng.uniform(0, w), rng.uniform(0, h)])
        if kind == "edge":            # a building: everything on one side of a line
            big = 4 * (w + h)
            pts = [c - d * big, c + d * big, c + d * big + n * big, c - d * big + n * big]
        elif kind == "pole":          # a long thin shadow
            L, W = rng.uniform(5, 25) / gsd, rng.uniform(0.15, 0.6) / gsd
            pts = [c - d * L / 2 - n * W / 2, c + d * L / 2 - n * W / 2,
                   c + d * L / 2 + n * W / 2, c - d * L / 2 + n * W / 2]
        else:                         # a parked aircraft / hangar overhang: wide band
            L, W = rng.uniform(8, 30) / gsd, rng.uniform(1.5, 4) / gsd
            pts = [c - d * L / 2 - n * W / 2, c + d * L / 2 - n * W / 2,
                   c + d * L / 2 + n * W / 2, c - d * L / 2 + n * W / 2]
        cv2.fillPoly(mask, [np.round(np.array(pts) * 4).astype(np.int32)], 1.0,
                     lineType=cv2.LINE_AA, shift=2)
        kinds.append(str(kind))
    penumbra = max(0.7, rng.uniform(0.05, 0.3) / gsd)        # soft edge, in pixels
    mask = cv2.GaussianBlur(mask, (0, 0), penumbra)
    k = uniform(rng, cc["cast_shadow_strength"])
    # Shadows are lit by blue sky only: red drops more than blue (BGR order).
    mult = 1 - k * mask[..., None] * np.array([0.85, 1.0, 1.1], np.float32)
    frame *= mult
    return kinds
