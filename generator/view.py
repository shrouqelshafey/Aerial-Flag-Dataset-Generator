"""Camera tilt: the gimbal is never pointing exactly straight down.

A camera that turns on its own spot sees a new picture that is an exact homography
of the old one, for any scene: H = K R K^-1 (K = camera matrix, R = the turn). So a
straight-down picture becomes an off-nadir picture with one cv2.warpPerspective, and
every flag corner goes through the same H: the labels stay exact. Off-nadir, the far
side of the picture is further away (smaller flags), the near side closer (bigger).

Two ways to fill the tilted picture:

  canvas_view()  generate.py builds the ground itself, so it builds the straight-down
                 area the tilted camera really sees (bigger than one frame on the far
                 side) and warps it with no zoom. Physically exact.
  zoom_view()    generate_batch.py starts from a fixed real photo, so the tilted picture
                 is zoomed in about the centre until no corner is empty (= a slightly
                 longer lens). The zoom grows fast with tilt (about 1.2x at 5 deg), so
                 keep the batch tilt small.

Each returns a dict: H (source -> frame), size (source width, height), quad (the 4
frame corners in source pixels: place flags inside it), tilt_deg, azimuth_deg,
magnification (linear scale at the frame centre).
"""
import math

import cv2
import numpy as np

from .noise import uniform


def _K(W, H, f_px):
    return np.array([[f_px, 0, W / 2], [0, f_px, H / 2], [0, 0, 1]], np.float64)


def _turn_matrix(W, H, f_px, tilt_deg, azimuth_deg):
    """Homography of a camera turned by `tilt` towards `azimuth` (0 = image right)."""
    a = math.radians(azimuth_deg)
    axis = np.array([-math.sin(a), math.cos(a), 0.0])
    R, _ = cv2.Rodrigues(axis * math.radians(tilt_deg))
    K = _K(W, H, f_px)
    return K @ R @ np.linalg.inv(K)


def _frame_corners(W, H):
    return np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)


def _map(pts, M):
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), M).reshape(-1, 2)


def magnification(M, x, y):
    """Linear scale of the warp M where it lands on output pixel (x, y)."""
    q = _map([[x, y]], np.linalg.inv(M))[0]
    a, b, c = _map([q, q + (1.0, 0.0), q + (0.0, 1.0)], M)
    return float(math.sqrt(abs(np.linalg.det(np.column_stack([b - a, c - a])))))


def nadir_view(W, H):
    return {"H": np.eye(3), "size": (W, H), "quad": _frame_corners(W, H).astype(np.float32),
            "tilt_deg": 0.0, "azimuth_deg": 0.0, "magnification": 1.0}


def canvas_view(W, H, f_px, tilt_deg, azimuth_deg):
    Ht = _turn_matrix(W, H, f_px, tilt_deg, azimuth_deg)
    quad = _map(_frame_corners(W, H), np.linalg.inv(Ht))       # what the camera sees, straight down
    x0, y0 = np.floor(quad.min(axis=0)) - 2
    x1, y1 = np.ceil(quad.max(axis=0)) + 2
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)   # canvas -> straight-down frame
    M = Ht @ T
    M /= M[2, 2]
    return {"H": M, "size": (int(x1 - x0), int(y1 - y0)), "quad": (quad - (x0, y0)).astype(np.float32),
            "tilt_deg": round(tilt_deg, 2), "azimuth_deg": round(azimuth_deg, 1),
            "magnification": round(magnification(M, W / 2, H / 2), 4)}


def zoom_view(W, H, f_px, tilt_deg, azimuth_deg):
    Ht = _turn_matrix(W, H, f_px, tilt_deg, azimuth_deg)

    def Z(s):
        return np.array([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)], [0, 0, 1]], np.float64)

    def fits(s):
        q = _map(_frame_corners(W, H), np.linalg.inv(Z(s) @ Ht))
        return bool((q[:, 0] >= -1e-6).all() and (q[:, 0] <= W + 1e-6).all()
                    and (q[:, 1] >= -1e-6).all() and (q[:, 1] <= H + 1e-6).all())

    lo, hi = 1.0, 1.0
    while not fits(hi):
        lo, hi = hi, hi * 1.25
        if hi > 8:
            raise ValueError(f"tilt {tilt_deg} deg is too large for this camera")
    for _ in range(40):                              # smallest zoom with no empty corner
        if hi - lo < 1e-5:
            break
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    M = Z(hi) @ Ht
    M /= M[2, 2]
    quad = _map(_frame_corners(W, H), np.linalg.inv(M)).astype(np.float32)
    return {"H": M, "size": (W, H), "quad": quad, "tilt_deg": round(tilt_deg, 2),
            "azimuth_deg": round(azimuth_deg, 1), "zoom": round(hi, 4),
            "magnification": round(magnification(M, W / 2, H / 2), 4)}


def sample_view(rng, cfg, W, H, f_px, mode="canvas", max_magnification=None):
    """One image's camera direction. cfg: {"prob": p, "tilt_deg": [lo, hi]} (None = off).

    max_magnification (zoom mode): the tilt is reduced until the centre is enlarged at
    most this much (generate_batch.py: a real photo must stay at >= the lowest altitude).
    """
    if not cfg or rng.random() >= float(cfg.get("prob", 0.0)):
        return nadir_view(W, H)
    tilt, azimuth = uniform(rng, cfg["tilt_deg"]), rng.uniform(0, 360)
    if mode == "canvas":
        return canvas_view(W, H, f_px, tilt, azimuth)
    while tilt >= 0.25:
        v = zoom_view(W, H, f_px, tilt, azimuth)
        if max_magnification is None or v["magnification"] <= max_magnification:
            return v
        tilt *= 0.8
    return nadir_view(W, H)


def is_tilted(view):
    return view["tilt_deg"] > 0


def warp_frame(img, view, W, H):
    if not is_tilted(view):
        return img
    return cv2.warpPerspective(img, view["H"], (W, H), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT)


def warp_points(pts, view):
    if not is_tilted(view):
        return np.asarray(pts, np.float32)
    return _map(pts, view["H"]).astype(np.float32)


def describe(view):
    """The JSON-friendly part of a view (no matrix)."""
    return {k: (list(v) if k == "size" else v) for k, v in view.items() if k not in ("H", "quad")}
