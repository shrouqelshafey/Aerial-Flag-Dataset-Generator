"""Stage 8 - labels.

Every flag is stored as its 4 corners in frame pixels. For any window (the full
frame or a 1024x1024 tile) we:
  1. clip the corner polygon to the window,
  2. drop it if less than min_visible of its area is inside,
  3. OBB  = cv2.minAreaRect of the visible polygon (4 corners),
     AABB = min/max of the visible polygon,
  4. divide by the window size so every number is 0..1.
"""
import cv2
import numpy as np


def _hull(poly):
    return cv2.convexHull(np.asarray(poly, np.float32)).reshape(-1, 2)


def clip_to_window(poly, x0, y0, x1, y1):
    """Return (visible polygon or None, visible area, full area)."""
    hull = _hull(poly)
    full = float(cv2.contourArea(hull))
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    area, inter = cv2.intersectConvexConvex(hull, rect)
    if inter is None or area <= 0:
        return None, 0.0, full
    return inter.reshape(-1, 2), float(area), full


def window_labels(objects, x0, y0, x1, y1, min_visible):
    """objects: list of {"cls": int, "corners": (4,2)}. Returns (kept, dropped_partial)."""
    kept, dropped = [], 0
    for o in objects:
        vis_poly, area, full = clip_to_window(o["corners"], x0, y0, x1, y1)
        if vis_poly is None:
            continue
        frac = area / max(full, 1e-6)
        if frac < min_visible:
            dropped += 1
            continue
        box = cv2.boxPoints(cv2.minAreaRect(vis_poly.astype(np.float32)))
        box = box - np.array([x0, y0], np.float32)
        vp = vis_poly - np.array([x0, y0], np.float32)
        kept.append({"cls": int(o["cls"]), "obb": box,
                     "aabb": (float(vp[:, 0].min()), float(vp[:, 1].min()),
                              float(vp[:, 0].max()), float(vp[:, 1].max())),
                     "visible": round(frac, 3)})
    return kept, dropped


def obb_line(lab, w, h):
    pts = lab["obb"] / np.array([w, h], np.float32)
    pts = np.clip(pts, 0, 1)
    return f"{lab['cls']} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1))


def detect_line(lab, w, h):
    ax0, ay0, ax1, ay1 = lab["aabb"]
    ax0, ax1 = max(0.0, ax0), min(float(w), ax1)
    ay0, ay1 = max(0.0, ay0), min(float(h), ay1)
    cx, cy = (ax0 + ax1) / 2 / w, (ay0 + ay1) / 2 / h
    bw, bh = (ax1 - ax0) / w, (ay1 - ay0) / h
    return f"{lab['cls']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
