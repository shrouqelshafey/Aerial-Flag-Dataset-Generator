"""Draw the labels FROM THE LABEL FILES on top of a frame, to check them by eye.

Reading the written .txt files (not the generator's own numbers) means a preview
shows exactly what YOLO will train on.
  green polygon = oriented box (obb)     thin blue rectangle = detect box
  text = country, class id, size in pixels; header = altitude, GSD, background
"""
import json
from pathlib import Path

import cv2
import numpy as np

GREEN, BLUE, WHITE, BLACK = (60, 220, 60), (255, 160, 40), (255, 255, 255), (0, 0, 0)


def _text(img, s, org, scale=0.8, color=WHITE):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_preview(run: Path, split: str, stem: str, names, variant="frames"):
    base = Path(run) / variant
    img = cv2.imread(str(base / "obb" / "images" / split / f"{stem}.jpg"))
    h, w = img.shape[:2]
    obb = (base / "obb" / "labels" / split / f"{stem}.txt").read_text().split("\n")
    det = (base / "detect" / "labels" / split / f"{stem}.txt").read_text().split("\n")
    for line in filter(None, det):
        c, cx, cy, bw, bh = map(float, line.split())
        x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x0, y0), (x1, y1), BLUE, 1, cv2.LINE_AA)
    for line in filter(None, obb):
        v = list(map(float, line.split()))
        pts = np.array(v[1:], np.float32).reshape(4, 2) * [w, h]
        cv2.polylines(img, [pts.round().astype(np.int32)], True, GREEN, 2, cv2.LINE_AA)
        e1, e2 = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])
        x, y = pts[:, 0].min(), pts[:, 1].min()
        _text(img, f"{names[int(v[0])]} ({int(v[0])}) {max(e1, e2):.0f}x{min(e1, e2):.0f}px",
              (int(x), max(20, int(y) - 8)), 0.7, GREEN)
    meta_path = Path(run) / "metadata" / f"{stem}.json"
    if variant == "frames" and meta_path.exists():
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        _text(img, f"{stem}  |  {split}  |  altitude {m['altitude_m']:.1f} m  |  GSD {m['gsd_m']*100:.2f} cm/px"
                   f"  |  2 m flag = {m['flag_long_px']:.0f} px  |  ground: {m['background']['type']}"
                   f" ({m['background']['file']})", (20, 45), 1.1)
    return img
