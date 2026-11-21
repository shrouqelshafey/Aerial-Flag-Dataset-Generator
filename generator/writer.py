"""Writes images + both label formats in the folder layout Ultralytics expects.

dataset/<run>/
  frames/detect/{images,labels}/{train,val,test}/   YOLO boxes: class cx cy w h
  frames/obb/{images,labels}/{train,val,test}/      YOLO-OBB:   class x1 y1 ... x4 y4
  tiles/detect/...   tiles/obb/...                  the same, cut into 1024 px tiles
  metadata/<image>.json        everything needed to explain / reproduce one frame
  previews/                    frames with their labels drawn on top
  generation_logs/generation_report.csv   one row per flag (and per empty frame)
  config_used.yaml  summary.json
Each <variant>/<format>/ folder has its own data.yaml. The detect and obb folders
share the same image files (hard links), so the images take disk space once.
"""
import json
import os
import shutil
from pathlib import Path

import cv2

from .split import SPLITS

FORMATS = ("detect", "obb")


class DatasetWriter:
    def __init__(self, run_dir, class_names, quality=92, variants=("frames", "tiles")):
        self.run = Path(run_dir)
        self.names = list(class_names)
        self.quality = int(quality)
        self.variants = variants
        for v in variants:
            for f in FORMATS:
                for kind in ("images", "labels"):
                    for s in SPLITS:
                        (self.run / v / f / kind / s).mkdir(parents=True, exist_ok=True)
        for d in ("metadata", "previews", "generation_logs"):
            (self.run / d).mkdir(parents=True, exist_ok=True)

    def save(self, variant, split, stem, img, obb_lines, det_lines):
        base = self.run / variant
        img_det = base / "detect" / "images" / split / f"{stem}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise IOError(f"JPEG encoding failed for {stem}")
        img_det.write_bytes(buf.tobytes())
        img_obb = base / "obb" / "images" / split / f"{stem}.jpg"
        if img_obb.exists():
            img_obb.unlink()
        try:
            os.link(img_det, img_obb)            # same file, no extra disk space
        except OSError:
            shutil.copyfile(img_det, img_obb)
        # An image with no flags still gets an (empty) label file.
        for fmt, lines in (("obb", obb_lines), ("detect", det_lines)):
            (base / fmt / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def save_meta(self, stem, meta):
        with open(self.run / "metadata" / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)

    def write_yaml(self):
        for v in self.variants:
            for f in FORMATS:
                root = (self.run / v / f).resolve()
                lines = [f"# Ultralytics dataset ({v}, {'oriented' if f == 'obb' else 'axis-aligned'} boxes)",
                         f"path: {root.as_posix()}",
                         "train: images/train", "val: images/val", "test: images/test",
                         "names:"]
                lines += [f"  {i}: {n}" for i, n in enumerate(self.names)]
                (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
