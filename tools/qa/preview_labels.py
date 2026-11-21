"""Draw the labels on top of images, to check them by eye.

    python tools/qa/preview_labels.py dataset/synthetic_n200_s42            first 12 frames
    python tools/qa/preview_labels.py dataset/synthetic_n200_s42 --n 30 --tiles

Writes to <run>/previews/ (tiles go to <run>/previews/tiles/).
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from generator.preview import draw_preview  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tiles", action="store_true", help="preview tiles instead of full frames")
    a = ap.parse_args()
    run = Path(a.run)
    names = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))["classes"]
    variant = "tiles" if a.tiles else "frames"
    out = run / "previews" / ("tiles" if a.tiles else "")
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((run / variant / "obb" / "images").rglob("*.jpg"))[:a.n]
    for p in imgs:
        img = draw_preview(run, p.parent.name, p.stem, names, variant)
        cv2.imwrite(str(out / p.name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Wrote {len(imgs)} previews to {out}")


if __name__ == "__main__":
    main()
