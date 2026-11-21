"""Check a generated run for label mistakes. Exit code 1 if anything is wrong.

    python tools/qa/validate_labels.py dataset/synthetic_n200_s42

Checks (quality_control in config/config.yaml):
  * every image has a label file, every label file has an image
  * image resolution = camera output (frames) or tile size (tiles)
  * every line has the right number of values, class id in range, coordinates in 0..1
  * detect box >= annotations.minimum_bbox_width_px x minimum_bbox_height_px
  * the detect box sits inside the extent of the oriented box (same flag, same class)
  * negative frames have an EMPTY label file
"""
import json
import sys
from pathlib import Path

import cv2
import yaml


def check_dataset(ds: Path, n_classes: int, size, min_box, errors: list):
    fmt = ds.name                                    # "obb" or "detect"
    n_img = n_obj = 0
    images = {p.relative_to(ds / "images").with_suffix("") for p in (ds / "images").rglob("*.jpg")}
    labels = {p.relative_to(ds / "labels").with_suffix("") for p in (ds / "labels").rglob("*.txt")}
    for extra in sorted(labels - images)[:10]:
        errors.append(f"{ds}: label without image: {extra}")
    for rel in sorted(images):
        n_img += 1
        if n_img <= 30 or n_img % 25 == 0:           # resolution: sample, reading 4K is slow
            h, w = cv2.imread(str(ds / "images" / f"{rel}.jpg"), cv2.IMREAD_REDUCED_GRAYSCALE_2).shape
            if (w * 2, h * 2) != tuple(size):
                errors.append(f"{ds}/images/{rel}: size {w*2}x{h*2}, expected {size[0]}x{size[1]}")
        lab = ds / "labels" / f"{rel}.txt"
        if not lab.exists():
            errors.append(f"{ds}: missing label for {rel}")
            continue
        for ln, line in enumerate(lab.read_text().splitlines(), 1):
            parts = line.split()
            where = f"{lab.relative_to(ds)}:{ln}"
            need = 9 if fmt == "obb" else 5
            if len(parts) != need:
                errors.append(f"{where}: expected {need} values, got {len(parts)}")
                continue
            c, vals = int(parts[0]), [float(v) for v in parts[1:]]
            n_obj += 1
            if not 0 <= c < n_classes:
                errors.append(f"{where}: class {c} out of range")
            if any(v < 0 or v > 1 for v in vals):
                errors.append(f"{where}: coordinate outside 0..1")
            if fmt == "detect":
                bw, bh = vals[2] * size[0], vals[3] * size[1]
                if bw < min_box[0] - 0.01 or bh < min_box[1] - 0.01:
                    errors.append(f"{where}: box {bw:.1f}x{bh:.1f} px below the minimum")
    return n_img, n_obj


def cross_check(run: Path, variant: str, errors: list):
    """The axis-aligned box must sit inside the extent of the oriented box."""
    obb_dir, det_dir = run / variant / "obb" / "labels", run / variant / "detect" / "labels"
    for f in obb_dir.rglob("*.txt"):
        o = [list(map(float, l.split())) for l in f.read_text().splitlines()]
        d = [list(map(float, l.split())) for l in (det_dir / f.relative_to(obb_dir)).read_text().splitlines()]
        if len(o) != len(d):
            errors.append(f"{variant}/{f.name}: {len(o)} OBB vs {len(d)} detect lines")
            continue
        for a, b in zip(o, d):
            xs, ys = a[1::2], a[2::2]
            x0, y0, x1, y1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
            tol = 0.002
            if int(a[0]) != int(b[0]) or x0 < min(xs) - tol or x1 > max(xs) + tol \
                    or y0 < min(ys) - tol or y1 > max(ys) + tol:
                errors.append(f"{variant}/{f.name}: detect box not inside the oriented box")


def check_negatives(run: Path, errors: list):
    """A frame where no flag was drawn must have an empty label file."""
    for m in (run / "metadata").glob("*.json"):
        meta = json.loads(m.read_text(encoding="utf-8"))
        lab = run / "frames" / "detect" / "labels" / meta["split"] / f"{meta['image']}.txt"
        if not meta["flags"] and lab.read_text().strip():
            errors.append(f"{lab.name}: negative frame has labels")


def main():
    run = Path(sys.argv[1])
    s = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
    n_classes = len(s["classes"])
    min_box = s["min_box_px"]
    errors = []
    for variant in ("frames", "tiles"):
        if not (run / variant).exists():
            continue
        size = (s["camera"]["width"], s["camera"]["height"]) if variant == "frames" \
            else (s["tiles"]["size"], s["tiles"]["size"])
        for fmt in ("detect", "obb"):
            ds = run / variant / fmt
            n_img, n_obj = check_dataset(ds, n_classes, size, min_box, errors)
            print(f"{variant:6s} {fmt:6s}: {n_img} images, {n_obj} objects")
            if not (ds / "data.yaml").exists():
                errors.append(f"{ds}: data.yaml missing")
        cross_check(run, variant, errors)
    check_negatives(run, errors)
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors[:50]:
            print("  " + e)
        sys.exit(1)
    print("\nAll labels valid.")


if __name__ == "__main__":
    main()
