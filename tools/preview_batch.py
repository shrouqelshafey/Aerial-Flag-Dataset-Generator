"""Draw previews of a generate_batch.py run, a few images of every look.

    python tools/preview_batch.py dataset/batch_demo
    python tools/preview_batch.py dataset/batch_demo --per-look 6 --seed 3

Reads the written label files (what Roboflow / YOLO will see) and writes
<run>/previews/<look>/<image>.jpg: green oriented boxes, country names, and a
strip of 1:1 zooms on every flag. Looks: pale_sand, airfield, none (photo as is),
empty (no flag).
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_batch import preview  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--per-look", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="change it to see other images")
    a = ap.parse_args()

    run = Path(a.run)
    names = yaml.safe_load((run / "roboflow_obb" / "data.yaml").read_text(encoding="utf-8"))["names"]
    rows = list(csv.DictReader(open(run / "generation_report.csv", encoding="utf-8")))
    groups = {}
    for r in rows:
        key = "empty" if r["n_flags"] == "0" else (r["look"] or "none")
        groups.setdefault(key, []).append(r)

    out_root = run / "previews"
    for d in out_root.glob("*"):
        if d.is_dir():
            shutil.rmtree(d)
    rng = np.random.default_rng(a.seed)
    total = 0
    for key, rs in sorted(groups.items()):
        out = out_root / key
        out.mkdir(parents=True, exist_ok=True)
        # prefer images with more flags, so the zooms show more
        rs = sorted(rs, key=lambda r: (-int(r["n_flags"]), rng.random()))
        for r in rs[:a.per_look]:
            stem = r["image"][:-4]
            img = cv2.imread(str(run / "roboflow_obb" / r["split"] / "images" / r["image"]))
            H, W = img.shape[:2]
            kept = []
            for line in (run / "roboflow_obb" / r["split"] / "labels" / f"{stem}.txt").read_text().splitlines():
                v = line.split()
                pts = np.array(v[1:], np.float32).reshape(4, 2) * [W, H]
                kept.append({"cls": int(v[0]), "obb": pts,
                             "aabb": (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())})
            title = (f"{stem} | {r['split']} | look: {key} | GSD {r['gsd_cm']} cm/px | "
                     f"2 m flag = {float(r['flag_long_px']):.0f} px | {r['countries'] or 'no flag'}")
            cv2.imwrite(str(out / r["image"]), preview(img, kept, names, title), [cv2.IMWRITE_JPEG_QUALITY, 90])
            total += 1
        print(f"  {key:10s} {min(len(rs), a.per_look)} of {len(rs)} images -> {out}")
    print(f"Wrote {total} previews to {out_root}")


if __name__ == "__main__":
    main()
