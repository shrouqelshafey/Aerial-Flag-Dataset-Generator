"""Turn a generate.py run into a folder Roboflow imports with its split kept.

    python tools/export_roboflow.py dataset/train_3000_s42            # full frames, YOLO boxes
    python tools/export_roboflow.py dataset/train_3000_s42 --obb      # oriented boxes
    python tools/export_roboflow.py dataset/train_3000_s42 --zip      # also a .zip to drag in

Writes <run>/roboflow_detect/{train,valid,test}/{images,labels} + data.yaml (Roboflow calls
the validation split "valid"). Images are hard-linked, so no extra disk space is used.
Then upload it in the Roboflow web app, or with tools/upload_roboflow.py.
"""
import argparse
import os
import shutil
from pathlib import Path

import yaml

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="a generate.py run folder, e.g. dataset/train_3000_s42")
    ap.add_argument("--obb", action="store_true", help="oriented boxes instead of axis-aligned")
    ap.add_argument("--zip", action="store_true", help="also write the folder as a .zip")
    a = ap.parse_args()

    run = Path(a.run)
    kind = "obb" if a.obb else "detect"
    src = run / "frames" / kind
    out = run / f"roboflow_{kind}"
    if not (src / "data.yaml").exists():
        raise SystemExit(f"{src} has no data.yaml - is {run} a generate.py run?")
    if out.exists():
        shutil.rmtree(out)

    names = yaml.safe_load(open(src / "data.yaml", encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    counts = {}
    for sp, rf in SPLITS.items():
        imgs = sorted((src / "images" / sp).glob("*.jpg"))
        (out / rf / "images").mkdir(parents=True)
        (out / rf / "labels").mkdir(parents=True)
        for im in imgs:
            link(im, out / rf / "images" / im.name)
            lab = src / "labels" / sp / f"{im.stem}.txt"
            shutil.copy2(lab, out / rf / "labels" / lab.name)   # empty file = image without flags
        counts[rf] = len(imgs)

    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": "train/images", "val": "valid/images", "test": "test/images",
                        "nc": len(names), "names": list(names)}, f, sort_keys=False)
    print(f"{out}: {counts}, {len(names)} classes")

    if a.zip:
        z = shutil.make_archive(str(out), "zip", out)
        print(f"zip: {z} ({Path(z).stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
