"""Make stand-in "drone photos" to try generate_batch.py before the real photos arrive.

    python tools/make_demo_backgrounds.py --n 200
    python generate_batch.py --backgrounds assets/batch_backgrounds_demo

Each demo photo is a full 4000 x 3000 frame built from the ground photos in
assets/backgrounds (at a random altitude of flight.altitude_m, with painted runway
lines on some), plus an index.yaml with its exact GSD and its source group, so the
split by background still keeps one source photo in one split.
These are NOT real drone photos: replace them with real ones for the real test.
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.background import BackgroundLibrary  # noqa: E402
from generator.settings import ROOT, load_settings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "assets" / "batch_backgrounds_demo"))
    ap.add_argument("--size", type=int, nargs=2, default=[4000, 3000], metavar=("W", "H"))
    ap.add_argument("--markings", type=float, default=0.4, help="share with painted runway lines")
    a = ap.parse_args()

    s = load_settings()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bg = s["backgrounds"]
    lib = BackgroundLibrary(bg["dir"], bg["max_upsample"], bg["type_distribution"])
    jitter = dict(bg["jitter"], markings_prob=a.markings)
    W, H = a.size
    tan = math.tan(math.radians(s["camera"]["hfov_deg"]) / 2)
    lines = ["# Demo backgrounds made by tools/make_demo_backgrounds.py (not real photos)"]
    for i in range(a.n):
        rng = np.random.default_rng([s["seed"], 77, i])
        alt = rng.uniform(*s["altitude_m"])
        gsd = 2 * alt * tan / W                  # same camera HFOV, 4000 px wide
        img, info = lib.sample(rng, lib.groups(), gsd, W, H, jitter)
        name = f"demo_{i:04d}.jpg"
        cv2.imwrite(str(out / name), (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        lines.append(f"- {{file: {name}, gsd_m: {gsd:.6f}, group: {info['group']}}}"
                     f"   # {alt:.0f} m, from {info['file']}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{a.n}", flush=True)
    (out / "index.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {a.n} demo backgrounds + index.yaml to {out}")


if __name__ == "__main__":
    main()
