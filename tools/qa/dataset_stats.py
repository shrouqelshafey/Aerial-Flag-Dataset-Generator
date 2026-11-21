"""Print the numbers that tell you if a run looks right.

    python tools/qa/dataset_stats.py dataset/synthetic_n200_s42

  * flag size in pixels per training.object_size_groups (small / medium / large)
    - must match the camera maths: 45 px at 100 m, 90 px at 50 m
  * objects per class (should be roughly flat for the classes that have an image)
  * flags per frame, negative frames, split sizes, background types
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml


def bar(n, top, width=40):
    return "#" * int(round(width * n / max(top, 1)))


def main():
    run = Path(sys.argv[1])
    s = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
    names = s["classes"]
    sizes, per_class, per_frame, splits, bg = [], Counter(), Counter(), Counter(), Counter()
    for split in ("train", "val", "test"):
        for lab in (run / "frames" / "obb" / "labels" / split).glob("*.txt"):
            meta = json.loads((run / "metadata" / f"{lab.stem}.json").read_text(encoding="utf-8"))
            W, H = meta["camera"]["width"], meta["camera"]["height"]
            lines = lab.read_text().splitlines()
            per_frame[len(lines)] += 1
            splits[split] += 1
            bg[meta["background"]["type"]] += 1
            for l in lines:
                v = list(map(float, l.split()))
                per_class[names[int(v[0])]] += 1
                p = [(v[1 + 2 * i] * W, v[2 + 2 * i] * H) for i in range(4)]
                sizes.append(max(math.dist(p[0], p[1]), math.dist(p[1], p[2])))

    print(f"Frames per split: {dict(splits)}")
    print("\nFlags per frame (0 = negative image):")
    for k in sorted(per_frame):
        print(f"  {k}: {per_frame[k]:5d} {bar(per_frame[k], max(per_frame.values()))}")
    print("\nBackground types:")
    for k, n in bg.most_common():
        print(f"  {k:26s} {n:5d} {bar(n, max(bg.values()))}")
    if sizes:
        srt = sorted(sizes)
        print(f"\nFlag long side (px): min {srt[0]:.1f}  median {srt[len(srt)//2]:.1f}  max {srt[-1]:.1f}")
        groups = {"small": [45, 59], "medium": [60, 74], "large": [75, 90]}   # training.object_size_groups
        for g, (lo, hi) in groups.items():
            n = sum(lo <= x < hi + 1 for x in sizes)
            print(f"  {g:7s} {lo}-{hi} px {n:5d} {bar(n, len(sizes))}")
        n_out = sum(not 45 <= x < 91 for x in sizes)
        if n_out:
            print(f"  outside 45-90 px: {n_out} (oriented box of a flag cut by the frame edge is smaller)")
    print("\nObjects per class:")
    for n in names:
        if per_class[n]:
            print(f"  {n:22s} {per_class[n]:5d} {bar(per_class[n], max(per_class.values()))}")
    missing = [n for n in names if not per_class[n]]
    if missing:
        print(f"  no objects: {', '.join(missing)}")


if __name__ == "__main__":
    main()
