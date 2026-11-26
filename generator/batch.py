"""Batch recipe: a folder of real background photos -> a small, exactly balanced dataset.

    200 backgrounds -> 50 stay empty (1 image each) + 150 x 3 images with 1-3 flags = 500

plan() decides everything up front, so the numbers are exact, not "about":
  * empty backgrounds: negative_fraction of them, spread evenly over train/valid/test
  * flags per image:   equal shares of 1, 2 and 3 (flags_per_image), shuffled at random
  * Egypt:             in exactly focus_share of the images that have flags, chosen at random
  * other countries:   never twice in one image; the least-used one goes first (ties
                       broken at random), so the other countries get about the same count
  * split:             BY BACKGROUND - all images of one photo are in the same split, so
                       the model is never tested on ground it was trained on

Background scale (how many pixels a 2 m flag gets), per photo, first match wins
in <backgrounds_dir>/index.yaml (optional):
    - {file: IMG_0012.jpg, gsd_m: 0.031}           metres per pixel, if known
    - {file: IMG_0013.jpg, altitude_m: 80}         flown altitude, camera HFOV from config
    - {file: IMG_0014.jpg, group: site2, use: false}
A photo that is not listed is taken as a full camera frame (HFOV from config) at a
random altitude in flight.altitude_m. group = same place/flight (default: the file
itself); photos of one group stay in one split.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .split import SPLITS, assign_groups

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Background:
    path: Path
    group: str
    gsd_m: float = None        # known GSD
    altitude_m: float = None   # else: GSD from altitude + HFOV once the width is known
    split: str = ""
    negative: bool = False
    info: dict = field(default_factory=dict)

    def gsd(self, width_px, hfov_deg):
        if self.gsd_m:
            return float(self.gsd_m)
        return 2 * self.altitude_m * math.tan(math.radians(hfov_deg) / 2) / width_px


def load_backgrounds(bg_dir, altitude_range, seed):
    bg_dir = Path(bg_dir)
    if not bg_dir.is_dir():
        raise FileNotFoundError(f"{bg_dir} does not exist - put the background photos there "
                                "(or pass --backgrounds).")
    idx_path = bg_dir / "index.yaml"
    index = {}
    if idx_path.exists():
        with open(idx_path, encoding="utf-8") as f:
            index = {e["file"]: e for e in (yaml.safe_load(f) or [])}
    rng = np.random.default_rng([seed, 11])
    out = []
    for p in sorted(bg_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        e = index.get(p.name, {})
        if not e.get("use", True):
            continue
        alt = rng.uniform(*altitude_range)        # drawn for every file: stable if index changes
        b = Background(p, str(e.get("group", p.stem)), split=str(e.get("split", "")))
        if e.get("gsd_m"):
            b.gsd_m, b.info["scale_from"] = float(e["gsd_m"]), "index gsd_m"
        elif e.get("altitude_m"):
            b.altitude_m, b.info["scale_from"] = float(e["altitude_m"]), "index altitude_m"
        else:
            b.altitude_m, b.info["scale_from"] = float(alt), "random altitude (not in index.yaml)"
        out.append(b)
    if not out:
        raise FileNotFoundError(f"No background images in {bg_dir}")
    return out


def _quota(shares, n, rng):
    """Exactly n items split by `shares` (largest remainder), in random order."""
    keys = sorted(shares)
    p = np.array([float(shares[k]) for k in keys])
    p = p / p.sum() * n
    cnt = np.floor(p).astype(int)
    for i in np.argsort(-(p - cnt))[: n - cnt.sum()]:
        cnt[i] += 1
    out = np.repeat(keys, cnt)
    rng.shuffle(out)
    return [int(v) for v in out]


def plan(backgrounds, b, split_ratios, drawable, focus_id, seed):
    """Fill in split + negative on each background and return the list of images."""
    rng = np.random.default_rng([seed, 2024])

    # 1. split by group (one place never in two splits)
    groups = sorted({bg.group for bg in backgrounds})
    fixed = {bg.group: bg.split for bg in backgrounds if bg.split}     # `split:` in index.yaml
    g2s = {}
    for sp, gs in assign_groups(groups, split_ratios, seed, fixed).items():
        for g in gs:
            g2s.setdefault(g, sp)                 # < 3 groups: everything goes to train
    for bg in backgrounds:
        bg.split = g2s[bg.group]

    # 2. empty backgrounds, spread evenly over the splits
    n_neg = int(round(b["negative_fraction"] * len(backgrounds)))
    keys = []
    for sp in SPLITS:
        ids = [i for i, bg in enumerate(backgrounds) if bg.split == sp]
        for k, i in enumerate(rng.permutation(ids)):
            keys.append(((k + rng.random()) / len(ids), int(i)))
    for _, i in sorted(keys)[:n_neg]:
        backgrounds[i].negative = True

    # 3. the images
    images = []
    for i, bg in enumerate(backgrounds):
        if bg.negative:
            images.append({"bg": i, "variant": 0, "classes": []})
        else:
            for v in range(int(b["images_per_background"])):
                images.append({"bg": i, "variant": v + 1, "classes": None})
    pos = [im for im in images if im["classes"] is None]

    counts = _quota(b["flags_per_image"], len(pos), rng)
    has_focus = set()
    if focus_id is not None:
        n_focus = int(round(float(b["focus_share"]) * len(pos)))
        has_focus = set(rng.choice(len(pos), n_focus, replace=False).tolist())
    others = [c for c in drawable if c != focus_id]
    used = {c: 0 for c in others}
    for j in rng.permutation(len(pos)):
        cls = [focus_id] if j in has_focus else []
        need = min(counts[j] - len(cls), len(others))
        order = sorted(others, key=lambda c: (used[c], rng.random()))
        for c in order[:need]:
            used[c] += 1
            cls.append(c)
        pos[j]["classes"] = [int(c) for c in cls]
    return images
