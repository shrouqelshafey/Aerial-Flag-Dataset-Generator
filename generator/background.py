"""Stage 2 - background: a top-down ground photo resampled to the frame's GSD.

Every background file must say how much ground one of its pixels covers (GSD,
metres per pixel). That lives in assets/backgrounds/index.yaml:

    - {file: Rocky desert.png, gsd_m: 0.03, group: rocky_desert, type: desert_sand}

  group = where/when it was shot. Train/val/test are split BY GROUP so the same
          patch of ground never appears in both train and test.
  type  = one of synthesis.environment.background_types; frames pick a type with
          the probabilities in background_distribution.
  use: false  keeps a file in the folder but out of the generator.

The photo is first resized to the frame's GSD, then (if it is still smaller than
the frame) filled with soft-edged random patches of it (see _patch_fill), else
cropped at a random spot.
"""
from pathlib import Path

import cv2
import numpy as np
import yaml

from .noise import smooth_noise, uniform

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
COVER_STRETCH_MAX = 1.15


class BackgroundLibrary:
    def __init__(self, bg_dir, max_upsample=2.0, type_distribution=None):
        self.dir = Path(bg_dir)
        self.max_upsample = max_upsample
        self.type_dist = type_distribution or {}
        idx_path = self.dir / "index.yaml"
        if not idx_path.exists():
            raise FileNotFoundError(f"{idx_path} is missing: every background needs its GSD.")
        with open(idx_path, encoding="utf-8") as f:
            index = {e["file"]: e for e in (yaml.safe_load(f) or [])}

        self.items, self.unlisted, self.skipped = [], [], []
        for p in sorted(self.dir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            e = index.get(p.name)
            if e is None:
                self.unlisted.append(p.name)       # no GSD known -> cannot be scaled correctly
                continue
            if not e.get("use", True):
                self.skipped.append(p.name)
                continue
            self.items.append({"path": p, "file": p.name, "gsd_m": float(e["gsd_m"]),
                               "group": str(e.get("group", p.stem)),
                               "type": str(e.get("type", "unknown"))})
        if not self.items:
            raise FileNotFoundError(f"No usable background in {self.dir} (check index.yaml).")
        self._cache = {}

    def groups(self):
        return sorted({it["group"] for it in self.items})

    def _load(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            if len(self._cache) > 6:               # keep memory bounded
                self._cache.pop(next(iter(self._cache)))
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Cannot read {path}")
            self._cache[path] = img
        return self._cache[path]

    def _pick(self, rng, groups, target_gsd):
        pool = [it for it in self.items if it["group"] in groups]
        sharp = [it for it in pool if target_gsd * self.max_upsample >= it["gsd_m"]]
        cands = sharp or sorted(pool, key=lambda it: it["gsd_m"])[:1]
        # Weight by background_distribution: P(type) shared equally by the files of that type.
        counts = {}
        for it in cands:
            counts[it["type"]] = counts.get(it["type"], 0) + 1
        w = np.array([self.type_dist.get(it["type"], 0.0) / counts[it["type"]] for it in cands])
        if w.sum() <= 0:
            w = np.ones(len(cands))
        return cands[int(rng.choice(len(cands), p=w / w.sum()))]

    def sample(self, rng, groups, target_gsd, out_w, out_h, cfg_bg):
        """Return (float32 BGR image out_h x out_w at target_gsd, info dict)."""
        it = self._pick(rng, groups, target_gsd)
        src = self._load(it["path"])

        scale = target_gsd / it["gsd_m"]           # source pixels per output pixel
        rot90 = int(rng.integers(4))
        sh, sw = src.shape[:2]
        rw, rh = (sh, sw) if rot90 % 2 else (sw, sh)            # size after the turn
        # Almost big enough (within 15 %)? Enlarge a little to cover the frame: one
        # clean photo beats a few huge patches with visible seams. The error is far
        # smaller than the uncertainty of the gsd_m estimates; it is recorded.
        cover = max(out_w / (rw / scale), out_h / (rh / scale))
        stretch = cover if 1.0 < cover <= COVER_STRETCH_MAX else 1.0
        scale /= stretch
        sw = max(1, int(np.ceil(sw / scale - 1e-6)))
        sh = max(1, int(np.ceil(sh / scale - 1e-6)))
        interp = cv2.INTER_AREA if scale > 1 else cv2.INTER_CUBIC
        img = cv2.resize(src, (sw, sh), interpolation=interp)   # now at the frame's GSD
        img = np.rot90(img, rot90)
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if img.shape[1] >= out_w and img.shape[0] >= out_h:
            x0 = int(rng.integers(0, img.shape[1] - out_w + 1))
            y0 = int(rng.integers(0, img.shape[0] - out_h + 1))
            img, patches = img[y0:y0 + out_h, x0:x0 + out_w], 0
        else:
            img, patches = _patch_fill(img, out_w, out_h, rng)
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0

        img = _photometric_jitter(img, rng, cfg_bg)
        marked = False
        if rng.random() < cfg_bg["markings_prob"]:
            img = paint_markings(img, target_gsd, rng, cfg_bg)
            marked = True
        info = {"file": it["file"], "group": it["group"], "type": it["type"],
                "src_gsd_m": it["gsd_m"], "resample_factor": round(scale, 3),
                "upsampled": scale < 1, "cover_stretch": round(stretch, 3),
                "fill_patches": patches, "rot90": rot90, "markings": marked}
        return img, info


def _feather_mask(p, feather):
    """Square mask: 1 inside, smooth fall to 0 over `feather` px at the edges."""
    r = np.minimum(np.arange(p), np.arange(p)[::-1]).astype(np.float32)
    r = np.clip(r / max(feather, 1), 0, 1)
    r = r * r * (3 - 2 * r)                          # smoothstep
    return np.minimum(r[:, None], r[None, :])[..., None]


def _patch_fill(src, out_w, out_h, rng):
    """Fill an out_w x out_h frame from a SMALLER photo without visible mirroring.

    Mirroring a small photo gives a kaleidoscope (a pattern a detector can learn).
    Instead: square patches cut at random spots of the photo, each turned/flipped at
    random, are laid on a jittered grid in random order with soft edges, so the
    seams are irregular. Under the patches is a mirrored copy, so no gap can remain.
    """
    h, w = src.shape[:2]
    base = src
    while base.shape[1] < out_w:
        base = np.concatenate([base, base[:, ::-1]], axis=1)
    while base.shape[0] < out_h:
        base = np.concatenate([base, base[::-1]], axis=0)
    canvas = base[:out_h, :out_w].astype(np.float32)

    p = max(8, int(0.9 * min(h, w)))                 # patch side
    feather = max(2, p // 8)
    step = p - 2 * feather
    mask = _feather_mask(p, feather)
    spots = [(x, y) for y in range(-feather, out_h, step) for x in range(-feather, out_w, step)]
    order = rng.permutation(len(spots))
    for i in order:
        x, y = spots[i]
        x += int(rng.integers(-feather // 2, feather // 2 + 1))
        y += int(rng.integers(-feather // 2, feather // 2 + 1))
        sx, sy = int(rng.integers(0, w - p + 1)), int(rng.integers(0, h - p + 1))
        patch = np.rot90(src[sy:sy + p, sx:sx + p], int(rng.integers(4)))
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        # clip the patch to the frame
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + p, out_w), min(y + p, out_h)
        if x1 <= x0 or y1 <= y0:
            continue
        pt = patch[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
        m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        region = canvas[y0:y1, x0:x1]
        region += (pt - region) * m
    return canvas, len(spots)


def _photometric_jitter(img, rng, cfg_bg):
    b = uniform(rng, cfg_bg["brightness"])
    c = uniform(rng, cfg_bg["contrast"])
    cast = 1.0 + rng.uniform(-cfg_bg["color_cast"], cfg_bg["color_cast"], size=3)
    mean = img.mean()
    img = (img - mean) * c + mean
    img = img * b * cast.astype(np.float32)
    return np.clip(img, 0, 1)


def paint_markings(img, gsd, rng, cfg_bg):
    """Paint worn white/yellow runway-style lines. Sizes are in metres -> pixels."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    n = int(rng.integers(1, 4))
    angle = rng.uniform(0, np.pi)                  # lines are parallel, like a runway
    d = np.array([np.cos(angle), np.sin(angle)])
    nrm = np.array([-d[1], d[0]])
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    for i in range(n):
        width_px = uniform(rng, cfg_bg["marking_width_m"]) / gsd
        length_px = uniform(rng, cfg_bg["marking_length_m"]) / gsd
        offset = (i - n / 2) * rng.uniform(3, 8) / gsd
        c = np.array([cx, cy]) + nrm * offset + d * rng.uniform(-0.3, 0.3) * length_px
        pts = np.array([c - d * length_px / 2 - nrm * width_px / 2,
                        c + d * length_px / 2 - nrm * width_px / 2,
                        c + d * length_px / 2 + nrm * width_px / 2,
                        c - d * length_px / 2 + nrm * width_px / 2])
        cv2.fillPoly(mask, [np.round(pts * 4).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=2)
    wear = np.clip(0.75 + 0.35 * smooth_noise(h, w, 6, rng, octaves=3), 0, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.7) * wear * rng.uniform(0.5, 0.9)
    color = np.array([0.88, 0.88, 0.86] if rng.random() < 0.7 else [0.15, 0.70, 0.85], np.float32)
    return img * (1 - mask[..., None]) + color * mask[..., None]
