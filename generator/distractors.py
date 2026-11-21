"""Stage 6 - distractors: things that look a bit like a flag but are NOT labelled.

Two sources:
  * "image": the photos in assets/distractors (tarpaulin, fabric, plastic sheet),
    each cropped to its clean centre and given a real size by index.yaml;
  * drawn kinds: plain tarps, flag-coloured cloth, bags, cars, striped sheets.

They go through the same renderer and compositor as flags, so the only thing that
separates a flag from a distractor is its content, never "how it was pasted".
"""
from pathlib import Path

import numpy as np
import yaml

from .flag_render import load_texture
from .noise import smooth_noise

# BGR colours (0..1)
TARP = [(0.70, 0.40, 0.10), (0.20, 0.50, 0.20), (0.10, 0.45, 0.90), (0.55, 0.55, 0.55),
        (0.85, 0.85, 0.85), (0.20, 0.35, 0.50)]
FLAGLIKE = [(0.15, 0.10, 0.80), (0.90, 0.90, 0.90), (0.10, 0.10, 0.10), (0.15, 0.55, 0.10),
            (0.10, 0.75, 0.90), (0.60, 0.30, 0.10)]
CAR = [(0.92, 0.92, 0.92), (0.70, 0.70, 0.72), (0.08, 0.08, 0.08), (0.15, 0.15, 0.70),
       (0.55, 0.25, 0.10)]


def load_distractor_images(folder):
    """Read assets/distractors/index.yaml -> list of {name, tex, long_m}."""
    folder = Path(folder)
    idx = folder / "index.yaml"
    if not idx.exists():
        return []
    out = []
    for e in yaml.safe_load(idx.read_text(encoding="utf-8")) or []:
        tex = load_texture(folder / e["file"])
        h, w = tex.shape[:2]
        x0, y0, x1, y1 = e.get("crop", [0, 0, 1, 1])        # fractions of width / height
        tex = tex[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if tex.shape[0] > tex.shape[1]:                     # long side horizontal
            tex = np.ascontiguousarray(np.transpose(tex, (1, 0, 2)))
        out.append({"name": e["file"], "tex": tex, "long_m": e["long_m"]})
    return out


def _plain(color, h, w, rng, grain=0.03):
    t = np.ones((h, w, 3), np.float32) * np.array(color, np.float32)
    return np.clip(t + grain * smooth_noise(h, w, 3, rng)[..., None], 0, 1)


def make_distractor(kind, rng, images=()):
    """Return (texture BGR float, (long_m, short_m), rigid, name)."""
    if kind == "image" and images:
        it = images[int(rng.integers(len(images)))]
        L = rng.uniform(*it["long_m"])
        S = L * it["tex"].shape[0] / it["tex"].shape[1]
        return it["tex"], (L, S), False, it["name"]
    if kind == "tarp" or kind == "image":
        L = rng.uniform(1.5, 4.0); S = L * rng.uniform(0.4, 1.0)
        return _plain(TARP[rng.integers(len(TARP))], 64, 128, rng), (L, S), False, "tarp"
    if kind == "cloth":            # flag-coloured plain cloth: the hard negative
        L = rng.uniform(0.8, 2.5); S = L * rng.uniform(0.3, 1.0)
        return _plain(FLAGLIKE[rng.integers(len(FLAGLIKE))], 64, 128, rng), (L, S), False, kind
    if kind == "bag":
        L = rng.uniform(0.4, 1.0); S = L * rng.uniform(0.5, 0.9)
        c = np.array(TARP[rng.integers(len(TARP))]) * rng.uniform(0.3, 1.0)
        return _plain(c, 32, 64, rng, 0.05), (L, S), False, kind
    if kind == "car":
        L = rng.uniform(4.2, 4.9); S = rng.uniform(1.7, 1.95)
        tex = _plain(CAR[rng.integers(len(CAR))], 40, 100, rng, 0.02)
        tex[6:34, 28:42] *= 0.25       # windscreen
        tex[6:34, 70:80] *= 0.30       # rear window
        tex[:, :] *= 1 - 0.15 * np.abs(np.linspace(-1, 1, 40))[:, None, None]  # roof highlight
        return tex, (L, S), True, kind
    # striped_sheet: many stripes, wrong aspect - a towel, a mat, a beach sheet
    L = rng.uniform(1.0, 2.5); S = L * rng.uniform(0.4, 1.0)
    n = int(rng.integers(5, 10))
    cols = [FLAGLIKE[i] for i in rng.integers(len(FLAGLIKE), size=n)]
    tex = np.zeros((n * 8, 128, 3), np.float32)
    for i, c in enumerate(cols):
        tex[i * 8:(i + 1) * 8] = c
    if rng.random() < 0.5:
        tex = np.ascontiguousarray(np.transpose(tex, (1, 0, 2)))
    return tex, (L, S), False, "striped_sheet"


def sample_kind(rng, kinds: dict):
    names = list(kinds)
    p = np.array([kinds[k] for k in names], float)
    return names[int(rng.choice(len(names), p=p / p.sum()))]
