"""Cut large drone maps (orthomosaics) into background FRAMES and ground TEXTURES.

    python tools/cut_orthomosaics.py                         # defaults below
    python tools/cut_orthomosaics.py --frames 300 --textures-per-map 4 --seed 3

Reads assets/background_sources.yaml and the GeoTIFFs in assets/batch_backgrounds_sources/.

1. FRAMES for generate_batch.py  ->  assets/batch_backgrounds/<name>_cNN_KK.jpg (+ index.yaml)
   Each frame is what the SIYI A8 mini would see: 3840 x 2160 at a random altitude in
   flight.altitude_m. The ground footprint (2 * altitude * tan(HFOV / 2) wide) is cut
   from the map and resized to the frame, so index.yaml gets the exact gsd_m. A map
   coarser than the frame GSD is never enlarged more than --max-upsample: the altitude
   range of that map starts higher instead.
2. TEXTURES for generate.py      ->  assets/backgrounds/<name>_tNN.jpg (+ entries in its index.yaml)
   The biggest clean square in every cell, plus the biggest clean rectangles in the
   ground outside the cells that still cover a --strip-altitude frame (group <name>_rNN,
   always train). Stored at the sharpest GSD the generator needs (the 50 m frame), so the
   generator only ever shrinks them.

No ground in two splits: each map is divided into square CELLS a bit larger than the
biggest footprint (a 100 m frame). Every cut lies inside one cell, and the cell is the
split group (<name>_cNN), so train / valid / test never share ground. The cell's split
is written into both index files (`split:`), so it is the same in both generators.
Black or transparent map borders are never used.

Re-running replaces only what this tool wrote (the marked block in each index.yaml);
your own photos and their index.yaml entries are kept.
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generator.settings import DEFAULT_CONFIG, load_settings  # noqa: E402
from generator.split import assign_groups  # noqa: E402

MADE_BY = "cut_orthomosaics"
BLOCK = 16          # validity is checked on 16 x 16 px blocks of the map


def read_map(path):
    """GeoTIFF -> (BGR uint8 image, bool mask of real ground)."""
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # GeoTIFF tags are "unknown" to OpenCV
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"OpenCV cannot read {path}")
    if img.dtype != np.uint8:
        raise ValueError(f"{path.name}: expected 8-bit RGB, got {img.dtype}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        valid = img[..., 3] > 0
        img = np.ascontiguousarray(img[..., :3])
    else:
        valid = img.max(axis=2) > 8          # pure black = no data
    return img, valid


def block_mask(valid):
    """True for every BLOCK x BLOCK block of real ground (map border excluded, 1 block margin).

    A block counts as ground when >= 95 % of its pixels are; lone 'bad' blocks are dark
    spots of real ground (deep shadow, dark stones), not map border, and are ignored.
    """
    h, w = valid.shape
    hb, wb = h // BLOCK, w // BLOCK
    frac = valid[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))
    bad = (frac < 0.95).astype(np.uint8)
    bad = cv2.morphologyEx(bad, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))    # drop lone specks
    bad = cv2.dilate(bad, np.ones((3, 3), np.uint8))                          # margin to the border
    return bad == 0


def window_ok(integral, x, y, w, h):
    """Is the source window (pixels) fully on good blocks? integral = of BAD blocks."""
    bx0, by0 = x // BLOCK, y // BLOCK
    bx1, by1 = -(-(x + w) // BLOCK), -(-(y + h) // BLOCK)
    if by1 >= integral.shape[0] or bx1 >= integral.shape[1]:
        return False
    bad = integral[by1, bx1] - integral[by0, bx1] - integral[by1, bx0] + integral[by0, bx0]
    return bad == 0


def find_window(integral, cell, w, h, rng, tries=300):
    cx, cy, cs = cell
    if w > cs or h > cs:
        return None
    for _ in range(tries):
        x = cx + int(rng.integers(0, cs - w + 1))
        y = cy + int(rng.integers(0, cs - h + 1))
        if window_ok(integral, x, y, w, h):
            return x, y
    return None


def cells_of(valid_blocks, cell_px, min_valid=0.6):
    """Non-overlapping square cells (x, y, side) that are mostly real ground.

    A regular grid of cells, tried at many offsets; the offset that gives the most cells
    (then the most ground) wins. Fits more cells into an irregular map than a grid
    that always starts at the corner.
    """
    hb, wb = valid_blocks.shape
    cb = -(-cell_px // BLOCK)
    if hb < cb or wb < cb:
        return []
    ii = cv2.integral(valid_blocks.astype(np.uint8)).astype(np.int64)
    step = max(1, cb // 16)
    best = (0, 0, [])
    for oy in range(0, min(cb, hb - cb + 1), step):
        for ox in range(0, min(cb, wb - cb + 1), step):
            cells, ground = [], 0
            for by in range(oy, hb - cb + 1, cb):
                for bx in range(ox, wb - cb + 1, cb):
                    n = ii[by + cb, bx + cb] - ii[by, bx + cb] - ii[by + cb, bx] + ii[by, bx]
                    if n >= min_valid * cb * cb:
                        cells.append((bx * BLOCK, by * BLOCK, cb * BLOCK))
                        ground += n
            if (len(cells), ground) > best[:2]:
                best = (len(cells), ground, cells)
    return best[2]


def biggest_square(good, cell):
    """Biggest square of clean blocks inside a cell -> (x, y, side) in map pixels."""
    cx, cy, cs = cell
    bx0, by0, cb = cx // BLOCK, cy // BLOCK, cs // BLOCK
    sub = good[by0:by0 + cb, bx0:bx0 + cb]
    dp = np.zeros((sub.shape[0] + 1, sub.shape[1] + 1), np.int32)   # side of the square ending here
    for i in range(sub.shape[0]):
        row, up = dp[i + 1], dp[i]
        for j in np.flatnonzero(sub[i]):
            row[j + 1] = 1 + min(up[j + 1], row[j], up[j])
    i, j = np.unravel_index(int(dp.argmax()), dp.shape)
    n = int(dp[i, j])
    return (bx0 + j - n) * BLOCK, (by0 + i - n) * BLOCK, n * BLOCK


def biggest_rect(mask):
    """Biggest all-True rectangle of a block mask -> (x, y, w, h) in blocks."""
    h, w = mask.shape
    heights = np.zeros(w, np.int64)
    best = (0, 0, 0, 0, 0)
    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack = []                                  # (start x, height), heights increasing
        for x in range(w + 1):
            cur = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                if sh * (x - sx) > best[0]:
                    best = (sh * (x - sx), sx, y - sh + 1, x - sx, sh)
                start = sx
            stack.append((start, cur))
    return best[1:]


BEGIN = f"# --- {MADE_BY}: begin (written by tools/cut_orthomosaics.py, do not edit by hand) ---"
END = f"# --- {MADE_BY}: end ---"


def read_index(path):
    """(text before our block, text after it, newline) - hand-written lines stay untouched."""
    if not path.exists():
        return "", "", "\n"
    raw = path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if BEGIN in text and END in text:
        before, rest = text.split(BEGIN, 1)
        old_block, after = rest.split(END, 1)
        for e in yaml.safe_load(old_block) or []:            # delete our earlier output
            (path.parent / e["file"]).unlink(missing_ok=True)
        return before.rstrip("\n") + "\n", after.lstrip("\n"), nl
    return text.rstrip("\n") + "\n" if text.strip() else "", "", nl


def write_index(path, before, after, nl, entries, header=""):
    if not before.strip():
        before = header.rstrip("\n") + "\n" if header else ""
    lines = [BEGIN]
    for e in entries:
        lines.append("- " + yaml.safe_dump(e, default_flow_style=True, sort_keys=False,
                                           allow_unicode=True, width=1000).strip())
    lines.append(END)
    text = before + ("\n" if before else "") + "\n".join(lines) + "\n" + (("\n" + after) if after.strip() else "")
    path.write_bytes(text.replace("\n", nl).encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--sources", default=str(ROOT / "assets" / "background_sources.yaml"))
    ap.add_argument("--maps-dir", default=str(ROOT / "assets" / "batch_backgrounds_sources"))
    ap.add_argument("--frames", type=int, default=200, help="frames in total, spread over all cells")
    ap.add_argument("--frames-dir", default=str(ROOT / "assets" / "batch_backgrounds"))
    ap.add_argument("--textures-per-map", type=int, default=0, help="cells per map that get a "
                    "texture (0 = all)")
    ap.add_argument("--textures-dir", default=str(ROOT / "assets" / "backgrounds"))
    ap.add_argument("--strip-altitude", type=float, default=75.0, help="ground outside the cells "
                    "becomes a texture if a clean rectangle there covers a frame from this altitude")
    ap.add_argument("--max-upsample", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    s = load_settings(a.config)
    W, H = s["camera"]["width"], s["camera"]["height"]
    t = math.tan(math.radians(s["camera"]["hfov_deg"]) / 2)
    alt_lo, alt_hi = s["altitude_m"]
    gsd_at = lambda alt: 2 * alt * t / W                      # noqa: E731
    cell_m = 1.03 * W * gsd_at(alt_hi)                        # biggest footprint (100 m frame) + 3 %
    sources = yaml.safe_load(Path(a.sources).read_text(encoding="utf-8"))
    maps_dir, frames_dir, tex_dir = Path(a.maps_dir), Path(a.frames_dir), Path(a.textures_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([a.seed, 77])

    f_parts = read_index(frames_dir / "index.yaml")
    t_parts = read_index(tex_dir / "index.yaml")
    f_index, t_index = [], []

    # pass 1: open every map, find its cells
    maps = []
    for src in sources:
        path = maps_dir / src["file"]
        if not path.exists():
            print(f"  missing {path.name} - download it from its url in {Path(a.sources).name}")
            continue
        img, valid = read_map(path)
        g = float(src["gsd_m"])
        good = block_mask(valid)
        bad_int = cv2.integral((~good).astype(np.uint8)).astype(np.int64)
        cells = cells_of(good, int(math.ceil(cell_m / g)))
        alt_min = max(alt_lo, g / a.max_upsample * W / (2 * t))   # never enlarge more than max_upsample
        print(f"  {src['name']:16s} {img.shape[1]}x{img.shape[0]} px, {g * 100:.1f} cm/px, "
              f"{img.shape[1] * g:.0f} x {img.shape[0] * g:.0f} m, {len(cells)} cells, "
              f"altitude {alt_min:.0f}-{alt_hi:.0f} m")
        if not cells or alt_min > alt_hi:
            print("    skipped: too small or too coarse")
            continue
        maps.append(dict(src=src, img=img, gsd=g, good=good, integral=bad_int, cells=cells, alt_min=alt_min))

    # pass 2: frames, spread evenly over all cells of all maps
    all_cells = [(m, k) for m in maps for k in range(len(m["cells"]))]
    order = rng.permutation(len(all_cells))
    per_cell = {}
    for i in range(a.frames):
        m, k = all_cells[order[i % len(all_cells)]]
        per_cell.setdefault((id(m), k), [m, k, 0])[2] += 1
    n_frames = 0
    for m, k, n in per_cell.values():
        name, g = m["src"]["name"], m["gsd"]
        for j in range(n):
            for _ in range(20):                                 # new altitude / turn if it does not fit
                alt = rng.uniform(m["alt_min"], alt_hi)
                fg = gsd_at(alt)
                rot = int(rng.integers(4))                      # map north is not special
                w, h = round(W * fg / g), round(H * fg / g)
                sw, sh = (h, w) if rot % 2 else (w, h)
                pos = find_window(m["integral"], m["cells"][k], sw, sh, rng)
                if pos is not None:
                    break
            if pos is None:
                print(f"    {name} cell {k}: no clean spot for frame {j}")
                continue
            x, y = pos
            crop = np.rot90(m["img"][y:y + sh, x:x + sw], rot)
            interp = cv2.INTER_AREA if fg >= g else cv2.INTER_CUBIC
            frame = cv2.resize(np.ascontiguousarray(crop), (W, H), interpolation=interp)
            fname = f"{name}_c{k:02d}_{j:02d}.jpg"
            cv2.imwrite(str(frames_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            f_index.append({"file": fname, "gsd_m": round(fg, 5), "altitude_m": round(alt, 1),
                            "group": f"{name}_c{k:02d}", "type": m["src"]["type"]})
            n_frames += 1

    # one split for every piece of map ground, written into BOTH index files, so a cell
    # is in the same split in generate.py and generate_batch.py (the batch's valid/test
    # ground is never in generate.py's train). Same draw as generate_batch.py used before.
    cell_groups = sorted(f"{m['src']['name']}_c{k:02d}" for m in maps for k in range(len(m["cells"])))
    g2s = {g: sp for sp, gs in assign_groups(cell_groups, s["split"], s["seed"]).items() for g in gs}
    for e in f_index:
        e["split"] = g2s[e["group"]]

    # pass 3: textures for generate.py: the biggest clean square in EVERY cell (a full
    # cell is ~176 m: covers a 100 m frame without patching), at the sharpest GSD the
    # generator needs (the 50 m frame), so it only ever shrinks them
    n_tex, tex_area = 0, 0.0
    for m in maps:
        name, g = m["src"]["name"], m["gsd"]
        tg = max(g, gsd_at(alt_lo))
        cells = range(len(m["cells"]))
        if a.textures_per_map:
            cells = sorted(rng.permutation(len(m["cells"]))[:a.textures_per_map])
        pieces = []
        for k in cells:
            x, y, side = biggest_square(m["good"], m["cells"][k])
            if side * g >= cell_m / 2:
                pieces.append((x, y, side, side, f"{name}_c{k:02d}", g2s[f"{name}_c{k:02d}"]))
        # pass 4: ground outside the cells: the biggest clean rectangles that still cover
        # a 75 m frame (long strips along the map edges). Their own group, always train.
        rest = m["good"].copy()
        for x, y, cs in m["cells"]:
            rest[y // BLOCK:-(-(y + cs) // BLOCK), x // BLOCK:-(-(x + cs) // BLOCK)] = False
        min_long = W * gsd_at(a.strip_altitude) / g / BLOCK
        min_short = H * gsd_at(a.strip_altitude) / g / BLOCK
        for j in range(20):
            bx, by, bw, bh = biggest_rect(rest)
            if max(bw, bh) < min_long or min(bw, bh) < min_short:
                break
            rest[by:by + bh, bx:bx + bw] = False
            pieces.append((bx * BLOCK, by * BLOCK, bw * BLOCK, bh * BLOCK, f"{name}_r{j:02d}", "train"))
        for i, (x, y, w, h, group, split) in enumerate(pieces):
            crop = m["img"][y:y + h, x:x + w]
            out = (round(w * g / tg), round(h * g / tg))
            tex = cv2.resize(crop, out, interpolation=cv2.INTER_AREA) if out != (w, h) else crop
            fname = f"{name}_t{i:02d}.jpg"
            cv2.imwrite(str(tex_dir / fname), tex, [cv2.IMWRITE_JPEG_QUALITY, 95])
            t_index.append({"file": fname, "gsd_m": round(tg, 5), "group": group,
                            "type": m["src"]["type"], "split": split})
            n_tex += 1
            tex_area += w * h * g * g
            print(f"    texture {fname}: {w * g:.0f} x {h * g:.0f} m, {group}, {split}")

    f_header = ("# Background frames for generate_batch.py (see generator/batch.py).\n"
                "# Add your own photos OUTSIDE the cut_orthomosaics block, as\n"
                "#   - {file: IMG_0012.jpg, gsd_m: 0.031}   or   - {file: IMG_0013.jpg, altitude_m: 80}\n"
                "# Credits / licences of the drone maps: assets/background_sources.yaml\n")
    write_index(frames_dir / "index.yaml", *f_parts, f_index, f_header)
    write_index(tex_dir / "index.yaml", *t_parts, t_index)
    print(f"Wrote {n_frames} frames to {frames_dir} and {n_tex} textures "
          f"({tex_area / 1e4:.1f} ha of ground) to {tex_dir}")
    print("Splits of the map cells: " + ", ".join(
        f"{sp}: {' '.join(gs)}" for sp, gs in assign_groups(cell_groups, s["split"], s["seed"]).items()))


if __name__ == "__main__":
    main()
