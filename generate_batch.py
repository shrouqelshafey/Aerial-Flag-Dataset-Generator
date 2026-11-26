"""Paste flags on REAL background photos: a small, exactly balanced test set for Roboflow.

    python generate_batch.py                                   uses generator.batch in config.yaml
    python generate_batch.py --backgrounds assets/batch_backgrounds_demo --zip

With 200 photos (generator.batch defaults):
    50 photos stay empty (1 image each)  +  150 photos x 3 images with 1, 2 or 3 flags
    = 500 images. Egypt is in exactly 60 % of the flag images, no country twice in an
    image, the other countries are balanced, and all images of one photo are in the
    same split (train / valid / test), so valid/test show ground the model never saw.

Per image: the photo (random flip / 180 deg turn) -> flags rendered at their true size
for that photo (2 m / GSD) -> placed fully inside the picture at a random angle -> given
the photo's own blur, noise and haze (generator/match.py) -> camera tilt (generator/view.py)
and extra blur on the whole picture -> for some images, the colour and contrast of a
reference dataset (generator/look.py) -> labels from the same corners.

Output dataset/<name>/:
    roboflow_detect/{train,valid,test}/{images,labels}  + data.yaml   (YOLO boxes)
    roboflow_obb/   {train,valid,test}/{images,labels}  + data.yaml   (YOLO oriented boxes)
    previews/  generation_report.csv  summary.json  config_used.yaml
Upload roboflow_detect (folder or --zip) to Roboflow; keep its train/valid/test split.
"""
import argparse
import csv
import json
import math
import os
import shutil
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from generator.batch import load_backgrounds, plan
from generator.camera import apply_blur
from generator.compose import make_scene, object_corners, place_object
from generator.flag_render import render_object
from generator.flags import load_flag_textures
from generator.labels import detect_line, obb_line, window_labels
from generator.look import apply_look
from generator.match import for_image, measure
from generator.settings import DEFAULT_CONFIG, ROOT, load_settings
from generator.split import SPLITS
from generator.view import is_tilted, sample_view, warp_frame, warp_points

FOLDER = {"train": "train", "val": "valid", "test": "test"}     # Roboflow's names
FORMATS = ("detect", "obb")
GREEN, BLACK, WHITE = (60, 220, 60), (0, 0, 0), (255, 255, 255)
G = {}


def init_worker(s, run_dir):
    cv2.setNumThreads(1)
    G["s"], G["run"] = s, Path(run_dir)
    L, S = s["flag_size_m"]
    G["flags"], G["missing"] = load_flag_textures(s["flags_dir"], s["classes"], s["flag_files"], L / S)


# ---------------------------------------------------------------------------
def find_spot(rng, quad, long_px, short_px, margin_px, taken, spacing_px):
    """A centre that keeps the whole flag inside `quad` (the part of the photo that ends
    up in the picture) with `margin_px` to spare, and away from the other flags."""
    r = 0.5 * float(np.hypot(long_px, short_px))
    poly = np.asarray(quad, np.float32).reshape(-1, 1, 2)
    (x0, y0), (x1, y1) = quad.min(axis=0) + r + margin_px, quad.max(axis=0) - r - margin_px
    if x1 <= x0 or y1 <= y0:
        return None
    for _ in range(200):
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), True) < r + margin_px:
            continue
        if all(np.hypot(cx - tx, cy - ty) > r + tr + spacing_px for tx, ty, tr in taken):
            return cx, cy, r
    return None


def flip(img, k):
    """k: 0 none, 1 left-right, 2 up-down, 3 both (= 180 deg turn). Size is kept."""
    if k & 1:
        img = img[:, ::-1]
    if k & 2:
        img = img[::-1]
    return np.ascontiguousarray(img)


def save(run, fmt, split, stem, jpg_bytes_or_path, lines):
    base = run / f"roboflow_{fmt}" / FOLDER[split]
    img_path = base / "images" / f"{stem}.jpg"
    if isinstance(jpg_bytes_or_path, Path):          # 2nd format: same file, no extra space
        if img_path.exists():
            img_path.unlink()
        try:
            os.link(jpg_bytes_or_path, img_path)
        except OSError:
            shutil.copyfile(jpg_bytes_or_path, img_path)
    else:
        img_path.write_bytes(jpg_bytes_or_path)
    (base / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                 encoding="utf-8")
    return img_path


def preview(img, kept, names, title, width=1800):
    """Downscaled copy with the oriented boxes drawn on, plus a zoom on every flag."""
    h, w = img.shape[:2]
    k = min(1.0, width / w)
    out = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    zooms = []
    for lab in kept:
        pts = lab["obb"] * k
        cv2.polylines(out, [pts.round().astype(np.int32)], True, GREEN, 2, cv2.LINE_AA)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        for col, th in ((BLACK, 4), (GREEN, 2)):
            cv2.putText(out, names[lab["cls"]], (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, col, th, cv2.LINE_AA)
        x0, y0, x1, y1 = (int(v) for v in lab["aabb"])
        c, r = ((x0 + x1) // 2, (y0 + y1) // 2), max(x1 - x0, y1 - y0)
        crop = img[max(0, c[1] - r):c[1] + r, max(0, c[0] - r):c[0] + r]
        if crop.size:
            zooms.append(cv2.resize(crop, (220, 220), interpolation=cv2.INTER_NEAREST))
    for col, th in ((BLACK, 5), (WHITE, 2)):
        cv2.putText(out, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, th, cv2.LINE_AA)
    if zooms:                       # strip of 1:1 zooms under the picture
        strip = np.full((240, out.shape[1], 3), 30, np.uint8)
        for i, z in enumerate(zooms[: out.shape[1] // 230]):
            strip[10:230, 10 + i * 230:230 + i * 230] = z
        out = np.vstack([out, strip])
    return out


# ---------------------------------------------------------------------------
def do_background(job):
    """All images of one background photo (1 if it stays empty, else images_per_background)."""
    s, run = G["s"], G["run"]
    b, names = s["batch"], s["classes"]
    bg, images, preview_stems = job["bg"], job["images"], job["preview"]
    src = cv2.imread(str(bg["path"]), cv2.IMREAD_COLOR)
    if src is None:
        raise IOError(f"Cannot read {bg['path']}")
    H, W = src.shape[:2]
    hfov = s["camera"]["hfov_deg"]
    gsd = bg["gsd_m"] or 2 * bg["altitude_m"] * math.tan(math.radians(hfov) / 2) / W
    photo = src.astype(np.float32) / 255.0
    photo_look = measure(photo, b["match_camera"])
    Lm, Sm = s["flag_size_m"]
    ss, fx, cc = s["supersample"], s["flag_effects"], s["compose"]
    margin = b["edge_margin_m"] / gsd
    spacing = s["min_spacing_m"] / gsd
    f_px = (W / 2) / math.tan(math.radians(hfov) / 2)
    # a tilted photo is zoomed in (= seen from lower): keep it at >= the lowest altitude
    max_mag = max(1.0, gsd * f_px / s["altitude_m"][0])
    blur_cfg = {"defocus_prob": 0.0, "defocus_sigma_px": [0.0, 0.0], "motion_prob": 0.0,
                "speed_mps": s["camera_sim"]["speed_mps"], "exposure_s": s["camera_sim"]["exposure_s"],
                **(b.get("blur") or {})}
    rows = []
    for im in images:
        rng = np.random.default_rng([s["seed"], 3, bg["index"], im["variant"]])
        k = int(rng.integers(4)) if b["random_flip"] else 0
        view = sample_view(rng, b.get("view"), W, H, f_px, mode="zoom", max_magnification=max_mag)
        frame = flip(photo, k).copy()
        scene = make_scene(frame, rng, cc)
        match = for_image(photo_look, rng, b["match_camera"])
        taken, flags, no_room = [], [], []
        for c in im["classes"]:
            obj = render_object(G["flags"][c], (Lm, Sm), gsd, ss, rng, fx)
            Lp, Sp = obj.long_px / ss, obj.short_px / ss
            for _ in range(20):                      # the flag must end up fully in the picture
                spot = find_spot(rng, view["quad"], Lp, Sp, margin, taken, spacing)
                if spot is None:
                    break
                cx, cy, r = spot
                angle = rng.uniform(0, 360)
                dst = object_corners((cx, cy), angle, Lp, Sp, rng, fx["tilt_jitter"])
                out_pts = warp_points(dst, view)
                if (out_pts.min(axis=0) >= margin / 2).all() and (out_pts + margin / 2 <= (W, H)).all():
                    break
                spot = None
            if spot is None:
                no_room.append(names[c])
                continue
            info = place_object(frame, obj, dst, ss, rng, cc, scene, gsd, match=match)
            taken.append((cx, cy, r))
            flags.append({"cls": c, "corners": dst, "angle": angle, "effects": info})
        # camera: tilt (pixels and corners through the same matrix), then blur on the
        # whole picture so the flags are exactly as soft as the ground
        frame = warp_frame(frame, view, W, H)
        for f in flags:
            f["corners"] = warp_points(f["corners"], view)
        gsd_out = gsd / view["magnification"]
        frame, blur = apply_blur(frame, gsd_out, rng, blur_cfg)
        kept, _ = window_labels(flags, 0, 0, W, H, s["min_visible_fraction"])
        look = None
        if b.get("look"):                            # some images get a reference dataset's look
            frame, look = apply_look(frame, rng, b["look"])
        out = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, b["jpeg_quality"]])
        if not ok:
            raise IOError("JPEG encoding failed")
        stem = f"bg{bg['index']:04d}_" + ("empty" if not im["classes"] else str(im["variant"]))
        first = save(run, "detect", bg["split"], stem, buf.tobytes(), [detect_line(l, W, H) for l in kept])
        save(run, "obb", bg["split"], stem, first, [obb_line(l, W, H) for l in kept])
        if stem in preview_stems:
            tilt = f"tilt {view['tilt_deg']:.1f} deg" if is_tilted(view) else "straight down"
            blurred = " ".join(f"{k} {v}" for k, v in blur.items()) or "no extra blur"
            title = (f"{stem} | {bg['split']} | GSD {gsd_out * 100:.2f} cm/px | 2 m flag = {Lm / gsd_out:.0f} px"
                     f" | {tilt} | {blurred} | look: {look['look'] if look else 'photo as is'}"
                     f" | {bg['path'].name}")
            cv2.imwrite(str(run / "previews" / f"{stem}.jpg"),
                        preview(out, kept, names, title), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append({
            "image": f"{stem}.jpg", "split": FOLDER[bg["split"]], "background": bg["path"].name,
            "group": bg["group"], "width": W, "height": H, "gsd_cm": round(gsd_out * 100, 3),
            "scale_from": bg["scale_from"], "flag_long_px": round(Lm / gsd_out, 1), "flip": k,
            "tilt_deg": view["tilt_deg"], "tilt_zoom": view.get("zoom", 1.0),
            "defocus_sigma_px": blur.get("defocus_sigma_px", ""), "motion_px": blur.get("motion_px", ""),
            "n_flags": len(kept), "countries": " ".join(names[l["cls"]] for l in kept),
            "no_room": " ".join(no_room), "blur_sigma_px": round(match["blur_sigma"], 3),
            "noise_sigma": round(photo_look["noise"], 4),
            "black_level": " ".join(f"{v:.3f}" for v in photo_look["black"])
            if photo_look["black"] is not None else "",
            "look": look["look"] if look else "",
            "look_strength": look["strength"] if look else "",
        })
    return rows


# ---------------------------------------------------------------------------
def write_data_yaml(run, names):
    for fmt in FORMATS:
        root = run / f"roboflow_{fmt}"
        lines = [f"# YOLO dataset ({'oriented' if fmt == 'obb' else 'axis-aligned'} boxes), "
                 "Roboflow folder layout",
                 "train: train/images", "val: valid/images", "test: test/images",
                 f"nc: {len(names)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
        (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check(run, rows, backgrounds, names, focus):
    """Re-read the written label files and check the recipe was followed exactly."""
    problems = []
    per_class, per_n, split_of = Counter(), Counter(), {}
    for r in rows:
        for fmt, need in (("detect", 5), ("obb", 9)):
            base = run / f"roboflow_{fmt}" / r["split"]
            stem = r["image"][:-4]
            if not (base / "images" / r["image"]).exists():
                problems.append(f"{fmt}: image missing {r['image']}")
            lines = [l.split() for l in (base / "labels" / f"{stem}.txt").read_text().splitlines()]
            cls = [int(l[0]) for l in lines]
            if any(len(l) != need for l in lines):
                problems.append(f"{fmt}/{stem}: wrong number of values")
            if any(not 0 <= float(v) <= 1 for l in lines for v in l[1:]):
                problems.append(f"{fmt}/{stem}: coordinate outside 0..1")
            if len(cls) != len(set(cls)):
                problems.append(f"{fmt}/{stem}: same country twice")
            if len(cls) != r["n_flags"]:
                problems.append(f"{fmt}/{stem}: {len(cls)} labels, expected {r['n_flags']}")
        per_n[r["n_flags"]] += 1
        for c in r["countries"].split():
            per_class[c] += 1
        if split_of.setdefault(r["group"], r["split"]) != r["split"]:
            problems.append(f"background group {r['group']} is in two splits")
        if r["no_room"]:
            problems.append(f"{r['image']}: no room for {r['no_room']}")
    with_flags = sum(v for k, v in per_n.items() if k > 0)
    focus_imgs = sum(focus in r["countries"].split() for r in rows) if focus else 0
    return problems, {
        "images": len(rows),
        "backgrounds": len(backgrounds),
        "empty_backgrounds": sum(bg["negative"] for bg in backgrounds),
        "images_per_split": dict(Counter(r["split"] for r in rows)),
        "images_by_flag_count": {str(k): per_n[k] for k in sorted(per_n)},
        f"images_with_{focus}": focus_imgs,
        f"{focus}_share_of_flag_images": round(focus_imgs / max(with_flags, 1), 3),
        "flags_per_country": dict(per_class.most_common()),
        "flag_long_side_px": [min(r["flag_long_px"] for r in rows), max(r["flag_long_px"] for r in rows)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--backgrounds", default=None, help="folder of background photos "
                    "(default: generator.batch.backgrounds_dir)")
    ap.add_argument("--name", default=None, help="run folder inside dataset/")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--zip", action="store_true", help="also write roboflow_detect.zip / roboflow_obb.zip")
    a = ap.parse_args()

    s = load_settings(a.config, a.seed)
    b = s["batch"]
    bg_dir = Path(a.backgrounds) if a.backgrounds else ROOT / b["backgrounds_dir"]
    names = s["classes"]
    backgrounds = load_backgrounds(bg_dir, s["altitude_m"], s["seed"])
    name = a.name or f"batch_{bg_dir.name}_s{s['seed']}"
    run = s["output_dir"] / name
    if run.exists():
        shutil.rmtree(run)                     # a batch is always rebuilt from scratch
    for fmt in FORMATS:
        for sp in SPLITS:
            for kind in ("images", "labels"):
                (run / f"roboflow_{fmt}" / FOLDER[sp] / kind).mkdir(parents=True, exist_ok=True)
    (run / "previews").mkdir(parents=True)

    init_worker(s, run)
    drawable = sorted(G["flags"])
    focus = b.get("focus_class")
    focus_id = names.index(focus) if focus else None
    if focus_id is not None and focus_id not in drawable:
        raise FileNotFoundError(f"focus_class {focus} has no flag image in {s['flags_dir']}")
    images = plan(backgrounds, b, s["split"], drawable, focus_id, s["seed"])

    print(f"Backgrounds: {len(backgrounds)} in {bg_dir}  ->  "
          f"{sum(bg.negative for bg in backgrounds)} stay empty, "
          f"{sum(not bg.negative for bg in backgrounds)} x {b['images_per_background']} with flags"
          f"  =  {len(images)} images")
    print(f"Flags drawn from: {', '.join(names[c] for c in drawable)}"
          + (f"  (focus: {focus} in {b['focus_share']:.0%} of the flag images)" if focus else ""))
    scales = Counter(bg.info["scale_from"] for bg in backgrounds)
    print("Scale: " + ", ".join(f"{n} from {k}" for k, n in scales.items()))

    # previews: the first images with flags + 2 empty ones
    pos = [im for im in images if im["classes"]][: max(0, b["preview_images"] - 2)]
    neg = [im for im in images if not im["classes"]][:2]
    stem = lambda im: f"bg{im['bg']:04d}_" + ("empty" if not im["classes"] else str(im["variant"]))
    previews = {stem(im) for im in pos + neg}
    jobs = []
    for i, bg in enumerate(backgrounds):
        jobs.append({"bg": {"index": i, "path": bg.path, "group": bg.group, "split": bg.split,
                            "gsd_m": bg.gsd_m, "altitude_m": bg.altitude_m,
                            "scale_from": bg.info["scale_from"], "negative": bg.negative},
                     "images": [im for im in images if im["bg"] == i], "preview": previews})

    with open(run / "config_used.yaml", "w", encoding="utf-8") as f:
        f.write(f"# Settings used for this run, read from {s['config_path']}\n")
        yaml.safe_dump(json.loads(json.dumps({**s, "backgrounds_used": str(bg_dir)}, default=str)),
                       f, sort_keys=False)
    write_data_yaml(run, names)

    t0, rows = time.time(), []
    print(f"Generating with {a.workers} worker(s) -> {run}")
    step = max(1, len(jobs) // 10)
    if a.workers <= 1:
        it = map(do_background, jobs)
        pool = None
    else:
        pool = Pool(a.workers, initializer=init_worker, initargs=(s, run))
        it = pool.imap(do_background, jobs)
    for i, r in enumerate(it, 1):
        rows += r
        if i % step == 0 or i == len(jobs):
            print(f"  {i:4d}/{len(jobs)} backgrounds  {time.time() - t0:5.0f} s", flush=True)
    if pool:
        pool.close()
        pool.join()

    rows.sort(key=lambda r: r["image"])
    with open(run / "generation_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    problems, summary = check(run, rows, [j["bg"] for j in jobs], names, focus)
    summary = {"run": name, "seconds": round(time.time() - t0, 1), **summary,
               "problems": problems[:50]}
    with open(run / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    if a.zip:
        for fmt in FORMATS:
            z = shutil.make_archive(str(run / f"roboflow_{fmt}"), "zip", run / f"roboflow_{fmt}")
            print(f"Zipped {z}")
    print(("All checks passed." if not problems else f"{len(problems)} PROBLEM(S) - see summary.json")
          + f"\nDone -> {run}\nUpload {run / 'roboflow_detect'} to Roboflow (keep the existing split).")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
