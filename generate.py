"""Build a synthetic top-down flag dataset from config/config.yaml.

    python generate.py --n 200
    python generate.py --n 1000 --name big_run --workers 6
    python generate.py --n 20 --workers 1          (one process: easier to debug)

One image = one simulated SIYI A8 mini frame (3840 x 2160) from 50-100 m. Per frame:
  1 scene (altitude -> GSD) -> 2 background at that GSD -> 3 flag rendered at true size
  -> 4 placement -> 5 blending -> 6 distractors -> 7 camera tilt + effects -> 8 labels
  -> 9 tiles -> 10 split + save
Output: dataset/<run>/  (see generator/writer.py for the folder layout)
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from generator.background import BackgroundLibrary
from generator.camera import apply_camera
from generator.compose import cast_shadows, make_scene, object_corners, place_object
from generator.distractors import load_distractor_images, make_distractor, sample_kind
from generator.flag_render import render_object
from generator.flags import load_flag_textures
from generator.labels import detect_line, obb_line, window_labels
from generator.noise import uniform
from generator.preview import draw_preview
from generator.scale import camera_from_settings
from generator.settings import DEFAULT_CONFIG, load_settings
from generator.split import SPLITS, assign_groups, pick_split
from generator.tiling import tile_origins
from generator.view import describe, sample_view, warp_frame, warp_points
from generator.writer import DatasetWriter

G = {}   # per-process state (each worker process loads its own copy)
LOG_FIELDS = ["scene_id", "image_name", "split", "country_class", "class_id", "camera_altitude",
              "projected_width_px", "projected_height_px", "background_type", "generation_status"]


# ---------------------------------------------------------------------------
def init_worker(s, run_dir, variants):
    cv2.setNumThreads(1)                      # we parallelise over images instead
    G["s"] = s
    G["cam"] = camera_from_settings(s)
    bg = s["backgrounds"]
    G["bg"] = BackgroundLibrary(bg["dir"], bg["max_upsample"], bg["type_distribution"])
    L, S = s["flag_size_m"]
    G["flags"], G["missing"] = load_flag_textures(s["flags_dir"], s["classes"], s["flag_files"], L / S)
    G["drawable"] = sorted(G["flags"])        # class ids that have a flag image
    G["class_p"] = class_weights(G["drawable"], s["focus"])
    G["distractor_images"] = load_distractor_images(s["distractors"]["dir"]) \
        if s["distractors"]["enabled"] else []
    G["split_groups"] = assign_groups(G["bg"].groups(), s["split"], s["seed"], G["bg"].fixed_splits())
    G["writer"] = DatasetWriter(run_dir, s["classes"], s["jpeg_quality"], variants)


def class_weights(drawable, focus):
    """P(class) per drawn flag: the focus classes share `share`, the rest share the remainder."""
    fset = [c for c in focus["classes"] if c in drawable]
    share = focus["share"] if fset and len(fset) < len(drawable) else 0.0
    if share <= 0:
        return np.full(len(drawable), 1.0 / len(drawable))
    rest = len(drawable) - len(fset)
    return np.array([share / len(fset) if c in fset else (1 - share) / rest for c in drawable])


def sample_count(rng, dist):
    ks = sorted(dist)
    p = np.array([dist[k] for k in ks], float)
    return ks[int(rng.choice(len(ks), p=p / p.sum()))]


def find_spot(rng, quad, long_px, short_px, taken, spacing_px, allow_partial):
    """Pick a centre inside `quad` (the part of the canvas the camera sees) that keeps
    every object `spacing_px` away from the others. Sometimes (allow_partial) the
    object is put across the picture edge instead."""
    r = 0.5 * float(np.hypot(long_px, short_px))
    poly = np.asarray(quad, np.float32).reshape(-1, 1, 2)
    (x0, y0), (x1, y1) = quad.min(axis=0), quad.max(axis=0)
    partial = rng.random() < allow_partial          # straddle the picture edge
    for _ in range(200):
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        d = cv2.pointPolygonTest(poly, (float(cx), float(cy)), True)   # > 0 inside, in px
        if (abs(d) > 0.6 * r) if partial else (d < r + 1):
            continue
        if all(np.hypot(cx - tx, cy - ty) > r + tr + spacing_px for tx, ty, tr in taken):
            return cx, cy, r
    return None


def big_enough(labels, min_w, min_h):
    """annotations.minimum_bbox_width_px / height_px, checked on the detect box."""
    keep, small = [], 0
    for lab in labels:
        x0, y0, x1, y1 = lab["aabb"]
        if x1 - x0 >= min_w and y1 - y0 >= min_h:
            keep.append(lab)
        else:
            small += 1
    return keep, small


def window(flags, x0, y0, x1, y1, s):
    kept, cut = window_labels(flags, x0, y0, x1, y1, s["min_visible_fraction"])
    kept, small = big_enough(kept, *s["min_box_px"])
    return kept, cut, small


# ---------------------------------------------------------------------------
def generate_one(idx):
    s, cam = G["s"], G["cam"]
    names = s["classes"]
    rng = np.random.default_rng([s["seed"], idx])          # one RNG per image = reproducible
    W, H, ss = cam.width, cam.height, s["supersample"]
    fx, cc = s["flag_effects"], s["compose"]

    # 1. scene: altitude -> GSD
    split = pick_split(idx, s["split"])
    altitude = uniform(rng, s["altitude_m"])
    gsd = cam.gsd(altitude)

    # camera direction: straight down, or tilted a little (generator.view). Tilted, the
    # ground the camera sees is bigger than one frame: it is built straight down on a
    # larger canvas and turned into the tilted picture in step 7.
    view = sample_view(rng, s["view"], W, H, cam.f_px)
    Wc, Hc = view["size"]
    quad = view["quad"]

    # 2. background at the frame's GSD
    frame, bg_info = G["bg"].sample(rng, G["split_groups"][split], gsd, Wc, Hc, s["backgrounds"]["jitter"])
    scene = make_scene(frame, rng, cc)

    # 3-5. flags: render at true size -> place -> blend
    Lm, Sm = s["flag_size_m"]
    drawable = G["drawable"]
    n_flags = min(sample_count(rng, s["flags_per_image"]), len(drawable))
    # no country twice in a frame; look-alike flags (generator.focus_classes) more often
    cls_ids = rng.choice(drawable, size=n_flags, replace=False, p=G["class_p"]) if n_flags else []
    spacing = s["min_spacing_m"] / gsd
    taken, flags, no_room = [], [], []
    for c in cls_ids:
        obj = render_object(G["flags"][int(c)], (Lm, Sm), gsd, ss, rng, fx)
        Lp, Sp = obj.long_px / ss, obj.short_px / ss
        spot = find_spot(rng, quad, Lp, Sp, taken, spacing, s["allow_partial_prob"])
        if spot is None:
            no_room.append(int(c))
            continue
        cx, cy, r = spot
        angle = rng.uniform(0, 360)
        dst = object_corners((cx, cy), angle, Lp, Sp, rng, fx["tilt_jitter"])
        info = place_object(frame, obj, dst, ss, rng, cc, scene, gsd)
        taken.append((cx, cy, r))
        flags.append({"cls": int(c), "name": names[int(c)], "corners": dst,
                      "angle_deg": round(angle, 2), "effects": {**obj.effects, **info}})

    # 6. distractors (same renderer + compositor, never labelled)
    distractors = []
    if s["distractors"]["enabled"]:
        rigid_fx = dict(fx, fold_prob=0.0)
        lo, hi = s["distractors"]["count"]
        for _ in range(int(rng.integers(lo, hi + 1))):
            kind = sample_kind(rng, s["distractors"]["kinds"])
            tex, size_m, rigid, dname = make_distractor(kind, rng, G["distractor_images"])
            obj = render_object(tex, size_m, gsd, ss, rng, rigid_fx if rigid else fx)
            Lp, Sp = obj.long_px / ss, obj.short_px / ss
            spot = find_spot(rng, quad, Lp, Sp, taken, spacing, s["allow_partial_prob"])
            if spot is None:
                continue
            cx, cy, r = spot
            dst = object_corners((cx, cy), rng.uniform(0, 360), Lp, Sp, rng, fx["tilt_jitter"])
            place_object(frame, obj, dst, ss, rng, cc, scene, gsd)
            taken.append((cx, cy, r))
            distractors.append({"kind": dname, "size_m": [round(v, 2) for v in size_m],
                                "center_px": [round(cx, 1), round(cy, 1)]})

    shadows = cast_shadows(frame, gsd, rng, cc)

    # 7. camera: the tilt (pixels and flag corners through the same homography), then
    #    the camera effects on the final picture
    frame = warp_frame(frame, view, W, H)
    for f in flags:
        f["corners"] = warp_points(f["corners"], view)
    for d in distractors:
        d["center_px"] = np.round(warp_points([d["center_px"]], view)[0], 1).tolist()
    img, cam_params = apply_camera(frame, gsd, rng, s["camera_sim"])

    # 8. labels for the full frame
    stem = f"{s['image_prefix']}_{idx:0{s['zero_padding']}d}"
    kept, cut, small = window(flags, 0, 0, W, H, s)
    G["writer"].save("frames", split, stem, img,
                     [obb_line(l, W, H) for l in kept], [detect_line(l, W, H) for l in kept])

    # 9. tiles
    n_tiles = 0
    t = s["tiles"]
    if t["enabled"]:
        T = t["size"]
        for x, y in tile_origins(W, H, T, t["overlap"]):
            kt, _, _ = window(flags, x, y, x + T, y + T, s)
            if not kt and rng.random() >= t["keep_empty_prob"]:
                continue
            G["writer"].save("tiles", split, f"{stem}_x{x}_y{y}", img[y:y + T, x:x + T],
                             [obb_line(l, T, T) for l in kt], [detect_line(l, T, T) for l in kt])
            n_tiles += 1

    # 10. metadata + generation log
    meta = {"index": idx, "image": stem, "split": split, "seed": s["seed"],
            "camera": {"model": cam.name, "width": W, "height": H,
                       "hfov_deg": cam.hfov_deg, "f_px": round(cam.f_px, 2)},
            "altitude_m": round(altitude, 2), "gsd_m": round(gsd, 5),
            "flag_long_px": round(Lm / gsd, 2), "flag_short_px": round(Sm / gsd, 2),
            "background": bg_info, "view": describe(view),
            "scene": {"tint": [round(float(v), 3) for v in scene.tint], "light": round(scene.light, 3)},
            "flags": [{**{k: v for k, v in f.items() if k != "corners"},
                       "corners_px": np.round(f["corners"], 2).tolist()} for f in flags],
            "labelled_in_frame": len(kept), "cut_flags_not_labelled": cut,
            "too_small_not_labelled": small, "no_room_for": [names[c] for c in no_room],
            "distractors": distractors, "cast_shadows": shadows, "camera_effects": cam_params,
            "tiles_written": n_tiles}
    G["writer"].save_meta(stem, meta)

    rows = []
    base = {"scene_id": idx, "image_name": f"{stem}.jpg", "split": split,
            "camera_altitude": round(altitude, 2), "projected_width_px": round(Lm / gsd, 1),
            "projected_height_px": round(Sm / gsd, 1), "background_type": bg_info["type"]}
    kept_ids = [l["cls"] for l in kept]       # each country appears at most once per frame
    for f in flags:
        status = "labelled" if f["cls"] in kept_ids else "not_labelled_cut_or_too_small"
        rows.append({**base, "country_class": f["name"], "class_id": f["cls"], "generation_status": status})
    for c in no_room:
        rows.append({**base, "country_class": names[c], "class_id": c, "generation_status": "skipped_no_room"})
    if not flags:
        rows.append({**base, "country_class": "", "class_id": "", "generation_status": "negative_image"})
    return {"split": split, "labels": len(kept), "cut": cut, "small": small, "tiles": n_tiles,
            "upsampled": bg_info["upsampled"], "rows": rows, "stem": stem,
            "classes": kept_ids}


# ---------------------------------------------------------------------------
def progress(it, total, every=0.05):
    t0, step, out = time.time(), max(1, int(total * every)), []
    for i, r in enumerate(it, 1):
        out.append(r)
        if i % step == 0 or i == total:
            el = time.time() - t0
            print(f"  {i:5d}/{total}  {el:6.0f} s  (~{el / i * (total - i):.0f} s left)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--n", type=int, default=200, help="number of full frames")
    ap.add_argument("--start", type=int, default=0, help="first image index (to extend a run)")
    ap.add_argument("--name", default=None, help="run folder name inside dataset/")
    ap.add_argument("--seed", type=int, default=None, help="overrides project.random_seed")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--no-tiles", action="store_true")
    args = ap.parse_args()

    s = load_settings(args.config, args.seed)
    if args.no_tiles:
        s["tiles"]["enabled"] = False
    name = args.name or f"synthetic_n{args.n}_s{s['seed']}"
    run_dir = s["output_dir"] / name
    variants = ("frames", "tiles") if s["tiles"]["enabled"] else ("frames",)

    init_worker(s, run_dir, variants)    # also checks every asset in the main process
    cam = G["cam"]
    a0, a1 = s["altitude_m"]
    L = s["flag_size_m"][0]
    print(f"Camera {cam.name}: {cam.width}x{cam.height}, HFOV {cam.hfov_deg} deg, f = {cam.f_px:.0f} px")
    print(f"Altitude {a0:.0f}-{a1:.0f} m -> GSD {cam.gsd(a0)*100:.1f}-{cam.gsd(a1)*100:.1f} cm/px, "
          f"{L:.0f} m flag = {cam.size_px(L, a0):.0f}-{cam.size_px(L, a1):.0f} px")
    have = [s["classes"][c] for c in G["drawable"]]
    print(f"Flags: {len(have)} of {len(s['classes'])} classes have an image: {', '.join(have)}")
    fc = [s["classes"][c] for c in s["focus"]["classes"] if c in G["drawable"]]
    if fc and s["focus"]["share"] > 0:
        print(f"Focus: {', '.join(fc)} get {s['focus']['share']:.0%} of all flags "
              f"({s['focus']['share'] / len(fc):.1%} each, the others "
              f"{(1 - s['focus']['share']) / (len(have) - len(fc)):.1%} each)")
    if G["missing"]:
        print(f"  no image yet (not drawn, class ids kept): {', '.join(G['missing'])}")
    bgl = G["bg"]
    print(f"Backgrounds: {len(bgl.items)} used"
          + (f", skipped (use: false): {bgl.skipped}" if bgl.skipped else "")
          + (f", NOT in index.yaml (ignored): {bgl.unlisted}" if bgl.unlisted else ""))
    print(f"Split by background group: {G['split_groups']}")
    on = [k for k, v in s["layers"].items() if v]
    print(f"Layers on: {', '.join(on) or 'none'};  folds {'on' if s['flag_effects']['fold_prob'] else 'off'}, "
          f"flag shadows {'on' if s['compose']['contact_shadow_prob'] else 'off'}")
    G["writer"].write_yaml()
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        f.write(f"# Settings used for this run, read from {s['config_path']}\n")
        yaml.safe_dump(json.loads(json.dumps(s, default=str)), f, sort_keys=False)

    t0 = time.time()
    idxs = range(args.start, args.start + args.n)
    print(f"Generating {args.n} frames with {args.workers} worker(s) -> {run_dir}")
    if args.workers <= 1:
        results = progress((generate_one(i) for i in idxs), args.n)
    else:
        with Pool(args.workers, initializer=init_worker, initargs=(s, run_dir, variants)) as pool:
            results = progress(pool.imap(generate_one, idxs, chunksize=2), args.n)

    log = run_dir / "generation_logs" / "generation_report.csv"
    with open(log, "a" if args.start else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not args.start:
            w.writeheader()
        for r in results:
            w.writerows(r["rows"])

    n_prev = min(s["previews"], len(results))
    for r in results[:n_prev]:
        img = draw_preview(run_dir, r["split"], r["stem"], s["classes"])
        cv2.imwrite(str(run_dir / "previews" / f"{r['stem']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    per_class = {}
    for r in results:
        for c in r["classes"]:
            per_class[s["classes"][c]] = per_class.get(s["classes"][c], 0) + 1
    summary = {"run": name, "frames": len(results), "seconds": round(time.time() - t0, 1),
               "per_split": {sp: sum(r["split"] == sp for r in results) for sp in SPLITS},
               "flags_labelled": sum(r["labels"] for r in results),
               "flags_per_class": dict(sorted(per_class.items())),
               "negative_frames": sum(r["labels"] == 0 for r in results),
               "cut_flags_not_labelled": sum(r["cut"] for r in results),
               "too_small_not_labelled": sum(r["small"] for r in results),
               "tiles": sum(r["tiles"] for r in results),
               "frames_with_upsampled_background": sum(r["upsampled"] for r in results),
               "classes_without_image": G["missing"]}
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"Done -> {run_dir}\nCheck it:  python tools/qa/validate_labels.py {run_dir}")


if __name__ == "__main__":
    main()
