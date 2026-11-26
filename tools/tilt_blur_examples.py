"""Before / after pictures for the camera tilt + blur step (generator/view.py, camera.apply_blur).

    python tools/tilt_blur_examples.py            -> dataset/examples_tilt_blur/*.jpg

The SAME scene (same ground, same flags, same places) is shown twice:
  before  straight down, no extra blur  (the generators before this step)
  after   camera tilted + lens blur + motion blur, labels pushed through the same warp
Example 1 is a generate.py frame (ground built from textures, tilt 10 deg).
Example 2 is a generate_batch.py image (real drone photo, tilt 4 deg, zoomed to fill).
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.background import BackgroundLibrary            # noqa: E402
from generator.camera import motion_kernel                     # noqa: E402
from generator.compose import make_scene, object_corners, place_object   # noqa: E402
from generator.flag_render import render_object                # noqa: E402
from generator.flags import load_flag_textures                 # noqa: E402
from generator.match import for_image, measure                 # noqa: E402
from generator.scale import camera_from_settings               # noqa: E402
from generator.settings import ROOT, load_settings             # noqa: E402
from generator.view import _frame_corners, _map, _turn_matrix, zoom_view   # noqa: E402

OUT = ROOT / "dataset" / "examples_tilt_blur"
GREEN, BLACK, WHITE = (60, 220, 60), (0, 0, 0), (255, 255, 255)
FLAGS = ["egypt", "iraq", "syria", "yemen"]


def blur(img, gsd, sigma, speed, exposure, angle):
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    length = speed * exposure / gsd
    img = cv2.filter2D(img, -1, motion_kernel(length, angle), borderType=cv2.BORDER_REFLECT)
    return img, length


def to8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def put(img, text, org, scale=0.8):
    for col, th in ((BLACK, 5), (WHITE, 2)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def panel(before, after, flags_b, flags_a, names, title_b, title_a, header):
    """Two frames side by side, then before/after zooms of every flag."""
    H, W = before.shape[:2]
    k = 1200 / W
    row = []
    for img, flags, title in ((before, flags_b, title_b), (after, flags_a, title_a)):
        small = cv2.resize(img, (1200, round(H * k)), interpolation=cv2.INTER_AREA)
        for f in flags:
            cv2.polylines(small, [(f["corners"] * k).round().astype(np.int32)], True, GREEN, 2, cv2.LINE_AA)
        put(small, title, (12, 34))
        row.append(small)
    top = np.hstack([row[0], np.full((row[0].shape[0], 12, 3), 255, np.uint8), row[1]])
    zooms = []
    for fb, fa in zip(flags_b, flags_a):
        pair = []
        for img, f, tag in ((before, fb, "before"), (after, fa, "after")):
            c = f["corners"].mean(axis=0)
            r = 60
            x0, y0 = int(c[0] - r), int(c[1] - r)
            crop = img[max(0, y0):y0 + 2 * r, max(0, x0):x0 + 2 * r]
            crop = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST)
            p = ((f["corners"] - (max(0, x0), max(0, y0))) * (240 / (2 * r))).round().astype(np.int32)
            cv2.polylines(crop, [p], True, GREEN, 1, cv2.LINE_AA)
            long_px = 0.5 * (np.linalg.norm(f["corners"][1] - f["corners"][0])
                             + np.linalg.norm(f["corners"][2] - f["corners"][3]))
            put(crop, f"{tag}: {long_px:.0f} px", (6, 232), 0.5)
            pair.append(crop)
        z = np.hstack(pair)
        lab = np.full((30, z.shape[1], 3), 30, np.uint8)
        put(lab, names[fb["cls"]], (6, 22), 0.6)
        zooms.append(np.vstack([lab, z]))
    strip = np.full((zooms[0].shape[0] + 20, top.shape[1], 3), 30, np.uint8)
    for i, z in enumerate(zooms):
        x = 10 + i * (z.shape[1] + 20)
        if x + z.shape[1] <= strip.shape[1]:
            strip[10:10 + z.shape[0], x:x + z.shape[1]] = z
    head = np.full((50, top.shape[1], 3), 255, np.uint8)
    cv2.putText(head, header, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)
    return np.vstack([head, top, strip])


def place(frame, flags_tex, ids, spots, gsd, s, rng, scene, match=None):
    Lm, Sm = s["flag_size_m"]
    ss, fx, cc = s["supersample"], s["flag_effects"], s["compose"]
    out = []
    for c, (cx, cy, ang) in zip(ids, spots):
        obj = render_object(flags_tex[c], (Lm, Sm), gsd, ss, rng, dict(fx, tilt_jitter=0.0))
        dst = object_corners((cx, cy), ang, obj.long_px / ss, obj.short_px / ss, rng, 0.0)
        place_object(frame, obj, dst, ss, rng, cc, scene, gsd, match=match)
        out.append({"cls": c, "corners": dst})
    return out


def example_synthetic(s, flags_tex, ids):
    cam = camera_from_settings(s)
    W, H = cam.width, cam.height
    alt, tilt, az = 60.0, 10.0, 70.0
    gsd = cam.gsd(alt)
    rng = np.random.default_rng(7)
    # one straight-down canvas that holds both the straight-down frame and what the
    # tilted camera sees
    Mt = _turn_matrix(W, H, cam.f_px, tilt, az)
    quad = _map(_frame_corners(W, H), np.linalg.inv(Mt))
    pts = np.vstack([quad, _frame_corners(W, H)])
    x0, y0 = np.floor(pts.min(axis=0)) - 2
    x1, y1 = np.ceil(pts.max(axis=0)) + 2
    Wc, Hc = int(x1 - x0), int(y1 - y0)
    bgl = BackgroundLibrary(s["backgrounds"]["dir"], s["backgrounds"]["max_upsample"],
                            s["backgrounds"]["type_distribution"])
    groups = sorted({it["group"] for it in bgl.items if it["type"] == "desert_sand"}) or bgl.groups()
    canvas, _ = bgl.sample(rng, groups, gsd, Wc, Hc, s["backgrounds"]["jitter"])
    scene = make_scene(canvas, rng, s["compose"])
    # flags at the far side, centre and near side of the picture (straight-down frame coords)
    spots_frame = [(0.20 * W, 0.30 * H, 20), (0.50 * W, 0.50 * H, 75), (0.80 * W, 0.70 * H, 140),
                   (0.35 * W, 0.75 * H, 300)]
    spots = [(x - x0, y - y0, a) for x, y, a in spots_frame]
    flags = place(canvas, flags_tex, ids, spots, gsd, s, rng, scene)
    # before: the straight-down frame
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)          # frame -> canvas
    before = canvas[int(-y0):int(-y0) + H, int(-x0):int(-x0) + W]           # x0, y0 are whole numbers
    fb = [{**f, "corners": f["corners"] + (x0, y0)} for f in flags]
    # after: tilted camera, then lens + motion blur on the whole picture
    M = Mt @ np.linalg.inv(T)                                                 # canvas -> tilted frame
    after = cv2.warpPerspective(canvas, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    after, length = blur(after, gsd, sigma=0.9, speed=12, exposure=1 / 250, angle=0.4)
    fa = [{**f, "corners": _map(f["corners"], M)} for f in flags]
    return before, after, fb, fa, (
        f"Example 1 - generate.py (ground built from textures), altitude {alt:.0f} m, GSD {gsd * 100:.1f} cm/px",
        "BEFORE: straight down, no blur",
        f"AFTER: tilt {tilt:.0f} deg + lens blur 0.9 px + motion blur {length:.1f} px")


def example_real(s, flags_tex, ids):
    path = ROOT / "assets" / "batch_backgrounds" / "iraq_gaar_c00_05.jpg"
    gsd, alt, tilt, az = 0.04108, 92.3, 4.0, 200.0
    src = cv2.imread(str(path)).astype(np.float32) / 255
    H, W = src.shape[:2]
    f_px = (W / 2) / math.tan(math.radians(s["camera"]["hfov_deg"]) / 2)
    rng = np.random.default_rng(11)
    view = zoom_view(W, H, f_px, tilt, az)
    look = measure(src, s["batch"]["match_camera"])
    match = for_image(look, rng, s["batch"]["match_camera"])
    frame = src.copy()
    scene = make_scene(frame, rng, s["compose"])
    q = view["quad"]                                    # the part of the photo the tilted picture shows
    cx, cy = q.mean(axis=0)
    spots = []
    for (u, v, a) in ((0.25, 0.30, 35), (0.5, 0.5, 100), (0.75, 0.70, 160), (0.35, 0.72, 250)):
        x = q[0, 0] + (q[1, 0] - q[0, 0]) * u
        y = q[0, 1] + (q[3, 1] - q[0, 1]) * v
        spots.append((x, y, a))
    flags = place(frame, flags_tex, ids, spots, gsd, s, rng, scene, match=match)
    before = frame
    after = cv2.warpPerspective(frame, view["H"], (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    gsd_out = gsd / view["magnification"]
    after, length = blur(after, gsd_out, sigma=0.7, speed=12, exposure=1 / 250, angle=1.2)
    fa = [{**f, "corners": _map(f["corners"], view["H"])} for f in flags]
    return before, after, flags, fa, (
        f"Example 2 - generate_batch.py (real drone photo {path.name}), altitude {alt:.0f} m",
        "BEFORE: photo as it is, no extra blur",
        f"AFTER: tilt {tilt:.0f} deg (zoom {view['zoom']:.2f}x) + lens blur 0.7 px + motion blur {length:.1f} px")


def main():
    s = load_settings()
    names = s["classes"]
    L, S = s["flag_size_m"]
    flags_tex, _ = load_flag_textures(s["flags_dir"], names, s["flag_files"], L / S)
    ids = [names.index(n) for n in FLAGS]
    OUT.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate((example_synthetic, example_real), 1):
        before, after, fb, fa, (header, tb, ta) = ex(s, flags_tex, ids)
        img = panel(to8(before), to8(after), fb, fa, names, tb, ta, header)
        out = OUT / f"example{i}_before_after.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(out)


if __name__ == "__main__":
    main()
