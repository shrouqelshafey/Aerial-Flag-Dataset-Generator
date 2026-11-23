"""Give a finished image the colour and contrast of a REFERENCE dataset.

The reference drone photos (Roboflow Universe uavc/flags_gamed_awy and
uavc/uav2024-flags) are bright, pale, warm and very flat: in Lab, the lightness
spread is only 2-5 units, against 10-20 for most ground photos. Real flags in
them look pastel and soft. A model trained only on crisp, contrasty images has
never seen that.

Method, on the whole image (flags included, as a camera / haze would):
  1. white balance + exposure: per-channel GAINS so the mean colour becomes the
     reference mean (gains, not an added colour: a blue flag stays blue)
  2. contrast: in Lab, (value - mean) * k + mean with k = reference spread / spread
     (Reinhard), never below min_contrast (lightness) / min_colour (a, b), so a
     flag never fades away completely
Statistics are measured on a 500 px wide copy, the same way the reference numbers
in config.yaml were measured.
"""
import cv2
import numpy as np

from .noise import uniform


def lab_stats(img_bgr01):
    h, w = img_bgr01.shape[:2]
    small = cv2.resize(img_bgr01, (500, max(1, round(500 * h / w))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32), cv2.COLOR_BGR2Lab).reshape(-1, 3)
    return lab.mean(axis=0), lab.std(axis=0) + 1e-6


def sample_target(rng, presets):
    name = list(presets)[int(rng.integers(len(presets)))]
    p = presets[name]
    mean = np.array([uniform(rng, p["mean_L"]), uniform(rng, p["mean_a"]), uniform(rng, p["mean_b"])])
    std = np.array([uniform(rng, p["std_L"]), uniform(rng, p["std_a"]), uniform(rng, p["std_b"])])
    return name, mean.astype(np.float32), std.astype(np.float32)


def apply_look(img, rng, cfg):
    """img: float32 BGR 0..1. Returns (image, info dict or None if not applied)."""
    if rng.random() >= cfg.get("prob", 0.0):
        return img, None
    name, t_mean, t_std = sample_target(rng, cfg["presets"])
    s = uniform(rng, cfg["strength"])               # 0 = unchanged, 1 = full match
    img = np.clip(img, 0, 1).astype(np.float32)

    # 1. gains: mean colour -> reference mean colour
    t_bgr = cv2.cvtColor(t_mean.reshape(1, 1, 3), cv2.COLOR_Lab2BGR).reshape(3)
    h, w = img.shape[:2]
    small = cv2.resize(img, (500, max(1, round(500 * h / w))), interpolation=cv2.INTER_AREA)
    gain = 1 + s * (t_bgr / (small.reshape(-1, 3).mean(axis=0) + 1e-6) - 1)
    img = np.clip(img * gain.astype(np.float32), 0, 1)

    # 2. contrast and colourfulness around the (new) mean
    mean, std = lab_stats(img)
    k = t_std / std
    k[0] = np.clip(k[0], cfg["min_contrast"], 1.0)
    k[1:] = np.clip(k[1:], cfg["min_colour"], 1.0)
    k = 1 + s * (k - 1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab = (lab - mean) * k + mean
    out = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(out, 0, 1), {"look": name, "strength": round(s, 3),
                                "contrast_k": [round(float(v), 3) for v in k]}
