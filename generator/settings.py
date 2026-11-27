"""Read config/config.yaml and turn it into the flat settings the generator uses.

config/config.yaml stays the single source of truth:
  * the project spec (camera, flight, flag, countries, dataset, annotations, ...)
    gives the physical facts - camera, altitude, flag size, classes, split, labels;
  * the `generator:` section at the end gives what the spec leaves open - how many
    flags per frame, effect strengths, and an on/off switch for every realism layer.

If a value exists in the spec, it is read from the spec, never duplicated.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def _on(d, *keys, default=False):
    """Nested lookup of an `enabled`-style boolean: _on(cfg, "flag", "shadows", "enabled")."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return bool(d)


def load_settings(path=DEFAULT_CONFIG, seed=None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "generator" not in cfg:
        raise KeyError(f"{path} has no `generator:` section - see README, 'Synthetic generator'.")
    g = cfg["generator"]
    layers = g["layers"]
    fx, cc, cs = g["flag_effects"], g["compose"], g["camera_sim"]
    ie = cfg["image_effects"]

    # --- the frame -------------------------------------------------------------
    cam = cfg["camera"]
    width, height = cam["output_resolution"]
    alt = cfg["flight"]["altitude_m"]

    # --- flags per frame: negatives come from dataset.negative_image_ratio --------
    neg = float(cfg["dataset"]["negative_image_ratio"])
    pos = {int(k): float(v) for k, v in g["flags_per_image"].items() if int(k) > 0}
    tot = sum(pos.values())
    flags_per_image = {0: neg, **{k: (1 - neg) * v / tot for k, v in pos.items()}}

    names = list(cfg["countries"]["class_names"])
    fc = g.get("focus_classes") or {}
    unknown = [c for c in fc.get("classes", []) if c not in names]
    if unknown:
        raise KeyError(f"generator.focus_classes: not in countries.class_names: {unknown}")
    focus = {"classes": [names.index(c) for c in fc.get("classes", [])],
             "share": float(fc.get("share", 0.0))}

    place = cfg["flag"]["placement"]
    partial = g["partial_flag_prob"] if place.get("allow_partial_image_truncation", True) \
        and not place.get("avoid_image_truncation", False) else 0.0

    # --- flag cloth: folds/wrinkles from flag.fabric -------------------------------
    fabric = cfg["flag"]["fabric"]
    short_m = float(cfg["flag"]["physical_height_m"])
    folds_on = fabric.get("enabled", False) and (fabric.get("folds") or fabric.get("wrinkles"))
    max_disp = float(fabric.get("maximum_surface_displacement_m", 0.03))

    flag_effects = {
        "fade_prob": fx["fade_prob"] if layers["flag_fading"] else 0.0,
        "fade_strength": fx["fade_strength"],
        "dirt_prob": fx["dirt_prob"] if layers["flag_dirt"] else 0.0,
        "dirt_strength": fx["dirt_strength"],
        "fold_prob": fx["fold_prob"] if folds_on else 0.0,
        "fold_amplitude": [0.0, max_disp / short_m],       # fraction of the short side
        "fold_shading": fx["fold_shading"],
        "tilt_jitter": fx["tilt_jitter"],
        "edge_feather_px": fx["edge_feather_px"],
    }

    shadows = cfg["flag"]["shadows"]
    light = cfg["synthesis"]["environment"]["lighting"]
    compose = {
        "tint_strength": cc["tint_strength"] if layers["colour_match"] else 0.0,
        "light_gain": cc["light_gain"] if layers["colour_match"] else [1.0, 1.0],
        "contact_shadow_prob": cc["contact_shadow_prob"]
        if shadows.get("enabled") and shadows.get("cast_on_ground") else 0.0,
        "contact_shadow_strength": cc["contact_shadow_strength"],
        "contact_shadow_offset_m": cc["contact_shadow_offset_m"],
        "cast_shadow_prob": cc["cast_shadow_prob"]
        if layers["scene_shadows"] and _on(light, "shadows", "enabled") else 0.0,
        "cast_shadow_strength": cc["cast_shadow_strength"],
    }

    # --- whole-frame effects: image_effects.*.enabled switch them on -------------
    def rng_if(on, value, off):
        return value if on else off

    bv, cv = ie["brightness_variation"], ie["contrast_variation"]
    camera_sim = {
        "haze_prob": rng_if(layers["haze"], cs["haze_prob"], 0.0),
        "haze_strength": cs["haze_strength"],
        "vignette_strength": rng_if(layers["vignette"], cs["vignette_strength"], [0.0, 0.0]),
        "defocus_prob": rng_if(ie["defocus_blur"]["enabled"], ie["defocus_blur"]["probability"], 0.0),
        "defocus_sigma_px": cs["defocus_sigma_px"],
        "motion_prob": rng_if(ie["motion_blur"]["enabled"], ie["motion_blur"]["probability"], 0.0),
        "speed_mps": cs["speed_mps"],
        "exposure_s": cs["exposure_s"],
        "read_noise": rng_if(ie["sensor_noise"]["enabled"], cs["read_noise"], [0.0, 0.0]),
        "shot_noise": rng_if(ie["sensor_noise"]["enabled"], cs["shot_noise"], [0.0, 0.0]),
        "gamma": rng_if(layers["gamma"], cs["gamma"], [1.0, 1.0]),
        "brightness": rng_if(bv["enabled"], bv["range"], [1.0, 1.0]),
        "contrast": rng_if(cv["enabled"], cv["range"], [1.0, 1.0]),
    }

    bg = g["backgrounds"]
    env = cfg["synthesis"]["environment"]
    dist = g["distractors"]
    tiling = cfg["training"]["tiling"]
    ann = cfg["annotations"]
    comp = ie["compression"]

    return {
        "config_path": str(path),
        "seed": int(seed if seed is not None else cfg["project"]["random_seed"]),
        "camera": {"name": cam["model"], "width": int(width), "height": int(height),
                   "hfov_deg": float(cam["hfov_deg"])},
        "altitude_m": [float(alt["min"]), float(alt["max"])],
        "flag_size_m": [float(cfg["flag"]["physical_width_m"]), short_m],
        "supersample": int(g["supersample"]),
        "flags_per_image": flags_per_image,
        "focus": focus,                               # bigger share for look-alike flags
        "allow_partial_prob": float(partial),
        "min_visible_fraction": float(ann["minimum_visible_fraction"]),
        "min_box_px": (float(ann["minimum_bbox_width_px"]), float(ann["minimum_bbox_height_px"])),
        "min_spacing_m": float(g["min_spacing_m"]) if place.get("avoid_overlapping_flags", True) else 0.0,
        "classes": list(cfg["countries"]["class_names"]),
        "flags_dir": ROOT / cfg["countries"]["texture_lookup_directory"],
        "flag_files": dict(g.get("flag_files") or {}),
        "backgrounds": {
            "dir": ROOT / bg["directory"],
            "max_upsample": float(bg["max_upsample"]),
            "type_distribution": dict(env.get("background_distribution") or {}),
            "jitter": {"brightness": [1.0, 1.0], "contrast": [1.0, 1.0],
                       "color_cast": float(bg["color_cast"]),
                       "markings_prob": bg["markings_prob"] if layers["runway_markings"] else 0.0,
                       "marking_width_m": bg["marking_width_m"],
                       "marking_length_m": bg["marking_length_m"]},
        },
        "flag_effects": flag_effects,
        "compose": compose,
        "distractors": {
            "enabled": bool(layers["distractors"]),
            "dir": ROOT / dist["directory"],
            "count": dist["count"],
            "kinds": dist["kinds"],
        },
        "camera_sim": camera_sim,
        "view": dict(g.get("view") or {}),            # camera tilt (generator/view.py)
        "jpeg_quality": int(comp["jpeg_quality"]) if comp.get("enabled", True) else 95,
        "output_dir": ROOT / cfg["dataset"]["output_directory"],
        "image_prefix": cfg["dataset"]["image_naming"]["prefix"],
        "zero_padding": int(cfg["dataset"]["image_naming"]["zero_padding"]),
        "split": dict(cfg["dataset"]["split"]),
        "tiles": {"enabled": bool(tiling["enabled"]), "size": int(tiling["tile_size"]),
                  "overlap": int(round(float(tiling["overlap"]) * int(tiling["tile_size"]))),
                  "keep_empty_prob": float(g["keep_empty_tile_prob"])},
        "previews": int(g["preview_frames"]) if cfg.get("previews", {}).get("enabled", True) else 0,
        "layers": dict(layers),
        "batch": dict(g.get("batch") or {}),          # generate_batch.py (real backgrounds)
    }
