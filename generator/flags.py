"""Flag artwork -> one texture per class, ready to print on a 2 x 1 m sheet.

1. tools/flags/flag_loader.py loads every image in assets/flags as RGBA
   (class name = file name, lowercase).
2. Each class in countries.class_names finds its file: same name, or the name
   given in generator.flag_files (united_states -> america.png).
3. The transparent margin around the artwork is cut off (alpha > 50 %).
4. The artwork is put on a 2:1 sheet WITHOUT stretching it
   (flag.design_mapping: preserve_aspect_ratio); the empty strip is filled by
   extending the flag's own edge colours (unused_area_handling: flag_background_fill).

Classes with no image are reported and simply not drawn; their class id stays
the same, so labels stay valid when the image is added later.
"""
import cv2
import numpy as np

from tools.flags.flag_loader import flag_to_array, load_all_flags


def artwork_to_sheet(rgba: np.ndarray, aspect: float) -> np.ndarray:
    """RGBA uint8 (RGB order) -> float32 BGR 0..1 texture with width/height == aspect."""
    a = rgba[..., 3]
    ys, xs = np.where(a > 128)
    if len(xs) == 0:
        raise ValueError("flag image is fully transparent")
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA).astype(np.float32) / 255.0
    alpha = bgr[..., 3:4]
    tex = bgr[..., :3] * alpha + (1 - alpha)        # any soft edge pixel over white cloth

    h, w = tex.shape[:2]
    if w / h < aspect:                              # too narrow: widen left + right
        extra = int(round(h * aspect)) - w
        tex = cv2.copyMakeBorder(tex, 0, 0, extra // 2, extra - extra // 2, cv2.BORDER_REPLICATE)
    elif w / h > aspect:                            # too wide: add top + bottom
        extra = int(round(w / aspect)) - h
        tex = cv2.copyMakeBorder(tex, extra // 2, extra - extra // 2, 0, 0, cv2.BORDER_REPLICATE)
    return tex


def load_flag_textures(flags_dir, class_names, flag_files, aspect):
    """Return ({class_id: texture}, [missing class names])."""
    images = load_all_flags(flags_dir)
    textures, missing = {}, []
    for cid, name in enumerate(class_names):
        key = str(flag_files.get(name, name)).lower()
        if key not in images:
            missing.append(name)
            continue
        textures[cid] = artwork_to_sheet(flag_to_array(images[key]), aspect)
    if not textures:
        raise FileNotFoundError(f"No flag image in {flags_dir} matches countries.class_names.")
    return textures, missing
