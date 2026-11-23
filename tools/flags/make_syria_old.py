"""Build assets/flags/syria_old.png (Syria's flag before March 2025) from syria.png (the 2025 flag).

    python tools/flags/make_syria_old.py

The 2025 flag: green / white / black bands, three red stars at 1/4, 1/2, 3/4 of the width.
The old flag:  red / white / black bands, two green stars at 1/3 and 2/3 of the width,
same star size. Both use the same red and green, so the script only swaps the colours
and moves a cut-out of the middle star; nothing is drawn by hand.
"""
from pathlib import Path

import numpy as np
from PIL import Image

FLAGS = Path(__file__).resolve().parents[2] / "assets" / "flags"


def mix(white, colour, t):
    """Anti-aliased pixel: t = 0 -> white, t = 1 -> colour."""
    return white * (1 - t) + colour * t


def amount(img, white, colour):
    """How far each pixel is from white towards colour (0..1)."""
    return np.clip(np.abs(img - white).sum(axis=2) / np.abs(colour - white).sum(), 0, 1)[..., None]


def main():
    new = np.array(Image.open(FLAGS / "syria.png").convert("RGB")).astype(np.float32)
    h, w = new.shape[:2]
    green = new[h // 8, w // 2].copy()        # top band
    red = new[h // 2, w // 2].copy()          # centre of the middle star
    white = new[h // 2, w // 40].copy()       # left end of the white band
    y0, y1 = h // 3 - 4, 2 * h // 3 + 4       # white band (+ its anti-aliased edges)

    old = new.copy()
    top = slice(0, y0 + 8)                    # top band and its edge: green -> red
    old[top] = mix(white, red, amount(new[top], white, green))

    sw = w // 4                               # middle star cut-out, centred on w / 2
    star_t = amount(new[y0 + 8:y1 - 8, w // 2 - sw // 2:w // 2 + sw // 2], white, red)
    old[y0 + 8:y1 - 8] = white                # empty white band
    for cx in (w // 3, 2 * w // 3):
        old[y0 + 8:y1 - 8, cx - sw // 2:cx + sw // 2] = mix(white, green, star_t)

    Image.fromarray(np.clip(np.rint(old), 0, 255).astype(np.uint8)).save(FLAGS / "syria_old.png")
    print(f"wrote {FLAGS / 'syria_old.png'} ({w}x{h})")


if __name__ == "__main__":
    main()
