"""Stage 1 - scale math (pinhole camera / similar triangles).

f_px = (W / 2) / tan(HFOV / 2)      focal length expressed in pixels
GSD  = H / f_px                      metres of ground covered by one pixel
size_px = size_m / GSD               how many pixels an object of size_m covers

SIYI A8 mini, 3840 x 2160, HFOV 81 deg  ->  f = 2248 px
  50 m: GSD 2.2 cm/px, 2 m flag = 90 px      100 m: GSD 4.4 cm/px, flag = 45 px

    python -m generator.scale          prints this table from config/config.yaml
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Camera:
    name: str
    width: int  # image width in pixels
    height: int  # image height in pixels
    hfov_deg: float  # horizontal field of view in degrees

    @property
    def f_px(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    def gsd(self, altitude_m: float) -> float:
        """Ground sample distance in metres per pixel at this altitude."""
        return altitude_m / self.f_px

    def size_px(self, size_m: float, altitude_m: float) -> float:
        return size_m / self.gsd(altitude_m)

    def footprint_m(self, altitude_m: float):
        """Width and height of the ground area seen by the camera, in metres."""
        g = self.gsd(altitude_m)
        return self.width * g, self.height * g


def camera_from_settings(s: dict) -> Camera:
    c = s["camera"]
    return Camera(c["name"], int(c["width"]), int(c["height"]), float(c["hfov_deg"]))


if __name__ == "__main__":
    from .settings import load_settings
    s = load_settings()
    cam = camera_from_settings(s)
    print(f"{cam.name}: {cam.width}x{cam.height}, HFOV {cam.hfov_deg} deg, f = {cam.f_px:.0f} px")
    L, S = s["flag_size_m"]
    print(f"{'altitude':>9s} {'GSD':>9s} {'ground seen':>15s} {'flag':>13s}")
    lo, hi = s["altitude_m"]
    for h in sorted({lo, 60, 70, 80, 90, hi}):
        w, g = cam.footprint_m(h)
        print(f"{h:7.0f} m {cam.gsd(h)*100:6.2f} cm {w:6.0f} x {g:3.0f} m "
              f"{cam.size_px(L, h):5.1f} x {cam.size_px(S, h):4.1f} px")
