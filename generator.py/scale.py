import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Camera:
    name: str
    width: int  # image width in pixels
    height: int  # image height in pixels
    hfov_deg: float  # horizontal field of view in degrees