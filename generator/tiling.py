"""Stage 9 - cut a full 4K frame into overlapping square tiles (config: training.tiling).

Why: a 3840 px frame shrunk to a 1024 px training image makes a 45 px flag only
12 px - too small to tell the countries apart. A tile keeps the full resolution.
The overlap (20 % of 1024 = 205 px) is bigger than the biggest flag (~100 px
corner to corner), so every flag is whole in at least one tile.
"""


def _axis(n, tile, overlap):
    if n <= tile:
        return [0]
    stride = tile - overlap
    xs = list(range(0, n - tile + 1, stride))
    if xs[-1] != n - tile:
        xs.append(n - tile)
    return xs


def tile_origins(width, height, tile=1024, overlap=205):
    return [(x, y) for y in _axis(height, tile, overlap) for x in _axis(width, tile, overlap)]
