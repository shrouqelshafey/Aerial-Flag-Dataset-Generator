"""Stage 10 - train / val / test split BY BACKGROUND GROUP.

If the same patch of ground is in train and in test, the test score is too
optimistic: the model has already "seen" that ground. So each background group
(one site / one shooting session) belongs to exactly one split.
"""
import numpy as np

SPLITS = ("train", "val", "test")


def assign_groups(groups, ratios, seed):
    g = sorted(groups)
    rng = np.random.default_rng([seed, 7])
    rng.shuffle(g)
    n = len(g)
    if n < 3:
        print(f"WARNING: only {n} background group(s): train/val/test share backgrounds. "
              "Add more backgrounds before trusting val/test scores.")
        return {s: list(g) for s in SPLITS}
    n_val = max(1, round(n * ratios["val"]))
    n_test = max(1, round(n * ratios["test"]))
    n_train = max(1, n - n_val - n_test)
    return {"train": g[:n_train], "val": g[n_train:n_train + n_val],
            "test": g[n_train + n_val:n_train + n_val + n_test]}


def pick_split(idx, ratios):
    """Deterministic and evenly spread: any 20 consecutive images follow the ratios.

    (idx * golden ratio) mod 1 is a low-discrepancy sequence - it fills 0..1 evenly,
    so 200 images give ~140/30/30 instead of a random 141/40/19.
    """
    u = (idx * 0.6180339887498949) % 1.0
    p = np.array([ratios[s] for s in SPLITS], float)
    cum = np.cumsum(p / p.sum())
    return SPLITS[int(np.searchsorted(cum, u, side="right"))]
