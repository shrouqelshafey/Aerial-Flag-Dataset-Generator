"""Stage 10 - train / val / test split BY BACKGROUND GROUP.

If the same patch of ground is in train and in test, the test score is too
optimistic: the model has already "seen" that ground. So each background group
(one site / one shooting session) belongs to exactly one split.
"""
import numpy as np

SPLITS = ("train", "val", "test")


def assign_groups(groups, ratios, seed, fixed=None):
    """{split: [groups]}. fixed = {group: split} written in an index.yaml (`split:`):
    those groups keep their split, so ground cut from the drone maps is in the SAME
    split in generate.py and generate_batch.py. The other groups are shared out by ratio."""
    fixed = {g: s for g, s in (fixed or {}).items() if g in set(groups)}
    bad = {g: s for g, s in fixed.items() if s not in SPLITS}
    if bad:
        raise ValueError(f"unknown split in index.yaml: {bad} (use train / val / test)")
    out = {s: sorted(g for g, sp in fixed.items() if sp == s) for s in SPLITS}
    g = sorted(set(groups) - set(fixed))
    rng = np.random.default_rng([seed, 7])
    rng.shuffle(g)
    n = len(g)
    if n < 3:
        if fixed and all(out.values()):
            out["train"] += g                 # every split already has fixed ground
            return out
        if not fixed:
            print(f"WARNING: only {n} background group(s): train/val/test share backgrounds. "
                  "Add more backgrounds before trusting val/test scores.")
            return {s: list(g) for s in SPLITS}
    n_val = max(1, round(n * ratios["val"]))
    n_test = max(1, round(n * ratios["test"]))
    n_train = max(1, n - n_val - n_test)
    out["train"] += g[:n_train]
    out["val"] += g[n_train:n_train + n_val]
    out["test"] += g[n_train + n_val:n_train + n_val + n_test]
    for s in SPLITS:
        if not out[s]:                        # never leave a split without ground
            out[s] = list(out["train"])
            print(f"WARNING: no background group for {s}: it shares ground with train.")
    return out


def pick_split(idx, ratios):
    """Deterministic and evenly spread: any 20 consecutive images follow the ratios.

    (idx * golden ratio) mod 1 is a low-discrepancy sequence - it fills 0..1 evenly,
    so 200 images give ~140/30/30 instead of a random 141/40/19.
    """
    u = (idx * 0.6180339887498949) % 1.0
    p = np.array([ratios[s] for s in SPLITS], float)
    cum = np.cumsum(p / p.sum())
    return SPLITS[int(np.searchsorted(cum, u, side="right"))]
