"""How much does each kind of reconstruction error cost on each metric?

Corrupts the true poses of all 20 public episodes with one error type at a time and
prints the mean score per metric. Rows marked "frozen" and "oracle" are the floor and
ceiling for comparison.

    .venv/bin/python t3/sweep.py
"""

from __future__ import annotations

import json

import numpy as np

import score as L

CASES = [
    ("oracle", None, 0.0),
    ("frozen", None, 0.0),
    ("jitter", "jitter", 0.005), ("jitter", "jitter", 0.01), ("jitter", "jitter", 0.03),
    ("drift", "drift", 0.03), ("drift", "drift", 0.05), ("drift", "drift", 0.10),
    ("scale", "scale", 0.05), ("scale", "scale", 0.10), ("scale", "scale", 0.20),
]
METRICS = ("add_auc", "spider_sr", "maniptrans_sr", "rpe_cm", "mppe_cm")


def main():
    episodes = json.load(open(L.DATA / "meta" / "info.json"))["episode_indices"]
    data = []
    for e in episodes:
        names, ref = L.load_reference(e)
        data.append((e, ref, [L.load_scan(n) for n in names]))

    print(f"{'case':8s} {'size':>6s} " + " ".join(f"{m:>13s}" for m in METRICS))
    for label, kind, size in CASES:
        rows = []
        for e, ref, scans in data:
            if label == "oracle":
                pred = ref.copy()
            elif label == "frozen":
                pred = np.repeat(ref[:1], len(ref), axis=0)
            else:
                pred = L.perturb(ref, np.random.default_rng(e), kind, size)
            rows.append(L.score_episode(e, pred, scans, ref, scans))
        means = [np.mean([r[m] for r in rows]) for m in METRICS]
        print(f"{label:8s} {size:6.3f} " + " ".join(f"{v:13.3f}" for v in means), flush=True)


if __name__ == "__main__":
    main()
