"""EXPLORATORY (PREREG_v2.md D2.9a, definitions fixed before this code): why does the v1 metric on exact
trajectories differ from v1's own numbers?

Runs v1's own process (x marginal; cold ~ N(1, 0.05^2), hot ~ N(0, 2^2); Euler-Maruyama eta = 0.005,
30000 steps; p_left every 50 steps; v1 metric with the tail-estimated equilibrium) at 4 points chosen by
the pre-fixed rule (2 largest and 2 smallest |exact-solver metric - v1 parquet mean|), with and without
v1's gradient/position clipping, at N = 5000 and N = 50000, and compares with the exact-solver value.
"""

import sys

import numpy as np
import pandas as pd
from numba import njit, prange

ETA, N_STEPS, EVERY = 0.005, 30000, 50


@njit(parallel=True, cache=True)
def _p_left(x0, barrier, temperature, clip_on, seed):
    n = x0.size
    n_rec = N_STEPS // EVERY
    left = np.zeros((n, n_rec), dtype=np.uint8)
    noise_std = np.sqrt(2.0 * temperature * ETA)
    for i in prange(n):
        np.random.seed(seed + i)
        x = x0[i]
        j = 0
        for s in range(1, N_STEPS + 1):
            g = 4.0 * barrier * x * (x * x - 1.0)
            if clip_on:
                g = min(max(g, -100.0), 100.0)
            x = x - ETA * g + noise_std * np.random.normal()
            if clip_on:
                x = min(max(x, -50.0), 50.0)
            if s % EVERY == 0:
                left[i, j] = 1 if x < 0.0 else 0
                j += 1
    return left.sum(axis=0) / n


def v1_pct(cold, hot):
    tail = max(1, len(cold) // 10)
    eq = (cold[-tail:].mean() + hot[-tail:].mean()) / 2.0
    return 100.0 * float((np.abs(hot - eq) < np.abs(cold - eq)).mean())


def run(barrier, temperature, n, clip_on, seed):
    rng = np.random.default_rng(seed)
    cold = _p_left(rng.normal(1.0, 0.05, n), barrier, temperature, clip_on, seed)
    hot = _p_left(rng.normal(0.0, 2.0, n), barrier, temperature, clip_on, seed + 10_000_000)
    return v1_pct(cold, hot)


def main() -> int:
    cmp = pd.read_csv("prereg_v2/v1_compare_points.csv").sort_values("abs_diff")
    pts = pd.concat([cmp.tail(2), cmp.head(2)])  # 2 largest, 2 smallest differences
    variants = [
        ("N=5000  clip", 5000, True, [42, 43, 44, 45, 46]),
        ("N=5000  noclip", 5000, False, [42, 43, 44, 45, 46]),
        ("N=50000 clip", 50000, True, [42, 43]),
        ("N=50000 noclip", 50000, False, [42, 43]),
    ]
    fin_ok = clip_ok = 0
    for _, r in pts.iterrows():
        b, t, exact = float(r["b"]), float(r["T"]), float(r["pl_v1pct"])
        print(
            f"\nb={b} T={t}: exact-solver v1-metric {exact:.1f}%   v1 parquet {r['v1_mean']:.1f} +- "
            f"{r['v1_sd']:.1f} (10 seeds)"
        )
        means = {}
        for name, n, clip_on, seeds in variants:
            vals = np.array([run(b, t, n, clip_on, s) for s in seeds])
            means[name] = vals.mean()
            print(
                f"  {name:15s}: {vals.mean():6.1f} +- {vals.std(ddof=1) if len(vals) > 1 else 0:5.1f}"
                f"  (n_seeds {len(vals)})"
            )
        gap0 = abs(means["N=5000  clip"] - exact)
        fin = gap0 - abs(means["N=50000 clip"] - exact) >= 0.5 * gap0 and gap0 > 0
        clp = gap0 - abs(means["N=5000  noclip"] - exact) >= 0.5 * gap0 and gap0 > 0
        fin_ok += int(fin)
        clip_ok += int(clp)
        print(
            f"  initial gap {gap0:.1f} pp; larger N closes >= half: {fin}; no clipping closes >= half: {clp}"
        )
    print(
        f"\nreading rule (D2.9a): finite-sample supported at {fin_ok}/4 points, clipping at {clip_ok}/4 points "
        f"(threshold 3/4): finite-sample {'YES' if fin_ok >= 3 else 'no'}, clipping {'YES' if clip_ok >= 3 else 'no'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
