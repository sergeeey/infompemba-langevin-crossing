"""Why does the G1 lambda1 estimate scatter ~8% between runs while the block error says ~2%?

Exploratory diagnostic (PREREG D1.4). Point (b,T) = (1.0, 0.3), kappa = 0. Independent seeds; three
estimators of the slowest rate from <sign x>(t):
  A  the gate's estimator: unweighted log-fit over the NOISY window 0.05 < s < 0.5
  B  unweighted log-fit over a FIXED time window [0.7 tau, 3 tau] (tau from eigvalsh, data-independent)
  C  weighted log-fit over the same fixed window (delta method: sd(ln s) = sqrt((1-s^2)/N) / s)
plus a particle bootstrap of A's standard error on one draw. Read-only.
"""

import sys

import numpy as np

sys.path.insert(0, r"E:\Метрологический эффект Мпемба в информационной геометрии нейросетей")
from run_gates_v2_langevin import (
    DT,
    N_PARTICLES,
    _fit_rate,
    _langevin,
    _sample_from_pmf,
)
from src.mpemba_exact import cold_ic, lambda1_eig, make_grid, potential

B, T = 1.0, 0.3
N_SEEDS = 8


def fn(x):
    return potential(x, B, 0.0)


grid = make_grid(fn, T, 15.0, 0.01)
lam_eig = lambda1_eig(make_grid(fn, T, 4.0, 0.01))
tau = 1.0 / lam_eig
t_end = 3.0 * tau
t_rec = np.unique(
    np.concatenate([np.geomspace(0.02, t_end, 30), np.linspace(0.05 * tau, t_end, 40)])
)
rec_steps = np.unique(np.round(t_rec / DT).astype(np.int64))
t_rec = rec_steps * DT
fixed = (t_rec >= 0.7 * tau) & (t_rec <= 3.0 * tau)


def fit_fixed(s, weighted):
    ok = fixed & (s > 0)
    if ok.sum() < 6:
        return np.nan
    sd_ln = np.sqrt(np.maximum(1.0 - s[ok] ** 2, 1e-12) / N_PARTICLES) / s[ok]
    w = 1.0 / sd_ln if weighted else None
    return -np.polyfit(t_rec[ok], np.log(s[ok]), 1, w=w)[0]


est = {"A adaptive window": [], "B fixed window": [], "C fixed window, weighted": []}
first_signs = None
for k in range(N_SEEDS):
    seed = 7_000_000 + 1_000_000 * k
    rng = np.random.default_rng(seed)
    x = _langevin(
        _sample_from_pmf(grid, cold_ic(grid), N_PARTICLES, rng), B, T, DT, rec_steps, seed
    )
    signs = np.sign(x)
    s = signs.mean(axis=0)
    est["A adaptive window"].append(_fit_rate(t_rec, s))
    est["B fixed window"].append(fit_fixed(s, False))
    est["C fixed window, weighted"].append(fit_fixed(s, True))
    if k == 0:
        first_signs = signs

print(
    f"eigvalsh lambda1 = {lam_eig:.4e}  (tau = {tau:.2f}); {N_SEEDS} independent draws, N = {N_PARTICLES}"
)
for name, vals in est.items():
    v = np.array(vals)
    print(
        f"  {name:26s} mean {v.mean():.4e} (bias {v.mean() / lam_eig - 1:+.3f})  "
        f"sd {v.std(ddof=1):.2e} ({v.std(ddof=1) / v.mean():.3f} rel)  "
        f"min/max {v.min():.3e}/{v.max():.3e}"
    )

# particle bootstrap of estimator A on the first draw
rng = np.random.default_rng(1)
boots = []
for _ in range(60):
    idx = rng.integers(0, N_PARTICLES, N_PARTICLES)
    boots.append(_fit_rate(t_rec, first_signs[idx].mean(axis=0)))
boots = np.array(boots)
print(
    f"  bootstrap SE of A on draw 0: {np.nanstd(boots):.2e} ({np.nanstd(boots) / np.nanmean(boots):.3f} rel), "
    f"nan fits: {int(np.isnan(boots).sum())}"
)
