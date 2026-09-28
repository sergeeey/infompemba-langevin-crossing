"""Where does the G2b W1 deviation (0.02-0.033) come from? Exploratory diagnostic, read-only."""

import sys

import numpy as np
from numba import njit, prange

sys.path.insert(0, r"E:\Метрологический эффект Мпемба в информационной геометрии нейросетей")
from run_gates_v2_langevin import _sample_from_pmf, _w1_empirical
from src.mpemba_exact import (
    cold_ic,
    gaussian_ic,
    lambda1_eig,
    make_grid,
    potential,
    richardson,
    run_pair,
)


@njit(parallel=True)
def langevin(x0, barrier, temperature, dt, rec_steps, lim):
    n = x0.size
    out = np.empty((n, rec_steps.size))
    last = rec_steps[-1]
    for i in prange(n):
        x = x0[i]
        j = 0
        for s in range(1, last + 1):
            m = 1 + int(dt * 12.0 * barrier * x * x / lim)
            h = dt / m
            amp = np.sqrt(2.0 * temperature * h)
            for _ in range(m):
                x += -4.0 * barrier * x * (x * x - 1.0) * h + amp * np.random.normal()
            if s == rec_steps[j]:
                out[i, j] = x
                j += 1
    return out


b, T = 1.0, 0.3
fn = lambda x: potential(x, b, 0.0)
grid = make_grid(fn, T, 15.0, 0.01)
tau = 1.0 / lambda1_eig(make_grid(fn, T, 4.0, 0.01))
t_end = 3 * tau
coarse, fine = run_pair(grid, np.stack([cold_ic(grid), gaussian_ic(grid, 3.0)], axis=1), t_end)
n = min(coarse.t.size, fine.t.size)
ref_t = coarse.t[1:n]
w1_s = richardson(coarse.w1[1:n], fine.w1[1:n])[0]

t_rec = np.unique(
    np.concatenate([np.geomspace(0.02, t_end, 30), np.linspace(0.05 * tau, t_end, 40)])
)
for dt in (0.002, 0.0005):
    rec = np.unique(np.round(t_rec / dt).astype(np.int64))
    tr = rec * dt
    s_c = np.interp(np.log(tr), np.log(ref_t), w1_s[:, 0])
    s_h = np.interp(np.log(tr), np.log(ref_t), w1_s[:, 1])
    for lim in (0.5, 0.05):
        rng = np.random.default_rng(7)
        xc = langevin(_sample_from_pmf(grid, cold_ic(grid), 100_000, rng), b, T, dt, rec, lim)
        xh = langevin(np.clip(rng.normal(0, 3, 100_000), -15, 15), b, T, dt, rec, lim)
        wc = np.array([_w1_empirical(grid, xc[:, k]) for k in range(tr.size)])
        wh = np.array([_w1_empirical(grid, xh[:, k]) for k in range(tr.size)])
        dc, dh = np.abs(wc - s_c), np.abs(wh - s_h)
        kc, kh = int(dc.argmax()), int(dh.argmax())
        print(
            f"dt={dt} lim={lim}: cold max dev {dc[kc]:.4f} at t={tr[kc]:.3g} "
            f"(L {wc[kc]:.4f} S {s_c[kc]:.4f}); hot max dev {dh[kh]:.4f} at t={tr[kh]:.3g} "
            f"(L {wh[kh]:.4f} S {s_h[kh]:.4f}); late (t>2tau) cold dev {dc[tr > 2 * tau].mean():.4f} "
            f"hot dev {dh[tr > 2 * tau].mean():.4f}"
        )
# statistical floor: W1 of an exact equilibrium sample against the grid pi
rng = np.random.default_rng(11)
floors = [_w1_empirical(grid, _sample_from_pmf(grid, grid.pi, 100_000, rng)) for _ in range(5)]
print("W1 statistical floor at N=1e5 (equilibrium sample vs pi):", np.round(floors, 4))
