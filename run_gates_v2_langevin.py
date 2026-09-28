"""Gates G1/G2 for the exact solver: independent Langevin cross-check and grid refinement.

Tolerances were recorded in PREREG_v2.md (Deviations D1.2) before this code was written.
Langevin here is Euler-Maruyama, so it has O(dt) weak error; a failed gate means the method of
comparison or the solver is fixed, never the tolerance.
"""

import sys
import time

import numpy as np
from numba import njit, prange

from src.mpemba_exact import (
    cold_ic,
    gaussian_ic,
    lambda1_eig,
    lambda1_estimate,
    make_grid,
    potential,
    richardson,
    run_pair,
)

POINTS = [(1.0, 0.3), (1.0, 0.2), (0.5, 0.2)]  # (barrier, temperature), kappa = 0
N_PARTICLES = 100_000
DT = 0.002
TOL_G1 = 0.10
TOL_G2A = 1e-2
D_FLOOR = 1e-6
TOL_G2B_W1 = 0.02
GAP_MIN = 0.03
N_BLOCKS = 10
MIN_STRONG_POINTS = 10  # non-vacuity guard for the gap-sign test (added after review, D1.4)


@njit(parallel=True, cache=True)
def _langevin(x0, barrier, temperature, dt, rec_steps, seed):
    """Euler-Maruyama for U = b (x^2-1)^2; positions at the recorded step indices.

    Steps are subdivided where dt * U'' would exceed 0.05 (far tails of the hot state; the first
    run used 0.5 and failed G2b through Euler error there, see PREREG D1.3), so that
    every particle advances the same physical time per outer step.

    Reproducibility (D1.4): particle i is seeded with seed + i at the start of its own loop, so
    its noise stream does not depend on which thread runs it or on scheduling order.
    """
    n = x0.size
    out = np.empty((n, rec_steps.size))
    last = rec_steps[-1]
    for i in prange(n):
        np.random.seed(seed + i)
        x = x0[i]
        j = 0
        for s in range(1, last + 1):
            m = 1 + int(dt * 12.0 * barrier * x * x / 0.05)
            h = dt / m
            amp = np.sqrt(2.0 * temperature * h)
            for _ in range(m):
                x += -4.0 * barrier * x * (x * x - 1.0) * h + amp * np.random.normal()
            if s == rec_steps[j]:
                out[i, j] = x
                j += 1
    return out


def _sample_from_pmf(grid, pmf, n, rng):
    idx = np.searchsorted(np.cumsum(pmf), rng.random(n))
    idx = np.minimum(idx, grid.x.size - 1)
    return grid.x[idx] + grid.dx * (rng.random(n) - 0.5)


def _fit_rate(t, s):
    sel = (s > 0.05) & (s < 0.5)
    if sel.sum() < 6:
        return np.nan
    return -np.polyfit(t[sel], np.log(s[sel]), 1)[0]


def _w1_empirical(grid, samples):
    xs = np.sort(samples)
    cdf = np.searchsorted(xs, grid.x, side="right") / xs.size
    return float(np.sum(np.abs(cdf - grid.cdf_pi)) * grid.dx)


def run_point(barrier: float, temperature: float, seed: int) -> tuple[bool, bool, bool]:
    print(f"\n--- b={barrier}, T={temperature}, kappa=0 ---")
    fn = lambda x: potential(x, barrier, 0.0)
    grid = make_grid(fn, temperature, half_width=15.0, dx=0.01)
    lam_eig = lambda1_eig(make_grid(fn, temperature, half_width=4.0, dx=0.01))
    lam_est = lambda1_estimate(fn, temperature)
    tau = 1.0 / lam_eig
    t_end = 3.0 * tau

    # solver (coarse+fine, Richardson) at two resolutions
    runs = {}
    for dx in (0.01, 0.005):
        g = make_grid(fn, temperature, half_width=15.0, dx=dx)
        rho0 = np.stack([cold_ic(g), gaussian_ic(g, 3.0)], axis=1)
        # WHY 1.05: the recorded solver times stop 1-2% before t_end (stride alignment), and
        # np.interp clamps there; extending the run keeps the last Langevin times inside range.
        coarse, fine = run_pair(g, rho0, 1.05 * t_end)
        n = min(coarse.t.size, fine.t.size)
        runs[dx] = {
            "t": coarse.t[:n],
            "kl": richardson(coarse.kl[:n], fine.kl[:n])[0],
            "w1": richardson(coarse.w1[:n], fine.w1[:n])[0],
        }

    # G2a: grid refinement
    ok2a = True
    for name in ("kl", "w1"):
        a, b = runs[0.01][name], runs[0.005][name]
        sel = b >= D_FLOOR
        rel = float((np.abs(a[sel] - b[sel]) / b[sel]).max())
        passed = rel <= TOL_G2A
        ok2a &= passed
        print(
            f"  G2a {name}: max rel change dx 0.01 -> 0.005 = {rel:.2e} {'PASS' if passed else 'FAIL'}"
        )

    # Langevin
    rng = np.random.default_rng(seed)
    t_rec = np.unique(
        np.concatenate([np.geomspace(0.02, t_end, 30), np.linspace(0.05 * tau, t_end, 40)])
    )
    rec_steps = np.unique(np.round(t_rec / DT).astype(np.int64))
    t_rec = rec_steps * DT
    t0 = time.time()
    x_cold = _langevin(
        _sample_from_pmf(grid, cold_ic(grid), N_PARTICLES, rng),
        barrier,
        temperature,
        DT,
        rec_steps,
        seed,
    )
    x_hot0 = np.clip(rng.normal(0.0, 3.0, N_PARTICLES), -15.0, 15.0)
    x_hot = _langevin(x_hot0, barrier, temperature, DT, rec_steps, seed + 10_000_000)
    print(
        f"  Langevin: {N_PARTICLES} particles x 2, {rec_steps[-1]} steps, {time.time() - t0:.0f}s"
    )

    # G1: lambda1 from decay of <sign x>
    s_all = np.sign(x_cold).mean(axis=0)
    lam_l = _fit_rate(t_rec, s_all)
    blocks = [_fit_rate(t_rec, np.sign(x_cold[k::N_BLOCKS]).mean(axis=0)) for k in range(N_BLOCKS)]
    err = float(np.nanstd(blocks) / np.sqrt(N_BLOCKS))
    ok1 = True
    for name, ref in (("eig", lam_eig), ("quadrature", lam_est)):
        dev = abs(lam_l / ref - 1.0)
        passed = dev <= TOL_G1
        ok1 &= passed
        print(
            f"  G1 lambda1 Langevin {lam_l:.4e} (+-{err:.1e}) vs {name} {ref:.4e}: "
            f"dev {dev:.3f} {'PASS' if passed else 'FAIL'}"
        )

    # G2b: W1 of both states vs solver, and sign of the gap
    w1_c = np.array([_w1_empirical(grid, x_cold[:, k]) for k in range(t_rec.size)])
    w1_h = np.array([_w1_empirical(grid, x_hot[:, k]) for k in range(t_rec.size)])
    ref_t = runs[0.005]["t"][1:]
    s_c = np.interp(np.log(t_rec), np.log(ref_t), runs[0.005]["w1"][1:, 0])
    s_h = np.interp(np.log(t_rec), np.log(ref_t), runs[0.005]["w1"][1:, 1])
    dev = max(np.abs(w1_c - s_c).max(), np.abs(w1_h - s_h).max())
    passed_dev = dev <= TOL_G2B_W1
    gap_l, gap_s = w1_h - w1_c, s_h - s_c
    strong = np.abs(gap_s) > GAP_MIN
    # WHY min count: np.all over an empty selection is True; require the test to have real content
    sign_ok = int(strong.sum()) >= MIN_STRONG_POINTS and bool(
        np.all(np.sign(gap_l[strong]) == np.sign(gap_s[strong]))
    )
    print(
        f"  G2b max|W1_L - W1_S| = {dev:.4f} {'PASS' if passed_dev else 'FAIL'}; "
        f"gap sign agrees on {int(strong.sum())} strong points: {'PASS' if sign_ok else 'FAIL'}"
    )
    return ok1, ok2a, passed_dev and sign_ok


def main() -> int:
    t0 = time.time()
    base = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"seed base: {base}")
    res = [run_point(b, t, seed=base + 100_000 * i) for i, (b, t) in enumerate(POINTS)]
    g1, g2a, g2b = (all(r[k] for r in res) for k in range(3))
    print(f"\nelapsed {time.time() - t0:.0f}s")
    print(
        f"G1: {'PASS' if g1 else 'FAIL'}  G2a: {'PASS' if g2a else 'FAIL'}  G2b: {'PASS' if g2b else 'FAIL'}"
    )
    return 0 if (g1 and g2a and g2b) else 1


if __name__ == "__main__":
    sys.exit(main())
