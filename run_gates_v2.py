"""Validation gates V1-V3 for the exact solver (PREREG_v2.md, deviation D1).

Tolerances were recorded in PREREG_v2.md before this code existed. A failed gate means the solver
is fixed, never the tolerance. Read-only with respect to project data; prints a numeric report.
Revised after review (D1.3): V1 covers every recorded moment, V2 checks every step, V3 covers
every grid point with b/T <= 12 for all kappa.
"""

import sys
import time

import numpy as np
from scipy.special import ndtr

from src.mpemba_exact import (
    apply_q,
    cold_ic,
    evolve,
    gaussian_ic,
    kl_div,
    lambda1_eig,
    lambda1_estimate,
    make_grid,
    out_rates,
    potential,
    richardson,
    run_pair,
    time_grid,
)

TOL_V1 = 5e-3
D_FLOOR = 1e-6
TOL_V2_RESID = 1e-10
TOL_V2_MASS = 1e-10
TOL_V2_STAT = 1e-18
TOL_V2_L1 = 1e-6
TOL_V3_FACTOR = 1.5

BARRIERS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
TEMPERATURES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
KAPPAS = [0.0, 0.02, 0.05, 0.1]


def exact_gaussian_metrics(m0: float, s0: float, temperature: float, t: np.ndarray):
    """OU dX = -X dt + sqrt(2T) dW, X0 ~ N(m0, s0^2); equilibrium N(0, T)."""
    mean = m0 * np.exp(-t)
    var = temperature + (s0**2 - temperature) * np.exp(-2.0 * t)
    kl = 0.5 * (var / temperature + mean**2 / temperature - 1.0 - np.log(var / temperature))
    xs = np.linspace(-14.0, 14.0, 140001)
    dxs = xs[1] - xs[0]
    cdf_eq = ndtr(xs / np.sqrt(temperature))
    w1 = np.array(
        [
            np.sum(np.abs(ndtr((xs - mu) / np.sqrt(v)) - cdf_eq)) * dxs
            for mu, v in zip(mean, var, strict=True)
        ]
    )
    return kl, w1


def gate_v1() -> bool:
    print("\n=== V1: harmonic OU vs analytic, every recorded moment (tol rel 5e-3, D >= 1e-6) ===")
    ok = True
    for temperature in (0.2, 1.0):
        grid = make_grid(lambda x: 0.5 * x**2, temperature, half_width=8.0, dx=0.01)
        rho0 = gaussian_ic(grid, sigma=0.3, centre=1.5)
        coarse, fine = run_pair(grid, rho0, t_end=12.0)
        n = min(coarse.t.size, fine.t.size)
        t = coarse.t[:n]
        if not np.allclose(t, fine.t[:n], rtol=1e-9, atol=0):
            raise RuntimeError("coarse and fine time grids are misaligned")
        kl_ex, w1_ex = exact_gaussian_metrics(1.5, 0.3, temperature, t)
        for name, cs, fs, ex in (
            ("KL", coarse.kl[:n, 0], fine.kl[:n, 0], kl_ex),
            ("W1", coarse.w1[:n, 0], fine.w1[:n, 0], w1_ex),
        ):
            extrap, _ = richardson(cs, fs)
            sel = ex >= D_FLOOR
            worst_c = float((np.abs(cs[sel] - ex[sel]) / ex[sel]).max())
            worst_f = float((np.abs(fs[sel] - ex[sel]) / ex[sel]).max())
            worst = float((np.abs(extrap[sel] - ex[sel]) / ex[sel]).max())
            passed = worst <= TOL_V1
            ok &= passed
            print(
                f"  T={temperature:<4} {name}: raw coarse {worst_c:.2e}  raw fine {worst_f:.2e}  "
                f"richardson {worst:.2e}  points={int(sel.sum())}/{n}  "
                f"{'PASS' if passed else 'FAIL'}"
            )
    return ok


def gate_v2() -> bool:
    print("\n=== V2: discrete equilibrium, conservation, positivity (every step) ===")
    ok = True
    barrier, temperature, kappa = 2.0, 0.2, 0.05
    grid = make_grid(lambda x: potential(x, barrier, kappa), temperature, half_width=15.0, dx=0.01)
    res = np.abs(apply_q(grid, grid.pi))
    mask = grid.pi > 1e-200
    resid = float((res[mask] / (grid.pi[mask] * out_rates(grid)[mask])).max())
    passed = resid <= TOL_V2_RESID
    ok &= passed
    print(f"  (a) max relative stationarity residual: {resid:.2e}  {'PASS' if passed else 'FAIL'}")

    times = time_grid(1e6, 1.002)
    traj = evolve(grid, grid.pi, times, record_stride=1, check_every_step=True)
    worst = float(traj.kl.max())
    passed = worst <= TOL_V2_STAT
    ok &= passed
    print(
        f"  (d) max D_KL over all {traj.t.size} steps from rho=pi: {worst:.2e}  "
        f"{'PASS' if passed else 'FAIL'}"
    )

    rho0 = np.stack([cold_ic(grid), gaussian_ic(grid, 3.0)], axis=1)
    traj = evolve(grid, rho0, times, record_stride=10, check_every_step=True)
    passed_b = traj.mass_drift <= TOL_V2_MASS
    ok &= passed_b
    print(
        f"  (b) max |sum(rho) - 1| over all steps: {traj.mass_drift:.2e}  "
        f"{'PASS' if passed_b else 'FAIL'}"
    )
    passed_c = traj.min_rho >= 0.0
    ok &= passed_c
    print(
        f"  (c) min rho over all steps: {traj.min_rho:.2e}; no clipping is applied (positivity "
        f"holds by construction)  {'PASS' if passed_c else 'FAIL'}"
    )

    def fn0(x):
        return potential(x, barrier, 0.0)

    grid0 = make_grid(fn0, temperature, half_width=15.0, dx=0.01)
    lam1 = lambda1_estimate(fn0, temperature)
    t_max = 8.0 / lam1
    traj = evolve(grid0, gaussian_ic(grid0, 3.0), time_grid(t_max, 1.002), record_stride=50)
    rho = traj.final_rho[:, 0]
    l1 = float(np.abs(rho - grid0.pi).sum())
    passed = l1 <= TOL_V2_L1
    ok &= passed
    print(
        f"  (e) b=2,T=0.2,k=0: lambda1_est={lam1:.3e} t_max={t_max:.3e}  ||rho-pi||_1 = {l1:.2e}  "
        f"final KL {kl_div(grid0, rho):.2e}  {'PASS' if passed else 'FAIL'}  "
        "(equilibrium reached; does NOT validate lambda1, see D1.3)"
    )
    return ok


def gate_v3() -> bool:
    print("\n=== V3: lambda1 quadrature vs eigvalsh, all grid points b/T <= 12, all kappa ===")
    factors = []
    failures = []
    for kappa in KAPPAS:
        for barrier in BARRIERS:
            for temperature in TEMPERATURES:
                if barrier / temperature > 12.0:
                    continue

                def fn(x, b=barrier, k=kappa):
                    return potential(x, b, k)

                half = 4.0 if barrier >= 0.5 else 6.0
                grid = make_grid(fn, temperature, half_width=half, dx=0.01)
                lam_est = lambda1_estimate(fn, temperature)
                lam_eig = lambda1_eig(grid)
                if (
                    not (np.isfinite(lam_est) and np.isfinite(lam_eig))
                    or min(lam_est, lam_eig) <= 0
                ):
                    failures.append((kappa, barrier, temperature, "non-positive or non-finite"))
                    continue
                factor = max(lam_est / lam_eig, lam_eig / lam_est)
                factors.append((factor, kappa, barrier, temperature, lam_est, lam_eig))
                if factor > TOL_V3_FACTOR:
                    failures.append((kappa, barrier, temperature, round(factor, 3)))
    factors.sort(reverse=True)
    print(f"  points with a valid factor: {len(factors)}")
    for f in factors[:5]:
        print(
            f"  worst: kappa={f[1]} b={f[2]} T={f[3]} est={f[4]:.4e} eig={f[5]:.4e} factor={f[0]:.3f}"
        )
    passed = not failures
    print(f"  failures: {failures if failures else 'none'}  {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    t0 = time.time()
    results = {"V1": gate_v1(), "V2": gate_v2(), "V3": gate_v3()}
    print(f"\nelapsed {time.time() - t0:.0f}s")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
