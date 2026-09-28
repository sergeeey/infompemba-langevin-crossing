"""G3 (positive control) and G4 (negative control) for PREREG_v2, D2.2 / D2.3 / D2.6.

G3: symmetric hot states at b/T = 10, kappa = 0 must be classified EFFECT (theory guarantees the
crossing: no slow-mode overlap). G4: pair (A = cold evolved to s*, B = cold at t = 0); by the data
processing inequality D_KL(A(t)) <= D_KL(B(t)) for every t, so no crossing is possible; the pipeline
must not report one. Criteria fixed before this code (PREREG_v2.md D2); a failed gate means the
pipeline is fixed, never the criterion.
"""

import sys
import time

import numpy as np

from src.mpemba_analysis import classify_metric, curves
from src.mpemba_exact import (
    cold_ic,
    evolve,
    gaussian_ic,
    kl_div,
    lambda1_eig,
    make_grid,
    potential,
    time_grid,
)

G3_CONFIGS = [(2.0, 0.2), (1.0, 0.1), (3.0, 0.3), (5.0, 0.5), (0.5, 0.05)]  # b/T = 10, kappa = 0
G4_CONFIGS = [(b, t, 0.0) for b, t in G3_CONFIGS] + [
    (2.0, 0.2, 0.05),
    (1.0, 0.1, 0.05),
    (3.0, 0.3, 0.1),
]
TOL_G4 = 1e-12
THERMAL_FACTOR = 8.0


def _setup(barrier: float, temperature: float, kappa: float):
    def fn(x):
        return potential(x, barrier, kappa)

    half = 4.0 if barrier >= 0.5 else 6.0
    lam = lambda1_eig(make_grid(fn, temperature, half, 0.01))
    return fn, lam, 8.0 / lam


def _thermal_ic(temperature: float):
    def build(grid):
        w = np.exp(-(grid.energy - grid.energy.min()) / (THERMAL_FACTOR * temperature))
        return cold_ic(grid), w / w.sum()

    return build


def gate_g3() -> bool:
    print("\n=== G3: positive control (b/T = 10, kappa = 0) ===")
    ok = True
    for barrier, temperature in G3_CONFIGS:
        fn, lam, t_max = _setup(barrier, temperature, 0.0)
        t0 = time.time()
        rows = {}
        for name, ic_fn in (
            ("gauss3", lambda g: (cold_ic(g), gaussian_ic(g, 3.0))),
            ("thermal8T", _thermal_ic(temperature)),
        ):
            cv = curves(fn, temperature, ic_fn, 1.05 * t_max)
            rows[name] = {m: classify_metric(cv.t, cv.d[m], cv.err[m], t_max) for m in ("kl", "w1")}
        g_ok = rows["gauss3"]["kl"].label == "EFFECT" and rows["gauss3"]["w1"].label == "EFFECT"
        t_ok = rows["thermal8T"]["kl"].label == "EFFECT"
        ok &= g_ok and t_ok
        print(
            f"  b={barrier} T={temperature} lam1={lam:.2e} t_max={t_max:.2e} "
            f"({time.time() - t0:.0f}s)\n"
            f"    gauss3   : KL {rows['gauss3']['kl'].label} (t*={rows['gauss3']['kl'].t_star:.3g}, "
            f"P0 {rows['gauss3']['kl'].p0_ratio:.3g}), W1 {rows['gauss3']['w1'].label} "
            f"(P0 {rows['gauss3']['w1'].p0_ratio:.3g})  {'PASS' if g_ok else 'FAIL'}\n"
            f"    thermal8T: KL {rows['thermal8T']['kl'].label} "
            f"(t*={rows['thermal8T']['kl'].t_star:.3g}, P0 {rows['thermal8T']['kl'].p0_ratio:.3g}), "
            f"W1 {rows['thermal8T']['w1'].label} (P0 {rows['thermal8T']['w1'].p0_ratio:.3g}, "
            f"not counted)  {'PASS' if t_ok else 'FAIL'}"
        )
    return ok


def _g4_pair(lam: float):
    def build(grid):
        cold = cold_ic(grid)
        target = 0.8 * kl_div(grid, cold)
        times = time_grid(4.0 / lam, 1.002)
        probe = evolve(grid, cold, times, record_stride=1)
        hits = np.flatnonzero(probe.kl[:, 0] <= target)
        if hits.size == 0:
            raise RuntimeError("cold state did not relax to 0.8 of its initial KL")
        k = int(hits[0])
        state = evolve(grid, cold, times[: k + 1], record_stride=10**9).final_rho[:, 0]
        return state, cold  # column 0 plays "cold" (closer), column 1 plays "hot" (farther)

    return build


def gate_g4() -> bool:
    print("\n=== G4: negative control, monotone relaxation (max KL gap <= 1e-12 on raw runs) ===")
    ok = True
    for barrier, temperature, kappa in G4_CONFIGS:
        fn, lam, t_max = _setup(barrier, temperature, kappa)
        t0 = time.time()
        cv = curves(fn, temperature, _g4_pair(lam), 1.05 * t_max)
        gap_c = cv.raw_coarse["kl"][:, 0] - cv.raw_coarse["kl"][:, 1]
        gap_f = cv.raw_fine["kl"][:, 0] - cv.raw_fine["kl"][:, 1]
        worst = float(max(gap_c.max(), gap_f.max()))
        gap_x = cv.d["kl"][:, 0] - cv.d["kl"][:, 1]
        w1_gap = float((cv.d["w1"][:, 0] - cv.d["w1"][:, 1]).max())
        p0 = cv.d["kl"][0, 1] / cv.d["kl"][0, 0]
        out = classify_metric(cv.t, cv.d["kl"], cv.err["kl"], t_max)
        passed = (
            worst <= TOL_G4 and p0 >= 1.25 and out.label in ("NO_CROSSING", "CROSSING_NO_MARGIN")
        )
        ok &= passed
        print(
            f"  b={barrier} T={temperature} k={kappa} ({time.time() - t0:.0f}s): P0(KL)={p0:.3f}; "
            f"max raw KL gap {worst:.2e}; max extrapolated gap {gap_x.max():.2e}; "
            f"label {out.label}; W1 max gap (report only) {w1_gap:.2e}  {'PASS' if passed else 'FAIL'}"
        )
    return ok


def main() -> int:
    t0 = time.time()
    results = {"G3": gate_g3(), "G4": gate_g4()}
    print(f"\nelapsed {time.time() - t0:.0f}s")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
