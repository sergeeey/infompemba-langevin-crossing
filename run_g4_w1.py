"""G4 for W1 (PREREG_v2.md D2.9b, definition fixed before this code): a negative control for the
Wasserstein-1 metric on a convex potential.

For U = x^2 / 2 the overdamped flow is a W1 contraction (synchronous coupling), so the pair
(A = B evolved to s*, B) satisfies W1(A_t, pi) <= W1(B_t, pi) for all t; a pipeline that reports an
EFFECT in W1 here is broken. Pass: W1 label in {NO_CROSSING, CROSSING_NO_MARGIN} and P0(W1) >= 1.25.
The double-well W1 gaps of the original G4 pairs were report-only; this closes the W1 half on a system
where the guarantee holds.
"""

import sys

import numpy as np

from src.mpemba_analysis import classify_metric, curves
from src.mpemba_exact import evolve, gaussian_ic, time_grid, w1_dist

TEMPERATURES = [0.2, 0.5, 1.0]
T_MAX = 8.0  # lambda1 = 1 for U = x^2 / 2


def _pair(grid):
    b = gaussian_ic(grid, 0.3, 1.5)
    target = 0.8 * w1_dist(grid, b)
    times = time_grid(T_MAX, 1.002)
    probe = evolve(grid, b, times, record_stride=1)
    hits = np.flatnonzero(probe.w1[:, 0] <= target)
    if hits.size == 0:
        raise RuntimeError("state did not relax to 0.8 of its initial W1")
    k = int(hits[0])
    a = evolve(grid, b, times[: k + 1], record_stride=10**9).final_rho[:, 0]
    return a, b  # column 0 = closer ('cold' role), column 1 = farther ('hot' role)


def main() -> int:
    ok = True
    print("G4-W1: convex potential U = x^2/2, pair (A = B evolved to s*, B)")
    for temperature in TEMPERATURES:
        cv = curves(lambda x: 0.5 * x**2, temperature, _pair, 1.05 * T_MAX, half_width=8.0)
        out = classify_metric(cv.t, cv.d["w1"], cv.err["w1"], T_MAX)
        gap_c = cv.raw_coarse["w1"][:, 0] - cv.raw_coarse["w1"][:, 1]
        gap_f = cv.raw_fine["w1"][:, 0] - cv.raw_fine["w1"][:, 1]
        passed = out.label in ("NO_CROSSING", "CROSSING_NO_MARGIN") and out.p0_ratio >= 1.25
        ok &= passed
        print(
            f"  T={temperature}: P0(W1)={out.p0_ratio:.3f}; label {out.label}; max raw W1 gap "
            f"coarse {gap_c.max():.2e} fine {gap_f.max():.2e}  {'PASS' if passed else 'FAIL'}"
        )
    print(f"\nG4-W1: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
