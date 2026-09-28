"""G5b: power check of the overlap criterion R, ADDED AFTER the main run (post-hoc, labelled as such).

Why: in the main run every eligible point had r = |c1_hot| / |c1_cold| in 0.003 .. 0.43, so R predicted
'crossing' everywhere and the 224/224 agreement could not have failed in the other direction. Here the
roles are arranged so that R predicts NO crossing (r > 1): the 'hot' (farther) state is the right-well
local equilibrium (|c1| ~ 1), the 'cold' (closer) state is a symmetric thermal state (c1 ~ 0 at kappa = 0).
Selection rule, fixed before running and blind to the outcome: among thermal factors f in FACTORS the
largest f with KL_C <= 0.8 * KL_H (so P0 >= 1.25 in KL). Expected by theory: no sustained crossing.
Pass criteria: r > 1.05 (the pair has the intended sign), direct label != EFFECT, last KL gap < 0.
Not used to change any K verdict; a failure would indicate a bug in R or in the pipeline.
"""

import sys
import time

import numpy as np

from src.mpemba_analysis import classify_metric, curves, slow_mode_overlaps
from src.mpemba_exact import cold_ic, kl_div, lambda1_eig, make_grid, potential

CONFIGS = [(2.0, 0.2, 0.0), (1.0, 0.1, 0.0), (3.0, 0.3, 0.0), (5.0, 0.5, 0.0), (0.5, 0.05, 0.0)] + [
    (2.0, 0.2, 0.05),
    (1.0, 0.1, 0.05),
    (3.0, 0.3, 0.1),
]
FACTORS = [1.05, 1.1, 1.2, 1.3, 1.5, 2.0]


def main() -> int:
    ok = True
    print("G5b (post-hoc): pairs where R predicts NO crossing")
    for barrier, temperature, kappa in CONFIGS:

        def fn(x, b=barrier, k=kappa):
            return potential(x, b, k)

        half = 4.0 if barrier >= 0.5 else 6.0
        t_max = 8.0 / lambda1_eig(make_grid(fn, temperature, half, 0.01))
        chosen = {}

        def ic_fn(grid, temperature=temperature, chosen=chosen):
            far = cold_ic(grid)
            kl_far = kl_div(grid, far)
            best = None
            for f in FACTORS:
                w = np.exp(-(grid.energy - grid.energy.min()) / (f * temperature))
                w = w / w.sum()
                if kl_div(grid, w) <= 0.8 * kl_far:
                    best = (f, w)
            if best is None:
                raise RuntimeError("no thermal factor satisfies P0")
            chosen["f"] = best[0]
            return best[1], far  # column 0 = closer ('cold' role), column 1 = farther ('hot')

        t0 = time.time()
        cv = curves(fn, temperature, ic_fn, 1.05 * t_max)
        out = classify_metric(cv.t, cv.d["kl"], cv.err["kl"], t_max)
        rho_c, rho_h = ic_fn(cv.grid)
        _, (c1_c, c1_h) = slow_mode_overlaps(cv.grid, [rho_c, rho_h])
        r = abs(c1_h) / abs(c1_c) if abs(c1_c) > 0 else np.inf
        passed = (
            r > 1.05 and out.label in ("NO_CROSSING", "CROSSING_NO_MARGIN") and out.last_gap < 0
        )
        ok &= passed
        print(
            f"  b={barrier} T={temperature} k={kappa} f={chosen['f']} ({time.time() - t0:.0f}s): "
            f"|c1_C|={abs(c1_c):.2e} |c1_H|={abs(c1_h):.3f} r={r:.2e}; P0(KL)={out.p0_ratio:.3g}; "
            f"label {out.label}; last gap {out.last_gap:.3e}  {'PASS' if passed else 'FAIL'}"
        )
    print(f"\nG5b: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
