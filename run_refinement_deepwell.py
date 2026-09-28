"""Deep-well grid-refinement check (PREREG_v2.md D2.10 part 2, rule fixed before this code).

Reruns the five points with the lowest late-time KL margin at dx in {0.005, 0.0025} and compares with the
main run (dx in {0.01, 0.005}). EFFECT in KL must persist; the difference of D at the last valid time between
dx = 0.0025 and dx = 0.005 is reported next to the main run's error estimate at the same time.
"""

import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.mpemba_analysis import D_FLOOR, MARGIN, classify_metric, curves
from src.mpemba_exact import cold_ic, gaussian_ic, lambda1_estimate, potential

POINTS = [
    (5.0, 0.05, 0.02),
    (5.0, 0.05, 0.05),
    (5.0, 0.05, 0.1),
    (3.0, 0.05, 0.05),
    (5.0, 0.1, 0.05),
]


def run(args):
    b, t, k = args
    t0 = time.time()

    def fn(x):
        return potential(x, b, k)

    with open("prereg_v2/summary_main.csv", encoding="utf-8") as fh:
        row = next(
            r
            for r in csv.DictReader(fh)
            if (float(r["b"]), float(r["T"]), float(r["kappa"])) == (b, t, k)
        )
    t_max = float(row["t_max"])
    assert abs(t_max / (8.0 / lambda1_estimate(fn, t)) - 1.0) < 1e-9  # same t_max as the main run
    cv = curves(
        fn, t, lambda g: (cold_ic(g), gaussian_ic(g, 3.0)), 1.05 * t_max, dxs=(0.005, 0.0025)
    )
    old = np.load(f"prereg_v2/out/b{b}_T{t}_k{k}.npz")
    res = {"cfg": (b, t, k), "seconds": time.time() - t0}
    for m in ("kl", "w1"):
        out = classify_metric(cv.t, cv.d[m], cv.err[m], t_max)
        sel = (cv.t > 0) & (cv.t <= t_max) & (cv.d[m][:, 0] >= D_FLOOR)
        i = int(np.flatnonzero(sel)[-1])  # last valid time
        gap_new = cv.d[m][i, 0] - cv.d[m][i, 1]
        ratio_new = gap_new / (MARGIN * (cv.err[m][i, 0] + cv.err[m][i, 1]))
        n = min(old["t"].size, cv.t.size)
        assert np.allclose(old["t"][:n], cv.t[:n], rtol=1e-9), "time grids differ"
        d_old, e_old = old[f"d_{m}"][:n], old[f"err_{m}"][:n]
        gap_old = d_old[i, 0] - d_old[i, 1]
        ratio_old = gap_old / (MARGIN * (e_old[i, 0] + e_old[i, 1]))
        diff = np.abs(cv.d[m][i] - d_old[i])  # dx = 0.0025 vs 0.005, cold and hot
        res[m] = {
            "label": out.label,
            "ratio_new": float(ratio_new),
            "ratio_old": float(ratio_old),
            "dx_diff_cold_hot": [float(diff[0]), float(diff[1])],
            "main_err_cold_hot": [float(e_old[i, 0]), float(e_old[i, 1])],
            "gap_old_new": [float(gap_old), float(gap_new)],
        }
    return res


def main() -> int:
    ok = True
    with ProcessPoolExecutor(max_workers=len(POINTS)) as pool:
        futures = [pool.submit(run, p) for p in POINTS]
        for fut in as_completed(futures):
            r = fut.result()
            print(f"\n{r['cfg']} ({r['seconds']:.0f}s)")
            for m in ("kl", "w1"):
                d = r[m]
                print(
                    f"  {m.upper()}: label {d['label']}; late ratio old (dx .01/.005) {d['ratio_old']:.2f} -> "
                    f"new (dx .005/.0025) {d['ratio_new']:.2f}; gap old {d['gap_old_new'][0]:.4e} new "
                    f"{d['gap_old_new'][1]:.4e}; |D(.0025) - D(.005)| cold/hot "
                    f"{d['dx_diff_cold_hot'][0]:.2e}/{d['dx_diff_cold_hot'][1]:.2e} vs main error estimate (dx + time Richardson terms) "
                    f"{d['main_err_cold_hot'][0]:.2e}/{d['main_err_cold_hot'][1]:.2e}"
                )
            ok &= r["kl"]["label"] == "EFFECT"
    print(f"\nKL EFFECT persists at dx = 0.0025 for all five points: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
