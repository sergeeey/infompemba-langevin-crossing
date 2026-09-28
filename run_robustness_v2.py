"""Post-hoc design-sensitivity sweep (PREREG_v2.md D2.8), defined before this code was written.

Hot = Gauss(mu, sigma) for mu in MUS, sigma in SIGMAS; cold = right-well restricted Boltzmann; 15
(b, T, kappa) contexts; D2.1 protocol unchanged. Reports P0/labels, the overlap ratio
r = |c1_hot| / |c1_cold| and the agreement of the spectral criterion R with the direct late-time KL
gap sign on ALL runs with P0 in KL, without the r in [0.95, 1.05] exclusion. Descriptive only.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.mpemba_analysis import classify_metric, curves, slow_mode_overlaps
from src.mpemba_exact import cold_ic, gaussian_ic, lambda1_eig, make_grid, potential

BT = [(2.0, 0.2), (1.0, 0.1), (3.0, 0.3), (5.0, 0.5), (0.5, 0.05)]
KAPPAS = [0.0, 0.05, 0.1]
MUS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SIGMAS = [1.0, 1.5, 3.0]


def run_one(args):
    barrier, temperature, kappa, mu, sigma = args

    def fn(x):
        return potential(x, barrier, kappa)

    half = 4.0 if barrier >= 0.5 else 6.0
    t_max = 8.0 / lambda1_eig(make_grid(fn, temperature, half, 0.01))

    def ic(grid):
        return cold_ic(grid), gaussian_ic(grid, sigma, mu)

    cv = curves(fn, temperature, ic, 1.05 * t_max)
    kl = classify_metric(cv.t, cv.d["kl"], cv.err["kl"], t_max)
    w1 = classify_metric(cv.t, cv.d["w1"], cv.err["w1"], t_max)
    rho_c, rho_h = ic(cv.grid)
    _, (c_c, c_h) = slow_mode_overlaps(cv.grid, [rho_c, rho_h])
    return {
        "b": barrier,
        "T": temperature,
        "kappa": kappa,
        "mu": mu,
        "sigma": sigma,
        "kl_label": kl.label,
        "w1_label": w1.label,
        "kl_p0": kl.p0_ratio,
        "w1_p0": w1.p0_ratio,
        "kl_lastgap": kl.last_gap,
        "r": abs(c_h) / abs(c_c) if abs(c_c) > 0 else np.inf,
    }


def main() -> int:
    tasks = [(b, t, k, mu, s) for (b, t) in BT for k in KAPPAS for mu in MUS for s in SIGMAS]
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(tasks)} runs, {workers} workers")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, a) for a in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done, {time.time() - t0:.0f}s", flush=True)

    import csv

    rows.sort(key=lambda r: (r["b"], r["T"], r["kappa"], r["mu"], r["sigma"]))
    with open("prereg_v2/summary_robustness.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nlabels by hot width sigma (KL EFFECT / W1 EFFECT / W1 NO_TEST) out of 75 runs each:")
    for sg in SIGMAS:
        sub = [r for r in rows if r["sigma"] == sg]
        print(
            f"  sigma={sg}: KL EFFECT {sum(r['kl_label'] == 'EFFECT' for r in sub)}, "
            f"W1 EFFECT {sum(r['w1_label'] == 'EFFECT' for r in sub)}, "
            f"W1 NO_TEST {sum(r['w1_label'] == 'NO_TEST' for r in sub)}, "
            f"KL no-margin {sum(r['kl_label'] == 'CROSSING_NO_MARGIN' for r in sub)}"
        )
    n = len(rows)
    counts = {}
    for r in rows:
        key = (r["kl_label"], r["w1_label"])
        counts[key] = counts.get(key, 0) + 1
    print(f"\nlabels (KL, W1) over {n} runs: {counts}")

    p0 = [r for r in rows if r["kl_label"] != "NO_TEST" and np.isfinite(r["kl_lastgap"])]
    print(f"runs with P0 in KL and a valid late gap: {len(p0)} of {n}")
    print("\nR (sign of 1 - r) versus direct late-time KL gap sign, no exclusion:")
    agree = [r for r in p0 if (r["kl_lastgap"] > 0) == (r["r"] < 1.0)]
    marginal = [r for r in p0 if abs(r["r"] - 1.0) <= 0.05]
    flagged = [
        r for r in p0 if abs(r["r"] - 1.0) > 0.05 and (r["kl_lastgap"] > 0) != (r["r"] < 1.0)
    ]
    print(
        f"  agree {len(agree)} / {len(p0)}; marginal |r-1|<=0.05: {len(marginal)}; "
        f"FLAGGED disagreements (|r-1|>0.05): {len(flagged)}"
    )
    print(
        f"  runs with r < 1: {sum(r['r'] < 1 for r in p0)}, r > 1: {sum(r['r'] > 1 for r in p0)}; "
        f"r in (0.5, 2): {sum(0.5 < r['r'] < 2 for r in p0)}"
    )
    for r in flagged[:12]:
        print(
            f"    FLAG b={r['b']} T={r['T']} k={r['kappa']} mu={r['mu']} sigma={r['sigma']} "
            f"r={r['r']:.3f} lastgap={r['kl_lastgap']:.3e} KL {r['kl_label']}"
        )

    print(
        "\nKL label by kappa (rows) and hot family (mu, sigma): E effect, . no-test, 0 none, c no-margin"
    )
    short = {"EFFECT": "E", "NO_TEST": ".", "NO_CROSSING": "0", "CROSSING_NO_MARGIN": "c"}
    for kappa in KAPPAS:
        print(f"  kappa={kappa}")
        print("        " + " ".join(f"s={s:<4}" for s in SIGMAS))
        for mu in MUS:
            cells = []
            for s in SIGMAS:
                sub = [r for r in rows if r["kappa"] == kappa and r["mu"] == mu and r["sigma"] == s]
                cells.append("".join(sorted(short[r["kl_label"]] for r in sub)))
            print(f"    mu={mu:<4} " + " ".join(f"{c:<6}" for c in cells))
    print(f"\nelapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
