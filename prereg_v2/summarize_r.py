"""Read-only: overlap ratios r = |c1_hot| / |c1_cold| and lambda1 sources from summary_main.csv."""

import csv

import numpy as np

with open("prereg_v2/summary_main.csv", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
for r in rows:
    for k in r:
        if k != "lam_src" and not k.endswith("_label"):
            r[k] = float(r[k])
for kappa in (0.0, 0.02, 0.05, 0.1):
    sub = [r for r in rows if r["kappa"] == kappa and np.isfinite(r["r"])]
    rr = np.array([r["r"] for r in sub])
    c_h = np.array([abs(r["c1_hot"]) for r in sub])
    c_c = np.array([abs(r["c1_cold"]) for r in sub])
    print(
        f"kappa={kappa}: n={len(sub)}  r min/median/max = {rr.min():.2e} / {np.median(rr):.2e} / {rr.max():.2e}  "
        f"|c1_hot| max {c_h.max():.2e}  |c1_cold| min/max {c_c.min():.3f} / {c_c.max():.3f}"
    )
print(
    f"lambda1 from eigvalsh: {sum(r['lam_src'] == 'eig' for r in rows)}, from quadrature: "
    f"{sum(r['lam_src'] == 'quad' for r in rows)}"
)
print(
    f"b/T range: {min(r['bt'] for r in rows):.2f} .. {max(r['bt'] for r in rows):.0f}; "
    f"t_max range {min(r['t_max'] for r in rows):.2g} .. {max(r['t_max'] for r in rows):.2g}"
)
kl_p0 = np.array([r["kl_p0"] for r in rows])
w1_p0 = np.array([r["w1_p0"] for r in rows])
print(
    f"KL P0 ratio min {kl_p0.min():.3g}; W1 P0 ratio min {w1_p0.min():.3f}, max {w1_p0.max():.3f}"
)
print(
    f"seconds per config: median {np.median([r['seconds'] for r in rows]):.0f}, max {max(r['seconds'] for r in rows):.0f}"
)
