"""Post-run inspection of the main run (read-only): odd rows, degeneracy of the energy comparison,
margin robustness of the KL/W1 EFFECT labels. Reads prereg_v2/summary_main.csv and out/*.npz."""

import csv
import sys

import numpy as np

sys.path.insert(0, ".")
from src.mpemba_analysis import D_FLOOR, MARGIN

with open("prereg_v2/summary_main.csv", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
for r in rows:
    for k in r:
        if k not in ("lam_src",) and not k.endswith("_label"):
            r[k] = float(r[k])

print(f"rows: {len(rows)}")
print("\n--- W1 NO_TEST rows (P0 failed) ---")
for r in rows:
    if r["w1_label"] == "NO_TEST":
        print(
            f"  b={r['b']} T={r['T']} kappa={r['kappa']}  W1 P0 ratio {r['w1_p0']:.3f}  KL P0 {r['kl_p0']:.3g}"
        )

print("\n--- energy (<U>) NO_CROSSING rows and kappa dependence ---")
for kappa in (0.0, 0.02, 0.05, 0.1):
    sub = [r for r in rows if r["kappa"] == kappa]
    counts = {}
    for r in sub:
        counts[r["u_label"]] = counts.get(r["u_label"], 0) + 1
    print(f"  kappa={kappa}: {counts}")
for r in rows:
    if r["u_label"] == "NO_CROSSING":
        print(f"  b={r['b']} T={r['T']} kappa={r['kappa']} u_p0={r['u_p0']:.3g}")

print("\n--- energy at kappa=0: is the cold distance at the round-off floor? ---")
for name in ("b2.0_T0.2_k0.0", "b1.0_T0.5_k0.0", "b0.1_T1.0_k0.0"):
    z = np.load(f"prereg_v2/out/{name}.npz")
    d = z["d_u"]
    print(
        f"  {name}: D_u,cold(0) = {d[0, 0]:.2e}, D_u,cold max = {d[:, 0].max():.2e}, "
        f"D_u,hot(0) = {d[0, 1]:.2e}, err max = {z['err_u'].max():.2e}"
    )

print(
    "\n--- margin over the window INCLUDING t* (about 1 by construction; see margin_robustness.py for the informative statistics) ---"
)
worst = {"kl": (np.inf, None), "w1": (np.inf, None)}
tstar = {"kl": [], "w1": []}
for r in rows:
    z = np.load(f"prereg_v2/out/b{r['b']}_T{r['T']}_k{r['kappa']}.npz")
    t = z["t"]
    for m in ("kl", "w1"):
        if r[f"{m}_label"] != "EFFECT":
            continue
        d, e = z[f"d_{m}"], z[f"err_{m}"]
        sel = (t >= r[f"{m}_tstar"]) & (t <= r["t_max"]) & (d[:, 0] >= D_FLOOR)
        ratio = ((d[:, 0] - d[:, 1]) / (MARGIN * (e[:, 0] + e[:, 1])))[sel].min()
        tstar[m].append(r[f"{m}_tstar"] / r["t_max"])
        if ratio < worst[m][0]:
            worst[m] = (float(ratio), (r["b"], r["T"], r["kappa"]))
for m in ("kl", "w1"):
    ts = np.array(tstar[m])
    print(
        f"  {m}: min margin ratio {worst[m][0]:.3g} at {worst[m][1]}; t*/t_max range "
        f"{ts.min():.2e} .. {ts.max():.2e}, median {np.median(ts):.2e}"
    )
