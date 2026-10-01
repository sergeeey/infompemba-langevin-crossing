"""Run the input guards (src/mpemba_guards.py) over the stored main-run curves (post-hoc, 2026-10-01).

For every one of the 288 configurations and every metric (kl, w1, u, var):
  - the guard's problems for the stored (t, d, err) are collected;
  - for kl and w1 the guarded label is compared with the label in summary_main.csv.
Prints counts; exit code 0 only if the guard changes no kl/w1 label and flags no kl/w1 curve.
Needs prereg_v2/out/*.npz (see REPRODUCE.md section 3).
"""

import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.mpemba_guards import INVALID_INPUT, check_curve_inputs, classify_metric_guarded

M = pd.read_csv("prereg_v2/summary_main.csv")
flagged = {m: Counter() for m in ("kl", "w1", "u", "var")}
changed = []
ratios = {"kl": [], "w1": [], "var": [], "u (kappa != 0)": [], "u (kappa = 0)": []}
for _, r in M.iterrows():
    z = np.load(f"prereg_v2/out/b{r.b}_T{r['T']}_k{r.kappa}.npz")
    for m, cnt in flagged.items():
        cnt.update(check_curve_inputs(z["t"], z[f"d_{m}"], z[f"err_{m}"]))
    for m in ("kl", "w1", "var", "u"):
        d0 = z[f"d_{m}"][0]
        key = m if m != "u" else ("u (kappa = 0)" if r.kappa == 0 else "u (kappa != 0)")
        if d0[0] > 0:
            ratios[key].append(d0[1] / d0[0])
    for m in ("kl", "w1"):
        out = classify_metric_guarded(z["t"], z[f"d_{m}"], z[f"err_{m}"], r.t_max)
        if out.label != r[f"{m}_label"]:
            changed.append((r.b, r["T"], r.kappa, m, r[f"{m}_label"], out.label))

for m, cnt in flagged.items():
    print(f"{m}: problems found: {sum(cnt.values())}")
    for p, n in cnt.items():
        print(f"    {n:4d} x {p}")
for key, vals in ratios.items():
    if vals:
        print(
            f"hot/cold distance ratio at t = 0, {key}: n = {len(vals)} (cold distance > 0), "
            f"min {min(vals):.4g}, max {max(vals):.4g}"
        )
print(
    f"kl/w1 labels changed by the guard: {len(changed)} (INVALID_INPUT label = {INVALID_INPUT!r})"
)
for row in changed[:10]:
    print("   ", row)
kl_w1_flagged = sum(flagged["kl"].values()) + sum(flagged["w1"].values())
sys.exit(0 if not changed and kl_w1_flagged == 0 else 1)
