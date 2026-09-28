"""Robust margin statistics for the EFFECT labels (read-only; replaces the misleading 'minimum margin').

By the definition of EFFECT, at t* (the first record after the last bad point) gap/(5*err) is just above 1;
a minimum over a window that includes t* is therefore ~1 by construction. Meaningful statistics: the median
ratio over the window, the ratio at the last valid time, and the fraction of window records with ratio >= 2.
"""

import csv
import sys

import numpy as np

sys.path.insert(0, ".")
from src.mpemba_analysis import D_FLOOR, MARGIN

with open("prereg_v2/summary_main.csv", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
for m in ("kl", "w1"):
    med, late, frac2, at_t, cells = [], [], [], [], []
    for r in rows:
        if r[f"{m}_label"] != "EFFECT":
            continue
        b, T, k = float(r["b"]), float(r["T"]), float(r["kappa"])
        z = np.load(f"prereg_v2/out/b{b}_T{T}_k{k}.npz")
        t, d, e = z["t"], z[f"d_{m}"], z[f"err_{m}"]
        ts, tm = float(r[f"{m}_tstar"]), float(r["t_max"])
        sel = (t >= ts) & (t <= tm) & (d[:, 0] >= D_FLOOR)
        ratio = ((d[:, 0] - d[:, 1]) / (MARGIN * (e[:, 0] + e[:, 1])))[sel]
        med.append((float(np.median(ratio)), b, T, k))
        late.append((float(ratio[-1]), b, T, k))
        frac2.append(float((ratio >= 2).mean()))
        at_t.append(float(ratio[0]))
        cells.append((T / (8 * b)) ** 0.5 / 0.005)
    med.sort()
    late.sort()
    print(f"[{m}] EFFECT points: {len(med)}")
    print(
        f"  ratio at t* (definitional, just above 1): min {min(at_t):.3f}, median {np.median(at_t):.3f}"
    )
    print(
        f"  median ratio over the window: min {med[0][0]:.1f} at {med[0][1:]}, 5th lowest {med[4][0]:.1f}, "
        f"median over points {np.median([x[0] for x in med]):.3g}"
    )
    print(
        f"  ratio at the last valid time: min {late[0][0]:.1f} at {late[0][1:]}, "
        f"median over points {np.median([x[0] for x in late]):.3g}"
    )
    print(
        f"  fraction of window records with ratio >= 2: min {min(frac2):.3f}, median {np.median(frac2):.3f}"
    )
    print(f"  cells per well width sqrt(T/8b) at dx = 0.005: min {min(cells):.1f}")
    print("  five lowest late-time ratios:", [f"{x[0]:.2f} {x[1:]}" for x in late[:5]])
