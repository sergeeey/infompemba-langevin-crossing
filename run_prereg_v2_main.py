"""Main run for PREREG_v2 (D2.1): 288 configurations, per-point classification, K-table inputs.

Deterministic (no seeds). Curves are saved to prereg_v2/out/*.npz (git-ignored; manifest with
sha256 is committed); the per-point summary goes to prereg_v2/summary_main.csv (committed).
`--smoke` runs a few configurations and reports timing and crashes only (no outcomes printed).
"""

import csv
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.mpemba_analysis import METRICS, classify_metric, curves, slow_mode_overlaps
from src.mpemba_exact import (
    cold_ic,
    gaussian_ic,
    lambda1_eig,
    lambda1_estimate,
    make_grid,
    potential,
)

BARRIERS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
TEMPERATURES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
KAPPAS = [0.0, 0.02, 0.05, 0.1]
R_MAX_BT = 12.0
R_MARGINAL = (0.95, 1.05)
OUT_DIR = "prereg_v2/out"
SUMMARY = "prereg_v2/summary_main.csv"
MANIFEST = "prereg_v2/out_manifest.csv"


def _hot_gauss3(grid):
    return cold_ic(grid), gaussian_ic(grid, 3.0)


def run_config(args):
    barrier, temperature, kappa = args
    t0 = time.time()

    def fn(x):
        return potential(x, barrier, kappa)

    half = 4.0 if barrier >= 0.5 else 6.0
    if barrier / temperature <= R_MAX_BT:
        lam, src = lambda1_eig(make_grid(fn, temperature, half, 0.01)), "eig"
    else:
        lam, src = lambda1_estimate(fn, temperature), "quad"
    t_max = 8.0 / lam
    cv = curves(fn, temperature, _hot_gauss3, 1.05 * t_max)
    row = {
        "b": barrier,
        "T": temperature,
        "kappa": kappa,
        "bt": barrier / temperature,
        "lam1": lam,
        "lam_src": src,
        "t_max": t_max,
    }
    for m in METRICS:
        o = classify_metric(cv.t, cv.d[m], cv.err[m], t_max)
        row[f"{m}_label"] = o.label
        row[f"{m}_tstar"] = o.t_star
        row[f"{m}_p0"] = o.p0_ratio
        row[f"{m}_lastgap"] = o.last_gap
    row.update(c1_cold=np.nan, c1_hot=np.nan, r=np.nan)
    if barrier / temperature <= R_MAX_BT:
        try:
            _, (c_c, c_h) = slow_mode_overlaps(cv.grid, list(_hot_gauss3(cv.grid)))
            row.update(c1_cold=c_c, c1_hot=c_h, r=abs(c_h) / abs(c_c))
        except ValueError:
            pass
    name = f"{OUT_DIR}/b{barrier}_T{temperature}_k{kappa}.npz"
    arrays = {"t": cv.t}
    for m in METRICS:
        arrays[f"d_{m}"] = cv.d[m]
        arrays[f"err_{m}"] = cv.err[m]
    np.savez_compressed(name, **arrays)
    row["seconds"] = time.time() - t0
    return row


def _write_csv(rows):
    keys = list(rows[0].keys())
    with open(SUMMARY, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "sha256"])
        for fname in sorted(os.listdir(OUT_DIR)):
            with open(os.path.join(OUT_DIR, fname), "rb") as f:
                writer.writerow([fname, hashlib.sha256(f.read()).hexdigest()])


def _cell(rows, b, t, kappa, key):
    for r in rows:
        if r["b"] == b and r["T"] == t and r["kappa"] == kappa:
            return r[key]
    return "?"


def summarize(rows):
    short = {"NO_TEST": ".", "NO_CROSSING": "0", "CROSSING_NO_MARGIN": "c", "EFFECT": "E"}
    print("\n=== label counts per metric (all kappa) ===")
    for m in METRICS:
        counts = {}
        for r in rows:
            counts[r[f"{m}_label"]] = counts.get(r[f"{m}_label"], 0) + 1
        print(f"  {m:4s}: {counts}")
    for m in ("kl", "w1"):
        print(
            f"\n=== kappa = 0 map, {m.upper()} (rows b, cols T; E effect, c crossing w/o margin, "
            "0 none, . no-test) ==="
        )
        print("        " + "  ".join(f"{t:>5}" for t in TEMPERATURES))
        for b in BARRIERS:
            cells = "  ".join(
                f"{short[_cell(rows, b, t, 0.0, m + '_label')]:>5}" for t in TEMPERATURES
            )
            print(f"  b={b:<4} {cells}")

    k0 = [r for r in rows if r["kappa"] == 0.0]
    p0_both = [r for r in k0 if r["kl_label"] != "NO_TEST" and r["w1_label"] != "NO_TEST"]
    eff_both = [r for r in p0_both if r["kl_label"] == "EFFECT" and r["w1_label"] == "EFFECT"]
    kl_only = [r for r in k0 if r["kl_label"] == "EFFECT" and r["w1_label"] != "EFFECT"]
    print(
        f"\n=== K1/K2 (kappa = 0) ===\n  points with P0 in both metrics: {len(p0_both)} of {len(k0)}"
    )
    print(f"  EFFECT_BOTH: {len(eff_both)} (K1 threshold 10)   KL_ONLY: {len(kl_only)}")
    if len(p0_both) < 36:  # section 3a: K1 does not apply, the design is inadequate
        k1 = "n/a (fewer than 36 of 72 points satisfy P0: design inadequate, not REJECT)"
    else:
        k1 = "REJECT" if len(eff_both) < 10 else "pass"
    all_eff_both = [r for r in rows if r["kl_label"] == "EFFECT" and r["w1_label"] == "EFFECT"]
    all_kl_only = [r for r in rows if r["kl_label"] == "EFFECT" and r["w1_label"] != "EFFECT"]
    k2 = "REJECT (metric artifact)" if (not all_eff_both and all_kl_only) else "pass"
    print(
        f"  K1: {k1}   K2 (whole grid, {len(all_eff_both)} EFFECT_BOTH / {len(all_kl_only)} KL_ONLY): {k2}"
    )

    print("\n=== R (Lu-Raz overlap) vs direct late-time KL gap sign, b/T <= 12 ===")
    dropped = [r for r in rows if r["bt"] <= R_MAX_BT and not np.isfinite(r["r"])]
    print(f"  rows with b/T <= {R_MAX_BT:g} where R was undefined (excluded): {len(dropped)}")
    if dropped:
        print("  G5 ANOMALY: R undefined at moderate b/T; inspect before trusting the agreement")
    verdicts = {}
    for label, sel in (
        ("kappa != 0 (G5)", lambda r: r["kappa"] != 0.0),
        ("all kappa (K3)", lambda r: True),
    ):
        pool = [
            r
            for r in rows
            if sel(r)
            and np.isfinite(r["r"])
            and not (R_MARGINAL[0] <= r["r"] <= R_MARGINAL[1])
            and r["kl_label"] != "NO_TEST"
            and np.isfinite(r["kl_lastgap"])
        ]
        bad = [r for r in pool if (r["kl_lastgap"] > 0) != (r["r"] < 1.0)]
        frac = 1.0 - len(bad) / len(pool) if pool else float("nan")
        verdicts[label] = (frac, len(bad), len(pool))
        print(f"  {label}: eligible {len(pool)}, disagree {len(bad)}, agreement {frac:.3f}")
        for r in bad[:10]:
            print(
                f"    disagree: b={r['b']} T={r['T']} k={r['kappa']} r={r['r']:.3f} "
                f"lastgap={r['kl_lastgap']:.3e} KL {r['kl_label']}"
            )
    g5 = verdicts["kappa != 0 (G5)"]
    k3 = verdicts["all kappa (K3)"]
    g5_ok = g5[0] >= 0.95 and not dropped
    print(f"  G5 gate (agreement >= 0.95 and no undefined R): {'PASS' if g5_ok else 'FAIL -> K0'}")
    print(
        f"  K3 comparison: agreement {k3[0]:.3f}, disagreements {k3[1]} "
        f"({'K3: known criterion' if k1 == 'pass' and k2 == 'pass' and k3[0] >= 0.95 else 'see D2.5'})"
    )

    print("\n=== observables (descriptive; not used by any K decision) ===")
    print(
        "  DEGENERATE = P0 ratio D_hot(0)/D_cold(0) > 1e12, i.e. the cold distance is at "
        "round-off and P0 holds vacuously"
    )
    for kappa in KAPPAS:
        sub = [r for r in rows if r["kappa"] == kappa]
        for m in ("u", "var"):
            counts = {}
            for r in sub:
                key = "DEGENERATE" if r[f"{m}_p0"] > 1e12 else r[f"{m}_label"]
                counts[key] = counts.get(key, 0) + 1
            print(f"  kappa={kappa:<5} {m:4s}: {counts}")


def _read_summary():
    with open(SUMMARY, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in r:
            if k != "lam_src" and not k.endswith("_label"):
                r[k] = float(r[k])
    return rows


def main() -> int:
    if "--summarize" in sys.argv:  # re-run the K-table logic on the saved CSV (no recomputation)
        summarize(_read_summary())
        return 0
    smoke = "--smoke" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [(b, t, k) for k in KAPPAS for b in BARRIERS for t in TEMPERATURES]
    if smoke:
        tasks = [(2.0, 0.2, 0.0), (5.0, 0.05, 0.05), (0.1, 1.0, 0.1)]
    tasks.sort(key=lambda a: -a[0] / a[1])
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(tasks)} configurations, {workers} workers")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_config, a): a for a in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done, {time.time() - t0:.0f}s elapsed", flush=True)
    rows.sort(key=lambda r: (r["kappa"], r["b"], r["T"]))
    if smoke:
        print(f"smoke ok; per-config seconds: {[round(r['seconds']) for r in rows]}")
        return 0
    _write_csv(rows)
    summarize(rows)
    print(f"\nelapsed {time.time() - t0:.0f}s; wrote {SUMMARY} and {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
