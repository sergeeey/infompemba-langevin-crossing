"""Test of the degeneracy explanation for the non-reproduced v1 map (post-hoc hypothesis; prediction fixed here
BEFORE the script was run, after the attribution test D2.9a returned 'unexplained' and showed bimodal values).

Hypothesis: with the equilibrium estimated from the tails of the two compared trajectories, the distance gap
is identically zero whenever both trajectories are stationary in the last-10% window (cold stuck near 0, hot at
0.5, eq = 0.25 for both), so the v1 metric returns the sign of noise or of round-off there.
Predictions (exact-solver max|gap| < 1e-6 marks a degenerate point):
  P1. degenerate points: median v1-parquet seed-to-seed sd of crossed_pct > 25 pp; non-degenerate: < 5 pp.
  P2. on non-degenerate points |exact-solver metric - parquet mean| is small (< 5 pp on average); on degenerate
      points it is large.
  P3. on degenerate points the individual parquet runs are bimodal (crossed_pct < 10 or > 90) in most runs.
Read-only.
"""

import numpy as np
import pandas as pd

cmp = pd.read_csv("prereg_v2/v1_compare_points.csv")
v1 = pd.read_parquet("results/phase_diagram.parquet")
cmp["degenerate"] = cmp["pl_max_abs_gap"] < 1e-6
print(f"degenerate points (max|gap| < 1e-6): {int(cmp['degenerate'].sum())} of {len(cmp)}")
print(
    "max|gap| range on degenerate points: "
    f"{cmp.loc[cmp['degenerate'], 'pl_max_abs_gap'].max():.1e} max; on the others: "
    f"{cmp.loc[~cmp['degenerate'], 'pl_max_abs_gap'].min():.1e} min"
)
for name, grp in cmp.groupby("degenerate"):
    label = "DEGENERATE    " if name else "non-degenerate"
    keys = list(zip(grp["b"], grp["T"], strict=True))
    runs = v1[[(b, t) in set(keys) for b, t in zip(v1["barrier"], v1["T"], strict=True)]]
    extreme = float(((runs["crossed_pct"] < 10) | (runs["crossed_pct"] > 90)).mean())
    print(
        f"{label}: n={len(grp):2d}  median parquet sd = {grp['v1_sd'].median():5.1f} pp  "
        f"mean |exact - parquet mean| = {grp['abs_diff'].mean():5.1f} pp  "
        f"share of individual parquet runs at an extreme (<10 or >90): {extreme:.2f}"
    )
deg, non = cmp[cmp["degenerate"]], cmp[~cmp["degenerate"]]
if len(deg) and len(non):
    p1 = deg["v1_sd"].median() > 25 and non["v1_sd"].median() < 5
    p2 = non["abs_diff"].mean() < 5 and deg["abs_diff"].mean() > non["abs_diff"].mean()
    print(f"\nP1 (sd separation): {p1};  P2 (reproduction only off the degenerate set): {p2}")

# ---- POST-HOC (added after P2 failed; the threshold below is chosen after seeing that failure) ----
print(
    "\nPOST-HOC: sensitivity to the degeneracy threshold (the sampling noise of p_left at N=5000 is ~0.007)"
)
for thr in (1e-6, 1e-3, 1e-2, 3e-2, 1e-1):
    d = cmp[cmp["pl_max_abs_gap"] < thr]
    o = cmp[cmp["pl_max_abs_gap"] >= thr]
    if len(d) and len(o):
        print(
            f"  max|gap| < {thr:g}: n={len(d):2d}, mean |exact - parquet| {d['abs_diff'].mean():5.1f}, "
            f"median sd {d['v1_sd'].median():5.1f} | rest: n={len(o):2d}, mean |exact - parquet| "
            f"{o['abs_diff'].mean():5.1f}, median sd {o['v1_sd'].median():5.1f}"
        )

# ---- POST-HOC noise-floor model (theta = 0.02 fixed from sampling theory, not fitted) ----
err_exact = (cmp["pl_v1pct"] - cmp["v1_mean"]).abs()
err_model = (cmp["pl_noise_model"] - cmp["v1_mean"]).abs()
print(
    "\nPOST-HOC noise-floor model: records with |gap| <= 0.02 are coin flips (50%), the rest deterministic"
)
print(
    f"  mean |exact-solver metric - parquet mean| = {err_exact.mean():.1f} pp (Pearson r "
    f"{np.corrcoef(cmp['pl_v1pct'], cmp['v1_mean'])[0, 1]:.2f})"
)
print(
    f"  mean |noise model      - parquet mean| = {err_model.mean():.1f} pp (Pearson r "
    f"{np.corrcoef(cmp['pl_noise_model'], cmp['v1_mean'])[0, 1]:.2f}); within 5 pp: "
    f"{int((err_model <= 5).sum())}/{len(cmp)}, within 10 pp: {int((err_model <= 10).sum())}/{len(cmp)}"
)

print(
    "\nPOST-HOC robustness of the noise model to theta (0.02 was fixed from sampling theory first):"
)
for th in (0.005, 0.01, 0.02, 0.03, 0.05):
    col = "pl_noise_model" if th == 0.02 else f"pl_noise_model_t{th}"
    e = (cmp[col] - cmp["v1_mean"]).abs()
    print(
        f"  theta={th:<5}: mean |model - parquet| {e.mean():5.1f} pp, Pearson r "
        f"{np.corrcoef(cmp[col], cmp['v1_mean'])[0, 1]:.2f}, within 5 pp {int((e <= 5).sum())}/72"
    )
