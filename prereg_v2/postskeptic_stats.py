"""Read-only statistics requested by the third skeptic pass and the code review (post-hoc; CSV inputs only).

1. Hot-family sweep: W1 labels stratified by (mu, sigma); the CROSSING_NO_MARGIN runs and their |r - 1|.
2. Overlap of the two '57 of 72' sets in the v1 analysis.
3. Noise-floor model vs baselines, stratified by degenerate (max|gap| < 1e-6) / other points, with a
   shrinkage baseline; out-of-sample prediction at N = 50000 (theta scaled by 1/sqrt(10)).
4. Sensitivity of the headline discrepancy statistics to the four points decided by round-off (max|gap| < 1e-13).
"""

import numpy as np
import pandas as pd

sw = pd.read_csv("prereg_v2/summary_robustness.csv")
print("=== 1. W1 labels by (mu, sigma); counts over the 15 contexts (5 (b,T) x 3 kappa) ===")
print("  E = EFFECT, . = NO_TEST, c = CROSSING_NO_MARGIN")
short = {"EFFECT": "E", "NO_TEST": ".", "CROSSING_NO_MARGIN": "c", "NO_CROSSING": "0"}
for sg in sorted(sw["sigma"].unique()):
    cells = []
    for mu in sorted(sw["mu"].unique()):
        sub = sw[(sw["sigma"] == sg) & (sw["mu"] == mu)]
        cells.append(
            f"mu={mu:+.0f}: E{(sub['w1_label'] == 'EFFECT').sum():2d} .{(sub['w1_label'] == 'NO_TEST').sum():2d}"
        )
    print(f"  sigma={sg}: " + " | ".join(cells))
nm = sw[sw["kl_label"] == "CROSSING_NO_MARGIN"]
print(
    f"\nKL CROSSING_NO_MARGIN runs: {len(nm)}; with |r-1| <= 0.05: {int((np.abs(nm['r'] - 1) <= 0.05).sum())}; "
    f"r range {nm['r'].min():.3f}..{nm['r'].max():.3f}"
)
print("  by (sigma, |mu|):", nm.groupby(["sigma", nm["mu"].abs()]).size().to_dict())
marg = sw[np.abs(sw["r"] - 1) <= 0.05]
print(
    f"runs with |r-1| <= 0.05: {len(marg)}; of them KL label counts: {marg['kl_label'].value_counts().to_dict()}"
)

cmp = pd.read_csv("prereg_v2/v1_compare_points.csv")
print("\n=== 2. The two '57 of 72' sets ===")
a = set(cmp.index[cmp["pl_v1pct"] > 70])
b = set(cmp.index[cmp["pl_frac_below_theta"] > 0.5])
print(
    f"  crossed% > 70: {len(a)}; more than half of records below theta: {len(b)}; intersection {len(a & b)}"
)

print("\n=== 3. Model vs baselines, stratified ===")
deg = cmp["pl_max_abs_gap"] < 1e-6
f = cmp["pl_frac_below_theta"]
pred = {
    "noise-free v1 metric": cmp["pl_v1pct"],
    "grand mean of the map": pd.Series(cmp["v1_mean"].mean(), index=cmp.index),
    "50% at degenerate, else noise-free": pd.Series(
        np.where(deg, 50.0, cmp["pl_v1pct"]), index=cmp.index
    ),
    "shrinkage: (1-f)*noise-free + 50*f": (1 - f) * cmp["pl_v1pct"] + 50.0 * f,
    "noise-floor model (theta=0.02)": cmp["pl_noise_model"],
}
se = cmp["v1_sd"] / np.sqrt(10)
for name, p in pred.items():
    e = (p - cmp["v1_mean"]).abs()
    print(
        f"  {name:38s}: all {e.mean():5.1f} | degenerate (n={int(deg.sum())}) {e[deg].mean():5.1f} | "
        f"other (n={int((~deg).sum())}) {e[~deg].mean():5.1f}"
    )
print(
    f"  target noise floor (0.8 x mean SE): all {0.8 * se.mean():.1f} | degenerate {0.8 * se[deg].mean():.1f} | "
    f"other {0.8 * se[~deg].mean():.1f}"
)
e50 = (cmp["pl_noise_model_t0.0063"] - cmp["v1_mean"]).abs()
print(
    f"  (theta scaled to N=50000 is 0.0063: model error vs the N=5000 map {e50.mean():.1f}, not an out-of-sample test by itself)"
)

print("\n=== 4. Sensitivity to the points decided by round-off (max|gap| < 1e-13) ===")
rnd = cmp["pl_max_abs_gap"] < 1e-13
print(
    f"  points: {int(rnd.sum())}: {list(zip(cmp.loc[rnd, 'b'], cmp.loc[rnd, 'T'], cmp.loc[rnd, 'pl_v1pct'].round(1)))}"
)
alt = np.where(rnd, 50.0, cmp["pl_v1pct"])
print(
    f"  Pearson r: {np.corrcoef(cmp['pl_v1pct'], cmp['v1_mean'])[0, 1]:.2f} -> {np.corrcoef(alt, cmp['v1_mean'])[0, 1]:.2f}; "
    f"mean |diff|: {(cmp['pl_v1pct'] - cmp['v1_mean']).abs().mean():.1f} -> {np.abs(alt - cmp['v1_mean']).mean():.1f}; "
    f"points > 70%: {int((cmp['pl_v1pct'] > 70).sum())} -> {int((alt > 70).sum())} (v1: {int((cmp['v1_mean'] > 70).sum())})"
)
