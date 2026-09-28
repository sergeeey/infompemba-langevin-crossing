"""EXPLORATORY (outside the K criteria): reproduce the v1 phase diagram with the exact solver.

v1 (run_phase_diagram.py): cold ~ N((1,0), 0.05^2 I), hot ~ N(0, 2^2 I), overdamped Langevin
dt = 0.005, 30000 steps, p_left recorded every 50 steps, O_eq = mean of the last 10% of both
trajectories, crossed_pct = 100 * mean(dist_hot < dist_cold) with NO check that the hot state
started farther. Here the same initial conditions are evolved with the exact 1D solver (the x
marginal decouples from y), the same 600 recorded times (t = 0.25 .. 150) and the same v1 metric are
applied, then compared with results/phase_diagram.parquet. Then the same trajectories are scored
with the analytic equilibrium and the P0 precondition. Energy and <x^2> get the same treatment
(y-part of the energy analytically). Read-only.
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.mpemba_exact import _implicit_step, make_grid, potential

BARRIERS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
TEMPS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
N_REC, DT_REC, SUBSTEPS = 600, 0.25, 10
THETA_SWEEP = [0.005, 0.0063, 0.01, 0.03, 0.05]
NOISE_THETA = (
    0.02  # ~2 sqrt(2) * sqrt(0.25 / 5000): sampling noise of the difference of two p_left estimates
)


def gauss(x, mu, s):
    w = np.exp(-0.5 * ((x - mu) / s) ** 2)
    return w / w.sum()


def v1_metric(traj_cold, traj_hot):
    tail = max(1, len(traj_cold) // 10)
    eq = (traj_cold[-tail:].mean() + traj_hot[-tail:].mean()) / 2.0
    gap = np.abs(traj_hot - eq) - np.abs(traj_cold - eq)
    idx = np.flatnonzero(gap < 0)
    return 100.0 * (gap < 0).mean(), (int(idx[0]) if idx.size else -1), eq


def p0_and_cross(traj_cold, traj_hot, start_cold, start_hot, eq):
    """P0 with the pre-registered 1.25 ratio and any-time crossing under the analytic equilibrium."""
    d_c0, d_h0 = abs(start_cold - eq), abs(start_hot - eq)
    p0 = d_h0 >= 1.25 * d_c0 and d_h0 > 1e-12
    gap = np.abs(traj_hot - eq) - np.abs(traj_cold - eq)
    return p0, bool((gap < 0).any())


def p_left(grid, r):
    """Fraction with x < 0 (half weight on the node at x = 0)."""
    return r[grid.x < 0].sum(axis=0) + 0.5 * r[grid.x.size // 2]


def obs(grid, r, y_var):
    """Total energy (x part on the grid, y part analytic) and <x^2>."""
    e_x = (r * grid.energy[:, None]).sum(axis=0)
    x2 = (r * (grid.x**2)[:, None]).sum(axis=0)
    return e_x + 0.3 * y_var, x2


rows = []
for b in BARRIERS:
    for T in TEMPS:
        grid = make_grid(lambda x, b=b: potential(x, b, 0.0), T, 15.0, 0.01)
        rho = np.stack([gauss(grid.x, 1.0, 0.05), gauss(grid.x, 0.0, 2.0)], axis=1)
        y_eq = T / 0.6
        y_var0 = np.array([0.05**2, 2.0**2])
        pl0 = p_left(grid, rho)
        e0, x20 = obs(grid, rho, y_var0)
        pl, en, x2 = [pl0], [e0], [x20]
        t = 0.0
        h = DT_REC / SUBSTEPS
        for _ in range(N_REC):
            for _ in range(SUBSTEPS):
                _implicit_step(grid.up, grid.dn, h, rho)
            t += DT_REC
            y_var = y_eq + (y_var0 - y_eq) * np.exp(-1.2 * t)
            e, x2v = obs(grid, rho, y_var)
            pl.append(p_left(grid, rho))
            en.append(e)
            x2.append(x2v)
        pl, en, x2 = np.array(pl), np.array(en), np.array(x2)
        eq_e, eq_x2 = grid.u_eq + 0.3 * y_eq, float(np.sum(grid.pi * grid.x**2))
        row = {"b": b, "T": T}
        # ablation: analytic equilibrium (0.5) but still no P0 check
        row["pl_analytic_eq_pct"] = 100.0 * float(
            (np.abs(pl[1:, 1] - 0.5) < np.abs(pl[1:, 0] - 0.5)).mean()
        )
        # degeneracy probe: v1's tail-estimated equilibrium makes the two distances equal whenever
        # both trajectories are stationary in the last 10% window, i.e. gap == 0 identically
        _c, _h = pl[1:, 0], pl[1:, 1]
        _eq = (_c[-60:].mean() + _h[-60:].mean()) / 2.0
        row["pl_max_abs_gap"] = float(np.abs(np.abs(_h - _eq) - np.abs(_c - _eq)).max())
        # POST-HOC noise-floor model of the v1 metric: records with |gap| <= theta are coin flips at N = 5000
        _gap = np.abs(_h - _eq) - np.abs(_c - _eq)
        row["pl_frac_below_theta"] = float((np.abs(_gap) <= NOISE_THETA).mean())
        row["pl_noise_model"] = 100.0 * float(
            (_gap < -NOISE_THETA).mean() + 0.5 * (np.abs(_gap) <= NOISE_THETA).mean()
        )
        for _th in THETA_SWEEP:  # robustness of the post-hoc model to theta
            row[f"pl_noise_model_t{_th}"] = 100.0 * float(
                (_gap < -_th).mean() + 0.5 * (np.abs(_gap) <= _th).mean()
            )
        for name, series, eq in (("pl", pl, 0.5), ("en", en, eq_e), ("x2", x2, eq_x2)):
            c, h_ = series[1:, 0], series[1:, 1]  # v1 records from step 50 on (t >= 0.25)
            v1_pct, first, eq_tail = v1_metric(c, h_)
            p0, crossed = p0_and_cross(c, h_, series[0, 0], series[0, 1], eq)
            row.update(
                {
                    f"{name}_v1pct": v1_pct,
                    f"{name}_first": first,
                    f"{name}_eq_tail": eq_tail,
                    f"{name}_eq": eq,
                    f"{name}_p0": p0,
                    f"{name}_cross_given_p0": crossed if p0 else None,
                }
            )
        rows.append(row)
df = pd.DataFrame(rows)

v1 = pd.read_parquet("results/phase_diagram.parquet")
agg = (
    v1.groupby(["barrier", "T"])
    .agg(v1_mean=("crossed_pct", "mean"), v1_sd=("crossed_pct", "std"))
    .reset_index()
)
m = df.merge(agg, left_on=["b", "T"], right_on=["barrier", "T"])
diff = (m["pl_v1pct"] - m["v1_mean"]).abs()
m.assign(abs_diff=diff)[
    ["b", "T", "pl_v1pct", "v1_mean", "v1_sd", "abs_diff", "pl_max_abs_gap", "pl_noise_model"]
    + [f"pl_noise_model_t{t}" for t in THETA_SWEEP]
    + ["pl_frac_below_theta"]
].to_csv("prereg_v2/v1_compare_points.csv", index=False)
print(f"grid points compared: {len(m)}")
print("p_left, v1 metric on EXACT trajectories vs v1's own 10-seed means:")
print(
    f"  Pearson r = {np.corrcoef(m['pl_v1pct'], m['v1_mean'])[0, 1]:.4f}; mean |diff| = "
    f"{diff.mean():.2f} pct points; max |diff| = {diff.max():.2f}; points within 5 pp: "
    f"{int((diff <= 5).sum())}/{len(m)}"
)
strong_exact = int((m["pl_v1pct"] > 70).sum())
strong_v1 = int((m["v1_mean"] > 70).sum())
print(
    f"  points with mean crossed_pct > 70: exact-solver {strong_exact}, v1 parquet {strong_v1}; "
    f"peak exact {m['pl_v1pct'].max():.1f} at "
    f"{m.loc[m['pl_v1pct'].idxmax(), ['b', 'T']].tolist()}; peak v1 {m['v1_mean'].max():.1f}"
)
print(
    f"  exact-solver first crossing at record 0 (hot 'closer' from the start): "
    f"{int((df['pl_first'] == 0).sum())}/{len(df)} points"
)
print(
    f"  eq_tail (v1 estimate) vs 0.5: mean |dev| {np.abs(df['pl_eq_tail'] - 0.5).mean():.3f}, "
    f"max {np.abs(df['pl_eq_tail'] - 0.5).max():.3f}"
)

print("\nSame trajectories scored with the analytic equilibrium and the P0 rule (ratio >= 1.25):")
for name, label in (("pl", "p_left"), ("en", "energy"), ("x2", "<x^2>")):
    n_p0 = int(df[f"{name}_p0"].sum())
    n_cross = int(df[f"{name}_cross_given_p0"].eq(True).sum())
    print(
        f"  {label:7s}: P0 satisfied at {n_p0}/72 points; any-time crossing among them: {n_cross}"
    )
print("\nAblation on p_left (72 points): which defect is sufficient?")
n_a = int((df["pl_v1pct"] > 70).sum())
n_b = int((df["pl_analytic_eq_pct"] > 70).sum())
print(
    f"  A) v1 metric (tail-estimated O_eq, no P0):  {n_a}/72 points above 70%, mean {df['pl_v1pct'].mean():.1f}%"
)
print(
    f"  B) analytic O_eq = 0.5, still no P0 check:  {n_b}/72 points above 70%, "
    f"mean {df['pl_analytic_eq_pct'].mean():.1f}%, min {df['pl_analytic_eq_pct'].min():.1f}%"
)
print(
    f"  C) either estimate + P0 check: P0 satisfied at {int(df['pl_p0'].sum())}/72 points (NO_TEST elsewhere)"
)
print("\nv1-metric 'strong' counts for the other observables (crossed_pct > 70), for context:")
for name, label in (("en", "energy"), ("x2", "<x^2>")):
    print(f"  {label:7s}: {int((df[f'{name}_v1pct'] > 70).sum())}/72")
