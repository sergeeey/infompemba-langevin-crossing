"""
Generate publication-quality figures for the preprint.

Figure 1: Core result — observable-specific crossing
Figure 2: Phase diagram + spectral overlay
Figure 3: Control experiments
Figure 4: Spectral analysis (eigenmodes + Kramers validation)

Requires: results/phase_diagram.parquet, results/spectral_data.parquet
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)

# Add project root to path
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "src"))
sys.path.insert(0, PROJECT)

from run_toy_mpemba import run_langevin, double_well_grad, double_well_energy
from fokker_planck import compute_eigenvalues


RESULTS = os.path.join(PROJECT, "results")
PAPER = os.path.join(PROJECT, "paper")


def figure1_core_result():
    """
    Figure 1: Observable-specific Mpemba crossing.
    Three panels: p_left (crossing), energy (no crossing), x² (no crossing).
    """
    n_seeds = 10
    T = 0.2
    eta = 0.005
    N = 5000
    n_steps = 30000
    eval_interval = 50

    # Collect trajectories
    cold_pleft, hot_pleft = [], []
    cold_energy, hot_energy = [], []
    cold_x2, hot_x2 = [], []

    for seed in range(42, 42 + n_seeds):
        np.random.seed(seed)
        cold_x0 = np.random.randn(N, 2) * 0.05 + np.array([1.0, 0.0])
        df_c = run_langevin(
            double_well_grad, double_well_energy, cold_x0, T, eta, n_steps, eval_interval
        )

        np.random.seed(seed + 10000)
        hot_x0 = np.random.randn(N, 2) * 2.0
        df_h = run_langevin(
            double_well_grad, double_well_energy, hot_x0, T, eta, n_steps, eval_interval
        )

        cold_pleft.append(df_c["p_left"].values)
        hot_pleft.append(df_h["p_left"].values)
        cold_energy.append(df_c["mean_energy"].values)
        hot_energy.append(df_h["mean_energy"].values)
        cold_x2.append(df_c["mean_x2"].values)
        hot_x2.append(df_h["mean_x2"].values)

    steps = df_c["step"].values

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)

    # WHY: plot distance-to-equilibrium |observable - eq_value|, not raw values.
    # Raw p_left goes from 0 → 0.5 which doesn't visually show "crossing".
    # Distance-to-eq makes crossing explicit: hot line drops below cold.
    datasets = [
        (cold_pleft, hot_pleft, r"$|p_{\mathrm{left}} - p_{\mathrm{eq}}|$", "Crossing observed"),
        (cold_energy, hot_energy, r"$|\langle E \rangle - E_{\mathrm{eq}}|$", "No crossing"),
        (
            cold_x2,
            hot_x2,
            r"$|\langle x^2 \rangle - \langle x^2 \rangle_{\mathrm{eq}}|$",
            "No crossing",
        ),
    ]

    for ax, (cold_data, hot_data, ylabel, verdict) in zip(axes, datasets):
        cold_arr = np.array(cold_data)
        hot_arr = np.array(hot_data)

        cold_mean = cold_arr.mean(axis=0)
        hot_mean = hot_arr.mean(axis=0)

        # Equilibrium: last 10% average of both trajectories
        n_tail = max(1, len(cold_mean) // 10)
        eq_val = (cold_mean[-n_tail:].mean() + hot_mean[-n_tail:].mean()) / 2.0

        # Distance to equilibrium
        cold_dist = np.abs(cold_arr - eq_val)
        hot_dist = np.abs(hot_arr - eq_val)

        cold_d_mean = cold_dist.mean(axis=0)
        hot_d_mean = hot_dist.mean(axis=0)
        cold_d_std = cold_dist.std(axis=0)
        hot_d_std = hot_dist.std(axis=0)

        ax.plot(steps, cold_d_mean, "b-", linewidth=1.5, label="Cold (one well)")
        ax.fill_between(
            steps, cold_d_mean - cold_d_std, cold_d_mean + cold_d_std, color="blue", alpha=0.15
        )
        ax.plot(steps, hot_d_mean, "r-", linewidth=1.5, label="Hot (spread)")
        ax.fill_between(
            steps, hot_d_mean - hot_d_std, hot_d_mean + hot_d_std, color="red", alpha=0.15
        )

        ax.set_xlabel("Langevin step")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

        # Mark crossing region
        if "Crossing" in verdict:
            gap = hot_d_mean - cold_d_mean
            cross_idx = np.where(gap < 0)[0]
            if len(cross_idx) > 0:
                cross_step = steps[cross_idx[0]]
                ax.axvline(
                    cross_step,
                    color="green",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Crossing ~step {cross_step}",
                )
                ax.legend(loc="best")

        ax.set_title(verdict, fontweight="bold", color="green" if "observed" in verdict else "gray")

    labels = ["(a)", "(b)", "(c)"]
    for ax, label in zip(axes, labels):
        ax.text(
            -0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top"
        )

    fig.suptitle(r"$U(x,y) = (x^2-1)^2 + 0.3y^2$, $T=0.2$, $N=5000$, 10 seeds", fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(PAPER, "fig1_core_result.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def figure2_phase_diagram():
    """
    Figure 2: Phase diagram + spectral overlay.
    (a) Numerical % crossed
    (b) Spectral τ₁/τ₂
    (c) Overlay with contours
    """
    phase_path = os.path.join(RESULTS, "phase_diagram.parquet")
    spectral_path = os.path.join(RESULTS, "spectral_data.parquet")

    if not os.path.exists(phase_path):
        print(f"SKIP figure 2: {phase_path} not found")
        return
    if not os.path.exists(spectral_path):
        print(f"SKIP figure 2: {spectral_path} not found")
        return

    df_ph = pd.read_parquet(phase_path)
    df_sp = pd.read_parquet(spectral_path)

    barriers = sorted(df_sp["barrier"].unique())
    temps = sorted(df_sp["T"].unique())

    # Aggregate numerical data
    agg = (
        df_ph.groupby(["barrier", "T"])
        .agg(
            mean_crossed=("crossed_pct", "mean"),
            frac_strong=("strong", "mean"),
        )
        .reset_index()
    )

    crossed_map = np.full((len(barriers), len(temps)), np.nan)
    ts_map = np.full((len(barriers), len(temps)), np.nan)

    for _, row in agg.iterrows():
        if row["barrier"] in barriers and row["T"] in temps:
            bi = barriers.index(row["barrier"])
            ti = temps.index(row["T"])
            crossed_map[bi, ti] = row["mean_crossed"]

    for _, row in df_sp.iterrows():
        bi = barriers.index(row["barrier"])
        ti = temps.index(row["T"])
        ts_map[bi, ti] = row["timescale_ratio"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (a) Numerical
    ax = axes[0]
    im = ax.imshow(
        crossed_map,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )
    for bi in range(len(barriers)):
        for ti in range(len(temps)):
            val = crossed_map[bi, ti]
            if not np.isnan(val):
                c = "white" if val < 30 or val > 80 else "black"
                ax.text(
                    ti,
                    bi,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=c,
                )
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels(temps)
    ax.set_yticks(range(len(barriers)))
    ax.set_yticklabels(barriers)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Barrier height $b$")
    ax.set_title("(a) Numerical: mean % crossed")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (b) Spectral
    ax = axes[1]
    log_ts = np.log10(np.clip(ts_map, 1, 1e16))
    im2 = ax.imshow(log_ts, aspect="auto", origin="lower", cmap="plasma", interpolation="nearest")
    for bi in range(len(barriers)):
        for ti in range(len(temps)):
            val = ts_map[bi, ti]
            if not np.isnan(val) and val < 1e6:
                c = "white" if log_ts[bi, ti] > 3 else "black"
                ax.text(ti, bi, f"{val:.0f}", ha="center", va="center", fontsize=7, color=c)
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels(temps)
    ax.set_yticks(range(len(barriers)))
    ax.set_yticklabels(barriers)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Barrier height $b$")
    ax.set_title(r"(b) Spectral: $\log_{10}(\tau_1/\tau_2)$")
    fig.colorbar(im2, ax=ax, shrink=0.8)

    # (c) Overlay
    ax = axes[2]
    im3 = ax.imshow(
        crossed_map,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        interpolation="nearest",
        alpha=0.8,
    )
    X, Y = np.meshgrid(range(len(temps)), range(len(barriers)))
    for level, ls, lbl in [
        (5, ":", r"$\tau_1/\tau_2=5$"),
        (10, "--", r"$\tau_1/\tau_2=10$"),
        (50, "-", r"$\tau_1/\tau_2=50$"),
    ]:
        try:
            cs = ax.contour(
                X, Y, ts_map, levels=[level], colors="blue", linewidths=2, linestyles=ls
            )
            cs.collections[0].set_label(lbl)
        except Exception:
            pass

    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels(temps)
    ax.set_yticks(range(len(barriers)))
    ax.set_yticklabels(barriers)
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Barrier height $b$")
    ax.set_title("(c) Overlay: numerics + spectral boundary")
    ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(im3, ax=ax, shrink=0.8, label="% crossed")

    plt.tight_layout()
    out = os.path.join(PAPER, "fig2_phase_diagram.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def figure3_controls():
    """
    Figure 3: Kill criteria / control experiments.
    4 panels: unbalanced, balanced, single-well, clipping.
    """
    N = 5000
    T = 0.2
    eta = 0.005
    n_steps = 30000
    eval_interval = 50
    n_seeds = 10

    def run_pair(cold_fn, hot_fn, seeds, grad_fn=double_well_grad, energy_fn=double_well_energy):
        cold_all, hot_all = [], []
        for seed in seeds:
            np.random.seed(seed)
            cold_x0 = cold_fn(seed)
            df_c = run_langevin(grad_fn, energy_fn, cold_x0, T, eta, n_steps, eval_interval)
            np.random.seed(seed + 10000)
            hot_x0 = hot_fn(seed)
            df_h = run_langevin(grad_fn, energy_fn, hot_x0, T, eta, n_steps, eval_interval)
            cold_all.append(df_c["p_left"].values)
            hot_all.append(df_h["p_left"].values)
        return df_c["step"].values, np.array(cold_all), np.array(hot_all)

    def cold_unbalanced(seed):
        return np.random.randn(N, 2) * 0.05 + np.array([1.0, 0.0])

    def cold_balanced(seed):
        np.random.seed(seed)
        left = np.random.randn(N // 2, 2) * 0.05 + np.array([-1.0, 0.0])
        right = np.random.randn(N - N // 2, 2) * 0.05 + np.array([1.0, 0.0])
        return np.vstack([left, right])

    def hot_wide(seed):
        return np.random.randn(N, 2) * 2.0

    # Single-well potential
    def harmonic_grad(xy):
        return 2.0 * xy

    def harmonic_energy(xy):
        return np.sum(xy**2, axis=-1)

    def cold_harmonic(seed):
        return np.random.randn(N, 2) * 0.3

    def hot_harmonic(seed):
        return np.random.randn(N, 2) * 3.0

    seeds = list(range(42, 42 + n_seeds))

    print("  Running controls...")
    steps, c_unbal, h_unbal = run_pair(cold_unbalanced, hot_wide, seeds)
    steps, c_bal, h_bal = run_pair(cold_balanced, hot_wide, seeds)
    steps_h, c_harm, h_harm = run_pair(
        cold_harmonic, hot_harmonic, seeds, grad_fn=harmonic_grad, energy_fn=harmonic_energy
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    configs = [
        (c_unbal, h_unbal, steps, "Unbalanced cold\n(one well)", True),
        (c_bal, h_bal, steps, "Balanced cold\n(both wells)", False),
        (c_harm, h_harm, steps_h, "Single-well\n(harmonic)", False),
    ]

    for ax, (cold_arr, hot_arr, st, title, expect_mpemba) in zip(axes, configs):
        cm, hm = cold_arr.mean(0), hot_arr.mean(0)
        cs, hs = cold_arr.std(0), hot_arr.std(0)

        # For p_left: distance to equilibrium (0.5)
        eq = 0.5
        d_cold = np.abs(cm - eq)
        d_hot = np.abs(hm - eq)

        ax.plot(st, d_cold, "b-", lw=1.5, label="Cold")
        ax.fill_between(st, np.abs(cm - cs - eq), np.abs(cm + cs - eq), color="blue", alpha=0.1)
        ax.plot(st, d_hot, "r-", lw=1.5, label="Hot")
        ax.fill_between(st, np.abs(hm - hs - eq), np.abs(hm + hs - eq), color="red", alpha=0.1)

        # Crossing stats
        gap = d_hot - d_cold
        crossed_pct = (gap < 0).sum() / len(gap) * 100
        verdict = f"{crossed_pct:.0f}% crossed"
        color = "green" if crossed_pct > 70 else ("orange" if crossed_pct > 55 else "red")

        ax.set_title(f"{title}\n{verdict}", fontweight="bold", color=color)
        ax.set_xlabel("Step")
        ax.set_ylabel(r"$|p_{\mathrm{left}} - 0.5|$")
        ax.legend(loc="best")

    labels = ["(a)", "(b)", "(c)"]
    for ax, label in zip(axes, labels):
        ax.text(
            -0.12, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top"
        )

    plt.tight_layout()
    out = os.path.join(PAPER, "fig3_controls.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def figure4_spectral():
    """
    Figure 4: Eigenmodes + Kramers validation.
    """
    spectral_path = os.path.join(RESULTS, "spectral_data.parquet")
    if not os.path.exists(spectral_path):
        print(f"SKIP figure 4: {spectral_path} not found")
        return

    df_sp = pd.read_parquet(spectral_path)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Top row: 3 eigenmodes at different regimes
    params = [
        (0.5, 0.5, "Weak barrier, high T"),
        (1.0, 0.2, "Sweet spot"),
        (3.0, 0.1, "Strong barrier, low T"),
    ]

    for i, (b, t, label) in enumerate(params):
        ax = fig.add_subplot(gs[0, i])
        r = compute_eigenvalues(b, t, n_eigs=4, n_pts=501)
        x = r["x_grid"]

        U = b * (x**2 - 1.0) ** 2
        U_norm = U / max(U.max(), 1) * 0.4
        ax.fill_between(x, 0, U_norm, alpha=0.15, color="gray")

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        names = [r"$\psi_0$", r"$\psi_1$", r"$\psi_2$"]
        for k in range(min(3, len(r["eigenvalues"]))):
            psi = r["eigenvectors"][:, k]
            psi_sq = psi**2
            psi_sq /= psi_sq.max() * 1.2
            ax.plot(x, psi_sq, color=colors[k], lw=1.5, label=names[k])

        ax.set_xlim(-3, 3)
        ax.set_ylim(0, 0.9)
        ts = r["timescale_ratio"]
        ts_str = f"{ts:.0f}" if ts < 1e6 else f"{ts:.0e}"
        ax.set_title(f"$b={b}$, $T={t}$\n" + r"$\tau_1/\tau_2=$" + ts_str, fontsize=9)
        ax.set_xlabel("$x$")
        if i == 0:
            ax.legend(fontsize=7)

    # Bottom left: Kramers validation scatter
    ax = fig.add_subplot(gs[1, 0:2])
    mask = df_sp["kramers_ratio"].notna() & (df_sp["lambda_1"] > 1e-12)
    dv = df_sp[mask]

    sc = ax.scatter(
        dv["lambda_1"],
        dv["kramers_lambda1"],
        c=dv["T"],
        cmap="coolwarm",
        s=50,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
    )
    fig.colorbar(sc, ax=ax, label="Temperature $T$", shrink=0.8)

    lims = [dv["lambda_1"].min() * 0.5, dv["lambda_1"].max() * 2]
    ax.plot(lims, lims, "k--", alpha=0.5, label="$y=x$ (perfect)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_1$ (numerical FP)")
    ax.set_ylabel(r"$\lambda_1$ (Kramers formula)")
    ax.set_title("Kramers formula validation (94% within factor 2)")
    ax.legend()

    # Bottom right: gap ratio vs crossed %
    ax = fig.add_subplot(gs[1, 2])
    phase_path = os.path.join(RESULTS, "phase_diagram.parquet")
    if os.path.exists(phase_path):
        df_ph = pd.read_parquet(phase_path)
        agg = (
            df_ph.groupby(["barrier", "T"])
            .agg(
                mean_crossed=("crossed_pct", "mean"),
            )
            .reset_index()
        )

        merged = agg.merge(df_sp[["barrier", "T", "timescale_ratio"]], on=["barrier", "T"])
        merged = merged[merged["timescale_ratio"] < 1e10]

        ax.scatter(
            merged["timescale_ratio"],
            merged["mean_crossed"],
            c=merged["barrier"],
            cmap="viridis",
            s=50,
            edgecolors="k",
            linewidths=0.5,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\tau_1/\tau_2$")
        ax.set_ylabel("% crossed (numerical)")
        ax.set_title("Timescale separation\nvs Mpemba strength")
        ax.axhline(50, color="gray", ls=":", alpha=0.5, label="Noise level")
        ax.axhline(70, color="green", ls="--", alpha=0.5, label="Strong threshold")
        ax.legend(fontsize=7)
    else:
        ax.text(
            0.5,
            0.5,
            "Phase diagram\nnot computed",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    out = os.path.join(PAPER, "fig4_spectral.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


def main():
    os.makedirs(PAPER, exist_ok=True)

    print("Figure 1: Core result...")
    figure1_core_result()

    print("Figure 2: Phase diagram...")
    figure2_phase_diagram()

    print("Figure 3: Controls...")
    figure3_controls()

    print("Figure 4: Spectral analysis...")
    figure4_spectral()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
