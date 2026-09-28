"""
Phase 2: Fokker-Planck spectral analysis.

Computes eigenvalues of the FP operator for the same (barrier, T) grid
as the phase diagram, then overlays the timescale separation boundary
onto the numerical Mpemba results.

Output:
  results/spectral_data.parquet     — eigenvalues for each (barrier, T)
  results/spectral_vs_mpemba.png    — theory vs numerics overlay
  results/eigenmodes.png            — eigenvector visualization for key points
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import time

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fokker_planck import compute_eigenvalues, kramers_lambda1, sweep_phase_diagram


def run_spectral_sweep():
    """Compute spectral data for the same grid as phase diagram."""
    barriers = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    temperatures = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

    print(
        f"Computing FP eigenvalues: {len(barriers)}×{len(temperatures)} = {len(barriers)*len(temperatures)} points"
    )
    t0 = time.time()

    results = sweep_phase_diagram(barriers, temperatures, n_pts=501)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    df = pd.DataFrame(results)
    return df


def plot_spectral_vs_mpemba(df_spectral, df_mpemba_path="results/phase_diagram.parquet"):
    """
    Overlay: heatmap of numerical Mpemba + contour of spectral gap ratio.

    The hypothesis: Mpemba appears where timescale_ratio is large (λ₁ ≪ λ₂).
    """
    if not os.path.exists(df_mpemba_path):
        print(f"WARNING: {df_mpemba_path} not found. Plotting spectral data only.")
        df_mpemba = None
    else:
        df_mpemba = pd.read_parquet(df_mpemba_path)

    barriers = sorted(df_spectral["barrier"].unique())
    temps = sorted(df_spectral["T"].unique())

    # Build spectral maps
    gap_ratio_map = np.full((len(barriers), len(temps)), np.nan)
    timescale_map = np.full((len(barriers), len(temps)), np.nan)
    kramers_accuracy_map = np.full((len(barriers), len(temps)), np.nan)

    for _, row in df_spectral.iterrows():
        bi = barriers.index(row["barrier"])
        ti = temps.index(row["T"])
        gap_ratio_map[bi, ti] = row["gap_ratio"]
        timescale_map[bi, ti] = row["timescale_ratio"]
        kramers_accuracy_map[bi, ti] = row["kramers_ratio"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # ── Panel 1: Timescale ratio (τ₁/τ₂) ──
    ax = axes[0]
    log_ts = np.log10(np.clip(timescale_map, 1e-3, 1e6))
    im = ax.imshow(log_ts, aspect="auto", origin="lower", cmap="plasma", interpolation="nearest")
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t}" for t in temps])
    ax.set_yticks(range(len(barriers)))
    ax.set_yticklabels([f"{b}" for b in barriers])
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Barrier height b")
    ax.set_title(r"$\log_{10}(\tau_1/\tau_2)$ — timescale separation")
    fig.colorbar(im, ax=ax)

    for bi in range(len(barriers)):
        for ti in range(len(temps)):
            val = timescale_map[bi, ti]
            if not np.isnan(val):
                txt = f"{val:.0f}" if val < 100 else f"{val:.0e}"
                color = "white" if log_ts[bi, ti] > 2 else "black"
                ax.text(ti, bi, txt, ha="center", va="center", fontsize=7, color=color)

    # ── Panel 2: Mpemba numerical (if available) ──
    ax = axes[1]
    if df_mpemba is not None:
        agg = (
            df_mpemba.groupby(["barrier", "T"])
            .agg(
                mean_crossed=("crossed_pct", "mean"),
            )
            .reset_index()
        )

        crossed_map = np.full((len(barriers), len(temps)), np.nan)
        for _, row in agg.iterrows():
            if row["barrier"] in barriers and row["T"] in temps:
                bi = barriers.index(row["barrier"])
                ti = temps.index(row["T"])
                crossed_map[bi, ti] = row["mean_crossed"]

        im2 = ax.imshow(
            crossed_map,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            interpolation="nearest",
        )
        fig.colorbar(im2, ax=ax, label="% crossed")

        for bi in range(len(barriers)):
            for ti in range(len(temps)):
                val = crossed_map[bi, ti]
                if not np.isnan(val):
                    color = "white" if val < 30 or val > 80 else "black"
                    ax.text(
                        ti,
                        bi,
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=color,
                    )

        # Overlay: contour of timescale ratio
        # Contour at τ₁/τ₂ = 5, 10, 50
        X, Y = np.meshgrid(range(len(temps)), range(len(barriers)))
        for level, ls in [(5, ":"), (10, "--"), (50, "-")]:
            try:
                ax.contour(
                    X,
                    Y,
                    timescale_map,
                    levels=[level],
                    colors="blue",
                    linewidths=1.5,
                    linestyles=ls,
                )
            except Exception:
                pass
    else:
        ax.text(
            0.5,
            0.5,
            "Phase diagram\nnot yet computed",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )

    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t}" for t in temps])
    ax.set_yticks(range(len(barriers)))
    ax.set_yticklabels([f"{b}" for b in barriers])
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Barrier height b")
    ax.set_title("Numerical Mpemba (%) + spectral contours (blue)")

    # ── Panel 3: Kramers vs numerical λ₁ ──
    ax = axes[2]
    # Scatter: Kramers prediction vs numerical
    mask = df_spectral["kramers_ratio"].notna() & (df_spectral["lambda_1"] > 1e-15)
    df_valid = df_spectral[mask]

    sc = ax.scatter(
        df_valid["lambda_1"],
        df_valid["kramers_lambda1"],
        c=df_valid["barrier"],
        cmap="viridis",
        s=60,
        edgecolors="k",
        linewidths=0.5,
    )
    fig.colorbar(sc, ax=ax, label="Barrier height")

    # Reference line y=x
    lims = [1e-10, max(df_valid["lambda_1"].max(), df_valid["kramers_lambda1"].max()) * 2]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_1$ (numerical)")
    ax.set_ylabel(r"$\lambda_1$ (Kramers)")
    ax.set_title("Kramers formula validation")
    ax.legend()

    fig.suptitle(
        "Fokker-Planck Spectral Analysis vs Numerical Mpemba Effect\n"
        r"$U(x) = b(x^2-1)^2$, overdamped Langevin",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("results/spectral_vs_mpemba.png", dpi=150, bbox_inches="tight")
    print("Saved: results/spectral_vs_mpemba.png")
    plt.close()


def plot_eigenmodes(barriers_to_show=None, T_to_show=None):
    """Visualize eigenmodes for a few key (barrier, T) points."""
    if barriers_to_show is None:
        barriers_to_show = [0.5, 1.0, 3.0]
    if T_to_show is None:
        T_to_show = [0.1, 0.2, 0.5]

    fig, axes = plt.subplots(
        len(barriers_to_show),
        len(T_to_show),
        figsize=(5 * len(T_to_show), 4 * len(barriers_to_show)),
        sharex=True,
    )
    if len(barriers_to_show) == 1:
        axes = axes[np.newaxis, :]
    if len(T_to_show) == 1:
        axes = axes[:, np.newaxis]

    for bi, b in enumerate(barriers_to_show):
        for ti, t in enumerate(T_to_show):
            ax = axes[bi, ti]
            r = compute_eigenvalues(b, t, n_eigs=4, n_pts=501)
            x = r["x_grid"]

            # Plot potential (scaled)
            U = b * (x**2 - 1.0) ** 2
            U_scaled = U / U.max() * 0.5
            ax.fill_between(x, 0, U_scaled, alpha=0.15, color="gray", label="U(x)")

            # Plot first 3 eigenvectors (squared = probability)
            colors = ["C0", "C1", "C2"]
            labels = [
                r"$\psi_0$ (stationary)",
                r"$\psi_1$ (inter-basin)",
                r"$\psi_2$ (intra-basin)",
            ]
            for k in range(min(3, len(r["eigenvalues"]))):
                psi = r["eigenvectors"][:, k]
                psi_sq = psi**2
                psi_sq /= psi_sq.max()  # normalize for visual
                ax.plot(x, psi_sq * 0.8, color=colors[k], linewidth=1.5, label=f"{labels[k]}")

            ax.set_xlim(-3, 3)
            ax.set_ylim(0, 1)
            ts_ratio = r["timescale_ratio"]
            ax.set_title(f"b={b}, T={t}\n" + r"$\tau_1/\tau_2$" + f"={ts_ratio:.1f}", fontsize=10)
            if bi == 0 and ti == 0:
                ax.legend(fontsize=7, loc="upper right")
            if bi == len(barriers_to_show) - 1:
                ax.set_xlabel("x")

    fig.suptitle("FP Eigenmodes: stationary, inter-basin (slow), intra-basin (fast)", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/eigenmodes.png", dpi=150, bbox_inches="tight")
    print("Saved: results/eigenmodes.png")
    plt.close()


def print_spectral_summary(df):
    """Print key findings."""
    print("\n" + "=" * 70)
    print("SPECTRAL ANALYSIS SUMMARY")
    print("=" * 70)

    # Where is timescale separation large?
    strong_sep = df[df["timescale_ratio"] > 10].sort_values("timescale_ratio", ascending=False)
    if len(strong_sep) > 0:
        print(f"\nStrong timescale separation (τ₁/τ₂ > 10): {len(strong_sep)} points")
        for _, row in strong_sep.head(10).iterrows():
            print(
                f"  b={row['barrier']:.1f}, T={row['T']:.2f}: "
                f"τ₁/τ₂={row['timescale_ratio']:.1f}, "
                f"λ₁={row['lambda_1']:.2e}, λ₂={row['lambda_2']:.2e}"
            )
    else:
        print("\nNo strong timescale separation found.")

    # Kramers accuracy
    mask = df["kramers_ratio"].notna() & (df["lambda_1"] > 1e-10)
    if mask.any():
        ratios = df.loc[mask, "kramers_ratio"]
        print(f"\nKramers formula accuracy:")
        print(f"  Median ratio (Kramers/numerical): {ratios.median():.3f}")
        print(f"  Range: {ratios.min():.3f} — {ratios.max():.3f}")
        good = ((ratios > 0.5) & (ratios < 2.0)).sum()
        print(f"  Within factor 2: {good}/{len(ratios)} ({good/len(ratios)*100:.0f}%)")


def main():
    os.makedirs("results", exist_ok=True)

    # Step 1: Spectral sweep
    df = run_spectral_sweep()
    df.to_parquet("results/spectral_data.parquet", index=False)
    print(f"Saved: results/spectral_data.parquet ({len(df)} rows)")

    # Step 2: Overlay plot
    plot_spectral_vs_mpemba(df)

    # Step 3: Eigenmodes
    plot_eigenmodes()

    # Step 4: Summary
    print_spectral_summary(df)


if __name__ == "__main__":
    main()
