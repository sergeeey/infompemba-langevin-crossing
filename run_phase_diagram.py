"""
Phase diagram: barrier × temperature for Mpemba effect in double-well Langevin.

Potential: U(x,y) = barrier * (x^2 - 1)^2 + 0.3 * y^2
Observable: p_left (basin occupancy)

Grid: 9 barriers × 8 temperatures × 10 seeds = 720 runs
Each run: 5000 particles, 30000 steps, eval every 50 steps

Output:
  results/phase_diagram.parquet   — raw data (720 rows)
  results/phase_diagram.png       — heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

matplotlib.use("Agg")


# ── Parametric potential ────────────────────────────────────


def make_grad_fn(barrier: float):
    """Return gradient function for U = barrier*(x^2-1)^2 + 0.3*y^2."""

    def grad_fn(xy):
        x, y = xy[..., 0], xy[..., 1]
        gx = 4.0 * barrier * x * (x**2 - 1.0)
        gy = 0.6 * y
        return np.stack([gx, gy], axis=-1)

    return grad_fn


def make_energy_fn(barrier: float):
    """Return energy function for U = barrier*(x^2-1)^2 + 0.3*y^2."""

    def energy_fn(xy):
        x, y = xy[..., 0], xy[..., 1]
        return barrier * (x**2 - 1.0) ** 2 + 0.3 * y**2

    return energy_fn


# ── Langevin simulator ─────────────────────────────────────


def run_langevin(grad_fn, x0, T, eta, n_steps, eval_interval=50):
    """Overdamped Langevin for N particles in 2D. Returns p_left trajectory."""
    N = x0.shape[0]
    xy = x0.copy()
    noise_std = np.sqrt(2.0 * T * eta)
    p_left_traj = []

    for step in range(n_steps):
        grad = grad_fn(xy)
        grad = np.clip(grad, -100.0, 100.0)
        xy = xy - eta * grad + noise_std * np.random.randn(N, 2)
        xy = np.clip(xy, -50.0, 50.0)

        if (step + 1) % eval_interval == 0:
            p_left_traj.append(np.mean(xy[:, 0] < 0))

    return np.array(p_left_traj)


# ── Single point measurement ──────────────────────────────


def measure_one(barrier, T, seed, n_particles=5000, eta=0.005, n_steps=30000, eval_interval=50):
    """Run cold + hot for one (barrier, T, seed). Return crossing stats."""
    grad_fn = make_grad_fn(barrier)

    # Cold: all in right well
    np.random.seed(seed)
    cold_x0 = np.random.randn(n_particles, 2) * 0.05 + np.array([1.0, 0.0])
    cold_pl = run_langevin(grad_fn, cold_x0, T, eta, n_steps, eval_interval)

    # Hot: wide gaussian, covers both wells
    np.random.seed(seed + 10000)
    hot_x0 = np.random.randn(n_particles, 2) * 2.0
    hot_pl = run_langevin(grad_fn, hot_x0, T, eta, n_steps, eval_interval)

    # Equilibrium: last 10%
    n_tail = max(1, len(cold_pl) // 10)
    eq_val = (cold_pl[-n_tail:].mean() + hot_pl[-n_tail:].mean()) / 2.0

    dist_cold = np.abs(cold_pl - eq_val)
    dist_hot = np.abs(hot_pl - eq_val)
    gap = dist_hot - dist_cold

    total = len(gap)
    crossed_pct = (gap < 0).sum() / total * 100.0

    # First crossing step
    crossed_idx = np.where(gap < 0)[0]
    first_cross_step = int(crossed_idx[0] * eval_interval) if len(crossed_idx) > 0 else -1

    # Sustained crossing: after first crossing, how many consecutive?
    sustained = 0
    if len(crossed_idx) > 0:
        start = crossed_idx[0]
        for i in range(start, total):
            if gap[i] < 0:
                sustained += 1

    # Effect strength: mean gap in second half (negative = Mpemba)
    half = total // 2
    mean_gap_second_half = gap[half:].mean()

    return {
        "barrier": barrier,
        "T": T,
        "seed": seed,
        "crossed_pct": crossed_pct,
        "first_cross_step": first_cross_step,
        "sustained_pct": sustained
        / max(1, total - (crossed_idx[0] if len(crossed_idx) > 0 else 0))
        * 100.0
        if len(crossed_idx) > 0
        else 0.0,
        "mean_gap_2nd_half": mean_gap_second_half,
        "eq_p_left": eq_val,
        "strong": crossed_pct > 70,
    }


def run_point(args):
    """Wrapper for ProcessPoolExecutor."""
    barrier, T, seed = args
    return measure_one(barrier, T, seed)


# ── Phase diagram ──────────────────────────────────────────


def run_phase_diagram(
    barriers=None,
    temperatures=None,
    seeds=None,
    max_workers=None,
):
    if barriers is None:
        barriers = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    if temperatures is None:
        temperatures = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    if seeds is None:
        seeds = list(range(42, 52))  # 10 seeds

    tasks = [(b, t, s) for b in barriers for t in temperatures for s in seeds]
    total = len(tasks)
    print(
        f"Phase diagram: {len(barriers)} barriers × {len(temperatures)} temps × {len(seeds)} seeds = {total} runs"
    )

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 12)

    results = []
    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_point, task): task for task in tasks}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                eta_s = elapsed / done * (total - done) if done > 0 else 0
                print(
                    f"  {done}/{total} ({done/total*100:.0f}%) — {elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining"
                )

    df = pd.DataFrame(results)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return df


# ── Visualization ──────────────────────────────────────────


def plot_phase_diagram(df, output_path="results/phase_diagram.png"):
    """Generate heatmap: barrier × T, color = mean % crossed across seeds."""
    # Aggregate across seeds
    agg = (
        df.groupby(["barrier", "T"])
        .agg(
            mean_crossed=("crossed_pct", "mean"),
            std_crossed=("crossed_pct", "std"),
            frac_strong=("strong", "mean"),
            mean_first_step=(
                "first_cross_step",
                lambda x: x[x > 0].mean() if (x > 0).any() else -1,
            ),
        )
        .reset_index()
    )

    barriers = sorted(agg["barrier"].unique())
    temps = sorted(agg["T"].unique())

    # Build 2D arrays
    crossed_map = np.full((len(barriers), len(temps)), np.nan)
    strong_map = np.full((len(barriers), len(temps)), np.nan)

    for _, row in agg.iterrows():
        bi = barriers.index(row["barrier"])
        ti = temps.index(row["T"])
        crossed_map[bi, ti] = row["mean_crossed"]
        strong_map[bi, ti] = row["frac_strong"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap 1: Mean % crossed
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
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t}" for t in temps])
    ax.set_yticks(range(len(barriers)))
    ax.set_yticklabels([f"{b}" for b in barriers])
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Barrier height")
    ax.set_title("Mean % crossed (p_left)")
    fig.colorbar(im, ax=ax, label="% crossed")

    # Annotate
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
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )

    # Heatmap 2: Fraction of STRONG seeds
    ax = axes[1]
    im2 = ax.imshow(
        strong_map,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t}" for t in temps])
    ax.set_yticks(range(len(barriers)))
    ax.set_yticklabels([f"{b}" for b in barriers])
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Barrier height")
    ax.set_title("% seeds with STRONG effect (>70% crossed)")
    fig.colorbar(im2, ax=ax, label="% seeds STRONG")

    for bi in range(len(barriers)):
        for ti in range(len(temps)):
            val = strong_map[bi, ti]
            if not np.isnan(val):
                color = "white" if val < 30 or val > 80 else "black"
                ax.text(
                    ti,
                    bi,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )

    fig.suptitle(
        r"Mpemba Phase Diagram: $U = b(x^2-1)^2 + 0.3y^2$, observable = $p_{\mathrm{left}}$"
        "\n5000 particles, 30000 steps, 10 seeds per point",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(df):
    """Print text summary of phase diagram."""
    agg = (
        df.groupby(["barrier", "T"])
        .agg(
            mean_crossed=("crossed_pct", "mean"),
            frac_strong=("strong", "mean"),
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("PHASE DIAGRAM SUMMARY")
    print("=" * 70)

    # Find strong region
    strong_region = agg[agg["frac_strong"] >= 0.7]
    if len(strong_region) > 0:
        print(f"\nSTRONG effect region ({len(strong_region)} points, >=70% seeds STRONG):")
        for _, row in strong_region.iterrows():
            print(
                f"  barrier={row['barrier']:.1f}, T={row['T']:.2f}: "
                f"{row['mean_crossed']:.1f}% crossed, {row['frac_strong']*100:.0f}% seeds strong"
            )

        b_range = (strong_region["barrier"].min(), strong_region["barrier"].max())
        t_range = (strong_region["T"].min(), strong_region["T"].max())
        print(f"\n  Barrier range: {b_range[0]:.1f} — {b_range[1]:.1f}")
        print(f"  Temperature range: {t_range[0]:.2f} — {t_range[1]:.2f}")
    else:
        print("\nNo strong region found!")

    # Dead zones
    dead = agg[agg["mean_crossed"] < 55]
    if len(dead) > 0:
        print(f"\nDEAD zones ({len(dead)} points, <55% crossed = noise level):")
        for _, row in dead.iterrows():
            print(f"  barrier={row['barrier']:.1f}, T={row['T']:.2f}: {row['mean_crossed']:.1f}%")


# ── Main ───────────────────────────────────────────────────


def main():
    os.makedirs("results", exist_ok=True)

    df = run_phase_diagram()
    df.to_parquet("results/phase_diagram.parquet", index=False)
    print(f"Saved: results/phase_diagram.parquet ({len(df)} rows)")

    plot_phase_diagram(df)
    print_summary(df)


if __name__ == "__main__":
    main()
