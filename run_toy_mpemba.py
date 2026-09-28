"""
Toy 2D Langevin Mpemba test.

Double-well potential: U(x,y) = (x^2 - 1)^2 + 0.3*y^2
Overdamped Langevin: dx = -dU/dx dt + sqrt(2T) dW

Step 1: Harmonic well baseline (sanity check)
Step 2: Double-well (Mpemba search)
"""

import numpy as np
import pandas as pd
import time
import os


# ── Potentials ──────────────────────────────────────────────


def harmonic_grad(xy):
    """U = x^2 + y^2, grad = (2x, 2y)"""
    return 2.0 * xy


def double_well_grad(xy):
    """U = (x^2-1)^2 + 0.3*y^2, grad = (4x(x^2-1), 0.6y)"""
    x, y = xy[..., 0], xy[..., 1]
    gx = 4.0 * x * (x**2 - 1.0)
    gy = 0.6 * y
    return np.stack([gx, gy], axis=-1)


def double_well_energy(xy):
    """U = (x^2-1)^2 + 0.3*y^2"""
    x, y = xy[..., 0], xy[..., 1]
    return (x**2 - 1.0) ** 2 + 0.3 * y**2


# ── Simulator ───────────────────────────────────────────────


def run_langevin(
    grad_fn,
    energy_fn,
    x0,  # (N, 2) initial positions
    T,  # temperature
    eta,  # step size
    n_steps,
    eval_interval=10,
):
    """
    Overdamped Langevin dynamics for N particles in 2D.
    Returns metrics at eval points.
    """
    N = x0.shape[0]
    xy = x0.copy()
    noise_std = np.sqrt(2.0 * T * eta)

    results = []

    for step in range(n_steps):
        # Langevin step with gradient clipping for stability
        grad = grad_fn(xy)
        grad = np.clip(grad, -100.0, 100.0)
        xy = xy - eta * grad + noise_std * np.random.randn(N, 2)
        xy = np.clip(xy, -50.0, 50.0)

        if (step + 1) % eval_interval == 0:
            energy = energy_fn(xy)
            # Basin occupancy: fraction in left well (x < 0)
            p_left = np.mean(xy[:, 0] < 0)
            results.append(
                {
                    "step": step + 1,
                    "mean_energy": np.mean(energy),
                    "std_energy": np.std(energy),
                    "mean_x2": np.mean(xy[:, 0] ** 2),
                    "p_left": p_left,
                    "mean_x": np.mean(xy[:, 0]),
                    "mean_y2": np.mean(xy[:, 1] ** 2),
                }
            )

    return pd.DataFrame(results)


# ── Experiment ──────────────────────────────────────────────


def run_experiment(
    potential="double_well",
    T=0.5,
    eta=0.005,
    n_particles=5000,
    n_steps=20000,
    eval_interval=10,
    cold_center=(1.0, 0.0),
    cold_std=0.1,
    hot_center=(0.0, 0.0),
    hot_std=2.0,
    seed=42,
):
    np.random.seed(seed)

    if potential == "harmonic":
        grad_fn = harmonic_grad
        energy_fn = lambda xy: np.sum(xy**2, axis=-1)
        eq_label = "harmonic"
    else:
        grad_fn = double_well_grad
        energy_fn = double_well_energy
        eq_label = "double_well"

    # Initial distributions
    cold_x0 = np.random.randn(n_particles, 2) * cold_std + np.array(cold_center)
    hot_x0 = np.random.randn(n_particles, 2) * hot_std + np.array(hot_center)

    print(f"Potential: {eq_label}, T={T}, eta={eta}, N={n_particles}, steps={n_steps}")
    print(f"Cold: center={cold_center}, std={cold_std}")
    print(f"Hot:  center={hot_center}, std={hot_std}")
    print(f"Cold init energy: {np.mean(energy_fn(cold_x0)):.4f}")
    print(f"Hot  init energy: {np.mean(energy_fn(hot_x0)):.4f}")

    # Run cold
    print("\nRunning cold...")
    t0 = time.time()
    df_cold = run_langevin(grad_fn, energy_fn, cold_x0, T, eta, n_steps, eval_interval)
    df_cold["init_type"] = "cold"
    print(f"  done in {time.time() - t0:.1f}s")

    # Run hot
    print("Running hot...")
    np.random.seed(seed + 1)  # different noise realization
    t0 = time.time()
    df_hot = run_langevin(grad_fn, energy_fn, hot_x0, T, eta, n_steps, eval_interval)
    df_hot["init_type"] = "hot"
    print(f"  done in {time.time() - t0:.1f}s")

    df = pd.concat([df_cold, df_hot], ignore_index=True)
    return df


def analyze(df, metric="mean_energy"):
    """Check for crossing in a given metric."""
    cold = df[df.init_type == "cold"].set_index("step")[metric]
    hot = df[df.init_type == "hot"].set_index("step")[metric]

    # Equilibrium = last 10% average
    eq_cold = cold.iloc[-len(cold) // 10 :].mean()
    eq_hot = hot.iloc[-len(hot) // 10 :].mean()
    eq_val = (eq_cold + eq_hot) / 2

    # Distance to equilibrium
    dist_cold = np.abs(cold - eq_val)
    dist_hot = np.abs(hot - eq_val)

    gap = dist_hot - dist_cold  # negative = hot closer to eq

    print(f"\n{'=' * 55}")
    print(f"METRIC: {metric}")
    print(f"{'=' * 55}")
    print(f"Equilibrium value: {eq_val:.4f}")
    print(f"Cold start dist: {dist_cold.iloc[0]:.4f}")
    print(f"Hot  start dist: {dist_hot.iloc[0]:.4f}")
    print(f"Cold final dist: {dist_cold.iloc[-1]:.4f}")
    print(f"Hot  final dist: {dist_hot.iloc[-1]:.4f}")

    crossings = gap[gap < 0]
    if len(crossings) > 0:
        first = crossings.index[0]
        sustained = 0
        first_idx = list(gap.index).index(first)
        for i in range(first_idx, len(gap)):
            if gap.iloc[i] < 0:
                sustained += 1

        pct = sustained / (len(gap) - first_idx) * 100
        print(f"\n*** CROSSING at step {first}! ***")
        print(f"Sustained: {sustained}/{len(gap) - first_idx} steps ({pct:.0f}%) after crossing")

        if pct > 70:
            print(">>> STRONG Mpemba effect")
        elif pct > 40:
            print(">>> WEAK Mpemba effect (noisy)")
        else:
            print(">>> NOISE, not real Mpemba")
    else:
        print("\nNo crossing. Hot always farther from equilibrium.")
        print(f"Min gap: {gap.min():.4f} at step {gap.idxmin()}")

    # Print trajectory
    print(f"\nStep  | dist_cold | dist_hot  | gap")
    print("-" * 55)
    steps = list(gap.index)
    for s in steps[:: max(1, len(steps) // 20)]:
        marker = " <<<" if gap[s] < 0 else ""
        print(f"{s:6d} | {dist_cold[s]:9.4f} | {dist_hot[s]:9.4f} | {gap[s]:+8.4f}{marker}")

    return gap


def main():
    os.makedirs("results", exist_ok=True)

    # ── Step 1: Harmonic baseline ──
    print("=" * 60)
    print("STEP 1: HARMONIC WELL (sanity check)")
    print("=" * 60)

    df_harm = run_experiment(
        potential="harmonic",
        T=0.5,
        eta=0.005,
        n_particles=5000,
        n_steps=10000,
        eval_interval=10,
        cold_center=(0.0, 0.0),
        cold_std=0.3,
        hot_center=(0.0, 0.0),
        hot_std=3.0,
    )
    df_harm.to_parquet("results/toy_harmonic.parquet", index=False)
    gap_harm = analyze(df_harm, "mean_energy")

    # ── Step 2: Double-well ──
    print("\n\n" + "=" * 60)
    print("STEP 2: DOUBLE-WELL POTENTIAL (Mpemba search)")
    print("=" * 60)

    df_dw = run_experiment(
        potential="double_well",
        T=0.5,
        eta=0.005,
        n_particles=5000,
        n_steps=20000,
        eval_interval=10,
        cold_center=(1.0, 0.0),
        cold_std=0.1,
        hot_center=(0.0, 0.0),
        hot_std=2.0,
    )
    df_dw.to_parquet("results/toy_double_well.parquet", index=False)

    print("\n--- Energy metric ---")
    gap_e = analyze(df_dw, "mean_energy")

    print("\n--- Basin occupancy (p_left) ---")
    gap_p = analyze(df_dw, "p_left")

    print("\n--- Mean x^2 ---")
    gap_x2 = analyze(df_dw, "mean_x2")

    # ── Step 3: Double-well with lower T (stronger effect) ──
    print("\n\n" + "=" * 60)
    print("STEP 3: DOUBLE-WELL, LOW T=0.2 (stronger barrier)")
    print("=" * 60)

    df_dw2 = run_experiment(
        potential="double_well",
        T=0.2,
        eta=0.005,
        n_particles=5000,
        n_steps=30000,
        eval_interval=10,
        cold_center=(1.0, 0.0),
        cold_std=0.05,
        hot_center=(0.0, 0.0),
        hot_std=2.0,
    )
    df_dw2.to_parquet("results/toy_double_well_lowT.parquet", index=False)

    print("\n--- Energy metric ---")
    analyze(df_dw2, "mean_energy")

    print("\n--- Basin occupancy (p_left) ---")
    analyze(df_dw2, "p_left")

    print("\n\n" + "=" * 60)
    print("DONE. Results in results/toy_*.parquet")
    print("=" * 60)


if __name__ == "__main__":
    main()
