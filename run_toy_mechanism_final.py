"""
Final mechanism tests:
1. Cold with balanced basin occupancy (kills "imbalance" explanation if effect persists)
2. Energy-only metric (kills "p_left bias" if effect persists)
3. No clipping control (kills "clipping artifact" if effect persists)
"""

import numpy as np
import pandas as pd
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toy_mpemba import run_langevin, double_well_grad, double_well_energy


def run_langevin_noclip(grad_fn, energy_fn, x0, T, eta, n_steps, eval_interval=50):
    """Langevin without any clipping — pure dynamics."""
    N = x0.shape[0]
    xy = x0.copy()
    noise_std = np.sqrt(2.0 * T * eta)
    results = []
    for step in range(n_steps):
        grad = grad_fn(xy)
        xy = xy - eta * grad + noise_std * np.random.randn(N, 2)
        if (step + 1) % eval_interval == 0:
            energy = energy_fn(xy)
            p_left = np.mean(xy[:, 0] < 0)
            results.append({
                "step": step + 1,
                "mean_energy": np.mean(energy),
                "std_energy": np.std(energy),
                "mean_x2": np.mean(xy[:, 0]**2),
                "p_left": p_left,
                "mean_x": np.mean(xy[:, 0]),
                "mean_y2": np.mean(xy[:, 1]**2),
            })
    return pd.DataFrame(results)


def measure_crossing(df_cold, df_hot, metric):
    cold = df_cold.set_index("step")[metric]
    hot = df_hot.set_index("step")[metric]
    eq_val = (cold.iloc[-len(cold)//10:].mean() + hot.iloc[-len(hot)//10:].mean()) / 2
    dist_cold = np.abs(cold - eq_val)
    dist_hot = np.abs(hot - eq_val)
    gap = dist_hot - dist_cold
    crossed_pct = (gap < 0).sum() / len(gap) * 100
    return crossed_pct, gap


def run_test(label, cold_x0, hot_x0, T=0.2, eta=0.005, n_steps=30000,
             eval_interval=50, seeds=range(42, 52), use_clipping=True, metrics=None):
    """Run test across seeds and multiple metrics."""
    if metrics is None:
        metrics = ["p_left", "mean_energy", "mean_x2"]

    langevin_fn = run_langevin if use_clipping else run_langevin_noclip
    results = {m: [] for m in metrics}

    for seed in seeds:
        np.random.seed(seed)
        cx0 = cold_x0(seed)
        hx0 = hot_x0(seed)

        df_c = langevin_fn(double_well_grad, double_well_energy, cx0, T, eta, n_steps, eval_interval)
        np.random.seed(seed + 10000)
        df_h = langevin_fn(double_well_grad, double_well_energy, hx0, T, eta, n_steps, eval_interval)

        for m in metrics:
            pct, _ = measure_crossing(df_c, df_h, m)
            results[m].append(pct)

    print(f"\n  {label}:")
    for m in metrics:
        arr = np.array(results[m])
        strong = np.sum(arr > 70)
        print(f"    {m:15s}: crossed={np.mean(arr):5.1f}%, strong={strong}/{len(arr)}")

    return results


def main():
    seeds = range(42, 52)
    N = 5000

    print("=" * 65)
    print("TEST A: COLD WITH BALANCED BASIN OCCUPANCY")
    print("  Cold: 50% in left well (x=-1), 50% in right well (x=+1)")
    print("  Hot:  wide gaussian centered at 0")
    print("  If effect disappears -> mechanism = basin imbalance")
    print("=" * 65)

    def cold_balanced(seed):
        np.random.seed(seed)
        n_left = N // 2
        n_right = N - n_left
        left = np.random.randn(n_left, 2) * 0.05 + np.array([-1.0, 0.0])
        right = np.random.randn(n_right, 2) * 0.05 + np.array([1.0, 0.0])
        return np.vstack([left, right])

    def cold_unbalanced(seed):
        np.random.seed(seed)
        return np.random.randn(N, 2) * 0.05 + np.array([1.0, 0.0])

    def hot_wide(seed):
        np.random.seed(seed + 5000)
        return np.random.randn(N, 2) * 2.0

    # A1: Original (unbalanced cold)
    run_test("A1: Unbalanced cold (original)", cold_unbalanced, hot_wide, seeds=seeds)

    # A2: Balanced cold
    run_test("A2: Balanced cold (50/50)", cold_balanced, hot_wide, seeds=seeds)

    print(f"\n{'='*65}")
    print("TEST B: NO CLIPPING CONTROL")
    print("  Same as original but without gradient/position clipping")
    print("  Using smaller eta=0.002 for stability")
    print("=" * 65)

    run_test("B1: With clipping (eta=0.005)", cold_unbalanced, hot_wide,
             eta=0.005, seeds=seeds, use_clipping=True)
    run_test("B2: No clipping (eta=0.002)", cold_unbalanced, hot_wide,
             eta=0.002, seeds=seeds, use_clipping=False)

    print(f"\n{'='*65}")
    print("SUMMARY")
    print("=" * 65)
    print("""
If A2 (balanced cold) shows ~50% crossed:
  -> Mechanism CONFIRMED: basin imbalance drives Mpemba

If A2 still shows >80%:
  -> Mechanism INCOMPLETE: intra-basin dynamics also matter

If B2 (no clipping) matches B1:
  -> Clipping is NOT an artifact

If B2 differs significantly from B1:
  -> Clipping AFFECTS the result
""")


if __name__ == "__main__":
    main()
