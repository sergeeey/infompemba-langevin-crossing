"""
Robustness test: does the Mpemba effect survive seed changes?
Kill criterion: effect must appear in >70% of seeds.
"""

import numpy as np
import pandas as pd
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toy_mpemba import run_langevin, double_well_grad, double_well_energy


def test_seed(seed, T=0.2, eta=0.005, n_particles=5000, n_steps=30000, eval_interval=50):
    np.random.seed(seed)
    cold_x0 = np.random.randn(n_particles, 2) * 0.05 + np.array([1.0, 0.0])
    hot_x0 = np.random.randn(n_particles, 2) * 2.0

    df_cold = run_langevin(double_well_grad, double_well_energy, cold_x0, T, eta, n_steps, eval_interval)
    np.random.seed(seed + 10000)
    df_hot = run_langevin(double_well_grad, double_well_energy, hot_x0, T, eta, n_steps, eval_interval)

    cold_pl = df_cold.set_index("step")["p_left"]
    hot_pl = df_hot.set_index("step")["p_left"]

    eq_val = (cold_pl.iloc[-len(cold_pl)//10:].mean() + hot_pl.iloc[-len(hot_pl)//10:].mean()) / 2

    dist_cold = np.abs(cold_pl - eq_val)
    dist_hot = np.abs(hot_pl - eq_val)
    gap = dist_hot - dist_cold

    # Check sustained crossing (hot closer to eq)
    total = len(gap)
    crossed = (gap < 0).sum()
    pct = crossed / total * 100

    # First crossing step
    crossings = gap[gap < 0]
    first_step = crossings.index[0] if len(crossings) > 0 else -1

    return {
        "seed": seed,
        "crossed_pct": pct,
        "first_cross_step": first_step,
        "eq_val": eq_val,
        "cold_final_dist": dist_cold.iloc[-1],
        "hot_final_dist": dist_hot.iloc[-1],
        "strong": pct > 70,
        "weak": 40 < pct <= 70,
        "none": pct <= 40,
    }


def main():
    seeds = list(range(42, 72))  # 30 seeds
    print(f"Testing {len(seeds)} seeds for Mpemba robustness (p_left, double-well, T=0.2)")
    print("=" * 70)

    results = []
    t0 = time.time()

    for i, seed in enumerate(seeds):
        r = test_seed(seed)
        results.append(r)
        label = "STRONG" if r["strong"] else ("weak" if r["weak"] else "NONE")
        print(f"  seed {seed:3d}: {r['crossed_pct']:5.1f}% crossed, first@step {r['first_cross_step']:5d} -> {label}")

    elapsed = time.time() - t0

    df = pd.DataFrame(results)
    df.to_parquet("results/toy_robustness.parquet", index=False)

    strong = df["strong"].sum()
    weak = df["weak"].sum()
    none_ct = df["none"].sum()

    print(f"\n{'='*70}")
    print(f"ROBUSTNESS RESULT ({len(seeds)} seeds, {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  STRONG (>70% crossed):  {strong}/{len(seeds)} ({strong/len(seeds)*100:.0f}%)")
    print(f"  Weak   (40-70%):        {weak}/{len(seeds)} ({weak/len(seeds)*100:.0f}%)")
    print(f"  None   (<40%):          {none_ct}/{len(seeds)} ({none_ct/len(seeds)*100:.0f}%)")
    print(f"  Mean crossed%:          {df['crossed_pct'].mean():.1f}%")
    print(f"  Median crossed%:        {df['crossed_pct'].median():.1f}%")
    print()

    if strong / len(seeds) >= 0.7:
        print(">>> ROBUST Mpemba effect confirmed across seeds")
    elif (strong + weak) / len(seeds) >= 0.7:
        print(">>> Mpemba effect present but noisy")
    else:
        print(">>> Mpemba effect NOT robust — fails kill criterion")


if __name__ == "__main__":
    main()
