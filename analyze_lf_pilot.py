"""
Analysis script for LangevinFisher discriminating pilot.
Checks: crossing existence, 50-step persistence, Welch t-test.
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys


def analyze_pilot(parquet_path):
    """Full pilot analysis with crossing detection."""
    df = pd.read_parquet(parquet_path)
    print(f"Records: {len(df)}")
    print(f"Runs: {df.run_id.nunique()}")
    print(f"Steps: {df.step.min()} - {df.step.max()}")
    print(f"Eval interval: {df.step.diff().drop_duplicates().mode().iloc[0] if len(df) > 1 else 'N/A'}")
    print()

    # === 1. Init validation ===
    step_0 = df[df.step == df.step.min()]
    cold_0 = step_0[step_0.init_type == 'cold']
    hot_0 = step_0[step_0.init_type == 'hot']

    print("=== INIT VALIDATION ===")
    if len(cold_0) > 0:
        print(f"  KL(cold) = {cold_0.kl_div_star.mean():.4f} (required < 0.5)")
    if len(hot_0) > 0:
        print(f"  KL(hot)  = {hot_0.kl_div_star.mean():.4f} (required > 5.0)")
    print()

    # === 2. Mean trajectories ===
    all_steps = sorted(df.step.unique())
    print(f"=== TRAJECTORY ANALYSIS ({len(all_steps)} eval points) ===")

    crossing_points = []
    persistent_crossing = False
    crossing_start = None

    for step in all_steps:
        sdf = df[df.step == step]
        c = sdf[sdf.init_type == 'cold'].kl_div_star
        h = sdf[sdf.init_type == 'hot'].kl_div_star

        if len(c) == 0 or len(h) == 0:
            continue

        c_mean, h_mean = c.mean(), h.mean()
        c_std, h_std = c.std(), h.std()
        c_n, h_n = len(c), len(h)

        # Welch t-test (односторонний, ТЗ Section 6)
        # H0: KL_cold - KL_hot <= 0; H1: KL_cold - KL_hot > 0
        if c_n > 1 and h_n > 1:
            t_stat, p_two = stats.ttest_ind(c, h, equal_var=False)
            # Односторонний p-value
            if t_stat > 0:
                p_val = p_two / 2.0
            else:
                p_val = 1.0 - p_two / 2.0
        else:
            t_stat, p_val = float('nan'), float('nan')

        is_crossing = h_mean < c_mean

        if is_crossing:
            crossing_points.append(step)

        # Print every 20 steps or at crossing
        if step % 200 == 0 or is_crossing:
            sig = "***" if is_crossing else ""
            print(f"  Step {step:4d}: KL_cold={c_mean:.4f}+-{c_std:.4f}  "
                  f"KL_hot={h_mean:.4f}+-{h_std:.4f}  "
                  f"t={t_stat:+.2f}  p={p_val:.4f}  {'CROSSING' if is_crossing else ''} {sig}")

    # === 3. Persistent crossing check (50 consecutive steps = 5 eval points) ===
    print()
    print("=== PERSISTENT CROSSING CHECK ===")
    print("  Required: 50 consecutive steps with KL_hot < KL_cold")
    print(f"  At eval_interval=10: >= 5 consecutive eval points")

    # Find first persistent crossing
    consecutive = 0
    first_persistent_step = None
    for i, step in enumerate(all_steps):
        if step in crossing_points:
            consecutive += 1
            if consecutive >= 5:
                first_persistent_step = all_steps[i - 4]  # First step of the sequence
                persistent_crossing = True
                break
        else:
            consecutive = 0

    if persistent_crossing:
        print(f"  *** PERSISTENT CROSSING DETECTED at step {first_persistent_step} ***")
        print(f"  Crossing maintained for >= 50 steps")

        # Welch test at crossing point
        sdf = df[df.step == first_persistent_step]
        c = sdf[sdf.init_type == 'cold'].kl_div_star
        h = sdf[sdf.init_type == 'hot'].kl_div_star
        t_stat, p_val = stats.ttest_ind(c, h, equal_var=False)

        print(f"  At crossing: KL_cold={c.mean():.4f}, KL_hot={h.mean():.4f}")
        print(f"  Welch t={t_stat:.4f}, p={p_val:.6f}")
        print(f"  Significant at alpha=0.01: {p_val < 0.01}")
        print(f"  Significant at alpha=0.05: {p_val < 0.05}")
    else:
        if crossing_points:
            print(f"  Temporary crossings at steps: {crossing_points[:10]}...")
            print(f"  But NONE persisted for 50 consecutive steps")
        else:
            print(f"  *** NO CROSSING AT ALL ***")
            print(f"  KL_hot >= KL_cold at all {len(all_steps)} eval points")

    # === 4. Bootstrap CI for final step ===
    print()
    print("=== BOOTSTRAP CI (final step) ===")
    final_step = all_steps[-1]
    sdf = df[df.step == final_step]
    c = sdf[sdf.init_type == 'cold'].kl_div_star.values
    h = sdf[sdf.init_type == 'hot'].kl_div_star.values

    if len(c) > 1 and len(h) > 1:
        n_bootstrap = 10000
        diff_means = []
        for _ in range(n_bootstrap):
            c_sample = np.random.choice(c, size=len(c), replace=True)
            h_sample = np.random.choice(h, size=len(h), replace=True)
            diff_means.append(h_sample.mean() - c_sample.mean())

        diff_means = np.array(diff_means)
        ci_lower = np.percentile(diff_means, 2.5)
        ci_upper = np.percentile(diff_means, 97.5)

        print(f"  Diff = mean(KL_hot) - mean(KL_cold) = {h.mean() - c.mean():.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  CI includes 0: {ci_lower < 0 < ci_upper}")
        print(f"  P(hot < cold) = {(diff_means < 0).mean():.4f}")

    # === 5. VERDICT ===
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    init_ok = (len(cold_0) > 0 and cold_0.kl_div_star.mean() < 0.5 and
               len(hot_0) > 0 and hot_0.kl_div_star.mean() > 5.0)

    if not init_ok:
        print("[INSUFFICIENT DATA] Init thresholds not met")
    elif persistent_crossing:
        print("[PASSED] Persistent crossing detected")
        print("  -> Proceed to Phase B: confirmatory run (200+ runs)")
    elif crossing_points:
        print("[FAILED] Temporary crossings but not persistent")
        print("  -> Do NOT proceed to full run")
    else:
        print("[FAILED] No crossing at all")
        print("  -> Do NOT proceed to full run")

    return persistent_crossing, crossing_points


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/trajectories_lf_pilot.parquet"
    analyze_pilot(path)
