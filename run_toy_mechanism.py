"""
Mechanism verification tests:
1. Single-well baseline (expect: no Mpemba)
2. Barrier height sweep (expect: effect depends on barrier)
3. Temperature sweep (expect: effect in specific T range)
"""

import numpy as np
import pandas as pd
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_toy_mpemba import run_langevin


def make_double_well(barrier_scale=1.0):
    """U(x,y) = barrier_scale * (x^2-1)^2 + 0.3*y^2"""
    def grad(xy):
        x, y = xy[..., 0], xy[..., 1]
        gx = barrier_scale * 4.0 * x * (x**2 - 1.0)
        gy = 0.6 * y
        return np.stack([gx, gy], axis=-1)
    def energy(xy):
        x, y = xy[..., 0], xy[..., 1]
        return barrier_scale * (x**2 - 1.0)**2 + 0.3 * y**2
    return grad, energy


def harmonic_grad(xy):
    return 2.0 * xy

def harmonic_energy(xy):
    return np.sum(xy**2, axis=-1)


def measure_mpemba(grad_fn, energy_fn, T, eta=0.005, n_particles=5000,
                   n_steps=30000, eval_interval=50, seed=42,
                   cold_center=(1.0, 0.0), cold_std=0.05,
                   hot_center=(0.0, 0.0), hot_std=2.0):
    """Returns crossed% for p_left metric."""
    np.random.seed(seed)
    cold_x0 = np.random.randn(n_particles, 2) * cold_std + np.array(cold_center)
    hot_x0 = np.random.randn(n_particles, 2) * hot_std + np.array(hot_center)

    df_cold = run_langevin(grad_fn, energy_fn, cold_x0, T, eta, n_steps, eval_interval)
    np.random.seed(seed + 10000)
    df_hot = run_langevin(grad_fn, energy_fn, hot_x0, T, eta, n_steps, eval_interval)

    cold_pl = df_cold.set_index("step")["p_left"]
    hot_pl = df_hot.set_index("step")["p_left"]

    eq_val = (cold_pl.iloc[-len(cold_pl)//10:].mean() +
              hot_pl.iloc[-len(hot_pl)//10:].mean()) / 2

    dist_cold = np.abs(cold_pl - eq_val)
    dist_hot = np.abs(hot_pl - eq_val)
    gap = dist_hot - dist_cold

    crossed_pct = (gap < 0).sum() / len(gap) * 100
    return crossed_pct


def run_multi_seed(label, grad_fn, energy_fn, T, seeds=range(42, 52), **kwargs):
    """Run over multiple seeds and return stats."""
    results = []
    for s in seeds:
        pct = measure_mpemba(grad_fn, energy_fn, T, seed=s, **kwargs)
        results.append(pct)
    arr = np.array(results)
    strong = np.sum(arr > 70)
    mean_pct = np.mean(arr)
    return {
        "label": label,
        "T": T,
        "mean_crossed_pct": mean_pct,
        "strong_count": int(strong),
        "total_seeds": len(results),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def main():
    os.makedirs("results", exist_ok=True)
    all_results = []
    seeds = range(42, 52)  # 10 seeds per test

    # ── TEST 1: Single-well baseline ──
    print("=" * 65)
    print("TEST 1: SINGLE-WELL BASELINE (expect: no Mpemba)")
    print("=" * 65)
    r = run_multi_seed("single_well", harmonic_grad, harmonic_energy, T=0.2, seeds=seeds,
                       cold_center=(0.5, 0.0), cold_std=0.05,
                       hot_center=(0.0, 0.0), hot_std=2.0)
    all_results.append(r)
    print(f"  Mean crossed: {r['mean_crossed_pct']:.1f}%, Strong: {r['strong_count']}/{r['total_seeds']}")
    verdict = "NO effect (expected)" if r['strong_count'] < 3 else "UNEXPECTED effect!"
    print(f"  >>> {verdict}")

    # ── TEST 2: Barrier height sweep ──
    print(f"\n{'='*65}")
    print("TEST 2: BARRIER HEIGHT SWEEP (expect: stronger barrier = stronger effect)")
    print("=" * 65)

    barriers = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    for b in barriers:
        grad_fn, energy_fn = make_double_well(barrier_scale=b)
        r = run_multi_seed(f"barrier_{b}", grad_fn, energy_fn, T=0.2, seeds=seeds)
        all_results.append(r)
        bar = "#" * int(r['mean_crossed_pct'] / 5)
        print(f"  barrier={b:4.1f}: crossed={r['mean_crossed_pct']:5.1f}%, "
              f"strong={r['strong_count']}/{r['total_seeds']} |{bar}")

    # ── TEST 3: Temperature sweep ──
    print(f"\n{'='*65}")
    print("TEST 3: TEMPERATURE SWEEP (expect: effect in specific T range)")
    print("=" * 65)

    grad_fn, energy_fn = make_double_well(barrier_scale=1.0)
    temps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]
    for t in temps:
        r = run_multi_seed(f"T_{t}", grad_fn, energy_fn, T=t, seeds=seeds)
        all_results.append(r)
        bar = "#" * int(r['mean_crossed_pct'] / 5)
        print(f"  T={t:4.2f}: crossed={r['mean_crossed_pct']:5.1f}%, "
              f"strong={r['strong_count']}/{r['total_seeds']} |{bar}")

    # ── Summary ──
    df = pd.DataFrame(all_results)
    df.to_parquet("results/toy_mechanism_tests.parquet", index=False)

    print(f"\n{'='*65}")
    print("MECHANISM VERIFICATION SUMMARY")
    print("=" * 65)

    # Check kill criteria
    sw = df[df.label == "single_well"].iloc[0]
    dw = df[df.label == "barrier_1.0"].iloc[0]

    print(f"\n1. Single-well vs double-well:")
    print(f"   Single-well: {sw['mean_crossed_pct']:.1f}%")
    print(f"   Double-well: {dw['mean_crossed_pct']:.1f}%")
    if sw['mean_crossed_pct'] < 60 and dw['mean_crossed_pct'] > 80:
        print("   >>> PASS: effect requires multi-basin structure")
    else:
        print("   >>> FAIL: effect not basin-dependent")

    barrier_rows = df[df.label.str.startswith("barrier_")]
    crossed_vals = barrier_rows["mean_crossed_pct"].values
    print(f"\n2. Barrier dependence:")
    if np.std(crossed_vals) > 5:
        print(f"   Std of crossed%: {np.std(crossed_vals):.1f} (spread > 5)")
        print("   >>> PASS: effect depends on barrier height")
    else:
        print(f"   Std of crossed%: {np.std(crossed_vals):.1f} (spread <= 5)")
        print("   >>> FAIL: effect independent of barrier")

    temp_rows = df[df.label.str.startswith("T_")]
    t_crossed = temp_rows["mean_crossed_pct"].values
    print(f"\n3. Temperature dependence:")
    if np.std(t_crossed) > 5:
        print(f"   Std of crossed%: {np.std(t_crossed):.1f} (spread > 5)")
        print("   >>> PASS: effect depends on temperature")
    else:
        print(f"   Std of crossed%: {np.std(t_crossed):.1f} (spread <= 5)")
        print("   >>> FAIL: effect independent of temperature")

    print(f"\nResults saved to results/toy_mechanism_tests.parquet")


if __name__ == "__main__":
    main()
