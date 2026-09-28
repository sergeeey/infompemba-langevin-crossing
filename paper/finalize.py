"""
Finalize all paper outputs after phase diagram completes.

1. Re-run spectral overlay with phase diagram data
2. Generate Figures 2 and 4 (need phase_diagram.parquet)
3. Print final statistics for the paper
"""

import numpy as np
import pandas as pd
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "src"))

RESULTS = os.path.join(PROJECT, "results")


def final_statistics():
    """Print all key numbers for the paper."""
    phase_path = os.path.join(RESULTS, "phase_diagram.parquet")
    spectral_path = os.path.join(RESULTS, "spectral_data.parquet")

    print("=" * 70)
    print("FINAL STATISTICS FOR PAPER")
    print("=" * 70)

    if not os.path.exists(phase_path):
        print("ERROR: phase_diagram.parquet not found")
        return

    df_ph = pd.read_parquet(phase_path)
    df_sp = pd.read_parquet(spectral_path)

    # Phase diagram stats
    agg = (
        df_ph.groupby(["barrier", "T"])
        .agg(
            mean_crossed=("crossed_pct", "mean"),
            std_crossed=("crossed_pct", "std"),
            frac_strong=("strong", "mean"),
        )
        .reset_index()
    )

    strong = agg[agg["frac_strong"] >= 0.7]
    dead = agg[agg["mean_crossed"] < 55]
    moderate = agg[(agg["mean_crossed"] >= 55) & (agg["frac_strong"] < 0.7)]

    print(f"\nPhase diagram: {len(agg)} points ({len(df_ph)} total runs)")
    print(f"  STRONG region (>=70% seeds strong): {len(strong)}/{len(agg)} points")
    print(f"  MODERATE region: {len(moderate)}/{len(agg)} points")
    print(f"  DEAD region (<55% crossed): {len(dead)}/{len(agg)} points")

    if len(strong) > 0:
        print(f"\n  Strong region boundaries:")
        print(f"    Barrier: {strong['barrier'].min():.1f} — {strong['barrier'].max():.1f}")
        print(f"    Temperature: {strong['T'].min():.2f} — {strong['T'].max():.2f}")
        print(
            f"    Peak: {strong['mean_crossed'].max():.1f}% at "
            f"b={strong.loc[strong['mean_crossed'].idxmax(), 'barrier']:.1f}, "
            f"T={strong.loc[strong['mean_crossed'].idxmax(), 'T']:.2f}"
        )

    # Spectral analysis stats
    print(f"\nSpectral analysis: {len(df_sp)} points")
    mask = df_sp["kramers_ratio"].notna() & (df_sp["lambda_1"] > 1e-12)
    kr = df_sp.loc[mask, "kramers_ratio"]
    within2 = ((kr > 0.5) & (kr < 2.0)).sum()
    print(f"  Kramers accuracy: median ratio {kr.median():.3f}")
    print(f"  Within factor 2: {within2}/{len(kr)} ({within2/len(kr)*100:.0f}%)")

    # Correlation: timescale separation vs Mpemba
    merged = agg.merge(df_sp[["barrier", "T", "timescale_ratio"]], on=["barrier", "T"])
    # Binarize
    merged["mpemba_strong"] = merged["frac_strong"] >= 0.7
    merged["ts_high"] = merged["timescale_ratio"] > 10

    # Confusion matrix
    tp = ((merged["mpemba_strong"]) & (merged["ts_high"])).sum()
    fp = ((~merged["mpemba_strong"]) & (merged["ts_high"])).sum()
    fn = ((merged["mpemba_strong"]) & (~merged["ts_high"])).sum()
    tn = ((~merged["mpemba_strong"]) & (~merged["ts_high"])).sum()

    print(f"\n  Spectral prediction (threshold: ts_ratio > 10):")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"    Precision: {precision:.2f}, Recall: {recall:.2f}")

    # Key numbers for abstract
    print(f"\n{'='*70}")
    print("KEY NUMBERS FOR ABSTRACT")
    print(f"{'='*70}")
    print(
        f"  Grid: {len(df_ph['barrier'].unique())} barriers x {len(df_ph['T'].unique())} temperatures x {len(df_ph['seed'].unique())} seeds"
    )
    print(f"  Total runs: {len(df_ph)}")
    print(f"  Strong Mpemba region: {len(strong)} of {len(agg)} parameter combinations")
    print(
        f"  Kramers formula: {within2/len(kr)*100:.0f}% within factor 2 (median ratio {kr.median():.2f})"
    )
    print(f"  Spectral prediction precision: {precision:.0%}, recall: {recall:.0%}")


def main():
    # Re-run spectral overlay with phase diagram
    print("Re-running spectral overlay with phase diagram data...")
    from run_spectral_analysis import plot_spectral_vs_mpemba

    df_sp = pd.read_parquet(os.path.join(RESULTS, "spectral_data.parquet"))
    plot_spectral_vs_mpemba(df_sp)

    # Generate Figures 2 and 4
    print("\nGenerating publication figures 2 and 4...")
    from paper.generate_figures import figure2_phase_diagram, figure4_spectral

    figure2_phase_diagram()
    figure4_spectral()

    # Final stats
    final_statistics()


if __name__ == "__main__":
    main()
